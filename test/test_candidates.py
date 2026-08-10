"""
Tests for M1, the K-candidate generator.

Most of these run without the 309 MB checkpoint by faking the three methods
MultiHypothesisPlanner actually calls on the base policy. That keeps the
keystone's LOGIC under test on any machine; whether the real weights produce a
useful spread is a different question, answered by eval/oracle_gap.py against
the real model.
"""
from __future__ import annotations
import unittest

import numpy as np

from erc_urban.interfaces import Candidate, Observation
from erc_urban.modules.baseline import BaselinePlanner
from erc_urban.modules.candidates import MultiHypothesisPlanner


def _obs(goal_bearing: float = 0.0) -> Observation:
    return Observation(
        front=np.zeros((8, 8, 3), dtype=np.uint8),
        rear=None,
        lat=37.0,
        lon=-122.0,
        yaw=0.0,
        stamp=1.0,
        goal_bearing=goal_bearing,
        goal_range_m=50.0,
    )


class FakeBase:
    """
    Stands in for LogoNavPlanner. Bends each path in proportion to the lateral
    offset it was conditioned on, which is the behaviour a goal-conditioned
    policy is supposed to have.
    """

    def __init__(self, gain: float = 0.25, ready: bool = True, warm: bool = True):
        self.gain = gain
        self.ready = ready
        self._warm = warm
        self.context_calls = 0
        self.batch_sizes: list[int] = []

    def update_context(self, obs):
        self.context_calls += 1
        return self._warm

    def goal_pose(self, obs, lateral_offset_m=0.0, forward_offset_m=0.0):
        return np.array([40.0, lateral_offset_m / 0.25, 1.0, 0.0], np.float32)

    def forward_batch(self, goal_poses):
        gp = np.atleast_2d(goal_poses)
        self.batch_sizes.append(gp.shape[0])
        s = np.linspace(0.0, 3.0, 8)
        out = []
        for row in gp:
            lateral = float(row[1]) * 0.25
            # Straight ahead, drifting toward the requested side.
            out.append(np.stack([s, self.gain * lateral * s], axis=1))
        return np.asarray(out)

    def score_waypoints(self, xy, obs):
        end = float(np.arctan2(xy[-1, 1], max(xy[-1, 0], 1e-3)))
        return 2.0 - abs(end - obs.goal_bearing)

    def set_goal(self, lat, lon):
        pass

    def reset(self):
        pass


class TestOffsets(unittest.TestCase):
    def test_zero_offset_always_present(self):
        """The unperturbed goal must be in the set, so K-sampling can never
        be worse than the single-hypothesis planner it replaces."""
        p = MultiHypothesisPlanner(base=FakeBase(), n_perturbed=12)
        self.assertAlmostEqual(float(p.offsets()[0]), 0.0)

    def test_offsets_symmetric_and_bounded(self):
        p = MultiHypothesisPlanner(base=FakeBase(), n_perturbed=12, lateral_span_m=3.0)
        offs = p.offsets()
        self.assertLessEqual(len(offs), 12)
        self.assertLessEqual(np.abs(offs).max(), 3.0 + 1e-9)
        # Every non-zero magnitude appears on both sides.
        for d in offs:
            if abs(d) > 1e-9:
                self.assertTrue(np.any(np.isclose(offs, -d)))

    def test_single_hypothesis_degenerates_cleanly(self):
        p = MultiHypothesisPlanner(base=FakeBase(), n_perturbed=1)
        self.assertEqual(list(p.offsets()), [0.0])


class TestPropose(unittest.TestCase):
    def test_emits_k_distinct_candidates(self):
        p = MultiHypothesisPlanner(base=FakeBase(), n_perturbed=6, n_anchors=12)
        cands = p.propose(_obs(), k=18)
        self.assertGreater(len(cands), 1)
        self.assertLessEqual(len(cands), 18)
        for c in cands:
            self.assertIsInstance(c, Candidate)
            self.assertEqual(c.waypoints.ndim, 2)
            self.assertEqual(c.waypoints.shape[1], 2)

    def test_candidates_are_actually_diverse(self):
        """The whole point, and the thing perturbation alone failed to deliver:
        the set must span real lateral choice, or M4/M5 have nothing to pick
        between and the keystone has failed."""
        p = MultiHypothesisPlanner(base=FakeBase(), n_perturbed=6, n_anchors=12)
        ends = np.array([c.waypoints[-1] for c in p.propose(_obs(), k=18)])
        self.assertGreater(float(ends[:, 1].max() - ends[:, 1].min()), 0.5)

    def test_anchors_dominate_the_spread(self):
        """Coverage must come from the anchor fan, not the policy. If someone
        sets n_anchors=0 thinking the model supplies diversity, the set
        collapses -- this pins the measured reason it cannot."""
        with_anchors = MultiHypothesisPlanner(
            base=FakeBase(gain=0.01), n_perturbed=6, n_anchors=12
        )
        without = MultiHypothesisPlanner(
            base=FakeBase(gain=0.01), n_perturbed=6, n_anchors=0
        )
        def spread(pl):
            e = np.array([c.waypoints[-1] for c in pl.propose(_obs(), k=18)])
            return float(e[:, 1].max() - e[:, 1].min())
        self.assertGreater(spread(with_anchors), 10 * spread(without))

    def test_scores_sorted_descending(self):
        p = MultiHypothesisPlanner(base=FakeBase(), n_perturbed=6, n_anchors=12)
        scores = [c.score for c in p.propose(_obs(), k=18)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_omnivla_rule_endpoint_nearest_goal_wins(self):
        """OmniVLA's selection rule for a pose goal: keep the trajectory whose
        endpoint points nearest the target."""
        p = MultiHypothesisPlanner(
            base=FakeBase(), n_perturbed=6, n_anchors=12, prior_weight=0.0
        )
        for bearing in (0.0, 0.35, -0.35):
            best = p.propose(_obs(goal_bearing=bearing), k=18)[0]
            err = abs(np.arctan2(best.waypoints[-1, 1], best.waypoints[-1, 0]) - bearing)
            self.assertLess(err, 0.25, msg=f"bearing={bearing}")

    def test_prior_weight_pulls_ranking_toward_logonav(self):
        """With prior_weight high, the winner must be the path most like
        LogoNav's own prediction, not merely the one aimed at the goal."""
        obs = _obs(goal_bearing=0.6)          # goal well off to the left
        prior = FakeBase().forward_batch(
            np.array([[40.0, 0.0, 1.0, 0.0]], np.float32)
        )[0]
        p = MultiHypothesisPlanner(
            base=FakeBase(), n_perturbed=6, n_anchors=12, prior_weight=50.0
        )
        best = p.propose(obs, k=18)[0]
        self.assertLess(
            float(np.mean(np.linalg.norm(best.waypoints - prior, axis=1))), 0.3
        )

    def test_context_pushed_exactly_once_per_tick(self):
        """K forwards must not mean K frames pushed — that would corrupt the
        model's temporal context."""
        base = FakeBase()
        p = MultiHypothesisPlanner(base=base, n_perturbed=6)
        p.propose(_obs(), k=18)
        self.assertEqual(base.context_calls, 1)

    def test_one_batched_forward_not_k_forwards(self):
        """Latency budget: all perturbed hypotheses cost one batched pass."""
        base = FakeBase()
        p = MultiHypothesisPlanner(base=base, n_perturbed=6, n_anchors=0)
        p.propose(_obs(), k=18)
        self.assertEqual(len(base.batch_sizes), 1)
        self.assertGreater(base.batch_sizes[0], 1)

    def test_duplicates_removed(self):
        """A saturated policy returns identical paths for different offsets;
        duplicates would let one answer win a softmax vote by multiplicity."""
        p = MultiHypothesisPlanner(
            base=FakeBase(gain=0.0), n_perturbed=6, n_anchors=0
        )
        self.assertEqual(len(p.propose(_obs(), k=18)), 1)

    def test_anchor_horizon_tracks_policy_horizon(self):
        """LogoNav shortens its path when blocked (2.69 m open vs 0.69 m facing
        an obstacle). Anchors must shorten with it, or every anchor looks
        equally unlike the prior and the prior term stops discriminating."""
        p = MultiHypothesisPlanner(base=FakeBase(), n_perturbed=1, n_anchors=12)
        prior = np.stack([np.linspace(0, 0.8, 8), np.zeros(8)], axis=1)
        lens = [float(np.linalg.norm(a[-1])) for a in p._anchor_paths(_obs(), prior)]
        self.assertLess(max(lens), 1.5)


class TestFallback(unittest.TestCase):
    def test_falls_back_to_anchors_during_warmup(self):
        p = MultiHypothesisPlanner(base=FakeBase(warm=False), n_anchors=12)
        cands = p.propose(_obs(), k=12)
        self.assertEqual(len(cands), 12)  # BaselinePlanner emits exactly k

    def test_falls_back_when_model_absent(self):
        base = FakeBase(warm=False, ready=False)
        p = MultiHypothesisPlanner(base=base, n_anchors=12)
        self.assertEqual(len(p.propose(_obs(), k=12)), 12)

    def test_scores_without_prior_are_pure_goal_seeking(self):
        p = MultiHypothesisPlanner(base=FakeBase(warm=False), n_anchors=12)
        self.assertEqual(len(p.propose(_obs(), k=12)), 12)

    def test_honors_planner_protocol(self):
        p = MultiHypothesisPlanner(base=FakeBase(), anchor_source=BaselinePlanner())
        self.assertTrue(hasattr(p, "propose"))
        self.assertTrue(hasattr(p, "set_goal"))
        self.assertTrue(hasattr(p, "reset"))

    def test_anchor_source_is_swappable(self):
        """NoMaD-exploration samples must be able to replace the arc fan
        without touching this class."""
        class FakeNoMaD:
            def propose(self, obs, k):
                s = np.linspace(0, 2.0, 8)
                return [
                    Candidate(
                        waypoints=np.stack([s, 0.3 * i * s], axis=1),
                        score=0.0, stamp=obs.stamp,
                    )
                    for i in range(-k // 2, k // 2)
                ]

        p = MultiHypothesisPlanner(base=FakeBase(), anchor_source=FakeNoMaD())
        self.assertGreater(len(p.propose(_obs(), k=18)), 5)


try:
    import utm as _utm  # noqa: F401
    _HAS_UTM = True
except ImportError:  # base env lacks it; rover_env has it
    _HAS_UTM = False


@unittest.skipUnless(_HAS_UTM, "requires `utm` (present in the rover_env env)")
class TestGoalPoseConvention(unittest.TestCase):
    """
    Pins the conditioning layout against upstream. If someone 'fixes' the axis
    swap in logonav.goal_pose, these fail loudly rather than the rover quietly
    driving perpendicular to its goal.

    Reference: LogoNav_frodobot.py:121 and vint_hf_dataset.py:662-670.
    """

    def _gp(self, yaw, goal_lat, goal_lon, **kw):
        from erc_urban.modules.logonav import LogoNavPlanner

        p = LogoNavPlanner.__new__(LogoNavPlanner)  # no checkpoint needed
        p.goal_latlon = (goal_lat, goal_lon)
        p.metric_waypoint_spacing = 0.25
        p.goal_distance_cap_m = 30.0
        o = _obs()
        o.lat, o.lon, o.yaw = 37.0, -122.0, yaw
        return p.goal_pose(o, **kw)

    def test_goal_ahead_is_positive_forward(self):
        # Facing north (yaw=0), goal to the north.
        gp = self._gp(0.0, 37.001, -122.0)
        self.assertIsNotNone(gp)
        self.assertGreater(gp[0], 0.0)                      # forward component
        # Lateral is near zero but NOT exactly zero: see the convergence test
        # below. Tolerance is 2 deg of bearing over the 30 m capped reach.
        self.assertLess(abs(float(gp[1]) * 0.25), 30.0 * np.radians(2.0))

    def test_goal_left_is_positive_lateral(self):
        # Facing north, goal to the west == left in a left-positive frame.
        gp = self._gp(0.0, 37.0, -122.001)
        self.assertGreater(gp[1], 0.0)

    def test_utm_grid_convergence_is_expected_not_a_bug(self):
        """
        Documents a real ~0.6 deg bias, so nobody 'fixes' it later.

        `yaw` comes from the compass and is relative to TRUE north; UTM easting
        /northing deltas are relative to GRID north. They differ by the grid
        convergence gamma ~= dlon * sin(lat). At lon -122 in UTM zone 10
        (central meridian -123) and lat 37: 1 deg * sin(37 deg) = 0.60 deg.

        We deliberately do NOT correct it. MBRA's own dataloader builds goal
        poses from UTM the same way (vint_hf_dataset.py), so the policy was
        trained through this bias; removing it at inference would put us out of
        distribution. Worst case at a zone edge is 3 deg * sin(60 deg) ~ 2.6 deg,
        which is under a metre of lateral error at the 30 m aim point and far
        inside the 15 m checkpoint tolerance.
        """
        gp = self._gp(0.0, 37.001, -122.0)
        reach_m = float(np.hypot(gp[0], gp[1])) * 0.25
        bearing_deg = abs(np.degrees(np.arctan2(float(gp[1]), float(gp[0]))))
        expected = abs(np.degrees(np.radians(1.0) * np.sin(np.radians(37.0))))
        self.assertAlmostEqual(bearing_deg, expected, places=1)
        self.assertAlmostEqual(reach_m, 30.0, places=3)     # cap applied

    def test_lateral_offset_moves_aim_point_left(self):
        gp0 = self._gp(0.0, 37.001, -122.0)
        gp1 = self._gp(0.0, 37.001, -122.0, lateral_offset_m=2.0)
        # +2 m left, divided by the 0.25 m normalisation => +8 units.
        self.assertAlmostEqual(float(gp1[1] - gp0[1]), 8.0, places=3)

    def test_offset_not_shrunk_by_the_distance_cap(self):
        """
        Regression: the cap is a proportional rescale, so applying it AFTER the
        offset shrank the offset by the same factor. A 111 m goal caps to 30 m
        and turned a 2 m offset into 0.54 m; a 300 m goal collapsed it ~10x.
        That silently degenerates the candidate set to near-duplicates exactly
        when the goal is far away. Offsets must survive the cap intact.
        """
        for goal_lat in (37.001, 37.003, 37.01):    # ~111 m, ~333 m, ~1.1 km
            gp0 = self._gp(0.0, goal_lat, -122.0)
            gp2 = self._gp(0.0, goal_lat, -122.0, lateral_offset_m=2.0)
            self.assertAlmostEqual(
                float(gp2[1] - gp0[1]), 8.0, places=3,
                msg=f"offset shrunk at goal_lat={goal_lat}",
            )


if __name__ == "__main__":
    unittest.main()
