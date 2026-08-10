"""
Tests for M4, the VLM selector and Probability Fusion.

A mock VLM stands in for Gemini, so the fusion algebra, the overlay geometry and
the staleness behaviour are all testable offline and for free.
"""
from __future__ import annotations
import math
import time
import unittest

import numpy as np

from erc_urban.calib import CameraCalibration
from erc_urban.interfaces import Candidate, Observation
from erc_urban.modules.selector import (
    PROMPT, VLMSelector, _similarity, _softmax, render_overlay,
)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _cal():
    return CameraCalibration(width=512, height=288, fx=170.0, fy=170.0,
                             cx=256.0, cy=144.0, height_m=0.148,
                             pitch_rad=math.radians(3.0), source="test")


def _cands(n=5, span=1.5):
    out = []
    s = np.linspace(0.3, 2.0, 8)
    for i, lat in enumerate(np.linspace(-span, span, n)):
        xy = np.stack([s, lat * (s / s[-1])], axis=1)
        out.append(Candidate(waypoints=xy, score=-abs(float(lat)), stamp=1.0))
    return out


def _obs(stamp=1.0, cal=None):
    cal = cal or _cal()
    return Observation(
        front=np.full((cal.height, cal.width, 3), 90, np.uint8), rear=None,
        lat=37.0, lon=-122.0, yaw=0.0, stamp=stamp,
        goal_bearing=0.0, goal_range_m=20.0,
    )


class TestPromptDiscipline(unittest.TestCase):
    def test_prompt_hides_goal_and_scores(self):
        """Their ablation: shown planner scores or the goal, the VLM defers to
        the planner's ranking instead of applying its own judgement."""
        low = PROMPT.lower()
        for leak in ("score", "goal", "gps", "waypoint", "checkpoint", "confidence"):
            self.assertNotIn(leak, low, msg=f"prompt leaks '{leak}'")

    def test_prompt_asks_for_a_bare_index(self):
        self.assertIn("only the number", PROMPT.lower())


@unittest.skipUnless(_HAS_CV2, "requires opencv")
class TestOverlay(unittest.TestCase):
    def test_overlay_actually_draws(self):
        """The bug this replaces: upgrades._render_overlay returned the frame
        unchanged, so the VLM had nothing to select over."""
        cal = _cal()
        base = _obs(cal=cal).front
        out = render_overlay(base, _cands(), cal)
        self.assertEqual(out.shape, base.shape)
        self.assertGreater(int((out != base).any(axis=2).sum()), 500)

    def test_candidates_are_drawn_in_distinct_colours(self):
        cal = _cal()
        out = render_overlay(_obs(cal=cal).front, _cands(5), cal)
        colours = {tuple(c) for c in out.reshape(-1, 3) if tuple(c) != (90, 90, 90)}
        self.assertGreater(len(colours), 5)

    def test_left_candidate_draws_left_of_right_candidate(self):
        """A mirrored overlay would make the VLM confidently pick the wrong
        side; pinned against the projection convention in test_calib."""
        cal = _cal()
        base = _obs(cal=cal).front
        left = render_overlay(base, [_cands(3)[2]], cal)     # +y = left
        right = render_overlay(base, [_cands(3)[0]], cal)    # -y = right
        lx = np.argwhere((left != base).any(axis=2))[:, 1].mean()
        rx = np.argwhere((right != base).any(axis=2))[:, 1].mean()
        self.assertLess(lx, rx)

    def test_empty_candidate_set_is_safe(self):
        cal = _cal()
        base = _obs(cal=cal).front
        np.testing.assert_array_equal(render_overlay(base, [], cal), base)


class TestSimilarity(unittest.TestCase):
    def test_identical_path_scores_highest(self):
        a = np.stack([np.linspace(0.3, 2.0, 8), np.zeros(8)], axis=1)
        b = a + np.array([0.0, 0.5])
        self.assertGreater(_similarity(a, a, 0.0, 1.0), _similarity(b, a, 0.0, 1.0))

    def test_progress_alignment_compares_the_remaining_segment(self):
        """Without it a stale pick is penalised simply because the rover moved
        along it, which is the opposite of the intended behaviour."""
        stale = np.stack([np.linspace(0.0, 2.0, 9), np.zeros(9)], axis=1)
        ahead = np.stack([np.linspace(1.0, 2.0, 9), np.zeros(9)], axis=1)
        naive = _similarity(ahead, stale, 0.0, 1.0)
        aligned = _similarity(ahead, stale, 1.0, 1.0)
        self.assertGreater(aligned, naive)

    def test_horizon_truncation_handles_unequal_lengths(self):
        a = np.stack([np.linspace(0.3, 2.0, 4), np.zeros(4)], axis=1)
        b = np.stack([np.linspace(0.3, 2.0, 12), np.zeros(12)], axis=1)
        self.assertTrue(np.isfinite(_similarity(a, b, 0.0, 1.0)))

    def test_degenerate_stale_path_is_neutral(self):
        self.assertEqual(_similarity(np.zeros((4, 2)), np.zeros((1, 2)), 0.0, 1.0), 0.0)


class TestProbabilityFusion(unittest.TestCase):
    def _sel(self, pick_idx=None, **kw):
        s = VLMSelector(_cal(), vlm_call=None, **kw)
        if pick_idx is not None:
            c = _cands()
            s._pick = {"waypoints": c[pick_idx].waypoints.copy(),
                       "stamp": 1.0, "idx": pick_idx}
        return s

    def test_falls_back_to_planner_without_a_pick(self):
        s = self._sel()
        c = _cands()
        self.assertEqual(s.select(_obs(), c), int(np.argmax([x.score for x in c])))
        self.assertEqual(s.last_note, "planner_only")

    def test_default_lambda_bounds_the_vlm_to_half_the_weight(self):
        """
        Probability Fusion's whole point, and the reason the paper prefers it
        over Score Fusion: alpha = (lam/(lam+1))*w, so at the default lam=1 the
        VLM can contribute AT MOST 0.5 and cannot steamroll a confident planner.
        Score Fusion's similarity term is unbounded and has no such guarantee.
        """
        s = self._sel(pick_idx=0)
        s.select(_obs(stamp=1.0), _cands())
        self.assertAlmostEqual(s.last_alpha, 0.5, places=6)

    def test_fresh_pick_wins_when_the_vlm_is_trusted(self):
        c = _cands()
        planner_best = int(np.argmax([x.score for x in c]))
        self.assertNotEqual(planner_best, 0)
        s = self._sel(pick_idx=0, lam=1000.0)      # alpha -> ~1
        self.assertEqual(s.select(_obs(stamp=1.0), c), 0)

    def test_fresh_pick_shifts_probability_even_at_default_lambda(self):
        """Bounded influence must still be real influence."""
        c = _cands()
        s = self._sel(pick_idx=0)
        s.select(_obs(stamp=1.0), c)
        self.assertGreater(s.last_alpha, 0.0)

    def test_stale_pick_loses_influence(self):
        """The whole latency mechanism: an old pick must decay away rather than
        keep steering."""
        c = _cands()
        s = self._sel(pick_idx=0, lam=1000.0)
        fresh = s.select(_obs(stamp=1.0), c)
        old = s.select(_obs(stamp=1.0 + 12.0), c)
        self.assertEqual(fresh, 0)
        self.assertEqual(old, int(np.argmax([x.score for x in c])))

    def test_alpha_is_bounded_by_one(self):
        for lam in (0.5, 1.0, 10.0, 1000.0):
            s = self._sel(pick_idx=0, lam=lam)
            for dt in (0.0, 1.0, 5.0, 14.0):
                s.select(_obs(stamp=1.0 + dt), _cands())
                self.assertLessEqual(s.last_alpha, 1.0)
                self.assertGreaterEqual(s.last_alpha, 0.0)

    def test_alpha_decays_monotonically(self):
        s = self._sel(pick_idx=0)
        alphas = []
        for dt in (0.0, 1.0, 3.0, 6.0, 12.0):
            s.select(_obs(stamp=1.0 + dt), _cands())
            alphas.append(s.last_alpha)
        self.assertEqual(alphas, sorted(alphas, reverse=True))

    def test_very_stale_pick_is_discarded_entirely(self):
        s = self._sel(pick_idx=0, max_stale_s=10.0)
        s.select(_obs(stamp=1.0 + 30.0), _cands())
        self.assertIn("too_stale", s.last_note)

    def test_empty_candidates_is_safe(self):
        self.assertEqual(self._sel().select(_obs(), []), 0)

    def test_softmax_is_a_distribution(self):
        p = _softmax(np.array([-3.0, 0.0, 1.0]))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertTrue((p > 0).all())


@unittest.skipUnless(_HAS_CV2, "requires opencv")
class TestAsyncQuerying(unittest.TestCase):
    def test_select_never_blocks_on_a_slow_vlm(self):
        """Measured: the rover has a command watchdog, so command RATE is
        effective speed (single command +0.7 deg, 5 Hz streaming +18.6 deg).
        A 1-3 s in-tick VLM call would physically slow the rover to a crawl."""
        def slow(_img, _k):
            time.sleep(1.5)
            return 0

        s = VLMSelector(_cal(), vlm_call=slow, query_hz=100.0, gate_queries=False)
        c = _cands()
        t0 = time.monotonic()
        for _ in range(5):
            s.select(_obs(), c)
        self.assertLess(time.monotonic() - t0, 0.5)

    def test_only_one_query_in_flight(self):
        calls = []

        def slow(_img, _k):
            calls.append(1)
            time.sleep(0.4)
            return 0

        s = VLMSelector(_cal(), vlm_call=slow, query_hz=1000.0, gate_queries=False)
        c = _cands()
        for _ in range(10):
            s.select(_obs(), c)
        time.sleep(0.6)
        self.assertEqual(len(calls), 1)

    def test_answer_is_applied_once_it_arrives(self):
        s = VLMSelector(_cal(), vlm_call=lambda i, k: 0, query_hz=1000.0,
                        lam=1000.0, gate_queries=False)
        c = _cands()
        s.select(_obs(), c)
        time.sleep(0.4)
        self.assertEqual(s.select(_obs(stamp=1.0), c), 0)
        # >=1 rather than ==1: query_hz is deliberately huge here, so a second
        # select() is entitled to fire another (instant, mocked) query.
        self.assertGreaterEqual(s.n_answers, 1)

    def test_vlm_failure_degrades_to_the_planner(self):
        def boom(_img, _k):
            raise RuntimeError("api down")

        s = VLMSelector(_cal(), vlm_call=boom, query_hz=1000.0, gate_queries=False)
        c = _cands()
        s.select(_obs(), c)
        time.sleep(0.3)
        self.assertEqual(s.select(_obs(), c), int(np.argmax([x.score for x in c])))

    def test_out_of_range_index_is_ignored(self):
        s = VLMSelector(_cal(), vlm_call=lambda i, k: 999, query_hz=1000.0, gate_queries=False)
        c = _cands()
        s.select(_obs(), c)
        time.sleep(0.3)
        self.assertIsNone(s._pick)


if __name__ == "__main__":
    unittest.main()


class TestDecisionGating(unittest.TestCase):
    """
    The VLM is a high-level planner, not a per-tick controller. Querying it
    everywhere costs money, 1.6-2.1 s of staleness, and -- because the rover has
    a command watchdog -- tick rate, which is literally speed.
    """

    def _sel(self, **kw):
        return VLMSelector(_cal(), vlm_call=lambda i, k: 0, **kw)

    def test_routine_scene_is_skipped(self):
        """Clear winner, no hazard, wide corridor -> the planner can handle it."""
        s = self._sel()
        want, why = s.is_decision_point(_cands(5), hazard_frac=0.0,
                                        corridor_width=0.9)
        self.assertFalse(want)
        self.assertEqual(why, "routine")

    def test_hazard_forces_a_query(self):
        s = self._sel()
        want, why = s.is_decision_point(_cands(5), hazard_frac=0.10,
                                        corridor_width=0.9)
        self.assertTrue(want)
        self.assertIn("hazard", why)

    def test_ambiguous_corridor_forces_a_query(self):
        s = self._sel()
        want, why = s.is_decision_point(_cands(5), corridor_width=0.05)
        self.assertTrue(want)
        self.assertIn("ambiguous", why)

    def test_contested_candidates_force_a_query(self):
        """Near-tied top scores mean the planner has no strong opinion."""
        s = self._sel()
        # Top two nearly tied WITHIN a wide overall range. If every candidate
        # were tied, margin == spread and the ratio is 1.0 -- that is a
        # degenerate set, not a contested decision.
        scores = [-0.50, -0.51, -1.0, -1.5, -2.0]
        tied = [Candidate(c.waypoints, score=sc, stamp=1.0)
                for c, sc in zip(_cands(5), scores)]
        want, why = s.is_decision_point(tied, corridor_width=0.9)
        self.assertTrue(want)
        self.assertIn("contested", why)

    def test_gating_actually_suppresses_the_call(self):
        calls = []
        s = VLMSelector(_cal(), vlm_call=lambda i, k: calls.append(1) or 0,
                        query_hz=1000.0)
        for _ in range(5):
            s.select(_obs(), _cands(5), corridor_width=0.9)
        time.sleep(0.2)
        self.assertEqual(len(calls), 0)
        self.assertGreater(s.n_skipped, 0)

    def test_gating_can_be_disabled(self):
        s = self._sel(gate_queries=False, query_hz=1000.0)
        want, _ = s.is_decision_point(_cands(5), corridor_width=0.9)
        self.assertFalse(want)          # still routine...
        s.select(_obs(), _cands(5), corridor_width=0.9)
        time.sleep(0.2)
        self.assertEqual(s.n_skipped, 0)   # ...but not skipped
