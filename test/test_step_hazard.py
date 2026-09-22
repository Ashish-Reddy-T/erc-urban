"""
Regression tests for the mission-2 step collision.

The rover drove into a raised step edge while three signals said stop. All
three failure modes are pinned here, because each of them independently would
have prevented the crash:

  1. the traversability veto returned "all_untraversable" for ~10 consecutive
     ticks and run.py only appended it to a log string
  2. depth clearance fell 1.20 m -> 0.06 m but depth had been demoted to
     advisory whenever a costmap existed
  3. the costmap reported `ground_found=False` with 0 obstacle points and
     3.50 m clearance -- a step edge presents almost nothing above the fitted
     plane, so RANSAC failed and the geometric veto went blind while still
     outranking depth
"""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban.calib import CameraCalibration
from erc_urban.interfaces import Candidate, Observation
from erc_urban.modules.traversability import TraversabilityVeto


def _cal():
    return CameraCalibration(width=256, height=144, fx=85.0, fy=85.0,
                             cx=128.0, cy=72.0, height_m=0.148,
                             pitch_rad=math.radians(3.0), source="test")


def _obs():
    return Observation(front=np.zeros((144, 256, 3), np.uint8), rear=None,
                       lat=30.48, lon=114.30, yaw=0.0, stamp=1.0,
                       goal_bearing=0.0, goal_range_m=83.0)


def _cands(n=7):
    out = []
    s = np.linspace(0.2, 2.0, 8)
    for lat in np.linspace(-1.2, 1.2, n):
        out.append(Candidate(np.stack([s, lat * s / s[-1]], axis=1),
                             score=-abs(float(lat)), stamp=1.0))
    return out


class Source:
    ready = True

    def __init__(self, frac):
        self.frac = frac

    def traversable(self, rgb, out_hw):
        m = np.zeros(out_hw, bool)
        if self.frac > 0:
            m[-max(1, int(out_hw[0] * self.frac)):, :] = True
        return m


class TestUntraversableIsReported(unittest.TestCase):
    def test_all_untraversable_is_surfaced_not_swallowed(self):
        """
        `apply` must SAY when nothing is drivable. run.py acts on this string;
        if it silently returned the original index with note 'ok', the caller
        could not tell a safe choice from a forced one -- which is precisely
        how the rover kept driving into a step.
        """
        veto = TraversabilityVeto(source=Source(0.0), calib=_cal(),
                                  min_traversable_frac=0.0)
        idx, note = veto.apply(_obs(), _cands(), 0)
        self.assertTrue(note.startswith("all_untraversable"),
                        msg=f"note was {note!r}")

    def test_partial_traversability_still_selects(self):
        """A scene with some drivable surface must not trip the stop."""
        veto = TraversabilityVeto(source=Source(0.9), calib=_cal(),
                                  min_traversable_frac=0.0)
        idx, note = veto.apply(_obs(), _cands(), 0)
        self.assertFalse(note.startswith("all_untraversable"))

    def test_degrading_scene_eventually_reports_untraversable(self):
        """
        Mirrors the observed sequence: near-field traversability decayed
        0.94 -> 0.83 -> 0.54 -> 0.26 -> 0.05 over ~10 ticks. The verdict has to
        flip somewhere on that ramp, well before the surface runs out.
        """
        notes = []
        for frac in (0.94, 0.83, 0.54, 0.26, 0.12, 0.05, 0.0):
            veto = TraversabilityVeto(source=Source(frac), calib=_cal(),
                                      min_traversable_frac=0.0)
            notes.append(veto.apply(_obs(), _cands(), 0)[1])
        self.assertTrue(any(n.startswith("all_untraversable") for n in notes),
                        msg=f"never tripped across the ramp: {notes}")


class TestDepthAuthority(unittest.TestCase):
    """
    The costmap may only outrank depth while it can actually see. These pin the
    two conditions under which it cannot.
    """

    def test_no_ground_plane_means_costmap_is_blind(self):
        from erc_urban.modules.costmap import DepthCostmap, CostmapConfig

        # A scene with no fittable ground plane: pure noise at long range.
        rng = np.random.default_rng(0)
        depth = rng.uniform(3.0, 3.4, (144, 256))
        res = DepthCostmap(CostmapConfig(max_range_m=3.5)).compute(depth)
        # Whatever it decides about obstacles, `ground_found` is the flag
        # run.py keys on to hand authority back to depth.
        self.assertIn(res.ground_found, (True, False))
        if not res.ground_found:
            self.assertEqual(res.reason, "no_ground_plane")
            self.assertEqual(res.n_obstacle_points, 0)

    def test_no_plane_does_not_invent_obstacles(self):
        """Camera-horizontal fallback treated every dirt pixel as occupied
        (10k pts / 0.92 m stop). Unknown ground is empty, not full."""
        from erc_urban.modules.costmap import DepthCostmap, CostmapConfig

        rng = np.random.default_rng(1)
        depth = rng.uniform(0.08, 0.35, (144, 256))
        res = DepthCostmap(CostmapConfig(max_range_m=3.5)).compute(depth)
        if not res.ground_found:
            self.assertEqual(res.n_obstacle_points, 0)
            self.assertGreaterEqual(res.forward_clearance_m, 3.0)

    def test_near_field_roots_are_not_obstacles(self):
        """A valid plane plus 20 cm bumps at 0.4 m is park dirt, not a wall."""
        from erc_urban.modules.costmap import DepthCostmap, CostmapConfig

        h, w = 144, 256
        # Fronto-parallel ground at 1.2 m with a low bump in the lower third.
        depth = np.full((h, w), 1.20, np.float32)
        depth[int(h * 0.65):, :] = 0.40
        res = DepthCostmap(CostmapConfig(max_range_m=3.5, obstacle_height_m=0.18)).compute(depth)
        if res.ground_found:
            self.assertGreater(res.forward_clearance_m, 1.0)

    def test_hard_stop_threshold_is_below_observed_impact_reading(self):
        """Depth read 0.06 m immediately before impact and 0.17/0.37/0.54 on
        the approach. The hard floor must sit above those."""
        from erc_urban import run as R

        self.assertGreater(R.DEPTH_HARD_STOP_M, 0.06)
        self.assertGreaterEqual(R.DEPTH_HARD_STOP_M, 0.37)

    def test_hard_stop_is_not_conditioned_on_the_costmap_agreeing(self):
        """
        The step crash read depth_stop(0.79 -> 0.06 m) against cost(fwd=3.50m
        pts=0) with the ground plane FOUND on most of those ticks. So requiring
        the costmap to corroborate the hard stop -- "it also sees something" --
        would switch the guard off on precisely the failure that motivated it.
        A costmap that cannot see step edges does not get a vote on whether a
        step edge is there.

        Pinned by reading the source, because the condition lives inside a
        1000-line `step()` that cannot be exercised without a rover.
        """
        import inspect
        from erc_urban import run as R

        src = inspect.getsource(R.step)
        i = src.index("DEPTH_HARD_STOP_M")
        clause = src[max(0, i - 900):i]
        # The clause must be reached without the costmap having to assert an
        # obstacle. `n_obstacle_points` appearing here would be that gate.
        self.assertNotIn("n_obstacle_points >", clause)
        self.assertIn("depth_view.forward_m <", src[i - 40:i + 40])
        self.assertIn("_rough_but_driveable", src)
        self.assertGreater(src.count("_rough_but_driveable"), 1)

    def test_driveable_dirt_skips_phantom_stops(self):
        from erc_urban.run import _rough_but_driveable

        self.assertTrue(_rough_but_driveable(1.0, np.zeros((8, 8), bool)))
        self.assertFalse(_rough_but_driveable(0.2, np.zeros((8, 8), bool)))
        self.assertFalse(_rough_but_driveable(1.0, np.ones((8, 8), bool)))


class TestManoeuvreGuard(unittest.TestCase):
    """
    Collision gates must be suppressed while the rover is turning in place, and
    ONLY then.

    Both halves matter. A 148 mm camera swept across the ground by a pivot
    reports contact that is not there -- eight escapes fired that way in one
    run, against a rover that had never commanded forward motion. But keying
    the suppression on "we were moving last tick" would leave the guard off for
    the first tick that rolls forward, which is the tick that drives at the
    step. So the test is `is THIS tick an in-place manoeuvre`, which the first
    forward tick is not.
    """

    def _guard_source(self):
        import inspect
        from erc_urban import run as R

        src = inspect.getsource(R.step)
        i = src.index("maneuvering = (")
        return src[i:i + 400]

    def test_guard_is_keyed_on_this_tick_not_the_last_command(self):
        g = self._guard_source()
        self.assertNotIn("last_lin", g)
        self.assertIn("pivot_busy", g)
        self.assertIn("want_align", g)

    def test_every_collision_gate_is_guarded(self):
        """
        The `obs_view` branch had no manoeuvre check at all, so a lamp-on-wall
        reading could fire an escape mid-pivot while the other three branches
        were correctly suppressed.
        """
        import inspect
        from erc_urban import run as R

        src = inspect.getsource(R.step)
        start = src.index("maneuvering = (")
        block = src[start:src.index("if hit:", start)]
        # Four gates: no path visible, costmap, depth, classical obstacle.
        # Matched with the leading `and` so prose in comments does not count.
        self.assertEqual(block.count("and not maneuvering"), 4,
                         msg="a collision gate is missing the guard")

    def test_no_path_visible_is_measured_on_drivable_not_track_only(self):
        """
        Observed live on a grass site: sem(hard=0.00 off=0.81
        track=0.07/near=0.08). Zero hazard, 81% of the frame soft — and the
        rover declared `no_path_visible(track_near=0.000)` and started escaping
        while sitting on perfectly drivable grass.

        Every other module already treats soft as drivable: semantics.py calls
        it "drivable but off-track ... penalised, not vetoed", traversability.py
        refuses to veto it, the costmap only costs it. Measuring the stop on
        `good` alone made this one check disagree with all of them.
        """
        import inspect
        from erc_urban import run as R

        src = inspect.getsource(R.step)
        # Anchor on the assignment, not the bare word — the word also appears
        # in the comment explaining this very fix.
        i = src.index('why = f"no_path_visible')
        gate = src[src.rindex("\n        if ", 0, i):i]
        self.assertIn("drive_frac", gate)
        self.assertNotIn("track_frac <", gate)
        # And the drivable mask must actually be track-or-verge.
        self.assertIn("sem.track(good_m | soft_m)", src)

    def test_grass_only_scene_is_drivable(self):
        """The numbers from the live log, end to end through categorize()."""
        from erc_urban.modules.semantics import SemanticTraversability

        sem = SemanticTraversability.__new__(SemanticTraversability)
        sem.hard, sem.soft, sem.good, sem.notground = {1}, {2}, {3}, {4}
        sem.conf, sem.min_blob_frac = 0.6, 0.004
        labels = np.full((120, 160), 2, np.int16)   # all grass
        hard, soft, good = sem.categorize(labels)
        _, track_band, _ = sem.track(good)
        _, drive_band, _ = sem.track(good | soft)
        from erc_urban import run as R
        self.assertLess(track_band, R.TRACK_MIN)       # the old signal stopped
        self.assertGreater(drive_band, R.TRACK_MIN)    # the new one drives

    def test_the_blind_gps_align_no_longer_outranks_vision(self):
        """
        The single worst bug in the loop: an unconditional
        `elif abs(degrees(bearing)) > ALIGN_DEG` that discarded RouteAligner's
        policy and span the rover in place on a clear sidewalk. It survives for
        STAGE 0 only, where PulsePivot does not run.
        """
        import inspect
        from erc_urban import run as R

        src = inspect.getsource(R.step)
        i = src.index("> ALIGN_DEG")
        line_start = src.rindex("\n", 0, i)
        self.assertIn("STAGE == 0", src[line_start:i])


if __name__ == "__main__":
    unittest.main()
