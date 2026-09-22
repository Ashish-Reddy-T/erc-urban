"""Tests for M2, route-visual alignment."""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban_test.modules.route import RouteAligner, RouteAim


class TestConfidence(unittest.TestCase):
    def test_no_corridor_is_zero_confidence(self):
        r = RouteAligner()
        self.assertEqual(r.confidence(0.0), 0.0)
        self.assertEqual(r.confidence(r.corridor_min_width), 0.0)

    def test_wide_corridor_is_full_confidence(self):
        r = RouteAligner()
        self.assertEqual(r.confidence(0.9), 1.0)

    def test_confidence_ramps_not_switches(self):
        """A hard mode flip would make the rover twitch as a corridor comes and
        goes at the threshold."""
        r = RouteAligner()
        mid = (r.corridor_min_width + r.corridor_full_width) / 2.0
        c = r.confidence(mid)
        self.assertGreater(c, 0.0)
        self.assertLess(c, 1.0)


class TestCorridorFollowing(unittest.TestCase):
    def test_wide_corridor_overrides_gps(self):
        """On a single-track path the line is the way through, even when the
        goal points elsewhere."""
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(40.0),
                      corridor_bearing=math.radians(-5.0), corridor_width=0.8)
        self.assertEqual(aim.source, "corridor")
        # Must stay near the corridor, not near the 40 deg goal.
        self.assertLess(abs(math.degrees(aim.bearing) - (-5.0)), 10.0)

    def test_goal_bias_is_bounded(self):
        """The load-bearing bound. Blending a fraction of a large goal bearing
        parks the aim outside the corridor and loses the path (run.py:984)."""
        r = RouteAligner()
        for goal_deg in (30.0, 60.0, 100.0, 119.0):
            aim = r.align(goal_bearing=math.radians(goal_deg),
                          corridor_bearing=0.0, corridor_width=0.9)
            off = abs(math.degrees(aim.bearing))
            self.assertLessEqual(
                off, math.degrees(r.goal_bias_max) + 1e-6,
                msg=f"goal {goal_deg} deg pulled aim {off:.1f} deg off corridor",
            )

    def test_goal_biases_in_the_right_direction(self):
        r = RouteAligner()
        left = r.align(goal_bearing=math.radians(60.0),
                       corridor_bearing=0.0, corridor_width=0.9)
        right = r.align(goal_bearing=math.radians(-60.0),
                        corridor_bearing=0.0, corridor_width=0.9)
        self.assertGreater(left.bearing, 0.0)
        self.assertLess(right.bearing, 0.0)

    def test_narrow_corridor_slows_down(self):
        r = RouteAligner()
        narrow = r.align(goal_bearing=0.0, corridor_bearing=0.0,
                         corridor_width=r.corridor_min_width * 1.5)
        wide = r.align(goal_bearing=0.0, corridor_bearing=0.0, corridor_width=0.9)
        self.assertLess(narrow.speed_scale, wide.speed_scale)


class TestFallbacks(unittest.TestCase):
    def test_track_used_when_no_corridor(self):
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(40.0), corridor_bearing=None,
                      track_bearing=math.radians(-20.0), track_frac=0.5)
        self.assertEqual(aim.source, "track")
        self.assertLess(math.degrees(aim.bearing), 40.0)
        self.assertGreater(math.degrees(aim.bearing), -20.0)

    def test_gps_when_nothing_visual(self):
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(33.0))
        self.assertEqual(aim.source, "gps")
        self.assertAlmostEqual(math.degrees(aim.bearing), 33.0, places=6)
        self.assertEqual(aim.confidence, 0.0)

    def test_track_below_threshold_is_ignored(self):
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(33.0),
                      track_bearing=math.radians(-80.0), track_frac=0.001)
        self.assertEqual(aim.source, "gps")


class TestTurnaround(unittest.TestCase):
    def test_goal_behind_triggers_turnaround(self):
        """Only when the goal is genuinely behind. A corridor is not worth
        following backwards."""
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(170.0),
                      corridor_bearing=0.0, corridor_width=0.9)
        self.assertEqual(aim.source, "turnaround")
        self.assertEqual(aim.speed_scale, 0.0)

    def test_goal_ahead_of_threshold_keeps_corridor(self):
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(110.0),
                      corridor_bearing=0.0, corridor_width=0.9)
        self.assertEqual(aim.source, "corridor")

    def test_turnaround_is_symmetric(self):
        r = RouteAligner()
        for d in (150.0, -150.0):
            self.assertEqual(
                r.align(goal_bearing=math.radians(d), corridor_bearing=0.0,
                        corridor_width=0.9).source, "turnaround")


class TestFrozenGpsScenario(unittest.TestCase):
    """
    Measured on the live Mini+: lat/lon came back bit-identical across 24
    stationary samples and across 30 s of driving. A stale GPS bearing must not
    be able to drag the rover off a visible path.
    """

    def test_stale_gps_cannot_override_a_good_corridor(self):
        r = RouteAligner()
        corridor = math.radians(10.0)
        aims = [
            r.align(goal_bearing=math.radians(g), corridor_bearing=corridor,
                    corridor_width=0.85).bearing
            for g in (-100.0, -40.0, 0.0, 40.0, 100.0)
        ]
        spread = math.degrees(max(aims) - min(aims))
        self.assertLess(spread, 2 * math.degrees(r.goal_bias_max) + 1.0)
        for a in aims:
            self.assertLess(abs(math.degrees(a) - 10.0),
                            math.degrees(r.goal_bias_max) + 1.0)


if __name__ == "__main__":
    unittest.main()


class TestOpenGroundIsNotACorridor(unittest.TestCase):
    """
    Measured live on a grass field, and it cost a checkpoint.

    `semantics.corridor` found every column open and reported a corridor of
    width 1.00 — the whole frame. `RouteAligner` scored that at full confidence
    and then, per the corridor policy, allowed the goal to move the aim by at
    most `goal_bias_max_deg`. Logged: src=corridor(1.00@+8deg) with the goal at
    +90deg and cmd ang=0.02, about 2 deg/s of correction against a 90 degree
    error. The rover drove a straight line past the checkpoint with every gate
    reporting clear, and the range grew 24.4 m -> 30.5 m.

    A corridor spanning the whole frame is not a corridor. It is open ground,
    and open ground imposes no lateral constraint on where to drive.
    """

    def test_a_corridor_is_defined_by_its_edges(self):
        """
        Width alone cannot tell a wide path from open ground — both score high.
        Edges can: `corridor()` returns the longest run of drivable columns, so
        1.00 means no column anywhere was closed and there is nothing to follow.
        A wide corridor with real edges must keep full confidence.
        """
        r = RouteAligner()
        self.assertEqual(r.confidence(0.05), 0.0)     # too narrow to be real
        self.assertEqual(r.confidence(0.50), 1.0)     # a genuine corridor
        self.assertEqual(r.confidence(0.90), 1.0)     # wide, but it has edges
        self.assertEqual(r.confidence(0.994), 0.0)    # one stray closed column
        self.assertEqual(r.confidence(1.00), 0.0)     # no edges at all

    def test_full_frame_corridor_yields_the_goal_bearing(self):
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(90.0),
                      corridor_bearing=math.radians(8.0), corridor_width=1.00,
                      track_bearing=0.0, track_frac=1.00, goal_trusted=True)
        self.assertEqual(aim.confidence, 0.0)
        self.assertAlmostEqual(math.degrees(aim.bearing), 90.0, places=3)

    def test_a_real_corridor_still_wins(self):
        """The behaviour that matters on a sidewalk must be untouched."""
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(90.0),
                      corridor_bearing=math.radians(8.0), corridor_width=0.50,
                      track_bearing=0.0, track_frac=0.60, goal_trusted=True)
        self.assertEqual(aim.source, "corridor")
        self.assertEqual(aim.confidence, 1.0)
        self.assertLess(abs(math.degrees(aim.bearing)), 20.0)

    def test_saturated_near_band_is_not_a_track_bearing(self):
        """
        `semantics.track` returns the centroid of traversable pixels, so when
        the whole band is traversable the centroid is the middle of the frame
        by construction. Live, that printed as `@+0deg` on every tick while the
        goal sat 90 degrees away. It is an artifact, not an observation.
        """
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(90.0), corridor_bearing=None,
                      corridor_width=0.0, track_bearing=0.0, track_frac=1.00,
                      goal_trusted=True)
        self.assertEqual(aim.source, "gps")

    def test_a_discriminating_track_still_steers(self):
        r = RouteAligner()
        aim = r.align(goal_bearing=math.radians(90.0), corridor_bearing=None,
                      corridor_width=0.0, track_bearing=math.radians(20.0),
                      track_frac=0.30, goal_trusted=True)
        self.assertEqual(aim.source, "track")
