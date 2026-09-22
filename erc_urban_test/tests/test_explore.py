"""
Regression tests for the gap explorer (M8c).

The behaviour under test is the one the rover conspicuously lacked: with a goal
ahead and something in the way, pick a side and KEEP picking it. The observed
failure was eight escapes alternating sign while the rover stayed put, so the
central assertion here is sign consistency across many ticks, not a single
tick's answer.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from erc_urban_test.modules.explore import GapExplorer
from erc_urban_test.modules.localmap import LocalMap, PolarScan
from erc_urban_test.modules.odometry import OdomDelta
from erc_urban_test.modules.topograph import TopoGraph, BLOCKED


def _still():
    return OdomDelta(0.0, 0.0, 0.0, 1.0)


def _scan(clearance, unknown=0.0, n=36):
    c = np.asarray(clearance, np.float64)
    return PolarScan(
        bin_centers=np.linspace(-math.pi, math.pi, n, endpoint=False),
        clearance_m=c,
        unknown_frac=np.full(n, unknown, np.float64),
        max_range_m=8.0,
    )


def _drive(wall_lo, wall_hi, ticks=20, goal_bearing=0.0, topo=None):
    """Hold a wall 2 m ahead for `ticks` and collect the aim each tick."""
    m, ex = LocalMap(), GapExplorer()
    tg = topo if topo is not None else TopoGraph()
    wall = np.stack(
        [np.full(80, 2.0), np.linspace(wall_lo, wall_hi, 80)], axis=1)
    aims = []
    for t in range(ticks):
        m.update_pose(_still(), yaw=0.0)
        m.integrate(obstacles_robot=wall)
        tg.update(m.x, m.y, m.theta, m.travelled_m, float(t))
        a = ex.aim(m.clearance_polar(n_bins=36, max_range_m=8.0),
                   goal_bearing=goal_bearing, theta=m.theta,
                   travelled_m=m.travelled_m, blocked_score=tg.blocked_score)
        aims.append(a)
    return aims, ex, m, tg


class TestGapExtraction(unittest.TestCase):
    def test_a_wall_is_not_a_gap(self):
        """
        The bug this pins: with `min_gap_m` as a fixed 1.2 m and a wall 2 m
        ahead, the wall's own bearings passed the openness test, the "gap" found
        WAS the wall, and the rover drove into it. The threshold has to scale
        with the clearance actually on offer.
        """
        c = np.full(36, 8.0)
        c[16:20] = 1.7                     # wall straight ahead
        ex = GapExplorer()
        s = _scan(c)
        self.assertGreater(ex.open_threshold(s), 1.7)
        for g in ex.gaps(s):
            self.assertGreater(g.depth_m, 1.7)

    def test_a_tight_corridor_still_yields_a_gap(self):
        """The adaptive threshold must not decide a narrow world is impassable
        and hand the caller nothing."""
        ex = GapExplorer()
        self.assertTrue(ex.gaps(_scan(np.full(36, 1.6))))

    def test_a_slot_too_narrow_for_the_rover_is_rejected(self):
        c = np.full(36, 0.4)
        c[18] = 8.0                        # one 10-degree bin at 8 m
        # 10 degrees at 8 m is 1.4 m wide, so widen the rover past it.
        ex = GapExplorer(rover_width_m=3.0)
        self.assertEqual(ex.gaps(_scan(c)), [])

    def test_a_gap_spanning_the_seam_is_one_gap(self):
        """A run wrapping +/-pi must not be split into two half-gaps, and its
        centre must not average to 0 — i.e. to straight ahead."""
        c = np.full(36, 0.4)
        c[:3] = 8.0
        c[-3:] = 8.0                       # contiguous across the seam
        gaps = GapExplorer().gaps(_scan(c))
        self.assertEqual(len(gaps), 1)
        self.assertGreater(abs(gaps[0].bearing), math.radians(150))


class TestSideChoiceAndCommitment(unittest.TestCase):
    def test_open_left_goes_left_and_stays_left(self):
        aims, _, _, _ = _drive(-1.5, 0.6)
        signs = {1 if a.bearing >= 0 else -1 for a in aims}
        self.assertEqual(len(signs), 1, msg=f"flip-flopped: {signs}")
        self.assertGreater(aims[0].bearing, 0)

    def test_open_right_goes_right_and_stays_right(self):
        aims, _, _, _ = _drive(-0.6, 1.5)
        signs = {1 if a.bearing >= 0 else -1 for a in aims}
        self.assertEqual(len(signs), 1)
        self.assertLess(aims[0].bearing, 0)

    def test_the_turn_is_a_step_around_not_a_turn_back(self):
        """
        Aiming at a wide valley's CENTRE sends the rover backwards: with a wall
        ahead, every bearing from ~+35 round the back to ~-35 is one open run
        whose centre is 180 degrees. VFH+ aims at the border nearest the goal
        instead, which is what stepping past the wall's edge actually requires.
        """
        for lo, hi in ((-1.5, 0.6), (-0.6, 1.5)):
            aims, _, _, _ = _drive(lo, hi, ticks=4)
            self.assertLess(abs(math.degrees(aims[0].bearing)), 90.0,
                            msg=f"turned {math.degrees(aims[0].bearing):.0f} deg")

    def test_a_detour_is_committed_and_reported(self):
        aims, ex, _, _ = _drive(-1.5, 0.6, ticks=5)
        self.assertEqual(aims[-1].source, "detour")
        self.assertIsNotNone(ex.detour)

    def test_open_ground_produces_no_detour(self):
        """
        A detour must mean "something is in the way", not "the goal is not
        straight ahead". This conflation made the explorer invent a detour
        around nothing whenever the goal sat off to one side.
        """
        m, ex = LocalMap(), GapExplorer()
        for _ in range(5):
            m.update_pose(_still(), yaw=0.0)
            m.integrate(clearance_by_bin=np.full(16, 6.0),
                        bin_centers=np.linspace(-0.98, 0.98, 16))
            a = ex.aim(m.clearance_polar(n_bins=36, max_range_m=8.0),
                       goal_bearing=math.radians(-97.0), theta=m.theta,
                       travelled_m=m.travelled_m)
        self.assertEqual(a.source, "goal_clear")
        self.assertIsNone(ex.detour)

    def test_no_gap_anywhere_is_reported_not_invented(self):
        """With nothing drivable the explorer must say so and let the caller's
        escape ladder run, rather than emit a bearing on faith."""
        ex = GapExplorer(rover_width_m=5.0)
        a = ex.aim(_scan(np.full(36, 0.3)), 0.0, 0.0, 0.0)
        self.assertEqual(a.source, "no_gap")
        self.assertEqual(a.confidence, 0.0)
        self.assertEqual(a.speed_scale, 0.0)


class TestBlockedMemoryChangesTheChoice(unittest.TestCase):
    def test_a_direction_that_failed_here_loses_to_one_that_has_not(self):
        """
        The anti-livelock property. With the left side already recorded as
        failed at this place, the explorer must prefer the right — even though
        pure geometry rates them the same.
        """
        # Bins 16..20 are centred -20, -10, 0, +10, +20: a wall symmetric about
        # straight ahead, so geometry alone cannot prefer either side.
        c = np.full(36, 8.0)
        c[16:21] = 1.0
        ex = GapExplorer()
        tg = TopoGraph(blocked_hard_n=1)
        tg.update(0.0, 0.0, 0.0, 0.0, 0.0)
        # The whole left side has failed here. Marked across the span rather
        # than at one bearing because outcomes are stored in 30-degree bins.
        for deg in (30.0, 60.0, 90.0):
            tg.mark(math.radians(deg), BLOCKED)
        a = ex.aim(_scan(c), goal_bearing=0.0, theta=0.0, travelled_m=0.0,
                   blocked_score=tg.blocked_score)
        self.assertLess(a.bearing, 0.0,
                        msg=f"chose the failed side: {math.degrees(a.bearing):.0f}")

    def test_blocked_goal_does_not_pick_a_rear_gap_when_forward_is_open(self):
        """Park log: GPS at +79°, dirt ahead open, explorer latched
        gap(+130° w=180°). If the hole in front is clear, stay in the
        forward cone instead of spinning toward a valley border."""
        c = np.full(36, 8.0)
        # Block the goal sector (~+80°) so this is a real detour, not goal_clear.
        c[25:29] = 0.8
        a = GapExplorer().aim(
            _scan(c),
            goal_bearing=math.radians(80.0),
            theta=0.0,
            travelled_m=0.0,
        )
        self.assertNotEqual(a.source, "no_gap")
        self.assertLess(abs(a.bearing), math.radians(70.0))

    def test_open_forward_refuses_a_uturn_detour(self):
        """Once a right-side detour is latched, Bug2 will follow it behind
        the rover. Drop it if the hole in front is still open."""
        c = np.full(36, 8.0)
        c[16:21] = 1.0
        ex = GapExplorer()
        ex.aim(_scan(c), 0.0, 0.0, 0.0)
        self.assertIsNotNone(ex.detour)
        open_fwd = np.full(36, 8.0)
        a = ex.aim(_scan(open_fwd), goal_bearing=math.radians(20.0),
                   theta=0.0, travelled_m=0.1)
        self.assertNotEqual(a.source, "detour")
        self.assertLess(abs(a.bearing), math.radians(45.0))

    def test_without_memory_the_symmetric_case_is_still_decided(self):
        """No memory is not a reason to stall — some side must be picked."""
        c = np.full(36, 8.0)
        c[16:21] = 1.0
        a = GapExplorer().aim(_scan(c), 0.0, 0.0, 0.0)
        self.assertIsNotNone(a.bearing)
        self.assertNotEqual(a.source, "no_gap")


class TestDetourLifecycle(unittest.TestCase):
    def test_detour_expires_on_its_travel_budget(self):
        """A wrong commitment must cost metres, not the mission."""
        c = np.full(36, 8.0)
        c[16:20] = 1.0
        ex = GapExplorer(commit_budget_m=2.0)
        ex.aim(_scan(c), 0.0, 0.0, 0.0)
        self.assertIsNotNone(ex.detour)
        ex.aim(_scan(c), 0.0, 0.0, travelled_m=10.0)
        # Dropped and re-decided rather than grinding on the stale one.
        self.assertAlmostEqual(ex.detour.started_travel_m, 10.0, places=6)

    def test_detour_is_dropped_once_the_goal_line_is_clear(self):
        """Bug2's rejoin condition."""
        blocked = np.full(36, 8.0)
        blocked[16:20] = 1.0
        ex = GapExplorer()
        ex.aim(_scan(blocked), 0.0, 0.0, 0.0)
        self.assertIsNotNone(ex.detour)
        a = ex.aim(_scan(np.full(36, 8.0)), 0.0, 0.0, 0.1)
        self.assertIsNone(ex.detour)
        self.assertEqual(a.source, "goal_clear")

    def test_commitment_holds_while_the_rover_turns(self):
        """
        The detour is stored as a WORLD bearing, so rotating the rover must not
        change where it is trying to go. Store it robot-relative and the
        commitment evaporates on the very turn it exists to perform.
        """
        c = np.full(36, 8.0)
        c[16:21] = 1.0
        ex = GapExplorer()
        ex.aim(_scan(c), 0.0, theta=0.0, travelled_m=0.0)
        world0 = ex.detour.bearing
        # Rotate the rover 40 degrees left. A thing that was at world bearing 0
        # is now at robot bearing -40, so the scan rolls toward LOWER indices.
        c2 = np.roll(c, -4)
        ex.aim(_scan(c2), _wrap(0.0 - math.radians(40)),
               theta=math.radians(40), travelled_m=0.1)
        self.assertIsNotNone(
            ex.detour, msg="commitment evaporated on the turn it exists to make")
        self.assertLess(abs(_wrap(ex.detour.bearing - world0)),
                        math.radians(60))


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


if __name__ == "__main__":
    unittest.main()
