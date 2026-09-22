"""
Regression tests for the rolling local map (M8a).

The map exists because the stack had no memory of space, and the specific
defect it replaces is `DepthCostmap`'s robot-frame `memory_frames` history,
which was OR-ed across ticks without ever being transported. Under an in-place
pivot that smeared past obstacles across the fan at the wrong bearings and
vetoed every candidate on empty ground — logged as veto=4/12 then 5/12 then
all_untraversable 12/12 while the very same line reported pts=0.

So the load-bearing test here is the one that spins the rover on the spot and
demands the map stay empty.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from erc_urban.modules.localmap import LocalMap
from erc_urban.modules.odometry import OdomDelta


def _still(conf=1.0):
    return OdomDelta(0.0, 0.0, 0.0, conf)


class TestPose(unittest.TestCase):
    def test_rotation_does_not_translate(self):
        m = LocalMap()
        for k in range(24):
            m.update_pose(_still(), yaw=math.radians(k * 15))
        self.assertAlmostEqual(m.x, 0.0, places=6)
        self.assertAlmostEqual(m.y, 0.0, places=6)

    def test_heading_comes_from_the_compass_not_odometry(self):
        """
        odometry.py measures accumulated VO yaw ~3x LOW over a spin. If the map
        integrated that instead of the compass, every obstacle laid down while
        turning would land at roughly a third of its true bearing.
        """
        m = LocalMap()
        # VO claims a tiny rotation; the compass says 90 degrees.
        m.update_pose(OdomDelta(0.0, 0.0, math.radians(5.0), 1.0),
                      yaw=math.radians(90.0))
        self.assertAlmostEqual(m.theta, math.radians(-90.0), places=6)

    def test_vo_yaw_is_the_fallback_when_the_compass_drops_out(self):
        m = LocalMap()
        m.update_pose(_still(), yaw=0.0)
        m.update_pose(OdomDelta(0.0, 0.0, math.radians(10.0), 1.0), yaw=None)
        self.assertAlmostEqual(math.degrees(m.theta), 10.0, places=4)

    def test_translation_uses_the_heading_the_delta_was_measured_in(self):
        """
        The odometry delta is expressed in the PREVIOUS robot frame, so it must
        be rotated by the old heading before the heading is replaced. Getting
        this backwards smears every observation along a turn.
        """
        m = LocalMap()
        m.update_pose(_still(), yaw=0.0)               # facing world +X
        # Drive 1 m forward, and only THEN turn 90 degrees.
        m.update_pose(OdomDelta(1.0, 0.0, 0.0, 1.0), yaw=math.radians(90.0))
        self.assertAlmostEqual(m.x, 1.0, places=6)
        self.assertAlmostEqual(m.y, 0.0, places=6)

    def test_zero_confidence_odometry_is_ignored(self):
        m = LocalMap()
        m.update_pose(OdomDelta(5.0, 5.0, 0.0, 0.0), yaw=0.0)
        self.assertAlmostEqual(m.x, 0.0, places=6)


class TestOccupancy(unittest.TestCase):
    def test_spinning_on_the_spot_paints_no_obstacles(self):
        """
        THE regression. A 360-degree in-place pivot on empty ground must leave
        the occupancy map empty. The old costmap memory failed exactly here.
        """
        m = LocalMap()
        for k in range(36):
            m.update_pose(_still(), yaw=math.radians(k * 10))
            m.integrate(
                clearance_by_bin=np.full(16, 6.0),
                bin_centers=np.linspace(-0.98, 0.98, 16),
            )
        self.assertEqual(int(m.occupied.sum()), 0)

    def test_one_observation_is_not_enough_to_occupy(self):
        """Log-odds, not a boolean: Depth Anything hallucinates, and a single
        bad frame must not become a permanent wall."""
        m = LocalMap()
        m.update_pose(_still(), yaw=0.0)
        m.integrate(obstacles_robot=np.array([[2.0, 0.0]]))
        self.assertEqual(int(m.occupied.sum()), 0)

    def test_repeated_observation_does_occupy(self):
        m = LocalMap()
        for _ in range(3):
            m.update_pose(_still(), yaw=0.0)
            m.integrate(obstacles_robot=np.array([[2.0, 0.0]]))
        self.assertGreater(int(m.occupied.sum()), 0)

    def test_free_space_evidence_erases_a_phantom(self):
        m = LocalMap()
        for _ in range(3):
            m.update_pose(_still(), yaw=0.0)
            m.integrate(obstacles_robot=np.array([[2.0, 0.0]]))
        self.assertGreater(int(m.occupied.sum()), 0)
        for _ in range(30):
            m.update_pose(_still(), yaw=0.0)
            m.integrate(clearance_by_bin=np.full(16, 6.0),
                        bin_centers=np.linspace(-0.98, 0.98, 16))
        self.assertEqual(int(m.occupied.sum()), 0)

    def test_driving_over_ground_marks_it_driven_and_clears_it(self):
        m = LocalMap()
        m.update_pose(_still(), yaw=0.0)
        m.integrate()
        self.assertGreater(int(m.driven.sum()), 0)


class TestPolarScan(unittest.TestCase):
    def _wall_map(self, lo=-1.5, hi=1.5):
        m = LocalMap()
        wall = np.stack([np.full(80, 2.0), np.linspace(lo, hi, 80)], axis=1)
        for _ in range(3):
            m.update_pose(_still(), yaw=0.0)
            m.integrate(obstacles_robot=wall)
        return m

    def test_wall_ahead_shows_as_reduced_forward_clearance(self):
        s = self._wall_map().clearance_polar(n_bins=36, max_range_m=8.0)
        fwd = s.clearance_m[int(np.argmin(np.abs(s.bin_centers)))]
        self.assertLess(fwd, 2.2)
        self.assertGreater(fwd, 0.5)

    def test_open_directions_keep_full_range(self):
        s = self._wall_map().clearance_polar(n_bins=36, max_range_m=8.0)
        behind = s.clearance_m[int(np.argmin(np.abs(
            np.abs(s.bin_centers) - math.pi)))]
        self.assertAlmostEqual(behind, 8.0, places=6)

    def test_obstacle_persists_after_the_rover_turns_away(self):
        """
        The whole point of the map. A single-frame costmap forgets an obstacle
        the moment it leaves view, so the goal term drags the rover straight
        back into it — which is what made the detour oscillate.
        """
        m = self._wall_map()
        before = m.clearance_polar(n_bins=36, max_range_m=8.0)
        fwd_before = before.clearance_m[int(np.argmin(np.abs(before.bin_centers)))]
        # Turn 90 degrees left and observe nothing new.
        m.update_pose(_still(), yaw=math.radians(-90.0))
        after = m.clearance_polar(n_bins=36, max_range_m=8.0)
        # The wall is now on our right: bearing -90 in the robot frame.
        k = int(np.argmin(np.abs(after.bin_centers + math.pi / 2)))
        self.assertLess(after.clearance_m[k], fwd_before + 0.5)

    def test_unknown_fraction_is_measured_only_in_front_of_the_obstacle(self):
        """Space BEHIND a wall is not frontier — counting it would send the
        rover at walls that happen to have unmapped space behind them."""
        s = self._wall_map().clearance_polar(n_bins=36, max_range_m=8.0)
        self.assertTrue(np.all(s.unknown_frac >= 0.0))
        self.assertTrue(np.all(s.unknown_frac <= 1.0))


class TestRecentring(unittest.TestCase):
    def test_rover_stays_inside_the_grid_over_a_long_drive(self):
        m = LocalMap()
        for _ in range(200):
            m.update_pose(OdomDelta(0.1, 0.0, 0.0, 1.0), yaw=0.0)
            m.integrate()
        ix, iy = m.world_to_cell(np.array([[m.x, m.y]]))
        self.assertTrue(0 <= int(ix[0]) < m.n)
        self.assertTrue(0 <= int(iy[0]) < m.n)
        self.assertAlmostEqual(m.x, 20.0, places=4)

    def test_recentring_does_not_wrap_stale_evidence_around(self):
        """Rolled-in strips are wrapped garbage, not observations."""
        m = LocalMap()
        for _ in range(3):
            m.update_pose(_still(), yaw=0.0)
            m.integrate(obstacles_robot=np.array([[1.0, 0.0]]))
        self.assertGreater(int(m.occupied.sum()), 0)
        for _ in range(300):
            m.update_pose(OdomDelta(0.1, 0.0, 0.0, 1.0), yaw=0.0)
        # Driven 30 m; the obstacle is long out of the window and must not have
        # reappeared on the far side.
        s = m.clearance_polar(n_bins=36, max_range_m=8.0)
        self.assertAlmostEqual(
            float(s.clearance_m[int(np.argmin(np.abs(s.bin_centers)))]),
            8.0, places=6,
        )


if __name__ == "__main__":
    unittest.main()
