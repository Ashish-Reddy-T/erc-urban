"""World-frame obstacle memory, full-circle heading choice, and the grid plan."""
import math
import os
import unittest

import numpy as np

from erc_urban.calib import CameraCalibration
from erc_urban.modules.freespace import FreeSpaceUrban
from erc_urban.modules.obstacle_memory import ObstacleMemory, choose_full, plan_heading

HW = (288, 512)
CAL = CameraCalibration.load_or_default(
    os.path.join(os.path.dirname(__file__), "..", "erc_urban", "camera_front.json"), 1024, 576)


class TestMemoryFrames(unittest.TestCase):
    def test_point_stays_put_in_the_world(self):
        m = ObstacleMemory()
        m.add(np.array([[1.0, 0.0]]), 0.0, now=0.0)          # 1 m ahead, facing north
        m.update_pose(0.5, 0.0, 0.0)                          # drive 0.5 m north
        np.testing.assert_allclose(m.to_robot(0.0), [[0.5, 0.0]], atol=1e-6)
        # Facing east now: the point (north of us) is on the LEFT.
        np.testing.assert_allclose(m.to_robot(math.radians(90.0)), [[0.0, 0.5]], atol=1e-6)

    def test_gyro_frame_turn(self):
        """Turn 90 deg left by odometry: a point that was ahead is now on the right."""
        m = ObstacleMemory()
        m.add(np.array([[1.0, 0.0]]), m.yaw, now=0.0)
        m.step_odom(0.0, 0.0, math.radians(90.0))
        np.testing.assert_allclose(m.to_robot(m.yaw), [[0.0, -1.0]], atol=1e-6)

    def test_reverse_moves_backwards(self):
        m = ObstacleMemory()
        m.add(np.array([[1.0, 0.0]]), 0.0, now=0.0)
        m.update_pose(-0.5, 0.0, 0.0)
        np.testing.assert_allclose(m.to_robot(0.0), [[1.5, 0.0]], atol=1e-6)

    def test_seen_free_clears_confirmed_after_three_ticks(self):
        m = ObstacleMemory()
        m.add(np.array([[2.0, 0.0]]), 0.0, now=0.0)
        for k in range(2):
            m.clear_seen(np.zeros((0, 2)), 0.0)
            self.assertEqual(len(m), 1, k)
        m.clear_seen(np.zeros((0, 2)), 0.0)
        self.assertEqual(len(m), 0)

    def test_felt_obstacle_survives_open_view(self):
        """Machakos: wheels stalled on a root under grass; vision says open."""
        m = ObstacleMemory()
        m.add(np.array([[0.6, 0.0]]), 0.0, now=0.0, sticky=True)
        m.add(np.array([[2.0, 1.0]]), 0.0, now=0.0)
        for _ in range(10):
            m.clear_seen(np.zeros((0, 2)), 0.0)
        self.assertEqual(len(m), 1)
        m.step_odom(45.0, 0.0, 0.0)                      # far past keep_m
        m.prune(1.0)
        self.assertEqual(len(m), 0)

    def test_still_there_is_kept(self):
        m = ObstacleMemory()
        m.add(np.array([[2.0, 0.0]]), 0.0, now=0.0)
        for _ in range(5):
            m.clear_seen(np.array([[2.05, 0.0]]), 0.0)
        self.assertEqual(len(m), 1)

    def test_seen_points_are_short_lived(self):
        m = ObstacleMemory(seen_keep_m=8.0)
        m.observe(np.array([[1.5, 0.3], [0.2, 0.0], [4.0, 0.0]]), 0.0, now=0.0)
        self.assertEqual(len(m), 1)                            # only 0.5..2.5 m
        m.update_pose(0.0, 8.5, 0.0)                           # sidestep 8.5 m
        m.prune(now=1.0)
        self.assertEqual(len(m), 0)

    def test_passers_by_are_not_planned_around(self):
        """Seen once (a walker) is ignored at min_hits=3; seen 3x (static) is kept."""
        m = ObstacleMemory()
        m.observe(np.array([[1.5, 0.0]]), 0.0, now=0.0)
        self.assertEqual(len(m.to_robot(0.0, min_hits=3)), 0)
        m.observe(np.array([[1.5, 0.0]]), 0.0, now=0.3)
        m.observe(np.array([[1.5, 0.0]]), 0.0, now=0.6)
        self.assertEqual(len(m.to_robot(0.0, min_hits=3)), 1)
        m.add(np.array([[2.0, 1.0]]), 0.0, now=1.0)            # confirmed: always
        self.assertEqual(len(m.to_robot(0.0, min_hits=3)), 2)

    def test_transient_batch_is_dropped(self):
        m = ObstacleMemory()
        b = m.add(np.array([[1.0, 0.0], [1.0, 0.2]]), 0.0, now=0.0)
        m.drop_batch(b)
        self.assertEqual(len(m), 0)


class TestChooseFull(unittest.TestCase):
    def setUp(self):
        self.fs = FreeSpaceUrban(CAL, seg_hw=HW).estimate(np.ones(HW, bool))

    def test_open_goes_to_goal_in_view(self):
        h, which, inview = choose_full(self.fs, 10.0, 1.2, 0.8, None)
        self.assertEqual((h, which, inview), (10.0, "track", True))

    def test_grass_field_is_crossed_not_turned_away_from(self):
        """Panama: track ends 0.2 m out, grass clear 5 m, goal ahead -> drive on grass."""
        track = np.zeros(HW, bool)
        grass = np.ones(HW, bool)
        fs = FreeSpaceUrban(CAL, seg_hw=HW).estimate(track, grass)
        h, which, inview = choose_full(fs, 0.0, 1.2, 0.8, None)
        self.assertTrue(inview)
        self.assertEqual(which, "grass")
        self.assertLessEqual(abs(h), 5.0)

    def test_goal_behind_turns_round(self):
        h, _, inview = choose_full(self.fs, 180.0, 1.2, 0.8, None)
        self.assertFalse(inview)
        self.assertGreater(abs(h), 150.0)

    def test_remembered_wall_is_not_turned_into(self):
        """Wall remembered on the right (out of view), goal behind it."""
        wall = np.stack([np.linspace(-2.0, 3.0, 51), np.full(51, -1.0)], 1)
        h, _, _ = choose_full(self.fs, -90.0, 1.2, 0.8, wall)
        self.assertGreater(h, -50.0)

    def test_corner_beside_blocks_turning_into_it(self):
        corner = np.array([[0.05, 0.22]])                      # just off the left front
        h, _, _ = choose_full(self.fs, 60.0, 1.2, 0.8, corner)
        self.assertLessEqual(h, 15.0)


class TestPlanHeading(unittest.TestCase):
    def test_nothing_remembered(self):
        self.assertIsNone(plan_heading(np.zeros((0, 2)), 20.0, 10.0))

    def test_free_follows_goal(self):
        for want in (0.0, 20.0, -35.0, 120.0):
            h = plan_heading(np.array([[0.0, 5.0]]), want, 10.0)
            self.assertLess(abs((h - want + 180) % 360 - 180), 5.0, want)

    def test_wall_goes_round_the_near_end(self):
        wall = np.stack([np.full(41, 1.0), np.linspace(-3.0, 1.0, 41)], 1)
        self.assertGreater(plan_heading(wall, 0.0, 10.0), 30.0)

    def test_pocket_backs_out(self):
        pk = np.vstack([np.stack([np.full(25, 1.0), np.linspace(-1.2, 1.2, 25)], 1),
                        np.stack([np.linspace(-2, 1, 31), np.full(31, 1.2)], 1),
                        np.stack([np.linspace(-2, 1, 31), np.full(31, -1.2)], 1)])
        self.assertGreater(abs(plan_heading(pk, 0.0, 10.0)), 120.0)


class TestClosedLoop(unittest.TestCase):
    """erc_urban.sim_fan: the real _fan_step in a 2D world of walls."""

    def _run(self, name):
        from erc_urban import sim_fan
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return sim_fan.run(sim_fan.SCENARIOS[name])

    def test_pocket(self):
        r = self._run("pocket")
        self.assertTrue(r.reached)
        self.assertEqual(r.collisions, 0)

    def test_wall_end(self):
        r = self._run("offset_wall")
        self.assertTrue(r.reached)
        self.assertEqual(r.collisions, 0)

    def test_pillar_tie(self):
        r = self._run("pillar")
        self.assertTrue(r.reached)
        self.assertEqual(r.collisions, 0)


if __name__ == "__main__":
    unittest.main()
