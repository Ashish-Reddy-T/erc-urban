"""Tests for Stage-0 planner + pure pursuit."""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban_test.interfaces import Observation
from erc_urban_test.modules.baseline import BaselinePlanner, PurePursuitController


def _obs(goal_bearing: float = 0.0) -> Observation:
    return Observation(
        front=np.zeros((8, 8, 3), dtype=np.uint8),
        rear=None,
        lat=0.0,
        lon=0.0,
        yaw=0.0,
        stamp=1.0,
        goal_bearing=goal_bearing,
        goal_range_m=50.0,
    )


class TestBaselinePlanner(unittest.TestCase):
    def test_propose_count_and_shape(self):
        planner = BaselinePlanner()
        cands = planner.propose(_obs(), k=12)
        self.assertEqual(len(cands), 12)
        for c in cands:
            self.assertEqual(c.waypoints.shape[1], 2)
            self.assertEqual(c.waypoints.shape[0], 8)
            self.assertLessEqual(c.score, 0.0)

    def test_prefers_left_when_goal_is_left(self):
        planner = BaselinePlanner()
        # Goal ~30° left
        cands = planner.propose(_obs(goal_bearing=math.pi / 6), k=12)
        best = max(cands, key=lambda c: c.score)
        # Best end should have positive y (left)
        self.assertGreater(best.waypoints[-1, 1], 0.0)

    def test_prefers_straight_when_goal_ahead(self):
        planner = BaselinePlanner()
        cands = planner.propose(_obs(goal_bearing=0.0), k=11)  # odd → includes 0 curv
        best = max(cands, key=lambda c: c.score)
        self.assertAlmostEqual(best.score, 0.0, places=3)
        self.assertAlmostEqual(best.waypoints[-1, 1], 0.0, places=3)


class TestPurePursuit(unittest.TestCase):
    def test_straight_path_mostly_forward(self):
        ctrl = PurePursuitController()
        wp = np.stack([np.linspace(0, 3, 8), np.zeros(8)], axis=1)
        lin, ang = ctrl.command(wp)
        self.assertGreater(lin, 0.0)
        self.assertAlmostEqual(ang, 0.0, places=3)
        self.assertLessEqual(lin, 1.0)

    def test_left_path_positive_angular(self):
        ctrl = PurePursuitController()
        s = np.linspace(0, 3, 8)
        # gentle left arc endpoints with +y
        wp = np.stack([s, 0.3 * s], axis=1)
        lin, ang = ctrl.command(wp)
        self.assertGreater(ang, 0.0)
        self.assertGreaterEqual(lin, 0.0)
        self.assertLessEqual(abs(ang), 1.0)

    def test_right_path_negative_angular(self):
        ctrl = PurePursuitController()
        s = np.linspace(0, 3, 8)
        wp = np.stack([s, -0.3 * s], axis=1)
        _, ang = ctrl.command(wp)
        self.assertLess(ang, 0.0)


if __name__ == "__main__":
    unittest.main()
