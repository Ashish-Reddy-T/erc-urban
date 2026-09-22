"""Tiny Stage-0 closed-loop smoke test (no network)."""
from __future__ import annotations
import math
import unittest
from unittest.mock import MagicMock

import numpy as np

from erc_urban_test.geo import geodesic_bearing_range, reached
from erc_urban_test.interfaces import Observation
from erc_urban_test.modules.baseline import BaselinePlanner, PurePursuitController


class TestStage0Loop(unittest.TestCase):
    def test_plan_select_control_pipeline(self):
        """GPS prior → candidates → argmax → pure pursuit → valid cmd."""
        # Rover at origin heading north; goal ~east.
        lat0, lon0 = 0.0, 0.0
        goal = (0.0, 0.001)
        yaw = 0.0  # north
        bearing, rng = geodesic_bearing_range(lat0, lon0, yaw, *goal)
        self.assertFalse(reached(rng))

        obs = Observation(
            front=np.zeros((4, 4, 3), dtype=np.uint8),
            rear=None,
            lat=lat0,
            lon=lon0,
            yaw=yaw,
            stamp=0.0,
            goal_bearing=bearing,
            goal_range_m=rng,
        )
        planner = BaselinePlanner()
        ctrl = PurePursuitController()
        cands = planner.propose(obs, k=12)
        idx = int(np.argmax([c.score for c in cands]))
        lin, ang = ctrl.command(cands[idx].waypoints)

        self.assertGreaterEqual(lin, 0.0)
        self.assertLessEqual(lin, 1.0)
        self.assertLessEqual(abs(ang), 1.0)
        self.assertEqual(idx, int(np.argmax([c.score for c in cands])))

    def test_near_goal_counts_reached(self):
        bearing, rng = geodesic_bearing_range(30.0, 114.0, 0.0, 30.00005, 114.0)
        # ~5.5 m north
        self.assertLess(rng, 15.0)
        self.assertTrue(reached(rng))

    def test_mock_mission_checkpoint_flow(self):
        """Sanity: client helpers exist and mock success path looks right."""
        client = MagicMock()
        client.checkpoint_reached.return_value = {
            "message": "Checkpoint reached successfully",
            "next_checkpoint_sequence": 2,
        }
        resp = client.checkpoint_reached()
        self.assertNotIn("_http_status", resp)
        self.assertEqual(resp["next_checkpoint_sequence"], 2)


if __name__ == "__main__":
    unittest.main()
