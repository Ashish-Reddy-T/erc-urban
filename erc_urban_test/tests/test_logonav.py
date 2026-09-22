"""LogoNav planner adapter tests (fallback always; live load if weights+torch)."""
from __future__ import annotations
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from erc_urban_test.interfaces import Observation
from erc_urban_test.modules.logonav import LogoNavPlanner, _DEFAULT_WEIGHTS


def _obs(**kw) -> Observation:
    base = dict(
        front=np.zeros((64, 64, 3), dtype=np.uint8),
        rear=None,
        lat=19.57,
        lon=-98.81,
        yaw=0.0,
        stamp=1.0,
        goal_bearing=0.0,
        goal_range_m=40.0,
    )
    base.update(kw)
    return Observation(**base)


class TestLogoNavFallback(unittest.TestCase):
    def test_missing_weights_falls_back_to_baseline(self):
        p = LogoNavPlanner(checkpoint_path="/tmp/does_not_exist_logonav.pth")
        self.assertFalse(p.ready)
        cands = p.propose(_obs(), k=7)
        self.assertEqual(len(cands), 7)
        self.assertEqual(cands[0].waypoints.shape[1], 2)

    def test_set_goal_clears_context(self):
        p = LogoNavPlanner(checkpoint_path="/tmp/does_not_exist_logonav.pth")
        p.context_queue = [object()]  # type: ignore
        p.set_goal(19.57, -98.81)
        self.assertEqual(p.goal_latlon, (19.57, -98.81))
        self.assertEqual(p.context_queue, [])


@unittest.skipUnless(
    _DEFAULT_WEIGHTS.is_file(),
    "logonav.pth not downloaded",
)
class TestLogoNavLiveLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
            import utm  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"torch/utm missing: {e}")
        cls.planner = LogoNavPlanner(device="cpu")
        if not cls.planner.ready:
            raise unittest.SkipTest(cls.planner._load_error or "not ready")

    def test_warmup_then_logo_candidate(self):
        p = self.planner
        p.set_goal(19.570546, -98.814396)
        obs = _obs(
            front=np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8),
            lat=19.57022,
            lon=-98.81461,
            yaw=math.radians(30.0),
            goal_bearing=0.1,
        )
        warm = None
        for _ in range(p.context_size):
            warm = p.propose(obs, k=4)
            self.assertEqual(len(warm), 4)
            # Still baseline-only during warmup (scores <= 0).
            self.assertTrue(all(c.score <= 0.0 for c in warm))
        hot = p.propose(obs, k=4)
        self.assertEqual(len(hot), 4)
        self.assertGreater(hot[0].score, 0.0)
        self.assertEqual(hot[0].waypoints.shape[1], 2)


if __name__ == "__main__":
    unittest.main()
