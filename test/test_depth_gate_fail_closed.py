"""Depth gate must fail closed when inference raises."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from erc_urban.modules.depth_gate import DepthObstacleGate


class TestDepthGateFailClosed(unittest.TestCase):
    def test_evaluate_blocks_when_estimator_raises(self):
        with patch.object(DepthObstacleGate, "_try_load"):
            gate = DepthObstacleGate(every_n=1)

        boom = MagicMock()
        boom.estimate.side_effect = RuntimeError("infer_fail")
        gate._estimator = boom
        gate._trav = MagicMock()

        frame = np.zeros((576, 1024, 3), dtype=np.uint8)
        out = gate.evaluate(frame, 0.0)

        self.assertTrue(out.blocked)
        self.assertEqual(out.linear_scale, 0.0)
        self.assertEqual(out.forward_m, 0.0)
        self.assertTrue(out.reason.startswith("depth_err:"))


if __name__ == "__main__":
    unittest.main()
