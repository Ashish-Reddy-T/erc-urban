"""Depth gate must fail closed when inference raises."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from erc_urban_test.modules.depth_gate import DepthClearance, DepthObstacleGate


def _gate(every_n: int) -> DepthObstacleGate:
    with patch.object(DepthObstacleGate, "_try_load"):
        gate = DepthObstacleGate(every_n=every_n)
    gate._estimator = MagicMock()
    gate._estimator.estimate.side_effect = RuntimeError("infer_fail")
    gate._trav = MagicMock()
    return gate


class TestDepthGateFailClosed(unittest.TestCase):
    frame = np.zeros((576, 1024, 3), dtype=np.uint8)

    def test_evaluate_blocks_when_estimator_raises(self):
        out = _gate(every_n=1).evaluate(self.frame, 0.0)

        self.assertTrue(out.blocked)
        self.assertEqual(out.linear_scale, 0.0)
        self.assertEqual(out.forward_m, 0.0)
        self.assertTrue(out.reason.startswith("depth_err:"))

    def test_error_persists_on_skipped_ticks(self):
        # With every_n > 1 the ticks between inferences replay _last. An
        # inference error must replace an earlier clear result there, or the
        # rover drives on the tick right after the crash.
        gate = _gate(every_n=2)
        gate._last = DepthClearance(False, 5.0, "depth_ok(5.00m)", 1.0, None)
        gate._tick = 1  # next evaluate() is a sampled tick

        first = gate.evaluate(self.frame, 0.0)
        skipped = gate.evaluate(self.frame, 0.0)

        self.assertEqual(gate._estimator.estimate.call_count, 1)
        self.assertTrue(first.blocked)
        self.assertTrue(skipped.blocked)
        self.assertEqual(skipped.linear_scale, 0.0)
        self.assertTrue(skipped.reason.startswith("depth_err:"))


if __name__ == "__main__":
    unittest.main()
