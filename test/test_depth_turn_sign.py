"""ERC-3 depth yaw is image-right-positive; urban bearings and waypoints are left-positive."""
from __future__ import annotations
import math
import unittest
from unittest.mock import patch

import numpy as np

from erc_urban.modules import depth_gate as dg_mod
from erc_urban.run import depth_evaluate, depth_turn_to_wp


class TestDepthTurnToWp(unittest.TestCase):
    def test_positive_erc3_turn_steers_right_in_urban_frame(self):
        # ERC-3 outdoor_traversability: arctan((u-cx)/fx) → positive = image right.
        # Urban robot frame: y left-positive, so right steer must be y < 0.
        x, y = depth_turn_to_wp(math.radians(30.0), scale=1.0)
        self.assertGreater(x, 0.0)
        self.assertLess(y, 0.0)

    def test_negative_erc3_turn_steers_left_in_urban_frame(self):
        x, y = depth_turn_to_wp(math.radians(-30.0), scale=1.0)
        self.assertGreater(x, 0.0)
        self.assertGreater(y, 0.0)


class _FixedDepth:
    def __init__(self, depth):
        self.depth = depth

    def estimate(self, front, target_size):
        return self.depth


def _gate(depth):
    """Real DepthObstacleGate + real ERC-3 OutdoorTraversability, fake depth model."""
    with patch.object(dg_mod.DepthObstacleGate, "_try_load", lambda self: None):
        gate = dg_mod.DepthObstacleGate(every_n=1)
    import sys
    s = str(dg_mod._ERC3_SRC)
    if s not in sys.path:
        sys.path.insert(0, s)
    from outdoor_traversability import (  # type: ignore
        OutdoorTraversability,
        OutdoorTraversabilityConfig,
    )
    gate._estimator = _FixedDepth(depth)
    gate._trav = OutdoorTraversability(OutdoorTraversabilityConfig(
        obstacle_distance_m=gate.obstacle_m,
        stop_distance_m=gate.stop_m,
        slow_distance_m=gate.slow_m,
        memory_frames=1,
    ))
    return gate


def _wall_ahead(open_left: bool, open_right: bool):
    """0.8 m wall across the middle third; each outer third open (20 m) or walled."""
    d = np.full((144, 256), 0.8, dtype=np.float32)
    if open_left:
        d[:, :85] = 20.0
    if open_right:
        d[:, 171:] = 20.0
    return d


class TestDepthGateEndToEnd(unittest.TestCase):
    """Bearing in, waypoint out, through the real traversability code."""

    FRONT = np.zeros((576, 1024, 3), dtype=np.uint8)

    def _wp_y(self, depth, bearing):
        view = depth_evaluate(_gate(depth), self.FRONT, bearing)
        self.assertFalse(view.blocked, view.reason)
        self.assertIsNotNone(view.turn_rad, view.reason)
        return depth_turn_to_wp(view.turn_rad, max(0.25, view.linear_scale))[1]

    def test_only_left_open_steers_left(self):
        self.assertGreater(self._wp_y(_wall_ahead(True, False), 0.0), 0.0)

    def test_only_right_open_steers_right(self):
        self.assertLess(self._wp_y(_wall_ahead(False, True), 0.0), 0.0)

    def test_both_open_goal_left_takes_left_gap(self):
        # Goal 30 deg to the LEFT (left-positive bearing). The gate must pick
        # the left gap, not its mirror.
        self.assertGreater(self._wp_y(_wall_ahead(True, True), math.radians(30.0)), 0.0)

    def test_both_open_goal_right_takes_right_gap(self):
        self.assertLess(self._wp_y(_wall_ahead(True, True), math.radians(-30.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
