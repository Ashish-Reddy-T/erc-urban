"""Depth-avoid override must survive pure pursuit's lookahead and slow the rover."""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban.modules.baseline import PurePursuitController
from erc_urban.run import depth_override_command


def _straight_plan() -> np.ndarray:
    return np.stack([np.linspace(0.4, 2.0, 5), np.zeros(5)], axis=1)


def _left_point(deg: float, scale: float) -> tuple[float, float]:
    """Robot-frame override point, left-positive, at the gate's `scale` distance."""
    a = math.radians(deg)
    return max(0.15, abs(math.cos(a)) * scale), math.sin(a) * scale


class TestDepthOverrideLookahead(unittest.TestCase):
    def test_left_dodge_at_turn_branch_scale_steers_left(self):
        # ERC-3's forward-blocked turn branch reports linear_scale 0.70, which
        # puts the override point inside both urban lookaheads.
        for lookahead in (0.8, 1.0):
            with self.subTest(lookahead=lookahead):
                ctl = PurePursuitController(lookahead_m=lookahead)
                _, ang = depth_override_command(
                    ctl, _straight_plan(), _left_point(40.0, 0.7), 0.7)
                self.assertGreater(ang, 0.0)

    def test_right_dodge_at_turn_branch_scale_steers_right(self):
        for lookahead in (0.8, 1.0):
            with self.subTest(lookahead=lookahead):
                ctl = PurePursuitController(lookahead_m=lookahead)
                _, ang = depth_override_command(
                    ctl, _straight_plan(), _left_point(-40.0, 0.7), 0.7)
                self.assertLess(ang, 0.0)

    def test_linear_scale_below_one_slows_the_rover(self):
        for lookahead in (0.8, 1.0):
            for deg in (0.0, 40.0):
                with self.subTest(lookahead=lookahead, deg=deg):
                    ctl = PurePursuitController(lookahead_m=lookahead)
                    slow, _ = depth_override_command(
                        ctl, _straight_plan(), _left_point(deg, 0.7), 0.7)
                    full, _ = depth_override_command(
                        ctl, _straight_plan(), _left_point(deg, 1.0), 1.0)
                    self.assertLess(slow, full)

    def test_planner_waypoints_are_not_mutated(self):
        wp = _straight_plan()
        before = wp.copy()
        depth_override_command(
            PurePursuitController(lookahead_m=0.8), wp, _left_point(40.0, 0.7), 0.7)
        np.testing.assert_array_equal(wp, before)


if __name__ == "__main__":
    unittest.main()
