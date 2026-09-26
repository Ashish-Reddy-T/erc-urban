"""ERC-3 turn_rad is image-right-positive; urban waypoints are left-positive."""
from __future__ import annotations
import math
import unittest

from erc_urban.run import depth_turn_to_wp


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


if __name__ == "__main__":
    unittest.main()
