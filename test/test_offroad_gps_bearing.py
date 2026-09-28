"""Offroad GPS relative bearing must be left-positive (matches urban geo)."""
from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from erc_offroad.auto import choose_line, gps_relative_bearing_rad
from erc_offroad.gpswatch import geo


class TestOffroadGpsBearing(unittest.TestCase):
    def test_facing_north_goal_due_east_is_right_negative(self):
        # gpswatch brg/yaw are CW-from-North degrees. Terrain/choose_line treat
        # positive as LEFT, so east-of-north must be negative (right).
        goal_b = gps_relative_bearing_rad(brg_cw_deg=90.0, yaw_cw_deg=0.0)
        self.assertAlmostEqual(goal_b, -math.pi / 2, places=5)

    def test_choose_line_steers_right_toward_east_checkpoint(self):
        # Facing north, checkpoint ~50 m due east. Uniformly clear terrain with
        # the same left-positive column bearings terrain.analyze produces.
        brg, _ = geo(40.0, -74.0, 40.0, -74.0 + 50.0 / 85000.0)
        self.assertAlmostEqual(brg, 90.0, delta=0.5)
        centres = (np.arange(12) + 0.5) / 12
        bearings = np.radians((0.5 - centres) * 90.0)
        tv = SimpleNamespace(bearings=bearings, clear=np.ones(12),
                             best_bearing=0.0)
        aim, _ = choose_line(tv, gps_relative_bearing_rad(brg, 0.0))
        self.assertLess(aim, 0.0, "east goal must pick a right (negative) column")


if __name__ == "__main__":
    unittest.main()
