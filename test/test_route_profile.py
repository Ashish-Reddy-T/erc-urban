"""Tests for mission route shaping / sidewalk rail targets."""
from __future__ import annotations

import math
import unittest

from erc_urban.geo import geodesic_bearing_range
from erc_urban.modules.route_profile import build_route_profile


RWARE = [
    (-0.4144954681, 36.95953369),
    (-0.4138350785, 36.95928192),
    (-0.4128639698, 36.95890808),
    (-0.4121999145, 36.95863724),
    (-0.4115463197, 36.95835114),
    (-0.411362201, 36.95827103),
    (-0.4109404683, 36.95808792),
    (-0.4099167585, 36.95764923),
    (-0.4091888368, 36.95849609),
    (-0.4099167585, 36.95764923),
    (-0.4109404683, 36.95808792),
    (-0.411362201, 36.95827103),
    (-0.4115463197, 36.95835114),
    (-0.4121999145, 36.95863724),
    (-0.4128639698, 36.95890808),
    (-0.4138350785, 36.95928192),
    (-0.4144954681, 36.95953369),
]


class TestRwareRouteProfile(unittest.TestCase):
    def test_auto_detects_known_rware_mission(self):
        route = build_route_profile(RWARE, profile="auto")
        self.assertIsNotNone(route)
        self.assertIn("rware_sidewalk", route.describe())

    def test_unrelated_goals_are_left_raw_in_auto(self):
        route = build_route_profile([(40.0, -73.0), (40.001, -73.001)], profile="auto")
        self.assertIsNone(route)

    def test_first_leg_target_is_short_lookahead_on_sidewalk_rail(self):
        route = build_route_profile(RWARE, profile="rware_sidewalk")
        target = route.target(1, *RWARE[0])
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target.lookahead_m, 10.0, places=3)
        self.assertAlmostEqual(target.lateral_offset_m, -3.0, places=3)

        raw_bearing, _ = geodesic_bearing_range(
            RWARE[0][0], RWARE[0][1], 0.0, RWARE[1][0], RWARE[1][1]
        )
        rail_bearing, _ = geodesic_bearing_range(
            RWARE[0][0], RWARE[0][1], 0.0, target.lat, target.lon
        )
        self.assertLess(abs(math.degrees(raw_bearing - rail_bearing)), 20.0)

    def test_return_leg_reuses_same_absolute_sidewalk_side(self):
        route = build_route_profile(RWARE, profile="rware_sidewalk")
        outbound_cp8 = route.route_xy[7]
        return_cp10 = route.route_xy[9]
        self.assertLess(float(((outbound_cp8 - return_cp10) ** 2).sum()), 1e-9)

    def test_default_rail_matches_live_sidewalk_fix(self):
        """Live run stopped on the actual sidewalk west of the checkpoint chord."""
        route = build_route_profile(RWARE, profile="rware_sidewalk")
        live_fix = (-0.41447386145591736, 36.95949935913086)
        target = route.target(1, *live_fix)
        self.assertLess(abs(target.cross_track_m), 1.0)


if __name__ == "__main__":
    unittest.main()
