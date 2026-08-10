"""Tests for GPS → robot-frame geometry."""
from __future__ import annotations
import math
import unittest

from erc_urban.geo import geodesic_bearing_range, reached


class TestGeo(unittest.TestCase):
    def test_reached_tolerance(self):
        self.assertTrue(reached(0.0))
        self.assertTrue(reached(15.0))
        self.assertFalse(reached(15.1))
        self.assertTrue(reached(20.0, tol_m=25.0))

    def test_zero_range_same_point(self):
        bearing, rng = geodesic_bearing_range(30.0, 114.0, 0.0, 30.0, 114.0)
        self.assertAlmostEqual(rng, 0.0, places=3)
        self.assertAlmostEqual(bearing, 0.0, places=5)

    def test_due_east_with_north_heading(self):
        # ~111 m east at equator-ish; use small dlon at lat 0.
        lat0, lon0 = 0.0, 0.0
        lat1, lon1 = 0.0, 0.001  # ~111 m east
        bearing, rng = geodesic_bearing_range(lat0, lon0, yaw=0.0, lat1=lat1, lon1=lon1)
        # yaw=0 = North, goal east → RIGHT in left-positive frame ≈ −π/2
        self.assertAlmostEqual(bearing, -math.pi / 2, places=2)
        self.assertGreater(rng, 100.0)
        self.assertLess(rng, 120.0)

    def test_orientation_degrees_converted_by_caller(self):
        # Facing east (90° = π/2 rad), goal also east → bearing ~0 in robot frame.
        lat0, lon0 = 0.0, 0.0
        lat1, lon1 = 0.0, 0.001
        yaw = math.radians(90.0)
        bearing, rng = geodesic_bearing_range(lat0, lon0, yaw, lat1, lon1)
        self.assertAlmostEqual(bearing, 0.0, places=2)
        self.assertGreater(rng, 100.0)

    def test_yaw_none_returns_north_relative(self):
        # Without yaw, still left-positive vs North: east → −π/2
        bearing, rng = geodesic_bearing_range(0.0, 0.0, None, 0.0, 0.001)
        self.assertAlmostEqual(bearing, -math.pi / 2, places=2)


if __name__ == "__main__":
    unittest.main()
