"""Tests for the SE(2) pose-buffer realignment helper (erc_urban.modules.upgrades)."""
from __future__ import annotations

import math
import unittest

import numpy as np

from erc_urban.modules.upgrades import PoseBuffer


class FakeDelta:
    def __init__(self, dx=0.0, dy=0.0, dyaw=0.0, confidence=1.0):
        self.dx = dx
        self.dy = dy
        self.dyaw = dyaw
        self.confidence = confidence


class TestPoseBufferIntegrate(unittest.TestCase):
    def test_starts_at_origin(self):
        pb = PoseBuffer()
        self.assertEqual((pb.x, pb.y, pb.yaw), (0.0, 0.0, 0.0))

    def test_forward_motion_with_no_rotation_accumulates_x(self):
        pb = PoseBuffer()
        pb.integrate(1.0, FakeDelta(dx=1.0, dy=0.0, dyaw=0.0))
        self.assertAlmostEqual(pb.x, 1.0)
        self.assertAlmostEqual(pb.y, 0.0)
        self.assertAlmostEqual(pb.yaw, 0.0)

    def test_rotation_then_forward_motion_rotates_into_odom_frame(self):
        pb = PoseBuffer()
        # Turn 90 degrees left (positive yaw), then step "forward" in the new
        # robot frame -- that step must land along the odom-frame Y axis, not X.
        pb.integrate(1.0, FakeDelta(dyaw=math.pi / 2))
        pb.integrate(2.0, FakeDelta(dx=1.0, dy=0.0, dyaw=0.0))
        self.assertAlmostEqual(pb.x, 0.0, places=6)
        self.assertAlmostEqual(pb.y, 1.0, places=6)

    def test_none_delta_holds_position_but_still_records_a_sample(self):
        pb = PoseBuffer()
        pb.integrate(1.0, FakeDelta(dx=1.0))
        x0, y0 = pb.x, pb.y
        pb.integrate(2.0, None)
        self.assertEqual((pb.x, pb.y), (x0, y0))
        self.assertEqual(len(pb.buf), 2)
        self.assertEqual(pb.buf[-1][0], 2.0)

    def test_zero_confidence_delta_holds_position_but_records_a_sample(self):
        pb = PoseBuffer()
        pb.integrate(1.0, FakeDelta(dx=1.0))
        x0, y0 = pb.x, pb.y
        pb.integrate(2.0, FakeDelta(dx=5.0, dy=5.0, confidence=0.0))
        self.assertEqual((pb.x, pb.y), (x0, y0))


class TestPoseBufferNearest(unittest.TestCase):
    def test_nearest_on_empty_buffer_is_none(self):
        pb = PoseBuffer()
        self.assertIsNone(pb._nearest(1.0))

    def test_nearest_picks_closest_timestamp(self):
        pb = PoseBuffer()
        pb.push(1.0, 0.0, 0.0, 0.0)
        pb.push(3.0, 1.0, 0.0, 0.0)
        pb.push(5.0, 2.0, 0.0, 0.0)
        nearest = pb._nearest(3.4)
        self.assertEqual(nearest[0], 3.0)


class TestPoseBufferRealign(unittest.TestCase):
    def test_realign_returns_waypoints_unchanged_when_buffer_empty(self):
        pb = PoseBuffer()
        wp = np.array([[1.0, 0.0], [2.0, 0.0]])
        out = pb.realign(wp, t_capture=0.0, t_now=1.0)
        self.assertTrue(np.array_equal(out, wp))

    def test_realign_shifts_waypoints_by_translation_only(self):
        pb = PoseBuffer()
        pb.push(0.0, 0.0, 0.0, 0.0)
        pb.push(1.0, 0.5, 0.0, 0.0)  # moved 0.5m forward, no rotation
        wp = np.array([[2.0, 0.0]])
        out = pb.realign(wp, t_capture=0.0, t_now=1.0)
        self.assertTrue(np.allclose(out, np.array([[1.5, 0.0]])))

    def test_realign_accounts_for_rotation_between_capture_and_now(self):
        pb = PoseBuffer()
        pb.push(0.0, 0.0, 0.0, 0.0)
        pb.push(1.0, 0.0, 0.0, math.pi / 2)  # rover turned left 90deg, no translation
        wp = np.array([[1.0, 0.0]])
        out = pb.realign(wp, t_capture=0.0, t_now=1.0)
        # A waypoint that was "straight ahead" at capture time is now to the
        # robot's right (since the robot itself turned left).
        self.assertAlmostEqual(out[0, 0], 0.0, places=6)
        self.assertAlmostEqual(out[0, 1], -1.0, places=6)


if __name__ == "__main__":
    unittest.main()
