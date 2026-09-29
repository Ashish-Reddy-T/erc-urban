"""WheelOdometry over erc_indoor's Odometer, fed synthetic /data bursts."""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_indoor.odometry import Odometer
from erc_urban.modules.wheel_odom import (WheelOdometry, compass_turn_error, parse_gyro_axis,
                                          yaw_axis_from_wiggle)


def _burst(t0: float, rpm: float, wz: float, n: int = 5, dt: float = 0.02) -> dict:
    """One /data payload: n samples at 50 Hz, like the Mini+ sends."""
    ts = [t0 + i * dt for i in range(n)]
    return {
        "rpms": [[rpm, rpm, rpm, rpm, t] for t in ts],
        "gyros": [[0.0, 0.0, wz, t] for t in ts],
    }


def _feed(odo: Odometer, t_start: float, t_end: float, rpm: float, wz: float) -> None:
    t = t_start
    while t < t_end - 1e-9:
        odo._ingest(_burst(t, rpm, wz))
        t += 0.1


def _odometer(m_per_rev: float = 0.388) -> Odometer:
    odo = Odometer(client=None, m_per_rev=m_per_rev)   # thread never started
    odo.bias = 0.0
    return odo


class TestWheelOdometry(unittest.TestCase):
    def test_straight_line_distance(self):
        odo = _odometer()
        w = WheelOdometry(odo)
        _feed(odo, 100.0, 100.5, rpm=60.0, wz=0.0)
        self.assertEqual(w.update(now=100.5).reason, "wheel_first")
        _feed(odo, 100.5, 101.5, rpm=60.0, wz=0.0)
        d = w.update(now=101.5)
        self.assertEqual(d.reason, "wheel")
        self.assertEqual(d.confidence, 1.0)
        # 60 rpm = 1 rev/s = 0.388 m/s for ~1 s.
        self.assertAlmostEqual(d.dx, 0.388, delta=0.03)
        self.assertAlmostEqual(d.dy, 0.0, places=6)

    def test_left_turn_is_positive_dyaw_and_bends_left(self):
        odo = _odometer()
        w = WheelOdometry(odo)
        _feed(odo, 100.0, 100.5, rpm=60.0, wz=20.0)
        w.update(now=100.5)
        _feed(odo, 100.5, 101.5, rpm=60.0, wz=20.0)
        d = w.update(now=101.5)
        self.assertAlmostEqual(math.degrees(d.dyaw), 20.0, delta=2.0)
        self.assertGreater(d.dy, 0.0)

    def test_wheel_constant_is_a_parameter(self):
        odo = _odometer(m_per_rev=0.5)
        w = WheelOdometry(odo)
        _feed(odo, 100.0, 100.5, rpm=60.0, wz=0.0)
        w.update(now=100.5)
        _feed(odo, 100.5, 101.5, rpm=60.0, wz=0.0)
        self.assertAlmostEqual(w.update(now=101.5).dx, 0.5, delta=0.04)

    def test_stale_telemetry_has_zero_confidence(self):
        odo = _odometer()
        w = WheelOdometry(odo, stale_s=2.5)
        _feed(odo, 100.0, 101.0, rpm=60.0, wz=0.0)
        w.update(now=101.0)
        d = w.update(now=110.0)
        self.assertEqual(d.confidence, 0.0)
        self.assertEqual(d.reason, "wheel_stale")

    def test_distance_per_tick_is_capped(self):
        odo = _odometer()
        w = WheelOdometry(odo, max_speed_mps=0.9)
        _feed(odo, 100.0, 100.5, rpm=60.0, wz=0.0)
        w.update(now=100.5)
        _feed(odo, 100.5, 101.5, rpm=600.0, wz=0.0)   # 3.9 m/s: a glitch
        self.assertLessEqual(w.update(now=101.5).distance_m, 0.9 * 1.0 + 1e-6)

    def test_dead_rpm_detected_under_forward_command(self):
        """One Mini+ read [0,0,0,0] while driving: detect it, don't trust it."""
        odo = _odometer()
        w = WheelOdometry(odo, dead_after_s=4.0)
        for k in range(12):
            t = 100.0 + 0.5 * k
            _feed(odo, t, t + 0.5, rpm=0.0, wz=0.0)
            w.update(commanding_forward=True, now=t + 0.5)
        self.assertTrue(w.rpm_dead)

    def test_stalled_after_moving_is_not_dead(self):
        """Wheels that have reported travel and then stall are STUCK, not dead."""
        odo = _odometer()
        w = WheelOdometry(odo, dead_after_s=4.0)
        _feed(odo, 100.0, 101.0, rpm=60.0, wz=0.0)
        w.update(commanding_forward=True, now=101.0)
        for k in range(12):
            t = 101.0 + 0.5 * k
            _feed(odo, t, t + 0.5, rpm=0.0, wz=0.0)
            w.update(commanding_forward=True, now=t + 0.5)
        self.assertFalse(w.rpm_dead)

    def test_not_ready_before_samples(self):
        w = WheelOdometry(_odometer())
        d = w.update(now=1.0)
        self.assertEqual(d.confidence, 0.0)
        self.assertTrue(np.isfinite(d.dx))


def _spin(t0: float, axis: int, rate: float, bias=(0.36, 0.5, 1.2), n: int = 25) -> list:
    """Raw (hw_t, gx, gy, gz) at 10 Hz: 0.8 s at `rate` on `axis`, then still."""
    out = []
    for i in range(n):
        g = list(bias)
        if i < 8:
            g[axis] += rate
        out.append((t0 + 0.1 * i, *g))
    return out


class TestYawAxis(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_gyro_axis("z"), (2, 1.0))
        self.assertEqual(parse_gyro_axis("-x"), (0, -1.0))
        self.assertIsNone(parse_gyro_axis("auto"))

    def test_inert_eel_code_turns_on_x(self):
        """Wuhan inert-eel-code, 2026-09-25: left spin +34 deg/s on x, z flat."""
        res = yaw_axis_from_wiggle(_spin(0.0, 0, +34.0), _spin(5.0, 0, -33.0))
        self.assertEqual(res[:2], (0, 1.0))

    def test_mini_plus_z_and_flipped_sign(self):
        self.assertEqual(yaw_axis_from_wiggle(_spin(0, 2, 30.0), _spin(5, 2, -30.0))[:2], (2, 1.0))
        self.assertEqual(yaw_axis_from_wiggle(_spin(0, 1, -30.0), _spin(5, 1, 30.0))[:2], (1, -1.0))

    def test_no_spin_is_inconclusive(self):
        self.assertIsNone(yaw_axis_from_wiggle(_spin(0, 2, 0.0), _spin(5, 2, 0.0)))
        self.assertIsNone(yaw_axis_from_wiggle([], _spin(5, 2, -30.0)))

    def test_odometer_integrates_chosen_axis(self):
        odo = _odometer()
        odo.set_yaw_axis(0, 1.0)
        for k in range(10):
            t = 100.0 + 0.1 * k
            odo._ingest({"rpms": [[0, 0, 0, 0, t]], "gyros": [[30.0, 0.0, 0.0, t]]})
        self.assertAlmostEqual(odo.state().yaw, 27.0, delta=0.5)


class TestCompassCheck(unittest.TestCase):
    def test_good_compass_follows_the_wiggle(self):
        """inert-eel-code: a 25 deg left spin moved the compass 4 -> 339 and back."""
        self.assertLess(compass_turn_error(4.0, 339.0, 3.0, 25.0, -24.0), 5.0)

    def test_panama_compass_is_flagged(self):
        """hinge-moto-nova: a 38 deg left turn moved it 265 -> 73."""
        self.assertGreater(compass_turn_error(265.0, 73.0, 301.0, 38.0, -38.0), 20.0)

    def test_frozen_compass_is_flagged(self):
        self.assertGreater(compass_turn_error(120.0, 120.0, 120.0, 25.0, -25.0), 20.0)

    def test_missing_reading(self):
        self.assertIsNone(compass_turn_error(None, 10.0, 0.0, 25.0, -25.0))


class TestTiltGuard(unittest.TestCase):
    def test_tilt_from_startup_gravity(self):
        from types import SimpleNamespace
        import erc_urban.run as R
        old = R._GRAVITY
        try:
            R._GRAVITY = np.array([0.0, 0.0, 1.0])
            level = SimpleNamespace(raw={"accels": [[0.04, -0.05, 1.0, 1.0]]})
            up = SimpleNamespace(raw={"accels": [[0.6, 0.0, 0.8, 1.0]]})      # ~37 deg
            self.assertLess(R._tilt_deg(level), 5.0)
            self.assertGreater(R._tilt_deg(up), 30.0)
            R._GRAVITY = None
            self.assertIsNone(R._tilt_deg(up))
        finally:
            R._GRAVITY = old


if __name__ == "__main__":
    unittest.main()
