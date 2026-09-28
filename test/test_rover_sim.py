"""Tests for the offline rover sim: every modelled live fact gets a check."""
from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from erc_urban import sdk_client
from erc_urban.sdk_client import RoverClient, Telemetry

from .rover_sim import NO_FIX, FakeRoverClient, RoverSim, SimConfig


def _run(sim: RoverSim, seconds: float, dt: float = 0.05) -> None:
    for _ in range(int(round(seconds / dt))):
        sim.step(dt)


def _ang_diff(a: float, b: float) -> float:
    """Signed a - b in degrees, wrapped to (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


class TestKinematics(unittest.TestCase):
    def test_positive_angular_turns_left_compass_decreases(self):
        sim = RoverSim(SimConfig(start_heading_deg=90.0), seed=1)
        sim.command(0.0, 0.4)
        _run(sim, 1.0)
        self.assertLess(_ang_diff(sim.heading_deg, 90.0), 0.0)
        self.assertGreater(sim.yaw_rate_dps, 0.0)

    def test_negative_angular_turns_right(self):
        sim = RoverSim(SimConfig(start_heading_deg=90.0), seed=1)
        sim.command(0.0, -0.4)
        _run(sim, 1.0)
        self.assertGreater(_ang_diff(sim.heading_deg, 90.0), 0.0)

    def test_turn_rate_at_0p4_within_live_band_and_linear(self):
        sim = RoverSim(seed=1)
        sim.command(0.0, 0.4)
        _run(sim, 2.0)
        rate = -_ang_diff(sim.heading_deg, 0.0) / 2.0
        self.assertGreaterEqual(rate, 27.0)
        self.assertLessEqual(rate, 35.0)
        self.assertAlmostEqual(rate, 31.0, places=6)
        sim2 = RoverSim(seed=1)
        sim2.command(0.0, 0.2)
        _run(sim2, 2.0)
        self.assertAlmostEqual(-_ang_diff(sim2.heading_deg, 0.0) / 2.0, 15.5, places=6)

    def test_compass_is_clockwise_from_north(self):
        # Facing east (90) and driving forward moves east; facing north moves north.
        east = RoverSim(SimConfig(start_heading_deg=90.0), seed=1)
        east.command(1.0, 0.0)
        _run(east, 2.0)
        self.assertAlmostEqual(east.x, 2.0, places=6)
        self.assertAlmostEqual(east.y, 0.0, places=6)
        self.assertGreater(east.lon, east.cfg.origin_lon)
        north = RoverSim(SimConfig(start_heading_deg=0.0), seed=1)
        north.command(1.0, 0.0)
        _run(north, 2.0)
        self.assertAlmostEqual(north.y, 2.0, places=6)
        self.assertGreater(north.lat, north.cfg.origin_lat)
        # Right turn from north passes through east (clockwise).
        cw = RoverSim(seed=1)
        cw.command(0.0, -0.4)
        _run(cw, 90.0 / 31.0, dt=90.0 / 31.0 / 20)
        self.assertAlmostEqual(cw.heading_deg, 90.0, places=6)

    def test_orientation_in_0_360(self):
        sim = RoverSim(SimConfig(orientation_delay_s=0.0, orientation_period_s=0.0), seed=1)
        sim.command(0.0, 0.4)  # left from north wraps below 0
        _run(sim, 1.0)
        o = sim.data()["orientation"]
        self.assertGreaterEqual(o, 0.0)
        self.assertLess(o, 360.0)
        self.assertAlmostEqual(o, 360.0 - 31.0, places=1)

    def test_orientation_round_does_not_report_360(self):
        # 0.004 deg left of north is 359.996, which rounds to 360.00.
        sim = RoverSim(SimConfig(orientation_delay_s=0.0, orientation_period_s=0.0), seed=1)
        sim.command(0.0, 0.4)
        sim.step(0.004 / 31.0)
        o = sim.data()["orientation"]
        self.assertGreaterEqual(o, 0.0)
        self.assertLess(o, 360.0)
        self.assertAlmostEqual(o, 0.0, places=2)

    def test_arc_integration_closes_full_circle(self):
        sim = RoverSim(seed=1)
        sim.command(0.5, 0.4)
        period = 360.0 / 31.0
        _run(sim, period, dt=period / 7)  # coarse steps, exact arcs
        self.assertAlmostEqual(sim.x, 0.0, places=6)
        self.assertAlmostEqual(sim.y, 0.0, places=6)

    def test_command_clamped(self):
        sim = RoverSim(seed=1)
        sim.command(3.0, -2.0)
        self.assertEqual((sim.linear, sim.angular), (1.0, -1.0))


class TestOrientationLag(unittest.TestCase):
    def _turning_sim(self):
        sim = RoverSim(SimConfig(start_heading_deg=180.0), seed=1)
        sim.command(0.0, 0.4)
        return sim

    def test_dead_time_1p4s(self):
        sim = self._turning_sim()
        while sim.t < 1.4 - 1e-6:
            self.assertAlmostEqual(sim.data()["orientation"], 180.0, places=6, msg=f"t={sim.t:.2f}")
            sim.step(0.05)
        _run(sim, 0.4 + 0.05)
        self.assertLess(sim.data()["orientation"], 180.0)

    def test_reports_heading_from_1p4s_ago_on_grid(self):
        sim = self._turning_sim()
        _run(sim, 4.0)
        # At t=4.0 (a grid point) the report is truth at t=2.6.
        self.assertAlmostEqual(sim.data()["orientation"], 180.0 - 31.0 * 2.6, places=1)

    def test_updates_in_0p4s_steps(self):
        sim = self._turning_sim()
        _run(sim, 2.0)
        changes, last = [], sim.data()["orientation"]
        for _ in range(int(round(4.0 / 0.05))):
            sim.step(0.05)
            o = sim.data()["orientation"]
            if o != last:
                changes.append(round(sim.t, 6))
                last = o
        gaps = np.diff(changes)
        self.assertGreaterEqual(len(changes), 9)
        np.testing.assert_allclose(gaps, 0.4, atol=1e-6)
        # Each jump is one 0.4 s worth of turning.
        sim2 = self._turning_sim()
        _run(sim2, 2.0)
        a = sim2.data()["orientation"]
        _run(sim2, 0.4)
        self.assertAlmostEqual(_ang_diff(a, sim2.data()["orientation"]), 31.0 * 0.4, places=1)


class TestGyro(unittest.TestCase):
    def test_gyro_rows_shape_left_positive_deg_s(self):
        sim = RoverSim(SimConfig(gyro_gain=1.0), seed=1)
        sim.command(0.0, 0.4)
        _run(sim, 1.0)
        rows = sim.data()["gyros"]
        self.assertEqual(len(rows), sim.cfg.imu_rows)
        for r in rows:
            self.assertEqual(len(r), 4)
            self.assertAlmostEqual(r[2], 31.0, places=3)
        ts = [r[3] for r in rows]
        self.assertEqual(ts, sorted(ts))

    def test_gyro_integral_within_25_percent(self):
        for seed in range(20):
            sim = RoverSim(seed=seed)
            self.assertGreaterEqual(sim.gyro_gain, 0.75)
            self.assertLessEqual(sim.gyro_gain, 1.25)
            sim.command(0.0, 0.4)
            seen, integral = set(), 0.0
            for _ in range(60):
                sim.step(0.05)
                for r in sim.data()["gyros"]:
                    if r[3] not in seen:
                        seen.add(r[3])
                        integral += r[2] * sim.cfg.imu_period_s
            truth = 31.0 * sim.t
            self.assertLessEqual(abs(integral / truth - 1.0), 0.25, msg=f"seed {seed}")

    def test_gyro_gain_seeded(self):
        self.assertEqual(RoverSim(seed=7).gyro_gain, RoverSim(seed=7).gyro_gain)
        self.assertNotEqual(RoverSim(seed=7).gyro_gain, RoverSim(seed=8).gyro_gain)

    def test_default_gyro_gain_reaches_live_worst_case(self):
        gains = [RoverSim(seed=s).gyro_gain for s in range(200)]
        self.assertGreaterEqual(min(gains), 0.75)
        self.assertLessEqual(max(gains), 1.25)
        # The old [0.85, 1.15] draw never reaches a 25 percent scale error.
        self.assertLess(min(gains), 0.85)
        self.assertGreater(max(gains), 1.15)

    def test_explicit_gyro_gain_extremes_scale_yaw_rate(self):
        for gain in (0.75, 1.25):
            sim = RoverSim(SimConfig(gyro_gain=gain), seed=1)
            sim.command(0.0, 0.4)
            _run(sim, 1.0)
            for row in sim.data()["gyros"]:
                self.assertAlmostEqual(row[2], 31.0 * gain, places=3)


class TestGps(unittest.TestCase):
    def test_noise_statistics(self):
        sigma = 3.0
        sim = RoverSim(SimConfig(gps_sigma_m=sigma, gps_period_s=1.0), seed=3)
        ex, ey = [], []
        for _ in range(2000):
            sim.step(1.0)
            d = sim.data()
            ey.append(math.radians(d["latitude"] - sim.lat) * 6_371_000.0)
            ex.append(math.radians(d["longitude"] - sim.lon) * 6_371_000.0 * math.cos(math.radians(sim.lat)))
        for e in (np.array(ex), np.array(ey)):
            self.assertLess(abs(e.mean()), 0.3)
            self.assertAlmostEqual(e.std(), sigma, delta=0.15 * sigma)

    def test_noise_held_within_gps_period(self):
        sim = RoverSim(SimConfig(gps_period_s=1.0), seed=3)
        sim.step(0.1)
        a = sim.data()["latitude"]
        sim.step(0.1)
        self.assertEqual(sim.data()["latitude"], a)

    def test_no_fix_reads_1000(self):
        sim = RoverSim(SimConfig(gps_fix=False), seed=3)
        sim.step(1.0)
        d = sim.data()
        self.assertEqual(d["latitude"], 1000)
        self.assertEqual(d["longitude"], 1000)
        self.assertEqual(NO_FIX, 1000)

    def test_gps_dropout_window_mid_run(self):
        # Half-open [1.0, 2.5): fix before and after, 1000 inside, even while moving.
        sim = RoverSim(SimConfig(gps_sigma_m=0.0, gps_dropouts=((1.0, 2.5),)), seed=3)
        sim.step(0.9)
        before = sim.data()
        self.assertNotEqual(before["latitude"], 1000)
        self.assertGreater(before["gps_signal"], 0.0)
        sim.step(0.2)  # t = 1.1
        dropped = sim.data()
        self.assertEqual(dropped["latitude"], 1000)
        self.assertEqual(dropped["longitude"], 1000)
        self.assertEqual(dropped["gps_signal"], 0.0)
        sim.command(1.0, 0.0)
        sim.step(1.5)  # t = 2.6
        after = sim.data()
        self.assertNotEqual(after["latitude"], 1000)
        self.assertAlmostEqual(after["latitude"], sim.lat, places=6)
        self.assertGreater(after["gps_signal"], 0.0)


class TestDeterminism(unittest.TestCase):
    def _trace(self, seed):
        sim = RoverSim(SimConfig(gyro_noise_dps=0.5), seed=seed)
        out = []
        for i in range(40):
            sim.command(0.5, 0.3 if i < 20 else -0.3)
            sim.step(0.25)
            out.append(sim.data())
        return out

    def test_same_seed_same_payloads(self):
        self.assertEqual(self._trace(11), self._trace(11))
        self.assertNotEqual(self._trace(11), self._trace(12))


class TestFakeClient(unittest.TestCase):
    def test_has_real_client_public_methods(self):
        real = {n for n in dir(RoverClient) if not n.startswith("_") and callable(getattr(RoverClient, n))}
        fake = {n for n in dir(FakeRoverClient) if not n.startswith("_") and callable(getattr(FakeRoverClient, n))}
        self.assertLessEqual(real, fake)

    def test_payload_keys_match_sdk_data(self):
        sim = RoverSim(seed=1)
        sim.step(1.0)
        keys = set(sim.data())
        for k in ("battery", "signal_level", "orientation", "lamp", "speed", "gps_signal",
                  "latitude", "longitude", "vibration", "timestamp", "accels", "gyros", "mags",
                  "rpms", "rtm_state", "rtm_reason", "rtm_age_s"):
            self.assertIn(k, keys)

    @patch("erc_urban.sdk_client.requests.get")
    def test_fake_get_data_matches_real_parser(self, mock_get):
        sim = RoverSim(SimConfig(gyro_gain=1.0), seed=5)
        client = FakeRoverClient(sim)
        client.send_cmd(0.6, 0.4)
        _run(sim, 3.0)
        body = sim.data()
        fake_tel = client.get_data()
        self.assertIsInstance(fake_tel, Telemetry)

        mock_get.return_value = MagicMock(status_code=200, json=lambda: body)
        mock_get.return_value.raise_for_status = lambda: None
        real_tel = RoverClient("http://localhost:8000", timeout=1.0).get_data()

        self.assertEqual(fake_tel.raw, real_tel.raw)
        self.assertEqual(fake_tel.lat, real_tel.lat)
        self.assertEqual(fake_tel.lon, real_tel.lon)
        self.assertEqual(fake_tel.yaw, real_tel.yaw)
        self.assertEqual(fake_tel.hw_ts, real_tel.hw_ts)
        self.assertEqual(fake_tel.yaw_rate, real_tel.yaw_rate)
        self.assertEqual(fake_tel.wheels_turning, real_tel.wheels_turning)
        self.assertAlmostEqual(fake_tel.yaw_rate, 31.0, places=3)
        self.assertTrue(fake_tel.wheels_turning)
        self.assertAlmostEqual(fake_tel.yaw, math.radians(body["orientation"]))
        # Sim clock drives stamp and age, not the wall clock.
        self.assertEqual(fake_tel.stamp, sim.t)
        self.assertGreaterEqual(fake_tel.age_s, 0.0)
        self.assertLess(fake_tel.age_s, 1.0)

    def test_fake_client_restores_real_module(self):
        client = FakeRoverClient(RoverSim(seed=1))
        client.get_data()
        import requests
        import time
        self.assertIs(sdk_client.requests, requests)
        self.assertIs(sdk_client.time, time)

    def test_send_cmd_clamps_and_drives_sim(self):
        sim = RoverSim(seed=1)
        client = FakeRoverClient(sim, auto_step=0.1)
        client.send_cmd(2.0, -3.0)
        self.assertEqual((sim.linear, sim.angular), (1.0, -1.0))
        for _ in range(10):
            client.get_data()
        self.assertAlmostEqual(sim.t, 1.0)
        client.stop()
        self.assertEqual((sim.linear, sim.angular), (0.0, 0.0))

    def test_no_fix_through_real_parser(self):
        client = FakeRoverClient(RoverSim(SimConfig(gps_fix=False), seed=1))
        tel = client.get_data()
        self.assertEqual((tel.lat, tel.lon), (1000.0, 1000.0))

    def test_dropout_visible_through_auto_step(self):
        # auto_step advances inside get_data, so a caller cannot toggle gps_fix
        # between the step and the read. The window has to live on sim time.
        sim = RoverSim(SimConfig(gps_sigma_m=0.0, gps_dropouts=((0.5, 0.8),)), seed=1)
        client = FakeRoverClient(sim, auto_step=0.1)
        seen_drop = False
        seen_fix_after = False
        for _ in range(12):
            tel = client.get_data()
            if tel.lat == 1000.0:
                seen_drop = True
            elif seen_drop:
                seen_fix_after = True
        self.assertTrue(seen_drop)
        self.assertTrue(seen_fix_after)

    def test_checkpoints(self):
        sim = RoverSim(seed=1)
        far = (sim.cfg.origin_lat + 0.001, sim.cfg.origin_lon)
        near = (sim.cfg.origin_lat + 0.00005, sim.cfg.origin_lon)
        client = FakeRoverClient(sim, checkpoints=[near, far])
        cps, latest = client.get_checkpoints()
        self.assertEqual([c.sequence for c in cps], [1, 2])
        self.assertEqual(latest, 0)
        self.assertNotIn("_http_status", client.checkpoint_reached())
        body = client.checkpoint_reached()
        self.assertEqual(body["_http_status"], 400)
        self.assertGreater(body["detail"]["proximate_distance_to_checkpoint"], 15.0)
        self.assertEqual(client.get_checkpoints()[1], 1)


if __name__ == "__main__":
    unittest.main()
