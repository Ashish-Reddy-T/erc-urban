"""Offline kinematic rover sim and a fake RoverClient backed by it.

A unicycle model of the EarthRover that reproduces the live telemetry quirks
the navigation code has to cope with, deterministic under a seed and advanced
only by an explicit ``step(dt)`` (no wall clock, no sleeps).

Modelled live facts (all configurable through ``SimConfig``):
  * positive angular command = LEFT turn: heading goes CCW, so the compass
    value (``orientation``) decreases.
  * turn rate scales linearly with the command: 31 deg/s at angular 0.4.
  * ``/data`` orientation is degrees clockwise from north in [0, 360). It is
    the true heading as of ``orientation_delay_s`` (1.4 s) ago, sampled and
    held on an ``orientation_period_s`` (0.4 s) grid.
  * gyros rows are ``[x, y, z, t]``, z = yaw rate in deg/s, LEFT-positive,
    scaled by a gain error (seeded uniform in [0.75, 1.25] unless given).
  * GPS carries Gaussian noise in metres, redrawn every ``gps_period_s``;
    with no fix latitude and longitude read 1000.

Public API:
    SimConfig(...)                       dataclass of the knobs above
    RoverSim(config=None, seed=0)
        .command(linear, angular)        set the held command, clamped to [-1, 1]
        .step(dt)                        advance sim time by dt seconds
        .t, .x, .y, .heading_deg         truth: seconds, metres east/north, compass deg
        .lat, .lon                       truth position in degrees
        .gyro_gain                       the gain error in use
        .data() -> dict                  SDK ``/data`` payload for the current time
    FakeRoverClient(sim, auto_step=None, checkpoints=(), client_module=None)
        Same methods as ``sdk_client.RoverClient`` (send_cmd, stop, get_data,
        get_frames, get_map_frame, get_status, ping, list_missions,
        start_mission, end_mission, get_checkpoints, checkpoint_reached).
        ``get_data`` runs the real ``RoverClient.get_data`` parser over
        ``sim.data()``, so it returns exactly the ``Telemetry`` the real client
        would. ``auto_step`` advances the sim by that many seconds per call.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional, Sequence
from unittest.mock import patch

import numpy as np

from erc_urban import sdk_client as _default_client_module

R_EARTH_M = 6_371_000.0
NO_FIX = 1000.0


@dataclass
class SimConfig:
    origin_lat: float = 40.6943
    origin_lon: float = -73.9865
    start_heading_deg: float = 0.0
    # 31 deg/s at angular 0.4, linear in the command.
    turn_rate_dps_per_unit: float = 31.0 / 0.4
    max_speed_mps: float = 1.0
    track_width_m: float = 0.3
    wheel_circumference_m: float = 0.388
    orientation_delay_s: float = 1.4
    orientation_period_s: float = 0.4
    # None = seeded uniform draw in [0.75, 1.25].
    gyro_gain: Optional[float] = None
    gyro_bias_dps: float = 0.0
    gyro_noise_dps: float = 0.0
    imu_period_s: float = 0.1
    imu_rows: int = 5
    gps_sigma_m: float = 2.0
    gps_period_s: float = 1.0
    gps_fix: bool = True
    battery: int = 100
    # Unix time that sim t = 0 maps to in payload timestamps.
    epoch: float = 1_760_000_000.0


class RoverSim:
    def __init__(self, config: Optional[SimConfig] = None, seed: int = 0):
        self.cfg = config or SimConfig()
        root = np.random.default_rng(seed)
        self._rng_gyro, self._rng_gps = (np.random.default_rng(s) for s in root.integers(0, 2**63, size=2))
        self.gyro_gain = (float(self.cfg.gyro_gain) if self.cfg.gyro_gain is not None
                          else float(root.uniform(0.75, 1.25)))
        self.t = 0.0
        self.x = 0.0
        self.y = 0.0
        self.heading_deg = self.cfg.start_heading_deg % 360.0
        self.linear = 0.0
        self.angular = 0.0
        # Unwrapped heading at every step boundary, for the delayed compass.
        self._hist_t = [0.0]
        self._hist_h = [self.heading_deg]
        self._next_imu = self.cfg.imu_period_s
        self._gyros: deque = deque(maxlen=self.cfg.imu_rows)
        self._accels: deque = deque(maxlen=self.cfg.imu_rows)
        self._rpms: deque = deque(maxlen=self.cfg.imu_rows)
        self._gps_slot: Optional[int] = None
        self._gps_noise = (0.0, 0.0)

    # ---- control / time ----------------------------------------------------
    def command(self, linear: float, angular: float) -> None:
        self.linear = float(np.clip(linear, -1.0, 1.0))
        self.angular = float(np.clip(angular, -1.0, 1.0))

    @property
    def speed_mps(self) -> float:
        return self.linear * self.cfg.max_speed_mps

    @property
    def yaw_rate_dps(self) -> float:
        """True body yaw rate, deg/s, LEFT-positive (CCW)."""
        return self.angular * self.cfg.turn_rate_dps_per_unit

    def step(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("dt must be >= 0")
        if dt == 0:
            return
        v, w = self.speed_mps, self.yaw_rate_dps
        t0 = self.t
        self._emit_imu(t0, t0 + dt, v, w)
        # Compass heading h (CW from north); CCW rate w lowers it.
        h0 = math.radians(self._hist_h[-1])
        h1 = h0 - math.radians(w) * dt
        if abs(w) < 1e-9:
            self.x += v * dt * math.sin(h0)
            self.y += v * dt * math.cos(h0)
        else:
            dh = h1 - h0
            self.x += v * dt * (math.cos(h0) - math.cos(h1)) / dh
            self.y += v * dt * (math.sin(h1) - math.sin(h0)) / dh
        self.t = t0 + dt
        self._hist_t.append(self.t)
        self._hist_h.append(math.degrees(h1))
        self.heading_deg = math.degrees(h1) % 360.0

    def _emit_imu(self, t0: float, t1: float, v: float, w: float) -> None:
        half = math.radians(w) * self.cfg.track_width_m / 2.0
        rpm_l = (v - half) / self.cfg.wheel_circumference_m * 60.0
        rpm_r = (v + half) / self.cfg.wheel_circumference_m * 60.0
        while self._next_imu <= t1 + 1e-9:
            ts = round(self.cfg.epoch + self._next_imu, 3)
            gz = (w * self.gyro_gain + self.cfg.gyro_bias_dps
                  + (self._rng_gyro.normal(0.0, self.cfg.gyro_noise_dps) if self.cfg.gyro_noise_dps else 0.0))
            self._gyros.append([0.0, 0.0, round(gz, 4), ts])
            self._accels.append([0.0, 0.0, 1.0, ts])
            self._rpms.append([round(rpm_l, 3), round(rpm_r, 3), round(rpm_l, 3), round(rpm_r, 3), ts])
            self._next_imu += self.cfg.imu_period_s

    # ---- sensors -----------------------------------------------------------
    def _heading_at(self, t: float) -> float:
        return float(np.interp(t, self._hist_t, self._hist_h))

    def reported_orientation(self) -> float:
        """Compass degrees as /data reports it: delayed, then held on a grid."""
        p = self.cfg.orientation_period_s
        sample_t = math.floor(self.t / p + 1e-9) * p if p > 0 else self.t
        return self._heading_at(sample_t - self.cfg.orientation_delay_s) % 360.0

    def _latlon(self, x: float, y: float) -> tuple[float, float]:
        lat = self.cfg.origin_lat + math.degrees(y / R_EARTH_M)
        lon = self.cfg.origin_lon + math.degrees(x / (R_EARTH_M * math.cos(math.radians(self.cfg.origin_lat))))
        return lat, lon

    @property
    def lat(self) -> float:
        return self._latlon(self.x, self.y)[0]

    @property
    def lon(self) -> float:
        return self._latlon(self.x, self.y)[1]

    def reported_latlon(self) -> tuple[float, float]:
        if not self.cfg.gps_fix:
            return NO_FIX, NO_FIX
        slot = math.floor(self.t / self.cfg.gps_period_s + 1e-9) if self.cfg.gps_period_s > 0 else None
        if slot is None or slot != self._gps_slot:
            self._gps_slot = slot
            s = self.cfg.gps_sigma_m
            self._gps_noise = (float(self._rng_gps.normal(0.0, s)), float(self._rng_gps.normal(0.0, s))) if s > 0 else (0.0, 0.0)
        return self._latlon(self.x + self._gps_noise[0], self.y + self._gps_noise[1])

    def data(self) -> dict:
        lat, lon = self.reported_latlon()
        now = self.cfg.epoch + self.t
        last_ts = self._gyros[-1][3] if self._gyros else None
        return {
            "battery": self.cfg.battery,
            "signal_level": 5,
            "orientation": round(self.reported_orientation(), 2) % 360.0,
            "lamp": 0,
            "speed": round(abs(self.speed_mps), 3),
            "gps_signal": 31.25 if self.cfg.gps_fix else 0.0,
            "latitude": lat,
            "longitude": lon,
            "vibration": 0.0,
            "timestamp": now,
            "accels": [list(r) for r in self._accels],
            "gyros": [list(r) for r in self._gyros],
            "mags": [],
            "rpms": [list(r) for r in self._rpms],
            "rtm_state": "CONNECTED",
            "rtm_reason": None,
            "rtm_age_s": round(now - last_ts, 2) if last_ts is not None else None,
        }


class _Resp:
    def __init__(self, body: Any, status_code: int = 200):
        self._body = body
        self.status_code = status_code
        self.text = ""

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


@dataclass
class FakeRoverClient:
    sim: RoverSim
    auto_step: Optional[float] = None
    checkpoints: Sequence[tuple[float, float]] = ()
    client_module: Any = None
    checkpoint_radius_m: float = 15.0
    latest_scanned: int = field(default=0, init=False)
    sent: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self._mod = self.client_module or _default_client_module
        self._real = self._mod.RoverClient("http://sim", timeout=1.0)

    # ---- control -----------------------------------------------------------
    def send_cmd(self, linear: float, angular: float, lamp: Optional[int] = None) -> None:
        linear = float(np.clip(linear, -1.0, 1.0))
        angular = float(np.clip(angular, -1.0, 1.0))
        self.sent.append((self.sim.t, linear, angular))
        self.sim.command(linear, angular)

    def stop(self) -> None:
        self.send_cmd(0.0, 0.0)

    # ---- telemetry ---------------------------------------------------------
    def get_data(self):
        if self.auto_step:
            self.sim.step(self.auto_step)
        body = self.sim.data()
        sim = self.sim
        fake_requests = SimpleNamespace(get=lambda *a, **k: _Resp(body))
        fake_time = SimpleNamespace(monotonic=lambda: sim.t, time=lambda: sim.cfg.epoch + sim.t)
        with patch.object(self._mod, "requests", fake_requests), patch.object(self._mod, "time", fake_time):
            return self._real.get_data()

    def get_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return None, None

    def get_map_frame(self) -> Optional[np.ndarray]:
        return None

    def get_status(self) -> Optional[dict]:
        return None

    def ping(self) -> bool:
        return True

    # ---- missions ----------------------------------------------------------
    def list_missions(self) -> list[dict]:
        return []

    def start_mission(self) -> dict:
        return {"message": "sim mission started"}

    def end_mission(self) -> dict:
        return {"message": "sim mission ended"}

    def get_checkpoints(self):
        cps = [self._mod.Checkpoint(id=i + 1, sequence=i + 1, lat=float(la), lon=float(lo))
               for i, (la, lo) in enumerate(self.checkpoints)]
        return cps, self.latest_scanned

    def checkpoint_reached(self) -> dict:
        if self.latest_scanned >= len(self.checkpoints):
            return {"detail": "no checkpoint left", "_http_status": 400}
        la, lo = self.checkpoints[self.latest_scanned]
        dy = math.radians(la - self.sim.lat) * R_EARTH_M
        dx = math.radians(lo - self.sim.lon) * R_EARTH_M * math.cos(math.radians(self.sim.lat))
        dist = math.hypot(dx, dy)
        if dist > self.checkpoint_radius_m:
            return {"detail": {"error": f"Bot is not within {self.checkpoint_radius_m:g} meters from the checkpoint",
                               "proximate_distance_to_checkpoint": round(dist, 2)},
                    "_http_status": 400}
        self.latest_scanned += 1
        return {"message": "Checkpoint reached", "next_checkpoint_sequence": self.latest_scanned + 1}
