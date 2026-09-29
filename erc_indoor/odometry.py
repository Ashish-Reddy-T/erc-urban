"""
Live wheel + gyro odometry from /data, on a background thread.

/data carries the last few rpm and gyro samples, each with its own hardware
timestamp, in 5-sample 50 Hz bursts every ~0.5 s. Samples are de-duplicated by
timestamp and integrated trapezoidally, so a missed burst is interpolated
across rather than held (holding inflated lap 2 from 125 m to 193 m).

Measured on this unit (indoor laps, 2026-09-22):
  * wheel: 0.388 m per revolution (lap 1 = 132 m, the published lap length)
  * gyro z is LEFT-positive, deg/s, bias ~ +0.27 deg/s while still — small, but
    integrated over a 10-minute lap it is ~160 deg, so it is re-measured at
    startup and yaw is only ever used over seconds (turn progress), never as
    absolute heading. The compass is useless indoors (swings +-200 deg).
  * the yaw axis is NOT the same on every unit: Wuhan inert-eel-code
    (2026-09-25) turns on gyro X (+34 deg/s on a left spin, z flat). yaw_axis /
    yaw_sign pick it; erc_urban detects it with a short wiggle at startup.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass

import numpy as np

from .lap import WHEEL_M_PER_REV

DEFAULT_GYRO_BIAS = 0.27
# Never integrate across a telemetry gap. Full-lap run 3: the SDK stalled for
# ~100 s; the next sample was joined to the last one trapezoidally at 0.51 m/s
# and 29 m of travel that never happened fired turn 3 in the wrong corridor.
MAX_INTEGRATION_DT = 0.6


@dataclass
class OdomState:
    s: float            # cumulative signed wheel distance, m
    yaw: float          # cumulative gyro yaw, deg, left-positive
    v: float            # latest wheel speed, m/s
    wz: float           # latest yaw rate, deg/s (bias-corrected)
    hw_ts: float        # hardware time of the newest sample
    age_s: float        # now - hw_ts at our receipt


class Odometer(threading.Thread):
    def __init__(self, client, poll_hz: float = 8.0,
                 m_per_rev: float = WHEEL_M_PER_REV, yaw_axis: int = 2,
                 yaw_sign: float = 1.0):
        super().__init__(daemon=True)
        self.client = client
        self.yaw_axis = yaw_axis
        self.yaw_sign = yaw_sign
        # (receipt time, hw time, gx, gy, gz): lets the caller find the yaw axis.
        self.raw_gyros: collections.deque = collections.deque(maxlen=600)
        self.poll_dt = 1.0 / poll_hz
        self.m_per_rev = m_per_rev
        self.bias = DEFAULT_GYRO_BIAS
        self._lock = threading.Lock()
        self._running = True
        self._s = 0.0
        self._yaw = 0.0
        self._last_rpm = None       # (t, v)
        self._last_gyro = None      # (t, wz_raw)
        self._v = 0.0
        self._wz = 0.0
        self._hw_ts = 0.0
        self._age = float("nan")
        self._raw_gyro_still: list[float] = []
        self.polls = 0
        # SDK login health (earth-rovers-sdk /data): REMOTE_LOGIN kick counter
        # and rovers other than ours that report to this login
        self.rtm_kicks = 0
        self.rtm_foreign: list[str] = []

    # -- thread ---------------------------------------------------------------
    def run(self) -> None:
        while self._running:
            t0 = time.time()
            try:
                tel = self.client.get_data()
                self._ingest(tel.raw or {})
                self.polls += 1
            except Exception:
                pass
            time.sleep(max(0.0, self.poll_dt - (time.time() - t0)))

    def stop(self) -> None:
        self._running = False

    def _ingest(self, d: dict) -> None:
        if isinstance(d.get("rtm_kicks"), int):
            self.rtm_kicks = d["rtm_kicks"]
            self.rtm_foreign = list(d.get("rtm_foreign_bots") or [])
        rpms = sorted((r for r in (d.get("rpms") or []) if isinstance(r, list) and len(r) >= 5),
                      key=lambda r: r[4])
        gyros = sorted((g for g in (d.get("gyros") or []) if isinstance(g, list) and len(g) >= 4),
                       key=lambda g: g[3])
        with self._lock:
            for r in rpms:
                t = float(r[4])
                if self._last_rpm is not None and t <= self._last_rpm[0]:
                    continue
                v = float(np.mean([float(x) for x in r[:4]])) / 60.0 * self.m_per_rev
                if self._last_rpm is not None:
                    t0, v0 = self._last_rpm
                    self._s += 0.5 * (v + v0) * min(t - t0, MAX_INTEGRATION_DT)
                self._last_rpm = (t, v)
                self._v = v
                self._hw_ts = max(self._hw_ts, t)
            still = self._last_rpm is not None and abs(self._last_rpm[1]) < 1e-6
            for g in gyros:
                t = float(g[3])
                if self._last_gyro is not None and t <= self._last_gyro[0]:
                    continue
                self.raw_gyros.append((time.time(), t, float(g[0]), float(g[1]), float(g[2])))
                wz = float(g[self.yaw_axis]) * self.yaw_sign
                if still:
                    self._raw_gyro_still.append(wz)
                    if len(self._raw_gyro_still) > 2000:
                        self._raw_gyro_still = self._raw_gyro_still[-1000:]
                if self._last_gyro is not None:
                    t0, w0 = self._last_gyro
                    self._yaw += 0.5 * ((wz - self.bias) + (w0 - self.bias)) * min(t - t0, MAX_INTEGRATION_DT)
                self._last_gyro = (t, wz)
                self._wz = wz - self.bias
                self._hw_ts = max(self._hw_ts, t)
            if self._hw_ts:
                self._age = time.time() - self._hw_ts

    # -- API ------------------------------------------------------------------
    def set_yaw_axis(self, axis: int, sign: float) -> None:
        """Integrate gyro[axis] * sign from now on (re-measure bias after this)."""
        with self._lock:
            self.yaw_axis, self.yaw_sign = int(axis), float(sign)
            self._last_gyro = None
            self._raw_gyro_still = []

    def calibrate_bias(self, seconds: float = 3.0) -> float:
        """Rover must be still. Median raw gyro z over the window."""
        with self._lock:
            self._raw_gyro_still = []
        time.sleep(seconds)
        with self._lock:
            vals = list(self._raw_gyro_still)
        if len(vals) >= 10:
            b = float(np.median(vals))
            if abs(b) < 3.0:
                self.bias = b
        return self.bias

    def state(self) -> OdomState:
        with self._lock:
            return OdomState(self._s, self._yaw, self._v, self._wz, self._hw_ts, self._age)

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._last_rpm is not None and self._last_gyro is not None
