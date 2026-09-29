"""
Wheel + gyro odometry as a drop-in for VisualOdometry.

VisualOdometry tracks ground texture under a 148 mm camera and, by its own
measurements, loses tracking above ~0.1 m per frame and under-reads spins ~3x.
erc_indoor's Odometer integrates the rpm and gyro bursts in /data instead, and
drove full indoor laps on a Mini+ with it (0.388 m/rev, lap 1 = 132 m). This
wraps that thread so every consumer here (GpsHealth, Breadcrumbs,
LocalMap.update_pose, costmap memory, CommitWindow) keeps receiving the same
per-tick OdomDelta.

rpm liveness is per unit: one Mini+ read [0,0,0,0] throughout 30 s of driving.
So `rpm_dead` is decided live: if forward motion has been commanded for a while
and the wheels have never reported any travel, the caller switches back to VO.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .odometry import OdomDelta


class WheelOdometry:
    def __init__(
        self,
        odometer,
        stale_s: float = 2.5,
        max_speed_mps: float = 0.9,
        dead_after_s: float = 4.0,
    ):
        """
        odometer: an erc_indoor.odometry.Odometer (already started).
        stale_s: newest wheel/gyro sample older than this -> confidence 0.
        max_speed_mps: clip on distance per tick; a telemetry hiccup must not
            teleport the map pose.
        dead_after_s: commanded forward this long with zero wheel travel means
            this unit's rpms are not live.
        """
        self.odo = odometer
        self.stale_s = stale_s
        self.max_speed_mps = max_speed_mps
        self.dead_after_s = dead_after_s
        self._last: Optional[tuple[float, float, float]] = None   # (t, s, yaw_deg)
        self._fwd_since: Optional[float] = None
        self._ever_moved = False
        self.rpm_dead = False
        self.last: Optional[OdomDelta] = None

    def update(self, commanding_forward: bool = False,
               now: Optional[float] = None) -> OdomDelta:
        """Motion since the previous call, in the previous robot frame."""
        now = time.time() if now is None else now
        if not getattr(self.odo, "ready", False):
            self.last = OdomDelta(0.0, 0.0, 0.0, 0.0, reason="wheel_not_ready")
            return self.last
        st = self.odo.state()
        self._check_alive(st.s, commanding_forward, now)
        prev, self._last = self._last, (now, st.s, st.yaw)
        if prev is None:
            self.last = OdomDelta(0.0, 0.0, 0.0, 0.0, reason="wheel_first")
            return self.last
        stale = not (st.hw_ts > 0 and now - st.hw_ts <= self.stale_s)
        if stale or self.rpm_dead:
            self.last = OdomDelta(0.0, 0.0, 0.0, 0.0,
                                  reason="wheel_dead" if self.rpm_dead else "wheel_stale")
            return self.last
        dt = max(now - prev[0], 1e-3)
        ds = st.s - prev[1]
        cap = self.max_speed_mps * dt
        ds = max(-cap, min(cap, ds))
        dyaw = math.radians(st.yaw - prev[2])          # gyro is left-positive
        # Midpoint heading: over one tick the rover arcs, it does not turn
        # then translate.
        self.last = OdomDelta(
            dx=ds * math.cos(dyaw / 2.0),
            dy=ds * math.sin(dyaw / 2.0),
            dyaw=dyaw,
            confidence=1.0,
            reason="wheel",
        )
        return self.last

    def _check_alive(self, s: float, commanding_forward: bool, now: float) -> None:
        # Only a unit whose wheels have NEVER reported travel is dead. Once they
        # have, zero travel under a forward command means stalled against
        # something, which is true and exactly what the stuck logic needs.
        if self.rpm_dead or self._ever_moved:
            return
        if abs(s) > 0.05:
            self._ever_moved = True
            return
        if not commanding_forward:
            self._fwd_since = None
            return
        if self._fwd_since is None:
            self._fwd_since = now
        elif now - self._fwd_since >= self.dead_after_s:
            self.rpm_dead = True


_AXES = "xyz"


def parse_gyro_axis(spec: str) -> Optional[tuple[int, float]]:
    """'z' -> (2, +1), '-x' -> (0, -1); 'auto' (or junk) -> None."""
    spec = (spec or "").strip().lower()
    sign = -1.0 if spec.startswith("-") else 1.0
    spec = spec.lstrip("+-")
    return (_AXES.index(spec), sign) if spec in ("x", "y", "z") else None


def _integrate(samples) -> list:
    """Per-axis integral (deg) of raw samples [(hw_t, gx, gy, gz), ...]."""
    tot = [0.0, 0.0, 0.0]
    samples = sorted(samples)
    for (t0, *g0), (t1, *g1) in zip(samples, samples[1:]):
        dt = min(max(t1 - t0, 0.0), 0.6)
        for a in range(3):
            tot[a] += 0.5 * (g0[a] + g1[a]) * dt
    return tot


def yaw_axis_from_wiggle(left: list, right: list, min_deg: float = 15.0,
                         min_ratio: float = 2.0) -> Optional[tuple[int, float, float]]:
    """
    Which gyro axis is yaw, from raw samples [(hw_t, gx, gy, gz), ...] recorded
    over a LEFT spin and then a RIGHT spin. The yaw axis integrates to + on one
    and - on the other; the others only see bias and bumps. Returns
    (axis, sign so that left is positive, left-minus-right degrees) or None
    if no axis stands out (spin did not happen, or telemetry was missing).
    """
    if len(left) < 3 or len(right) < 3:
        return None
    L, Rt = _integrate(left), _integrate(right)
    diff = [L[a] - Rt[a] for a in range(3)]
    order = sorted(range(3), key=lambda a: -abs(diff[a]))
    best, second = order[0], order[1]
    if abs(diff[best]) < min_deg or abs(diff[best]) < min_ratio * abs(diff[second]):
        return None
    return best, (1.0 if diff[best] > 0 else -1.0), diff[best]


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def compass_turn_error(c0, c1, c2, turn_left: float, turn_right: float) -> Optional[float]:
    """
    Worst disagreement (deg) between the compass and the gyro over the wiggle.
    c0/c1/c2: compass (deg CW from north) before, after the left spin, after
    the right spin. turn_*: gyro yaw change of each spin, left-positive, so a
    good compass moves by -turn. Panama hinge-moto-nova: a 38 deg left turn
    moved the compass 265 -> 73, i.e. an error of ~150 deg. None if any
    reading is missing.
    """
    if c0 is None or c1 is None or c2 is None:
        return None
    e1 = abs(_wrap180((c1 - c0) + turn_left))
    e2 = abs(_wrap180((c2 - c1) + turn_right))
    return max(e1, e2)


@dataclass
class Wiggle:
    axis: Optional[tuple[int, float, float]]   # yaw_axis_from_wiggle(), or None
    note: str
    compass_err: Optional[float] = None        # compass_turn_error(), or None


def _circ_median(vals: list) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    a = np.radians(vals)
    return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0


def detect_yaw_axis(odometer, send, ang: float = 0.5, spin_s: float = 0.8,
                    settle_s: float = 1.6, read_compass=None) -> Wiggle:
    """
    Spin left then right in place (net ~0) and find the yaw axis from the
    Odometer's raw gyro buffer. send(lin, ang, hold_s) commands the rover.
    settle_s covers the ~1 s command-to-telemetry lag seen on the link.
    read_compass() -> deg or None: if given, also measure whether the compass
    follows the two spins.
    """
    def compass():
        return _circ_median([read_compass() for _ in range(3)]) if read_compass else None

    c0 = compass()
    t0 = time.time()
    send(0.0, ang, spin_s)
    time.sleep(spin_s)
    send(0.0, 0.0, None)
    time.sleep(settle_s)
    c1 = compass()
    t1 = time.time()
    send(0.0, -ang, spin_s)
    time.sleep(spin_s)
    send(0.0, 0.0, None)
    time.sleep(settle_s)
    c2 = compass()
    t2 = time.time()
    raw = list(odometer.raw_gyros)
    left = [(r[1], r[2], r[3], r[4]) for r in raw if t0 <= r[0] < t1]
    right = [(r[1], r[2], r[3], r[4]) for r in raw if t1 <= r[0] < t2]
    res = yaw_axis_from_wiggle(left, right)
    note = f"{len(left)}+{len(right)} gyro samples"
    if left and right:
        L, Rt = _integrate(left), _integrate(right)
        note += " (L/R deg x %+.0f/%+.0f y %+.0f/%+.0f z %+.0f/%+.0f)" % (
            L[0], Rt[0], L[1], Rt[1], L[2], Rt[2])
    err = None
    if res is not None:
        a, sign = res[0], res[1]
        err = compass_turn_error(c0, c1, c2, sign * _integrate(left)[a],
                                 sign * _integrate(right)[a])
        if err is not None:
            note += f", compass {c0:.0f}->{c1:.0f}->{c2:.0f} (off {err:.0f} deg)"
    return Wiggle(res, note, err)
