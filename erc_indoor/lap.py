"""
Load a recorded lap — either format — into one common shape.

    lap = load_lap("erc_indoor/data/indoor_track_recording.h5")   # lap 1 (h5)
    lap = load_lap("runs/indoor/lap_20260922_204027")              # record.py dir

Everything is on ONE clock (time.time() seconds):
  * the h5 frame clock runs +4 h ahead of its IMU clock (the SDK stamps frames
    with datetime.utcnow().timestamp(), i.e. local-as-UTC) and is corrected here;
  * record.py fetched /v2/front faster than the camera updates, so byte-identical
    consecutive frames are dropped (14927 fetched -> 3430 unique on lap 2).

Wheel distance: the SDK delivers rpms in 5-sample, 50 Hz bursts every ~0.5 s.
Burst means are interpolated linearly between bursts. At WHEEL_M_PER_REV this
gives ~130 m for lap 1 and 125 m cone-to-cone on lap 2, against a published
132 m lap. Gyro yaw rate is handled the same
way; it is only trusted for short-horizon turn rate, never as absolute heading.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image

# Distance per wheel revolution. Fitted so lap 1 = 132 m (rules PDF); physically
# a ~0.12 m wheel. Re-check against lap 2 cone-to-cone distance.
WHEEL_M_PER_REV = 0.388
H5_FRAME_CLOCK_OFFSET_S = 14400.0
BURST_GAP_S = 0.1


@dataclass
class Lap:
    name: str
    frame_t: np.ndarray                     # (N,) seconds
    _load: Callable[[int], np.ndarray]      # i -> HxWx3 uint8 RGB
    wheel_t: np.ndarray                     # burst times
    wheel_v: np.ndarray                     # burst mean speed, m/s (signed)
    gyro_t: np.ndarray
    gyro_z: np.ndarray                      # burst mean yaw rate, deg/s (raw sign)
    ctrl_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ctrl_lin: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ctrl_ang: np.ndarray = field(default_factory=lambda: np.zeros(0))
    events: list = field(default_factory=list)   # [(t, name)]

    def __len__(self) -> int:
        return len(self.frame_t)

    def image(self, i: int) -> np.ndarray:
        return self._load(int(i))

    # -- odometry -------------------------------------------------------------
    def dist_at(self, t) -> np.ndarray:
        """Cumulative wheel distance (m) at time(s) t.

        Trapezoidal between bursts, NOT sample-and-hold: on lap 2 /data polling
        stalled for 3.6 s at a time and holding a burst taken while
        accelerating across that gap inflated the lap from 125 m to 193 m.
        """
        return np.interp(t, self.wheel_t, _cumtrapz(self.wheel_t, self.wheel_v))

    def yaw_at(self, t, bias: Optional[float] = None) -> np.ndarray:
        """Integrated gyro yaw (deg, left-positive). Short-horizon use only."""
        b = self.gyro_bias() if bias is None else bias
        return np.interp(t, self.gyro_t, _cumtrapz(self.gyro_t, self.gyro_z - b))

    def gyro_bias(self) -> float:
        v = np.interp(self.gyro_t, self.wheel_t, self.wheel_v)
        still = np.abs(v) < 1e-3
        return float(np.median(self.gyro_z[still])) if still.sum() > 10 else 0.0

    @property
    def frame_s(self) -> np.ndarray:
        return self.dist_at(self.frame_t)

    def event_times(self, name: str) -> list[float]:
        return [t for t, n in self.events if n == name]


def _cumtrapz(t: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))])


def _bursts(t: np.ndarray, vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(t)
    t, vals = t[order], vals[order]
    gid = np.concatenate([[0], np.cumsum(np.diff(t) > BURST_GAP_S)])
    n = gid.max() + 1
    bt = np.bincount(gid, weights=t, minlength=n) / np.bincount(gid, minlength=n)
    bv = np.bincount(gid, weights=vals, minlength=n) / np.bincount(gid, minlength=n)
    return bt, bv


def _rpm_to_mps(rpm4: np.ndarray) -> np.ndarray:
    return rpm4.mean(axis=1) / 60.0 * WHEEL_M_PER_REV


def _decode(raw: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def _load_h5(path: str) -> Lap:
    import h5py
    f = h5py.File(path, "r")
    ft = f["front_frames/timestamps"][:] - H5_FRAME_CLOCK_OFFSET_S
    data = f["front_frames/data"]
    r = f["rpms"][:]
    rpm4 = np.stack([r[k] for k in ("front_left", "front_right", "rear_left", "rear_right")], 1)
    wt, wv = _bursts(r["t"], _rpm_to_mps(rpm4))
    g = f["gyros"][:]
    gt, gz = _bursts(g["t"], g["z"].astype(float))
    c = f["controls"][:]
    return Lap(
        name=os.path.basename(path), frame_t=ft,
        _load=lambda i: _decode(bytes(data[i])),
        wheel_t=wt, wheel_v=wv, gyro_t=gt, gyro_z=gz,
        ctrl_t=c["timestamp"], ctrl_lin=c["linear"], ctrl_ang=c["angular"],
    )


def _jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def _load_dir(path: str) -> Lap:
    frames = _jsonl(os.path.join(path, "frames.jsonl"))
    keep, last = [], None
    for fr in frames:
        with open(os.path.join(path, "frames", fr["file"]), "rb") as fh:
            h = hashlib.md5(fh.read()).digest()
        if h != last:
            keep.append(fr)
        last = h
    files = [os.path.join(path, "frames", fr["file"]) for fr in keep]
    ft = np.array([fr["t"] for fr in keep])
    raw_to_unique = {fr["i"]: k for k, fr in enumerate(keep)}

    rpm_rows, gyro_rows = {}, {}
    for rec in _jsonl(os.path.join(path, "telemetry.jsonl")):
        d = rec.get("data") or {}
        for row in d.get("rpms") or []:
            if len(row) >= 5:
                rpm_rows[float(row[4])] = row[:4]
        for row in d.get("gyros") or []:
            if len(row) >= 4:
                gyro_rows[float(row[3])] = row[2]
    rt = np.array(sorted(rpm_rows))
    wt, wv = _bursts(rt, _rpm_to_mps(np.array([rpm_rows[t] for t in rt], float)))
    gtt = np.array(sorted(gyro_rows))
    gt, gz = _bursts(gtt, np.array([gyro_rows[t] for t in gtt], float))

    ctrl = _jsonl(os.path.join(path, "controls.jsonl"))
    events = []
    for e in _jsonl(os.path.join(path, "events.jsonl")):
        events.append((e["t"], e["event"]))
    return Lap(
        name=os.path.basename(path.rstrip("/")), frame_t=ft,
        _load=lambda i: _decode(open(files[i], "rb").read()),
        wheel_t=wt, wheel_v=wv, gyro_t=gt, gyro_z=gz,
        ctrl_t=np.array([c["t"] for c in ctrl]),
        ctrl_lin=np.array([c["linear"] for c in ctrl]),
        ctrl_ang=np.array([c["angular"] for c in ctrl]),
        events=events,
    )


def load_lap(path: str) -> Lap:
    return _load_h5(path) if path.endswith(".h5") else _load_dir(path)
