"""
Heading = gyro for turns, compass (and GPS course) for the absolute reference.

Panama (hinge-moto-nova): at rest on level ground the compass read 265, 305,
262, 301, 263, 321 ... (std 19 deg, ~40 deg tick to tick). Everything that
turned through it went wrong: the goal-behind pivot reversed direction every
burst, and a checkpoint 4 m away was orbited. The gyro is steady (bias
calibrated at start), so it carries every turn; the compass is only folded in
slowly (k per tick, gated), which averages its noise down to a few degrees,
and a straight forward stretch of GPS track corrects the absolute heading too.
On a good compass (Wuhan) the result is the compass, lightly smoothed.

Worse, on that unit the compass is miscalibrated, not just noisy: a 38 deg
left turn (gyro, agreeing with the camera) moved it 265 -> 73 and it then sat
steady there. So k_compass=0 is allowed: the compass only seeds the start,
and a GPS course that disagrees by more than snap_deg REPLACES the heading
(the seed may be off by anything) instead of nudging it.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def _gps_bearing(lat0, lon0, lat1, lon1) -> float:
    """Degrees clockwise from north, from (lat0, lon0) to (lat1, lon1)."""
    dn = (lat1 - lat0) * 110540.0
    de = (lon1 - lon0) * 111320.0 * math.cos(math.radians(lat0))
    return math.degrees(math.atan2(de, dn)) % 360.0


class HeadingFilter:
    def __init__(self, k_compass: float = 0.04, gate_deg: float = 60.0, init_n: int = 8,
                 reinit_after: int = 40, k_gps: float = 0.3, gps_seg_m: float = 4.0,
                 gps_straight_deg: float = 15.0, snap_deg: float = 60.0,
                 track_min_m: float = 4.0, track_win_m: float = 10.0):
        self.track_min_m = track_min_m
        self.track_win_m = track_win_m
        self._trk: list = []
        self.snap_deg = snap_deg
        self._pending = None
        self.k_compass = k_compass
        self.gate_deg = gate_deg
        self.init_n = init_n
        self.reinit_after = reinit_after
        self.k_gps = k_gps
        self.gps_seg_m = gps_seg_m
        self.gps_straight_deg = gps_straight_deg
        self.est: Optional[float] = None       # deg, clockwise from north
        self._init: list[float] = []
        self._outliers = 0
        self._seg = None                       # [lat, lon, fwd_m, turn_deg, sin_sum, cos_sum]
        self.fixes = 0
        self.last_gps_fix: Optional[float] = None

    def update(self, compass_deg: Optional[float], dyaw_left_deg: float = 0.0,
               dx_fwd_m: float = 0.0, lat: float = float("nan"),
               lon: float = float("nan")) -> Optional[float]:
        """One tick. Returns the fused heading (deg CW from north) or None."""
        if self.est is None:
            if compass_deg is not None:
                self._init.append(float(compass_deg))
            if len(self._init) < self.init_n:
                return compass_deg
            a = np.radians(self._init)
            self.est = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0
            self._init = []
            return self.est
        # Predict with the gyro (left-positive -> clockwise heading decreases).
        self.est = (self.est - dyaw_left_deg) % 360.0
        # Compass: small, gated nudges.
        if compass_deg is not None:
            innov = wrap180(float(compass_deg) - self.est)
            if self.k_compass <= 0.0:
                pass                                   # compass seeds the start only
            elif abs(innov) <= self.gate_deg:
                self.est = (self.est + self.k_compass * innov) % 360.0
                self._outliers = 0
            else:
                self._outliers += 1
                if self._outliers >= self.reinit_after:
                    # Persistently far off: the estimate is what is wrong.
                    self.est, self._outliers = None, 0
                    self._init = [float(compass_deg)]
                    self._trk = []
                    self._pending = None
                    return compass_deg
        # GPS vs odometry TRACK: dead-reckon the path with the current heading
        # (wheel distance along it) and compare it with the path GPS saw. The
        # rotation between the two displacements is the heading error, and it
        # does not need a straight stretch: requiring one (Panama) never
        # happened, the fan steers a little every tick, and the rover drove
        # 7 m the wrong way on a compass-seeded heading.
        if not (math.isnan(lat) or math.isnan(lon)):
            r = math.radians(self.est)
            if self._trk:
                e0, n0, ol = self._trk[-1][2], self._trk[-1][3], self._trk[-1][4]
            else:
                e0 = n0 = ol = 0.0
            self._trk.append((lat, lon, e0 + dx_fwd_m * math.sin(r),
                              n0 + dx_fwd_m * math.cos(r), ol + abs(dx_fwd_m)))
            while len(self._trk) > 2 and self._trk[-1][4] - self._trk[1][4] > self.track_win_m:
                self._trk.pop(0)
            a, b = self._trk[0], self._trk[-1]
            cl = math.cos(math.radians(lat))
            dge, dgn = (b[1] - a[1]) * 111320.0 * cl, (b[0] - a[0]) * 110540.0
            doe, don = b[2] - a[2], b[3] - a[3]
            g, o = math.hypot(dge, dgn), math.hypot(doe, don)
            if g >= self.track_min_m and o >= self.track_min_m and 0.5 <= g / o <= 1.8:
                innov = wrap180(math.degrees(math.atan2(dge, dgn)) - math.degrees(math.atan2(doe, don)))
                applied = False
                if abs(innov) > self.snap_deg:
                    # The first fix may snap at once (the seed can be anything);
                    # after that a big jump needs a second, agreeing track: GPS
                    # under trees gave a +66 deg "fix" one check after -8.
                    first = self.k_compass <= 0.0 and self.fixes == 0
                    confirmed = first or (
                        self._pending is not None and abs(wrap180(innov - self._pending)) < 30.0)
                    if confirmed:
                        self.est, self._pending = (self.est + innov) % 360.0, None
                        applied = True
                    else:
                        self._pending = innov
                else:
                    self.est = (self.est + self.k_gps * innov) % 360.0
                    self._pending = None
                    applied = True
                if applied:
                    self.fixes += 1
                    self.last_gps_fix = innov
                print(f"[heading] GPS track {g:.1f} m vs odometry {o:.1f} m: off {innov:+.0f} deg "
                      f"{'applied' if applied else 'pending confirmation'} -> heading {self.est:.0f}", flush=True)
                self._trk = []
        return self.est


def save_state(path: str, est: Optional[float], lat: float, lon: float, fixes: int) -> None:
    """Remember the heading across a restart (the rover does not move meanwhile)."""
    import json
    import time
    if est is None or math.isnan(lat):
        return
    try:
        with open(path, "w") as fh:
            json.dump({"est": est, "lat": lat, "lon": lon, "fixes": fixes, "t": time.time()}, fh)
    except OSError:
        pass


def load_state(path: str, lat: float, lon: float, max_age_s: float = 900.0,
               max_move_m: float = 2.0) -> Optional[float]:
    """Saved heading if fresh, GPS-corrected at least once, and the rover has not moved."""
    import json
    import time
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None
    if time.time() - d.get("t", 0) > max_age_s or d.get("fixes", 0) < 1 or math.isnan(lat):
        return None
    moved = math.hypot((lat - d["lat"]) * 110540.0,
                       (lon - d["lon"]) * 111320.0 * math.cos(math.radians(lat)))
    return float(d["est"]) if moved <= max_move_m else None
