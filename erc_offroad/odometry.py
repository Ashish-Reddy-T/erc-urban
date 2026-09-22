"""
Planar visual odometry from the ground plane — the only working motion sensor
on this rover.

Every alternative is dead or untrustworthy. Measured: `rpms` reads [0,0,0,0]
through 30 s of commanded driving, `speed` is pinned at 0, lat/lon are
bit-identical across both 24 stationary samples and that same 30 s, and the
compass ran to -141 deg then back to -7 deg while commanded one way throughout.
The camera works, and `calib.image_to_ground` converts a ground pixel to
robot-frame metres exactly, so tracking ground texture yields real displacement.

METHOD
------
Track corners in the ground region, lift both frames to robot-frame metres, then
fit the rigid 2-D motion mapping one to the other (Kabsch). A world-fixed point
seen from a rover that moved by (t, dyaw) appears at

    q = R(-dyaw) @ (p - t)

so solve for R and t, then invert for the rover's motion.

MEASURED LIMITS
---------------
  * Stationary is clean: 0.01 cm and 0.0 deg of drift over six frames 1.5 s
    apart, and identical frames give exactly zero. No phantom motion.
  * Per-frame motion above ~0.1 m loses tracking, because near-field ground
    under a 148 mm camera changes scale fast. At the measured ~1.5 Hz loop and
    0.3 m/s a tick is 0.2 m, so VO will drop out — as confidence 0, not as a
    wrong number.
  * Large per-step ROTATION is worse: over a 360 deg spin (22 deg per step)
    accumulated yaw came out ~3x low. Small per-tick rotations are fine.
  * Featureless ground gives nothing, and drift accumulates — which is why
    Breadcrumbs is bounded in distance and time rather than kept as a map.

Every estimate carries a confidence; callers are expected to stop trusting it
rather than assume it is always right.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OdomDelta:
    """Rover motion between two frames, in the earlier frame's robot frame."""

    dx: float                 # forward, metres
    dy: float                 # left, metres
    dyaw: float               # radians, left-positive
    confidence: float         # 0..1
    n_tracks: int = 0
    residual_m: float = 0.0
    reason: str = "ok"

    @property
    def distance_m(self) -> float:
        return float(np.hypot(self.dx, self.dy))


def _kabsch_2d(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid 2-D transform (R, t) minimising ||R @ p + t - q||."""
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return R, qc - R @ pc


class VisualOdometry:
    """Frame-to-frame planar odometry using the calibrated ground plane."""

    def __init__(
        self,
        calib,
        min_tracks: int = 40,
        max_range_m: float = 3.0,
        min_range_m: float = 0.20,
        horizon_margin_px: float = 25.0,
        max_residual_m: float = 0.08,
        max_step_m: float = 1.0,
    ):
        """
        max_range_m: drop ground points further than this. Near the horizon a
            single pixel spans a huge distance, so far points dominate the fit
            while carrying the least information.
        horizon_margin_px: stay clear of the horizon, for the same reason.
        max_step_m: a single-frame jump larger than this is a tracking failure,
            not motion. At ~1.5 Hz and 0.3 m/s the true step is ~0.2 m.
        """
        self.calib = calib
        self.min_tracks = min_tracks
        self.max_range_m = max_range_m
        self.min_range_m = min_range_m
        self.horizon_margin_px = horizon_margin_px
        self.max_residual_m = max_residual_m
        self.max_step_m = max_step_m
        self._prev_gray = None
        self.last: Optional[OdomDelta] = None

    def reset(self) -> None:
        self._prev_gray = None

    def _ground_points(self, cal, pts: np.ndarray):
        xy, ok = cal.image_to_ground(pts.astype(np.float64))
        r = np.hypot(xy[:, 0], xy[:, 1])
        good = ok & (r > self.min_range_m) & (r < self.max_range_m)
        return xy, good

    def update(self, frame: np.ndarray) -> OdomDelta:
        """Feed the current RGB frame; returns motion since the previous one."""
        try:
            import cv2
        except Exception:
            return OdomDelta(0.0, 0.0, 0.0, 0.0, reason="no_cv2")

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        prev, self._prev_gray = self._prev_gray, gray
        if prev is None or prev.shape != gray.shape:
            return OdomDelta(0.0, 0.0, 0.0, 0.0, reason="first_frame")

        h, w = gray.shape
        cal = self.calib.scaled_to(w, h)
        y_min = cal.horizon_v() + self.horizon_margin_px

        mask = np.zeros((h, w), np.uint8)
        mask[int(max(0, y_min)):, :] = 255
        p0 = cv2.goodFeaturesToTrack(prev, 900, 0.005, 8, mask=mask, blockSize=7)
        if p0 is None or len(p0) < self.min_tracks:
            return OdomDelta(0.0, 0.0, 0.0, 0.0, reason="no_features")

        lk = dict(winSize=(31, 31), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, **lk)
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, p1, None, **lk)
        fb = np.abs(p0 - p0b).reshape(-1, 2).max(axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
        A, B = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        if len(A) < self.min_tracks:
            return OdomDelta(0.0, 0.0, 0.0, 0.0, len(A), reason="tracking_lost")

        xy0, g0 = self._ground_points(cal, A)
        xy1, g1 = self._ground_points(cal, B)
        g = g0 & g1
        if int(g.sum()) < self.min_tracks:
            return OdomDelta(0.0, 0.0, 0.0, 0.0, int(g.sum()), reason="few_ground_pts")

        P, Q = xy0[g], xy1[g]
        R, t = _kabsch_2d(P, Q)
        resid = float(np.median(np.linalg.norm((P @ R.T + t) - Q, axis=1)))

        # Re-fit on inliers: a moving shadow or a pedestrian's foot otherwise
        # drags the whole rigid fit.
        err = np.linalg.norm((P @ R.T + t) - Q, axis=1)
        keep = err <= max(2.0 * resid, 0.01)
        if int(keep.sum()) >= self.min_tracks:
            R, t = _kabsch_2d(P[keep], Q[keep])
            resid = float(np.median(np.linalg.norm((P[keep] @ R.T + t) - Q[keep], axis=1)))

        # q = R(-dyaw) (p - t_rover)  =>  rover motion is the inverse map.
        dyaw = -float(np.arctan2(R[1, 0], R[0, 0]))
        t_rover = -(R.T @ t)
        dx, dy = float(t_rover[0]), float(t_rover[1])

        if not np.isfinite([dx, dy, dyaw]).all():
            return OdomDelta(0.0, 0.0, 0.0, 0.0, int(g.sum()), reason="non_finite")
        if np.hypot(dx, dy) > self.max_step_m:
            return OdomDelta(0.0, 0.0, 0.0, 0.0, int(g.sum()),
                             residual_m=resid, reason="implausible_step")

        conf = float(np.clip(1.0 - resid / max(self.max_residual_m, 1e-9), 0.0, 1.0))
        conf *= float(np.clip(int(keep.sum()) / (4.0 * self.min_tracks), 0.0, 1.0))
        self.last = OdomDelta(dx, dy, dyaw, conf, int(keep.sum()), resid)
        return self.last


class Breadcrumbs:
    """
    Short memory of where the rover has just been, in the CURRENT robot frame.

    The point is escape. With no rear camera, reversing is blind — a blind
    15-tick reverse drove this rover into a staircase — and ground already
    driven over is the best evidence a retreat is clear.

    Points are transported into the current frame on every update, so a caller
    reads them directly as "where I came from, from here". Bounded in distance
    and time because VO drifts and stale evidence about a dynamic scene is not
    evidence.
    """

    def __init__(self, max_points: int = 60, max_distance_m: float = 4.0,
                 max_age_s: float = 45.0, min_spacing_m: float = 0.05):
        self.pts: deque = deque(maxlen=max_points)     # (x, y, stamp)
        self.max_distance_m = max_distance_m
        self.max_age_s = max_age_s
        self.min_spacing_m = min_spacing_m
        self.travelled_m = 0.0

    def reset(self) -> None:
        self.pts.clear()
        self.travelled_m = 0.0

    def update(self, delta: OdomDelta, stamp: float) -> None:
        """Transport the trail into the new robot frame, then drop a crumb."""
        if delta.confidence <= 0.0:
            return
        c, s = np.cos(-delta.dyaw), np.sin(-delta.dyaw)
        R = np.array([[c, -s], [s, c]])
        moved = np.array([delta.dx, delta.dy])
        self.pts = deque(
            [(*(R @ (np.array([x, y]) - moved)), t) for x, y, t in self.pts],
            maxlen=self.pts.maxlen,
        )
        # The origin of the previous frame is now behind us: that is a crumb.
        prev_origin = R @ (-moved)
        if (
            not self.pts
            or float(np.hypot(*(np.array(self.pts[-1][:2]) - prev_origin)))
            >= self.min_spacing_m
        ):
            self.pts.append((float(prev_origin[0]), float(prev_origin[1]), stamp))
        self.travelled_m += delta.distance_m
        self._prune(stamp)

    def _prune(self, now: float) -> None:
        keep = [
            (x, y, t) for x, y, t in self.pts
            if (now - t) <= self.max_age_s
            and float(np.hypot(x, y)) <= self.max_distance_m
        ]
        self.pts = deque(keep, maxlen=self.pts.maxlen)

    def retreat_path(self, max_m: float = 1.5) -> Optional[np.ndarray]:
        """
        Recently-driven ground, nearest first, as an (N,2) robot-frame path to
        follow backwards. None when the trail is too short to be useful — in
        which case the caller must NOT invent one.
        """
        if len(self.pts) < 3:
            return None
        pts = np.array([[x, y] for x, y, _ in self.pts], dtype=np.float64)
        order = np.argsort(np.linalg.norm(pts, axis=1))
        pts = pts[order]
        d = np.linalg.norm(pts, axis=1)
        pts = pts[d <= max_m]
        return pts if len(pts) >= 3 else None

    def extent_m(self) -> float:
        """
        How far the trail reaches — net displacement, not path length.

        `travelled_m` sums |delta| per tick and so counts noise and skid: a rover
        yawing in place accumulated 4.0 m of "travel" and 37 crumbs without
        leaving a 1 m circle.
        """
        if not self.pts:
            return 0.0
        return float(max(np.hypot(x, y) for x, y, _ in self.pts))

    def retreat_bearing(
        self, max_m: float = 1.5, min_extent_m: float = 0.4
    ) -> Optional[float]:
        """
        Direction the rover came from, left-positive. None when unknowable.

        With no rear camera the better escape is usually to TURN toward this and
        drive forward, camera looking where it is going, rather than back along
        it blind.

        `min_extent_m` is load-bearing, not a tuning knob: under an in-place
        pivot odometry still reports 4-8 cm of skid per tick at high confidence,
        so crumbs accumulate and their centroid points in a RANDOM direction.
        That produced eight consecutive escapes at +116, +113, +134, +130, -160,
        +112, -165 and +101 degrees while the rover stayed put. A trail with no
        extent does not know where we came from and must say so.
        """
        p = self.retreat_path(max_m)
        if p is None:
            return None
        if self.extent_m() < min_extent_m:
            return None
        v = p.mean(axis=0)
        if float(np.hypot(*v)) < 1e-3:
            return None
        return float(np.arctan2(v[1], v[0]))
