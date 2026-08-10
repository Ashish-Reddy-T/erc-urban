"""
Module 5a — planar visual odometry from the ground plane.

WHY THIS EXISTS
---------------
plan.md's M5 says to fix `PoseBuffer` by integrating wheel RPM and yaw rate.
Neither is available on this rover. Measured on the live Mini+:

  * `rpms` reads [0, 0, 0, 0] permanently, through 30 s of commanded driving
  * `speed` is pinned at 0
  * `latitude`/`longitude` are bit-identical across 24 stationary samples AND
    across that same 30 s of driving
  * the compass ran to -141 deg then reversed to -7 deg while the rover was
    commanded one direction throughout (magnetometer interference)

So every odometry source the plan assumed is either dead or untrustworthy. The
one sensor that demonstrably works is the camera, and the rover drives on a
plane whose geometry we now know: `calib.image_to_ground` converts a ground
pixel to robot-frame metres exactly. Tracking ground texture between frames
therefore yields real displacement -- this is the same measurement used to
establish that GPS was frozen in the first place.

WHAT IT FEEDS
-------------
  * `PoseBuffer` (M5 transport). Today run.py pushes `(stamp, 0.0, 0.0, yaw)`,
    so AsyncShield's SE(2) realignment has an identically-zero translation term
    and does nothing but rotate.
  * The breadcrumb trail, so a recovery manoeuvre can retrace ground the rover
    has already driven over instead of reversing blind. `/v2/screenshot`
    returns no rear frame, and a blind 15-tick reverse put this rover into a
    staircase.
  * Range-to-goal when GPS is frozen: dead-reckoned displacement is the only
    thing that can move `rng` off its pinned value.

METHOD
------
Track corners in the ground region (below the horizon, inside the depth-
trustworthy band), lift both frames' points to robot-frame metres, then fit the
rigid 2-D motion that maps one to the other (Kabsch). A world-fixed point seen
from a rover that moved by (t, dyaw) appears at

    q = R(-dyaw) @ (p - t)

so we solve for R and t directly, then invert to get the rover's motion.

Honest about failure, with measured limits:

  * Stationary is clean. On six real frames 1.5 s apart the total drift was
    0.01 cm and 0.0 deg at confidence 1.00 over ~750 tracks, and identical
    frames give exactly zero. No phantom motion.
  * Per-frame motion above ~0.1 m loses tracking, because the near-field ground
    under a 148 mm camera changes scale drastically between frames. At 5 Hz and
    0.3 m/s a tick is 0.06 m, which is comfortable -- but the measured loop runs
    nearer 1.5 Hz, where the same speed gives 0.2 m and VO will start dropping
    out. That shows up as confidence 0, not as a wrong number.
  * Large per-step ROTATION is worse than translation: over the 360 deg spin
    (up to 22 deg per step) accumulated yaw came out ~3x low. Small per-tick
    rotations in the control loop are fine.
  * Featureless ground gives nothing at all.
  * Drift accumulates, which is why Breadcrumbs is bounded in both distance and
    time rather than kept as a map.

Every estimate carries a confidence and callers are expected to stop trusting
it rather than assume it is always right. A confidently wrong displacement
would poison both the pose buffer and the retreat path.
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
        max_range_m: ground points further than this are dropped. Far points sit
            near the horizon where a single pixel spans a huge distance, so they
            dominate the fit while carrying the least information.
        horizon_margin_px: stay well clear of the horizon for the same reason.
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

        # Re-fit on inliers: a few bad tracks (a moving shadow, a pedestrian's
        # foot) otherwise drag the whole rigid fit.
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

    The point is escape. With no rear camera, reversing is blind, and a blind
    15-tick reverse drove this rover into a staircase. Ground the rover has
    already driven over is the best evidence available that a retreat is
    clear -- not proof, since the world moves, but far better than nothing.

    Every stored point is transported into the current frame on each update, so
    a caller can read them directly as "where I came from, from here". Bounded
    in both distance and time because visual odometry drifts and because stale
    evidence about a dynamic scene is not evidence.
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
        follow backwards. None when there is not enough trail to be useful --
        in which case the caller must NOT invent one.
        """
        if len(self.pts) < 3:
            return None
        pts = np.array([[x, y] for x, y, _ in self.pts], dtype=np.float64)
        order = np.argsort(np.linalg.norm(pts, axis=1))
        pts = pts[order]
        d = np.linalg.norm(pts, axis=1)
        pts = pts[d <= max_m]
        return pts if len(pts) >= 3 else None

    def retreat_bearing(self, max_m: float = 1.5) -> Optional[float]:
        """
        Direction the rover came from, left-positive.

        With no rear camera the better escape is often to TURN toward this
        bearing and drive forward with the camera looking where it is going,
        rather than to back along it blind.
        """
        p = self.retreat_path(max_m)
        if p is None:
            return None
        v = p.mean(axis=0)
        if float(np.hypot(*v)) < 1e-3:
            return None
        return float(np.arctan2(v[1], v[0]))
