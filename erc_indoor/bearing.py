"""
Heading error between the live view and a teach-lap view (BearNav-style).

A pure yaw rotation shifts every feature horizontally by roughly the same
number of pixels, so the MODE of the horizontal displacements between matched
features is the heading difference. The mode (histogram vote), not the mean,
because matches on near-field floor tiles and reflections disagree with the
far corridor and must be outvoted rather than averaged in.

Sign convention (matches the rover: positive angular = LEFT):
  yaw_err_deg > 0  means the live camera points LEFT of the teach camera,
                   so the rover should turn RIGHT (negative angular).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

WORK_W, WORK_H = 512, 288
# fx measured by 360-degree spin loop closure at 1024 px (erc_urban/camera_front.json).
FX_1024 = 340.0
FX = FX_1024 * WORK_W / 1024.0
BIN_PX = 4.0
MAX_DY_PX = 24.0   # rotation is horizontal; big vertical shifts are wrong matches


@dataclass
class Bearing:
    yaw_err_deg: float
    confidence: float      # fraction of matches agreeing with the mode
    n_matches: int
    n_inliers: int

    @property
    def ok(self) -> bool:
        return self.n_inliers >= 12 and self.confidence >= 0.2


@dataclass
class Features:
    kp: np.ndarray        # (N, 2) xy at WORK size
    desc: Optional[np.ndarray]


_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_SIFT = cv2.SIFT_create(nfeatures=1000)
_BF = cv2.BFMatcher(cv2.NORM_L2)


def features(rgb: np.ndarray) -> Features:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
    g = _CLAHE.apply(g)
    kps, desc = _SIFT.detectAndCompute(g, None)
    xy = np.array([k.pt for k in kps], np.float32).reshape(-1, 2)
    return Features(xy, desc)


def px_to_deg(dx: float) -> float:
    return float(np.degrees(np.arctan2(dx, FX)))


def bearing(live: Features, ref: Features, ratio: float = 0.8) -> Bearing:
    if live.desc is None or ref.desc is None or len(live.kp) < 8 or len(ref.kp) < 8:
        return Bearing(0.0, 0.0, 0, 0)
    pairs = _BF.knnMatch(live.desc, ref.desc, k=2)
    good = [m for m, *rest in pairs if rest and m.distance < ratio * rest[0].distance]
    if len(good) < 8:
        return Bearing(0.0, 0.0, len(good), 0)
    a = live.kp[[m.queryIdx for m in good]]
    b = ref.kp[[m.trainIdx for m in good]]
    dx = a[:, 0] - b[:, 0]
    dy = a[:, 1] - b[:, 1]
    keep = np.abs(dy) < MAX_DY_PX
    dx = dx[keep]
    if len(dx) < 8:
        return Bearing(0.0, 0.0, len(good), 0)
    edges = np.arange(-WORK_W / 2, WORK_W / 2 + BIN_PX, BIN_PX)
    hist, _ = np.histogram(dx, bins=edges)
    hist = np.convolve(hist, [1, 2, 1], mode="same")
    k = int(hist.argmax())
    centre = 0.5 * (edges[k] + edges[k + 1])
    inl = np.abs(dx - centre) <= 2.5 * BIN_PX
    mode_dx = float(np.median(dx[inl])) if inl.any() else centre
    return Bearing(px_to_deg(mode_dx), float(inl.sum()) / len(dx), len(good), int(inl.sum()))


# ---- DINOv2 patch-grid heading (fallback when SIFT fails) -------------------
# One patch column of the 392-px-wide DINO input spans 1024/28 = 36.6 px of the
# 1024 frame, ~6.1 deg at fx 340. Validated against SIFT on lap-1 corners:
# +24.0 vs +22.2, +17.2 vs +15.2, -13.1 vs -13.1, -12.0 vs -11.6 deg.
DINO_PATCH_DEG = float(np.degrees(np.arctan((1024.0 / 28.0) / FX_1024)))


def dino_shift(live_tok, ref_tok, max_k: int = 10) -> tuple[float, float]:
    """(yaw_err_deg, margin). Same sign as bearing(): positive = live points
    LEFT of ref. margin = best mean patch similarity minus the median over
    shifts; ~0.2-0.35 on good pairs, <0.07 when the views do not correspond."""
    W = live_tok.shape[1]
    sc = []
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            a, b = live_tok[:, k:, :], ref_tok[:, :W - k, :]
        else:
            a, b = live_tok[:, :W + k, :], ref_tok[:, -k:, :]
        sc.append(float((a * b).sum(-1).mean()))
    sc = np.array(sc)
    i = int(sc.argmax())
    off = 0.0
    if 0 < i < len(sc) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float((i - max_k + off) * DINO_PATCH_DEG), float(sc.max() - np.median(sc))

