"""
Camera-only terrain reading: where can this rover actually drive, and how fast.

NO LIDAR, NO DEPTH MODEL, NO GPS. Monocular RGB only, classical, ~5 ms a frame.

WHY ROUGHNESS AND NOT DEPTH
---------------------------
A monocular depth model plus RANSAC ground fit is the obvious approach and it
provably fails here: `test_step_hazard.py` already documents that a slab edge
"presents almost nothing above the fitted plane", so the geometric veto goes
blind exactly at the hazard. In this rock garden that failure mode IS the whole
ground surface.

What actually separates driveable from not, in these frames, is TEXTURE. A flat
flagstone slab is a large region of near-uniform intensity with low local
gradient. Rounded river cobble is high-frequency gradient everywhere. Those
are trivially separable without a model, and -- unlike depth -- the signal gets
STRONGER the closer the hazard is, which is the direction that matters.

WHAT IT CANNOT SEE
------------------
The camera sits at ~0.148 m and roughly the first 0.4 m ahead of the wheels is
below the frame. Every failure on 2026-08-25 happened inside that blind zone.
This module therefore reports `blind_zone_m` honestly and the controller must
treat the near field as UNKNOWN, not as clear.

COLUMN GEOMETRY
---------------
Measured fx on this camera implies ~112.8 deg horizontal FOV (from the urban
stack's costmap calibration, closing a 360 deg pivot). Barrel distortion is
severe at the edges, so bearings from edge columns are only approximate --
which is fine, because they are used to RANK directions, not to servo.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FOV_H_DEG = 112.8
# Fraction of frame height treated as ground. Above this is scene, not floor.
GROUND_BAND = 0.45
N_COLS = 24
N_ROWS = 10
# Gradient energy above which a cell is "rough". Calibrated on 2026-08-25
# frames: flat slab cells sit well below, cobble fields well above.
ROUGH_THRESH = 14.0
# Cells this bright are blown highlights (sky through the roof, lamp glare)
# and carry no texture information either way.
BLOWN = 245.0


@dataclass
class TerrainView:
    rough: np.ndarray        # (N_ROWS, N_COLS) mean gradient, row 0 = nearest
    clear: np.ndarray        # (N_COLS,) rows of smooth run upward from bottom
    bearings: np.ndarray     # (N_COLS,) approx bearing, radians, left-positive
    best_bearing: float
    best_depth: float        # 0..1 fraction of the band that is clear ahead
    centre_depth: float      # same, straight ahead -- what we would hit going fwd
    open_frac: float         # fraction of all cells that are smooth
    blind_zone_m: float = 0.4

    def describe(self) -> str:
        return (f"terrain[best={np.degrees(self.best_bearing):+.0f}deg "
                f"depth={self.best_depth:.2f} centre={self.centre_depth:.2f} "
                f"open={self.open_frac:.2f}]")


def analyze(rgb: np.ndarray) -> Optional[TerrainView]:
    if rgb is None or rgb.ndim != 3:
        return None
    h, w = rgb.shape[:2]
    gray = rgb.astype(np.float32).mean(axis=2)
    band = gray[int(h * (1.0 - GROUND_BAND)):, :]
    bh, bw = band.shape

    gx = np.abs(np.diff(band, axis=1, prepend=band[:, :1]))
    gy = np.abs(np.diff(band, axis=0, prepend=band[:1, :]))
    grad = gx + gy
    # Blown-out cells have no texture to measure; mark them neutral-rough so
    # they are never mistaken for beautifully smooth pavement.
    grad = np.where(band > BLOWN, ROUGH_THRESH * 1.5, grad)

    rows = np.array_split(np.arange(bh), N_ROWS)
    cols = np.array_split(np.arange(bw), N_COLS)
    rough = np.zeros((N_ROWS, N_COLS), dtype=np.float32)
    for i, rr in enumerate(rows):
        for j, cc in enumerate(cols):
            rough[i, j] = float(grad[np.ix_(rr, cc)].mean())
    # Row 0 = nearest to the rover (bottom of the image).
    rough = rough[::-1, :]

    smooth = rough < ROUGH_THRESH
    clear = np.zeros(N_COLS, dtype=np.float32)
    for j in range(N_COLS):
        n = 0
        for i in range(N_ROWS):
            if not smooth[i, j]:
                break
            n += 1
        clear[j] = n

    centres = (np.arange(N_COLS) + 0.5) / N_COLS      # 0..1 left->right
    # +bearing = LEFT (verified on this unit), so flip the image x axis.
    bearings = np.radians((0.5 - centres) * FOV_H_DEG)

    # Prefer depth, but break ties toward the centre: an edge column's bearing
    # is the least trustworthy thing in this frame.
    score = clear - 0.6 * np.abs(np.arange(N_COLS) - (N_COLS - 1) / 2.0) / N_COLS
    j_best = int(np.argmax(score))
    c0, c1 = int(N_COLS * 0.375), int(N_COLS * 0.625)

    return TerrainView(
        rough=rough,
        clear=clear,
        bearings=bearings,
        best_bearing=float(bearings[j_best]),
        best_depth=float(clear[j_best] / N_ROWS),
        centre_depth=float(clear[c0:c1].mean() / N_ROWS),
        open_frac=float(smooth.mean()),
    )


def speed_policy(tv: TerrainView, pitch_deg: float, stuck_score: float,
                 v_min: float = 0.16, v_max: float = 0.45) -> tuple[float, str]:
    """
    How fast to drive, and why. This is the "max when it can, min when it must"
    decision, kept in ONE place so the reasoning is auditable in the log.

    Speed rises with how far the clear run extends ahead and falls for every
    reason to be careful. The nose-up term is the important one: on 2026-08-25
    every failure was a partial climb that slid back, and pitch is knowable
    BEFORE committing rather than only after.
    """
    reasons = []
    f = 0.35 + 0.65 * float(np.clip(tv.centre_depth, 0.0, 1.0))
    if tv.centre_depth < 0.25:
        f *= 0.5; reasons.append("near-field rough")
    if tv.open_frac < 0.30:
        f *= 0.7; reasons.append("cluttered")
    if pitch_deg > 8.0:
        # Nose-up: climbing. Needs authority, but bounded -- see climb.py for
        # why escalation is gated rather than automatic.
        f *= 1.25; reasons.append(f"climbing {pitch_deg:+.0f}deg")
    if pitch_deg < -10.0:
        f *= 0.6; reasons.append(f"nose-down {pitch_deg:+.0f}deg")
    if stuck_score > 0.5:
        f *= 1.2; reasons.append("stuck-boost")
    v = float(np.clip(v_min + (v_max - v_min) * np.clip(f, 0.0, 1.0),
                      v_min, v_max))
    return v, (", ".join(reasons) or "clear")
