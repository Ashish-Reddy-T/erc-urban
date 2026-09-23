"""
Free space per heading — the obstacle avoidance.

Live run 3 drove straight into the wall beside the piano-corridor doorway:
heading matched the teach lap (the rover was parallel to it) but it was offset
sideways, and nothing looked at what was actually in front. This module does.

Method: where does the FLOOR end? SegFormer-B2 (ADE20K, cached; the same model
erc_urban/modules/semantics.py uses) labels floor pixels. For each image column
scan up from the bottom row to the first non-floor pixel; that boundary row,
through the calibrated camera (fx 340 @1024, 0.148 m high, 3 deg pitch), is
the distance at which something stands on the floor in that direction. Walls,
doorframes, furniture, people and cones all end the floor.

Why not metric depth: Depth Anything V2 (hypersim) on this 15 cm-high camera
over a polished floor reads the floor ~16x too far (median floor ratio 0.06 on
both laps — it sees the ceiling reflected in it) and put a wall 2.3 m away one
second before touching it. The floor boundary needs no scale at all.

Every heading is judged by the rover-wide strip swept along it (path_clear),
so an obstacle beside the front corner counts even when it is not "ahead".
`steer(want_deg, need_m)` picks the heading nearest the teach heading whose
strip is clear — a wall ahead with a doorway to the left turns the
rover into the doorway instead of into the wall.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch

from .calib import CameraCalibration

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = os.environ.get("SEG_MODEL", "nvidia/segformer-b2-finetuned-ade-512-512")
FLOOR_IDS = (3, 28)            # ADE20K: floor/flooring, rug/carpet
SEG_W, SEG_H = 512, 288
N_BINS = 21
HALF_FOV_DEG = 50.0            # ignore the extreme, barrel-distorted edges
CAP_M = 5.0
BOTTOM_SKIP = 4                # rows of lens vignette at the very bottom
THIN_BAND_PX = 14              # non-floor bands thinner than this (rows @288) are thresholds
# Swept-path check. Full lap run: the rover clipped the left doorframe of the
# piano corridor ~6 times — the pillar edge was 0.2-0.3 m off the left front
# corner, outside the old +-7.5 deg "ahead" wedge (the body spans +-23 deg at
# 0.3 m). A heading is judged by the rover-wide strip swept along it.
HALF_WIDTH_M = 0.17            # 190 mm body / 2 + ~7.5 cm margin (0.20 made doorways look shut)
HEADING_STEP_DEG = 5.0


@dataclass
class FreeSpace:
    bins_deg: np.ndarray        # candidate headings, deg, left-positive
    clear_m: np.ndarray         # swept-path clearance along each heading, m (capped)
    ahead_m: float              # swept-path clearance straight ahead
    pts: Optional[np.ndarray] = None   # (N,2) obstacle feet in robot frame (x fwd, y left)

    def clear_at(self, deg: float) -> float:
        if self.pts is not None:
            return path_clear(self.pts, deg)
        return float(np.interp(deg, self.bins_deg, self.clear_m))


def path_clear(pts: np.ndarray, heading_deg: float, half_w: float = HALF_WIDTH_M) -> float:
    """Distance the rover can drive along heading before its body strip meets
    an obstacle foot."""
    if pts is None or len(pts) == 0:
        return CAP_M
    a = np.radians(heading_deg)
    c, s_ = np.cos(a), np.sin(a)
    x = pts[:, 0] * c + pts[:, 1] * s_          # along the heading
    y = -pts[:, 0] * s_ + pts[:, 1] * c         # across it
    m = (x > 0.0) & (np.abs(y) < half_w)
    return float(min(CAP_M, x[m].min())) if m.any() else CAP_M


class FreeSpaceEstimator:
    def __init__(self, device: Optional[str] = None):
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation
        from .place import auto_device
        self.device = device or auto_device()
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID).eval().to(self.device)
        c = CameraCalibration.load_or_default(os.path.join(HERE, "camera_front.json"), 1024, 576)
        self.calib = c.scaled_to(SEG_W, SEG_H)
        self.horizon = self.calib.horizon_v()
        u = np.arange(SEG_W) + 0.5
        v = np.arange(SEG_H) + 0.5
        uu, vv = np.meshgrid(u, v)
        xy, ok = self.calib.image_to_ground(np.stack([uu.ravel(), vv.ravel()], 1))
        rng = np.hypot(xy[:, 0], xy[:, 1]).reshape(SEG_H, SEG_W)
        self._rng = np.where(ok.reshape(SEG_H, SEG_W), np.minimum(rng, CAP_M), CAP_M)
        self._gx = xy[:, 0].reshape(SEG_H, SEG_W)
        self._gy = xy[:, 1].reshape(SEG_H, SEG_W)
        self._gok = ok.reshape(SEG_H, SEG_W)
        self._col_deg = np.degrees(np.arctan2(-(u - self.calib.cx), self.calib.fx))
        self.bins = np.arange(-HALF_FOV_DEG, HALF_FOV_DEG + 1e-6, HEADING_STEP_DEG)
        self.last_floor: Optional[np.ndarray] = None
        self.last_boundary: Optional[np.ndarray] = None

    @torch.no_grad()
    def floor_mask(self, rgb: np.ndarray) -> np.ndarray:
        small = cv2.resize(rgb, (SEG_W, SEG_H), interpolation=cv2.INTER_AREA)
        inp = self.proc(images=small, return_tensors="pt", do_resize=False)
        logits = self.model(pixel_values=inp["pixel_values"].to(self.device)).logits
        logits = torch.nn.functional.interpolate(logits, size=(SEG_H, SEG_W), mode="bilinear",
                                                 align_corners=False)
        lab = logits.argmax(1)[0].cpu().numpy()
        return np.isin(lab, FLOOR_IDS)

    def estimate(self, rgb: np.ndarray) -> FreeSpace:
        floor = self.floor_mask(rgb)
        # light cleanup: grout lines / specks must not end the floor
        floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((5, 3), np.uint8)).astype(bool)
        # A thin horizontal band of non-floor with floor beyond it is a door
        # threshold / seam / mat edge, not an obstacle (full lap run: the CP4
        # doorway threshold read as "floor ends 0.4 m" five times and the rover
        # boxed itself in). Close vertical gaps up to THIN_BAND_PX rows.
        floor = cv2.morphologyEx(floor.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((THIN_BAND_PX, 1), np.uint8)).astype(bool)
        self.last_floor = floor
        H = SEG_H - BOTTOM_SKIP
        rev = (~floor[:H])[::-1]
        has = rev.any(axis=0)
        first_from_bottom = np.where(has, rev.argmax(axis=0), H)
        boundary_row = np.clip((H - 1) - first_from_bottom, 0, SEG_H - 1)
        self.last_boundary = boundary_row
        cols = np.arange(SEG_W)
        # obstacle foot of every column, robot frame; boundary at/above the
        # horizon (floor runs to the far end) contributes nothing
        valid = (boundary_row > self.horizon + 1) & self._gok[boundary_row, cols]
        pts = np.stack([self._gx[boundary_row, cols], self._gy[boundary_row, cols]], 1)[valid]
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < CAP_M]
        clear = np.array([path_clear(pts, float(b)) for b in self.bins])
        ahead = path_clear(pts, 0.0)
        return FreeSpace(self.bins.copy(), clear, ahead, pts)

    @staticmethod
    def steer(fs: FreeSpace, want_deg: float, need_m: float) -> Optional[float]:
        """Heading (deg, left+) nearest `want_deg` whose clearance >= need_m,
        requiring the neighbouring bins too (the rover is wider than a ray).
        None if no heading in view is clear."""
        ok = fs.clear_m >= need_m     # already rover-width (swept path)
        if not ok.any():
            return None
        cand = fs.bins_deg[ok]
        return float(cand[np.argmin(np.abs(cand - want_deg))])
