"""
Checkpoint-cone detector: saturated colour blobs that stand on the floor and
are cone-sized.

Cones are NOT navigation targets (the route map is). They matter twice:
  1. they stand in the corridor, so they are obstacles to stop for and pass;
  2. seeing one pass by confirms a checkpoint in the log, and the orange one
     near the end of the map is the finish line.

Colour alone is not enough in this building — orange doors, blue recycling
bins, a green bin, red EXIT signs and a yellow cable on the floor all pass a
hue test, and the polished floor mirrors tall objects into cone-like blobs
below the horizon. What separates a cone is geometry: its base sits on the
floor below the horizon, its height from the base pixel's ground distance is
0.07-0.30 m, and it TAPERS (top fifth narrower than its widest row). Bins,
doors and their reflections are rectangles; signs have no floor contact; the
cable is flat.

Measured OpenCV hues (lap 2): orange cone 3, red cone 178 (= -2), blue 108,
green 84, yellow 27. Orange and red are not separable by hue, so both are
"red" here, and the map position says which checkpoint it is.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .calib import CameraCalibration

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_W, WORK_H = 512, 288
# (name, hue ranges) in OpenCV 0..180
HUES = {
    "red": [(0, 12), (168, 180)],
    "yellow": [(18, 38)],
    "green": [(60, 95)],
    "blue": [(100, 125)],
}
MIN_S, MIN_V = 120, 55
MIN_AREA = 40               # px at WORK size (~a cone 5 m away)
# The cones are ~15 cm tall (measured: their tops sit at the horizon row, i.e.
# at the 0.148 m camera height, at every distance).
CONE_H_MIN_M, CONE_H_MAX_M = 0.07, 0.30
ASPECT_MIN, ASPECT_MAX = 0.8, 4.5   # h / w; a cone is taller than wide
# A cone tapers: its top fifth is much narrower than its widest row. Bins,
# doors and the floor reflections of both are rectangles (taper ~1).
TAPER_MAX = 0.6
EDGE_PX = 3                 # blobs cut by the side edges are partial objects


@dataclass
class Cone:
    color: str
    x_m: float          # forward, robot frame
    y_m: float          # left
    height_m: float
    u: float            # base centre, WORK px
    v: float
    area: int

    @property
    def dist_m(self) -> float:
        return float(np.hypot(self.x_m, self.y_m))

    @property
    def bearing_deg(self) -> float:
        return float(np.degrees(np.arctan2(self.y_m, self.x_m)))


class ConeDetector:
    def __init__(self, calib: CameraCalibration | None = None):
        c = calib or CameraCalibration.load_or_default(
            os.path.join(HERE, "camera_front.json"), 1024, 576)
        self.calib = c.scaled_to(WORK_W, WORK_H)
        self.horizon = self.calib.horizon_v()
        self._k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, rgb: np.ndarray) -> list[Cone]:
        img = cv2.resize(rgb, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        sat = (s >= MIN_S) & (v >= MIN_V)
        out: list[Cone] = []
        for name, ranges in HUES.items():
            m = np.zeros_like(sat)
            for lo, hi in ranges:
                m |= (h >= lo) & (h < hi)
            m = (m & sat).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self._k)
            n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
            for k in range(1, n):
                x, y, w, hh, area = st[k]
                if area < MIN_AREA or w == 0:
                    continue
                aspect = hh / w
                if not ASPECT_MIN <= aspect <= ASPECT_MAX:
                    continue
                if x <= EDGE_PX or x + w >= WORK_W - EDGE_PX:
                    continue
                rows = (lab[y:y + hh, x:x + w] == k).sum(axis=1)
                top = rows[: max(1, hh // 5)].mean()
                if top > TAPER_MAX * rows.max():
                    continue
                vb = float(y + hh)
                # Blobs cut by the bottom edge: the base is below frame. Keep
                # them (a cone right in front of us) and mark them very close.
                if vb < self.horizon + 3:
                    continue           # no floor contact: signs, doors' upper parts
                ub = float(x + w / 2.0)
                xy, ok = self.calib.image_to_ground(np.array([[ub, vb]]))
                if not ok[0]:
                    continue
                gx, gy = float(xy[0, 0]), float(xy[0, 1])
                d = float(np.hypot(gx, gy))
                height = hh * d / self.calib.fy
                touches_bottom = vb >= WORK_H - 2
                if not touches_bottom and not CONE_H_MIN_M <= height <= CONE_H_MAX_M:
                    continue
                out.append(Cone(name, gx, gy, height, ub, vb, int(area)))
        out.sort(key=lambda c: c.dist_m)
        return out
