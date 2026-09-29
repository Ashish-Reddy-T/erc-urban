"""
Lane following from a ground mask (Mexico finals: one straight dirt lane,
~150 m, between stacks of dry grass).

The general planner scores every heading and hunts left/right when its aim
sources disagree (Mexico trial 2026-09-28: 16 side flips, 5 m in 6 min). A
lane needs less: cast rover-wide rays over the ground mask and steer along the
one that runs furthest, nudged toward the lane's compass bearing. In a lane the
longest ray is the one along it; off-centre, it points back up the middle.

(A centreline fit was tried first: near rows see ground across the whole image,
clipped by its edges, so only far rows carried direction and the tracked run
drifted - 000192 steered 47 deg left on a straight lane.)

Ground = the lane surface only (earth / dirt track, "water" when wet); the
grass stacks read plant / tree / grass and are walls. Mexico trial frames:
bottom-centre earth 55 % + dirt track 9 %, sides plant 44 % + tree 20 %.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

LANE_NAMES = {"earth", "dirt track", "road", "path", "land", "sand", "water", "sidewalk",
              "floor", "ground",
              # dark wet lane seen low: trial frame 000040 read 89 % "mountain";
              # the grass stacks read it on ~1 % of their pixels.
              "mountain", "hill"}


@dataclass
class LaneEst:
    steer_deg: Optional[float]     # + = left; None = no drivable ray
    ahead_m: float                 # clear distance straight ahead (rover-wide strip)
    best_m: float                  # clear distance along the chosen ray
    left_m: float                  # best clear distance among rays > +15 deg
    right_m: float                 # best clear distance among rays < -15 deg
    ground_frac: float = 1.0       # lane pixels in the image's lower half (0 = lane-blind)
    angles: np.ndarray = field(default_factory=lambda: np.zeros(0))
    clear: np.ndarray = field(default_factory=lambda: np.zeros(0))


class LaneFinder:
    def __init__(self, calib, hw: tuple[int, int] = (144, 256), max_m: float = 5.0,
                 half_width_m: float = 0.2, fill: float = 0.6,
                 angles_deg=tuple(range(-50, 51, 5)), prior_w: float = 0.015,
                 cap_m: float = 3.0, edge_deg: float = 10.0):
        self.h, self.w = hw
        self.cal = calib.scaled_to(self.w, self.h)
        self.max_m = max_m
        self.half_w = half_width_m
        self.fill = fill
        self.angles = np.array(angles_deg, float)
        self.prior_w = prior_w          # score per degree away from the prior (m/deg)
        self.cap_m = cap_m              # beyond this, farther is not better (lane can curve)
        self.edge_deg = edge_deg        # keep the steer this far inside a wide gap
        # Sample each ray at fixed ground distances; precompute pixel strips.
        self.xs = np.arange(0.35, max_m + 1e-6, 0.1)
        self._strips = []               # per angle: [(x, v, a, b)]
        for ang in self.angles:
            t = math.radians(ang)
            pts = np.stack([self.xs * math.cos(t), self.xs * math.sin(t)], 1)
            uv, ok = self.cal.project_ground(pts)
            strips = []
            for x, (u, v), good in zip(self.xs, uv, ok):
                if not good or not (0 <= v < self.h):
                    strips.append(None)
                    continue
                du = max(1, int(round(self.cal.fx * self.half_w / max(x, 0.1))))
                strips.append((x, int(v), int(round(u)) - du, int(round(u)) + du))
            self._strips.append(strips)

    def _ray(self, g: np.ndarray, k: int) -> float:
        best = 0.0
        for s in self._strips[k]:
            if s is None:
                break
            x, v, a, b = s
            if b < 0 or a >= self.w:
                break                               # ray left the image
            seg = g[v, max(0, a):min(self.w, b + 1)]
            # A strip clipped by the image edge still has to be mostly ground.
            if seg.size < 2 or seg.mean() < self.fill:
                break
            best = x
        return best

    def estimate(self, g: np.ndarray, prior_deg: float = 0.0) -> LaneEst:
        """g: bool ground mask at hw. prior_deg: where the lane should be (+ left)."""
        assert g.shape == (self.h, self.w), g.shape
        clear = np.array([self._ray(g, k) for k in range(len(self.angles))])
        frac = float(g[self.h // 2:].mean())
        i0 = int(np.argmin(np.abs(self.angles)))
        left = clear[self.angles > 15]
        right = clear[self.angles < -15]
        steer, best = None, 0.0
        if clear.max() > 0.0:
            # Gap following: the contiguous runs of long rays; steer at the
            # middle of the chosen run, which centres the rover in the lane.
            # Taking the single longest ray instead let the prior break the
            # tie between many capped rays and hugged one edge (sim_lane:
            # wheels in the grass for 44 s of a 330 s lane).
            thr = min(self.cap_m, 0.85 * float(clear.max()))
            good = clear >= thr
            runs, k = [], 0
            while k < len(good):
                if good[k]:
                    j = k
                    while j + 1 < len(good) and good[j + 1]:
                        j += 1
                    runs.append((k, j))
                    k = j + 1
                else:
                    k += 1
            def run_score(r):
                a, b = r
                centre = float(np.mean(self.angles[a:b + 1]))
                return (min(float(clear[a:b + 1].max()), self.cap_m)
                        - self.prior_w * abs(centre - prior_deg))
            a, b = max(runs, key=run_score)
            lo, hi = float(self.angles[a]), float(self.angles[b])
            # Narrow gap (the lane): its centre. Wide gap (open ground at the
            # start or a junction): toward the prior, kept `edge_deg` inside the
            # gap - pure gap-centring ignored the goal in the open and circled
            # (fake e2e: 171 m and 870 s for a 131 m / 380 s mission).
            if hi - lo > 2 * self.edge_deg:
                steer = float(np.clip(prior_deg, lo + self.edge_deg, hi - self.edge_deg))
            else:
                steer = 0.5 * (lo + hi)
            best = float(clear[a:b + 1].max())
        return LaneEst(steer, float(clear[i0]), best,
                       float(left.max()) if left.size else 0.0,
                       float(right.max()) if right.size else 0.0, frac, self.angles, clear)
