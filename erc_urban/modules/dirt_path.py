"""
Classical path perception for Stage 0/1 (no ML). Only reachable under
USE_DIRT_GATE=1.

Brown dirt and grey asphalt are traversable; green grass is the hard veto.
Brightness floors adapt to the scene mean so night roads are not falsely
blocked when dirt_frac looks low.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from ..interfaces import Candidate, Observation


@dataclass
class PathView:
    dirt_frac: float       # traversable fraction in near ROI (dirt + asphalt)
    grass_frac: float
    center_offset: float   # +1 = free mass left of center → turn left
    blocked: bool          # hard stop: grass wall / buried / no free corridor
    uncertain: bool        # dim / low-confidence — pivot, do not reverse
    dirt: np.ndarray       # HxW bool traversable
    mean_bright: float = 128.0


def analyze(front: np.ndarray, near_band: float = 0.35) -> PathView:
    """RGB uint8 → PathView. Uses lower `near_band` of the image as ROI."""
    if front is None or front.size == 0 or front.ndim != 3:
        z = np.zeros((1, 1), dtype=bool)
        return PathView(0.0, 1.0, 0.0, True, False, z, 0.0)

    h, w = front.shape[:2]
    mean_b = float(front.mean())
    night = mean_b < 70.0
    dirt = _traversable_mask(front, mean_b)
    grass = _grass_mask(front, mean_b)
    y0 = int(h * (1.0 - near_band))
    roi = dirt[y0:, :]
    grass_roi = grass[y0:, :]
    bright_roi = front[y0:].astype(np.float32).mean(axis=2)

    n = max(roi.size, 1)
    dirt_frac = float(roi.mean())
    grass_frac = float(grass_roi.mean())
    white_frac = float((bright_roi > 230).mean())
    hot_frac = float((bright_roi > 200).mean())

    xs = np.nonzero(roi)[1]
    if xs.size < max(20, n // 200):
        offset = 0.0
    else:
        offset = float(np.clip((w * 0.5 - xs.mean()) / (w * 0.5), -1.0, 1.0))

    c0, c1 = int(0.28 * w), int(0.72 * w)
    center = float(roi[:, c0:c1].mean()) if roi.size else 0.0
    left = float(roi[:, : int(0.22 * w)].mean()) if roi.size else 0.0
    right = float(roi[:, int(0.78 * w) :].mean()) if roi.size else 0.0
    side = 0.5 * (left + right)
    min_c = 0.06 if night else 0.10
    has_corridor = center >= min_c and center >= side - 0.02

    sky = _sky_frac(front)
    edges = _edge_frac(front, y0)
    # Washed / buried lens: blown whites filling the near field.
    washed = (not night) and (
        (white_frac > 0.30 and dirt_frac < 0.30)
        or (hot_frac > 0.40 and dirt_frac < 0.28)
    )
    buried = washed or (
        sky < 0.05 and edges > 0.20 and not has_corridor
    ) or (
        sky < 0.05 and dirt_frac < 0.12 and grass_frac > 0.15
    )

    hard_grass = grass_frac > (0.55 if night else 0.65)
    no_path = (not has_corridor) and grass_frac > (0.25 if night else 0.35)
    day_empty = (not night) and dirt_frac < 0.08 and not has_corridor
    blocked = hard_grass or buried or no_path or day_empty or (
        dirt_frac < 0.04 and grass_frac > 0.20
    )

    uncertain = False
    if night and dirt_frac < 0.08 and grass_frac < 0.15 and not buried:
        blocked = False
        uncertain = True
    elif (not blocked) and night and dirt_frac < 0.12 and grass_frac < 0.20:
        uncertain = True

    return PathView(
        dirt_frac, grass_frac, offset, blocked, uncertain, dirt, mean_b
    )


def rescore(cands: list[Candidate], path: PathView, goal_w: float = 0.6,
            path_w: float = 1.2) -> list[Candidate]:
    """Blend goal score with alignment to path center (left-positive)."""
    out: list[Candidate] = []
    path_bearing = path.center_offset * (35.0 * np.pi / 180.0)
    for c in cands:
        end = c.waypoints[-1]
        end_bearing = float(np.arctan2(end[1], max(end[0], 1e-3)))
        path_term = -abs(_wrap(end_bearing - path_bearing))
        score = goal_w * c.score + path_w * path_term
        if path.dirt_frac < 0.12 and not path.uncertain:
            score -= 0.5
        out.append(Candidate(waypoints=c.waypoints, score=float(score), stamp=c.stamp))
    return out


class DirtPathGate:
    """Hard veto on grass; at night, veto grass only."""

    def __init__(
        self,
        min_dirt: float = 0.28,
        max_grass: float = 0.30,
        sample_m: float = 1.8,
    ):
        self.min_dirt = min_dirt
        self.max_grass = max_grass
        self.sample_m = sample_m

    def mask(self, obs: Observation, cands: list[Candidate]) -> np.ndarray:
        path = analyze(obs.front)
        if path.blocked:
            return np.zeros(len(cands), dtype=bool)
        h, w = obs.front.shape[:2]
        grass = _grass_mask(obs.front, path.mean_bright)
        soft = path.uncertain or path.mean_bright < 70.0
        min_dirt = 0.12 if soft else self.min_dirt
        ok = np.zeros(len(cands), dtype=bool)
        for i, c in enumerate(cands):
            ok[i] = self._arc_ok(
                c.waypoints, path.dirt, grass, w, h, min_dirt, soft
            )
        return ok

    def _arc_ok(
        self,
        wp: np.ndarray,
        dirt: np.ndarray,
        grass: np.ndarray,
        w: int,
        h: int,
        min_dirt: float,
        soft: bool,
    ) -> bool:
        dirt_hits = 0
        grass_hits = 0
        n = 0
        for x, y in wp:
            if x < 0.2 or x > self.sample_m:
                continue
            u, v = _robot_to_uv(float(x), float(y), w, h)
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            n += 1
            vi, ui = int(v), int(u)
            if grass[vi, ui]:
                grass_hits += 1
            if dirt[vi, ui]:
                dirt_hits += 1
        if n == 0:
            return soft  # night: keep LogoNav cand if samples miss image
        if (grass_hits / n) > self.max_grass:
            return False
        if soft:
            return True  # grass veto only
        return (dirt_hits / n) >= min_dirt


def _traversable_mask(rgb: np.ndarray, mean_b: float | None = None) -> np.ndarray:
    """Brown soil + grey asphalt. Floors scale with scene brightness."""
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    bright = (r + g + b) / 3.0
    if mean_b is None:
        mean_b = float(bright.mean())
    green = _grass_mask(rgb, mean_b)
    lo = max(8.0, mean_b * 0.25)
    hi = min(195.0, max(110.0, mean_b * 2.0))  # blown whites are never road

    soil = (
        ~green
        & (bright > lo)
        & (bright < hi)
        & (r >= g - 20)
        & (g >= b - 15)
        & (r - b > 3)
    )
    asphalt = (
        ~green
        & (bright > lo)
        & (bright < hi)
        & (np.abs(r - g) < 28)
        & (np.abs(g - b) < 28)
        & (np.abs(r - b) < 32)
    )
    return soil | asphalt


def _grass_mask(rgb: np.ndarray, mean_b: float | None = None) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    bright = (r + g + b) / 3.0
    if mean_b is None:
        mean_b = float(bright.mean())
    g_floor = max(18.0, mean_b * 0.35)
    lush = (g > r + 10) & (g > b + 6) & (g > g_floor)
    pale = (
        (g >= r - 5)
        & (g > b + 8)
        & (bright > max(50.0, mean_b * 0.7))
        & (g > max(40.0, mean_b * 0.5))
        & (bright < 230)
    )
    return lush | pale


def _sky_frac(rgb: np.ndarray, top: float = 0.28) -> float:
    h = rgb.shape[0]
    band = rgb[: max(1, int(h * top))]
    r = band[:, :, 0].astype(np.float32)
    g = band[:, :, 1].astype(np.float32)
    b = band[:, :, 2].astype(np.float32)
    bright = (r + g + b) / 3.0
    blue = (b > r + 12) & (b >= g - 8) & (bright > 90) & (bright < 250)
    return float(blue.mean())


def _edge_frac(rgb: np.ndarray, y0: int) -> float:
    gray = rgb.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge = (gx + gy) > 28.0
    return float(edge[y0:].mean())


def _robot_to_uv(x_m: float, y_left_m: float, w: int, h: int) -> tuple[float, float]:
    """Crude ground-plane: near → image bottom; +y_left → image left."""
    look = max(x_m, 0.25)
    v = h * (0.95 - 0.40 * np.clip((look - 0.4) / 2.6, 0.0, 1.0))
    u = w * 0.5 - (y_left_m / look) * (0.45 * w)
    return float(u), float(v)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
