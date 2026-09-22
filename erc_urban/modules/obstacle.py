"""
Fast RGB contact checks. There is no LiDAR on FrodoBots, and Depth Anything
often reads ~3 m at true contact against a wall, so this is the near-field
backstop: lamp hotspot with dark periphery means something is against the nose.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class ObstacleView:
    blocked: bool
    dark_frac: float
    edge_frac: float
    bright_frac: float
    reason: str = ""
    openness: float = 0.0


def _stats(rgb: np.ndarray) -> tuple[float, float, float, float, float, float]:
    gray = rgb.astype(np.float32).mean(axis=2)
    h, w = gray.shape
    y0, y1 = int(0.35 * h), int(0.95 * h)
    x0, x1 = int(0.18 * w), int(0.82 * w)
    roi = gray[y0:y1, x0:x1]
    mean_b = float(gray.mean())
    dark_thr = 35.0 if mean_b < 70.0 else 55.0
    bright_thr = max(55.0, mean_b * 0.9)
    dark_frac = float((roi < dark_thr).mean()) if roi.size else 1.0
    bright_frac = float((roi > bright_thr).mean()) if roi.size else 0.0
    gx = np.abs(np.diff(roi, axis=1, prepend=roi[:, :1])) if roi.size else 0.0
    gy = np.abs(np.diff(roi, axis=0, prepend=roi[:1, :])) if roi.size else 0.0
    edge_frac = float(((gx + gy) > 12.0).mean()) if roi.size else 0.0

    upper = gray[: max(1, int(0.40 * h)), :]
    side_thr = max(85.0, mean_b * 0.95)
    left = upper[:, : max(1, int(0.28 * w))]
    right = upper[:, int(0.72 * w) :]
    far = 0.5 * (
        float((left > side_thr).mean()) + float((right > side_thr).mean())
    )
    row_std = float(gray.mean(axis=1).std())

    cy0, cy1 = int(0.22 * h), int(0.72 * h)
    cx0, cx1 = int(0.28 * w), int(0.72 * w)
    center = gray[cy0:cy1, cx0:cx1]
    center_mean = float(center.mean()) if center.size else mean_b
    hot = float((center > max(110.0, mean_b + 25.0)).mean()) if center.size else 0.0
    corners = np.concatenate([
        gray[: int(0.18 * h), : int(0.18 * w)].ravel(),
        gray[: int(0.18 * h), int(0.82 * w) :].ravel(),
        gray[int(0.82 * h) :, : int(0.18 * w)].ravel(),
        gray[int(0.82 * h) :, int(0.82 * w) :].ravel(),
    ])
    corner_mean = float(corners.mean()) if corners.size else mean_b

    openness = 0.80 * min(1.0, far / 0.05) + 0.20 * min(1.0, row_std / 40.0)
    lamp_hot = hot > 0.08 and center_mean > corner_mean + 18.0
    return dark_frac, edge_frac, bright_frac, openness, far, float(lamp_hot)


def analyze_obstacle(front: np.ndarray) -> ObstacleView:
    if front is None or front.size == 0 or front.ndim != 3:
        return ObstacleView(True, 1.0, 0.0, 0.0, "no_frame", 0.0)

    dark_frac, edge_frac, bright_frac, openness, far, lamp_hot = _stats(front)

    # Lamp on near wall: bright center vs corners; allow mild side-texture noise.
    lamp_wall = bool(lamp_hot) and far < 0.09
    dark_body = dark_frac > 0.50 and bright_frac < 0.35 and far < 0.06
    buried = far < 0.02 and edge_frac > 0.12 and bright_frac < 0.12

    if lamp_wall:
        return ObstacleView(True, dark_frac, edge_frac, bright_frac, "lamp_wall", openness)
    if dark_body:
        return ObstacleView(True, dark_frac, edge_frac, bright_frac, "dark_blob", openness)
    if buried:
        return ObstacleView(True, dark_frac, edge_frac, bright_frac, "buried", openness)
    return ObstacleView(False, dark_frac, edge_frac, bright_frac, "clear", openness)


def front_blocked_rear_clear(
    front: np.ndarray | None, rear: np.ndarray | None
) -> ObstacleView | None:
    """Nose closed / rear open → reverse even if depth says ~3 m."""
    if front is None or rear is None:
        return None
    f = analyze_obstacle(front)
    r = analyze_obstacle(rear)
    _, _, _, f_open, f_far, _ = _stats(front)
    _, _, _, r_open, r_far, _ = _stats(rear)

    if r_open > 0.40 and f_open < 0.45 and (r_open - f_open) > 0.12:
        return ObstacleView(
            True, f.dark_frac, f.edge_frac, f.bright_frac,
            f"front_closed_rear_open(f={f_open:.2f},r={r_open:.2f})",
            f_open,
        )
    if r_far > f_far + 0.04 and f_open < 0.50 and f.blocked:
        return ObstacleView(
            True, f.dark_frac, f.edge_frac, f.bright_frac,
            f"front_closed_rear_open(f={f_open:.2f},r={r_open:.2f})",
            f_open,
        )
    if f.blocked:
        return f
    return None
