"""
Views for the API driver (erc_urban/agent.py): the annotated front frame and
the north-up map the model looks at. The model picks WHICH WAY to go; the fan
planner still decides HOW (clearance, escapes, tilt guard, claims).

What a person driving by the camera does, every few metres: look at the view,
look at the map, know where the checkpoint is, pick the gap / path / sidewalk
that leads there. The model gets the same three things:

  1. the front frame with numbered candidate headings drawn on it (colour =
     the fan's measured clearance along that heading),
  2. a small north-up map: OSM paths, checkpoints, the rover and its trail,
  3. text: distance and bearing to the checkpoint, recent trouble.

The model backends and the action loop live in erc_urban/driver_llm.py and
erc_urban/agent.py.
"""
from __future__ import annotations

import base64
import math
from typing import Optional

import cv2
import numpy as np

CANDIDATES_DEG = (60, 40, 20, 0, -20, -40, -60)   # left-positive; numbered 1..7

def _jpeg_b64(img_bgr: np.ndarray, q: int = 80) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def annotate(front_rgb: np.ndarray, cal, clear_at, goal_rel_deg: Optional[float],
             width: int = 768) -> np.ndarray:
    """Front frame (BGR out) with numbered candidate arrows. clear_at(deg) -> m."""
    h0, w0 = front_rgb.shape[:2]
    img = cv2.resize(front_rgb, (width, int(width * h0 / w0)))[:, :, ::-1].copy()
    h, w = img.shape[:2]
    c = cal.scaled_to(w, h)
    for k, a in enumerate(CANDIDATES_DEG, start=1):
        r = math.radians(a)
        pts = np.array([[d * math.cos(r), d * math.sin(r)] for d in (0.6, 2.5)])
        uv, ok = c.project_ground(pts)
        if not ok.all():
            u = w / 2 - math.tan(r) * c.fx
            uv = np.array([[u, h - 5], [u, h * 0.62]])
        clr = float(clear_at(a))
        col = (0, 200, 0) if clr >= 1.2 else (0, 200, 255) if clr >= 0.6 else (0, 0, 230)
        p0 = tuple(int(v) for v in uv[0]); p1 = tuple(int(v) for v in uv[1])
        cv2.arrowedLine(img, p0, p1, col, 3, tipLength=0.15)
        cv2.circle(img, p1, 13, (0, 0, 0), -1)
        cv2.putText(img, str(k), (p1[0] - 7, p1[1] + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    if goal_rel_deg is not None and abs(goal_rel_deg) < 55:
        u = int(w / 2 - math.tan(math.radians(goal_rel_deg)) * c.fx)
        cv2.line(img, (u, 0), (u, int(h * 0.45)), (255, 0, 255), 2)
        cv2.putText(img, "CHECKPOINT", (max(0, u - 55), 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 0, 255), 2)
    return img


def render_map(here, yaw_cw_deg: float, goals, goal_idx: int, trail, payload,
               span_m: float = 80.0, size: int = 360) -> np.ndarray:
    """North-up map centred on the rover (BGR)."""
    img = np.full((size, size, 3), 245, np.uint8)
    lat0, lon0 = here
    kx = 111320.0 * math.cos(math.radians(lat0))
    s = size / span_m

    def px(ll):
        e = (ll[1] - lon0) * kx
        n = (ll[0] - lat0) * 110540.0
        return int(size / 2 + e * s), int(size / 2 - n * s)

    if payload:
        nodes = {int(e["id"]): (e["lat"], e["lon"]) for e in payload.get("elements", [])
                 if e.get("type") == "node"}
        for el in payload.get("elements", []):
            if el.get("type") == "way" and (el.get("tags") or {}).get("highway"):
                pts = [px(nodes[n]) for n in el.get("nodes", []) if n in nodes]
                if len(pts) > 1:
                    cv2.polylines(img, [np.array(pts, np.int32)], False, (190, 190, 190), 7)
    if len(trail) > 1:
        cv2.polylines(img, [np.array([px(p) for p in trail], np.int32)], False, (220, 120, 40), 2)
    for i, g in enumerate(goals):
        p = px(g)
        cv2.circle(img, p, 9 if i == goal_idx else 6, (0, 150, 0) if i == goal_idx else (120, 190, 120), -1)
        cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
    c = (size // 2, size // 2)
    r = math.radians(yaw_cw_deg)
    tip = (int(c[0] + 16 * math.sin(r)), int(c[1] - 16 * math.cos(r)))
    cv2.arrowedLine(img, c, tip, (0, 120, 255), 4, tipLength=0.5)
    cv2.putText(img, "N", (size - 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, f"{span_m:.0f} m across", (6, size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    return img


