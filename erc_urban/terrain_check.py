"""
Which FAN_TERRAIN for this site? One frame, both modes, side by side.

    python -m erc_urban.terrain_check                 # live frame from the SDK
    python -m erc_urban.terrain_check a.jpg b.jpg     # saved frames

For each frame it prints the top SegFormer classes in the ground half of the
image and, per mode, the clear distance straight ahead and the widest clear
heading, then writes runs/terrain_check/<stamp>_<name>.jpg:
left = urban (paved allow-list, grass only as a fallback), right = offroad
(everything but real obstacles is ground). Green = ground, orange = where
ground ends, bars = clear distance per heading (5 m max).

Rule of thumb: if urban shows the drivable surface green and stops at real
obstacles, use urban (it avoids grass verges and low stuff). If urban paints
the whole course red/short (dirt, leaf litter, grass paths, SegFormer calling
the ground "plant"), use FAN_TERRAIN=offroad.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.request

import cv2
import numpy as np

import erc_urban.run as R
from erc_urban.modules.freespace import FreeSpaceUrban


def _live_frame(url: str) -> np.ndarray:
    d = json.load(urllib.request.urlopen(url + "/v2/front", timeout=30))
    bgr = cv2.imdecode(np.frombuffer(base64.b64decode(d["front_frame"]), np.uint8), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _panel(rgb_small, ground, fs, title):
    img = rgb_small[:, :, ::-1].copy()
    img[ground] = (0.55 * img[ground] + [0, 110, 0]).astype(np.uint8)
    b = fs.drive_boundary
    for u in range(img.shape[1]):
        cv2.circle(img, (u, int(b[u])), 1, (0, 165, 255), -1)
    h, w = img.shape[:2]
    bins, clear = fs.drive.bins_deg, fs.drive.clear_m
    for a, c in zip(bins, clear):
        u = int(w / 2 - np.tan(np.radians(a)) * R.CALIB.scaled_to(w, h).fx)
        if 0 <= u < w:
            col = (0, 200, 0) if c >= R.FAN_NEED_M else (0, 200, 255) if c >= R.FAN_MIN_M else (0, 0, 230)
            cv2.line(img, (u, h - 2), (u, h - 2 - int(12 * c)), col, 3)
    cv2.rectangle(img, (0, 0), (w, 18), (0, 0, 0), -1)
    cv2.putText(img, title, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    return img


def check(rgb: np.ndarray, sem, name: str, out_dir: str) -> None:
    rgb = cv2.resize(rgb, (1024, 576))
    lab = sem.labels(rgb, R.FAN_SEG_HW)
    fsu = FreeSpaceUrban(R.CALIB, seg_hw=R.FAN_SEG_HW)
    id2 = sem.model.config.id2label
    below = lab[int(fsu.horizon) + 2:]
    ids, cnt = np.unique(below, return_counts=True)
    top = sorted(zip(cnt, ids), reverse=True)[:6]
    print(f"\n{name}: ground-half classes: " +
          ", ".join(f"{id2[int(i)]} {100 * c / below.size:.0f}%" for c, i in top))
    small = cv2.resize(rgb, (R.FAN_SEG_HW[1], R.FAN_SEG_HW[0]))
    panels = []
    for mode in ("urban", "offroad"):
        R.FAN_TERRAIN = mode
        fsu.min_obstacle_h_m = 0.12 if mode == "offroad" else 0.07
        g, gr = R._fan_masks(sem, lab)
        fs = fsu.estimate(g, gr if mode == "urban" else None)
        c = fs.drive.clear_m
        best = int(np.argmin(np.abs(fs.drive.bins_deg) + 1000 * (c < c.max() - 0.05)))
        line = (f"{mode:7s} ahead {fs.drive.ahead_m:.1f} m, widest {fs.drive.bins_deg[best]:+.0f} deg "
                f"{fs.drive.clear_m[best]:.1f} m, ground {100 * (g | gr).mean():.0f}% of frame")
        print("  " + line)
        panels.append(_panel(small, g | gr, fs, line))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, time.strftime("%H%M%S_") + os.path.splitext(name)[0] + ".jpg")
    cv2.imwrite(path, np.hstack(panels))
    print("  ->", path)


def main(argv: list[str]) -> int:
    sem = R._semantics()
    if sem is None:
        print("SegFormer unavailable")
        return 1
    out = os.path.join("runs", "terrain_check")
    if argv:
        for p in argv:
            check(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), sem, os.path.basename(p), out)
    else:
        check(_live_frame(os.environ.get("SDK_URL", "http://localhost:8000")), sem, "live", out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
