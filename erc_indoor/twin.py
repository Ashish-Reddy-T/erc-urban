"""
Night "twin" views for the daytime route map.

    python -m erc_indoor.twin                      # lap 2 (night) -> route_map/twin

The route map is the daytime lap. At night the glass atrium looks nothing like
it (windows black instead of bright): SIFT finds 0-3 matches there. Lap 2 was
driven at night along the same route, so each map node gets the lap-2 frame
taken at the same place. The map keeps defining progress, corners and finish;
the twin is only a second picture of each place, used for bearing and place
recognition whenever the daytime picture fails to match.

Only lap-2 frames taken while driving STRAIGHT are eligible (no spin
commanded in the preceding ~2 s, allowing for the ~1.2 s command lag): a
frame grabbed mid-correction faces the wrong way and would steer the rover
off course. Frames are placed on the map by running lap 2 through the same
localizer the live loop uses.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image

from .lap import load_lap
from .localizer import RouteLocalizer
from .place import PlaceEncoder
from .replay import odom_progress
from .route_map import DEFAULT_OUT, RouteMap

DEFAULT_LAP = "runs/indoor/lap_20260922_204027"
MAX_NODE_GAP = 20         # teach nodes (~0.1 m each): far features still give heading at 2 m
CMD_LAG_S = 1.2


def straight(lap, t: float) -> bool:
    m = (lap.ctrl_t >= t - CMD_LAG_S - 1.0) & (lap.ctrl_t <= t - CMD_LAG_S + 0.3)
    if m.sum() < 3:
        return False
    return bool(np.all(np.abs(lap.ctrl_ang[m]) < 1e-6) and np.mean(lap.ctrl_lin[m]) > 0.3)


def build(map_dir: str, lap_path: str) -> None:
    route = RouteMap(map_dir)
    lap = load_lap(lap_path)
    starts = lap.event_times("START_FINISH")
    t0, t1 = (starts[0], starts[-1] + 1.0) if len(starts) >= 2 else (lap.frame_t[0], lap.frame_t[-1])
    idx = [i for i in range(len(lap)) if t0 <= lap.frame_t[i] <= t1][::2]

    enc = PlaceEncoder()
    loc = RouteLocalizer(route)
    loc.reset(0.0, 2.0)
    aligned, descs, ok = [], [], []
    t_prev = lap.frame_t[idx[0]]
    for i in idx:
        t = lap.frame_t[i]
        loc.predict(odom_progress(lap, t_prev, t))
        t_prev = t
        d = enc.encode_one(lap.image(i))
        loc.update(d)
        aligned.append(loc.estimate().node)
        descs.append(d)
        ok.append(straight(lap, t))
    aligned, ok = np.array(aligned), np.array(ok)
    print(f"{len(idx)} lap frames, {ok.sum()} straight-driving")

    twin = np.full(len(route), -1, int)
    cand = np.nonzero(ok)[0]
    for n in range(len(route)):
        gap = np.abs(aligned[cand] - n)
        k = int(gap.argmin())
        if gap[k] <= MAX_NODE_GAP:
            twin[n] = int(cand[k])       # index into idx / descs
    used = sorted(set(twin[twin >= 0].tolist()))
    out = os.path.join(map_dir, "twin")
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    for k in used:
        Image.fromarray(lap.image(idx[k])).resize((512, 288), Image.BILINEAR).save(
            os.path.join(out, "frames", f"{k:05d}.jpg"), quality=90)
    D = np.zeros((len(route), descs[0].shape[0]), np.float16)
    for n in range(len(route)):
        if twin[n] >= 0:
            D[n] = descs[twin[n]]
    np.save(os.path.join(out, "desc.npy"), D)
    with open(os.path.join(out, "twin.json"), "w") as fh:
        json.dump({"lap": lap_path, "lap_frame": [int(idx[k]) for k in range(len(idx))],
                   "twin": twin.tolist()}, fh)
    cov = (twin >= 0).mean()
    print(f"twin coverage {cov:.0%} of map nodes, {len(used)} distinct night frames -> {out}")
    for a, b, name in [(0, 427, "start->piano (atrium)"), (459, 730, "CP2 corridor"),
                       (784, 998, "CP3 corridor"), (1054, 1275, "CP4 corridor"),
                       (1320, len(route), "final atrium")]:
        print(f"  {name:22s} coverage {(twin[a:b] >= 0).mean():.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=DEFAULT_OUT)
    ap.add_argument("--lap", default=DEFAULT_LAP)
    a = ap.parse_args()
    build(a.map, a.lap)


if __name__ == "__main__":
    main()
