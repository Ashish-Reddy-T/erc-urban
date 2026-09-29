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
MAX_YAW_DIFF = 15.0      # deg, corner-aligned laps only


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


def _turn_centres(lap, t0: float, t1: float, route) -> list:
    """Time of each map corner in the lap: the yaw-rate peak of a turn with
    the map turn's sign, within +-8 m of the map corner (by lap distance)."""
    d0 = lap.dist_at(t0)
    ts = np.arange(t0, t1, 0.25)
    yaw = np.array([lap.yaw_at(t) for t in ts])
    dist = np.array([lap.dist_at(t) - d0 for t in ts])
    rate = np.gradient(yaw, ts)
    k = 12                                          # +-3 s smoothing
    rate = np.convolve(rate, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    scale = dist[-1] / route.finish_s
    out = []
    for t in route.turns:
        m = np.abs(dist - t.s * scale) <= 8.0
        if not m.any():
            raise SystemExit(f"no lap data near the corner at map s={t.s:.1f}")
        r = np.where(m, rate * np.sign(t.angle_deg), -np.inf)
        i = int(np.argmax(r))
        out.append((float(ts[i]), float(dist[i]), t))
    return out


def build_by_corners(map_dir: str, lap_path: str, name: str) -> None:
    """Align a lap to the map by its corners, not by place recognition.

    Place recognition is what fails when a lap looks different from the map
    (the reason for adding the lap at all), but wheel distance between corners
    is good to a few percent. Anchors: start (0), each corner's turning peak,
    finish; map s is piecewise-linear in lap distance between them."""
    route = RouteMap(map_dir)
    lap = load_lap(lap_path)
    starts = lap.event_times("START_FINISH")
    if len(starts) < 2:
        raise SystemExit("the lap needs START_FINISH marks at both ends (key 0)")
    t0, t1 = starts[0], starts[-1]
    d0 = lap.dist_at(t0)
    corners = _turn_centres(lap, t0, t1, route)
    xs = [0.0] + [c[1] for c in corners] + [lap.dist_at(t1) - d0]
    ys = [0.0] + [c[2].s for c in corners] + [route.finish_s]
    for (tc, dc, tt) in corners:
        print(f"  corner {tt.angle_deg:+.0f} deg: lap {dc:.1f} m -> map {tt.s:.1f} m")
    if np.any(np.diff(xs) <= 0):
        raise SystemExit(f"corner anchors out of order: {xs}")

    idx = [i for i in range(len(lap)) if t0 <= lap.frame_t[i] <= t1]
    fs = np.array([np.interp(lap.dist_at(lap.frame_t[i]) - d0, xs, ys) for i in idx])
    ok = np.array([straight(lap, lap.frame_t[i]) for i in idx])
    in_turn = np.zeros(len(route), bool)
    for t in route.turns:
        in_turn[t.start:t.end + 1] = True
    # Heading check: each straight has its own yaw offset between the lap's
    # gyro and the map's; a frame facing more than MAX_YAW_DIFF off the
    # node's heading (e.g. already swinging toward the finish cone) would feed
    # a false bearing to the steering, so it is not used.
    lyaw = np.array([lap.yaw_at(lap.frame_t[i]) for i in idx])
    seg_edges = [0.0] + [c[2].s for c in corners] + [route.finish_s + 1]
    seg = np.searchsorted(seg_edges, fs, side="right") - 1
    twin = np.full(len(route), -1, int)
    cand = np.nonzero(ok)[0]
    node_seg = np.searchsorted(seg_edges, route.s, side="right") - 1
    # local offset: rolling median over +-8 m of the same straight (both gyros drift)
    dyaw = np.array([lyaw[j] - route.yaw[route.node_at_s(fs[j])] for j in cand])
    off_at = {}
    for q, j in enumerate(cand):
        m = (seg[cand] == seg[j]) & (np.abs(fs[cand] - fs[j]) <= 8.0)
        off_at[int(j)] = float(np.median(dyaw[m]))
    rejected = 0
    for n in range(len(route)):
        if in_turn[n]:
            continue
        j = cand[int(np.argmin(np.abs(fs[cand] - route.s[n])))]
        if abs(fs[j] - route.s[n]) > 0.5 or seg[j] != node_seg[n]:
            continue
        if abs(lyaw[j] - off_at[int(j)] - route.yaw[n]) > MAX_YAW_DIFF:
            rejected += 1
            continue
        twin[n] = int(j)
    print(f"  {rejected} nodes skipped: lap frame faced > {MAX_YAW_DIFF:.0f} deg off the map heading")
    used = sorted(set(twin[twin >= 0].tolist()))
    print(f"{len(idx)} lap frames, {ok.sum()} straight-driving, {len(used)} used")

    enc = PlaceEncoder()
    out = os.path.join(map_dir, f"twin_{name}")
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    desc = {}
    for k in used:
        img = lap.image(idx[k])
        Image.fromarray(img).resize((512, 288), Image.BILINEAR).save(
            os.path.join(out, "frames", f"{k:05d}.jpg"), quality=90)
        desc[k] = enc.encode_one(img)
    D = np.zeros((len(route), next(iter(desc.values())).shape[0]), np.float16)
    for n in range(len(route)):
        if twin[n] >= 0:
            D[n] = desc[twin[n]]
    np.save(os.path.join(out, "desc.npy"), D)
    with open(os.path.join(out, "twin.json"), "w") as fh:
        json.dump({"lap": lap_path, "align": "corners", "anchors_lap_m": xs, "anchors_map_s": ys,
                   "lap_frame": [int(i) for i in idx], "twin": twin.tolist()}, fh)
    print(f"coverage {(twin >= 0).mean():.0%} of map nodes -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=DEFAULT_OUT)
    ap.add_argument("--lap", default=DEFAULT_LAP)
    ap.add_argument("--name", default=None,
                    help="store as route_map/twin_<name>, aligned by corners (new laps)")
    a = ap.parse_args()
    if a.name:
        build_by_corners(a.map, a.lap, a.name)
    else:
        build(a.map, a.lap)


if __name__ == "__main__":
    main()
