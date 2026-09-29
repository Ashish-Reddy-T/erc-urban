"""
The route map: the daytime teach lap, indexed so the rover can follow it.

    python -m erc_indoor.route_map            # build from lap 1 (daytime)
    python -m erc_indoor.route_map --lap runs/indoor/lap_20260922_204027 \
        --out erc_indoor/data/route_map_night          # night map from lap 2

Each teach frame becomes a node with
  s     wheel distance from the start cone (m)
  yaw   gyro yaw relative to the start (deg, left-positive; short-horizon only)
  p     PROGRESS = s + TURN_M_PER_DEG * (degrees turned inside turn segments)
  desc  DINOv2 place descriptor
and a 512x288 JPEG for bearing matching.

Why `p` and not `s`: the teach rover turned IN PLACE at every corner (s stays
at 33.0 m for 30 frames while yaw goes +85 deg), so distance alone cannot say
where in a turn the rover is. Counting turned degrees as progress keeps p
strictly increasing through the corners.

Turn segments are detected from the teach yaw. Checkpoint s values were
measured by replaying lap 2 (cones out, marks pressed) against this map; they
label progress in the logs and are confirmed live by the cone detector —
navigation never depends on them.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LAP = os.path.join(HERE, "data", "indoor_track_recording.h5")
DEFAULT_OUT = os.path.join(HERE, "data", "route_map")

TURN_M_PER_DEG = 0.02          # a 90 deg corner counts as 1.8 m of progress
TURN_RATE_DPS = 1.5            # yaw rate that counts as "turning" (corner 3 is a slow ~2 deg/s arc)
TURN_MIN_DEG = 40.0            # smaller heading changes are corrections, not corners
TURN_MERGE_FRAMES = 15         # the teach turns came in bursts with pauses

# Cone positions in teach-lap metres, MEASURED by replaying lap 2 (recorded
# with cones out and a key pressed at each cone) through the localizer:
# marks landed at CP1 16.2, CP2 45.8, CP3 82.6, CP4 108.4 (yellow cone still
# ~2-3 m ahead in that frame), finish node 1356 s=131.7. Each mark was pressed
# with the cone still just ahead, so ~1.5 m is added. Log labels only — the
# cone detector confirms them live.
CHECKPOINT_S = {"CP1": 17.5, "CP2": 47.0, "CP3": 84.0, "CP4": 111.0}
FINISH_S = 131.8   # the teach lap ends at the finish cone


@dataclass
class Turn:
    start: int          # node index where the teach rover began turning
    end: int            # node index where it finished
    angle_deg: float    # left-positive
    s: float

    def to_json(self) -> dict:
        return {"start": self.start, "end": self.end, "angle_deg": self.angle_deg, "s": self.s}


class RouteMap:
    def __init__(self, root: str):
        self.root = root
        meta = json.load(open(os.path.join(root, "meta.json")))
        self.meta = meta
        self.s = np.array(meta["s"])
        self.yaw = np.array(meta["yaw"])
        self.p = np.array(meta["p"])
        self.turns = [Turn(**t) for t in meta["turns"]]
        self.checkpoints: dict[str, float] = meta["checkpoints"]
        self.finish_s: float = meta["finish_s"]
        self.desc = np.load(os.path.join(root, "desc.npy")).astype(np.float32)
        self._feat_cache: dict[int, object] = {}
        # Optional extra views (twin.py): more pictures of each node from
        # other laps. route_map/twin = the night lap; route_map/twin_<name> =
        # further laps (e.g. twin_today, recorded in competition-like light).
        # USE_TWIN=0 disables all; TWINS=night,today selects by name.
        self.twins: list[dict] = []
        self._twin_cache: dict[tuple, object] = {}
        want = os.environ.get("TWINS")
        want = set(want.split(",")) if want else None
        if os.environ.get("USE_TWIN", "1") != "0":
            for d in sorted(os.listdir(root)):
                if not (d == "twin" or d.startswith("twin_")):
                    continue
                tj = os.path.join(root, d, "twin.json")
                if not os.path.exists(tj):
                    continue
                name = "night" if d == "twin" else d[len("twin_"):]
                if want is not None and name not in want:
                    continue
                self.twins.append({
                    "name": name, "dir": d,
                    "idx": np.array(json.load(open(tj))["twin"]),
                    "desc": np.load(os.path.join(root, d, "desc.npy")).astype(np.float32),
                })
        # back-compat: the first extra view set
        self.twin = self.twins[0]["idx"] if self.twins else None
        self.twin_desc = self.twins[0]["desc"] if self.twins else None

    def __len__(self) -> int:
        return len(self.s)

    def image(self, i: int) -> np.ndarray:
        return np.asarray(Image.open(os.path.join(self.root, "frames", f"{int(i):05d}.jpg")).convert("RGB"))

    def features(self, i: int):
        from .bearing import features
        i = int(i)
        if i not in self._feat_cache:
            if len(self._feat_cache) > 400:
                self._feat_cache.clear()
            self._feat_cache[i] = features(self.image(i))
        return self._feat_cache[i]

    @property
    def n_twins(self) -> int:
        return len(self.twins)

    def twin_name(self, t: int = 0) -> str:
        return self.twins[t]["name"]

    def _twin_path(self, i: int, t: int):
        if t >= len(self.twins):
            return None
        k = int(self.twins[t]["idx"][int(i)])
        if k < 0:
            return None
        return os.path.join(self.root, self.twins[t]["dir"], "frames", f"{k:05d}.jpg")

    def twin_image(self, i: int, t: int = 0):
        path = self._twin_path(i, t)
        return None if path is None else np.asarray(Image.open(path).convert("RGB"))

    def twin_features(self, i: int, t: int = 0):
        """Extra-view features for node i from view set t, or None where it has none."""
        from .bearing import features
        path = self._twin_path(i, t)
        if path is None:
            return None
        if path not in self._twin_cache:
            if len(self._twin_cache) > 600:
                self._twin_cache.clear()
            self._twin_cache[path] = features(np.asarray(Image.open(path).convert("RGB")))
        return self._twin_cache[path]

    def node_at_p(self, p: float) -> int:
        return int(np.clip(np.searchsorted(self.p, p), 0, len(self.p) - 1))

    def node_at_s(self, s: float) -> int:
        return int(np.clip(np.searchsorted(self.s, s), 0, len(self.s) - 1))

    def next_turn(self, i: int):
        for k, t in enumerate(self.turns):
            if t.end >= i:
                return k, t
        return None, None

    def turn_containing(self, i: int):
        for k, t in enumerate(self.turns):
            if t.start <= i <= t.end:
                return k, t
        return None, None

    def checkpoint_label(self, s: float) -> str:
        passed = [k for k, v in sorted(self.checkpoints.items(), key=lambda kv: kv[1]) if s >= v]
        return passed[-1] if passed else "START"


def detect_turns(yaw: np.ndarray, frame_t: np.ndarray, s: np.ndarray) -> list[Turn]:
    n = len(yaw)
    k = 2
    rate = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        rate[i] = (yaw[b] - yaw[a]) / max(frame_t[b] - frame_t[a], 1e-3)
    turning = np.abs(rate) > TURN_RATE_DPS
    segs, i = [], 0
    while i < n:
        if not turning[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and (turning[j + 1] or turning[j + 1:j + 1 + TURN_MERGE_FRAMES].any()):
            j += 1
        segs.append((i, j))
        i = j + 1
    turns = []
    for a, b in segs:
        ang = float(yaw[b] - yaw[a])
        if abs(ang) >= TURN_MIN_DEG:
            turns.append(Turn(int(a), int(b), ang, float(s[a])))
    return turns


def progress(yaw: np.ndarray, s: np.ndarray, turns: list[Turn]) -> np.ndarray:
    turned = np.zeros(len(yaw))
    dy = np.abs(np.diff(yaw, prepend=yaw[0]))
    inside = np.zeros(len(yaw), bool)
    for t in turns:
        inside[t.start:t.end + 1] = True
    turned = np.cumsum(np.where(inside, dy, 0.0))
    p = s + TURN_M_PER_DEG * turned
    # strictly increasing so searchsorted is well defined
    return np.maximum.accumulate(p + np.arange(len(p)) * 1e-6)


MARK_EARLY_M = 1.5    # marks were pressed with the cone still just ahead


def _crop_and_label(lap):
    """Frame indices to keep, s origin, checkpoints and finish for this lap.

    record.py laps carry cone marks: the map starts at the first START_FINISH
    mark and ends at the last, and CP marks become checkpoints. The h5 teach
    lap has no marks; it began at the start cone, and its checkpoints were
    measured by replay (CHECKPOINT_S / FINISH_S).
    """
    starts = lap.event_times("START_FINISH")
    if len(starts) < 2:
        s = lap.frame_s
        return np.arange(len(lap)), float(s[0]), dict(CHECKPOINT_S), FINISH_S
    t0, t1 = starts[0], starts[-1]
    keep = np.nonzero((lap.frame_t >= t0) & (lap.frame_t <= t1 + 1.0))[0]
    s0 = float(lap.dist_at(t0))
    finish = float(lap.dist_at(t1)) - s0
    scale = finish / FINISH_S
    cps = {}
    for name, default in CHECKPOINT_S.items():
        ts = lap.event_times(name)
        cps[name] = round(float(lap.dist_at(ts[0])) - s0 + MARK_EARLY_M, 2) if ts \
            else round(default * scale, 2)
    return keep, s0, cps, round(finish, 2)


def build(lap_path: str, out: str, node_hz: float = 4.0) -> None:
    from .lap import load_lap
    from .place import PlaceEncoder

    lap = load_lap(lap_path)
    keep, s0, cps, finish_s = _crop_and_label(lap)
    # thin to ~node_hz: the 8.5 Hz record.py laps are mostly near-duplicates
    thin, last = [], -1e9
    for i in keep:
        if lap.frame_t[i] - last >= 1.0 / node_hz - 1e-3:
            thin.append(int(i))
            last = lap.frame_t[i]
    idx = np.array(thin)
    ft = lap.frame_t[idx]
    s = lap.dist_at(ft) - s0
    yaw = lap.yaw_at(ft)
    yaw = yaw - yaw[0]
    turns = detect_turns(yaw, ft, s)
    p = progress(yaw, s, turns)

    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    enc = PlaceEncoder()
    descs, batch = [], []
    for n, i in enumerate(idx):
        im = lap.image(i)
        Image.fromarray(im).resize((512, 288), Image.BILINEAR).save(
            os.path.join(out, "frames", f"{n:05d}.jpg"), quality=90)
        batch.append(im)
        if len(batch) == 32 or n == len(idx) - 1:
            descs.append(enc.encode(batch))
            batch = []
    np.save(os.path.join(out, "desc.npy"), np.concatenate(descs).astype(np.float16))
    meta = {
        "lap": os.path.relpath(lap_path, HERE),
        "s": s.round(3).tolist(), "yaw": yaw.round(2).tolist(), "p": p.round(4).tolist(),
        "turns": [t.to_json() for t in turns],
        "checkpoints": cps, "finish_s": finish_s,
        "turn_m_per_deg": TURN_M_PER_DEG,
    }
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    print(f"route map: {len(idx)} nodes, {s[-1]:.1f} m, p_end={p[-1]:.1f}")
    for t in turns:
        print(f"  turn nodes {t.start}-{t.end} at s={t.s:.1f} m: {t.angle_deg:+.0f} deg")
    print(f"  checkpoints {cps}  finish_s={finish_s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lap", default=DEFAULT_LAP)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--node-hz", type=float, default=4.0)
    a = ap.parse_args()
    build(a.lap, a.out, a.node_hz)


if __name__ == "__main__":
    main()
