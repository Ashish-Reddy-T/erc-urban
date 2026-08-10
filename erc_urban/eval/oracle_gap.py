"""
Oracle gap — the keystone validation. `python -m erc_urban.eval.oracle_gap --ep runs/ep1`

plan.md makes the K-candidate set the load-bearing requirement for M4/M5: if the
planner's own pick is already as good as the best candidate available, then a
selector has nothing to select and the whole VLM layer is decoration. Slow-Brain
report the gap they exploit as 1.64 m (planner top-1) vs 0.39 m (oracle best).
This measures the same quantity on our stack.

    top-1 ADE     = displacement error of the candidate the planner ranks first
    oracle ADE    = displacement error of the BEST candidate in the set
    gap           = top-1 - oracle = headroom a perfect selector could recover

REFERENCE TRAJECTORY, AND ITS HONEST LIMITATION
-----------------------------------------------
ADE needs a ground-truth future path. GPS on this rover is frozen, so the
reference is reconstructed from visual odometry: chain the per-tick SE(2)
deltas forward and express them in the frame of the tick being scored.

That reference is what the rover ACTUALLY DROVE, not what an expert would have
driven. Recorded under a fixed command it is an arc, not a demonstration. So
this measures two real things -- whether the candidate set COVERS the executed
trajectory, and how much selection headroom exists within it -- but it is not a
measure of expert imitation quality. Reading it as the latter would overclaim.
Swapping in FrodoBots episodes with `action_mbra` labels would fix that, and the
loader boundary here is where they would plug in.

Ticks whose VO confidence is zero are skipped: an unmeasured reference is worse
than no reference, because it silently rewards candidates that match nothing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

from ..calib import CameraCalibration
from ..interfaces import Observation


def future_path(ticks, i: int, horizon: int,
                max_dropout_frac: float = 0.15) -> np.ndarray | None:
    """
    Where the rover actually went over the next `horizon` ticks, expressed in
    the robot frame at tick i.

    Each stored delta is in the robot frame at the START of its step, so the
    chain has to be rotated as it accumulates -- adding them directly would
    make every turn corrupt the reference.
    """
    pts = []
    x = y = th = 0.0
    n_drop = 0
    span = min(i + horizon, len(ticks)) - i
    for j in range(i, min(i + horizon, len(ticks))):
        t = ticks[j]
        if t["conf"] <= 0.0:
            # Tolerate isolated dropouts as "no motion observed" rather than
            # discarding the window: on real teleop rides a strict rule threw
            # away most of the episode (13 usable windows out of 60). Too many
            # dropouts still means the reference is unmeasured, and an
            # unmeasured reference silently rewards candidates matching nothing.
            n_drop += 1
            if n_drop > max_dropout_frac * span:
                return None
            pts.append((x, y))
            continue
        c, s = np.cos(th), np.sin(th)
        x += c * t["dx"] - s * t["dy"]
        y += s * t["dx"] + c * t["dy"]
        th += t["dyaw"]
        pts.append((x, y))
    if len(pts) < 3:
        return None
    return np.asarray(pts, dtype=np.float64)


def _resample(path: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Sample a polyline at the given arc lengths."""
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return np.stack([np.interp(lengths, cum, path[:, 0]),
                     np.interp(lengths, cum, path[:, 1])], axis=1)


def ade(cand: np.ndarray, ref: np.ndarray, n_samples: int = 12) -> float:
    """
    Average displacement error over the COMMON ARC LENGTH of the two paths.

    Naively pairing waypoint i with reference point i is wrong here and the
    first run of this benchmark proved it: candidates run ~2.5 m while 8 ticks
    of reference covered 0.30 m, so index-wise ADE measured the length mismatch
    (a nonsensical 1.28 m median) rather than whether the shapes agree. The
    planner is predicting a path, not a speed, so the comparison has to be
    parameterised by distance travelled along each path, not by index.
    """
    if len(cand) < 2 or len(ref) < 2:
        return float("nan")
    lc = float(np.sum(np.linalg.norm(np.diff(cand, axis=0), axis=1)))
    lr = float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1)))
    common = min(lc, lr)
    if common < 0.05:
        return float("nan")
    s = np.linspace(0.0, common, n_samples)
    return float(np.mean(np.linalg.norm(_resample(cand, s) - _resample(ref, s), axis=1)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True, help="episode directory from eval.record")
    ap.add_argument("--k", type=int, default=18)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-ticks", type=int, default=60)
    ap.add_argument("--min-ref-m", type=float, default=0.8,
                    help="skip ticks whose reference path is shorter than this. "
                         "Teleop rides contain long stationary stretches, and "
                         "against a near-zero reference every short candidate "
                         "scores well, which flatters the planner and the "
                         "oracle equally and measures nothing.")
    a = ap.parse_args(argv)

    meta = json.load(open(os.path.join(a.ep, "episode.json")))
    ticks = meta["ticks"]
    frames = sorted(glob.glob(os.path.join(a.ep, "f*.jpg")))
    cal = CameraCalibration.load_or_default("erc_urban/camera_front.json", 1024, 576)

    from ..modules.candidates import MultiHypothesisPlanner
    from ..modules.logonav import LogoNavPlanner
    from ..modules.route import RouteAligner
    from ..modules.semantics import SemanticTraversability
    from ..modules.scoring import path_traversable_fractions

    sem = SemanticTraversability(device=a.device)
    route = RouteAligner()
    base = LogoNavPlanner(device=a.device)
    planner = MultiHypothesisPlanner(base=base)
    base.set_goal(-24.68990135, 25.87973595)

    rows = []
    n_eval = 0
    for i in range(0, min(len(ticks), len(frames)) - a.horizon, a.stride):
        if n_eval >= a.max_ticks:
            break
        ref = future_path(ticks, i, a.horizon)
        if ref is not None:
            arc = float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1)))
            if arc < a.min_ref_m:
                continue
        frame = np.asarray(Image.open(frames[i]).convert("RGB"))
        # Rank on the same bearing run.py would use. Hardcoding 0.0 here
        # made the planner rank against "straight ahead" while the rover
        # was arcing, which measured the harness, not the planner.
        gb = 0.0
        if sem.ready:
            lab = sem.labels(frame, (144, 256))
            if lab is not None:
                hd, sf, gd = sem.categorize(lab)
                cb, cw = sem.corridor(gd)
                _, tb_frac, tb = sem.track(gd)
                gb = route.align(0.0, cb, cw, tb, tb_frac,
                                 goal_trusted=False).bearing
        obs = Observation(front=frame, rear=None, lat=-24.690138, lon=25.879728,
                          yaw=0.0, stamp=float(ticks[i]["t"]),
                          goal_bearing=gb, goal_range_m=26.0)
        cands = planner.propose(obs, a.k)
        # Re-rank with perception. Candidates do not exist until propose() has
        # run, so traversability can only be applied as a second pass -- which
        # is exactly how run.py does it.
        if sem.ready and lab is not None and cands:
            trav = (gd & ~hd)
            tf = path_traversable_fractions(cands, trav, cal)
            cands = planner.rescore(cands, obs, prior=None, trav_fracs=tf)      # advances temporal context
        if ref is None or len(cands) < 2:
            continue
        errs = np.array([ade(np.asarray(c.waypoints, float), ref) for c in cands])
        ok = np.isfinite(errs)
        if not ok.any():
            continue
        scores = np.array([c.score for c in cands])
        top1 = int(np.argmax(np.where(ok, scores, -np.inf)))
        best = int(np.argmin(np.where(ok, errs, np.inf)))
        arc_len = float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1)))
        rows.append((errs[top1], errs[best], int(ok.sum()), arc_len))
        n_eval += 1

    if not rows:
        print("no evaluable ticks (VO confidence too low, or no candidates)")
        return 1

    arr = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
    top1, oracle, nk, reach = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    gap = top1 - oracle
    win = float(np.mean(oracle < top1 - 1e-6))

    print(f"\n  oracle gap over {len(rows)} ticks from {a.ep}")
    print(f"  horizon {a.horizon} ticks (~{np.median(reach):.2f} m of reference ARC), "
          f"K median {int(np.median(nk))}\n")
    print(f"  {'':22}{'median':>9}{'mean':>9}{'p90':>9}")
    print("  " + "-" * 49)
    print(f"  {'planner top-1 ADE':22}{np.median(top1):8.3f}m{np.mean(top1):8.3f}m"
          f"{np.percentile(top1, 90):8.3f}m")
    print(f"  {'oracle-best ADE':22}{np.median(oracle):8.3f}m{np.mean(oracle):8.3f}m"
          f"{np.percentile(oracle, 90):8.3f}m")
    print(f"  {'GAP (headroom)':22}{np.median(gap):8.3f}m{np.mean(gap):8.3f}m"
          f"{np.percentile(gap, 90):8.3f}m")
    print("  " + "-" * 49)
    rel = 100.0 * np.median(gap) / max(np.median(top1), 1e-9)
    print(f"  a perfect selector would cut median ADE by {rel:.0f}%")
    print(f"  oracle strictly beats top-1 on {100*win:.0f}% of ticks")
    print(f"  reference: Slow-Brain report 1.64 m -> 0.39 m (76% reduction)\n")

    if rel < 10.0:
        print("  VERDICT: little headroom. The planner is already picking near the")
        print("  best available candidate, so a VLM selector has little to add and")
        print("  M4/M5 should not be trusted to improve things on this data.\n")
    else:
        print(f"  VERDICT: real headroom ({rel:.0f}%). A selector that picks better")
        print("  than the planner's own ranking has room to help.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
