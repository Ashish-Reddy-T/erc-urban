"""
Per-term scoring diagnostic. `python -m erc_urban.eval.score_diag --ep runs/vn1`

"The ranking is worse than random" is a verdict, not a diagnosis. This asks
which TERM is responsible, correlating each against what we want: low ADE to
what a human drove.

Spearman per term, over all candidates at each tick:

    positive  the term ranks good candidates higher -- it is helping
    ~zero     the term carries no information about this choice
    negative  the term prefers candidates the human avoided, so its weight
              should drop or its sign is wrong

Rank correlation rather than Pearson because only the ORDER matters: the argmax
is what gets driven, not the score's magnitude.
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
from .oracle_gap import ade, future_path


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    # A constant term carries no ordering information, but argsort would rank it
    # by ARRAY POSITION and manufacture a correlation out of nothing. Caught in
    # practice: `prior` and `clearance` were both all-zero (not supplied) yet
    # reported an identical, confident-looking rho of 0.352.
    if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 1e-12 else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--k", type=int, default=18)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-ticks", type=int, default=40)
    ap.add_argument("--min-ref-m", type=float, default=1.0)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args(argv)

    meta = json.load(open(os.path.join(a.ep, "episode.json")))
    ticks = meta["ticks"]
    frames = sorted(glob.glob(os.path.join(a.ep, "f*.jpg")))
    cal = CameraCalibration.load_or_default("erc_urban/camera_front.json", 1024, 576)

    from ..modules.candidates import MultiHypothesisPlanner
    from ..modules.logonav import LogoNavPlanner
    from ..modules.route import RouteAligner
    from ..modules.semantics import SemanticTraversability
    from ..modules.scoring import CandidateScorer, path_traversable_fractions

    sem = SemanticTraversability(device=a.device)
    route = RouteAligner()
    base = LogoNavPlanner(device=a.device)
    planner = MultiHypothesisPlanner(base=base)
    base.set_goal(-24.68990135, 25.87973595)
    scorer = CandidateScorer()

    per_term = {k: [] for k in ("goal", "prior", "traversable", "clearance",
                                "smooth", "total")}
    n = 0
    for i in range(0, min(len(ticks), len(frames)) - a.horizon, a.stride):
        if n >= a.max_ticks:
            break
        ref = future_path(ticks, i, a.horizon)
        if ref is None:
            continue
        if float(np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1))) < a.min_ref_m:
            continue
        frame = np.asarray(Image.open(frames[i]).convert("RGB"))
        lab = sem.labels(frame, (144, 256)) if sem.ready else None
        gb, trav = 0.0, None
        if lab is not None:
            hd, sf, gd = sem.categorize(lab)
            cb, cw = sem.corridor(gd)
            _, tf_, tb = sem.track(gd)
            gb = route.align(0.0, cb, cw, tb, tf_, goal_trusted=False).bearing
            trav = gd & ~hd
        obs = Observation(front=frame, rear=None, lat=0.0, lon=0.0, yaw=0.0,
                          stamp=float(ticks[i]["t"]), goal_bearing=gb,
                          goal_range_m=26.0)
        cands = planner.propose(obs, a.k)
        if len(cands) < 4:
            continue
        errs = np.array([ade(np.asarray(c.waypoints, float), ref) for c in cands])
        ok = np.isfinite(errs)
        if ok.sum() < 4:
            continue

        tfrac = (path_traversable_fractions(cands, trav, cal)
                 if trav is not None else None)
        scorer.score(cands, gb, prior=None, trav_fracs=tfrac)
        terms = scorer.last
        good = -errs[ok]                       # higher = better
        for key in per_term:
            vals = np.array([getattr(t, key) for t in terms])[ok]
            per_term[key].append(_spearman(vals, good))
        n += 1

    if not n:
        print("no evaluable ticks")
        return 1

    print(f"\n  scoring diagnostic over {n} ticks from {a.ep}")
    print("  Spearman(term, -ADE): +1 term ranks perfectly, 0 no info,")
    print("  negative = the term prefers candidates the human avoided\n")
    print(f"  {'term':14}{'median rho':>12}{'mean':>9}{'% ticks helping':>18}")
    print("  " + "-" * 55)
    for key, vals in per_term.items():
        v = np.array([x for x in vals if x == x])
        if not len(v):
            continue
        helping = 100.0 * float(np.mean(v > 0))
        flag = "  <-- HARMFUL" if np.median(v) < -0.05 else (
            "  <-- no signal" if abs(np.median(v)) < 0.05 else "")
        print(f"  {key:14}{np.median(v):11.3f}{np.mean(v):9.3f}"
              f"{helping:16.0f}%{flag}")
    print("  " + "-" * 55 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
