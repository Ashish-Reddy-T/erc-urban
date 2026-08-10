"""
Selector benchmark. `python -m erc_urban.eval.selector_bench --ep runs/ep1`

Answers the question the oracle gap raises but cannot settle: the gap says a
perfect selector could cut ADE by 56-71%, but is OUR selector -- Gemini looking
at a rendered overlay -- actually better than the planner's own ranking, or just
different from it?

Live testing showed Gemini disagreeing with the planner (picking +6, +53, +65
deg where the planner wanted 0 deg) and, worryingly, disagreeing with ITSELF
across identical queries. Disagreement is not improvement. This measures which.

Reports, per tick:
    planner   ADE of the candidate the planner ranks first
    vlm       ADE of the candidate the VLM picks
    fused     ADE after Probability Fusion (what actually drives)
    oracle    ADE of the best candidate available
    random    ADE of a uniformly random candidate -- the floor a "selector"
              must beat to be doing anything at all

The random baseline matters. A selector that merely perturbs the choice will
land between planner and random; only a real one beats the planner.

Costs one VLM call per evaluated tick (~$0.0004 each on Flash-Lite).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from ..calib import CameraCalibration
from ..interfaces import Observation
from .oracle_gap import ade, future_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", required=True)
    ap.add_argument("--k", type=int, default=18)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-ticks", type=int, default=25)
    ap.add_argument("--min-ref-m", type=float, default=0.8)
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT",
                                                        "erc-urban-nav-2610"))
    ap.add_argument("--location", default="global")
    ap.add_argument("--model", default="gemini-3.5-flash-lite")
    ap.add_argument("--no-vlm", action="store_true",
                    help="planner/oracle/random only, no API calls")
    a = ap.parse_args(argv)

    meta = json.load(open(os.path.join(a.ep, "episode.json")))
    ticks = meta["ticks"]
    frames = sorted(glob.glob(os.path.join(a.ep, "f*.jpg")))
    cal = CameraCalibration.load_or_default("erc_urban/camera_front.json", 1024, 576)

    from ..modules.candidates import MultiHypothesisPlanner
    from ..modules.logonav import LogoNavPlanner
    from ..modules.route import RouteAligner
    from ..modules.semantics import SemanticTraversability
    from ..modules.selector import GeminiVertexVLM, VLMSelector, render_overlay
    from ..modules.scoring import path_traversable_fractions

    sem = SemanticTraversability(device=a.device)
    route = RouteAligner()
    base = LogoNavPlanner(device=a.device)
    planner = MultiHypothesisPlanner(base=base)
    base.set_goal(-24.68990135, 25.87973595)

    vlm = None if a.no_vlm else GeminiVertexVLM(
        project=a.project, location=a.location, model=a.model)
    sel = VLMSelector(cal, vlm_call=None)      # fusion only; we drive it manually

    rng = np.random.default_rng(0)
    rows = []
    lats = []
    n_agree = 0
    for i in range(0, min(len(ticks), len(frames)) - a.horizon, a.stride):
        if len(rows) >= a.max_ticks:
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
            cands = planner.rescore(cands, obs, prior=None, trav_fracs=tf)
        if ref is None or len(cands) < 3:
            continue
        errs = np.array([ade(np.asarray(c.waypoints, float), ref) for c in cands])
        ok = np.isfinite(errs)
        if ok.sum() < 3:
            continue
        scores = np.array([c.score for c in cands])
        i_plan = int(np.argmax(np.where(ok, scores, -np.inf)))
        i_orc = int(np.argmin(np.where(ok, errs, np.inf)))
        i_rand = int(rng.choice(np.flatnonzero(ok)))

        i_vlm = i_plan
        if vlm is not None:
            t0 = time.monotonic()
            pick = vlm(render_overlay(frame, cands, cal), len(cands))
            lats.append(time.monotonic() - t0)
            if pick is not None and ok[pick]:
                i_vlm = pick
        if i_vlm == i_plan:
            n_agree += 1

        # Probability Fusion exactly as it runs live: fresh pick, dt = 0.
        sel._pick = {"waypoints": cands[i_vlm].waypoints.copy(),
                     "stamp": obs.stamp, "idx": i_vlm}
        i_fused = sel.select(obs, cands)

        rows.append((errs[i_plan], errs[i_vlm], errs[i_fused], errs[i_orc],
                     errs[i_rand]))

    if not rows:
        print("no evaluable ticks")
        return 1

    arr = np.array(rows)
    names = ["planner top-1", "VLM pick", "fused (live)", "oracle best", "random"]
    print(f"\n  selector benchmark: {len(rows)} ticks from {a.ep}")

    # --- reference-quality check -----------------------------------------
    # A reference recorded under a FIXED command is an arc the rover drove
    # because it was told to, not because the scene called for it. Scoring a
    # perception-driven planner against that penalises it for correctly saying
    # "go straight" on open pavement, and rewards any selector with a matching
    # turn bias. That confound can masquerade as a large win, so measure it
    # rather than leave it to the reader.
    yaws = np.array([t["dyaw"] for t in ticks if t["conf"] > 0])
    dists = np.array([np.hypot(t["dx"], t["dy"]) for t in ticks if t["conf"] > 0])
    if len(yaws) > 5 and dists.sum() > 0.1:
        turn_rate = np.degrees(yaws) / np.maximum(dists, 1e-6)   # deg per metre
        consistency = float(np.mean(np.sign(yaws) == np.sign(np.median(yaws))))
        print(f"  reference: {np.degrees(yaws.sum()):+.0f} deg net turn over "
              f"{dists.sum():.1f} m, {100*consistency:.0f}% of ticks turn the "
              f"same way")
        if consistency > 0.8 and abs(np.median(turn_rate)) > 15.0:
            print("  *** WARNING: the reference is a near-constant arc, which is\n"
                  "  *** what a FIXED command produces. Any selector biased toward\n"
                  "  *** that turn direction will score well here for the wrong\n"
                  "  *** reason. Treat the comparison below as indicative only;\n"
                  "  *** use scene-driven or teleoperated reference data to decide.")
    if lats:
        print(f"  VLM {a.model} @ {a.location}: median {np.median(lats):.2f} s, "
              f"agreed with planner on {100*n_agree/len(rows):.0f}% of ticks")
    print(f"\n  {'':18}{'median ADE':>12}{'mean':>10}{'vs planner':>13}")
    print("  " + "-" * 55)
    base_med = float(np.median(arr[:, 0]))
    for j, nm in enumerate(names):
        med = float(np.median(arr[:, j]))
        delta = 100.0 * (med - base_med) / max(base_med, 1e-9)
        tag = "" if j == 0 else f"{delta:+9.0f}%"
        print(f"  {nm:18}{med:11.3f}m{float(np.mean(arr[:, j])):9.3f}m{tag:>13}")
    print("  " + "-" * 55)

    vlm_med = float(np.median(arr[:, 1]))
    fused_med = float(np.median(arr[:, 2]))
    rand_med = float(np.median(arr[:, 4]))
    print(f"\n  VLM beats planner on {100*np.mean(arr[:,1] < arr[:,0]):.0f}% of ticks")
    print(f"  fused beats planner on {100*np.mean(arr[:,2] < arr[:,0]):.0f}% of ticks")
    # RANDOM is the baseline that matters, not the planner. Comparing a
    # selector against a planner that is itself worse than chance produces a
    # flattering percentage that means nothing -- measured here, the VLM looked
    # "29% better than the planner" while being only 4% better than random.
    vs_rand = 100.0 * (rand_med - vlm_med) / max(rand_med, 1e-9)
    n = len(arr)
    print(f"  VLM beats RANDOM on {100*np.mean(arr[:,1] < arr[:,4]):.0f}% of ticks "
          f"({vs_rand:+.0f}% median, n={n})")

    if base_med > rand_med:
        print(f"\n  *** THE PLANNER IS WORSE THAN RANDOM ({base_med:.3f} m vs "
              f"{rand_med:.3f} m).\n"
              "  *** Its scoring mis-orders the candidate set, so every "
              "'improvement\n"
              "  *** over the planner' below is measured against a broken "
              "baseline, and\n"
              "  *** Probability Fusion is actively mixing that bad ranking back "
              "in.\n"
              "  *** Fix the planner's scoring before tuning the selector.")

    if vs_rand < 10.0:
        print("\n  VERDICT: the VLM is not meaningfully better than picking at"
              "\n  random. It is not earning its latency or its API cost yet.\n")
    elif vlm_med < base_med:
        print(f"\n  VERDICT: the VLM beats random by {vs_rand:.0f}% and the "
              f"planner by {100*(base_med-vlm_med)/base_med:.0f}%.\n")
    else:
        print("\n  VERDICT: better than random but worse than the planner.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
