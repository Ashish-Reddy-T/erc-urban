"""
Per-module tick cost. `python -m erc_urban.eval.tick_budget`

plan.md calls this non-negotiable: a module that blows the budget is a module
that does not ship. It matters more on this rover than the plan assumed,
because of a measured hardware fact -- the Mini+ has a command watchdog, so a
single /control POST produces +0.7 deg of yaw while the same command streamed
at 5 Hz produces +18.6 deg. Command RATE is effective speed. A slow tick does
not merely delay decisions, it physically slows the robot down.

The SDK itself is not the constraint: /control 4 ms, /v2/screenshot 21 ms,
/data 3 ms, so ~28 ms (35 Hz) is achievable before any inference. Everything
below that line is ours.

Runs on recorded frames by default so it is reproducible and needs no rover.
"""
from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time

import numpy as np
from PIL import Image

from ..calib import CameraCalibration
from ..interfaces import Observation


def _timeit(fn, n: int, warmup: int = 1):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000.0, min(ts) * 1000.0, max(ts) * 1000.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="test/fixtures/*.png")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--hz", type=float, default=5.0)
    a = ap.parse_args(argv)

    files = sorted(glob.glob(a.frames))
    if not files:
        print(f"no frames matched {a.frames}")
        return 1
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in files[:6]]
    frame = frames[0]
    cal = CameraCalibration.load_or_default(
        "erc_urban/camera_front.json", frame.shape[1], frame.shape[0]
    )
    budget_ms = 1000.0 / a.hz
    print(f"\n  tick budget @ {a.hz:g} Hz = {budget_ms:.0f} ms   "
          f"frames {frame.shape[1]}x{frame.shape[0]}   device={a.device}\n")

    obs = Observation(front=frame, rear=None, lat=-24.690138, lon=25.879728,
                      yaw=0.0, stamp=time.monotonic(),
                      goal_bearing=0.0, goal_range_m=26.0)
    rows = []

    # --- M1: planner -----------------------------------------------------
    try:
        from ..modules.logonav import LogoNavPlanner
        from ..modules.candidates import MultiHypothesisPlanner

        base = LogoNavPlanner(device=a.device)
        if base.ready:
            base.set_goal(-24.68990135, 25.87973595)
            planner = MultiHypothesisPlanner(base=base)
            for f in frames:
                planner.propose(
                    Observation(front=f, rear=None, lat=obs.lat, lon=obs.lon,
                                yaw=0.0, stamp=time.monotonic(),
                                goal_bearing=0.0, goal_range_m=26.0), 18)
            cands = planner.propose(obs, 18)
            rows.append(("M1 planner (K candidates)",
                         *_timeit(lambda: planner.propose(obs, 18), a.repeats)))
        else:
            cands = []
            rows.append(("M1 planner", float("nan"), float("nan"), float("nan")))
    except Exception as e:
        cands = []
        print(f"  M1 unavailable: {e}")

    # --- M6a: depth ------------------------------------------------------
    try:
        from ..modules.depth_gate import DepthObstacleGate

        dg = DepthObstacleGate(device=a.device, every_n=1)
        if dg.ready:
            rows.append(("M6 depth (Depth Anything)",
                         *_timeit(lambda: dg.evaluate(frame, 0.0), a.repeats)))
    except Exception as e:
        print(f"  depth unavailable: {e}")

    # --- M6b: semantics --------------------------------------------------
    sem = None
    try:
        from ..modules.semantics import SemanticTraversability

        sem = SemanticTraversability(device=a.device)
        if sem.ready:
            rows.append(("M6 semantics (SegFormer)",
                         *_timeit(lambda: sem.labels(frame, (144, 256)), a.repeats)))
    except Exception as e:
        print(f"  semantics unavailable: {e}")

    # --- M6c: costmap ----------------------------------------------------
    try:
        from ..modules.costmap import DepthCostmap, CostmapConfig

        depth = np.clip(np.abs(np.random.default_rng(0).normal(2.0, 0.4, (144, 256))),
                        0.2, 3.5)
        cm = DepthCostmap(CostmapConfig(fov_horizontal_deg=cal.fov_horizontal_deg,
                                        max_range_m=3.5))
        rows.append(("M6 costmap (RANSAC+grid)",
                     *_timeit(lambda: cm.compute(depth), a.repeats)))
        if cands:
            res = cm.compute(depth)
            rows.append(("M6 costmap veto",
                         *_timeit(lambda: cm.veto(res, cands), a.repeats)))
    except Exception as e:
        print(f"  costmap unavailable: {e}")

    # --- M6d: traversability veto ---------------------------------------
    if sem is not None and sem.ready and cands:
        from ..modules.traversability import TraversabilityVeto, SegFormerSource

        tv = TraversabilityVeto(source=SegFormerSource(semantics=sem), calib=cal)
        rows.append(("M6 semantic veto (incl. seg)",
                     *_timeit(lambda: tv.mask(obs, cands), a.repeats)))

    # --- M5: visual odometry ---------------------------------------------
    try:
        from ..modules.odometry import VisualOdometry

        vo = VisualOdometry(cal)
        vo.update(frames[0])
        i = {"n": 0}

        def _vo():
            i["n"] = (i["n"] + 1) % len(frames)
            vo.update(frames[i["n"]])

        rows.append(("M5 visual odometry", *_timeit(_vo, a.repeats * 2)))
    except Exception as e:
        print(f"  odometry unavailable: {e}")

    # --- M4: overlay render (the only in-tick part of the VLM path) ------
    if cands:
        from ..modules.selector import render_overlay

        rows.append(("M4 overlay render",
                     *_timeit(lambda: render_overlay(frame, cands, cal), a.repeats)))

    # --- M2/M7: cheap bits ------------------------------------------------
    from ..modules.route import RouteAligner
    from ..modules.baseline import PurePursuitController

    r = RouteAligner()
    rows.append(("M2 route align",
                 *_timeit(lambda: r.align(0.3, 0.1, 0.8, 0.1, 0.9), 200)))
    if cands:
        pp = PurePursuitController()
        rows.append(("M7 pure pursuit",
                     *_timeit(lambda: pp.command(cands[0].waypoints), 200)))

    print(f"  {'module':<32} {'median':>9} {'min':>8} {'max':>8}   {'% budget':>9}")
    print("  " + "-" * 74)
    total = 0.0
    for name, med, lo, hi in rows:
        if med == med:                      # not NaN
            total += med
        pct = 100.0 * med / budget_ms
        flag = "  <-- OVER" if pct > 100 else ""
        print(f"  {name:<32} {med:8.1f}ms {lo:7.1f} {hi:7.1f}   {pct:8.1f}%{flag}")
    print("  " + "-" * 74)
    sdk_ms = 28.0
    print(f"  {'SDK I/O (measured)':<32} {sdk_ms:8.1f}ms")
    grand = total + sdk_ms
    print(f"  {'TOTAL':<32} {grand:8.1f}ms {'':16}   {100*grand/budget_ms:8.1f}%")
    print(f"\n  achievable rate: {1000.0/grand:.2f} Hz "
          f"({'fits' if grand <= budget_ms else 'MISSES'} the {a.hz:g} Hz budget)")
    if grand > budget_ms:
        worst = max((r for r in rows if r[1] == r[1]), key=lambda r: r[1])
        print(f"  dominant cost: {worst[0]} at {worst[1]:.0f} ms "
              f"({100*worst[1]/grand:.0f}% of the tick)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
