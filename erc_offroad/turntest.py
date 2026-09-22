"""
Why won't it turn? Measure, don't guess.

    python -m erc_offroad.turntest

THREE HYPOTHESES, RANKED BY HOW LIKELY THEY ARE TO BE THE ANSWER
----------------------------------------------------------------
1. NOT ENOUGH AUTHORITY. `/control` in the SDK does no clamping at all -- it
   forwards the command untouched -- so angular is valid all the way to 1.0.
   Every command issued tonight used 0.14 to 0.35. That is a THIRD of the
   available authority, on the highest-friction surface this rover will ever
   meet. This is the most likely answer and the cheapest to test.

2. PURE SPIN IS THE WORST CASE FOR SKID-STEER. A 4-wheel skid-steer rotates by
   scrubbing its wheels sideways across the ground. In-place rotation is the
   maximum-scrub, maximum-torque manoeuvre in its repertoire, and rock is the
   worst surface for it. An ARC -- linear and angular together -- turns far more
   easily, because wheels already rolling have broken static friction and only
   have to overcome the much smaller kinetic term. Operator reports hearing the
   motors strain, which is exactly what a stalled scrub sounds like.

3. COMMAND GAPS. Commands stream at 5 Hz. If the rover's watchdog decays drive
   between messages, effective torque is lower than commanded. Raising the rate
   costs nothing to try.

Each strategy below gets the same treatment: settle, read heading, drive for a
fixed time, settle, read heading. Output is degrees achieved and deg/s, so the
strategies are directly comparable. Tilt guarding stays on throughout.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

from .creep import (
    ABORT_ABS_DEG, ABORT_RATE_DEG, _get, _post, _ref_attitude, _wrap180,
    attitude,
)


def settled_heading(timeout_s: float = 12.0) -> float | None:
    """Wait for the wheels to stop, then take a median heading."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        a = attitude(_get("/data"))
        if a is not None and not a.moving:
            break
        time.sleep(0.4)
    vals = []
    for _ in range(5):
        o = _get("/data").get("orientation")
        if o is not None:
            vals.append(float(o))
        time.sleep(0.25)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def drive(lin: float, ang: float, seconds: float, hz: float) -> tuple[bool, float]:
    """Stream a command. Returns (completed_without_abort, peak_abs_tilt)."""
    ref = _ref_attitude(n=2)
    n = max(1, int(seconds * hz))
    dt = 1.0 / hz
    peak = 0.0
    prev_r = prev_p = None
    for _ in range(n):
        _post("/control", {"command": {"linear": lin, "angular": ang}})
        try:
            a = attitude(_get("/data", timeout=8))
        except Exception:
            time.sleep(dt); continue
        if a is None:
            time.sleep(dt); continue
        peak = max(peak, a.tilt)
        rate = 0.0
        if prev_r is not None:
            rate = max(abs(a.roll - prev_r), abs(a.pitch - prev_p))
        prev_r, prev_p = a.roll, a.pitch
        if a.tilt > ABORT_ABS_DEG or rate > ABORT_RATE_DEG:
            _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
            return False, peak
        time.sleep(dt)
    _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
    return True, peak


def run(name: str, lin: float, ang: float, secs: float, hz: float,
        wiggle: bool = False) -> dict:
    h0 = settled_heading()
    if h0 is None:
        return {"name": name, "error": "no heading"}
    ok = True
    peak = 0.0
    if wiggle:
        # Alternate forward/back while turning. Each reversal breaks static
        # friction, which is the whole point -- a stalled scrub never gets the
        # chance to start rolling.
        for i in range(4):
            o, p = drive(0.22 if i % 2 == 0 else -0.22, ang, secs / 4.0, hz)
            ok = ok and o
            peak = max(peak, p)
            if not o:
                break
    else:
        ok, peak = drive(lin, ang, secs, hz)
    time.sleep(1.0)
    h1 = settled_heading()
    if h1 is None:
        return {"name": name, "error": "no heading after"}
    d = _wrap180(h1 - h0)
    return {"name": name, "lin": lin, "ang": ang, "secs": secs, "hz": hz,
            "deg": d, "dps": d / secs if secs else 0.0,
            "aborted": not ok, "peak_tilt": peak}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="erc_offroad.turntest")
    ap.add_argument("--secs", type=float, default=2.0)
    ap.add_argument("--out", default="runs/offroad/turntest.jsonl")
    a = ap.parse_args(argv)

    plan = [
        # name                       lin    ang   hz    wiggle
        ("baseline spin 0.35 @5Hz",  0.0,   0.35,  5.0, False),
        ("spin 0.60 @5Hz",           0.0,   0.60,  5.0, False),
        ("spin 0.90 @5Hz",           0.0,   0.90,  5.0, False),
        ("spin 0.90 @15Hz",          0.0,   0.90, 15.0, False),
        ("ARC fwd 0.25 + 0.60",      0.25,  0.60,  5.0, False),
        ("ARC rev -0.25 + 0.60",    -0.25,  0.60,  5.0, False),
        ("WIGGLE +/-0.22 + 0.90",    0.0,   0.90,  5.0, True),
    ]

    print(f"{'strategy':<26} {'deg':>7} {'deg/s':>7} {'tilt':>6}  note")
    print("-" * 62)
    results = []
    with open(a.out, "a") as f:
        for name, lin, ang, hz, wig in plan:
            r = run(name, lin, ang, a.secs, hz, wig)
            r["t"] = time.time()
            results.append(r)
            f.write(json.dumps(r) + "\n"); f.flush()
            if "error" in r:
                print(f"{name:<26} {'--':>7} {'--':>7} {'--':>6}  {r['error']}")
                continue
            note = "ABORTED" if r["aborted"] else ""
            print(f"{name:<26} {r['deg']:+7.0f} {r['dps']:+7.1f} "
                  f"{r['peak_tilt']:6.1f}  {note}")
            time.sleep(1.5)

    good = [r for r in results if "error" not in r and abs(r.get("deg", 0)) >= 8]
    print()
    if good:
        best = max(good, key=lambda r: abs(r["deg"]))
        print(f"BEST: {best['name']} -> {best['deg']:+.0f} deg "
              f"({best['dps']:+.1f} deg/s)")
        print("Use these as the turn primitive in auto.py.")
    else:
        print("NOTHING turned the rover more than 8 deg.")
        print("That is a mechanical constraint, not a control-parameter one:")
        print("the chassis is high-centred and the wheels cannot get purchase.")
    _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
        print("\ninterrupted - stop sent")
