"""
Escalating climb controller: attempt, measure whether it actually got anywhere,
and retry with more power ONLY when that is the safe reading of the failure.

THE PROBLEM THIS SOLVES
-----------------------
Measured on the live rover: three forward attempts (linear 0.30 and 0.26) all
failed identically -- the rover started up a rock face, could not finish, and
slid back. Operator: "movement is going on but the slope of the rock is making
it come backwards." Dropping the power changed nothing, so open-loop throttle
choice is not the lever. What was missing is a loop that NOTICES the rollback
and responds.

WHY ESCALATION IS GATED, NOT AUTOMATIC
--------------------------------------
"If it comes back to the same spot, increase the speed" is right only when the
failure was a clean rollback. A rollback that came with a large tilt excursion
is a different event: adding power there adds ENERGY to a manoeuvre that was
already close to going over, and the rover ends up on its back instead of at
the bottom of the rock. Measured excursions on these attempts reached 22 deg of
roll in a single tick and a transient 44 deg absolute.

So failures are classified, and only two of the four escalate:

  NO_MOTION  -> escalate. Wheels never turned; this is static friction.
  ROLLBACK   -> escalate. It moved, it came back, tilt stayed sane.
  UNSAFE     -> STOP. Tilt/lurch abort fired. More power here is how you flip.
  PROGRESS   -> done.

WHY NOT GPS OR AN IMU DOUBLE-INTEGRAL
-------------------------------------
GPS reads the 1000/1000 no-fix sentinel indoors, and double-integrating a
1-sample-per-tick accelerometer with ~0.8 s of transport lag produces drift far
larger than the 20-40 cm displacements being measured. The camera is the only
honest witness to "am I somewhere new".

The witness is a normalised-correlation signature over a 64x36 greyscale
thumbnail of the front frame. Calibrated on labelled frames from the 2026-08-25
session:

    no motion at all ............. corr +0.996
    pivot 37-39 deg .............. corr +0.02 .. +0.05
    forward -> lurch + rollback .. corr +0.387, +0.694
    reverse to a new pose ........ corr +0.138

Note the confound this implies: ROTATION changes the view far more than
TRANSLATION does, so correlation alone would score an honest straight-line creep
as a failure. It is therefore combined with wheel state and compass heading
rather than trusted on its own. Thresholds below are provisional -- they come
from six labelled samples, not six hundred.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .creep import (
    ABORT_ABS_DEG, ABORT_DELTA_DEG, ABORT_RATE_DEG, HZ,
    _get, _post, _ref_attitude, _stream, attitude, _wrap180,
)

# Above this the view is unchanged: the rover is where it started.
CORR_SAME = 0.90
# Below this the view genuinely changed: real net displacement.
CORR_MOVED = 0.55
# Heading change beyond this means the "forward" attempt actually slewed, so
# correlation is contaminated by rotation and must not be read as translation.
YAW_CONTAMINATION_DEG = 12.0


@dataclass
class Outcome:
    kind: str          # PROGRESS | ROLLBACK | NO_MOTION | UNSAFE
    corr: float
    d_yaw: float
    wheels_turned: bool
    peak_tilt: float
    note: str = ""


def signature(w: int = 64, h: int = 36) -> Optional[np.ndarray]:
    """Zero-mean unit-variance thumbnail of the front frame."""
    from PIL import Image
    try:
        d = _get("/v2/screenshot", timeout=90)
    except Exception:
        return None
    b = d.get("front_frame")
    if not b:
        return None
    if "," in b:
        b = b.split(",", 1)[1]
    im = Image.open(io.BytesIO(base64.b64decode(b))).convert("L").resize((w, h))
    a = np.asarray(im, dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def correlate(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    return float((a * b).mean())


def _heading() -> Optional[float]:
    try:
        o = _get("/data", timeout=10).get("orientation")
        return float(o) if o is not None else None
    except Exception:
        return None


def attempt(power: float, seconds: float, log) -> Outcome:
    """One bounded forward attempt, fully instrumented."""
    ref = _ref_attitude()
    if ref is None:
        return Outcome("UNSAFE", float("nan"), 0.0, False, 0.0,
                       "could not read attitude")

    a0 = attitude(_get("/data"))
    if a0 is not None and a0.tilt > ABORT_ABS_DEG:
        return Outcome("UNSAFE", float("nan"), 0.0, False, a0.tilt,
                       f"already at {a0.tilt:.1f} deg absolute tilt")

    sig0 = signature()
    yaw0 = _heading()
    print(f"  attempt: linear {power:+.2f} for {seconds:.1f}s  "
          f"(ref roll {ref['roll']:+.1f} pitch {ref['pitch']:+.1f})")

    ok, reason = _stream(power, 0.0, seconds, ref, log)

    # Let it settle before judging: the rollback happens AFTER the command ends.
    time.sleep(2.0)
    moving = True
    for _ in range(12):
        a = attitude(_get("/data"))
        if a is not None and not a.moving:
            moving = False
            break
        time.sleep(0.5)

    sig1 = signature()
    yaw1 = _heading()
    corr = correlate(sig0, sig1)
    d_yaw = (_wrap180(yaw1 - yaw0) if (yaw0 is not None and yaw1 is not None)
             else 0.0)
    a1 = attitude(_get("/data"))
    peak = a1.tilt if a1 else float("nan")

    if not ok:
        return Outcome("UNSAFE", corr, d_yaw, True, peak, reason)
    if abs(d_yaw) > YAW_CONTAMINATION_DEG:
        return Outcome("PROGRESS", corr, d_yaw, True, peak,
                       f"slewed {d_yaw:+.0f} deg - correlation not usable as "
                       "a translation measure")
    if corr > CORR_SAME:
        return Outcome("NO_MOTION", corr, d_yaw, False, peak,
                       "view unchanged - never broke static friction")
    if corr < CORR_MOVED:
        return Outcome("PROGRESS", corr, d_yaw, True, peak, "view changed")
    return Outcome("ROLLBACK", corr, d_yaw, True, peak,
                   "moved then returned toward the start")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="erc_offroad.climb")
    ap.add_argument("--p0", type=float, default=0.28, help="starting linear")
    ap.add_argument("--dp", type=float, default=0.06, help="escalation step")
    ap.add_argument("--pmax", type=float, default=0.45,
                    help="hard ceiling on linear; refuses to go past this")
    ap.add_argument("--seconds", type=float, default=1.6)
    ap.add_argument("--max-attempts", type=int, default=4)
    ap.add_argument("--out", default="runs/offroad")
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    power = a.p0
    with open(os.path.join(a.out, "climb_log.jsonl"), "a") as log:
        for i in range(1, a.max_attempts + 1):
            print(f"\n=== attempt {i}/{a.max_attempts}  power {power:.2f} ===")
            o = attempt(power, a.seconds, log)
            print(f"  -> {o.kind}  corr {o.corr:+.3f}  d_yaw {o.d_yaw:+.0f}  "
                  f"tilt {o.peak_tilt:.1f}   {o.note}")
            log.write(json.dumps({"t": time.time(), "attempt": i,
                                  "power": power, **o.__dict__}) + "\n")
            log.flush()

            if o.kind == "PROGRESS":
                print("\nPROGRESS - stopping here rather than pushing further.")
                return 0
            if o.kind == "UNSAFE":
                # The whole point of the gate: do NOT add power to a manoeuvre
                # that already tripped the tilt guard.
                print("\nUNSAFE - refusing to escalate power into a tilt event.")
                print("  This rock needs a different LINE, not more throttle.")
                return 2
            nxt = round(power + a.dp, 3)
            if nxt > a.pmax + 1e-9:
                print(f"\nreached power ceiling {a.pmax:.2f} without progress.")
                print("  Escalation is exhausted; this is a route problem.")
                return 3
            print(f"  {o.kind} -> escalating {power:.2f} -> {nxt:.2f}")
            power = nxt
    print("\nattempts exhausted without progress.")
    return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
        print("\ninterrupted - stop sent")
