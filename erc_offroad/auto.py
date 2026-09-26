"""
Autonomous off-road driver: pick a line, aim at it, modulate speed, notice when
that is not working, and do something different.

    python -m erc_offroad.auto --minutes 5
    python -m erc_offroad.auto --dry-run        # decides, sends nothing

FOUR THINGS IT DOES, IN THE ORDER THEY MATTER
---------------------------------------------
1. SAFETY. Tilt and lurch aborts from creep.py. Nothing below overrides them.
2. AIM. With a GPS fix, the bearing to the live checkpoint. Without one -- and
   there is no fix indoors, the SDK returns the 1000/1000 sentinel -- the aim is
   the most traversable direction, i.e. get out of the rocks and into open sky
   where a fix exists. Handing over to GPS is automatic the moment it appears.
3. LINE. Terrain is the CONSTRAINT and the goal only biases WITHIN it. Scoring
   each column by clearance minus a weighted heading error is VFH+; picking the
   goal-most column regardless of clearance is how you drive into a boulder.
4. SPEED. `terrain.speed_policy` -- fast on a long clear run, slow in clutter or
   nose-down, bounded extra authority when climbing.

TURNS MUST BE LONG
------------------
From the 2026-08-25 teleop log, turn commands shorter than the ~0.8 s telemetry
lag achieve LITERALLY NOTHING: ang +0.30 for 0.2 s gave 0 deg, repeatedly. The
same command sustained for 2.5 s in open terrain gave 38 deg. `MIN_TURN_S`
exists because of that measurement.

WEDGED IS NOT UNDER-POWERED
---------------------------
The single most important lesson in the log. These are all real rows:

    ang +0.30 for 4.2 s  ->   0 deg   wheels turning
    ang +0.14 for 15.4 s ->  -5 deg   wheels turning
    ang +0.14 for 8.5 s  ->  +1 deg   wheels turning

Motors running, chassis not rotating: the rover is high-centred on rock and the
wheels are slipping. Adding throttle to that spins harder against stone and
raises the odds of climbing a rock and going over -- the run touched 42.5 deg of
tilt. The correct response is BACK-AND-FILL: reverse off the high point, turn
the other way, try a different line. So `_wedged()` is checked before any
escalation, and it routes to recovery rather than to more power.
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
    ABORT_ABS_DEG, HZ, _get, _post, _ref_attitude, _stream, _wrap180, attitude,
)
from .climb import signature, correlate, CORR_SAME
from .terrain import analyze, speed_policy, TerrainView
from .gpswatch import valid_fix, geo, load_checkpoints

# Below this the command dies inside the ~0.8 s telemetry lag and does nothing.
MIN_TURN_S = 2.0
# Turns are DURATION-limited, not power-limited. Measured by turntest.py on
# this rock, 2 s per strategy:
#
#     spin 0.35 -> -13 deg      spin 0.90 -> 0 deg
#     spin 0.60 ->  -7 deg      arc / wiggle -> 0 deg, and both tripped
#                               the lurch guard
#
# More angular turns the rover LESS. A skid-steer rotates by scrubbing its
# wheels sideways; commanded hard against rock it simply stalls and grinds,
# which is what the operator could hear. Backing off lets the wheels find
# purchase. So TURN_ANG stays low and long turns are bought with TIME.
MAX_TURN_S = 6.0
TURN_RATE_DPS = 6.5        # measured; adapted at runtime below
FWD_S = 1.4
TURN_ANG = 0.34
# Heading error above which we turn instead of driving.
TURN_IF_DEG = 22.0
# A turn burst that changed heading by less than this, with motors running,
# means wedged -- not under-powered.
WEDGE_YAW_DEG = 4.0
GOAL_WEIGHT = 3.5      # column-score weight on heading error, in cells/radian


@dataclass
class Decision:
    action: str
    lin: float
    ang: float
    secs: float
    aim_deg: float
    why: str


def front_rgb() -> Optional[np.ndarray]:
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
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))


def choose_line(tv: TerrainView, goal_bearing: Optional[float]) -> tuple[float, str]:
    """
    VFH+ over the roughness columns. Clearance is the constraint; the goal, when
    we have one, only biases the choice among columns that are actually clear.
    """
    if goal_bearing is None:
        return tv.best_bearing, "terrain-only (no GPS)"
    err = np.abs(np.array([_wrapr(b - goal_bearing) for b in tv.bearings]))
    score = tv.clear - GOAL_WEIGHT * err
    j = int(np.argmax(score))
    return float(tv.bearings[j]), "terrain+goal"


def _wrapr(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def gps_relative_bearing_rad(brg_cw_deg: float, yaw_cw_deg: float) -> float:
    # brg and yaw are CW-from-North; terrain bearings are left-positive.
    return math.radians(_wrap180(-(brg_cw_deg - yaw_cw_deg)))


def _wedged(d_yaw: float, corr: float, moved: bool, was_turn: bool) -> bool:
    if not moved:
        return False
    if was_turn:
        return abs(d_yaw) < WEDGE_YAW_DEG
    return (not math.isnan(corr)) and corr > CORR_SAME


def recovery_side(tv: Optional[TerrainView], last: float) -> float:
    """
    Which way to turn during back-and-fill. Returns +1 (right) or -1 (left).

    An earlier version derived this from the failed command's sign, which for a
    forward move is 0.0 -- so it collapsed to "always right" and every recovery
    tried the identical escape from the identical pocket. Ten recoveries in one
    run, all turning the same way.

    Now: go toward whichever half of the frame actually has more clear terrain,
    and if that is the same side we just tried, flip it. Repeating an escape
    that did not work is the definition of being stuck.
    """
    if tv is not None:
        half = len(tv.clear) // 2
        left = float(tv.clear[:half].sum())      # low index = left of frame
        right = float(tv.clear[half:].sum())
        if abs(left - right) > 1.0:
            side = -1.0 if left > right else 1.0
            return -side if side == last else side
    return -last if last != 0.0 else 1.0


def back_and_fill(rec_log, side: float) -> str:
    """Reverse off the high point, then turn toward `side` (+1 right, -1 left)."""
    ref = _ref_attitude()
    if ref is None:
        return "no attitude"
    print("    back-and-fill: reverse")
    _stream(-0.32, 0.0, 1.0, ref, rec_log, allow_tilt_abort=False)
    time.sleep(0.8)
    ref = _ref_attitude() or ref
    print(f"    back-and-fill: turn {'right' if side > 0 else 'left'}")
    _stream(0.0, -0.34 * (1.0 if side > 0 else -1.0), MIN_TURN_S, ref, rec_log)
    return "back-and-fill done"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="erc_offroad.auto")
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="decide and log, send no commands")
    ap.add_argument("--out", default="runs/offroad")
    ap.add_argument("--max-consec-fail", type=int, default=4)
    a = ap.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "auto_log.jsonl"), "a")

    try:
        cps, latest = load_checkpoints()
    except Exception:
        cps, latest = [], 0
    idx = max(0, min(latest, len(cps) - 1)) if cps else 0

    t_end = time.time() + a.minutes * 60.0
    consec_fail = 0
    tick = 0
    last_side = 0.0
    turn_rate = TURN_RATE_DPS

    while time.time() < t_end:
        tick += 1
        raw = _get("/data")
        att = attitude(raw)
        if att is None:
            time.sleep(0.5); continue

        if att.tilt > ABORT_ABS_DEG:
            print(f"[{tick}] ABSOLUTE TILT {att.tilt:.1f} - stopping")
            _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
            break

        rgb = front_rgb()
        tv = analyze(rgb) if rgb is not None else None
        if tv is None:
            time.sleep(0.5); continue

        goal_b = None
        goal_note = "no-gps"
        if cps and valid_fix(raw.get("latitude"), raw.get("longitude")):
            brg, dist = geo(float(raw["latitude"]), float(raw["longitude"]),
                            cps[idx]["lat"], cps[idx]["lon"])
            yaw = raw.get("orientation")
            if yaw is not None:
                goal_b = gps_relative_bearing_rad(brg, float(yaw))
                goal_note = f"cp{cps[idx]['seq']} {dist:.0f}m"
                if dist <= 15.0 and idx < len(cps) - 1:
                    idx += 1

        aim, line_note = choose_line(tv, goal_b)
        aim_deg = math.degrees(aim)
        v, why = speed_policy(tv, att.pitch, min(consec_fail / 3.0, 1.0))

        if abs(aim_deg) > TURN_IF_DEG:
            # Buy the angle with time, at the rate this rock actually allows.
            secs = float(np.clip(abs(aim_deg) / max(turn_rate, 1.5),
                                 MIN_TURN_S, MAX_TURN_S))
            d = Decision("turn", 0.0, math.copysign(TURN_ANG, aim),
                         secs, aim_deg,
                         f"{line_note}; {goal_note}; rate~{turn_rate:.1f}dps")
        else:
            d = Decision("forward", v, 0.0, FWD_S, aim_deg,
                         f"{why}; {line_note}; {goal_note}")

        print(f"[{tick}] {d.action:7s} lin{d.lin:+.2f} ang{d.ang:+.2f} "
              f"{d.secs:.1f}s | aim{aim_deg:+5.0f} depth{tv.centre_depth:.2f} "
              f"open{tv.open_frac:.2f} tilt{att.tilt:4.1f} | {d.why}")
        log.write(json.dumps({"t": time.time(), "tick": tick,
                              **d.__dict__, "tilt": att.tilt,
                              "pitch": att.pitch,
                              "centre_depth": tv.centre_depth,
                              "open_frac": tv.open_frac}) + "\n")
        log.flush()

        if a.dry_run:
            time.sleep(1.0)
            continue

        sig0 = signature()
        yaw0 = raw.get("orientation")
        ref = _ref_attitude()
        if ref is None:
            continue
        ok, reason = _stream(d.lin, d.ang, d.secs, ref, log)
        time.sleep(1.5)

        after = _get("/data")
        yaw1 = after.get("orientation")
        d_yaw = (_wrap180(float(yaw1) - float(yaw0))
                 if (yaw0 is not None and yaw1 is not None) else 0.0)
        corr = correlate(sig0, signature())
        # We commanded motion and _stream did not abort, so the motors ran.
        # Whether the CHASSIS moved is exactly what d_yaw/corr decide below.
        moved = True

        # Adapt the rate estimate from what this turn actually achieved, so a
        # rover that frees itself starts making efficient turns again instead
        # of grinding out 6-second bursts forever.
        if d.action == "turn" and ok and abs(d_yaw) >= 3.0:
            measured = abs(d_yaw) / max(d.secs, 1e-3)
            turn_rate = float(np.clip(0.6 * turn_rate + 0.4 * measured,
                                      1.5, 25.0))
            print(f"    turn rate now ~{turn_rate:.1f} deg/s "
                  f"(this burst {measured:.1f})")

        if not ok:
            consec_fail += 1
            print(f"    guard: {reason}")
            last_side = recovery_side(tv, last_side)
            back_and_fill(log, last_side)
        elif _wedged(d_yaw, corr, moved, d.action == "turn"):
            consec_fail += 1
            print(f"    WEDGED (dyaw {d_yaw:+.0f}, corr {corr:+.2f}) - "
                  "back-and-fill, NOT more throttle")
            last_side = recovery_side(tv, last_side)
            back_and_fill(log, last_side)
        else:
            consec_fail = 0
            last_side = 0.0

        if consec_fail >= a.max_consec_fail:
            print(f"\n{consec_fail} consecutive failures - stopping. "
                  "This needs a different line than the rover can find alone.")
            break

    _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
    log.close()
    print("\nstopped, motors zeroed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
        print("\ninterrupted - stop sent")
