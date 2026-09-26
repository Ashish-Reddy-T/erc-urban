"""
Bounded-nudge creep driver for the off-road rock garden.

WHY NUDGES AND NOT AUTONOMY
---------------------------
Measured on this unit, on this course, before writing a line of this:

  * GPS is DEAD. /data returns latitude=1000, longitude=1000 (the SDK's
    no-fix sentinel) on every sample over 16 s, gps_signal ~28. The mission's
    checkpoints are GPS coordinates 39 m apart. Nothing can servo to them.
  * There is NO REAR CAMERA on /v2/screenshot. Reversing is blind.
  * Telemetry is ~0.8 s stale (measured hw_age 0.62-1.03 s). Any reaction to
    a tilt spike is already a second late.
  * The ground is stone slabs with gaps. `test_step_hazard.py` already pins
    that a slab edge presents almost nothing above a fitted ground plane, so
    the depth/RANSAC veto goes blind exactly here.

Four blind spots at once is not a place to let a planner drive. So motion is
issued as SHORT, BOUNDED nudges with a human deciding each one, and this module
supplies only the safety envelope around them.

THE COMMAND WATCHDOG
--------------------
The Mini+ stops when commands stop arriving, so a nudge STREAMS its command at
`HZ` for its duration and then sends an explicit stop. Sending once and sleeping
records a stationary robot -- the bug that `eval/record.py` documents.

TILT IS MEASURED RELATIVE TO A BASELINE
---------------------------------------
The rover is NOT level here: at rest on the rocks it reads roll +5.2 deg,
pitch +5.5 deg, total 7.6 deg. An absolute threshold is therefore either 7.6 deg
late or permanently tripped. Everything below is relative to a baseline captured
while stationary.

Gravity sits on accels index 2 on this unit (measured [-0.109, 0.098, 1.087]).
`src/imu_safety.py` in the ERC-3 repo documents index 0; that is WRONG here and
a guard built on it would compute nonsense.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Optional

import urllib.request
import urllib.error

URL = os.environ.get("SDK_URL", "http://localhost:8000")
HZ = 5.0
# Per-nudge duration cap. With ~0.8 s of telemetry lag, anything longer commits
# the rover to travel it cannot react inside.
MAX_NUDGE_S = 3.0
# Tilt above baseline that aborts a nudge.
ABORT_DELTA_DEG = 12.0
# Absolute tilt that aborts regardless of baseline -- the rover is going over.
ABORT_ABS_DEG = 28.0
# A single-tick jump this large is a LURCH (wheel dropping into a crevice
# between slabs), not a climb. Measured: an ordinary pivot on this rock moved
# attitude ~3 deg per tick, so this sits well above normal traverse.
ABORT_RATE_DEG = 7.0
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "tilt_baseline.json")


def _get(path: str, timeout: float = 30.0):
    with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as r:
        return json.load(r)


def _post(path: str, payload: dict, timeout: float = 15.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{URL}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read()[:300]


@dataclass
class Attitude:
    roll: float
    pitch: float
    tilt: float
    gz: float
    age_s: float
    moving: bool
    orientation: Optional[float]


def attitude(d: dict) -> Optional[Attitude]:
    acc = d.get("accels") or []
    if not acc or len(acc[-1]) < 4:
        return None
    a = acc[-1][:3]
    n = math.sqrt(sum(v * v for v in a)) or 1.0
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, a[2] / n))))
    roll = math.degrees(math.atan2(a[1], a[2]))
    pitch = math.degrees(math.atan2(-a[0], math.sqrt(a[1] ** 2 + a[2] ** 2)))
    gy = d.get("gyros") or [[0, 0, 0, 0]]
    gz = float(gy[-1][2]) if len(gy[-1]) >= 3 else 0.0
    rp = d.get("rpms") or [[0, 0, 0, 0, 0]]
    try:
        moving = any(abs(float(x)) > 1e-6 for x in rp[-1][:-1])
    except (TypeError, ValueError):
        moving = False
    return Attitude(roll, pitch, tilt, gz, time.time() - float(acc[-1][3]),
                    moving, d.get("orientation"))


def load_baseline() -> Optional[dict]:
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return None


def cmd_calib(_args) -> int:
    """Capture the at-rest attitude. Run this whenever the rover is re-placed."""
    print("Sampling at-rest attitude (rover must be STOPPED)...")
    rolls, pitches, gzs = [], [], []
    for _ in range(12):
        a = attitude(_get("/data"))
        if a is None:
            print("  no IMU in payload"); return 1
        if a.moving:
            print("  WHEELS TURNING - rover must be stopped to calibrate"); return 1
        rolls.append(a.roll); pitches.append(a.pitch); gzs.append(a.gz)
        time.sleep(0.4)
    base = {
        "roll": sum(rolls) / len(rolls),
        "pitch": sum(pitches) / len(pitches),
        "gz_bias": sum(gzs) / len(gzs),
        "stamp": time.time(),
    }
    with open(BASELINE_PATH, "w") as f:
        json.dump(base, f, indent=2)
    print(f"  roll {base['roll']:+.2f}  pitch {base['pitch']:+.2f}  "
          f"gz_bias {base['gz_bias']:+.3f} deg/s")
    print(f"  saved -> {BASELINE_PATH}")
    print("  abort fires at delta > "
          f"{ABORT_DELTA_DEG:.0f} deg from here, or {ABORT_ABS_DEG:.0f} deg absolute")
    return 0


def cmd_watch(args) -> int:
    """
    Live attitude. Flags on ABSOLUTE tilt and on tick-to-tick rate only.

    Deliberately NOT on displacement from a stored baseline. The rover's
    resting posture legitimately changes every time it moves on rock -- after
    one climb-and-rollback it sat at roll -7.5 where it had been +3.9, an
    identical 7.8 deg of absolute tilt in a mirrored posture. Flagging that as
    "OVER LIMIT" is a false alarm, and a guard that cries wolf gets ignored
    exactly when it matters.
    """
    print("  t    roll   pitch   tilt   rate  |  gz     age  wheels  lat/lon")
    t0 = time.time()
    prev_r = prev_p = None
    while time.time() - t0 < args.seconds:
        d = _get("/data")
        a = attitude(d)
        if a is None:
            print("no imu"); time.sleep(0.5); continue
        rate = 0.0
        if prev_r is not None:
            rate = max(abs(a.roll - prev_r), abs(a.pitch - prev_p))
        prev_r, prev_p = a.roll, a.pitch
        flag = ""
        if a.tilt > ABORT_ABS_DEG:
            flag = "  <-- ABSOLUTE TILT HIGH"
        elif rate > ABORT_RATE_DEG:
            flag = "  <-- LURCH"
        print(f"{time.time()-t0:5.1f} {a.roll:+6.2f} {a.pitch:+6.2f} {a.tilt:6.2f} "
              f"{rate:6.2f} | {a.gz:+6.2f} {a.age_s:5.2f} "
              f"{str(a.moving):>6s}  {d.get('latitude')},{d.get('longitude')}{flag}")
        time.sleep(args.period)
    return 0


def cmd_look(args) -> int:
    """Grab front (and rear if the unit has one) into the run dir."""
    from PIL import Image
    import base64, io
    os.makedirs(args.out, exist_ok=True)
    ts = time.strftime("%H%M%S")
    got = []
    for path, key, name in (("/v2/screenshot", "front_frame", "front"),
                            ("/v2/rear", "rear_frame", "rear")):
        try:
            d = _get(path, timeout=90)
        except Exception as e:
            print(f"  {name}: unavailable ({e})"); continue
        b = d.get(key)
        if not b:
            print(f"  {name}: not in payload"); continue
        if "," in b:
            b = b.split(",", 1)[1]
        im = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
        p = os.path.join(args.out, f"{ts}_{name}.png")
        im.save(p)
        got.append(p)
        print(f"  {name}: {im.size[0]}x{im.size[1]} -> {p}")
    return 0 if got else 1


def _ref_attitude(n: int = 4) -> Optional[dict]:
    """
    Attitude RIGHT NOW, as the reference for the nudge about to be issued.

    Re-referencing per nudge is not a nicety. The rover's resting attitude
    CHANGES every time it moves on this terrain -- one 39 deg pivot took it
    from roll +5.2/pitch +5.5 to roll +7.7/pitch +2.2. Measuring against a
    baseline captured minutes ago silently spends the safety margin on
    posture that is perfectly fine. What we actually care about is how far
    THIS nudge tilts the rover from where it started.
    """
    rolls, pitches = [], []
    for _ in range(n):
        try:
            a = attitude(_get("/data", timeout=10))
        except Exception:
            continue
        if a is None:
            continue
        rolls.append(a.roll); pitches.append(a.pitch)
        time.sleep(0.2)
    if not rolls:
        return None
    rolls.sort(); pitches.sort()
    return {"roll": rolls[len(rolls)//2], "pitch": pitches[len(pitches)//2]}


def _stream(lin: float, ang: float, seconds: float, base: dict, log,
            allow_tilt_abort: bool = True) -> tuple[bool, str]:
    """
    Stream one command for `seconds`, watching tilt every tick.

    Three independent aborts: displacement from the pre-nudge reference,
    absolute tilt, and single-tick rate (a lurch into a crevice).

    Returns (ok, reason). ok=False means the safety envelope stopped it.
    """
    prev_roll = prev_pitch = None
    n = max(1, int(seconds * HZ))
    dt = 1.0 / HZ
    for i in range(n):
        _post("/control", {"command": {"linear": lin, "angular": ang}})
        try:
            d = _get("/data", timeout=10)
        except Exception:
            time.sleep(dt); continue
        a = attitude(d)
        if a is None:
            time.sleep(dt); continue
        dr = a.roll - base["roll"]; dp = a.pitch - base["pitch"]
        worst = max(abs(dr), abs(dp))
        if log is not None:
            log.write(json.dumps({
                "t": time.time(), "lin": lin, "ang": ang,
                "roll": a.roll, "pitch": a.pitch, "tilt": a.tilt,
                "d_roll": dr, "d_pitch": dp, "gz": a.gz,
                "age": a.age_s, "moving": a.moving,
            }) + "\n")
            log.flush()
        print(f"    [{i+1}/{n}] roll{a.roll:+6.2f} pitch{a.pitch:+6.2f} "
              f"d{worst:5.2f} gz{a.gz:+6.2f} wheels={a.moving}")
        rate = 0.0
        if prev_roll is not None:
            rate = max(abs(a.roll - prev_roll), abs(a.pitch - prev_pitch))
        prev_roll, prev_pitch = a.roll, a.pitch

        # Absolute tilt always aborts: past it the rover is going over whichever
        # way it drives. allow_tilt_abort=False (recovery reverses) only drops
        # the delta and lurch checks, which reversing off a hump trips by design.
        if a.tilt > ABORT_ABS_DEG:
            _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
            return False, (f"ABS TILT ABORT abs={a.tilt:.1f} "
                           f"d_roll={dr:+.1f} d_pitch={dp:+.1f}")
        if allow_tilt_abort:
            if worst > ABORT_DELTA_DEG:
                _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
                return False, (f"TILT ABORT d_roll={dr:+.1f} d_pitch={dp:+.1f} "
                               f"abs={a.tilt:.1f}")
            if rate > ABORT_RATE_DEG:
                _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
                return False, f"LURCH ABORT {rate:.1f} deg in one tick"
        time.sleep(dt)
    _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
    return True, "ok"


def cmd_nudge(args) -> int:
    if args.seconds > MAX_NUDGE_S:
        print(f"refusing: {args.seconds}s exceeds MAX_NUDGE_S={MAX_NUDGE_S}")
        return 1

    base = _ref_attitude()
    if base is None:
        print("could not read attitude; refusing to move"); return 1
    a0 = attitude(_get("/data"))
    if a0 is not None and a0.tilt > ABORT_ABS_DEG:
        print(f"REFUSING: already at {a0.tilt:.1f} deg absolute tilt")
        return 1
    print(f"ref attitude: roll {base['roll']:+.2f} pitch {base['pitch']:+.2f}"
          + (f"   (abs tilt {a0.tilt:.1f})" if a0 else ""))

    lin, ang = args.linear, args.angular
    os.makedirs(args.out, exist_ok=True)
    logp = os.path.join(args.out, "creep_log.jsonl")
    print(f"nudge lin={lin:+.2f} ang={ang:+.2f} for {args.seconds:.1f}s "
          f"(streaming {HZ:.0f}Hz)")
    with open(logp, "a") as log:
        ok, reason = _stream(lin, ang, args.seconds, base, log)
        print(f"  -> {reason}")
        if not ok and args.auto_recover:
            # The user's stated recovery: quick back-acceleration, once or twice.
            # Blind (no rear camera), so it is short and it is bounded.
            for attempt in (1, 2):
                print(f"  RECOVER {attempt}/2: reverse burst 0.6s")
                rok, rreason = _stream(-abs(args.recover_linear), 0.0, 0.6, base, log,
                                       allow_tilt_abort=False)
                if not rok:
                    print(f"     {rreason} - stop and look at the rover")
                    return 2
                time.sleep(0.4)
                a = attitude(_get("/data"))
                if a is None:
                    continue
                dr = a.roll - base["roll"]; dp = a.pitch - base["pitch"]
                print(f"     after: d_roll={dr:+.1f} d_pitch={dp:+.1f} "
                      f"abs={a.tilt:.1f}")
                if max(abs(dr), abs(dp)) < ABORT_DELTA_DEG * 0.6:
                    print("     recovered")
                    return 0
            print("     STILL TILTED - stop and look at the rover")
            return 2
    return 0 if ok else 2


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def _settled_orientation(n: int = 6) -> Optional[float]:
    """Median compass heading while stopped. Rejects the sample if wheels turn."""
    vals = []
    for _ in range(n):
        d = _get("/data")
        a = attitude(d)
        if a is not None and a.moving:
            return None
        o = d.get("orientation")
        if o is not None:
            vals.append(float(o))
        time.sleep(0.3)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def cmd_signcheck(args) -> int:
    """
    Determine what POSITIVE angular actually does on this unit.

    Two independent signals, because either alone can lie here: the compass
    sits beside cinder block and possible rebar, and the gyro has a real
    -1.53 deg/s bias. If they disagree, trust neither and turn by hand.
    """
    base = load_baseline()
    if base is None:
        print("run calib first"); return 1
    gz_bias = base.get("gz_bias", 0.0)

    o0 = _settled_orientation()
    if o0 is None:
        print("could not read a settled heading (wheels turning?)"); return 1
    print(f"heading before: {o0:.1f} deg")
    print(f"commanding angular={args.angular:+.2f} for {args.seconds:.1f}s")

    n = max(1, int(args.seconds * HZ))
    dt = 1.0 / HZ
    gyro_yaw = 0.0
    for i in range(n):
        _post("/control", {"command": {"linear": 0.0, "angular": args.angular}})
        try:
            d = _get("/data", timeout=10)
        except Exception:
            time.sleep(dt); continue
        a = attitude(d)
        if a is None:
            time.sleep(dt); continue
        gyro_yaw += (a.gz - gz_bias) * dt
        dr = a.roll - base["roll"]; dp = a.pitch - base["pitch"]
        print(f"    [{i+1}/{n}] gz{a.gz:+6.2f} (corr {a.gz-gz_bias:+6.2f}) "
              f"intg{gyro_yaw:+7.2f} d_roll{dr:+5.2f} d_pitch{dp:+5.2f} "
              f"wheels={a.moving}")
        if max(abs(dr), abs(dp)) > ABORT_DELTA_DEG or a.tilt > ABORT_ABS_DEG:
            _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
            print("    TILT ABORT"); return 2
        time.sleep(dt)
    _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
    time.sleep(1.2)

    o1 = _settled_orientation()
    if o1 is None:
        print("could not read settled heading after"); return 1
    dc = _wrap180(o1 - o0)
    print(f"\nheading after:  {o1:.1f} deg")
    print(f"compass delta:  {dc:+.1f} deg  (orientation is CW-from-North)")
    print(f"gyro integral:  {gyro_yaw:+.1f} deg (bias-corrected)")

    if abs(dc) < 3.0:
        print("\nINCONCLUSIVE: heading barely moved. Either it did not turn "
              "(too little power on rock) or the compass is stuck.")
        return 3
    turned = "RIGHT (clockwise)" if dc > 0 else "LEFT (counter-clockwise)"
    print(f"\n=> angular {args.angular:+.2f} turns the rover {turned}")
    agree = (dc > 0) == (gyro_yaw > 0)
    if abs(gyro_yaw) < 2.0:
        print("   (gyro integral too small to corroborate)")
    elif agree:
        print("   gyro AGREES with compass - confident")
    else:
        print("   WARNING: gyro DISAGREES with compass. Do not trust either; "
              "confirm visually before any autonomous turn.")
    return 0


def cmd_stop(_args) -> int:
    for _ in range(3):
        _post("/control", {"command": {"linear": 0.0, "angular": 0.0}})
        time.sleep(0.15)
    print("stop sent x3")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="erc_offroad.creep")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("calib").set_defaults(fn=cmd_calib)
    sub.add_parser("stop").set_defaults(fn=cmd_stop)

    sc = sub.add_parser("signcheck"); sc.set_defaults(fn=cmd_signcheck)
    sc.add_argument("--angular", type=float, default=0.20)
    sc.add_argument("--seconds", type=float, default=1.2)

    w = sub.add_parser("watch"); w.set_defaults(fn=cmd_watch)
    w.add_argument("--seconds", type=float, default=30.0)
    w.add_argument("--period", type=float, default=1.0)

    lk = sub.add_parser("look"); lk.set_defaults(fn=cmd_look)
    lk.add_argument("--out", default="runs/offroad")

    for name, lin, ang in (("fwd", 0.25, 0.0), ("back", -0.25, 0.0),
                           ("left", 0.0, 0.20), ("right", 0.0, -0.20)):
        p = sub.add_parser(name); p.set_defaults(fn=cmd_nudge)
        p.add_argument("seconds", type=float, nargs="?", default=0.6)
        p.add_argument("--linear", type=float, default=lin)
        p.add_argument("--angular", type=float, default=ang)
        p.add_argument("--out", default="runs/offroad")
        p.add_argument("--recover-linear", type=float, default=0.35)
        p.add_argument("--no-auto-recover", dest="auto_recover",
                       action="store_false", default=True)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
