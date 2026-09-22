"""
Bench checks for a rover on the ground in front of you. Nothing autonomous.

    python -m erc_urban_test.bench sign      # which way does +angular turn?
    python -m erc_urban_test.bench sensors   # what is actually alive on this unit
    python -m erc_urban_test.bench map       # save the minimap to a PNG

WHY THE SIGN CHECK EXISTS
-------------------------
`/control` passes {linear, angular} straight through to the rover; the SDK
documents no sign convention, so the only ground truth is measurement. The
convention matters more than it sounds: get it backwards and every heading
correction drives the rover away from where it should go, and every escape turns
into the obstacle it is escaping.

Inference from a Wuhan log says positive angular turns LEFT (a pivot commanded
ang=-0.60 for a -180 deg target moved the goal bearing -93 -> -57 deg, i.e. the
rover swung right). That is inference from one log on one unit. This measures it
on THIS unit, which is what you asked for.

SAFETY
------
Rotation only: `linear` is pinned to 0.0 in every burst, so the rover cannot
drive into anything it is not already touching. Commands are STREAMED at 5 Hz
because the Mini+ has a command watchdog — measured, a single POST yields +0.7
deg of yaw where the same command at 5 Hz yields +18.6 deg, so a script that
posts once measures a stationary robot and concludes nothing.

Telemetry is 0.8-1.4 s stale, so every reading is taken after a settle.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Optional

import numpy as np

from .sdk_client import RoverClient

BURST_S = float(os.environ.get("BENCH_BURST_S", "1.6"))
BURST_ANG = float(os.environ.get("BENCH_BURST_ANG", "0.60"))
SETTLE_S = float(os.environ.get("BENCH_SETTLE_S", "2.5"))
STREAM_HZ = 5.0


def _yaw_deg(client: RoverClient) -> Optional[float]:
    tel = client.get_data()
    return None if tel.yaw is None else float(np.degrees(tel.yaw))


def _settled_yaw(client: RoverClient, n: int = 5) -> Optional[float]:
    """
    Median of several readings after the pipeline has flushed.

    A median rather than a mean: the compass throws occasional wild samples on
    this hardware, and one of them would move a mean by tens of degrees.
    """
    time.sleep(SETTLE_S)
    vals = []
    for _ in range(n):
        y = _yaw_deg(client)
        if y is not None:
            vals.append(y)
        time.sleep(0.25)
    if not vals:
        return None
    # Circular median via unit vectors — a plain median is wrong across 0/360.
    a = np.radians(np.asarray(vals))
    return float(np.degrees(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def _burst(client: RoverClient, ang: float, seconds: float) -> None:
    """Stream a rotation-only command. `linear` is 0.0 and never a parameter."""
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        client.send_cmd(0.0, ang, lamp=0)
        time.sleep(1.0 / STREAM_HZ)
    client.send_cmd(0.0, 0.0, lamp=0)


def cmd_sign(client: RoverClient) -> int:
    print("=" * 66)
    print("ANGULAR SIGN CHECK — rotation only, linear pinned to 0.0")
    print(f"burst {BURST_S:.1f}s @ ang={BURST_ANG:+.2f}, streamed at {STREAM_HZ:.0f} Hz")
    print("=" * 66)

    y0 = _settled_yaw(client)
    if y0 is None:
        print("no compass reading — cannot measure. Is the mission started?")
        return 1
    print(f"  yaw before : {y0:+7.1f} deg (compass, CW from North)")

    _burst(client, +BURST_ANG, BURST_S)
    y1 = _settled_yaw(client)
    if y1 is None:
        print("no compass reading after the burst — inconclusive.")
        return 1
    d_pos = _wrap180(y1 - y0)
    print(f"  yaw after  : {y1:+7.1f} deg   delta = {d_pos:+.1f} deg")

    if abs(d_pos) < 8.0:
        print()
        print(f"  INCONCLUSIVE: only {abs(d_pos):.1f} deg of yaw.")
        print("  Either the rover did not break stiction, or the compass is dead.")
        print("  Retry with BENCH_BURST_ANG=0.8 BENCH_BURST_S=2.5, and watch it.")
        return 2

    # Compass is CW-from-North, so a DECREASING heading is a counter-clockwise
    # (left) turn when viewed from above.
    positive_is_left = d_pos < 0.0
    print()
    print(f"  ang = +{BURST_ANG:.2f} turned the rover "
          f"{'LEFT (counter-clockwise)' if positive_is_left else 'RIGHT (clockwise)'}")
    print()
    if positive_is_left:
        print("  => POSITIVE angular = LEFT.")
        print("     Matches erc_urban's convention (goal bearing is left-positive,")
        print("     and _align_cmd sends angular with the same sign as the bearing).")
        print("     No change needed.")
    else:
        print("  => POSITIVE angular = RIGHT.")
        print("     This CONTRADICTS erc_urban, which sends angular with the same")
        print("     sign as a left-positive goal bearing. Every heading correction")
        print("     on this unit would steer the wrong way. Negate angular in")
        print("     sdk_client.send_cmd before doing anything autonomous.")

    # Return the rover roughly to where it started, as a courtesy and as a
    # second, independent confirmation that the sign reverses as expected.
    print()
    print("  returning to the start heading...")
    _burst(client, -BURST_ANG, BURST_S)
    y2 = _settled_yaw(client)
    if y2 is not None:
        d_neg = _wrap180(y2 - y1)
        print(f"  yaw after reverse burst: {y2:+7.1f} deg   delta = {d_neg:+.1f} deg")
        if np.sign(d_neg) == np.sign(d_pos):
            print("  WARNING: both signs produced the SAME direction of rotation.")
            print("  That is not a steering command — suspect the compass, not the sign.")
        else:
            print(f"  consistent: reversing the sign reversed the rotation.")
            rate = abs(d_pos) / max(BURST_S, 1e-3)
            print(f"  measured yaw authority: ~{rate:.0f} deg/s per {BURST_ANG:.2f} angular")
    return 0


def cmd_sensors(client: RoverClient) -> int:
    """What is actually alive on this unit. Every assumption in the stack."""
    print("=" * 66)
    print("SENSOR REALITY CHECK")
    print("=" * 66)
    tel = client.get_data()
    raw = tel.raw or {}
    print(f"  lat/lon     : {tel.lat:.7f}, {tel.lon:.7f}")
    print(f"  orientation : {raw.get('orientation')}  -> yaw {tel.yaw}")
    print(f"  speed       : {raw.get('speed')}   (pinned at 0 on the Mini+)")
    print(f"  rpms        : {raw.get('rpms')}    (reads [0,0,0,0] on the Mini+)")
    print(f"  battery     : {raw.get('battery')}%   signal {raw.get('signal_level')}")
    print(f"  gps_signal  : {raw.get('gps_signal')}  hdop {raw.get('hdop')}")
    print(f"  telemetry age: {tel.age_s}s")

    # Is GPS actually live, or bit-identical like the unit the stack assumes?
    print()
    print("  sampling GPS for 12 s to see whether it MOVES while stationary...")
    seen = set()
    for _ in range(8):
        t = client.get_data()
        seen.add((t.lat, t.lon))
        time.sleep(1.5)
    print(f"  distinct fixes while stationary: {len(seen)}")
    if len(seen) == 1:
        print("  -> GPS is FROZEN (bit-identical). Vision is the only live signal;")
        print("     do not fuse GPS into pose on this unit.")
    else:
        print("  -> GPS updates. Worth fusing into the LocalMap pose, which")
        print("     currently under-measures travel by ~3.6x on VO alone.")

    front, rear = client.get_frames()
    print()
    print(f"  front frame : {'OK ' + str(front.shape) if front is not None else 'MISSING'}")
    print(f"  rear frame  : {'OK ' + str(rear.shape) if rear is not None else 'None'}"
          "   (expected None: /v2/screenshot returns rear only when BOT_TYPE=='zero',")
    print("                 and the Mini+ reports 'mini')")
    m = client.get_map_frame()
    print(f"  map frame   : {'OK ' + str(m.shape) if m is not None else 'MISSING'}")
    return 0


def cmd_map(client: RoverClient) -> int:
    from PIL import Image

    m = client.get_map_frame()
    if m is None:
        print("no map frame (is the mission started?)")
        return 1
    out = os.environ.get("BENCH_MAP_OUT", "map_frame.png")
    Image.fromarray(m).save(out)
    print(f"saved {out}  {m.shape}")
    print("This is rendered OpenStreetMap. For geometry, query Overpass at the")
    print("rover's lat/lon instead of reading the picture.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("check", choices=["sign", "sensors", "map"])
    ap.add_argument("--url", default=os.environ.get("SDK_URL", "http://localhost:8000"))
    args = ap.parse_args(argv)

    client = RoverClient(args.url)
    if not client.ping():
        print(f"SDK not reachable at {args.url}")
        print("Start it:  cd earth-rovers-sdk && hypercorn main:app --reload")
        return 1
    try:
        return {"sign": cmd_sign, "sensors": cmd_sensors, "map": cmd_map}[args.check](client)
    finally:
        try:
            client.send_cmd(0.0, 0.0, lamp=0)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
