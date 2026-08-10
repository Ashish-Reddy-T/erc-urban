"""
Record a driving episode for offline benchmarking.

    python -m erc_urban.eval.record --out runs/ep1 --seconds 60

Saves frames plus, per tick, the visual-odometry delta and the command issued.
The VO track is what makes the recording useful: GPS on this rover is frozen
(bit-identical across 30 s of driving), so the only available ground truth for
"where did the rover actually go" is the one we compute ourselves.

Commands are STREAMED at `--hz`, not sent once. The Mini+ has a command
watchdog: a single POST yields +0.7 deg of yaw where the same command at 5 Hz
yields +18.6 deg. A recorder that sends once records a stationary robot.

SAFETY -- READ THIS BEFORE CHANGING ANYTHING BELOW
--------------------------------------------------
An earlier version of this file streamed a fixed command open-loop with NO
perception in the loop at all, and put the rover on its side. The failure is
worth stating exactly, because the information needed to prevent it was already
being recorded:

  * over the final 40 ticks -- 8 seconds of commanded 0.32 forward -- visual
    odometry measured 19 cm of travel, 0.3-0.8 cm per tick, at confidence
    0.94-0.98. High confidence means that was a real measurement of a jammed
    rover, not a tracking dropout.
  * the frames at that point were almost entirely sky: the rover had climbed a
    planter and pitched up.
  * the recorder kept commanding forward regardless, and it went over.

The whole stack has layers that catch all of this -- TRACK_MIN, the costmap
stop, the semantic veto, the IMU tilt -- and this file used none of them.
A benchmark harness that drives must not be less careful than the thing it is
benchmarking. Three independent guards now run every tick, any of which halts:

  1. IMU tilt        -- the rover is going over
  2. VO stall        -- commanded to move, measurably not moving = jammed
  3. traversability  -- no drivable surface in the near field

Use --unsafe only for a rover on blocks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from PIL import Image

from ..calib import CameraCalibration
from ..modules.odometry import VisualOdometry
from ..sdk_client import RoverClient


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--linear", type=float, default=0.32)
    ap.add_argument("--angular", type=float, default=0.0)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--unsafe", action="store_true",
                    help="disable the tilt/stall/traversability guards. Only "
                         "for a rover on blocks.")
    ap.add_argument("--max-tilt-deg", type=float, default=25.0)
    ap.add_argument("--stall-cm", type=float, default=1.0,
                    help="per-tick travel below this, while commanded to move, "
                         "counts as jammed")
    ap.add_argument("--stall-ticks", type=int, default=8)
    ap.add_argument(
        "--pattern", default="",
        help="cycle of 'lin:ang:ticks' segments, e.g. "
             "'0.32:0:25,0.32:0.25:12,0.32:0:25,0.32:-0.25:12'. "
             "A single fixed command produces a constant arc, which is useless "
             "as a reference for a perception-driven planner -- it penalises "
             "the planner for correctly going straight and rewards any selector "
             "with a matching turn bias. Vary the command.",
    )
    a = ap.parse_args(argv)

    segments = []
    for part in filter(None, a.pattern.split(",")):
        lin, ang, n = part.split(":")
        segments.append((float(lin), float(ang), int(n)))
    if not segments:
        segments = [(a.linear, a.angular, 10 ** 9)]

    os.makedirs(a.out, exist_ok=True)
    cal = CameraCalibration.load_or_default("erc_urban/camera_front.json", 1024, 576)
    c = RoverClient(a.url)
    vo = VisualOdometry(cal)

    print(f"recording {a.seconds:.0f}s to {a.out} at {a.hz:g} Hz "
          f"(lin={a.linear} ang={a.angular})")
    recs = []
    t0 = time.monotonic()
    i = 0
    seg_i, seg_left = 0, segments[0][2]
    stall = 0
    sem = None
    if not a.unsafe:
        try:
            from ..modules.semantics import SemanticTraversability
            sem = SemanticTraversability(device="auto")
            if not sem.ready:
                sem = None
        except Exception as e:
            print(f"  semantics unavailable ({e})")
        if sem is None:
            print("  REFUSING to drive: no traversability check available.\n"
                  "  Pass --unsafe only if the rover is on blocks.")
            return 2
    stop_reason = None
    try:
        while time.monotonic() - t0 < a.seconds:
            tick = time.monotonic()
            if seg_left <= 0:
                seg_i = (seg_i + 1) % len(segments)
                seg_left = segments[seg_i][2]
            lin, ang, _ = segments[seg_i]
            seg_left -= 1
            c.send_cmd(lin, ang, lamp=1)      # streamed every tick (watchdog)
            f, _ = c.get_frames()
            tel = c.get_data()
            if f is None:
                continue
            d = vo.update(f)

            if not a.unsafe:
                acc = (tel.raw.get("accels") or [[0, 0, 1, 0]])[-1][:3]
                nrm = float(np.linalg.norm(acc)) or 1.0
                tilt = float(np.degrees(np.arccos(np.clip(acc[2] / nrm, -1, 1))))
                if tilt > a.max_tilt_deg:
                    stop_reason = f"tilt {tilt:.0f} deg > {a.max_tilt_deg:.0f}"
                    break
                moving_cmd = abs(lin) > 0.05
                if moving_cmd and d.confidence > 0.5 and \
                        d.distance_m * 100.0 < a.stall_cm:
                    stall += 1
                    if stall >= a.stall_ticks:
                        stop_reason = (f"jammed: {a.stall_ticks} ticks under "
                                       f"{a.stall_cm} cm while commanded forward")
                        break
                else:
                    stall = 0
                labels = sem.labels(f, (144, 256))
                if labels is not None:
                    hard, soft, good = sem.categorize(labels)
                    _, near, _ = sem.track(good)
                    if moving_cmd and near < 0.04:
                        stop_reason = f"no drivable surface ahead (near={near:.3f})"
                        break
            Image.fromarray(f).save(f"{a.out}/f{i:04d}.jpg", quality=88)
            recs.append({
                "i": i, "t": tick - t0,
                "dx": d.dx, "dy": d.dy, "dyaw": d.dyaw,
                "conf": d.confidence, "reason": d.reason,
                "lin": lin, "ang": ang,
                "lat": None if np.isnan(tel.lat) else tel.lat,
                "lon": None if np.isnan(tel.lon) else tel.lon,
                "yaw": tel.yaw,
            })
            i += 1
            if i % 10 == 0:
                trav = sum(np.hypot(r["dx"], r["dy"]) for r in recs if r["conf"] > 0)
                print(f"  {i:4d} frames  t={tick-t0:5.1f}s  VO travel={trav:5.2f} m")
            time.sleep(max(0.0, 1.0 / a.hz - (time.monotonic() - tick)))
    finally:
        c.send_cmd(0.0, 0.0, lamp=0)
    if stop_reason:
        print(f"\n  *** STOPPED EARLY: {stop_reason}")

    with open(f"{a.out}/episode.json", "w") as fh:
        json.dump({"calib": "erc_urban/camera_front.json", "hz": a.hz,
                   "ticks": recs}, fh, indent=1)
    trav = sum(np.hypot(r["dx"], r["dy"]) for r in recs if r["conf"] > 0)
    yaw = sum(np.degrees(r["dyaw"]) for r in recs if r["conf"] > 0)
    print(f"\nsaved {len(recs)} ticks | VO travel {trav:.2f} m | net yaw {yaw:+.0f} deg")
    print(f"  usable VO ticks: {sum(1 for r in recs if r['conf'] > 0)}/{len(recs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
