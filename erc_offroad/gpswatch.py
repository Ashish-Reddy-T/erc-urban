"""
Live GPS proximity pinger for a human-driven (WASD) run.

Read-only. Sends NO control commands, ever -- safe to run alongside teleop.

    python -m erc_offroad.gpswatch
    python -m erc_offroad.gpswatch --quiet          # no voice, terminal only
    python -m erc_offroad.gpswatch --reach-m 15

WHY THIS EXISTS
---------------
The rover currently reports `latitude=1000, longitude=1000` -- the SDK's no-fix
sentinel -- because the rock garden is indoors. The moment the rover reaches
open sky the fix should appear, and the driver needs to know two things without
taking their eyes off the video: that GPS came alive, and how far the next
checkpoint is.

So this speaks out loud (macOS `say`) rather than only printing. It announces
GPS acquisition and loss, counts down through range bands, and calls out the
checkpoint as reached.

BEARINGS ARE GIVEN AS CLOCK POSITIONS
-------------------------------------
"Checkpoint at 2 o'clock, 40 metres" is usable while driving; "bearing 63.4
degrees relative" is not. Bearing is relative to the rover's own heading, so it
already accounts for which way the rover is pointing.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import urllib.request

URL = "http://localhost:8000"
R_EARTH = 6_371_000.0
# The SDK reports this exact pair when it has no fix.
NO_FIX = 1000.0


def _get(path: str, timeout: float = 20.0):
    with urllib.request.urlopen(f"{URL}{path}", timeout=timeout) as r:
        return json.load(r)


def say(text: str, quiet: bool) -> None:
    print(f">>> {text}")
    if quiet:
        return
    try:
        subprocess.Popen(["say", "-r", "200", text],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        # A missing `say` must never take down the monitor.
        print("\a", end="", flush=True)


def valid_fix(lat, lon) -> bool:
    if lat is None or lon is None:
        return False
    try:
        lat = float(lat); lon = float(lon)
    except (TypeError, ValueError):
        return False
    if math.isnan(lat) or math.isnan(lon):
        return False
    if abs(lat - NO_FIX) < 1e-6 and abs(lon - NO_FIX) < 1e-6:
        return False
    return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0


def geo(lat1, lon1, lat2, lon2) -> tuple[float, float]:
    """Returns (bearing_deg_CW_from_north, distance_m)."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    brg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    d = 2 * R_EARTH * math.asin(math.sqrt(
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2))
    return brg, d


def clock_position(rel_deg: float) -> str:
    """Relative bearing -> clock face. 0 deg = 12 o'clock."""
    h = int(round((rel_deg % 360.0) / 30.0)) or 12
    return f"{h} o'clock"


def load_checkpoints() -> tuple[list[dict], int]:
    d = _get("/checkpoints-list")
    cps = []
    for c in d.get("checkpoints_list") or []:
        cps.append({"id": c.get("id"), "seq": c.get("sequence"),
                    "lat": float(c["latitude"]), "lon": float(c["longitude"])})
    cps.sort(key=lambda c: c["seq"])
    return cps, int(d.get("latest_scanned_checkpoint") or 0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="erc_offroad.gpswatch")
    ap.add_argument("--reach-m", type=float, default=15.0,
                    help="checkpoint tolerance (competition uses ~15 m)")
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--quiet", action="store_true", help="no voice")
    a = ap.parse_args(argv)

    try:
        cps, latest = load_checkpoints()
    except Exception as e:
        print(f"could not read checkpoints: {e}")
        return 1
    if not cps:
        print("no checkpoints in this mission")
        return 1

    print(f"{len(cps)} checkpoints, latest_scanned={latest}")
    for c in cps:
        print(f"  seq {c['seq']}  id {c['id']}  {c['lat']:.7f}, {c['lon']:.7f}")
    # CP1 and CP3 are identical on this mission -- an out-and-back.
    idx = max(0, min(latest, len(cps) - 1))
    print(f"\ntargeting seq {cps[idx]['seq']} first. Ctrl-C to stop.")
    print("read-only: this sends NO control commands.\n")

    had_fix = False
    announced: set[str] = set()
    bands = [100.0, 50.0, 30.0, 20.0]
    last_print = 0.0

    while True:
        try:
            d = _get("/data")
        except Exception:
            time.sleep(a.period)
            continue

        lat, lon = d.get("latitude"), d.get("longitude")
        yaw = d.get("orientation")
        ok = valid_fix(lat, lon)

        if ok and not had_fix:
            had_fix = True
            announced.clear()
            say("GPS acquired", a.quiet)
        elif not ok and had_fix:
            had_fix = False
            say("GPS lost", a.quiet)

        if not ok:
            if time.time() - last_print > 5.0:
                print(f"  no fix (lat/lon {lat},{lon}  gps_signal "
                      f"{d.get('gps_signal')})")
                last_print = time.time()
            time.sleep(a.period)
            continue

        tgt = cps[idx]
        brg, dist = geo(float(lat), float(lon), tgt["lat"], tgt["lon"])
        rel = brg - float(yaw) if yaw is not None else brg
        rel = (rel + 180.0) % 360.0 - 180.0

        print(f"  seq {tgt['seq']}  {dist:7.1f} m  {clock_position(rel)} "
              f"(rel {rel:+.0f}deg)  fix {float(lat):.6f},{float(lon):.6f}")

        if dist <= a.reach_m:
            key = f"reach{idx}"
            if key not in announced:
                announced.add(key)
                say(f"Checkpoint {tgt['seq']} reached. {int(dist)} metres.",
                    a.quiet)
                print("\n    *** POST /checkpoint-reached NOW ***\n")
                if idx < len(cps) - 1:
                    idx += 1
                    announced.clear()
                    say(f"Next checkpoint {cps[idx]['seq']}", a.quiet)
                else:
                    say("Final checkpoint. Mission complete.", a.quiet)
                    return 0
        else:
            for b in bands:
                key = f"{idx}@{b}"
                if dist <= b and key not in announced:
                    announced.add(key)
                    say(f"{int(b)} metres. {clock_position(rel)}.", a.quiet)
                    break

        time.sleep(a.period)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")
