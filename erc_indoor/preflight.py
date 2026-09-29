"""
Pre-run checks for the indoor lap. Read-only: never moves the rover.

    python -m erc_indoor.preflight

Checks the things that went wrong in testing: wrong bot in .env, a second
headless Chrome logged in to the rover (REMOTE_LOGIN kicks), a stale RTM link,
a low fd limit / stuck sockets in the SDK ("Too many open files"), missing
wheel/gyro telemetry, a dead camera, a run already in progress, and the map
view sets. Exit code 0 = all good.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
import time

import requests

SDK_URL = os.environ.get("SDK_URL", "http://127.0.0.1:8000")
EXPECTED_BOT = os.environ.get("EXPECTED_BOT", "data-stem-cell")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_BATTERY = 40

ok_all = True


def check(name: str, ok: bool, detail: str, fix: str = "") -> None:
    global ok_all
    ok_all &= ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:14s} {detail}")
    if not ok and fix:
        print(f"      -> {fix}")


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    env = os.path.join(ROOT, "earth-rovers-sdk", ".env")
    slug = next((l.split("=", 1)[1].strip() for l in open(env) if l.startswith("BOT_SLUG=")), "")
    check("bot", slug == EXPECTED_BOT, f"BOT_SLUG={slug}",
          f"set BOT_SLUG={EXPECTED_BOT} in earth-rovers-sdk/.env and restart the SDK")

    n_chrome = int(sh("ps aux | grep -c '[G]oogle Chrome --disable-background-networking'") or 0)
    check("sdk browser", n_chrome == 1, f"{n_chrome} headless SDK Chrome(s)",
          "exactly one must run: stop the SDK (Ctrl+C), kill leftovers "
          "(pkill -f 'Google Chrome --disable-background-networking'), start the SDK again")

    runs = sh("pgrep -f 'erc_indoor.run'")
    check("no run active", not runs, "none" if not runs else f"pids {runs}",
          "a lap is already running: stop it first (Ctrl+C in its terminal)")

    try:
        d = requests.get(f"{SDK_URL}/data", timeout=5).json()
    except Exception as e:  # noqa: BLE001
        check("sdk /data", False, f"no answer ({e.__class__.__name__})",
              "start the SDK: cd earth-rovers-sdk && ulimit -n 4096 && "
              "conda run -n rover_env hypercorn main:app --bind 127.0.0.1:8000")
        return 1
    st, age = d.get("rtm_state"), d.get("rtm_age_s")
    check("link", st == "CONNECTED" and age is not None and age < 2.0, f"rtm {st}, data age {age} s",
          "wait ~20 s; if it stays stale, close every FrodoBots browser tab and restart the SDK")
    bat = d.get("battery")
    check("battery", bat is not None and bat >= MIN_BATTERY, f"{bat} %", "charge the rover")
    check("wheels+gyro", bool(d.get("rpms")) and bool(d.get("gyros")),
          f"rpms {'yes' if d.get('rpms') else 'NO'}, gyros {'yes' if d.get('gyros') else 'NO'}",
          "telemetry missing: restart the SDK")

    # telemetry must be updating, not frozen
    ts0 = d.get("timestamp")
    time.sleep(1.5)
    try:
        ts1 = requests.get(f"{SDK_URL}/data", timeout=5).json().get("timestamp")
    except Exception:  # noqa: BLE001
        ts1 = ts0
    check("telemetry live", ts1 != ts0, "timestamp advancing" if ts1 != ts0 else "timestamp frozen",
          "restart the SDK")

    try:
        f = requests.get(f"{SDK_URL}/v2/front", timeout=8).json()
        nbytes = len(base64.b64decode(f.get("front_frame", "")))
    except Exception:  # noqa: BLE001
        nbytes = 0
    check("camera", nbytes > 20000, f"front frame {nbytes // 1024} KB", "restart the SDK")

    pid = sh("pgrep -f 'hypercorn main:app' | head -1")
    worker = sh(f"pgrep -P {pid} | tail -1") if pid else ""
    if worker:
        cw = int(sh(f"lsof -nP -p {worker} 2>/dev/null | grep -c CLOSE_WAIT") or 0)
        check("sdk sockets", cw < 50, f"{cw} stuck (CLOSE_WAIT)",
              "restart the SDK with: ulimit -n 4096 && conda run -n rover_env hypercorn main:app --bind 127.0.0.1:8000")

    rm = os.path.join(ROOT, "erc_indoor", "data", "route_map")
    views = sorted(x for x in os.listdir(rm) if x.startswith("twin"))
    check("map views", "twin_today" in views, " ".join(views) or "none",
          "rebuild: python -m erc_indoor.twin --lap runs/indoor/lap_20260926_083325 --name today")

    free_gb = int(sh(f"df -g '{ROOT}' | awk 'NR==2{{print $4}}'") or 0)
    check("disk", free_gb >= 3, f"{free_gb} GB free", "free some disk (a lap log is ~0.1-1 GB)")

    print("\nALL GOOD — place the rover at the start cone facing down the atrium (windows on the right)."
          if ok_all else "\nFix the FAIL lines above before running.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
