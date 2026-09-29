"""
Optional pre-mission bot check (NOT part of the run; its numbers are for
reading only - the auto-set V_NOM 0.45 flipped the Argentina finals rover,
2026-09-28. Run with the tested V_NOM=0.28.)

    python -m erc_urban.bot_check            # checks + calibration moves
    python -m erc_urban.bot_check --no-move  # link / GPS / checkpoints only

Moves (net ~zero, about 10 s): turn ~40 deg left and back, then 1 s forward and
1 s back. Keep 0.5 m clear in front and behind.

What each check caught on the practice bots, 2026-09-22..25:
  * link: a Chrome left over from an earlier SDK logged into RTM under the same
    account and the two kicked each other off (REMOTE_LOGIN) every few seconds
    (Costa Rica). Fixed automatically: kill stale Chromes, restart the SDK.
  * gyro axis: a Zero had yaw on gyro x, not z (Wuhan inert-eel-code).
  * compass: Machakos read 80 deg off, Costa Rica under-read turns by 20 %.
    Used only if it follows the gyro within 15 deg.
  * speed: 0.45 -> 0.37 m/s on one unit, 1.37 m/s per unit on another; V_NOM is
    scaled so every bot drives at the same real speed.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
import time

import numpy as np

from erc_urban.sdk_client import RoverClient
from erc_urban.streamer import CommandStreamer
from erc_urban.modules.wheel_odom import _integrate, yaw_axis_from_wiggle

SDK = os.environ.get("SDK_URL", "http://localhost:8000")
TARGET_MPS = 0.38        # real speed the planner was tuned at (Wuhan Mini+, V_NOM 0.28)
BASE_V_NOM = 0.28


def _say(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + "[check] " + msg, flush=True)


def _headless_chromes() -> int:
    try:
        out = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    except Exception:
        return -1
    return sum(1 for l in out.splitlines()
               if "pyppeteer" in l and "--headless" in l and "Helper" not in l)


def _restart_sdk() -> None:
    from erc_urban.sdk_guard import SdkGuard
    g = SdkGuard(SDK, manage=True)
    g.restart()
    g.ensure_healthy()


def _rtm_states(c: RoverClient, n: int = 5) -> list:
    out = []
    for _ in range(n):
        try:
            out.append((c.get_data().raw or {}).get("rtm_state"))
        except Exception:
            out.append(None)
        time.sleep(1.0)
    return out


def fresh_sdk() -> None:
    """Start a fresh SDK from the current .env. The SDK reads .env (bot, mission)
    only at startup, and a detached SDK from the previous slot keeps running:
    reusing it would drive the previous site's bot. Kills only what listens on
    the SDK port and the SDK's own headless Chromes."""
    from erc_urban.sdk_guard import SdkGuard
    g = SdkGuard(SDK, manage=True)
    _say(f"starting a fresh SDK from earth-rovers-sdk/.env (log: {g.log_dir}/sdk_*.log)")
    if not g.restart():
        _say("SDK did not come up on its port - check runs/sdk/ log")
        return
    t0 = time.time()
    st = g.status()
    while st not in ("ok", "mission") and time.time() - t0 < 60:
        time.sleep(2.0)
        st = g.status()
    _say(f"SDK up ({st}) after {time.time() - t0:.0f} s")


def run_checks(move: bool = True, fresh: bool = True) -> dict:
    """Checks + calibration. Returns env settings to use (GYRO_AXIS,
    HEADING_K_COMPASS, V_NOM) - only the ones it could measure."""
    if fresh and os.environ.get("BOT_CHECK_FIX", "1") != "0" \
            and os.environ.get("SDK_MANAGE", "1") != "0":
        fresh_sdk()
    c = RoverClient(SDK, timeout=30)
    flags: dict[str, str] = {}

    # 1. Link: one SDK Chrome, RTM connected and staying connected.
    n = _headless_chromes()
    try:
        c.start_mission()                 # /data needs it; keeps progress if already started
    except Exception as e:
        _say(f"start-mission: {e}")
    states = _rtm_states(c)
    if (n > 1 or not all(s == "CONNECTED" for s in states)) \
            and os.environ.get("BOT_CHECK_FIX", "1") != "0":
        _say(f"link unstable ({n} SDK Chromes, rtm {states}) -> restarting the SDK")
        _restart_sdk()
        try:
            c.start_mission()
        except Exception as e:
            _say(f"start-mission after restart: {e}")
        states = _rtm_states(c)
    _say(f"link: rtm {states[-1]} ({sum(s == 'CONNECTED' for s in states)}/5 connected)")

    # 2. GPS, battery, checkpoints.
    d = {}
    for _ in range(20):
        d = c.get_data().raw or {}
        lat, lon = d.get("latitude"), d.get("longitude")
        if lat is not None and abs(lat) <= 90 and abs(lon) <= 180:
            break
        time.sleep(1.0)
    lat, lon = d.get("latitude"), d.get("longitude")
    _say(f"GPS {lat}, {lon}  battery {d.get('battery')}%")
    try:
        cps, latest = c.get_checkpoints()
        msg = f"{len(cps)} checkpoints, {latest} claimed"
        if lat is not None and abs(lat) <= 90:
            nxt = [cp for cp in cps if cp.sequence > latest][:1]
            for cp in nxt:
                dn = (cp.lat - lat) * 110540.0
                de = (cp.lon - lon) * 111320.0 * math.cos(math.radians(lat))
                msg += f"; next CP{cp.sequence} {math.hypot(dn, de):.0f} m at " \
                       f"{math.degrees(math.atan2(de, dn)) % 360:.0f} deg"
        _say(msg)
    except Exception as e:
        _say(f"checkpoints-list failed: {e}")
    acc = np.array([a[:3] for a in d.get("accels") or []], float)
    if len(acc):
        g = acc.mean(0)
        _say(f"gravity on accel {'xyz'[int(np.argmax(np.abs(g)))]} "
             f"(tilt {math.degrees(math.acos(min(1.0, float(np.abs(g).max()) / max(float(np.linalg.norm(g)), 1e-6)))):.0f} deg)")

    if not move:
        return flags

    # 3. Gyro axis + compass (turn left, turn right).
    from erc_indoor.odometry import Odometer
    st = CommandStreamer(c, send=True, hz=10, lamp=0)
    st.halt()
    st.start()
    odo = Odometer(c)
    odo.start()
    try:
        t0 = time.time()
        while not odo.ready and time.time() - t0 < 10:
            time.sleep(0.2)

        def compass():
            v = []
            for _ in range(3):
                y = c.get_data().yaw
                if y is not None:
                    v.append(math.degrees(y) % 360)
            if not v:
                return None
            a = np.radians(v)
            return math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360

        def spin(ang):
            t = time.time()
            st.set(0.0, ang, 1.2)
            time.sleep(1.2)
            st.halt()
            time.sleep(2.0)
            return t, time.time()

        c0 = compass()
        a0, a1 = spin(0.6)
        c1 = compass()
        b0, b1 = spin(-0.6)
        c2 = compass()
        raw = list(odo.raw_gyros)
        # Split by receipt time at the start of the right turn: the 2 s settle
        # covers the ~1 s telemetry lag. (Letting the left window run 1 s past
        # its end swallowed the start of the right turn: an honest compass
        # then looked 20 deg off and was rejected - caught by the fake-SDK e2e.)
        left = [(r[1], r[2], r[3], r[4]) for r in raw if a0 <= r[0] < b0]
        right = [(r[1], r[2], r[3], r[4]) for r in raw if b0 <= r[0] < b1 + 1.0]
        res = yaw_axis_from_wiggle(left, right)
        if res is None:
            flags["GYRO_AXIS"] = "z"
            flags["HEADING_K_COMPASS"] = "0"
            _say("gyro: no clear turn measured (rough ground?) -> z (Mini+), compass off")
        else:
            ax, sign = res[0], res[1]
            name = ("-" if sign < 0 else "") + "xyz"[ax]
            gl, gr = sign * _integrate(left)[ax], sign * _integrate(right)[ax]
            flags["GYRO_AXIS"] = name
            rate = abs(gl) / 1.2
            _say(f"gyro: yaw = {name}; turned {gl:+.0f} / {gr:+.0f} deg (~{rate:.0f} deg/s at 0.6)")
            if None not in (c0, c1, c2):
                dl = (c1 - c0 + 180) % 360 - 180
                dr = (c2 - c1 + 180) % 360 - 180
                err = max(abs(dl + gl), abs(dr + gr))
                ok = err <= 15
                flags["HEADING_K_COMPASS"] = "0.04" if ok else "0"
                _say(f"compass: {c0:.0f} -> {c1:.0f} -> {c2:.0f}, off by {err:.0f} deg -> "
                     f"{'USED' if ok else 'ignored (gyro + GPS track only)'}")
            else:
                flags["HEADING_K_COMPASS"] = "0"
            odo.set_yaw_axis(ax, sign)

        # 4. Speed: 1 s forward, 1 s back at 0.3.
        speeds = []
        for lin in (0.3, -0.3):
            s0 = odo.state().s
            st.set(lin, 0.0, 1.0)
            time.sleep(1.0)
            st.halt()
            time.sleep(1.6)
            speeds.append(abs(odo.state().s - s0))   # metres in ~1 s of command
        v = float(np.mean(speeds))
        if v >= 0.08:
            per_unit = v / 0.3
            v_nom = float(np.clip(TARGET_MPS / per_unit, 0.22, 0.45))
            flags["V_NOM"] = f"{v_nom:.2f}"
            _say(f"speed: 0.3 -> {v:.2f} m/s ({per_unit:.2f} m/s per unit) -> V_NOM {v_nom:.2f} "
                 f"(target {TARGET_MPS} m/s)")
        else:
            flags["V_NOM"] = "0.34"
            _say(f"speed: wheels barely moved ({v:.2f} m in 1 s at 0.3: grass/slope/stiction) "
                 f"-> V_NOM 0.34 for more push")
    finally:
        st.stop(lamp_off=False)
        odo.stop()
    return flags


def main(argv: list[str]) -> int:
    flags = run_checks(move="--no-move" not in argv, fresh="--keep-sdk" not in argv)
    if flags:
        print("\nMeasured (for information only; run with the tested V_NOM=0.28): "
              + " ".join(f"{k}={v}" for k, v in flags.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
