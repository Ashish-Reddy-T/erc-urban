"""
Finals driver: an API vision model drives, the local planner keeps it safe.

    SITE=panama python -m erc_urban.agent                  # run the mission in .env
    SITE=costarica python -m erc_urban.agent --missions mission-1,mission-2,mission-3
    SITE=unlp python -m erc_urban.agent --dry              # decisions only, no motion
    SITE=costarica python -m erc_urban.agent --replay runs/urban/run_X/frames   # offline

The ordinary mission runner (erc_urban.run) runs unchanged in the main thread:
start-mission, heading, obstacle stop, escapes, tilt guard, automatic claims,
SDK recovery, live overlay on :8090. A driver thread looks at the latest frame
every few seconds, asks the model for ONE action (auto / drive / turn / back,
see driver_llm.py), and steers the planner through run._AIM_OVERRIDE /
run._AGENT_BACK until the action is done or something happens. Model errors fall
back to "auto", so a network blip never parks the rover.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import threading
import time

from erc_urban import sites

NEAR_M = 6.0          # closer than this the planner's own final approach takes over
ESCAPE_PATIENCE_S = 12.0   # planner escaping longer than this: the driver takes over
ACTION_TIMEOUT_S = {"auto": 30.0, "drive": 30.0, "turn": 20.0, "back": 12.0, "push": 20.0}


KEYS_FILE = os.path.expanduser("~/.erc_driver_keys")


def _load_keys(path: str = KEYS_FILE) -> None:
    """KEY=VALUE lines (ANTHROPIC_API_KEY, OPENROUTER_API_KEY, GEMINI_API_KEY,
    DRIVER_BACKEND, DRIVER_MODEL) from a file outside the repo; the shell wins."""
    try:
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except OSError:
        pass


def apply_measured(flags: dict, environ, shell: set) -> None:
    """Measured bot values beat site defaults; values set in the shell win."""
    for k, v in flags.items():
        if k not in shell:
            environ[k] = v


def set_mission_slug(env_path: str, slug: str) -> None:
    """Rewrite MISSION_SLUG in the SDK .env (other lines untouched)."""
    lines = open(env_path).read().splitlines()
    if not any(l.startswith("MISSION_SLUG=") for l in lines):
        lines.append(f"MISSION_SLUG={slug}")
    lines = [f"MISSION_SLUG={slug}" if l.startswith("MISSION_SLUG=") else l for l in lines]
    open(env_path, "w").write("\n".join(lines) + "\n")


class _Tee:
    """Everything printed also goes to the run's terminal.log."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        try:
            self.fh.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self.stream.flush()
        try:
            self.fh.flush()
        except Exception:
            pass

    def __getattr__(self, k):
        return getattr(self.stream, k)


def _save_step(out_dir, step, images, status, a, V) -> None:
    """What the model saw (annotated front + map + status) and what it chose."""
    if not out_dir:
        return
    try:
        import base64
        d = os.path.join(out_dir, "steps")
        os.makedirs(d, exist_ok=True)
        for (label, b64), name in zip(images, ("front", "map")):
            with open(os.path.join(d, f"{step:04d}_{name}.jpg"), "wb") as fh:
                fh.write(base64.b64decode(b64))
        with open(os.path.join(d, f"{step:04d}_prompt.txt"), "w") as fh:
            fh.write(status + "\n\n-> " + _describe(a, V) + f" | {a.reason} [{a.latency_s:.1f}s]"
                     + (" FALLBACK" if a.fallback else "") + "\n")
    except Exception as e:
        print(f"[driver] step save failed: {e}", flush=True)


def _log_result(out_dir, rec: dict) -> None:
    if not out_dir:
        return
    try:
        import json
        with open(os.path.join(out_dir, "steps.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _ev_seq(R) -> int:
    """Count of events ever raised. Not len(_EVENTS): the runner keeps only the
    last 50, so after 50 events a length never grows and nothing looks new."""
    return getattr(R, "_EVENT_SEQ", 0) or len(R._EVENTS)


def _events_since(R, seq: int) -> list:
    evs = list(R._EVENTS)
    if evs and len(evs[0]) > 2:
        return [e[1] for e in evs if e[2] > seq]
    return [e[1] for e in evs[seq:]]


def _say(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def _images(R, V, obs, trail):
    """[(label, jpeg b64)] for the model: annotated front + map."""
    fs = obs["fs"]
    front = V.annotate(obs["front"], R.CALIB, fs.drive.clear_at, obs["want_gps"])
    info = R._MISSION_INFO
    route = info.get("route") or []
    gi = obs.get("goal_index", 0)
    mp = V.render_map((obs["lat"], obs["lon"]), obs["yaw"], route, gi, trail,
                      R._OSM.get("payload"))
    return [("FRONT camera:", V._jpeg_b64(front)), ("MAP (north up):", V._jpeg_b64(mp))]


def _status(R, obs, journal, events) -> str:
    route = R._MISSION_INFO.get("route") or []
    gi = obs.get("goal_index", 0)
    rel = obs["want_gps"]
    side = "left" if rel > 0 else "right"
    tilt = obs.get("tilt")
    lines = [
        f"Checkpoint {gi + 1} of {len(route) or '?'}: {obs['rng']:.0f} m away, "
        f"{abs(rel):.0f} deg to the {side}{' (BEHIND)' if abs(rel) > 100 else ''}.",
        f"Robot heading {obs['yaw']:.0f} deg; tilt {tilt:.0f} deg." if tilt is not None
        else f"Robot heading {obs['yaw']:.0f} deg.",
    ]
    if events:
        lines.append("Since last step: " + "; ".join(events[-4:]))
    if journal:
        lines.append("Your recent actions (oldest first):")
        lines += [f"  - {j}" for j in journal]
    lines.append("Choose the next action.")
    return "\n".join(lines)


def _describe(a, V) -> str:
    if a.action == "drive":
        return f"drive #{a.candidate} ({V.CANDIDATES_DEG[a.candidate - 1]:+d}°) {a.distance_m:.1f} m"
    if a.action == "turn":
        return f"turn {a.turn_deg:+d}°"
    if a.action == "back":
        return f"back {a.distance_m:.1f} m"
    if a.action == "push":
        return f"PUSH #{a.candidate} ({V.CANDIDATES_DEG[a.candidate - 1]:+d}°) {a.distance_m:.1f} m"
    return f"auto {a.distance_m:.0f} m"


def driver_loop(R, V, driver, stop: threading.Event, out_dir: str = "") -> None:
    journal: collections.deque = collections.deque(maxlen=6)
    trail: list = []
    n_ev = 0
    step = 0
    esc_since = None
    while not stop.is_set():
        obs = dict(R._OBS)
        now = time.monotonic()
        if not obs or now - obs.get("t", 0) > 3.0 or obs.get("yaw") is None:
            time.sleep(0.5)
            continue
        if not math.isnan(obs["lat"]) and (not trail or math.hypot(
                (obs["lat"] - trail[-1][0]) * 110540,
                (obs["lon"] - trail[-1][1]) * 111320) > 1.0):
            trail.append((obs["lat"], obs["lon"]))
        if obs["rng"] <= NEAR_M:
            R._AIM_OVERRIDE = None          # planner's own final approach
            R._AGENT_PUSH = None
            time.sleep(0.5)
            continue
        # Let the planner's escape run; if it is still escaping after
        # ESCAPE_PATIENCE_S (boxed in, looping), the driver takes over.
        esc_since = (esc_since or now) if obs["escaping"] else None
        if esc_since is not None and now - esc_since < ESCAPE_PATIENCE_S:
            time.sleep(0.5)
            continue
        takeover = esc_since is not None
        new_events = _events_since(R, n_ev)
        n_ev = _ev_seq(R)
        step += 1
        images = _images(R, V, obs, trail)
        status = _status(R, obs, list(journal), new_events)
        a = driver.decide(images, status)
        if driver.errors >= 2:
            a.action, a.distance_m = "auto", 4.0
        _say(f"[driver] step {step} | CP{obs['goal_index'] + 1} {obs['rng']:.0f} m "
             f"{obs['want_gps']:+.0f}° | -> {_describe(a, V)} | {a.reason} [{a.latency_s:.1f}s]"
             f"{' FALLBACK' if a.fallback else ''}")
        _save_step(out_dir, step, images, status, a, V)

        # Execute through the planner.
        if takeover:
            R._AGENT_TAKEOVER = True
            esc_since = None
            time.sleep(0.5)
        obs = dict(R._OBS)
        t0, s0, r0, ev0 = time.monotonic(), obs["travelled"], obs["rng"], _ev_seq(R)
        yaw = obs["yaw"]
        if a.action == "auto":
            R._AIM_OVERRIDE = None
            goal_m = s0 + a.distance_m
        elif a.action == "drive":
            R._AIM_OVERRIDE = {"world_deg": (yaw - V.CANDIDATES_DEG[a.candidate - 1]) % 360.0,
                               "until_m": s0 + a.distance_m, "t_until": t0 + ACTION_TIMEOUT_S["drive"]}
            goal_m = s0 + a.distance_m
        elif a.action == "turn":
            R._AIM_OVERRIDE = {"world_deg": (yaw - a.turn_deg) % 360.0,
                               "until_m": s0 + 0.5, "t_until": t0 + ACTION_TIMEOUT_S["turn"]}
            goal_m = s0 + 0.5
        elif a.action == "push":
            R._AIM_OVERRIDE = None
            R._AGENT_PUSH = {"world_deg": (yaw - V.CANDIDATES_DEG[a.candidate - 1]) % 360.0,
                             "until_m": s0 + a.distance_m, "t_until": t0 + ACTION_TIMEOUT_S["push"]}
            goal_m = s0 + a.distance_m
        else:
            R._AGENT_BACK = a.distance_m
            goal_m = None
        timeout = ACTION_TIMEOUT_S[a.action]
        why = "timeout"
        saw_escape = False
        while not stop.is_set() and time.monotonic() - t0 < timeout:
            time.sleep(0.3)
            o = R._OBS
            if _ev_seq(R) > ev0:
                last = R._EVENTS[-1][1]
                if last.startswith("claimed") or (last.startswith("escape") and a.action != "back"):
                    why = last
                    break
            if a.action == "back":
                saw_escape |= bool(o.get("escaping"))
                if saw_escape and not o.get("escaping"):
                    why = "backed off"
                    break
            elif a.action == "push" and R._AGENT_PUSH is None:
                why = "done" if o.get("travelled", 0.0) >= goal_m - 0.05 else "push stopped (guard)"
                break
            elif goal_m is not None and o.get("travelled", 0.0) >= goal_m and (
                    a.action == "auto" or R._AIM_OVERRIDE is None):
                why = "done"
                break
            if o.get("rng", 99) <= NEAR_M:
                why = "near checkpoint"
                break
        if a.action != "auto" and R._AIM_OVERRIDE is not None and why != "done":
            R._AIM_OVERRIDE = None
        R._AGENT_PUSH = None
        o = R._OBS
        moved = o.get("travelled", s0) - s0
        journal.append(f"{_describe(a, V)} -> {why}; moved {moved:.1f} m, "
                       f"checkpoint {r0:.0f} -> {o.get('rng', r0):.0f} m")
        _say(f"         result: {why}, moved {moved:.1f} m, CP {r0:.0f} -> {o.get('rng', r0):.0f} m")
        _log_result(out_dir, {"step": step, "time": time.strftime("%H:%M:%S"),
                              "cp": obs["goal_index"] + 1, "action": _describe(a, V),
                              "reason": a.reason, "latency_s": round(a.latency_s, 2),
                              "fallback": a.fallback, "result": why, "moved_m": round(moved, 2),
                              "rng_before": round(r0, 1), "rng_after": round(o.get("rng", r0), 1),
                              "lat": obs["lat"], "lon": obs["lon"], "yaw": obs["yaw"]})


def replay(frames_dir: str, notes: str, n: int, goal_m: float, goal_rel: float) -> None:
    """Offline: model decisions on saved frames (no rover). Saves the views it saw."""
    import cv2
    import numpy as np

    import erc_urban.run as R
    from erc_urban.driver_llm import Driver
    from erc_urban.modules import vlm_nav as V
    from erc_urban.modules.freespace import FreeSpaceUrban

    driver = Driver(notes)
    _say(f"[replay] model {driver.name}")
    sem = R._semantics()
    fsu = FreeSpaceUrban(R.CALIB, seg_hw=R.FAN_SEG_HW, min_obstacle_h_m=R.FAN_MIN_OBS_H)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    step = max(1, len(files) // n)
    out = os.path.join("runs", "agent_replay", time.strftime("%H%M%S"))
    os.makedirs(out, exist_ok=True)
    lat = []
    for f in files[::step][:n]:
        rgb = cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(frames_dir, f)), cv2.COLOR_BGR2RGB),
                         (1024, 576))
        lab = sem.labels(rgb, R.FAN_SEG_HW)
        g, gr = R._fan_masks(sem, lab)
        fs = fsu.estimate(g, gr if R.FAN_TERRAIN == "urban" else None)
        front = V.annotate(rgb, R.CALIB, fs.drive.clear_at, goal_rel)
        mp = V.render_map((0.0, 0.0), 0.0, [(goal_m * math.cos(math.radians(goal_rel)) / 110540,
                                            goal_m * math.sin(math.radians(-goal_rel)) / 111320)],
                          0, [], None)
        text = (f"Checkpoint 2 of 5: {goal_m:.0f} m away, {abs(goal_rel):.0f} deg to the "
                f"{'left' if goal_rel > 0 else 'right'}.\nRobot heading 0 deg.\nChoose the next action.")
        a = driver.decide([("FRONT camera:", V._jpeg_b64(front)), ("MAP (north up):", V._jpeg_b64(mp))], text)
        lat.append(a.latency_s)
        cv2.imwrite(os.path.join(out, f), front)
        _say(f"[replay] {f}: {_describe(a, V)} | {a.reason} [{a.latency_s:.1f}s]"
             f"{' FALLBACK' if a.fallback else ''}")
    if lat:
        _say(f"[replay] {len(lat)} decisions, median {np.median(lat):.1f}s, max {max(lat):.1f}s -> {out}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions", default="", help="comma list: run these missions in turn")
    ap.add_argument("--dry", action="store_true", help="SEND_CONTROL=0: decide but do not move")
    ap.add_argument("--skip-checks", action="store_true", help="skip the pre-run bot check")
    ap.add_argument("--keep-sdk", action="store_true",
                    help="use the SDK already running (default: start a fresh one from .env)")
    ap.add_argument("--replay", default="", help="frames dir: offline decisions only")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--goal-m", type=float, default=25.0)
    ap.add_argument("--goal-rel", type=float, default=30.0)
    args = ap.parse_args(argv)

    _load_keys()
    site = os.environ.get("SITE", "")
    if not site:
        raise SystemExit("set SITE=" + "|".join(sites.SITES))
    if site.lower() not in sites.SITES:
        raise SystemExit(f"unknown SITE={site!r}; one of {', '.join(sites.SITES)}")
    if not args.replay:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("runs", "agent", f"{site.lower()}_{stamp}")
        os.makedirs(out_dir, exist_ok=True)
        tlog = open(os.path.join(out_dir, "terminal.log"), "a", buffering=1)
        sys.stdout = _Tee(sys.stdout, tlog)
        sys.stderr = _Tee(sys.stderr, tlog)
        _say(f"[driver] saving this run to {out_dir}/ (terminal.log, ticks.log, steps/, steps.jsonl)")
    # Measure this bot before anything reads the settings (nobody is at the
    # keyboard during a slot): link, GPS, gyro axis, compass, real speed.
    # Measured values beat the site defaults; anything set in the shell wins.
    if not args.replay and not args.skip_checks:
        from erc_urban import bot_check
        apply_measured(bot_check.run_checks(move=not args.dry, fresh=not args.keep_sdk),
                       os.environ, set(os.environ))
    notes = sites.apply(site, os.environ)
    # The runner stops the mission after 3 boxed-in escapes ("needs walker
    # assist"); with a driver on board it keeps going and the driver decides.
    os.environ.setdefault("ESC_BOXED_MAX", "1000000")
    os.environ.setdefault("MAX_ESCAPES", "1000000")
    if args.dry:
        os.environ["SEND_CONTROL"] = "0"
    if args.replay:
        out_dir = ""
    else:
        os.environ.setdefault("TICK_LOG", os.path.join(out_dir, "ticks.log"))

    if args.replay:
        replay(args.replay, notes, args.n, args.goal_m, args.goal_rel)
        return 0

    import erc_urban.run as R                # reads the env set above
    from erc_urban.driver_llm import Driver
    from erc_urban.modules import vlm_nav as V
    from erc_urban.sdk_guard import SdkGuard

    driver = Driver(notes)
    _say(f"[driver] site={site} model={driver.name} terrain={os.environ['FAN_TERRAIN']} "
         f"gyro={os.environ['GYRO_AXIS']} compass_k={os.environ['HEADING_K_COMPASS']} "
         f"V_NOM={os.environ['V_NOM']} claim<={os.environ.get('CLAIM_NEAR_M', '15')} m "
         f"(planner log: {os.environ['TICK_LOG']})")
    stop = threading.Event()
    threading.Thread(target=driver_loop, args=(R, V, driver, stop, out_dir), daemon=True).start()

    missions = [m.strip() for m in args.missions.split(",") if m.strip()] or [None]
    env_path = os.path.join("earth-rovers-sdk", ".env")
    for k, m in enumerate(missions):
        if m is not None:
            set_mission_slug(env_path, m)
            if k > 0 or R.SDK_MANAGE:
                SdkGuard(os.environ.get("SDK_URL", "http://localhost:8000"), manage=True).restart()
            _say(f"[driver] ===== {m} =====")
        R._OBS.clear()
        R._EVENTS.clear()
        R._AIM_OVERRIDE = None
        R._AGENT_PUSH = None
        try:
            R.main()
        except SystemExit as e:
            _say(f"[driver] run ended ({e})")
    stop.set()
    _say(f"[driver] finished. Everything is in {out_dir}/ ; planner frames/overlays in "
         f"{getattr(R, '_RUN_DIR', '') or 'runs/urban/'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
