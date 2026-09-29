"""
Teach-lap recorder for the Mexico (Tepetlaoxtoc) lane, copied from
erc_indoor.record (unchanged) plus a live GPS range line to every mission
checkpoint, so CP marks go in at the right place.

    python -m erc_mexico.record                      # SDK already has a mission
    python -m erc_mexico.record --start-mission      # call /start-mission first

  Status line (1 Hz):  GPS fix, compass, then each checkpoint's distance and
  direction; "IN" = within the 15 m claim radius. Marks (1-4) also log the GPS
  position and every checkpoint distance.

  w .......... forward            s .......... reverse
  a / d ...... spin left / right  q / e ...... arc forward-left / forward-right
  space ...... STOP               + / - ...... speed up / down
  0 .......... mark START/FINISH cone (orange) as you pass it
  1-4 ........ mark CP1..CP4 cone (red, blue, green, yellow) as you pass it
  m .......... free marker (anything worth noting)
  x .......... quit (sends stop)

WHY THIS EXISTS
---------------
The first teach lap (erc_indoor/data/indoor_track_recording.h5) has an EMPTY
`controls` table and its frame clock runs +4 h ahead of the IMU clock (the SDK
stamps frames with `datetime.utcnow().timestamp()`, which is local-as-UTC).
This recorder stamps EVERYTHING with time.time() at receipt, so frames,
telemetry, commands and cone marks share one clock.

DRIVING MODEL
-------------
A raw terminal gets key-down only. Each keypress sets a command that expires
after HOLD_S unless the key repeats, so holding a key drives continuously and
letting go stops within HOLD_S. The command streams at CTRL_HZ because the
Mini+ stops when commands stop arriving. Positive angular = LEFT (measured).

OUTPUT
------
runs/indoor/<stamp>/
  frames/NNNNNN.<ext>   raw bytes from /v2/front
  frames.jsonl          {"i", "t", "file", "sdk_ts"}
  telemetry.jsonl       {"t", "data": raw /data}
  controls.jsonl        {"t", "linear", "angular"} every streamed command
  events.jsonl          {"t", "event", "frame_i"}  cone / free marks
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import select
import sys
import termios
import threading
import time
import tty

import requests

BASE = "http://localhost:8000"
CTRL_HZ = 10.0
HOLD_S = 0.6          # > terminal auto-repeat delay (~0.5 s) so a held key is seamless
SPEEDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
TURN = 0.5            # spin-in-place angular
ARC = 0.35            # angular added while arcing
EVENT_NAMES = {"0": "START_FINISH", "1": "CP1", "2": "CP2", "3": "CP3", "4": "CP4", "m": "MARK"}


class Log:
    def __init__(self, out: str):
        self.dir = out
        self.fdir = os.path.join(out, "frames")
        os.makedirs(self.fdir, exist_ok=True)
        self._lock = threading.Lock()
        self.files = {k: open(os.path.join(out, f"{k}.jsonl"), "a")
                      for k in ("frames", "telemetry", "controls", "events")}
        self.n_frames = 0

    def write(self, kind: str, rec: dict) -> None:
        with self._lock:
            self.files[kind].write(json.dumps(rec) + "\n")
            self.files[kind].flush()

    def close(self) -> None:
        for f in self.files.values():
            f.close()


class Rover:
    def __init__(self, log: Log):
        self.log = log
        self.lin = 0.0
        self.ang = 0.0
        self.expires = 0.0
        self.running = True
        self.latest: dict = {}
        self.cps: list = []          # [(seq, lat, lon)]
        self._lock = threading.Lock()

    def set(self, lin: float, ang: float, hold: float = HOLD_S) -> None:
        with self._lock:
            self.lin, self.ang, self.expires = lin, ang, time.time() + hold

    def halt(self) -> None:
        self.set(0.0, 0.0, 0.0)
        self._post(0.0, 0.0)

    def _post(self, lin: float, ang: float) -> None:
        try:
            requests.post(f"{BASE}/control",
                          json={"command": {"linear": lin, "angular": ang}}, timeout=2)
        except requests.RequestException:
            pass
        self.log.write("controls", {"t": time.time(), "linear": lin, "angular": ang})

    def control_loop(self) -> None:
        while self.running:
            t = time.time()
            with self._lock:
                lin, ang = (self.lin, self.ang) if t < self.expires else (0.0, 0.0)
            self._post(lin, ang)
            time.sleep(max(0.0, 1.0 / CTRL_HZ - (time.time() - t)))

    def frame_loop(self) -> None:
        while self.running:
            try:
                r = requests.get(f"{BASE}/v2/front", timeout=5)
                t = time.time()
                if r.status_code == 200:
                    d = r.json()
                    raw = base64.b64decode(d["front_frame"])
                    ext = "jpg" if raw[:2] == b"\xff\xd8" else "png"
                    i = self.log.n_frames
                    name = f"{i:06d}.{ext}"
                    with open(os.path.join(self.log.fdir, name), "wb") as fh:
                        fh.write(raw)
                    self.log.n_frames += 1
                    self.log.write("frames", {"i": i, "t": t, "file": name,
                                              "sdk_ts": d.get("timestamp")})
                    continue
            except (requests.RequestException, ValueError, KeyError):
                pass
            time.sleep(0.2)

    def telemetry_loop(self) -> None:
        while self.running:
            t0 = time.time()
            try:
                r = requests.get(f"{BASE}/data", timeout=5)
                if r.status_code == 200:
                    d = r.json()
                    if isinstance(d, dict):
                        self.latest = d
                    self.log.write("telemetry", {"t": time.time(), "data": d})
            except (requests.RequestException, ValueError):
                pass
            time.sleep(max(0.0, 0.2 - (time.time() - t0)))


CLAIM_M = 15.0


def fetch_checkpoints() -> list:
    """[(seq, lat, lon)] from the SDK's mission checkpoint list."""
    for method in (requests.get, requests.post):
        try:
            r = method(f"{BASE}/checkpoints-list", timeout=10)
            if r.status_code == 200:
                d = r.json()
                return [(int(c["sequence"]), float(c["latitude"]), float(c["longitude"]))
                        for c in d.get("checkpoints_list", [])]
        except (requests.RequestException, ValueError, KeyError):
            pass
    return []


def ranges(d: dict, cps: list) -> list:
    """[(seq, dist_m, bearing_deg, rel_deg or None)] from a /data payload."""
    lat, lon = d.get("latitude"), d.get("longitude")
    if lat is None or lon is None or abs(lat) > 90:
        return []
    c = math.cos(math.radians(lat))
    comp = d.get("orientation")
    out = []
    for seq, a, b in cps:
        n, e = (a - lat) * 110540.0, (b - lon) * 111320.0 * c
        brg = math.degrees(math.atan2(e, n)) % 360.0
        rel = ((brg - float(comp) + 180.0) % 360.0 - 180.0) if comp is not None else None
        out.append((seq, math.hypot(n, e), brg, rel))
    return out


def status_loop(rov: "Rover") -> None:
    while rov.running:
        rs = ranges(rov.latest, rov.cps)
        if rs:
            parts = []
            for seq, dist, brg, rel in rs:
                side = "" if rel is None else (f" R{rel:.0f}" if rel >= 0 else f" L{-rel:.0f}")
                parts.append(f"CP{seq} {dist:4.0f}m{side}{' IN' if dist <= CLAIM_M else ''}")
            comp = rov.latest.get("orientation")
            line = f"[compass {comp}] " + " | ".join(parts)
        else:
            line = "waiting for GPS fix..."
        sys.stdout.write("\r" + line[:150].ljust(150))
        sys.stdout.flush()
        time.sleep(1.0)


def preflight(start_mission: bool) -> None:
    if start_mission:
        r = requests.post(f"{BASE}/start-mission", timeout=60)
        print("start-mission:", r.status_code, r.text[:200])
    r = requests.get(f"{BASE}/data", timeout=60)
    if r.status_code == 400:
        sys.exit("SDK says a mission must be started first. Re-run with --start-mission.")
    r.raise_for_status()
    r = requests.get(f"{BASE}/v2/front", timeout=60)
    r.raise_for_status()
    print("SDK OK: /data and /v2/front both answer.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--start-mission", action="store_true")
    args = ap.parse_args()

    preflight(args.start_mission)
    out = args.out or os.path.join("runs", "mexico", time.strftime("lap_%Y%m%d_%H%M%S"))
    log = Log(out)
    rov = Rover(log)
    rov.cps = fetch_checkpoints()
    print("checkpoints:", ", ".join(f"CP{s} ({a:.6f}, {b:.6f})" for s, a, b in rov.cps) or "NONE (no range line)")
    threads = [threading.Thread(target=f, daemon=True)
               for f in (rov.control_loop, rov.frame_loop, rov.telemetry_loop)]
    threads.append(threading.Thread(target=status_loop, args=(rov,), daemon=True))
    for th in threads:
        th.start()

    speed_i = 1
    print(__doc__.split("WHY THIS EXISTS")[0])
    print(f"Recording to {out}   speed={SPEEDS[speed_i]}")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                continue
            k = sys.stdin.read(1)
            v = SPEEDS[speed_i]
            if k == "w":
                rov.set(v, 0.0)
            elif k == "s":
                rov.set(-v, 0.0)
            elif k == "a":
                rov.set(0.0, TURN)
            elif k == "d":
                rov.set(0.0, -TURN)
            elif k == "q":
                rov.set(v, ARC)
            elif k == "e":
                rov.set(v, -ARC)
            elif k == " ":
                rov.halt()
            elif k in "+=":
                speed_i = min(speed_i + 1, len(SPEEDS) - 1)
                print(f"\rspeed={SPEEDS[speed_i]}      ")
            elif k in "-_":
                speed_i = max(speed_i - 1, 0)
                print(f"\rspeed={SPEEDS[speed_i]}      ")
            elif k in EVENT_NAMES:
                ev = EVENT_NAMES[k]
                d = rov.latest
                rs = ranges(d, rov.cps)
                log.write("events", {"t": time.time(), "event": ev,
                                     "frame_i": log.n_frames - 1,
                                     "lat": d.get("latitude"), "lon": d.get("longitude"),
                                     "compass": d.get("orientation"),
                                     "cp_dist_m": {f"CP{sq}": round(dd, 1) for sq, dd, _, _ in rs}})
                near = " ".join(f"CP{sq}={dd:.1f}m" for sq, dd, _, _ in rs)
                print(f"\n*** marked {ev} at frame {log.n_frames - 1}  ({near})")
            elif k == "x":
                break
    finally:
        rov.running = False
        rov.halt()
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        time.sleep(0.3)
        log.close()
        print(f"\nSaved {log.n_frames} frames to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
