"""
Mexico finals runner (Tepetlaoxtoc, 2026-09-29 3:00 PM): a lean lane follower.

    python -m erc_urban.run_mexico 2>&1 | tee runs/mexico_m.log

The mission is one straight dirt lane between stacks of dry grass: CP1 at the
start, then CP2..CP4 every 45-56 m up the lane (bearing 35-40 deg), ~150 m in
all. The general runner (erc_urban.run) lost the trial run (2026-09-28): its
heading bootstrap and route aims fought over which way to go (16 left/right
flips, 5 m in 6 min), reverse-first escapes pushed it into the grass, and depth
+ planning held it to ~1.5 Hz. This runner does only what the lane needs:

  1. Frames and telemetry are fetched by background threads, so a tick never
     waits on the SDK. Segmentation only (no depth model): ~3-5 Hz.
  2. Heading = compass (it matched the gyro within 3 deg median on this bot).
     The lane bearing is the current leg (previous CP -> next CP).
  3. Facing more than 60 deg off the lane (it starts facing the wrong way):
     one committed turn in place, one direction, until within 20 deg.
  4. Otherwise steer along the longest rover-wide ray over the lane mask
     (erc_urban/modules/lane.py), nudged toward the lane bearing.
  5. Blocked (no ray > 0.6 m): stop, small turn toward the side with more lane.
     A short reverse only when the wheels are genuinely stuck, at most once per
     10 s. Never reverse-first.
  6. Claim each checkpoint as soon as it is within 15 m (straight lane: an early
     claim cannot set up a bad turn). A rejected claim keeps driving and
     retries 2 m closer or after 5 s.

The IMU accelerometer on the trial bot read 1.37 / -1.97 g at rest, so tilt is
not used. Ctrl-C stops the rover.

Env: SDK_URL, V_LANE (0.32), SEG_SIZE (512), SEND_CONTROL (1), OVERLAY_PORT
(8090), SAVE_EVERY (2 ticks), CLAIM_M (15).
"""
from __future__ import annotations

import json
import math
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

from erc_urban.calib import CameraCalibration
from erc_urban.modules.lane import LANE_NAMES, LaneFinder
from erc_urban.modules.semantics import SemanticTraversability, _match
from erc_urban.sdk_client import RoverClient
from erc_urban.streamer import CommandStreamer

SDK_URL = os.environ.get("SDK_URL", "http://localhost:8000")
V_LANE = float(os.environ.get("V_LANE", "0.32"))
V_MIN = 0.2
SEG_SIZE = int(os.environ.get("SEG_SIZE", "512"))   # 384 lost the far lane behind a lens drop (trial 000016-68)
SEND = os.environ.get("SEND_CONTROL", "1") != "0"
OVERLAY_PORT = int(os.environ.get("OVERLAY_PORT", "8090"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "2"))
CLAIM_M = float(os.environ.get("CLAIM_M", "15"))
TICK_S = 0.2                 # upper bound 5 Hz
SEG_HW = (144, 256)
TURN_ENTER_DEG, TURN_EXIT_DEG = 60.0, 20.0
TURN_ENTER_LANE_DEG = 100.0  # with the lane clearly in view, only turn round if really facing away
LANE_SEEN_M = 2.0
TURN_ANG = 0.45
MAX_ANG = 0.4
STEER_K = MAX_ANG / 30.0     # 30 deg of steer -> full angular
ANG_MIN = 0.12               # below ~0.1 a skid-steer rover does not turn at all
PURSUIT_M = float(os.environ.get("PURSUIT_M", "15"))   # prior aims this far up the CP line
BLOCK_M = 0.6
BLIND_FRAC = 0.03            # lane pixels in the lower image below this = lane-blind frame
BLIND_COAST_S = 1.0          # keep the last lane command this long through blind frames
BLIND_SEARCH_S = 3.0         # then stop; after this, turn slowly toward the lane bearing
STALE_S = 3.0
STUCK_S = 5.0
CAL = CameraCalibration.load_or_default(
    os.path.join(os.path.dirname(__file__), "camera_front.json"), 1024, 576)


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def geo(lat: float, lon: float, lat2: float, lon2: float) -> tuple[float, float]:
    """(distance m, bearing deg CW from north) from (lat, lon) to (lat2, lon2)."""
    n = (lat2 - lat) * 110540.0
    e = (lon2 - lon) * 111320.0 * math.cos(math.radians(lat))
    return math.hypot(n, e), math.degrees(math.atan2(e, n)) % 360.0


def pursuit_bearing(lat: float, lon: float, a, b, ahead_m: float) -> float:
    """Bearing to the point `ahead_m` beyond the rover's projection on the line
    a -> b (clamped at b): holds the CP line against drift, and turns back to b
    after an overshoot. a, b: objects with .lat/.lon."""
    c = math.cos(math.radians(lat))
    ax, ay = (a.lon - lon) * 111320.0 * c, (a.lat - lat) * 110540.0     # rover at origin (E, N)
    bx, by = (b.lon - lon) * 111320.0 * c, (b.lat - lat) * 110540.0
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return math.degrees(math.atan2(bx, by)) % 360.0
    s = (-ax * dx - ay * dy) / L                 # rover's along-track position from a
    s = min(max(s + ahead_m, 0.0), L)
    px, py = ax + dx * s / L, ay + dy * s / L
    return math.degrees(math.atan2(px, py)) % 360.0


def rel_left(heading: float, bearing: float) -> float:
    """Angle to `bearing` in the rover frame, + = left."""
    return wrap(heading - bearing)


class Feed:
    """Newest front frame and telemetry, each fetched in its own thread."""

    def __init__(self, client: RoverClient):
        self.c = client
        self.frame = None
        self.frame_t = 0.0
        self.frame_n = 0
        self.tel = None
        self.tel_t = 0.0
        self.running = True
        for f in (self._frames, self._data):
            threading.Thread(target=f, daemon=True).start()

    def _frames(self):
        while self.running:
            try:
                front, _ = self.c.get_frames()
                if front is not None:
                    self.frame, self.frame_t = front, time.time()
                    self.frame_n += 1
                    continue
            except Exception:
                pass
            time.sleep(0.2)

    def _data(self):
        while self.running:
            try:
                t = self.c.get_data()
                if t.raw:
                    self.tel, self.tel_t = t, time.time()
            except Exception:
                pass
            time.sleep(0.1)


def wheels_turning(tel) -> bool:
    rp = (tel.raw or {}).get("rpms") or []
    vals = [abs(x) for r in rp[-3:] if isinstance(r, list) for x in r[:4]]
    return bool(vals) and max(vals) > 3.0


class Mexico:
    def __init__(self, client: RoverClient):
        self.c = client
        self.stream = CommandStreamer(client, send=SEND, hz=10)
        self.feed = Feed(client)
        self.sem = SemanticTraversability(device="auto", size=SEG_SIZE)
        self.ids = sorted(_match(self.sem.model.config.id2label, LANE_NAMES))
        self.lane = LaneFinder(CAL, hw=SEG_HW)
        self.turning = 0.0            # committed turn direction (+1 left / -1 right / 0)
        self.fwd_since = None
        self.last_rev = 0.0
        self.hold = None              # (rng, t) of a rejected claim
        self.n = 0
        stamp = time.strftime("run_%Y%m%d_%H%M%S")
        self.dir = os.path.join("runs", "mexico", stamp)
        os.makedirs(os.path.join(self.dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "overlay"), exist_ok=True)
        self.ticks = open(os.path.join(self.dir, "ticks.jsonl"), "a", buffering=1)
        self.overlay = None
        try:
            from erc_indoor.overlay import OverlayServer
            for port in range(OVERLAY_PORT, OVERLAY_PORT + 6):
                try:
                    self.overlay = OverlayServer(port)
                    print(f"LIVE VIEW  http://localhost:{port}/stream.mjpg", flush=True)
                    break
                except OSError:
                    continue
        except Exception as e:
            print(f"overlay off ({e})")
        print(f"saving to {self.dir}/  (frames, overlay, ticks.jsonl)", flush=True)

    # ---- motion ----------------------------------------------------------
    def cmd(self, lin: float, ang: float, hold: float = 0.6) -> None:
        self.stream.set(lin, ang, hold)

    def halt(self) -> None:
        self.stream.halt()

    # ---- one tick ----------------------------------------------------------
    def decide(self, e, prior: float, tel, now: float) -> tuple[float, float, str]:
        """(lin, ang, mode) from the lane estimate and the lane prior (+ left)."""
        # 1. Committed turn when facing away from the lane.
        # A lane bending away from the CP line is not "facing away": with the
        # lane clearly in view, only a near-reversal starts a turn (sim_lane,
        # 4 m bends: a 60 deg rule turned it into the grass mid-run, 143 s).
        lane_seen = e.steer_deg is not None and e.best_m >= LANE_SEEN_M
        enter = TURN_ENTER_LANE_DEG if lane_seen else TURN_ENTER_DEG
        if self.turning == 0.0 and abs(prior) > enter:
            self.turning = 1.0 if prior > 0 else -1.0
        if self.turning != 0.0:
            if abs(prior) < TURN_EXIT_DEG:
                self.turning = 0.0
            else:
                self.fwd_since = None
                return 0.0, self.turning * TURN_ANG, f"TURN {'left' if self.turning > 0 else 'right'} to lane ({prior:+.0f})"
        # 2. Lane-blind frame (lens drop, glare, a burst of mislabels): not an
        #    obstacle. Coast on the last lane command, then stop, then search.
        #    Treating it as "blocked" turned the rover in place on every drop
        #    (sim_lane: 30 % blind frames -> 216 s with wheels in the grass).
        last = getattr(self, "last_lane", None)
        if getattr(e, "ground_frac", 1.0) < BLIND_FRAC:
            self.fwd_since = None
            blind_for = now - (getattr(self, "seen_t", None) or now)
            if last is not None and now - last[2] < BLIND_COAST_S:
                return 0.7 * last[0], last[1], "BLIND coast"
            if blind_for < BLIND_SEARCH_S:
                return 0.0, 0.0, "BLIND hold"
            return 0.0, (0.3 if prior >= 0 else -0.3), "BLIND search toward lane"
        self.seen_t = now
        # 3. Blocked: stop and turn a little toward more lane (never reverse first).
        if e.steer_deg is None or e.best_m < BLOCK_M:
            self.fwd_since = None
            if e.left_m > e.right_m + 0.2:
                side = 1.0
            elif e.right_m > e.left_m + 0.2:
                side = -1.0
            else:
                side = 1.0 if prior >= 0 else -1.0
            return 0.0, side * 0.35, f"BLOCKED (best {e.best_m:.1f}m) turn {'left' if side > 0 else 'right'}"
        # 4. Follow the lane.
        steer = e.steer_deg
        ang = float(np.clip(STEER_K * steer, -MAX_ANG, MAX_ANG))
        if abs(steer) >= 5.0 and abs(ang) < ANG_MIN:
            ang = math.copysign(ANG_MIN, steer)
        lin = V_LANE * float(np.clip(e.best_m / 2.0, 0.5, 1.0)) * (1.0 - min(abs(steer), 45.0) / 90.0)
        lin = max(V_MIN, lin)
        # 5. Genuinely stuck: wheels not turning under a forward command.
        if self.fwd_since is None:
            self.fwd_since = now
        if tel is not None and wheels_turning(tel):
            self.fwd_since = now
        if now - self.fwd_since > STUCK_S and now - self.last_rev > 10.0:
            self.last_rev, self.fwd_since = now, None
            return -0.25, 0.0, "STUCK: short reverse"
        self.last_lane = (lin, ang, now)
        return lin, ang, f"LANE steer {steer:+.0f} best {e.best_m:.1f}m"

    def draw(self, rgb, g, e, lines) -> np.ndarray:
        im = cv2.cvtColor(cv2.resize(rgb, (768, 432)), cv2.COLOR_RGB2BGR)
        m = cv2.resize(g.astype(np.uint8), (768, 432), interpolation=cv2.INTER_NEAREST).astype(bool)
        im[m] = (0.6 * im[m] + 0.4 * np.array([0, 200, 0])).astype(np.uint8)
        cal = CAL.scaled_to(768, 432)
        for ang, c in zip(e.angles, e.clear):
            if c <= 0:
                continue
            t = math.radians(ang)
            uv, _ = cal.project_ground(np.array([[0.35 * math.cos(t), 0.35 * math.sin(t)],
                                                 [c * math.cos(t), c * math.sin(t)]]))
            col = (0, 255, 255) if c >= 2 else ((0, 160, 255) if c >= 0.8 else (0, 0, 255))
            cv2.line(im, tuple(int(x) for x in uv[0]), tuple(int(x) for x in uv[1]), col, 1)
        if e.steer_deg is not None:
            t = math.radians(e.steer_deg)
            c = max(e.best_m, 0.8)
            uv, _ = cal.project_ground(np.array([[0.35 * math.cos(t), 0.35 * math.sin(t)],
                                                 [c * math.cos(t), c * math.sin(t)]]))
            cv2.arrowedLine(im, tuple(int(x) for x in uv[0]), tuple(int(x) for x in uv[1]),
                            (255, 255, 255), 3, tipLength=0.15)
        cv2.rectangle(im, (0, 0), (768, 20 * len(lines) + 6), (0, 0, 0), -1)
        for k, ln in enumerate(lines):
            cv2.putText(im, ln, (6, 18 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return im

    def run(self) -> int:
        c = self.c
        try:
            print("start-mission:", c.start_mission().get("message"), flush=True)
        except Exception as e:
            print("start-mission failed:", e, flush=True)
        cps, latest = c.get_checkpoints()
        print(f"mission checkpoints: {len(cps)} (latest scanned {latest})", flush=True)
        for cp in cps:
            print(f"  CP{cp.sequence} ({cp.lat:.6f}, {cp.lon:.6f})", flush=True)
        self.stream.start()
        self.halt()
        t0 = time.time()
        while (self.feed.tel is None or self.feed.frame is None
               or not np.isfinite(self.feed.tel.lat)) and time.time() - t0 < 60:
            time.sleep(0.3)
        if self.feed.tel is None:
            print("no telemetry after 60 s - is the SDK up and the mission started?")
            return 1
        remaining = [cp for cp in cps if cp.sequence > latest]
        last_frame_n = -1
        while remaining:
            goal = remaining[0]
            prev = next((cp for cp in cps if cp.sequence == goal.sequence - 1), None)
            tick0 = time.time()
            tel, frame = self.feed.tel, self.feed.frame
            if time.time() - self.feed.tel_t > STALE_S or time.time() - self.feed.frame_t > STALE_S:
                self.halt()
                print("  link stale: holding still", flush=True)
                time.sleep(0.5)
                continue
            rng, brg_goal = geo(tel.lat, tel.lon, goal.lat, goal.lon)
            # ---- claim
            retry_ok = (self.hold is None or rng < self.hold[0] - 2.0 or time.time() - self.hold[1] > 5.0)
            if rng <= CLAIM_M and retry_ok:
                self.halt()
                print(f"claiming CP{goal.sequence} at {rng:.1f} m", flush=True)
                resp = c.checkpoint_reached()
                if resp.get("_http_status", 200) >= 400:
                    print(f"  rejected: {resp}", flush=True)
                    self.hold = (rng, time.time())
                else:
                    print(f"  CLAIMED CP{goal.sequence}: {resp}", flush=True)
                    self.hold = None
                    remaining.pop(0)
                    continue
            # ---- lane bearing: the current leg (prev CP -> goal); direct if no prev
            lane_brg = (pursuit_bearing(tel.lat, tel.lon, prev, goal, PURSUIT_M)
                        if prev is not None else brg_goal)
            heading = math.degrees(tel.yaw) % 360.0 if tel.yaw is not None else None
            prior = rel_left(heading, lane_brg) if heading is not None else 0.0
            # ---- perception (only on a new frame)
            if self.feed.frame_n == last_frame_n:
                time.sleep(0.03)
                continue
            last_frame_n = self.feed.frame_n
            lab = self.sem.labels(frame, SEG_HW)
            if lab is None:
                self.halt()
                continue
            g = np.isin(lab, self.ids)
            e = self.lane.estimate(g, prior_deg=float(np.clip(prior, -45.0, 45.0)))
            lin, ang, mode = self.decide(e, prior, tel, time.time())
            self.cmd(lin, ang)
            # ---- log
            dt = time.time() - tick0
            self.n += 1
            line = (f"CP{goal.sequence} rng={rng:5.1f}m brg={brg_goal:3.0f} lane={lane_brg:3.0f} "
                    f"hdg={heading if heading is None else round(heading)} prior={prior:+4.0f} | "
                    f"{mode} | cmd {lin:+.2f},{ang:+.2f} | ahead {e.ahead_m:.1f} L{e.left_m:.1f} "
                    f"R{e.right_m:.1f} | {dt*1000:.0f}ms")
            print(time.strftime("%H:%M:%S ") + line, flush=True)
            self.ticks.write(json.dumps({"t": time.time(), "cp": goal.sequence, "rng": round(rng, 2),
                                         "lat": tel.lat, "lon": tel.lon, "hdg": heading,
                                         "lane_brg": round(lane_brg, 1), "prior": round(prior, 1),
                                         "steer": e.steer_deg, "best": e.best_m, "ahead": e.ahead_m,
                                         "lin": round(lin, 3), "ang": round(ang, 3), "mode": mode,
                                         "ms": round(dt * 1000)}) + "\n")
            if self.overlay is not None or self.n % SAVE_EVERY == 0:
                im = self.draw(frame, g, e, [line[:110], f"goal CP{goal.sequence}  {mode}"])
                if self.overlay is not None:
                    self.overlay.publish(im)
                if self.n % SAVE_EVERY == 0:
                    cv2.imwrite(os.path.join(self.dir, "overlay", f"{self.n:06d}.jpg"), im)
                    cv2.imwrite(os.path.join(self.dir, "frames", f"{self.n:06d}.jpg"),
                                cv2.cvtColor(cv2.resize(frame, (512, 288)), cv2.COLOR_RGB2BGR))
            time.sleep(max(0.0, TICK_S - (time.time() - tick0)))
        self.halt()
        print("done - all checkpoints claimed", flush=True)
        return 0


def main() -> int:
    client = RoverClient(SDK_URL, timeout=30)
    m = Mexico(client)

    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        return m.run()
    except KeyboardInterrupt:
        print("\ninterrupted - halting", flush=True)
        return 130
    finally:
        m.feed.running = False
        for _ in range(3):
            m.halt()
            time.sleep(0.1)
        try:
            m.stream.stop(lamp_off=False)
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
