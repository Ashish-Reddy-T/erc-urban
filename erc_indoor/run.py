"""
Main control loop for the NYU indoor track. This is the file you execute.

    python -m erc_indoor.run                    # drive the lap
    SEND_CONTROL=0 python -m erc_indoor.run     # dry run: localize + log, no motion
    START_S=34 python -m erc_indoor.run         # start mid-route (e.g. after the piano turn)

There is no SDK mission indoors: nothing is started, ended or claimed.
Checkpoints are judged at the cones.

Place the rover at the orange Start/Finish cone facing down the glass atrium
(windows on the RIGHT), the same way the teach lap began. It then follows the
teach lap: straight through CP1 (red), LEFT at the piano, straight through CP2
(blue) to the chairs, LEFT, through CP3 (green), LEFT, through CP4 (yellow),
RIGHT, and stops at the orange cone.

No GPS indoors (lat/lon read 1000). Navigation is visual teach-and-repeat
against the daytime teach lap (erc_indoor/data/route_map, built by
`python -m erc_indoor.route_map`) plus wheel/gyro odometry. The cones are never
navigation targets; the detector only pauses for one sitting in the path.

Tuning knobs are env vars (see navigator.py for the driving ones):
    SDK_URL, CTRL_HZ, SEND_CONTROL, USE_CONES, START_S, SAVE_FRAMES, ROUTE_MAP
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time

import numpy as np
import requests
from PIL import Image

from .cones import ConeDetector
from .localizer import RouteLocalizer
from .navigator import Cmd, Navigator, Obs
from .odometry import Odometer
from .place import PlaceEncoder
from .route_map import DEFAULT_OUT, RouteMap
from .sdk_client import RoverClient

SDK_URL = os.environ.get("SDK_URL", "http://localhost:8000")
# Perception is ~250-450 ms/tick (frame fetch + DINOv2 + SIFT); commands are
# streamed separately at STREAM_HZ so a slow tick never starves the watchdog.
CTRL_HZ = float(os.environ.get("CTRL_HZ", "4"))
STREAM_HZ = 10.0
SEND_CONTROL = os.environ.get("SEND_CONTROL", "1") != "0"
USE_CONES = os.environ.get("USE_CONES", "1") != "0"
USE_FREESPACE = os.environ.get("USE_FREESPACE", "1") != "0"
START_S = float(os.environ.get("START_S", "0.0"))
START_SIGMA_M = float(os.environ.get("START_SIGMA_M", "3.0"))
SAVE_FRAMES = int(os.environ.get("SAVE_FRAMES", "2"))   # save every Nth tick's frame; 0 = off
ROUTE_MAP = os.environ.get("ROUTE_MAP", DEFAULT_OUT)
LAMP = int(os.environ.get("LAMP", "0"))
MAX_RUN_S = float(os.environ.get("MAX_RUN_S", "1500"))
# Stop after this many metres of map progress (staged testing). 0 = full lap.
STOP_AT_S = float(os.environ.get("STOP_AT_S", "0"))
# Link watchdog. Run 4: the bot stopped answering — frames byte-static,
# wheel/gyro 350-430 s old, not one commanded spin happened — while the loop
# kept deciding on frozen data. If neither the image nor the odometry changes
# for this long while we command motion, hold zero until the link is back.
LINK_STALE_S = float(os.environ.get("LINK_STALE_S", "4.0"))
# Full-lap run 3: the SDK stalled; each /v2/front call blocked up to its 10 s
# timeout, so the in-loop watchdog could not run. Short timeouts + treat a
# slow tick or stale odometry as a lost link.
SDK_TIMEOUT_S = float(os.environ.get("SDK_TIMEOUT_S", "3.0"))
ODOM_STALE_S = float(os.environ.get("ODOM_STALE_S", "2.5"))
# Annotated camera view (overlay.py): saved per tick and served live as MJPEG
# at http://<mac>:OVERLAY_PORT/stream.mjpg for the operator / team website.
OVERLAY = os.environ.get("OVERLAY", "1") != "0"
OVERLAY_PORT = int(os.environ.get("OVERLAY_PORT", "8090"))


class CommandStreamer(threading.Thread):
    """Sends the latest command at STREAM_HZ until it expires, then zeros.

    The Mini+ stops when commands stop arriving, and a burst must last exactly
    its planned duration regardless of how long the perception tick takes.
    """

    def __init__(self, client: RoverClient, send: bool):
        super().__init__(daemon=True)
        self.client = client
        self.send = send
        self._lock = threading.Lock()
        self._cmd = (0.0, 0.0)
        self._expires = 0.0
        self._running = True

    def set(self, lin: float, ang: float, hold_s: float) -> None:
        with self._lock:
            self._cmd = (float(lin), float(ang))
            self._expires = time.time() + max(0.0, hold_s)

    def halt(self) -> None:
        self.set(0.0, 0.0, 0.0)
        self._post(0.0, 0.0)

    def _post(self, lin: float, ang: float) -> None:
        if not self.send:
            return
        try:
            self.client.send_cmd(lin, ang, lamp=LAMP)
        except Exception:
            pass

    def run(self) -> None:
        while self._running:
            t0 = time.time()
            with self._lock:
                lin, ang = self._cmd if t0 < self._expires else (0.0, 0.0)
            self._post(lin, ang)
            time.sleep(max(0.0, 1.0 / STREAM_HZ - (time.time() - t0)))

    def stop(self) -> None:
        self._running = False


def wait_for_sdk(client: RoverClient, timeout_s: float = 30.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if client.ping():
            print("SDK HTTP up")
            return
        print("waiting for SDK on", client.base, "...")
        time.sleep(1.0)
    raise RuntimeError(f"SDK not reachable at {client.base}.")


def wait_for_telemetry(client: RoverClient, odo: Odometer, timeout_s: float = 60.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if odo.ready and client.get_front() is not None:
            st = odo.state()
            print(f"telemetry ok: odometry age {st.age_s:.2f} s, front camera ok")
            return
        try:
            if requests.get(f"{client.base}/data", timeout=5).status_code == 400:
                print("SDK answers 400 'start-mission': there is no indoor mission — remove "
                      "MISSION_SLUG from earth-rovers-sdk/.env and restart the SDK")
        except requests.RequestException:
            pass
        print("waiting for /data rpms+gyros and /v2/front ...")
        time.sleep(1.0)
    raise RuntimeError("no telemetry/camera from the SDK")


def build_stack():
    route = RouteMap(ROUTE_MAP)
    print(f"route map: {len(route)} nodes, {route.s[-1]:.1f} m, {len(route.turns)} turns, "
          f"checkpoints {route.checkpoints}, finish s={route.finish_s}")
    enc = PlaceEncoder()
    enc.encode_one(route.image(0))   # warm up the MPS kernels before the clock starts
    loc = RouteLocalizer(route)
    loc.reset(route.p[route.node_at_s(START_S)], START_SIGMA_M)
    cones = ConeDetector() if USE_CONES else None
    free = None
    if USE_FREESPACE:
        from .freespace import FreeSpaceEstimator
        free = FreeSpaceEstimator()
        free.estimate(route.image(0))   # warm up
    return route, enc, loc, cones, free


def main() -> None:
    client = RoverClient(SDK_URL, timeout=SDK_TIMEOUT_S)
    route, enc, loc, cones, free = build_stack()
    start_p = float(route.p[route.node_at_s(START_S)]) if "START_S" in os.environ else None
    nav = Navigator(route, loc, enc, cones, freespace=free, start_p=start_p)
    print(f"SEND_CONTROL={SEND_CONTROL} USE_CONES={USE_CONES} USE_FREESPACE={USE_FREESPACE} "
          f"CTRL_HZ={CTRL_HZ} START_S={START_S}")

    wait_for_sdk(client)

    streamer = CommandStreamer(client, SEND_CONTROL)
    streamer.halt()
    odo = Odometer(client)
    odo.start()
    wait_for_telemetry(client, odo)
    bias = odo.calibrate_bias(3.0)
    print(f"gyro bias {bias:+.3f} deg/s (rover must be still)")
    streamer.start()

    run_dir = os.path.join("runs", "indoor", time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
    ov_server = None
    if OVERLAY:
        from . import overlay as ov
        os.makedirs(os.path.join(run_dir, "overlay"), exist_ok=True)
        try:
            ov_server = ov.OverlayServer(OVERLAY_PORT)
            print(f"overlay live at http://localhost:{OVERLAY_PORT}/stream.mjpg "
                  f"(and /latest.jpg)")
        except OSError as e:
            print(f"overlay server not started ({e}); frames still saved")
    logf = open(os.path.join(run_dir, "log.jsonl"), "a")
    print(f"logging to {run_dir}")

    stop = {"flag": False}

    def _sigint(*_):
        stop["flag"] = True
    # SIGTERM too: `kill` must stop the rover through the finally below, not
    # kill Python with the last command still streaming.
    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    dt = 1.0 / CTRL_HZ
    t_start = time.time()
    last_small = None
    last_change = time.time()
    last_hw = 0.0
    commanding_since = None
    link_down = False
    tick = 0
    last_mode = None
    try:
        while not stop["flag"]:
            t0 = time.time()
            if t0 - t_start > MAX_RUN_S:
                print("MAX_RUN_S reached — stopping")
                break
            frame = client.get_front()
            st = odo.state()
            tick_slow = (time.time() - t0) > LINK_STALE_S
            odom_stale = st.age_s == st.age_s and (time.time() - st.hw_ts) > ODOM_STALE_S \
                and commanding_since is not None
            if frame is None or tick_slow or odom_stale:
                streamer.set(0.0, 0.0, 0.3)
                if not link_down:
                    why = "no frame" if frame is None else ("frame fetch took %.0f s" % (time.time() - t0)
                                                              if tick_slow else "wheel/gyro data %.0f s old"
                                                              % (time.time() - st.hw_ts))
                    print(f"[LINK] {why} — holding zero until the SDK answers normally", flush=True)
                link_down = True
                time.sleep(0.3)
                continue
            # ---- link watchdog: is anything actually changing?
            if frame is not None:
                small = np.asarray(Image.fromarray(frame).resize((64, 36)).convert("L"), np.float32)
                if last_small is None or np.abs(small - last_small).mean() > 1.5:
                    last_change = time.time()
                last_small = small
            if st.hw_ts > last_hw:
                last_hw = st.hw_ts
                last_change = time.time()
            if commanding_since is not None and time.time() - max(last_change, commanding_since) > LINK_STALE_S:
                if not link_down:
                    print(f"[LINK] no new image or wheel/gyro data for {LINK_STALE_S:.0f} s while "
                          f"commanding motion (odometry age {st.age_s:.0f} s) — holding zero. "
                          "Check the rover / SDK connection.", flush=True)
                link_down = True
                streamer.set(0.0, 0.0, 0.3)
                time.sleep(0.5)
                continue
            if link_down:
                print("[LINK] data flowing again — resuming", flush=True)
                link_down = False
                commanding_since = None
            obs = Obs(time.time(), frame, st.s, st.yaw, st.v)
            try:
                cmd = nav.step(obs)
            except Exception as e:  # never leave the rover driving on a bug
                streamer.halt()
                print("tick error:", repr(e))
                cmd = Cmd(0.0, 0.0, 0.3, "ERROR", repr(e))
            streamer.set(cmd.lin, cmd.ang, cmd.hold_s)
            moving_cmd = abs(cmd.lin) > 1e-3 or abs(cmd.ang) > 1e-3
            if moving_cmd and commanding_since is None:
                commanding_since = time.time()
            elif not moving_cmd and cmd.mode not in ("TURN", "ESCAPE", "ESCAPE_TURN", "SEARCH", "AVOID"):
                commanding_since = None

            L = nav.log
            if cmd.mode != last_mode:
                print(f"[mode] {last_mode} -> {cmd.mode}")
                last_mode = cmd.mode
            if tick % max(1, int(CTRL_HZ)) == 0:
                err = f"{L.yaw_err:+5.1f}" if L.yaw_err is not None else "  -  "
                ahead = f"{L.ahead_m:.1f}" if L.ahead_m is not None else " - "
                print(f"t={t0 - t_start:6.1f} {cmd.mode:6s} s={L.s:6.1f} node={L.node:4d} "
                      f"spread={L.spread:4.1f} err={err} inl={L.inliers:3d}{L.ref_src[0]} ahead={ahead} "
                      f"cmd=({cmd.lin:+.2f},{cmd.ang:+.2f}) {L.checkpoint} | {cmd.note}", flush=True)
            rec = {"t": t0, "tick": tick, "mode": cmd.mode, "lin": cmd.lin, "ang": cmd.ang,
                   "hold": cmd.hold_s, "note": cmd.note, "odo_s": st.s, "odo_yaw": st.yaw,
                   "v": st.v, "odo_age": st.age_s, "node": L.node, "s": L.s, "p": L.p,
                   "spread": L.spread, "sim": L.sim, "ref": L.ref, "ref_src": L.ref_src, "yaw_err": L.yaw_err,
                   "head_src": L.head_src, "ahead_m": L.ahead_m, "avoid_deg": L.avoid_deg,
                   "corridor_deg": L.corridor_deg,
                   "inliers": L.inliers, "turn_k": L.turn_k, "spin_rate": L.spin_rate,
                   "cones": L.cones, "checkpoint": L.checkpoint, "frame": frame is not None}
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if OVERLAY and frame is not None:
                try:
                    fsr = nav.last_fs if (nav.last_fs is not None and nav.last_fs[1] == obs.now) else None
                    img = ov.render(frame, L, cmd,
                                    free.last_floor if (fsr is not None and free is not None) else None,
                                    free.last_boundary if (fsr is not None and free is not None) else None,
                                    (fsr[0].bins_deg, fsr[0].clear_m) if fsr is not None else None,
                                    route, nav.last_cones)
                    data = ov_server.publish(img) if ov_server is not None else None
                    if tick % max(1, SAVE_FRAMES) == 0:
                        with open(os.path.join(run_dir, "overlay", f"{tick:06d}.jpg"), "wb") as fh:
                            fh.write(data if data is not None else ov.cv2.imencode(".jpg", img)[1].tobytes())
                except Exception as e:  # the overlay must never stop the rover
                    if tick % 20 == 0:
                        print("overlay error:", repr(e))
            if SAVE_FRAMES and frame is not None and tick % SAVE_FRAMES == 0:
                Image.fromarray(frame).resize((512, 288)).save(
                    os.path.join(run_dir, "frames", f"{tick:06d}.jpg"), quality=80)
            if cmd.done:
                print("DONE — at the finish")
                break
            if STOP_AT_S > 0 and L.s >= STOP_AT_S and cmd.mode == "DRIVE":
                print(f"STOP_AT_S={STOP_AT_S} reached (s={L.s:.1f}) — stopping")
                break
            tick += 1
            time.sleep(max(0.0, dt - (time.time() - t0)))
    finally:
        streamer.stop()
        for _ in range(3):
            streamer.halt()
            time.sleep(0.1)
        odo.stop()
        logf.close()
        print(f"stopped. log: {run_dir}/log.jsonl")


if __name__ == "__main__":
    main()
