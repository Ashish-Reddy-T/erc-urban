"""
A stand-in for the FrodoBots SDK server, for end-to-end tests of the agent
without a bot. Same endpoints and payload shapes the real SDK serves:

  GET  /                   200 (ping)
  POST /start-mission      starts the mission; /data answers 400 before it
  GET  /data               5-sample bursts of gyros/rpms/accels (+ hw time), GPS at 1 Hz,
                           orientation (compass), rtm_state
  GET  /v2/screenshot      {"front_frame": b64 jpeg, "rear_frame": ...}
  POST /control            {"command": {"linear", "angular", "lamp"}}; a 0.6 s watchdog
  GET  /checkpoints-list   {"checkpoints_list": [...], "latest_scanned_checkpoint": n}
  POST /checkpoint-reached 200 if the next checkpoint is within 15 m, else 400

It can play the bots we met: yaw gyro on z (Mini+) or x (Zero), a compass
with a bias and/or wrong scale, a flapping RTM link, different wheel speeds.
"""
from __future__ import annotations

import base64
import json
import math
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAT0, LON0 = 9.791500, -84.105900
M_PER_REV = 0.388


class FakeRover:
    def __init__(self, checkpoints_en, gyro_axis="z", compass_bias=0.0, compass_scale=1.0,
                 speed_per_unit=1.37, yaw_per_unit=60.0, start_yaw=0.0, flap=False,
                 front_jpeg: bytes = b"", gps_noise_m=0.3, deadband=0.1, claim_radius=15.0):
        self.cps = checkpoints_en
        self.gyro_axis = gyro_axis
        self.compass_bias, self.compass_scale = compass_bias, compass_scale
        self.speed_per_unit, self.yaw_per_unit = speed_per_unit, yaw_per_unit
        self.deadband = deadband
        self.claim_radius = claim_radius        # < 15: the server's GPS disagrees with ours
        self.flap = flap
        self.front_b64 = base64.b64encode(front_jpeg).decode() if front_jpeg else ""
        self.gps_noise = gps_noise_m
        self.e = self.n = 0.0
        self.yaw = start_yaw                      # CW from north, deg
        self.cmd = (0.0, 0.0)
        self.cmd_t = 0.0
        self.started = False
        self.latest = 0
        self.claims: list[tuple[int, float]] = []
        self.path_m = 0.0
        self.max_ang_cmd = 0.0
        self.lock = threading.Lock()
        self.samples: list[tuple[float, float, float]] = []   # (ts, v, w_left)
        self.gps = (LAT0, LON0)
        self._gps_t = 0.0
        self._running = True
        threading.Thread(target=self._sim, daemon=True).start()

    def latlon(self, e, n):
        return (LAT0 + n / 110540.0, LON0 + e / (111320.0 * math.cos(math.radians(LAT0))))

    def _sim(self):
        dt = 0.02
        while self._running:
            t = time.time()
            with self.lock:
                lin, ang = self.cmd if t - self.cmd_t < 0.6 else (0.0, 0.0)
                v = lin * self.speed_per_unit if abs(lin) >= self.deadband else 0.0
                w = ang * self.yaw_per_unit if abs(ang) >= self.deadband else 0.0   # left +
                self.yaw = (self.yaw - w * dt) % 360.0
                r = math.radians(self.yaw)
                self.e += v * dt * math.sin(r)
                self.n += v * dt * math.cos(r)
                self.path_m += abs(v) * dt
                self.samples.append((t, v, w))
                del self.samples[:-10]
                if t - self._gps_t >= 1.0:
                    self._gps_t = t
                    ne = self.e + random.gauss(0, self.gps_noise)
                    nn = self.n + random.gauss(0, self.gps_noise)
                    self.gps = self.latlon(ne, nn)
            time.sleep(dt)

    def data(self) -> dict:
        with self.lock:
            s = list(self.samples[-5:])
            gyros, rpms, accels = [], [], []
            for ts, v, w in s:
                g = [0.3, 0.4, 0.2]                        # small bias on every axis
                g["xyz".index(self.gyro_axis)] += w
                gyros.append([round(x, 3) for x in g] + [ts])
                rpm = v / M_PER_REV * 60.0
                spin = w / 360.0 * math.pi * 0.16 / M_PER_REV * 60.0   # pivot: sides opposite
                rpms.append([rpm - spin, rpm + spin, rpm - spin, rpm + spin, ts])
                a = [0.01, 0.02, 0.01]
                a["xyz".index(self.gyro_axis)] = 1.0
                accels.append(a + [ts])
            compass = (self.compass_scale * self.yaw + self.compass_bias) % 360.0
            rtm = "CONNECTED"
            if self.flap and int(time.time()) % 3 == 0:
                rtm = "ABORTED"
            return {"battery": 90, "signal_level": 5, "orientation": int(compass),
                    "latitude": self.gps[0], "longitude": self.gps[1], "gps_signal": 45,
                    "gyros": gyros, "rpms": rpms, "accels": accels,
                    "timestamp": str(time.time()), "rtm_state": rtm}

    def next_cp_dist(self) -> float:
        if self.latest >= len(self.cps):
            return float("inf")
        e, n = self.cps[self.latest]
        return math.hypot(self.e - e, self.n - n)


def make_handler(rover: FakeRover):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _cps(self):
            out = []
            for i, (e, n) in enumerate(rover.cps):
                lat, lon = rover.latlon(e, n)
                out.append({"id": 100 + i, "sequence": i + 1, "latitude": f"{lat:.9f}",
                            "longitude": f"{lon:.9f}"})
            return {"checkpoints_list": out, "latest_scanned_checkpoint": rover.latest}

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                return self._send(200, {"ok": True})
            if p == "/data":
                if not rover.started:
                    return self._send(400, {"detail": "Call /start-mission first"})
                return self._send(200, rover.data())
            if p in ("/v2/screenshot", "/v2/front"):
                return self._send(200, {"front_frame": rover.front_b64, "rear_frame": rover.front_b64,
                                        "timestamp": time.time()})
            if p == "/checkpoints-list":
                return self._send(200, self._cps())
            return self._send(404, {"detail": "not found"})

        def do_POST(self):
            p = self.path.split("?")[0]
            n = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            if p == "/start-mission":
                rover.started = True
                return self._send(200, {"message": "Mission started successfully",
                                        "checkpoints_list": self._cps()})
            if p == "/control":
                c = body.get("command", {})
                with rover.lock:
                    rover.cmd = (float(c.get("linear", 0.0)), float(c.get("angular", 0.0)))
                    rover.cmd_t = time.time()
                    rover.max_ang_cmd = max(rover.max_ang_cmd, abs(rover.cmd[1]))
                return self._send(200, {"message": "ok"})
            if p == "/checkpoint-reached":
                d = rover.next_cp_dist()
                if d <= rover.claim_radius:
                    rover.latest += 1
                    rover.claims.append((rover.latest, d))
                    nxt = rover.latest + 1 if rover.latest < len(rover.cps) else None
                    return self._send(200, {"message": "Checkpoint reached successfully",
                                            "next_checkpoint_sequence": nxt})
                return self._send(400, {"detail": f"not within {rover.claim_radius:.0f} m ({d:.1f} m)"})
            if p == "/end-mission":
                return self._send(404, {"detail": "Failed to end mission"})
            return self._send(404, {"detail": "not found"})
    return H


def serve(rover: FakeRover, port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(rover))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def front_fixture() -> bytes:
    p = os.path.join(os.path.dirname(__file__), "fixtures", "e2e_front_plaza.jpg")
    import cv2
    img = cv2.resize(cv2.imread(p), (1024, 576))
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()
