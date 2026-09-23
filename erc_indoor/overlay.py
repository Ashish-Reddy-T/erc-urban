"""
Annotated camera view of what the navigator sees and decides — for the
operator's screen and the team website.

    python -m erc_indoor.overlay runs/indoor/run_XXXX       # render a finished run -> overlay.mp4

Live: run.py renders one per tick (OVERLAY=1, default) to
runs/indoor/run_*/overlay/NNNNNN.jpg and serves the newest frame as an MJPEG
stream on OVERLAY_PORT (default 8090):

    <img src="http://<mac-ip>:8090/stream.mjpg">       (live)
    http://<mac-ip>:8090/latest.jpg                     (single frame)

What is drawn:
  * green    floor that SegFormer-B2 (ADE20K) labels traversable
  * red line where the floor ends in each image column (obstacle foot)
  * fan      clearance per heading bin at the bottom (green >1.6 m, amber, red <0.6 m)
  * cyan     teach-lap heading (where the recorded lap was pointing)
  * magenta  corridor axis (farthest floor) when corridor-following
  * orange   avoidance heading when an obstacle forced a detour
  * boxes    detected cones
  * HUD      mode, route progress, checkpoint, command, localization spread
  * bar      the lap: corners (ticks), checkpoints (dots), rover position
"""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

OUT_W, OUT_H = 768, 432
FX_1024 = 340.0
CONE_COLORS = {"red": (0, 80, 255), "blue": (255, 120, 0), "green": (0, 200, 0), "yellow": (0, 220, 255)}


def _u_for_deg(deg: float, W: int) -> int:
    """Image column of a heading (deg, left-positive)."""
    fx = FX_1024 * W / 1024.0
    return int(round(W / 2 - fx * np.tan(np.radians(np.clip(deg, -75, 75)))))


def _clear_color(c: float) -> tuple[int, int, int]:
    if c < 0.6:
        return (40, 40, 230)
    if c < 1.6:
        return (0, 180, 255)
    return (60, 200, 60)


def render(frame_rgb: np.ndarray, log, cmd=None, floor: Optional[np.ndarray] = None,
           boundary: Optional[np.ndarray] = None, clear: Optional[tuple] = None,
           route=None, cone_list=None) -> np.ndarray:
    """Returns a BGR image OUT_W x OUT_H. Every argument but the frame is optional."""
    img = cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    H, W = img.shape[:2]

    # ---- traversable floor
    if floor is not None:
        fm = cv2.resize(floor.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
        green = np.zeros_like(img)
        green[fm] = (60, 200, 60)
        img = cv2.addWeighted(img, 1.0, green, 0.35, 0)
    if boundary is not None:
        sx, sy = W / len(boundary), H / float(floor.shape[0] if floor is not None else 288)
        pts = np.stack([np.arange(len(boundary)) * sx, boundary * sy], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, (40, 40, 230), 2, cv2.LINE_AA)

    # ---- clearance fan along the bottom
    if clear is not None:
        bins, cl = clear
        for b, c in zip(bins, cl):
            u = _u_for_deg(float(b), W)
            L = int(np.clip(c / 5.0, 0.05, 1.0) * 60)
            cv2.line(img, (u, H - 4), (u, H - 4 - L), _clear_color(float(c)), 5)

    # ---- heading guides
    def vline(deg, color, label, y0):
        u = _u_for_deg(deg, W)
        cv2.line(img, (u, int(H * 0.30)), (u, int(H * 0.62)), color, 2, cv2.LINE_AA)
        cv2.circle(img, (u, int(H * 0.30)), 5, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (u + 6, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if getattr(log, "yaw_err", None) is not None:
        vline(-float(log.yaw_err), (255, 230, 0), f"teach {-log.yaw_err:+.0f}", int(H * 0.28))
    if getattr(log, "corridor_deg", None) is not None:
        vline(float(log.corridor_deg), (230, 60, 230), f"corridor {log.corridor_deg:+.0f}", int(H * 0.24))
    if getattr(log, "avoid_deg", None) is not None:
        vline(float(log.avoid_deg), (0, 140, 255), f"avoid {log.avoid_deg:+.0f}", int(H * 0.20))
    # straight-ahead tick
    cv2.line(img, (W // 2, int(H * 0.60)), (W // 2, int(H * 0.66)), (255, 255, 255), 1)

    # ---- cones (Cone objects from cones.py, WORK 512x288 base coords)
    for c in cone_list or []:
        sx, sy = W / 512.0, H / 288.0
        u, v = int(c.u * sx), int(c.v * sy)
        hh = max(8, int(40 * sy / max(c.dist_m, 0.3)))
        col = CONE_COLORS.get(c.color, (255, 255, 255))
        cv2.rectangle(img, (u - hh // 2, v - hh), (u + hh // 2, v), col, 2)
        cv2.putText(img, f"{c.color} {c.dist_m:.1f}m", (u - hh // 2, v - hh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # ---- HUD
    panel = img.copy()
    cv2.rectangle(panel, (0, 0), (W, 46), (0, 0, 0), -1)
    img = cv2.addWeighted(panel, 0.55, img, 0.45, 0)
    mode = getattr(cmd, "mode", "") if cmd is not None else ""
    lin = getattr(cmd, "lin", 0.0) if cmd is not None else 0.0
    ang = getattr(cmd, "ang", 0.0) if cmd is not None else 0.0
    total = float(route.s[-1]) if route is not None else 132.0
    s = float(getattr(log, "s", 0.0))
    line1 = (f"{mode:<11s} s={s:5.1f}/{total:.0f} m  {getattr(log, 'checkpoint', '')}  "
             f"turn {min(getattr(log, 'turn_k', 0) + 1, 4)}/4")
    ahead = getattr(log, "ahead_m", None)
    line2 = (f"cmd lin {lin:+.2f} ang {ang:+.2f}   loc spread {getattr(log, 'spread', 0):.1f} m   "
             f"head {getattr(log, 'head_src', '') or '-'}   ahead {ahead:.1f} m" if ahead is not None else
             f"cmd lin {lin:+.2f} ang {ang:+.2f}   loc spread {getattr(log, 'spread', 0):.1f} m   "
             f"head {getattr(log, 'head_src', '') or '-'}")
    cv2.putText(img, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, line2, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # ---- lap progress bar
    if route is not None:
        x0, x1, y = 8, W - 8, 54
        cv2.line(img, (x0, y), (x1, y), (180, 180, 180), 2)
        f = lambda sv: int(x0 + (x1 - x0) * np.clip(sv / total, 0, 1))
        for t in route.turns:
            cv2.line(img, (f(t.s), y - 6), (f(t.s), y + 6), (255, 230, 0), 2)
        for name, cs in route.checkpoints.items():
            col = {"CP1": (0, 80, 255), "CP2": (255, 120, 0), "CP3": (0, 200, 0), "CP4": (0, 220, 255)}.get(name, (255, 255, 255))
            cv2.circle(img, (f(cs), y), 5, col, -1)
        cv2.circle(img, (f(route.finish_s), y), 6, (0, 140, 255), 2)
        cv2.circle(img, (f(s), y), 7, (255, 255, 255), -1)
    return img


# ---- live MJPEG server ------------------------------------------------------
class OverlayServer:
    """Serves the newest overlay JPEG at /latest.jpg and /stream.mjpg."""

    def __init__(self, port: int):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self._jpg: Optional[bytes] = None
        self._cond = threading.Condition()
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.startswith("/latest.jpg"):
                    with srv._cond:
                        data = srv._jpg
                    if data is None:
                        self.send_error(503)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if self.path.startswith("/stream.mjpg"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    try:
                        while True:
                            with srv._cond:
                                srv._cond.wait(timeout=2.0)
                                data = srv._jpg
                            if data is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                self.send_error(404)

        self.httpd = ThreadingHTTPServer(("0.0.0.0", port), H)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def publish(self, bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        data = buf.tobytes()
        with self._cond:
            self._jpg = data
            self._cond.notify_all()
        return data


# ---- offline: render a finished run ------------------------------------------
def render_run(run_dir: str, fps: float = 4.0) -> str:
    """Re-render a run's saved frames with its log -> overlay.mp4 (floor mask
    recomputed; headings/HUD from the log)."""
    import json
    from types import SimpleNamespace
    from PIL import Image
    from .freespace import FreeSpaceEstimator
    from .cones import ConeDetector
    from .route_map import DEFAULT_OUT, RouteMap

    route = RouteMap(DEFAULT_OUT)
    fs = FreeSpaceEstimator()
    cd = ConeDetector()
    recs = {r["tick"]: r for r in (json.loads(l) for l in open(os.path.join(run_dir, "log.jsonl")))}
    fdir = os.path.join(run_dir, "frames")
    files = sorted(os.listdir(fdir))
    out = os.path.join(run_dir, "overlay.mp4")
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_W, OUT_H))
    for f in files:
        tick = int(os.path.splitext(f)[0])
        r = recs.get(tick)
        if r is None:
            continue
        rgb = np.asarray(Image.open(os.path.join(fdir, f)).convert("RGB").resize((1024, 576)))
        est = fs.estimate(rgb)
        log = SimpleNamespace(**{k: r.get(k) for k in ("s", "spread", "turn_k", "checkpoint", "head_src",
                                                         "yaw_err", "corridor_deg", "avoid_deg")},
                              ahead_m=est.ahead_m)
        cmd = SimpleNamespace(mode=r["mode"], lin=r["lin"], ang=r["ang"])
        img = render(rgb, log, cmd, fs.last_floor, fs.last_boundary, (est.bins_deg, est.clear_m),
                     route, cd.detect(rgb))
        vw.write(img)
    vw.release()
    return out


if __name__ == "__main__":
    import sys
    print(render_run(sys.argv[1]))
