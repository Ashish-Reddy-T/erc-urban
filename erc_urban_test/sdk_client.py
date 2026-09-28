"""
From docs:
  - Server: `hypercorn main:app --reload` (default http://localhost:8000)
  - POST /control          {"command": {"linear", "angular"}}  (lamp optional)
  - GET  /data             telemetry; orientation
  - GET  /v2/screenshot    front/rear base64 frames
  - Mission: 
    - POST /missions
    - POST /start-mission
    - GET /checkpoints-list
    - POST /checkpoint-reached
    - POST /end-mission
  - GET  /status           pipeline health
"""
from __future__ import annotations
import base64
import io
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image


def _default_timeout_s() -> float:
    """Short default so a hung SDK cannot block a whole urban control tick."""
    return float(os.environ.get("SDK_TIMEOUT_S", "3.0"))


@dataclass
class Telemetry:
    lat: float
    lon: float
    # Heading in radians, 0 = North, increasing CW.
    yaw: Optional[float]
    # Monotonic capture time on OUR clock (latency / PoseBuffer).
    stamp: float
    raw: dict
    # Hardware timestamp of the newest IMU sample in this payload. The SDK
    # refreshes at ~2 Hz while /data answers at ~17 Hz, so this is the ONLY
    # way to tell a fresh reading from a repeat of the same one.
    hw_ts: Optional[float] = None
    # Age of that sample when we received it (measured ~1.1 s median).
    age_s: Optional[float] = None
    # Body yaw rate, deg/s (gyros z). None when absent.
    yaw_rate: Optional[float] = None
    # True when any wheel reports non-zero RPM — ground truth for "moving",
    # far more reliable than differencing a noisy GPS range.
    wheels_turning: Optional[bool] = None


@dataclass
class Checkpoint:
    id: Optional[int]
    sequence: int
    lat: float
    lon: float


class RoverClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: Optional[float] = None,
    ):
        # Default 3s (or SDK_TIMEOUT_S). Constructor override still wins.
        # Indoor uses ~3s so a hung HTTP call cannot stall the control tick.
        # The slow first request (Chrome launch) is paid once in warm_up().
        self.base = base_url.rstrip("/")
        self.timeout = _default_timeout_s() if timeout is None else float(timeout)

    # ---- control -----------------------------------------------------------
    def send_cmd(self, linear: float, angular: float, lamp: Optional[int] = None) -> None:
        """linear/angular clamped to [-1, 1]. lamp is 0/1 if provided."""
        linear = float(np.clip(linear, -1.0, 1.0))
        angular = float(np.clip(angular, -1.0, 1.0))
        command: dict[str, Any] = {"linear": linear, "angular": angular}
        if lamp is not None:
            command["lamp"] = int(lamp)
        requests.post(
            f"{self.base}/control",
            json={"command": command},
            timeout=self.timeout,
        )

    def stop(self) -> None:
        self.send_cmd(0.0, 0.0)

    # ---- perception --------------------------------------------------------
    def get_frames(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Returns (front_rgb, rear_rgb) as HxWx3 uint8, or (None, None)."""
        r = requests.get(
            f"{self.base}/v2/screenshot", 
            timeout=self.timeout
        )
        r.raise_for_status()
        d = r.json()
        if not isinstance(d, dict):
            return None, None
        return self._decode(d.get("front_frame")), self._decode(d.get("rear_frame"))

    def get_map_frame(self) -> Optional[np.ndarray]:
        """
        The dashboard minimap as HxWx3 uint8, or None.

        This is a Leaflet-rendered OpenStreetMap view with the rover drawn as a
        dot and the checkpoints as numbered pins — i.e. a RASTER of the very OSM
        data an Overpass query returns as vectors. Useful as a human-legible
        record of what the rover saw, and as a cross-check on GPS; a poor
        substitute for querying OSM directly if you want geometry.

        Note the endpoint: the minimap exists only on v1 `/screenshot`, not on
        `/v2/screenshot`. v1 works by driving the headless browser to screenshot
        DOM elements to PNG on disk and reading them back, so it costs far more
        than a v2 frame grab. Call it as a low-rate side channel, never per tick
        in the control loop.
        """
        try:
            r = requests.get(
                f"{self.base}/screenshot",
                params={"view_types": "map"},
                timeout=max(self.timeout, 15.0),
            )
            r.raise_for_status()
            return self._decode(r.json().get("map_frame"))
        except Exception:
            return None

    @staticmethod
    def _decode(b64: Optional[str]) -> Optional[np.ndarray]:
        if not b64 or not isinstance(b64, str):
            return None
        # SDK sometimes prefixes a data URI ("data:image/jpeg;base64,....").
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception:
            return None
        return np.asarray(img)

    # ---- telemetry ---------------------------------------------------------
    def get_data(self) -> Telemetry:
        stamp = time.monotonic()
        try:
            r = requests.get(f"{self.base}/data", timeout=self.timeout)
            r.raise_for_status()
            d = r.json()
        except Exception:
            return Telemetry(np.nan, np.nan, None, stamp, {})

        # SDK can return JSON null before RTM telemetry fills in.
        if not isinstance(d, dict):
            return Telemetry(np.nan, np.nan, None, stamp, {})

        def _finite(v) -> Optional[float]:
            if isinstance(v, bool):
                return None
            try:
                f = float(v)
            except (TypeError, ValueError, OverflowError):
                return None
            return f if math.isfinite(f) else None

        lat = _finite(d.get("latitude"))
        lon = _finite(d.get("longitude"))
        yaw_deg = _finite(d.get("orientation"))

        yaw = math.radians(yaw_deg) if yaw_deg is not None else None

        def _last_ts(key):
            v = d.get(key)
            if isinstance(v, list) and v and isinstance(v[-1], list) and v[-1]:
                return _finite(v[-1][-1])
            return None

        hw_ts = _last_ts("gyros") or _last_ts("accels")
        age = (time.time() - hw_ts) if hw_ts is not None else None
        
        yaw_rate = None
        gy = d.get("gyros")
        if isinstance(gy, list) and gy and isinstance(gy[-1], list) and len(gy[-1]) >= 3:
            yaw_rate = _finite(gy[-1][2])

        wheels = None
        rp = d.get("rpms")
        if isinstance(rp, list) and rp and isinstance(rp[-1], list):
            try:
                wheels = any(abs(float(x)) > 1e-6 for x in rp[-1][:-1])
            except (TypeError, ValueError, OverflowError):
                wheels = None

        return Telemetry(
            lat=lat if lat is not None else np.nan,
            lon=lon if lon is not None else np.nan,
            yaw=yaw,
            stamp=stamp,
            raw=d,
            hw_ts=hw_ts,
            age_s=age,
            yaw_rate=yaw_rate,
            wheels_turning=wheels,
        )

    def get_status(self) -> Optional[dict]:
        r = requests.get(f"{self.base}/status", timeout=self.timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def warm_up(self, timeout_s: float = 90.0) -> bool:
        """One /data GET with a long timeout, before the control loop starts.

        The first request makes the SDK launch headless Chrome, which takes far
        longer than the per-tick timeout. Paying it once here keeps the short
        timeout for every tick after.
        """
        try:
            r = requests.get(f"{self.base}/data", timeout=timeout_s)
            return r.status_code < 500
        except Exception:
            return False

    def ping(self) -> bool:
        """True if something is listening on http://localhost:8000."""
        try:
            r = requests.get(self.base, timeout=self.timeout)
            # 200 dashboard, or 400 "Call /start-mission..."
            return r.status_code < 500
        except Exception:
            return False

    # ---- missions ----------------------------------------------------------
    def list_missions(self) -> list[dict]:
        """GET /missions when available"""
        r = requests.get(f"{self.base}/missions", timeout=self.timeout)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        return r.json().get("missions", [])

    def start_mission(self) -> dict:
        """POST /start-mission"""
        r = requests.post(f"{self.base}/start-mission", timeout=self.timeout)
        try:
            body = r.json()
        except Exception:
            body = {"detail": r.text}
        if r.status_code >= 400:
            detail = body.get("detail", body)
            raise RuntimeError(f"start-mission failed ({r.status_code}): {detail}")
        return body

    def end_mission(self) -> dict:
        """POST /end-mission."""
        r = requests.post(
            f"{self.base}/end-mission", 
            timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def get_checkpoints(self) -> tuple[list[Checkpoint], int]:
        """GET /checkpoints-list"""
        r = requests.get(f"{self.base}/checkpoints-list", timeout=self.timeout)
        r.raise_for_status()
        d = r.json()
        raw_list = d.get("checkpoints_list") or []
        latest = d.get("latest_scanned_checkpoint") or 0

        cps: list[Checkpoint] = []
        for c in raw_list:
            cps.append(
                Checkpoint(
                    id=c.get("id"),
                    sequence=c.get("sequence"),
                    lat=float(c.get("latitude")),
                    lon=float(c.get("longitude")),
                )
            )
        cps.sort(key=lambda c: c.sequence) # Ascending for sanity-check
        return cps, latest

    def checkpoint_reached(self) -> dict:
        """POST /checkpoint-reached."""
        r = requests.post(
            f"{self.base}/checkpoint-reached",
            json={},
            timeout=self.timeout,
        )
        try:
            body = r.json()
        except Exception:
            body = {"detail": r.text}
        if r.status_code >= 400: # When outside the 15m proximity radius
            body.setdefault("_http_status", r.status_code)
        return body
