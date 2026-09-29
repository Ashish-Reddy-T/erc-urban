"""
Keep the earth-rovers-sdk alive under a run.

Wuhan, 2026-09-23: the SDK died twice in ~25 min, in two different ways:
  * /data hung ~23 s then returned `null` while /v2/front still served frames
    (the messaging session was gone, so /control was dead too);
  * its headless Chrome page closed (pyppeteer `Target closed`), and every
    endpoint failed after that.
The rover stopped safely both times (command expiry + link watchdog), but
nothing brought the SDK back, and after a restart a mission must be
re-started before the SDK answers anything. This does both.

What it kills is deliberately narrow: whatever listens on the SDK port, and
Chrome processes whose command line carries pyppeteer's `.dev_profile`
user-data-dir. Never a normal Chrome.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_SDK_DIR = _REPO / "earth-rovers-sdk"
PYPPETEER_MARK = "pyppeteer/.dev_profile"


def _pids(cmd: list[str]) -> list[int]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return []
    return [int(x) for x in out.split() if x.strip().isdigit()]


class SdkGuard:
    def __init__(self, url: str = "http://localhost:8000",
                 sdk_dir: Path = DEFAULT_SDK_DIR,
                 log_dir: Path = _REPO / "runs" / "sdk",
                 manage: bool = True):
        self.url = url.rstrip("/")
        self.port = int(self.url.rsplit(":", 1)[-1]) if ":" in self.url.split("//", 1)[-1] else 80
        self.sdk_dir = Path(sdk_dir)
        self.log_dir = Path(log_dir)
        self.manage = manage
        self.restarts = 0

    # ---- probes -----------------------------------------------------------
    def listening(self) -> bool:
        try:
            return requests.get(self.url, timeout=3).status_code < 500
        except Exception:
            return False

    def status(self, timeout: float = 15.0) -> str:
        """
        'ok'       /data returns telemetry
        'mission'  SDK up, waiting for /start-mission (400)
        'null'     /data answers but carries no telemetry (dead session)
        'down'     nothing answering / hanging
        """
        try:
            r = requests.get(f"{self.url}/data", timeout=timeout)
        except Exception:
            return "down"
        if r.status_code == 400 and "start-mission" in r.text:
            return "mission"
        if r.status_code != 200:
            return "down"
        try:
            d = r.json()
        except ValueError:
            return "null"
        return "ok" if isinstance(d, dict) and d else "null"

    # ---- process control --------------------------------------------------
    def _kill_sdk(self) -> None:
        for pid in _pids(["lsof", "-nP", f"-iTCP:{self.port}", "-sTCP:LISTEN", "-t"]):
            if pid == os.getpid():          # never ourselves (a fake SDK served in-process)
                continue
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        t0 = time.time()
        while time.time() - t0 < 8 and _pids(["lsof", "-nP", f"-iTCP:{self.port}", "-sTCP:LISTEN", "-t"]):
            time.sleep(0.3)
        for pid in _pids(["lsof", "-nP", f"-iTCP:{self.port}", "-sTCP:LISTEN", "-t"]):
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # The SDK's own headless Chrome outlives it (orphaned to launchd).
        for pid in _pids(["pgrep", "-f", PYPPETEER_MARK]):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def start(self) -> None:
        """Launch hypercorn detached, so it outlives this run for the next one."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = open(self.log_dir / time.strftime("sdk_%Y%m%d_%H%M%S.log"), "ab")
        hypercorn = Path(sys.executable).with_name("hypercorn")
        cmd = [str(hypercorn) if hypercorn.exists() else "hypercorn",
               "main:app", "--bind", f"127.0.0.1:{self.port}"]
        subprocess.Popen(cmd, cwd=self.sdk_dir, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)

    def wait_listening(self, timeout_s: float = 45.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if self.listening():
                return True
            time.sleep(1.0)
        return False

    def ensure_up(self) -> bool:
        """Start the SDK if nothing answers on its port. True when it answers."""
        if self.listening():
            return True
        if not self.manage:
            return False
        print(f"[SDK] nothing on {self.url} — starting earth-rovers-sdk", flush=True)
        self._kill_sdk()
        self.start()
        return self.wait_listening()

    def ensure_healthy(self, wait_s: float = 30.0) -> str:
        """
        Startup check: an SDK can be listening yet dead inside (the 'null'
        case). Give it `wait_s` to produce telemetry or ask for a mission, then
        restart it once.
        """
        if not self.ensure_up():
            return "down"
        t0 = time.time()
        while time.time() - t0 < wait_s:
            st = self.status()
            if st in ("ok", "mission"):
                return st
            time.sleep(3.0)
        print("[SDK] listening but no telemetry — restarting it", flush=True)
        return self.status() if self.restart() else "down"

    def restart(self) -> bool:
        if not self.manage:
            return False
        self.restarts += 1
        print(f"[SDK] restarting earth-rovers-sdk (restart #{self.restarts})", flush=True)
        self._kill_sdk()
        time.sleep(1.0)
        self.start()
        return self.wait_listening()

    # ---- the recovery a run needs ----------------------------------------
    def recover(self, client, mission_mode: bool, max_wait_s: float = 90.0) -> bool:
        """
        Bring telemetry back. Restarts the SDK only if it is genuinely dead;
        an SDK that is merely waiting for /start-mission just gets one.
        Returns True once /data carries telemetry again.
        """
        st = self.status()
        print(f"[SDK] recovery: /data status '{st}'", flush=True)
        if st in ("down", "null"):
            if not self.restart():
                print("[SDK] restart failed: SDK not answering", flush=True)
                return False
            st = self.status()
        t0 = time.time()
        while time.time() - t0 < max_wait_s:
            if st == "ok":
                print("[SDK] telemetry flowing again", flush=True)
                return True
            if st == "mission" and mission_mode:
                try:
                    resp = client.start_mission()
                    print(f"[SDK] start-mission (refresh): {resp.get('message', resp)}", flush=True)
                except Exception as e:
                    print(f"[SDK] start-mission failed: {e}", flush=True)
            time.sleep(3.0)
            st = self.status()
        print(f"[SDK] telemetry still '{st}' after {max_wait_s:.0f} s", flush=True)
        return False
