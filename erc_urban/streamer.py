"""
Command streaming: the decision loop SETS a command with an expiry, a thread
keeps POSTing it to /control until it expires, then posts zeros.

Copied from erc_indoor/run.py (proven on the Mini+ over full indoor laps)
rather than imported, because importing erc_indoor.run drags in the whole
indoor stack and its module globals (LAMP=0 there).

Why it matters here: the Mini+ has a command watchdog, so command RATE is
effective speed (one POST gave +0.7 deg of yaw, the same command at 5 Hz
+18.6 deg). The urban tick is ~0.7 s, so posting once per tick starved it. And
if the loop hangs, the rover stops when the hold runs out instead of whenever
the firmware gives up.
"""
from __future__ import annotations

import threading
import time
from typing import Optional


class LinkWatchdog:
    """
    Is anything actually changing? Neither the image nor the IMU timestamp
    moving for `stale_s` while we command motion means the link is frozen, and
    the loop is deciding on a picture of the past.

    Ported from erc_indoor/run.py. Indoor run 4: the bot stopped answering,
    frames byte-static, IMU 350-430 s old, and the loop kept deciding.
    """

    def __init__(self, stale_s: float = 4.0, diff_thresh: float = 1.5):
        self.stale_s = stale_s
        self.diff_thresh = diff_thresh
        self._small = None
        self._hw = 0.0
        self._last_change = time.time()
        self._moving_since: Optional[float] = None
        self.down = False

    def update(self, front, hw_ts: Optional[float], moving: bool,
               now: Optional[float] = None) -> bool:
        """Feed this tick's frame and IMU stamp. Returns True if the link is stale."""
        import numpy as np
        from PIL import Image

        now = time.time() if now is None else now
        changed = False
        if front is not None:
            small = np.asarray(
                Image.fromarray(front).resize((64, 36)).convert("L"), np.float32
            )
            if self._small is None or np.abs(small - self._small).mean() > self.diff_thresh:
                changed = True
            self._small = small
        if hw_ts is not None and hw_ts > self._hw:
            self._hw = hw_ts
            changed = True
        if changed:
            self._last_change = now
        if self.down:
            # Halting made `moving` False; that must not read as recovery.
            # Stay down until something genuinely new arrives.
            if not changed:
                return True
            self.down = False
            self._moving_since = None
            return False
        if not moving:
            self._moving_since = None
            return False
        if self._moving_since is None:
            self._moving_since = now
        if now - max(self._last_change, self._moving_since) > self.stale_s:
            self.down = True
            return True
        return False


class CommandStreamer(threading.Thread):
    """Sends the latest command at `hz` until it expires, then zeros."""

    def __init__(self, client, send: bool = True, hz: float = 10.0,
                 lamp: Optional[int] = None):
        super().__init__(daemon=True)
        self.client = client
        self.send = send
        self.hz = hz
        self.lamp = lamp
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

    # Both are about what the ROVER is being told. In a dry run (send=False) it
    # is told nothing, so neither the link watchdog nor the dead-rpm check may
    # expect motion.
    @property
    def forward(self) -> bool:
        """True while a forward (lin > 0) command is actually being sent."""
        with self._lock:
            return self.send and time.time() < self._expires and self._cmd[0] > 1e-3

    @property
    def moving(self) -> bool:
        """True while a non-zero command is actually being sent."""
        with self._lock:
            lin, ang = self._cmd
            return (self.send and time.time() < self._expires
                    and (abs(lin) > 1e-3 or abs(ang) > 1e-3))

    def _post(self, lin: float, ang: float, lamp: Optional[int] = None) -> None:
        if not self.send:
            return
        try:
            self.client.send_cmd(lin, ang, lamp=self.lamp if lamp is None else lamp)
        except Exception:
            pass

    def run(self) -> None:
        while self._running:
            t0 = time.time()
            with self._lock:
                lin, ang = self._cmd if t0 < self._expires else (0.0, 0.0)
            self._post(lin, ang)
            time.sleep(max(0.0, 1.0 / self.hz - (time.time() - t0)))

    def stop(self, lamp_off: bool = False) -> None:
        """Stop streaming and leave the rover halted (lamp off if asked)."""
        self._running = False
        for _ in range(3):
            self.set(0.0, 0.0, 0.0)
            self._post(0.0, 0.0, lamp=0 if lamp_off else None)
            time.sleep(0.1)
