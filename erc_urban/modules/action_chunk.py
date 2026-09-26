"""
Action-chunk latency compensation helpers.

When a policy emits H actions at fixed dt (e.g. 10 steps at 10 Hz), inference and
transport delay mean the first executable command is not chunk[0] but chunk[k] for
elapsed time since the chunk was produced. NavigationScaling and Real-Time Chunking
use the same idea; this module is a small pure index selector for future NS wiring.
"""
from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np


def chunk_index(elapsed_s: float, *, dt_s: float = 0.1, horizon: int = 10) -> int:
    """Index into an action chunk given time since chunk was produced.

    Clamps to [0, horizon-1]. Non-finite elapsed -> 0. Negative -> 0.
    """
    if elapsed_s < 0 or not math.isfinite(elapsed_s):
        return 0
    if horizon < 1:
        return 0
    if dt_s <= 0 or not math.isfinite(dt_s):
        return 0
    idx = int(elapsed_s / dt_s)
    return min(max(idx, 0), horizon - 1)


def select_action(chunk: Any, elapsed_s: float, *, dt_s: float = 0.1) -> np.ndarray:
    """chunk shape (H, A) or (H,); return row at chunk_index."""
    arr = np.asarray(chunk)
    if arr.ndim == 0:
        horizon = 1
    else:
        horizon = int(arr.shape[0])
    idx = chunk_index(elapsed_s, dt_s=dt_s, horizon=horizon)
    return np.asarray(arr[idx])


class ActionChunkExecutor:
    """Holds latest chunk + t0 monotonic; step(now) returns action for elapsed."""

    def __init__(self, *, dt_s: float = 0.1) -> None:
        self.dt_s = dt_s
        self._chunk: Optional[np.ndarray] = None
        self._t0: Optional[float] = None

    def push(self, chunk: Any, t0: float | None = None) -> None:
        self._chunk = np.asarray(chunk)
        self._t0 = t0 if t0 is not None else time.monotonic()

    def step(self, now: float | None = None) -> np.ndarray | None:
        if self._chunk is None or self._t0 is None:
            return None
        t = now if now is not None else time.monotonic()
        return select_action(self._chunk, t - self._t0, dt_s=self.dt_s)
