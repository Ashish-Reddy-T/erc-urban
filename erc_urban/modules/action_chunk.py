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
    # Epsilon absorbs float error at exact boundaries (0.3 / 0.1 == 2.9999999999999996).
    ratio = elapsed_s / dt_s + 1e-9
    if not math.isfinite(ratio):
        return horizon - 1
    idx = int(math.floor(ratio))
    return min(max(idx, 0), horizon - 1)


def _as_chunk(chunk: Any) -> np.ndarray:
    arr = np.asarray(chunk)
    if arr.ndim == 0 or arr.shape[0] < 1:
        raise ValueError(
            f"action chunk must have at least one row, got shape {arr.shape}"
        )
    return arr


def select_action(chunk: Any, elapsed_s: float, *, dt_s: float = 0.1) -> np.ndarray:
    """chunk shape (H, A) or (H,) with H >= 1; return a copy of the row at chunk_index.

    Raises ValueError for an empty or 0-d chunk.
    """
    arr = _as_chunk(chunk)
    idx = chunk_index(elapsed_s, dt_s=dt_s, horizon=int(arr.shape[0]))
    return np.array(arr[idx], copy=True)


class ActionChunkExecutor:
    """Holds latest chunk + t0; step(now) returns a copy of the action for elapsed.

    Clock contract: t0 and now must come from the same clock. The planner stamps
    time.monotonic() (the default when t0/now are omitted).

    push() stores a copy and rejects empty or 0-d chunks with ValueError.

    Elapsed past the chunk end returns the last row for finite elapsed, but row 0
    for non-finite elapsed (inf, NaN), same as negative elapsed.

    There is no staleness cutoff or clear() yet: step() keeps returning a row of the
    last chunk forever. The caller must enforce max age and stop behavior; max_age_s,
    the negative-age check and clear() are deferred to the NS wiring PR.
    """

    def __init__(self, *, dt_s: float = 0.1) -> None:
        self.dt_s = dt_s
        self._chunk: Optional[np.ndarray] = None
        self._t0: Optional[float] = None

    def push(self, chunk: Any, t0: float | None = None) -> None:
        self._chunk = np.array(_as_chunk(chunk), copy=True)
        self._t0 = t0 if t0 is not None else time.monotonic()

    def step(self, now: float | None = None) -> np.ndarray | None:
        if self._chunk is None or self._t0 is None:
            return None
        t = now if now is not None else time.monotonic()
        return select_action(self._chunk, t - self._t0, dt_s=self.dt_s)
