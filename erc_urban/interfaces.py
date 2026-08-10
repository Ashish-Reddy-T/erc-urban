"""
The contract. Every module is swappable as long as it honors these types.
This is the single most important file: it is why the baseline and every
upgrade can be developed and tested independently.

A "candidate" is one hypothesized short-horizon path in the ROBOT frame:
an (N,2) array of (x_forward, y_left) waypoints in metres, plus a scalar
score from whatever produced it. The entire Slow-Brain-Fast-Planner /
traversability / fusion machinery operates on SETS of these.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol
import numpy as np


@dataclass
class Candidate:
    waypoints: np.ndarray          # (N, 2) float, robot frame, metres
    score: float                   # higher = better (planner's own confidence)
    stamp: float                   # capture time (our monotonic clock)


@dataclass
class Observation:
    front: np.ndarray              # HxWx3 uint8
    rear: Optional[np.ndarray]     # HxWx3 uint8 or None
    lat: float
    lon: float
    yaw: Optional[float]
    stamp: float
    # goal as a GPS checkpoint (urban track). Bearing/range are precomputed
    # relative to current pose so policies get a clean directional prior.
    goal_bearing: float = 0.0      # radians, robot frame (0 = straight ahead)
    goal_range_m: float = 0.0


class Planner(Protocol):
    """Module 1: emits K scored candidate trajectories per tick."""
    def propose(self, obs: Observation, k: int) -> list[Candidate]: ...


class Selector(Protocol):
    """Module 4: picks / re-weights among candidates (may be async+stale)."""
    def select(self, obs: Observation, cands: list[Candidate]) -> int: ...


class SafetyGate(Protocol):
    """Module 6: returns a boolean mask, True = candidate is traversable/safe."""
    def mask(self, obs: Observation, cands: list[Candidate]) -> np.ndarray: ...


class Controller(Protocol):
    """Module 7: converts a chosen waypoint path to (linear, angular)."""
    def command(self, waypoints: np.ndarray) -> tuple[float, float]: ...