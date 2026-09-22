"""
A "candidate" is one hypothesised short-horizon path in the ROBOT frame:
an (N,2) array of (x_forward, y_left) waypoints in metres, plus a scalar score
from whatever produced it. The planning, traversability and fusion machinery
all operate on SETS of these.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Candidate:
    waypoints: np.ndarray          # (N, 2) float, robot frame, metres
    score: float                   # higher = better (planner's confidence metric)
    stamp: float                   # capture time


@dataclass
class Observation:
    front: np.ndarray              # HxWx3 uint8
    rear: Optional[np.ndarray]     # HxWx3 uint8 or None
    lat: float
    lon: float
    yaw: Optional[float]
    stamp: float
    # Goal bearing/range are precomputed relative to current pose, so policies
    # get a clean directional prior rather than a raw fix.
    goal_bearing: float = 0.0      # radians, robot frame (0 = straight ahead)
    goal_range_m: float = 0.0
