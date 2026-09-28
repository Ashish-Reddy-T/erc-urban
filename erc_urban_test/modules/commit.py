"""
The commit window: hold the chosen path for a bounded window, advancing it by
visual odometry and blending at the seam when a new plan arrives.

The loop runs near 1.5 Hz, not the nominal 5 Hz, and telemetry is ~1.3 s stale.
Re-planning from scratch every tick makes the command jump whenever perception
flickers, and leaves nothing to follow between plans. Committing also means a
one-frame segmentation glitch cannot swing the command.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class CommitWindow:
    """Holds a chosen path across ticks, advancing it by measured motion."""

    def __init__(
        self,
        max_age_s: float = 1.0,
        max_distance_m: float = 0.6,
        blend: float = 0.35,
        min_points: int = 2,
    ):
        """
        max_age_s / max_distance_m: how long a plan may be held. Short, because
            a stale plan is exactly what the traversability veto exists to
            prevent. Whichever bound trips first ends the commitment.
        blend: fraction of the NEW plan taken at the seam. 1.0 is plain
            re-planning (a jump); 0 never updates. 0.35 keeps continuity while
            still tracking a genuine change in the world within a few ticks.
        """
        self.max_age_s = max_age_s
        self.max_distance_m = max_distance_m
        self.blend = blend
        self.min_points = min_points
        self.path: Optional[np.ndarray] = None
        self.age_s = 0.0
        self.distance_m = 0.0
        self.last_reason = "empty"

    def reset(self) -> None:
        self.path = None
        self.age_s = 0.0
        self.distance_m = 0.0
        self.last_reason = "reset"

    @property
    def active(self) -> bool:
        return self.path is not None and len(self.path) >= self.min_points

    def advance(self, delta, dt: float) -> None:
        """
        Move the committed path into the current robot frame using measured
        motion. Without this the held path would stay pinned to where the rover
        was when it was planned, and following it would steer backwards.
        """
        self.age_s += max(dt, 0.0)
        if self.path is None:
            return
        if delta is None or delta.confidence <= 0.0:
            # No motion estimate: the path can no longer be trusted to be in
            # the right frame, so let it expire rather than follow a guess.
            self.last_reason = "no_odom"
            self.age_s = self.max_age_s + 1.0
            return
        c, s = np.cos(-delta.dyaw), np.sin(-delta.dyaw)
        R = np.array([[c, -s], [s, c]])
        self.path = (self.path - np.array([delta.dx, delta.dy])) @ R.T
        self.distance_m += delta.distance_m
        # Drop waypoints the rover has already driven past. If every waypoint
        # has fallen behind, this must also drop them (path becomes empty and
        # `active` goes False), not silently keep the stale, now-behind path:
        # `active` only checks length, so a stale full path would stay
        # "active" and get blended into the next offer() instead of expiring.
        keep = self.path[:, 0] > 0.0
        self.path = self.path[keep]

    def expired(self) -> bool:
        return (
            not self.active
            or self.age_s >= self.max_age_s
            or self.distance_m >= self.max_distance_m
        )

    def offer(self, new_path: np.ndarray) -> np.ndarray:
        """
        Give the window a freshly planned path; get back what to execute.

        While the commitment holds, the new plan is blended in at `blend`
        rather than replacing outright, which is the seam handling. When it has
        expired, the new plan is adopted wholesale.
        """
        new_path = np.asarray(new_path, dtype=np.float64)
        if not self.active or self.expired():
            self.path = new_path.copy()
            self.age_s = 0.0
            self.distance_m = 0.0
            self.last_reason = "adopted"
            return self.path

        n = min(len(self.path), len(new_path))
        if n < self.min_points:
            self.path = new_path.copy()
            self.age_s = 0.0
            self.distance_m = 0.0
            self.last_reason = "too_short"
            return self.path

        blended = (1.0 - self.blend) * self.path[:n] + self.blend * new_path[:n]
        self.path = blended
        self.last_reason = "held"
        return self.path

    def describe(self) -> str:
        if not self.active:
            return f"commit[{self.last_reason}]"
        return (f"commit[{self.last_reason} age={self.age_s:.1f}s "
                f"d={self.distance_m:.2f}m n={len(self.path)}]")
