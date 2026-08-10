"""
BASELINE (Stage 0): a runnable, ML-free loop that reaches GPS checkpoints.

This is deliberately dumb but COMPLETE — it closes the loop end to end so you
can validate the SDK harness, the control interface, and the scoring
before any learned component exists. Every later stage replaces one piece
of THIS with a paper-derived module, one at a time.

M1 (BaselinePlanner): fan out K constant-curvature arcs; score each by how
    well its endpoint bearing matches the goal bearing.
M7 (PurePursuitController): waypoint tracking for later stages.
VillageGotoController: Stage-0 heading creep for rough village dirt paths.
"""
from __future__ import annotations
import numpy as np
from ..interfaces import Candidate, Observation


class BaselinePlanner:
    def __init__(
        self,
        horizon_m: float = 3.0,
        n_pts: int = 8,
        max_curv: float = 0.9,
    ):
        self.horizon_m = horizon_m
        self.n_pts = n_pts
        self.max_curv = max_curv

    def propose(self, obs: Observation, k: int) -> list[Candidate]:
        curvs = np.linspace(-self.max_curv, self.max_curv, k)
        s = np.linspace(0.0, self.horizon_m, self.n_pts)
        cands: list[Candidate] = []
        for c in curvs:
            if abs(c) < 1e-6:
                xy = np.stack([s, np.zeros_like(s)], axis=1)
            else:
                r = 1.0 / c
                th = s / r
                xy = np.stack([r * np.sin(th), r * (1 - np.cos(th))], axis=1)
            end_bearing = np.arctan2(xy[-1, 1], xy[-1, 0])
            score = -abs(_wrap(end_bearing - obs.goal_bearing))
            cands.append(
                Candidate(waypoints=xy, score=float(score), stamp=obs.stamp)
            )
        return cands


class PurePursuitController:
    def __init__(
        self,
        lookahead_m: float = 1.0,
        v_nom: float = 0.22,
        wheelbase_scale: float = 0.35,
        max_angular: float = 0.25,
    ):
        self.lookahead_m = lookahead_m
        self.v_nom = v_nom
        self.k = wheelbase_scale
        self.max_angular = max_angular

    def command(self, waypoints: np.ndarray) -> tuple[float, float]:
        d = np.linalg.norm(waypoints, axis=1)
        idx = np.argmax(d >= self.lookahead_m)
        target = waypoints[idx] if d[idx] >= self.lookahead_m else waypoints[-1]
        x, y = float(target[0]), float(target[1])
        L2 = x * x + y * y
        curv = 0.0 if L2 < 1e-6 else (2.0 * y) / L2
        angular = float(np.clip(self.k * curv, -self.max_angular, self.max_angular))
        # Floor was 0.4: a saturated turn cut throttle to 40% of v_nom on every
        # LogoNav tick, so the rover crept and GPS never showed progress.
        turn_scale = max(0.6, 1.0 - abs(angular) / max(self.max_angular, 1e-6))
        linear = float(np.clip(self.v_nom * turn_scale, 0.0, 1.0))
        return linear, angular


class VillageGotoController:
    """
    Stage-0 creep controller. Prefer fused bearing from dirt-path + GPS
    (see run.py). Never drives when caller passes blocked=True.
    """

    def __init__(
        self,
        v_creep: float = 0.22,
        align_deg: float = 35.0,
        ok_deg: float = 15.0,
        max_ang_pivot: float = 0.14,
        max_ang_drive: float = 0.12,
        kp_deg: float = 0.006,
    ):
        self.v_creep = v_creep
        self.align_deg = align_deg
        self.ok_deg = ok_deg
        self.max_ang_pivot = max_ang_pivot
        self.max_ang_drive = max_ang_drive
        self.kp_deg = kp_deg

    def command_bearing(
        self,
        bearing_rad: float,
        range_m: float,
        blocked: bool = False,
    ) -> tuple[float, float]:
        if blocked:
            return 0.0, 0.0

        err_deg = float(np.degrees(bearing_rad))
        abs_err = abs(err_deg)
        sign = 1.0 if err_deg >= 0.0 else -1.0

        if abs_err > self.align_deg:
            angular = sign * self.max_ang_pivot
            linear = 0.0
        elif abs_err > self.ok_deg:
            angular = float(
                np.clip(err_deg * self.kp_deg, -self.max_ang_drive, self.max_ang_drive)
            )
            linear = self.v_creep * 0.7
        else:
            angular = float(
                np.clip(err_deg * self.kp_deg, -self.max_ang_drive, self.max_ang_drive)
            )
            linear = self.v_creep * max(0.6, 1.0 - abs_err / 90.0)

        if range_m < 25.0:
            linear *= 0.75
        return float(np.clip(linear, 0.0, 1.0)), float(np.clip(angular, -1.0, 1.0))


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
