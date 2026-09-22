"""
BaselinePlanner: a fan of K constant-curvature arcs scored by endpoint bearing.
    Used at every stage, not just Stage 0 — it is the anchor fan that
    MultiHypothesisPlanner ranks with LogoNav, and LogoNav's own fallback.
PurePursuitController: waypoint tracking.
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
        self.horizon_m = horizon_m    # distance to look ahead
        self.n_pts = n_pts            # number of points to generate for representation of the arc upto `horizon_m`
        self.max_curv = max_curv      # maximum curvature allowed for the arc

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
        self.lookahead_m = lookahead_m  # distance to look ahead for the target waypoint
        self.v_nom = v_nom              # nominal velocity
        self.k = wheelbase_scale        # wheelbase scale factor
        self.max_angular = max_angular  # maximum angular velocity

    def command(self, waypoints: np.ndarray) -> tuple[float, float]:
        d = np.linalg.norm(waypoints, axis=1)
        idx = np.argmax(d >= self.lookahead_m)
        target = waypoints[idx] if d[idx] >= self.lookahead_m else waypoints[-1]
        x, y = float(target[0]), float(target[1])
        L2 = x * x + y * y
        curv = 0.0 if L2 < 1e-6 else (2.0 * y) / L2
        angular = float(np.clip(self.k * curv, -self.max_angular, self.max_angular))
        # 0.6 floor, not 0.4: at 0.4 a saturated turn cut throttle to 40% of - changed to 0.7
        # v_nom on every LogoNav tick and the rover crept.
        turn_scale = max(0.7, 1.0 - abs(angular) / max(self.max_angular, 1e-6)) # Reduce speed linearly to 0.7 min as angular (turn rate) increases
        linear = float(np.clip(self.v_nom * turn_scale, 0.0, 1.0))
        return linear, angular


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
