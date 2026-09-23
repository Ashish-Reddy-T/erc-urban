"""
Where along the teach lap am I? A 1-D histogram filter over route-map nodes.

  predict(dp)   shift belief forward by odometry progress (wheels + turned
                degrees, same units as RouteMap.p), blurred by odometry noise
  update(desc)  multiply by place-recognition likelihood against every node

Why a filter and not nearest-neighbour retrieval: the long white corridors all
look alike (frames 800-1100 of the teach lap), so single-frame retrieval jumps
between them. Odometry is good to a few percent over tens of metres and
carries the estimate through aliased stretches; place recognition pins it down
wherever the view is distinctive (atrium, piano, chairs, exit signs).

The likelihood has a floor, so one bad match can only dent the belief, never
erase the true position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Estimate:
    node: int
    p: float
    s: float
    spread_m: float      # std of belief around the peak, metres of progress
    sim: float           # best descriptor similarity at the estimate

    @property
    def confident(self) -> bool:
        return self.spread_m < 2.0


class RouteLocalizer:
    def __init__(
        self,
        route,
        tau: float = 0.02,
        lik_floor: float = 0.05,
        sigma_base_m: float = 0.15,
        sigma_frac: float = 0.15,
        uniform_mix: float = 1e-5,
    ):
        self.route = route
        self.p = route.p.astype(np.float64)
        self.D = route.desc
        self.tau = tau
        self.lik_floor = lik_floor
        self.sigma_base_m = sigma_base_m
        self.sigma_frac = sigma_frac
        self.uniform_mix = uniform_mix
        self.b = np.full(len(self.p), 1.0 / len(self.p))
        self.last_sims: np.ndarray | None = None

    def reset(self, p0: float = 0.0, sigma_m: float = 3.0) -> None:
        b = np.exp(-0.5 * ((self.p - p0) / sigma_m) ** 2)
        self.b = b / b.sum()

    def predict(self, dp: float) -> None:
        sig = self.sigma_base_m + self.sigma_frac * abs(dp)
        src = np.nonzero(self.b > 1e-7)[0]
        if len(src) == 0:
            self.b[:] = 1.0 / len(self.b)
            return
        mu = self.p[src] + dp
        # deposit each source's mass onto nodes near its shifted position
        new = np.zeros_like(self.b)
        lo = np.searchsorted(self.p, mu - 4 * sig)
        hi = np.searchsorted(self.p, mu + 4 * sig)
        for j, a, c, w in zip(range(len(src)), lo, hi, self.b[src]):
            if c <= a:
                k = min(max(a, 0), len(self.p) - 1)
                new[k] += w
                continue
            ker = np.exp(-0.5 * ((self.p[a:c] - mu[j]) / sig) ** 2)
            new[a:c] += w * ker / ker.sum()
        new += self.uniform_mix / len(new)
        self.b = new / new.sum()

    def update(self, desc: np.ndarray) -> None:
        sims = self.D @ desc
        if getattr(self.route, "twin_desc", None) is not None:
            # best of the day and night picture of each place
            sims = np.maximum(sims, self.route.twin_desc @ desc)
        self.last_sims = sims
        lik = np.exp((sims - sims.max()) / self.tau) + self.lik_floor
        b = self.b * lik
        self.b = b / b.sum()

    def constrain(self, lo: float, hi: float, floor: float = 1e-9) -> None:
        """Belief outside [lo, hi] is (almost) removed. The navigator derives
        the window from the corner sequence: after corner k the rover cannot be
        before it. Live run 8: at the turn-4 service doorway the filter jumped
        from s=130 to s=33.6 (the piano doorway looks alike) within seconds."""
        m = (self.p >= lo) & (self.p <= hi)
        if not m.any():
            return
        b = np.where(m, self.b, self.b * floor)
        self.b = b / b.sum()

    def anchor(self, p: float, sigma_m: float = 1.5, floor: float = 0.02) -> None:
        """Soft prior after an event we KNOW happened (a completed corner)."""
        g = np.exp(-0.5 * ((self.p - p) / sigma_m) ** 2) + floor
        b = self.b * g
        self.b = b / b.sum()

    def estimate(self) -> Estimate:
        k = int(self.b.argmax())
        pk = self.p[k]
        win = np.abs(self.p - pk) < 5.0
        w = self.b[win] / self.b[win].sum()
        pm = float((w * self.p[win]).sum())
        spread = float(np.sqrt((w * (self.p[win] - pm) ** 2).sum()))
        # outside mass also counts as spread (multi-modal belief)
        out_mass = 1.0 - self.b[win].sum()
        spread += 20.0 * out_mass
        node = self.route.node_at_p(pm)
        sim = float(self.last_sims[node]) if self.last_sims is not None else 0.0
        return Estimate(node, pm, float(self.route.s[node]), spread, sim)
