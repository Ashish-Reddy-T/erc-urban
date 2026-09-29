"""
Closed-loop lane simulator for run_mexico: the real LaneFinder + Mexico.decide
drive a kinematic rover down a synthetic lane whose ground mask is rendered
through the real camera calibration.

    python -m erc_urban.sim_lane            # all scenarios, summary table

World: a lane (default 1.5 m wide) along a gently winding centreline ~150 m
north-east; outside it is grass (not ground). Checkpoints sit up to 5 m off the
lane centre like the real Mexico ones. The rover starts facing away. Compass:
true heading + noise, lagged 1 s. GPS: 1.5 m noise at 1 Hz. Motors: deadband.
"""
from __future__ import annotations

import collections
import math
import random
import types
from dataclasses import dataclass

import numpy as np

from erc_urban import run_mexico as M
from erc_urban.modules.lane import LaneFinder

LAT0, LON0 = 19.570217, -98.814691


def to_ll(e: float, n: float) -> tuple[float, float]:
    return LAT0 + n / 110540.0, LON0 + e / (111320.0 * math.cos(math.radians(LAT0)))


@dataclass
class LaneWorld:
    name: str
    width: float = 1.5
    bearing: float = 37.0        # lane direction, deg CW from north
    wiggle: float = 2.0          # lateral S-bend amplitude, m
    length: float = 150.0
    start_heading: float = 175.0
    cp_offset: float = 5.0       # CP distance from the lane centre, m
    gps_noise: float = 1.5
    compass_noise: float = 3.0
    compass_bias: float = 0.0    # systematic compass error, deg
    dropout: float = 0.0         # fraction of frames with no lane (lens drops, glare)
    blotches: int = 0            # random false non-ground blotches per frame
    seed: int = 0

    def centre(self, s: np.ndarray) -> np.ndarray:
        """(N,2) E,N of the centreline at along-lane distances s."""
        b = math.radians(self.bearing)
        fwd = np.array([math.sin(b), math.cos(b)])
        left = np.array([-math.cos(b), math.sin(b)])
        lat = self.wiggle * np.sin(2 * math.pi * s / 70.0)
        return s[:, None] * fwd + lat[:, None] * left

    def lateral(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(|offset from the centreline|, along-lane s) for (N,2) E,N points.
        Closed form: the wiggle's slope (2 m over 70 m) is small, so the offset
        across the lane axis is the distance to the centreline to within ~2 %."""
        b = math.radians(self.bearing)
        fwd = np.array([math.sin(b), math.cos(b)])
        left = np.array([-math.cos(b), math.sin(b)])
        s = pts @ fwd
        lat = pts @ left - self.wiggle * np.sin(2 * math.pi * s / 70.0)
        return np.abs(lat), s

    def offset(self, p: np.ndarray, s_hint: float = 0.0) -> tuple[float, float]:
        d, s = self.lateral(np.atleast_2d(p))
        return float(d[0]), float(s[0])


class LaneSim:
    def __init__(self, w: LaneWorld):
        self.w = w
        self.rng = random.Random(w.seed)
        self.lf = LaneFinder(M.CAL, hw=M.SEG_HW)
        h, wd = M.SEG_HW
        vs, us = np.mgrid[0:h, 0:wd]
        xy, ok = self.lf.cal.image_to_ground(np.stack([us.ravel(), vs.ravel()], 1).astype(float))
        far = np.hypot(xy[:, 0], xy[:, 1]) > 12.0
        self.pix_ok = ok & ~far
        self.pix_xy = xy                                   # rover frame (x fwd, y left)
        self.mask_shape = (h, wd)

    def mask(self, pos: np.ndarray, heading: float, s_hint: float) -> np.ndarray:
        """Ground mask seen from pos/heading: True where the ground point is in the lane."""
        t = math.radians(heading)
        fwd = np.array([math.sin(t), math.cos(t)])
        left = np.array([-math.cos(t), math.sin(t)])
        pts = pos + self.pix_xy[:, :1] * fwd + self.pix_xy[:, 1:2] * left
        d, _ = self.w.lateral(pts)
        g = (d <= self.w.width / 2) & self.pix_ok
        return g.reshape(self.mask_shape)

    def run(self, max_s: float = 900.0, dt: float = 0.2, verbose: bool = False) -> dict:
        w = self.w
        # checkpoints: CP1 at start (claimed), then every ~48 m, alternating side
        s_cps = [0.0, 48.0, 93.0, 149.0]
        cps = []
        for k, s in enumerate(s_cps):
            c = w.centre(np.array([s]))[0]
            b = math.radians(w.bearing)
            left = np.array([-math.cos(b), math.sin(b)])
            off = (w.cp_offset if k % 2 else -w.cp_offset) if k else 0.0
            e, n = c + off * left
            lat, lon = to_ll(e, n)
            cps.append(types.SimpleNamespace(sequence=k + 1, lat=lat, lon=lon, en=(e, n)))
        m = M.Mexico.__new__(M.Mexico)
        m.turning, m.fwd_since, m.last_rev = 0.0, None, 0.0
        pos = w.centre(np.array([8.0]))[0].astype(float)   # start 8 m up the lane from CP1
        pos = pos + 0.0
        heading = w.start_heading
        s_hint = 8.0
        comp_buf = collections.deque([heading] * int(1.0 / dt), maxlen=int(1.0 / dt))
        gps = pos.copy()
        remaining = cps[1:]
        t = 0.0
        max_off, off_lane_s, path = 0.0, 0.0, 0.0
        claims = []
        while remaining and t < max_s:
            goal = remaining[0]
            prev = cps[goal.sequence - 2]
            if int(t / 1.0) != int((t - dt) / 1.0):            # GPS at 1 Hz
                gps = pos + np.array([self.rng.gauss(0, w.gps_noise), self.rng.gauss(0, w.gps_noise)])
            lat, lon = to_ll(*gps)
            rng, _ = M.geo(lat, lon, goal.lat, goal.lon)
            true_rng = math.hypot(pos[0] - goal.en[0], pos[1] - goal.en[1])
            if rng <= M.CLAIM_M and true_rng <= 15.0:
                claims.append((goal.sequence, round(true_rng, 1), round(t)))
                remaining.pop(0)
                continue
            compass = (comp_buf[0] + w.compass_bias + self.rng.gauss(0, w.compass_noise)) % 360.0
            lane_brg = M.pursuit_bearing(lat, lon, prev, goal, M.PURSUIT_M)
            prior = M.rel_left(compass, lane_brg)
            g = self.mask(pos, heading, s_hint)
            if self.rng.random() < w.dropout:
                g = np.zeros_like(g)
            for _ in range(w.blotches):
                h_, w_ = g.shape
                r0, c0 = self.rng.randrange(h_ // 2, h_), self.rng.randrange(0, w_)
                g[max(0, r0 - 6):r0 + 6, max(0, c0 - 10):c0 + 10] = False
            e = self.lf.estimate(g, prior_deg=float(np.clip(prior, -45.0, 45.0)))
            tel = types.SimpleNamespace(raw={"rpms": [[50, 50, 50, 50, 0]]})
            lin, ang, mode = m.decide(e, prior, tel, t)
            # kinematics with deadband
            v = lin * 1.3 if abs(lin) >= 0.1 else 0.0
            om = ang * 45.0 if abs(ang) >= 0.1 else 0.0          # deg/s, + = left
            heading = (heading - om * dt) % 360.0
            hb = math.radians(heading)
            step = v * dt * np.array([math.sin(hb), math.cos(hb)])
            pos = pos + step
            path += abs(v) * dt
            comp_buf.append(heading)
            off, s_hint = w.offset(pos, s_hint)
            max_off = max(max_off, off) if t > 20 else max_off     # ignore the turn-around
            if off > w.width / 2 - 0.17:
                off_lane_s += dt
            if verbose and int(t / 5) != int((t - dt) / 5):
                print(f"t={t:5.1f} s={s_hint:6.1f} off={off:.2f} hdg={heading:5.1f} prior={prior:+.0f} {mode}")
            t += dt
        return {"name": w.name, "claims": claims, "done": not remaining, "t": round(t), "path": round(path),
                "max_off": round(max_off, 2), "wheels_off_lane_s": round(off_lane_s, 1)}


SCENARIOS = [
    LaneWorld("straight 1.5 m, facing away"),
    LaneWorld("narrow 1.2 m", width=1.2),
    LaneWorld("noisy compass 8 deg, GPS 3 m", compass_noise=8.0, gps_noise=3.0, seed=1),
    LaneWorld("CPs 6 m off lane", cp_offset=6.0, seed=2),
    LaneWorld("start facing along lane", start_heading=40.0, seed=3),
    LaneWorld("very narrow 1.0 m", width=1.0, seed=4),
    LaneWorld("sharper bends (4 m / 70 m)", wiggle=4.0, seed=5),
    LaneWorld("compass 15 deg off", compass_bias=15.0, seed=6),
    LaneWorld("30% frames lane-blind", dropout=0.3, seed=7),
    LaneWorld("false blotches x3", blotches=3, seed=8),
    LaneWorld("open ground (20 m wide)", width=20.0, seed=10),
    LaneWorld("all of it", width=1.2, wiggle=3.0, compass_bias=10.0, compass_noise=6.0,
              gps_noise=2.5, dropout=0.2, blotches=2, seed=9),
]


def main():
    print(f"{'scenario':32s} {'done':5s} {'t(s)':>5s} {'path':>5s} {'max_off':>7s} {'off-lane s':>10s} claims")
    for w in SCENARIOS:
        r = LaneSim(w).run()
        print(f"{r['name']:32s} {str(r['done']):5s} {r['t']:5d} {r['path']:5d} {r['max_off']:7.2f} "
              f"{r['wheels_off_lane_s']:10.1f} {r['claims']}")


if __name__ == "__main__":
    main()
