"""
Closed-loop sanity check of the NAV=fan decision code (run._fan_step) in a
flat 2D world of wall segments. Ground masks are rendered through the real
camera calibration, so FreeSpaceUrban, obstacle memory, escapes and pulse
pivots all run as on the rover; only the rover and the link are faked
(no latency, no GPS noise).

    python -m erc_urban.sim_fan            # all scenarios, summary table
    python -m erc_urban.sim_fan wall -v    # one scenario, per-tick log
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import erc_urban.run as R
from erc_urban.modules.obstacle_memory import ObstacleMemory
from erc_urban.modules.odometry import OdomDelta

SPEED_PER_UNIT = 1.37      # m/s per unit linear (0.3 -> 0.41 m/s, Wuhan Mini+)
YAW_PER_UNIT = 37.0        # deg/s per unit angular (streamed, Wuhan Mini+)
BODY_R = 0.18              # m, collision radius (Mini+ ~0.29 x 0.19 m, half-diagonal)


@dataclass
class World:
    walls: list                                  # [((e0, n0), (e1, n1)), ...]
    start: tuple = (0.0, 0.0, 0.0)               # e, n, yaw deg (CW from north)
    goal: tuple = (0.0, 10.0)
    max_ticks: int = 900
    name: str = ""
    grass: bool = False                          # all drivable ground is grass


def _ray(e, n, hdg_rad, walls, far=30.0):
    """Distance along world heading (CW from north) to the nearest wall."""
    dx, dy = math.sin(hdg_rad), math.cos(hdg_rad)
    best = far
    for (x0, y0), (x1, y1) in walls:
        ex, ey = x1 - x0, y1 - y0
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:
            continue
        t = ((x0 - e) * ey - (y0 - n) * ex) / den
        u = ((x0 - e) * dy - (y0 - n) * dx) / den
        if t > 0 and 0 <= u <= 1:
            best = min(best, t)
    return best


def _seg_dist(p, a, b):
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def render_ground(fsu, e, n, yaw_rad, walls):
    """Ground mask at fsu's seg size: ground up to the wall foot per column."""
    g = np.zeros((fsu.h, fsu.w), bool)
    hz = int(math.ceil(fsu.horizon)) + 1
    for u in range(fsu.w):
        a = float(fsu._az[u])                      # left-positive, robot frame
        r = _ray(e, n, yaw_rad - a, walls)
        col_r = np.hypot(fsu._gx[:, u], fsu._gy[:, u])
        ok = fsu._gok[:, u] & (np.arange(fsu.h) > hz) & (col_r < r)
        g[:, u] = ok
    return g


@dataclass
class Result:
    reached: bool
    ticks: int
    min_goal_m: float
    collisions: int
    escapes: int
    boxed_exit: bool
    path_m: float
    trail: list = field(default_factory=list)


def run(world: World, verbose: bool = False) -> Result:
    e, n, yaw_deg = world.start
    R._MEM = ObstacleMemory() if R.FAN_MEM else None
    R._CRUMBS.travelled_m = 0.0
    R._FSU = None
    stuck_state: dict = {}
    t = [0.0]
    cmd = {"lin": 0.0, "ang": 0.0, "hold": None}
    last = OdomDelta(0.0, 0.0, 0.0, 1.0, reason="sim")
    collisions = escapes = 0
    path = 0.0
    trail = [(e, n)]
    min_goal = 1e9
    boxed = False

    def _cmd(client, lin, ang, hold_s=None):
        cmd.update(lin=lin, ang=ang, hold=hold_s)

    def _halt(client):
        cmd.update(lin=0.0, ang=0.0, hold=None)

    dt = 0.35
    with patch.object(R, "_cmd", _cmd), patch.object(R, "_halt", _halt), \
            patch.object(R, "_publish_overlay", lambda *a, **k: None), \
            patch.object(R.time, "monotonic", lambda: t[0]):
        for k in range(world.max_ticks):
            gdx, gdy = world.goal[0] - e, world.goal[1] - n
            rng = math.hypot(gdx, gdy)
            min_goal = min(min_goal, rng)
            if rng < 1.0:
                return Result(True, k, min_goal, collisions, escapes, False, path, trail)
            g_hdg = math.degrees(math.atan2(gdx, gdy))
            rel = ((yaw_deg - g_hdg + 180.0) % 360.0) - 180.0      # left-positive
            aim = SimpleNamespace(bearing=math.radians(rel), speed_scale=1.0, source="gps")
            tel = SimpleNamespace(yaw=math.radians(yaw_deg), hw_ts=t[0])
            if R._FSU is None:
                R._FSU = R.FreeSpaceUrban(R.CALIB, seg_hw=R.FAN_SEG_HW)
            ground = render_ground(R._FSU, e, n, math.radians(yaw_deg), world.walls)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    if world.grass:
                        R._fan_step(None, tel, None, rng, math.radians(rel), aim,
                                    np.zeros_like(ground), ground, stuck_state, last, "", 0)
                    else:
                        R._fan_step(None, tel, None, rng, math.radians(rel), aim, ground, None,
                                    stuck_state, last, "", 0)
            except SystemExit:
                boxed = True
                if verbose:
                    print(buf.getvalue(), end="")
                break
            out = buf.getvalue()
            escapes += out.count("ESCAPE:")
            if verbose:
                print(f"[{k:4d}] e={e:+.2f} n={n:+.2f} yaw={yaw_deg:6.1f} " + out.strip().replace("\n", " | ")[:230])
            # Apply the command for this tick.
            apply = dt if cmd["hold"] is None else min(dt, max(cmd["hold"], 0.0))
            dyaw = -cmd["ang"] * YAW_PER_UNIT * apply           # yaw is CW
            ds = cmd["lin"] * SPEED_PER_UNIT * apply
            ny = yaw_deg + dyaw
            ne = e + ds * math.sin(math.radians(ny))
            nn = n + ds * math.cos(math.radians(ny))
            hit = any(_seg_dist((ne, nn), a, b) < BODY_R for a, b in world.walls)
            if hit and ds != 0.0:
                collisions += 1
                if verbose:
                    print(f"       HIT at e={ne:+.2f} n={nn:+.2f} yaw={ny:.0f} ds={ds:+.3f}")
                ds, ne, nn = 0.0, e, n
            e, n, yaw_deg = ne, nn, ny % 360.0
            path += abs(ds)
            R._CRUMBS.travelled_m += abs(ds)
            last = OdomDelta(ds, 0.0, math.radians(-dyaw), 1.0, reason="wheel")
            trail.append((e, n))
            t[0] += dt
    return Result(False, k + 1, min_goal, collisions, escapes, boxed, path, trail)


def _box(e0, n0, e1, n1):
    return [((e0, n0), (e1, n0)), ((e1, n0), (e1, n1)), ((e1, n1), (e0, n1)), ((e0, n1), (e0, n0))]


SCENARIOS = {
    "open": World([], (0, 0, 0), (0, 10), 400, "open ground, goal ahead"),
    "behind": World([], (0, 0, 180), (0, 8), 400, "goal behind"),
    "wall": World([((-6, 3), (6, 3))], (0, 0, 0), (0, 9), 900,
                  "12 m wall across the goal line (CP9 stairs)"),
    "offset_wall": World([((-9, 3), (2.5, 3))], (0, 0, 0), (0, 9), 900,
                         "wall ending 2.5 m right of the goal line"),
    "gap": World([((-8, 3), (-0.2, 3)), ((1.4, 3), (8, 3))], (0, 0, 0), (0, 9), 900,
                 "wall with a 1.6 m gap just right of the line"),
    "pocket": World([((-1.2, 2.5), (1.2, 2.5)), ((-1.2, -1.5), (-1.2, 2.5)),
                     ((1.2, -1.5), (1.2, 2.5))], (0, 0, 0), (0, 9), 900,
                    "dead-end pocket, open only behind (scooter rows)"),
    "pillar": World(_box(-0.4, 3.0, 0.4, 3.8), (0, 0, 0), (0, 9), 500,
                    "pillar on the goal line"),
    "grass": World([((-0.4, 4.0), (0.4, 4.0))], (0, 0, 40), (0, 12), 500,
                   "open grass field (no paving), goal 40 deg left, small obstacle", grass=True),
    "grass_behind": World([], (0, 0, 180), (0, 8), 500, "grass field, goal behind", grass=True),
}


def main(argv: list[str]) -> int:
    verbose = "-v" in argv
    names = [a for a in argv if not a.startswith("-")] or list(SCENARIOS)
    bad = 0
    print(f"{'scenario':12s} {'ok':3s} {'ticks':>5s} {'minGoal':>7s} {'path':>6s} "
          f"{'esc':>4s} {'hits':>4s}  description")
    for name in names:
        w = SCENARIOS[name]
        r = run(w, verbose=verbose)
        ok = r.reached and r.collisions == 0
        bad += not ok
        print(f"{name:12s} {'yes' if ok else 'NO':3s} {r.ticks:5d} {r.min_goal_m:7.2f} "
              f"{r.path_m:6.1f} {r.escapes:4d} {r.collisions:4d}  {w.name}"
              f"{'  [boxed exit]' if r.boxed_exit else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
