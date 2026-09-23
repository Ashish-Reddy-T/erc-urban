"""
Offline check: localize a recorded lap against the route map.

    python -m erc_indoor.replay runs/indoor/lap_20260922_204027

Feeds the lap's frames and wheel/gyro odometry through exactly the localizer
the live loop uses, then plots map node vs time next to raw nearest-neighbour
retrieval, and prints where each cone mark landed on the map.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .lap import load_lap
from .localizer import RouteLocalizer
from .place import PlaceEncoder
from .route_map import DEFAULT_OUT, RouteMap, TURN_M_PER_DEG, TURN_RATE_DPS


def odom_progress(lap, t0: float, t1: float) -> float:
    ds = float(lap.dist_at(t1) - lap.dist_at(t0))
    dyaw = float(lap.yaw_at(t1) - lap.yaw_at(t0))
    turning = abs(dyaw) / max(t1 - t0, 1e-3) > TURN_RATE_DPS
    return ds + (TURN_M_PER_DEG * abs(dyaw) if turning else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lap")
    ap.add_argument("--map", default=DEFAULT_OUT)
    ap.add_argument("--hz", type=float, default=3.0, help="simulated tick rate")
    ap.add_argument("--plot", default="runs/indoor/replay.png")
    ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--nav", action="store_true",
                    help="step the full Navigator open-loop (commands are not applied)")
    a = ap.parse_args()
    if a.nav:
        return replay_nav(a)

    route = RouteMap(a.map)
    lap = load_lap(a.lap)
    enc = PlaceEncoder()
    loc = RouteLocalizer(route)
    loc.reset(0.0, 3.0)

    # simulate the live tick: pick the latest frame every 1/hz seconds
    ticks = np.arange(lap.frame_t[0], lap.frame_t[-1], 1.0 / a.hz)
    fidx = np.clip(np.searchsorted(lap.frame_t, ticks) - 1, 0, len(lap) - 1)
    est_node, nn_node, spread, t_prev = [], [], [], ticks[0]
    for t, i in zip(ticks, fidx):
        loc.predict(odom_progress(lap, t_prev, t))
        t_prev = t
        d = enc.encode_one(lap.image(i))
        loc.update(d)
        e = loc.estimate()
        est_node.append(e.node)
        spread.append(e.spread_m)
        nn_node.append(int(loc.last_sims.argmax()))
    est_node, nn_node, spread = map(np.array, (est_node, nn_node, spread))

    print(f"{len(ticks)} ticks; median spread {np.median(spread):.2f} m; "
          f"frac confident {(spread < 2).mean():.2f}")
    for et, name in lap.events:
        k = int(np.clip(np.searchsorted(ticks, et), 0, len(ticks) - 1))
        n = est_node[k]
        print(f"  {name:13s} -> map node {n:5d}  s={route.s[n]:6.1f}  "
              f"(map {route.checkpoint_label(route.s[n])})  spread {spread[k]:.1f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    tt = ticks - ticks[0]
    ax[0].plot(tt, nn_node, ".", ms=2, alpha=0.4, label="raw nearest")
    ax[0].plot(tt, est_node, "-", lw=1.5, label="filter")
    for t in route.turns:
        ax[0].axhspan(t.start, t.end, color="orange", alpha=0.2)
    for et, name in lap.events:
        ax[0].axvline(et - ticks[0], color="k", lw=0.5)
        ax[0].text(et - ticks[0], 50, name, rotation=90, fontsize=7)
    ax[0].set_ylabel("map node"); ax[0].legend()
    ax[1].plot(tt, spread); ax[1].set_ylabel("spread m"); ax[1].set_xlabel("s")
    os.makedirs(os.path.dirname(a.plot) or ".", exist_ok=True)
    plt.savefig(a.plot, dpi=80)
    print("plot:", a.plot)


def replay_nav(a) -> None:
    """Open-loop: the lap's own motion stands in for the rover. Checks that
    every code path runs and that corners/finish trigger where the lap turned."""
    from collections import Counter
    from .cones import ConeDetector
    from .navigator import Navigator, Obs

    route = RouteMap(a.map)
    lap = load_lap(a.lap)
    loc = RouteLocalizer(route)
    loc.reset(0.0, 3.0)
    from .freespace import FreeSpaceEstimator
    nav = Navigator(route, loc, PlaceEncoder(), ConeDetector(), freespace=FreeSpaceEstimator())
    bias = lap.gyro_bias()
    ticks = np.arange(lap.frame_t[0], lap.frame_t[-1], 1.0 / a.hz)[: a.max_ticks or None]
    modes = Counter()
    for t in ticks:
        i = int(np.clip(np.searchsorted(lap.frame_t, t) - 1, 0, len(lap) - 1))
        v = float(np.interp(t, lap.wheel_t, lap.wheel_v))
        cmd = nav.step(Obs(t, lap.image(i), float(lap.dist_at(t)), float(lap.yaw_at(t, bias)), v))
        modes[cmd.mode] += 1
        if cmd.done:
            print(f"DONE at t={t - ticks[0]:.0f}s")
            break
    print("ticks per mode:", dict(modes))


if __name__ == "__main__":
    main()
