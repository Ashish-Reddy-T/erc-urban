"""
Short live motion probe: measures what the controller assumes.

    python -m erc_indoor.probe               # forward 1.5 s, spin L 1 s, spin R 1 s

Prints command->wheel lag, speed at V_NOM, spin rate at SPIN_ANG (gyro), and
the visual bearing change across the left spin (sign check: a LEFT spin must
give a POSITIVE yaw_err between the after- and before-frames).
Needs ~1 m of clear floor ahead. Halts on exit, including Ctrl-C.
"""
from __future__ import annotations

import os
import time

import numpy as np

from .bearing import bearing, features
from .navigator import SPIN_ANG, V_NOM
from .odometry import Odometer
from .run import CommandStreamer, SDK_URL, wait_for_telemetry
from .sdk_client import RoverClient


def _watch(odo: Odometer, seconds: float, log: list) -> None:
    t_end = time.time() + seconds
    while time.time() < t_end:
        st = odo.state()
        log.append((time.time(), st.s, st.yaw, st.v, st.wz, st.age_s))
        time.sleep(0.05)


def main() -> None:
    client = RoverClient(SDK_URL, timeout=10.0)
    streamer = CommandStreamer(client, send=True)
    odo = Odometer(client)
    odo.start()
    try:
        wait_for_telemetry(client, odo)
        streamer.start()
        log: list = []
        _watch(odo, 1.0, log)
        st0 = odo.state()
        print(f"idle: odometry age {st0.age_s:.2f} s, s={st0.s:.3f}, yaw={st0.yaw:.1f}")

        # forward
        t_cmd = time.time()
        streamer.set(V_NOM, 0.0, 1.5)
        _watch(odo, 4.0, log)
        arr = np.array(log)
        moving = arr[(arr[:, 0] > t_cmd) & (np.abs(arr[:, 3]) > 0.02)]
        st1 = odo.state()
        lag = (moving[0, 0] - t_cmd) if len(moving) else float("nan")
        vmax = np.abs(arr[arr[:, 0] > t_cmd, 3]).max() if len(arr) else float("nan")
        print(f"forward {V_NOM} x1.5 s: moved {st1.s - st0.s:.2f} m, peak v {vmax:.2f} m/s, "
              f"wheel data first moved {lag:.2f} s after command, odometry age {st1.age_s:.2f} s")

        # left spin with bearing check
        f_before = features(client.get_front())
        y0 = odo.state().yaw
        streamer.set(0.0, SPIN_ANG, 1.0)
        _watch(odo, 3.0, log)
        y1 = odo.state().yaw
        f_after = features(client.get_front())
        b = bearing(f_after, f_before)
        print(f"spin LEFT {SPIN_ANG} x1.0 s: gyro {y1 - y0:+.1f} deg; visual {b.yaw_err_deg:+.1f} deg "
              f"(inliers {b.n_inliers}, conf {b.confidence:.2f})")

        # right spin back
        y0 = odo.state().yaw
        streamer.set(0.0, -SPIN_ANG, 1.0)
        _watch(odo, 3.0, log)
        y1 = odo.state().yaw
        print(f"spin RIGHT {SPIN_ANG} x1.0 s: gyro {y1 - y0:+.1f} deg")
    finally:
        streamer.stop()
        for _ in range(3):
            streamer.halt()
            time.sleep(0.1)
        odo.stop()
        print("halted")


if __name__ == "__main__":
    main()
