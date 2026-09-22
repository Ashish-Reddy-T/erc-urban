"""
Convert a FrodoBots ride into the episode format `eval.record` produces, so the
benchmarks run on it unchanged.

    python -m erc_urban.eval.frodobots --ride <dir> --out runs/vn1 --hz 5

Our own recordings are confounded as reference data: an episode driven by a
FIXED command is an arc the rover drove because it was told to, so scoring a
perception-driven planner against it punishes the planner for correctly going
straight. On such an episode a uniformly RANDOM candidate beat the planner by
35%, which says nothing about the planner.

These rides are HUMAN TELEOPERATION, so the path reflects a person reacting to
the scene — the reference an imitation-style metric needs. The front camera is
1024x576, identical to ours, so the measured intrinsics transfer directly, and
rides span many countries.

The reference trajectory is VO, not the rides' 1 Hz GPS: consumer GPS error is
metres while the candidates we score span ~2.5 m. The rides do carry real wheel
RPMs, which our rover reports as a constant [0,0,0,0]; `--check-rpm` uses them
as an independent sanity check on VO.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

from ..calib import CameraCalibration
from ..modules.odometry import VisualOdometry


def decode(ride_dir: str, out_dir: str, hz: float) -> list[str]:
    """HLS .ts segments -> jpg frames at `hz`."""
    # Rides carry BOTH audio and video HLS streams, named "..._audio_*.ts" and
    # "..._video_*.ts". Globbing all .ts feeds ffmpeg an audio-only concat and
    # it exits with "Output file does not contain any stream" -- which reads
    # like a corrupt ride rather than the wrong file selection it is. Some
    # rides ship audio only; those are skipped explicitly.
    segs = sorted(glob.glob(os.path.join(ride_dir, "recordings", "*_video_*.ts")))
    if not segs:
        n_audio = len(glob.glob(os.path.join(ride_dir, "recordings", "*_audio_*.ts")))
        raise SystemExit(
            f"no VIDEO segments under {ride_dir}/recordings "
            f"({n_audio} audio segments present) — this ride is audio-only")
    os.makedirs(out_dir, exist_ok=True)
    listing = os.path.join(out_dir, "_segs.txt")
    with open(listing, "w") as fh:
        for s in segs:
            fh.write(f"file '{os.path.abspath(s)}'\n")
    # Real teleop recordings have dropped/corrupt segments -- these are 4G
    # uploads from a moving robot, not clean files. Discard bad packets and
    # keep going rather than abandoning the ride: a strict decode failed
    # outright on one ride while another decoded fine with NAL warnings.
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-err_detect", "ignore_err", "-fflags", "+discardcorrupt",
           "-f", "concat", "-safe", "0",
           "-i", listing, "-vf", f"fps={hz}", "-q:v", "3",
           os.path.join(out_dir, "f%04d.jpg")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(listing)
    frames = sorted(glob.glob(os.path.join(out_dir, "f*.jpg")))
    if not frames:
        raise SystemExit(
            f"ffmpeg produced no frames for {ride_dir}\n{proc.stderr[-500:]}")
    if proc.returncode != 0:
        print(f"  (ffmpeg reported errors but decoded {len(frames)} frames; "
              "corrupt segments skipped)")
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ride", required=True, help="extracted ride directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--check-rpm", action="store_true")
    a = ap.parse_args(argv)

    frames = decode(a.ride, a.out, a.hz)
    if a.max_frames and len(frames) > a.max_frames:
        for f in frames[a.max_frames:]:
            os.remove(f)
        frames = frames[:a.max_frames]
    print(f"decoded {len(frames)} frames at {a.hz:g} Hz")

    probe = np.asarray(Image.open(frames[0]).convert("RGB"))
    cal = CameraCalibration.load_or_default(
        "erc_urban/camera_front.json", probe.shape[1], probe.shape[0])
    print(f"frame {probe.shape[1]}x{probe.shape[0]}  "
          f"calib {cal.source} fx={cal.fx:.0f} fov={cal.fov_horizontal_deg:.0f}deg")

    ctl = glob.glob(os.path.join(a.ride, "control_data_*.csv"))
    ctl_df = pd.read_csv(ctl[0]) if ctl else None

    vo = VisualOdometry(cal)
    recs = []
    for i, p in enumerate(frames):
        f = np.asarray(Image.open(p).convert("RGB"))
        d = vo.update(f)
        t = i / a.hz
        lin = ang = 0.0
        if ctl_df is not None and len(ctl_df):
            t0 = float(ctl_df["timestamp"].iloc[0])
            j = int(np.searchsorted(ctl_df["timestamp"].values, t0 + t))
            j = min(max(j, 0), len(ctl_df) - 1)
            lin = float(ctl_df["linear"].iloc[j])
            ang = float(ctl_df["angular"].iloc[j])
        recs.append({"i": i, "t": t, "dx": d.dx, "dy": d.dy, "dyaw": d.dyaw,
                     "conf": d.confidence, "reason": d.reason,
                     "lin": lin, "ang": ang, "lat": None, "lon": None,
                     "yaw": None})

    with open(os.path.join(a.out, "episode.json"), "w") as fh:
        json.dump({"calib": "erc_urban/camera_front.json", "hz": a.hz,
                   "source": os.path.basename(a.ride.rstrip("/")),
                   "ticks": recs}, fh, indent=1)

    ok = [r for r in recs if r["conf"] > 0]
    trav = sum(np.hypot(r["dx"], r["dy"]) for r in ok)
    yaws = np.array([r["dyaw"] for r in ok])
    print(f"VO: {len(ok)}/{len(recs)} usable, travel {trav:.1f} m, "
          f"net yaw {np.degrees(yaws.sum()):+.0f} deg")
    if len(yaws) > 5:
        same = float(np.mean(np.sign(yaws) == np.sign(np.median(yaws))))
        print(f"turn consistency {100*same:.0f}%  "
              f"({'VARIED - good reference' if same < 0.8 else 'near-constant arc - poor reference'})")

    if a.check_rpm and ctl_df is not None and "rpm_1" in ctl_df:
        rpms = ctl_df[["rpm_1", "rpm_2", "rpm_3", "rpm_4"]].abs().mean(axis=1)
        moving = float((rpms > 1).mean())
        print(f"wheel RPM present: {moving*100:.0f}% of control samples show motion "
              f"(our own rover reports a constant [0,0,0,0])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
