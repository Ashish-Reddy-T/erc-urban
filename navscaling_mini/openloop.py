"""
Open-loop check of the NavigationScaling MLP-BC policy on FrodoBots-Mini rides.

    PYTHONPATH=repos/NavigationScaling python -m navscaling_mini.openloop \
        data/sites/selebi_phikwe/ride_*  [--every 20] [--goal-m 15] [--save-dir DIR]

For every sampled front frame where the human was driving forward:
  * heading  = direction of the human's own GPS path over the next ~3 m
  * goal     = the point 15 m further along that path (hindsight goal, as in training)
  * policy   = first (linear, angular) of the model's 10-step chunk
  * human    = mean (linear, angular) the operator commanded over the next second
It also scores a "turn toward the goal" baseline (angular = -k * direction), so the
policy's score can be read against pure goal pursuit. Angular: + = left
(SDK convention; the model was trained on SDK commands). Direction: + = goal to
the RIGHT, as in the policy's own compute_direction().
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

MODEL = "lauriasuo/EarthRovers_MLP-BC_nLoc153_nH-1"


def _hav(a, b):
    p0, p1 = math.radians(a[0]), math.radians(b[0])
    dl, dn = math.radians(b[1] - a[1]), p1 - p0
    h = math.sin(dn / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(h)))


def _bearing(a, b):
    p0, p1 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    return math.atan2(math.sin(dl) * math.cos(p1),
                      math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl))


def load_policy(device, head=None):
    from earthrovers.common.models.mlp_model import MLPModel
    m = MLPModel.from_pretrained(MODEL).eval().to(device)
    if head:
        # Fine-tuned goal_encoder/state_encoder/prediction_head from navscaling_mini.train.
        missing, unexpected = m.load_state_dict(torch.load(head, map_location="cpu"), strict=False)
        assert not unexpected, unexpected
    import torchvision.transforms.v2 as T
    tf = T.Compose([T.ToImage(), T.Resize((224, 224)), T.ToDtype(torch.float32, scale=True),
                    T.Normalize([0.5] * 3, [0.5] * 3)])
    return m, tf


def policy_cmd(m, tf, device, rgb, dist_km, direction):
    x = tf(rgb).unsqueeze(0).unsqueeze(0).to(device)
    g = torch.tensor([[[min(max(dist_km, 0.0), 1.0), direction]]], dtype=torch.float32, device=device)
    with torch.inference_mode():
        out = m(x, g)
    out = out["out"] if isinstance(out, dict) else out
    return out.squeeze().cpu().numpy()[0]          # (linear, angular)


def ride_samples(ride, every, goal_m):
    rid = os.path.basename(ride).split("_")[1]
    ts = pd.read_csv(glob.glob(f"{ride}/front_camera_timestamps_*.csv")[0])
    gps = pd.read_csv(glob.glob(f"{ride}/gps_data_*.csv")[0])
    ctl = pd.read_csv(glob.glob(f"{ride}/control_data_*.csv")[0])
    gps["t"] = gps.timestamp / 1000.0
    gps = gps[(gps.latitude.abs() <= 90)].drop_duplicates(["latitude", "longitude"])
    if len(gps) < 5:
        return
    pts = list(zip(gps.latitude, gps.longitude))
    cum = np.concatenate([[0.0], np.cumsum([_hav(a, b) for a, b in zip(pts, pts[1:])])])
    tg = gps.t.to_numpy()
    video = glob.glob(f"{ride}/recordings/*uid_s_1000__uid_e_video.m3u8")[0]
    cap = cv2.VideoCapture(video)
    for i, t in enumerate(ts.timestamp.to_numpy()):
        ok, frame = cap.read()
        if not ok:
            break
        if i % every:
            continue
        k = int(np.searchsorted(tg, t))
        if k <= 0 or k >= len(tg):
            continue
        here_c = cum[k - 1] + (cum[k] - cum[k - 1]) * (t - tg[k - 1]) / max(tg[k] - tg[k - 1], 1e-3)
        j3 = int(np.searchsorted(cum, here_c + 3.0))
        jg = int(np.searchsorted(cum, here_c + goal_m))
        if jg >= len(pts) or j3 >= len(pts):
            continue
        here = pts[k - 1]
        c = ctl[(ctl.timestamp >= t) & (ctl.timestamp < t + 1.0)]
        if len(c) == 0:
            continue
        hv, hw = float(c.linear.mean()), float(c.angular.mean())
        if hv < 0.1:
            continue                                   # only while the human drives forward
        heading = _bearing(here, pts[j3])
        d = _bearing(here, pts[jg]) - heading
        direction = ((d + math.pi) % (2 * math.pi) - math.pi) / math.pi
        yield rid, i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), _hav(here, pts[jg]) / 1000.0, direction, hv, hw


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("rides", nargs="+")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--goal-m", type=float, default=15.0)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--head", default=None, help="fine-tuned head from navscaling_mini.train")
    a = ap.parse_args(argv)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m, tf = load_policy(device, a.head)
    rows = []
    for ride in a.rides:
        for rid, i, rgb, dist, direction, hv, hw in ride_samples(ride, a.every, a.goal_m):
            pv, pw = policy_cmd(m, tf, device, rgb, dist, direction)
            rows.append(dict(ride=rid, frame=i, dir_deg=direction * 180, human_v=hv, human_w=hw,
                             pol_v=float(pv), pol_w=float(pw), base_w=float(np.clip(-0.6 * direction, -1, 1))))
            if a.save_dir and len(rows) % 10 == 0:
                os.makedirs(a.save_dir, exist_ok=True)
                img = cv2.cvtColor(cv2.resize(rgb, (640, 360)), cv2.COLOR_RGB2BGR)
                cv2.putText(img, f"goal {direction*180:+.0f}deg  human w={hw:+.2f}  policy w={pw:+.2f} v={pv:.2f}",
                            (8, 24), 0, 0.6, (0, 255, 255), 2)
                cv2.imwrite(os.path.join(a.save_dir, f"{rid}_{i:05d}.jpg"), img)
    df = pd.DataFrame(rows)
    if df.empty:
        print("no usable samples")
        return 1
    turn = df[df.human_w.abs() > 0.15]
    def score(col):
        r = np.corrcoef(df[col], df.human_w)[0, 1]
        sign = (np.sign(turn[col]) == np.sign(turn.human_w)).mean() if len(turn) else float("nan")
        mae = (df[col] - df.human_w).abs().mean()
        return r, sign, mae
    print(f"{len(df)} samples from {df.ride.nunique()} rides; {len(turn)} with a clear human turn")
    for name, col in (("policy", "pol_w"), ("goal-pursuit baseline", "base_w")):
        r, sign, mae = score(col)
        print(f"  {name:22s} angular corr {r:+.2f}   turn-sign agreement {sign:.0%}   MAE {mae:.2f}")
    print(f"  linear: human mean {df.human_v.mean():.2f}, policy mean {df.pol_v.mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
