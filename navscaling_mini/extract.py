"""
Stream FrodoBots-Mini-4K rides into cached NavigationScaling features.

    PYTHONPATH=.:repos/NavigationScaling python -m navscaling_mini.extract \
        --out data/navscaling_ft [--per-city 8]

One shard (~2 GB) at a time: download, extract the rides of the chosen cities,
decode the front camera at 4 Hz (their training rate), run the FROZEN Theia
encoder + compressor of EarthRovers_MLP-BC_nLoc153_nH-1 on the frame and on
its mirror image, save per ride a small npz, delete video and shard. Disk
never holds more than one shard plus one ride.

Per sample: features (1176, fp16, plain + flipped), the next 10 human commands
at 10 Hz (their label), GPS-path position/heading for goal sampling at train
time. Heading = direction of the rover's own GPS path over the next 3 m (the
Mini compass is raw magnetometer counts and unreliable per unit).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import shutil
import sys
import tarfile

import cv2
import numpy as np
import pandas as pd
import torch

REPO = "BitRobot/FrodoBots-Mini-4K"
MODEL = "lauriasuo/EarthRovers_MLP-BC_nLoc153_nH-1"
CITIES = {"selebi_phikwe": ("city", "Selebi Phikwe"),
          "costa_rica": ("country", "Costa Rica"),
          "tepetlaoxtoc": ("city", "Tepetlaoxtoc De Hidalgo")}
# Held out for evaluation (never extracted for training).
TEST_RIDES = {"ride_106082_dq19d2_20250306055111", "ride_115538_dq19d2_20250415134150",
              "ride_115589_dq19d2_20250416050727", "ride_115630_dq19d2_20250416105454"}


def _hav(a0, o0, a1, o1):
    p0, p1 = np.radians(a0), np.radians(a1)
    dp, dl = p1 - p0, np.radians(o1 - o0)
    h = np.sin(dp / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(dl / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.minimum(1.0, np.sqrt(h)))


def _bearing(a0, o0, a1, o1):
    p0, p1, dl = math.radians(a0), math.radians(a1), math.radians(o1 - o0)
    return math.atan2(math.sin(dl) * math.cos(p1),
                      math.cos(p0) * math.sin(p1) - math.sin(p0) * math.cos(p1) * math.cos(dl))


class Encoder:
    def __init__(self, device):
        from earthrovers.common.models.mlp_model import MLPModel
        self.m = MLPModel.from_pretrained(MODEL).eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, rgb_224: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(rgb_224).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / 0.5
        f = self.m.compressor(self.m.img_encoder(x))
        ff = self.m.compressor(self.m.img_encoder(torch.flip(x, dims=[3])))
        return f.half().cpu().numpy(), ff.half().cpu().numpy()


def process_ride(ride: str, enc: Encoder, out_path: str, every: int = 5) -> int:
    ts = pd.read_csv(glob.glob(f"{ride}/front_camera_timestamps_*.csv")[0])
    gps = pd.read_csv(glob.glob(f"{ride}/gps_data_*.csv")[0])
    ctl = pd.read_csv(glob.glob(f"{ride}/control_data_*.csv")[0]).sort_values("timestamp")
    gps["t"] = gps.timestamp / 1000.0
    gps = gps[(gps.latitude.abs() <= 90) & (gps.longitude.abs() <= 180)]
    gps = gps.drop_duplicates(["latitude", "longitude"]).sort_values("t")
    vids = glob.glob(f"{ride}/recordings/*uid_s_1000__uid_e_video.m3u8")
    if len(gps) < 5 or len(ctl) < 20 or not vids:
        return 0
    la, lo, tg = gps.latitude.to_numpy(), gps.longitude.to_numpy(), gps.t.to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(_hav(la[:-1], lo[:-1], la[1:], lo[1:]))])
    ct, cl, ca = ctl.timestamp.to_numpy(), ctl.linear.to_numpy(), ctl.angular.to_numpy()
    cap = cv2.VideoCapture(vids[0])
    keep, frames = [], []
    for i, t in enumerate(ts.timestamp.to_numpy()):
        if i % every:
            if not cap.grab():
                break
            continue
        ok, fr = cap.read()
        if not ok:
            break
        if t < tg[0] or t > tg[-1] or t < ct[0] or t + 1.0 > ct[-1]:
            continue
        tk = t + 0.1 * np.arange(10)
        chunk = np.stack([np.interp(tk, ct, cl), np.interp(tk, ct, ca)], 1)
        if np.abs(chunk).max() < 0.05:
            continue                                        # idle
        s = float(np.interp(t, tg, cum))
        if s + 3.0 > cum[-1]:
            continue
        lat, lon = float(np.interp(s, cum, la)), float(np.interp(s, cum, lo))
        hla, hlo = float(np.interp(s + 3.0, cum, la)), float(np.interp(s + 3.0, cum, lo))
        keep.append((t, s, lat, lon, _bearing(lat, lon, hla, hlo), chunk))
        frames.append(cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (224, 224), interpolation=cv2.INTER_AREA))
    cap.release()
    if len(keep) < 10:
        return 0
    feats, flips = [], []
    for k in range(0, len(frames), 64):
        f, ff = enc(np.stack(frames[k:k + 64]))
        feats.append(f)
        flips.append(ff)
    np.savez_compressed(
        out_path, feat=np.concatenate(feats), feat_flip=np.concatenate(flips),
        chunk=np.stack([k[5] for k in keep]).astype(np.float32),
        t=np.array([k[0] for k in keep]), s=np.array([k[1] for k in keep]),
        lat=np.array([k[2] for k in keep]), lon=np.array([k[3] for k in keep]),
        heading=np.array([k[4] for k in keep]),
        poly_lat=la, poly_lon=lo, poly_s=cum)
    return len(keep)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/navscaling_ft")
    ap.add_argument("--per-city", type=int, default=8, help="shards per city, most footage first")
    ap.add_argument("--cities", nargs="*", default=list(CITIES))
    a = ap.parse_args(argv)
    from huggingface_hub import hf_hub_download
    meta = pd.read_parquet(hf_hub_download(REPO, "metadata.parquet", repo_type="dataset"))
    wanted = {}                                            # shard -> {folder: out subdir}
    for city in a.cities:
        col, val = CITIES[city]
        rows = meta[(meta[col] == val) & ~meta.folder.isin(TEST_RIDES)]
        order = rows.groupby("shard").front_footage_sec.sum().sort_values(ascending=False)
        # One held-out ride per city (Selebi's four are already in test/): the
        # longest ride of the city's top shard goes to test/, never to training.
        hold = None
        if city != "selebi_phikwe" and len(order):
            top = rows[rows.shard == order.index[0]]
            hold = top.loc[top.front_footage_sec.idxmax(), "folder"]
        for shard in order.index[:a.per_city]:
            for f in rows[rows.shard == shard].folder:
                wanted.setdefault(shard, {})[f] = "test" if f == hold else city
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    enc = Encoder(device)
    tmp = os.path.join(a.out, "_tmp")
    total = 0
    for n, (shard, folders) in enumerate(sorted(wanted.items()), 1):
        todo = {f: c for f, c in folders.items()
                if not os.path.exists(os.path.join(a.out, c, f + ".npz"))}
        if not todo:
            continue
        free_gb = shutil.disk_usage(a.out).free / 1e9
        if free_gb < 4.0:
            print(f"   only {free_gb:.1f} GB free — stopping before {shard}", flush=True)
            break
        print(f"[{n}/{len(wanted)}] {shard}: {len(todo)} rides ({free_gb:.0f} GB free)", flush=True)
        try:
            path = hf_hub_download(REPO, shard, repo_type="dataset")
        except Exception as e:
            print("   download failed:", e, flush=True)
            continue
        with tarfile.open(path) as tar:
            for f, city in todo.items():
                os.makedirs(tmp, exist_ok=True)
                tar.extractall(tmp, members=[m for m in tar.getmembers()
                                             if m.name == f or m.name.startswith(f + "/")])
                os.makedirs(os.path.join(a.out, city), exist_ok=True)
                try:
                    k = process_ride(os.path.join(tmp, f), enc, os.path.join(a.out, city, f + ".npz"))
                except Exception as e:
                    k = 0
                    print("   ride failed:", f, e, flush=True)
                total += k
                print(f"   {city:13s} {f}: {k} samples (total {total})", flush=True)
                shutil.rmtree(os.path.join(tmp, f), ignore_errors=True)
        real = os.path.realpath(path)
        os.remove(real)
        if os.path.islink(path):
            os.remove(path)
    print("done:", total, "samples", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
