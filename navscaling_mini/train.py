"""
Fine-tune the NavigationScaling MLP-BC head on cached Mini-4K features.

    PYTHONPATH=.:repos/NavigationScaling python -m navscaling_mini.train \
        --data data/navscaling_ft [--epochs 6] [--out data/navscaling_ft/head_best.pt]

Inputs are the npz files written by navscaling_mini.extract (frozen Theia +
compressor features, plain and mirrored, of 4 Hz front frames). Only
goal_encoder, state_encoder, prediction_head are trained, starting from
lauriasuo/EarthRovers_MLP-BC_nLoc153_nH-1. Recipe mirrors their
frodobots_dataset / ScaledMSELoss: goals re-drawn every epoch along the ride's
own future GPS path (beta(5,1) x up to 150 m), mirror augmentation negates the
goal direction and the angular target together.

Epoch 0 is the untouched pretrained head, so every number is read against it
and against the goal-pursuit baseline (w = clip(-0.6 * direction)).
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch

MODEL = "lauriasuo/EarthRovers_MLP-BC_nLoc153_nH-1"
TRAINED = ("goal_encoder", "state_encoder", "prediction_head")


def _bearing(la0, lo0, la1, lo1):
    p0, p1, dl = np.radians(la0), np.radians(la1), np.radians(lo1 - lo0)
    return np.arctan2(np.sin(dl) * np.cos(p1), np.cos(p0) * np.sin(p1) - np.sin(p0) * np.cos(p1) * np.cos(dl))


def _hav(la0, lo0, la1, lo1):
    p0, p1 = np.radians(la0), np.radians(la1)
    h = np.sin((p1 - p0) / 2) ** 2 + np.cos(p0) * np.cos(p1) * np.sin(np.radians(lo1 - lo0) / 2) ** 2
    return 2 * 6371000.0 * np.arcsin(np.minimum(1.0, np.sqrt(h)))


class Rides:
    """All samples of a set of rides, with per-ride GPS polylines for goals."""

    def __init__(self, files, rng):
        self.rng = rng
        feats, flips, chunks, meta = [], [], [], []
        self.polys = []
        for r, f in enumerate(files):
            d = np.load(f)
            feats.append(d["feat"])
            flips.append(d["feat_flip"])
            chunks.append(d["chunk"])
            n = len(d["chunk"])
            meta.append(np.stack([np.full(n, r), d["s"], d["lat"], d["lon"], d["heading"]], 1))
            self.polys.append((d["poly_s"], d["poly_lat"], d["poly_lon"]))
        self.feat = np.concatenate(feats)
        self.flip = np.concatenate(flips)
        self.chunk = np.concatenate(chunks).astype(np.float32)
        self.meta = np.concatenate(meta)
        self.city = [os.path.basename(os.path.dirname(f)) for f in files]
        self.ride_city = np.array([self.city[int(r)] for r in self.meta[:, 0]])

    def __len__(self):
        return len(self.chunk)

    def goals(self, fixed_m: float | None = None) -> np.ndarray:
        """(N, 2) = [dist_km, direction/pi]; + = goal to the right."""
        out = np.zeros((len(self), 2), np.float32)
        for r, (ps, pla, plo) in enumerate(self.polys):
            idx = np.nonzero(self.meta[:, 0] == r)[0]
            s = self.meta[idx, 1]
            remaining = np.maximum(ps[-1] - s, 0.0)
            if fixed_m is None:
                ahead = self.rng.beta(5, 1, size=len(idx)) * np.minimum(150.0, remaining)
            else:
                ahead = np.minimum(fixed_m, remaining)
            ahead = np.maximum(ahead, np.minimum(2.0, remaining))
            gla, glo = np.interp(s + ahead, ps, pla), np.interp(s + ahead, ps, plo)
            la, lo, hd = self.meta[idx, 2], self.meta[idx, 3], self.meta[idx, 4]
            d = _bearing(la, lo, gla, glo) - hd
            out[idx, 0] = np.clip(_hav(la, lo, gla, glo) / 1000.0, 0.0, 1.0)
            out[idx, 1] = ((d + np.pi) % (2 * np.pi) - np.pi) / np.pi
        return out


def head_forward(m, feat, goal):
    x = m.state_encoder(feat, m.goal_encoder(goal))
    return m.output_shaper(m.prediction_head(x))            # (B, 10, 2)


def scaled_mse(pred, target):
    w = 1.0 + 9.0 * target[..., 1].abs().amax(dim=1, keepdim=True)   # their ScaledMSELoss, 1..10
    return ((pred - target) ** 2 * w.unsqueeze(-1)).mean()


@torch.inference_mode()
def evaluate(m, ds, device, goal_m=15.0):
    g = torch.from_numpy(ds.goals(fixed_m=goal_m)).to(device)
    preds = []
    for k in range(0, len(ds), 2048):
        f = torch.from_numpy(ds.feat[k:k + 2048].astype(np.float32)).to(device)
        preds.append(head_forward(m, f, g[k:k + 2048]).cpu().numpy())
    p = np.concatenate(preds)
    hw = ds.chunk[:, :, 1].mean(1)                          # human angular, next second
    pw = p[:, 0, 1]                                          # policy's first command
    bw = np.clip(-0.6 * g[:, 1].cpu().numpy(), -1, 1)
    out = {}
    for name, sel in [("all", np.ones(len(ds), bool))] + [(c, ds.ride_city == c) for c in sorted(set(ds.ride_city))]:
        if sel.sum() < 20:
            continue
        turn = sel & (np.abs(hw) > 0.15)
        row = {}
        for tag, w in (("policy", pw), ("goal", bw)):
            r = np.corrcoef(w[sel], hw[sel])[0, 1]
            sgn = (np.sign(w[turn]) == np.sign(hw[turn])).mean() if turn.any() else float("nan")
            row[tag] = (sgn, r, np.abs(w[sel] - hw[sel]).mean())
        out[name] = (int(sel.sum()), int(turn.sum()), row)
    return out


def show(tag, res):
    for name, (n, nt, row) in res.items():
        (ps, pr, pm), (gs, gr, gm) = row["policy"], row["goal"]
        print(f"  {tag:9s} {name:14s} n={n:6d} turns={nt:5d} | policy sign {ps:4.0%} corr {pr:+.2f} "
              f"mae {pm:.2f} | goal-baseline sign {gs:4.0%} corr {gr:+.2f}", flush=True)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/navscaling_ft")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", default="data/navscaling_ft/head_best.pt")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    train_files = sorted(f for f in glob.glob(f"{a.data}/*/*.npz") if "/test/" not in f)
    test_files = sorted(glob.glob(f"{a.data}/test/*.npz"))
    tr, te = Rides(train_files, rng), Rides(test_files, rng)
    # Label test rides by where they came from (Selebi's 4 are known; others by GPS).
    te.ride_city = np.array(["selebi_phikwe" if -23 < la < -21 else "costa_rica" if 8 < la < 12
                             else "tepetlaoxtoc" if 19 < la < 20.5 else "other"
                             for la in te.meta[:, 2]])
    print(f"train {len(tr)} samples / {len(train_files)} rides ({dict(zip(*np.unique([os.path.basename(os.path.dirname(f)) for f in train_files], return_counts=True)))}); "
          f"test {len(te)} samples / {len(test_files)} rides", flush=True)

    from earthrovers.common.models.mlp_model import MLPModel
    m = MLPModel.from_pretrained(MODEL).to(device)
    assert not m.goal_embed and m.state_fusion_type == "mlp" and m.action_decoder_type == "mlp"
    for n, p in m.named_parameters():
        p.requires_grad = n.startswith(TRAINED)
    params = [p for p in m.parameters() if p.requires_grad]
    print("trainable %.2fM" % (sum(p.numel() for p in params) / 1e6), flush=True)
    m.eval()
    show("epoch 0", evaluate(m, te, device))

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best = float("inf")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        g = tr.goals()
        order = rng.permutation(len(tr))
        m.train()
        tot = 0.0
        for k in range(0, len(order), a.batch):
            idx = order[k:k + a.batch]
            flip = rng.random(len(idx)) < 0.5
            f = np.where(flip[:, None], tr.flip[idx], tr.feat[idx]).astype(np.float32)
            goal = g[idx].copy()
            goal[flip, 1] *= -1
            tgt = tr.chunk[idx].copy()
            tgt[flip, :, 1] *= -1
            pred = head_forward(m, torch.from_numpy(f).to(device), torch.from_numpy(goal).to(device))
            loss = scaled_mse(pred, torch.from_numpy(tgt).to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        m.eval()
        with torch.inference_mode():
            gt = torch.from_numpy(te.goals(fixed_m=15.0)).to(device)
            vl = float(scaled_mse(head_forward(m, torch.from_numpy(te.feat.astype(np.float32)).to(device), gt),
                                  torch.from_numpy(te.chunk).to(device)))
        print(f"epoch {ep}: train loss {tot / len(tr):.4f}  test loss {vl:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        show(f"epoch {ep}", evaluate(m, te, device))
        if vl < best:
            best = vl
            torch.save({n: p.detach().cpu() for n, p in m.state_dict().items() if n.startswith(TRAINED)}, a.out)
            print(f"  saved {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
