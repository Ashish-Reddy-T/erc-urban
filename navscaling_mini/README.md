# navscaling_mini: NavigationScaling policy on Earth Rover Mini footage

The [NavigationScaling](https://lasuomela.github.io/navigation_scaling/) MLP-BC
policy (`lauriasuo/EarthRovers_MLP-BC_nLoc153_nH-1`, code in
`repos/NavigationScaling`) was trained and deployed on Earth Rover **Zero**
bots, whose camera is about 0.8 m up. Our competition bot is the Mini+, with
its camera at 0.15 m. This folder checks it on Mini footage and fine-tunes its
head on Mini rides from our finals sites (FrodoBots-Mini-4K). Existing stacks
are untouched.

| Script | What it does |
|---|---|
| `openloop.py` | Replay Mini rides: frame + hindsight goal (15 m along the human's path) → policy (v, ω) vs human command vs a goal-pursuit baseline. `--head` loads a fine-tuned head. |
| `extract.py` | Stream Mini-4K shards (one ~2 GB at a time), cache frozen Theia+compressor features (plain + mirrored) at 4 Hz with 10×10 Hz human commands and GPS path, then delete the video. Output: `data/navscaling_ft/<city>/*.npz`. Held-out rides go to `data/navscaling_ft/test/`. |
| `train.py` | Fine-tune `goal_encoder` + `state_encoder` + `prediction_head` (4.8M params) with their recipe: beta(5,1) goals up to 150 m, mirror augmentation, ScaledMSELoss. Epoch 0 is the pretrained head, for comparison. |

Setup: `pip install git+https://github.com/bdaiinstitute/theia.git --no-deps`,
then run everything with `PYTHONPATH=.:repos/NavigationScaling`.

## Results

**Zero-shot, 2026-09-25.** 4 held-out Selebi Phikwe Mini rides, 342 samples:
the policy's turn direction matched the human's 47% of the time (chance),
angular corr +0.13. The goal-pursuit baseline scored 88% / +0.39.

**Head fine-tune, early run, 2026-09-25.** 59k Mini samples: 45 Costa Rica
rides, 8 Selebi, 4 Tepetlaoxtoc. Test set: 4 Selebi rides + 1 Costa Rica ride,
3,861 samples, goal 15 m ahead.

| Model | Turn-sign agreement | Angular corr |
|---|---|---|
| Pretrained head (epoch 0) | 55% | +0.08 |
| Fine-tuned, lr 1e-4, 6 epochs | 56–58% | +0.10 |
| Fine-tuned, lr 1e-3, 15 epochs | 47–53% | ~0 |
| Goal-pursuit baseline | 71% | +0.23 |

At lr 1e-3 the training loss falls 0.19 → 0.13 while the test metrics sit at
chance: the head memorises the training rides but does not generalise. From
frozen Theia features, the policy does not predict held-out human steering on
the Mini viewpoint. The decision gate (≥ 75% and ≥ 0.4) is far off, so it is
**not integrated**. Logs: `data/navscaling_ft/train_early*.log`.
