# ERC Urban Track — modular navigation stack (original design notes)

A staged, runnable scaffold. **Stage 0 runs today** against the FrodoBots
SDK; each later stage swaps ONE module (a paper-derived component) into a
fixed interface. Build order = the order these were validated to matter.

## What is real vs. what you must wire
- **Real & runnable now:** the SDK client (verified against
  `frodobots-org/earth-rovers-sdk`: `POST /control {"command":{"linear","angular"}}`,
  `/data`, `/v2/screenshot`), the GPS goal geometry, the geometric baseline
  planner, the pure-pursuit controller, and the staged main loop.
- **Marked `# >>> WIRE:` (you edit after cloning the repo):** the learned
  planner checkpoint, the VLM API call, the SAM2 traversability projection,
  and the mission-checkpoint fetch. I left these as honest seams rather than
  hallucinated signatures — a wrong guess there wastes more time than it saves.

## The staged path

| Stage | Adds | Module | From (paper / repo) | Interface |
|------|------|--------|---------------------|-----------|
| 0 | Minimum safe loop | `BaselinePlanner`, `PurePursuitController` | geometric baseline | Planner, Controller |
| 1 | GPS→robot goal prior | `geo.geodesic_bearing_range` | route conditioning (UrbanVLA idea) | — |
| 2 | Survive 4G + stay on sidewalk | `PoseBuffer` (AsyncShield SE(2)), `TraversabilityGate` (GeNIE/SAM2) | AsyncShield; GeNIE | SafetyGate |
| 3 | Learned planner + semantic selection | `LearnedPlanner` (MBRA/LogoNav or NoMaD), `VLMSelectorFusion` (Slow-Brain, Fast-Planner) | MBRA; NoMaD/ViNT; arXiv 2606.20458 | Planner, Selector |
| 4 | Reactivity (offline) | RL residual post-train the planner | S2E (arXiv 2507.22028) | — (training) |

## Per-module provenance & why it beats alternatives
- **M1 planner — MBRA/LogoNav base, NoMaD-style K-sampling.** Native to
  FrodoBots-2K/7K (MBRA-labeled) and emits the scored candidate set the whole
  stack consumes. **Keystone risk:** validate calibrated multi-hypothesis
  output first; without it, M4/M5 have nothing to select over.
- **M4/M5(slow) — Slow-Brain, Fast-Planner.** Training-free VLM candidate
  selector + Probability Fusion with exponential staleness decay; closes the
  junction/pedestrian scoring gap and survives 1–3 s cellular latency.
  Defaults follow their ablations (hide goal+scores, K≈18, Prob Fusion).
- **M5(transport) — AsyncShield SE(2) realignment.** De-stales the chosen
  waypoint over the 4G round-trip using odometry; server-side only (no
  onboard adapter possible on a thin-client rover).
- **M6 — GeNIE/SAM2 traversability veto.** The ERC winner's core; vetoes
  candidates onto road/grass. Sorokin's sidewalk study shows curb-cuts and
  grass-vs-pavement are the dominant sidewalk-violation failures.
- **M3 — S2E RL residual post-training.** Adds pedestrian/obstacle reactivity
  that passive FrodoBots imitation can't teach. Highest effort → last.
- **M7 — pure pursuit.** What S2E / UrbanVLA / the Slow-Brain deployment use.

## Data / training
Use HuggingFace **LeRobot's `earthrover_mini_plus`** integration for
collection (`lerobot-record`) and fine-tuning. Pipeline: MBRA-relabel
FrodoBots-2K/7K → SFT pose policy → (Stage 4) S2E-style RL residual in
MetaUrban/URBAN-SIM. Eval in NavBench-GS / MetaUrban-12K / SidewalkBench
before spending real Mini+ test hours.
