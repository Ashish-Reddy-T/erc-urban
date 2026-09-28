# EarthRover Challenge — navigation stacks

Autonomy code for the FrodoBots EarthRover Mini+ in the ERC 2026 tracks.
Every stack talks to the rover through the local FrodoBots SDK server
(`earth-rovers-sdk/`, HTTP on `localhost:8000`).

| Folder | What it is | Entry point |
|---|---|---|
| `erc_indoor/` | NYU indoor track: visual teach-and-repeat against a recorded lap (DINOv2 place matching + SIFT bearing + wheel/gyro odometry, SegFormer free space, cone pause) | `python -m erc_indoor.run` |
| `erc_urban/` | Urban track: LogoNav planner + Depth Anything obstacles + SegFormer semantics, drives SDK mission checkpoints | `python -m erc_urban.run` |
| `erc_urban_test/` | Sandbox copy of `erc_urban` for local free-drive testing + its unit tests | `erc_urban_test/local_run.sh` |
| `erc_offroad/` | Off-road trial tools (teleop, climb/creep tests); see `erc_offroad/FINDINGS.md` | `python -m erc_offroad.<tool>` |
| `earth-rovers-sdk/` | FrodoBots SDK, with local fixes (RTM re-login, telemetry filtered to the commanded bot) | `hypercorn main:app` |
| `ERC-3-earthrover-challenge/` | Vendored snapshot (commit `45a37cf`) of [vivekmattam02/ERC-3-earthrover-challenge](https://github.com/vivekmattam02/ERC-3-earthrover-challenge). `erc_urban` imports its LogoNav (`mbra_repo/`) and Depth Anything (`src/`, `third_party/`) code | — |

## Setup (once)

Needs conda (or any Python 3.10 env), `git`, `curl`, and Google Chrome.

```bash
git clone git@github.com:Ashish-Reddy-T/erc-urban.git earth-rover-comp
cd earth-rover-comp
conda create -n rover_env python=3.10 -y
conda activate rover_env
./setup.sh
```

`setup.sh` is safe to re-run. It:

1. `pip install -r requirements.txt` into the active env (torch, transformers, …).
2. Creates `earth-rovers-sdk/.venv` with the SDK's own pins (it needs numpy 1.26;
   the stacks use numpy 2, so the two don't share an env).
3. Downloads the checkpoints git can't hold (~510 MB, from HuggingFace, sha256-checked):
   - `logonav.pth` → `ERC-3-earthrover-challenge/mbra_repo/deployment/model_weights/`
   - Depth Anything V2 metric hypersim + vkitti (ViT-S) → `ERC-3-earthrover-challenge/third_party/Depth-Anything-V2/checkpoints/`
4. Pre-caches DINOv2 ViT-S/14 (`~/.cache/torch/hub`) and
   `nvidia/segformer-b2-finetuned-ade-512-512` (`~/.cache/huggingface`).
   **Run setup with internet before going to a venue**: at run time nothing
   should need to download.

Already in git, no action needed:
- `earth-rovers-sdk/.env` — the team's shared SDK key, bot and Chrome path.
- `erc_indoor/data/route_map/` — the indoor teach map (day lap + night twin views).
- Camera calibrations (`*/camera_front.json`).

## Before you drive: things that will bite you

- **One SDK per bot at a time.** Everyone shares the same `SDK_API_TOKEN`. If two
  people run the SDK for the same bot, the second login kicks the first out of RTM
  (`REMOTE_LOGIN`) and the rover freezes mid-run. Coordinate who is connected.
- **Check `BOT_SLUG`** in `earth-rovers-sdk/.env` is the rover you're standing next
  to. The committed value is `moto-stem-opal`. After changing it, restart the SDK.
- **Chrome path.** The SDK drives a headless Chrome. On Apple Silicon it must be
  native Chrome (pyppeteer's bundled Chromium is Intel-only):
  `CHROME_EXECUTABLE_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"`.
  On Linux, point it at `google-chrome`/`chromium`. `setup.sh` checks the path.
- **Bind the SDK to 127.0.0.1**, not `0.0.0.0`, on shared Wi-Fi.

## Run

Terminal 1, the SDK (leave it running; open http://localhost:8000 to see the video):

```bash
cd earth-rovers-sdk
.venv/bin/hypercorn main:app --bind 127.0.0.1:8000
```

Terminal 2, a stack (`conda activate rover_env` first):

### Indoor (`erc_indoor`)

How the whole stack works, start to finish: [erc_indoor/doc/HOW_IT_WORKS.md](erc_indoor/doc/HOW_IT_WORKS.md).

No SDK mission and no GPS indoors; checkpoints are judged at the cones.
Put the rover at the orange Start/Finish cone, facing down the glass atrium with
the windows on the RIGHT (how the teach lap began).

```bash
SEND_CONTROL=0 python -m erc_indoor.run   # dry run: localize + log, no motion
python -m erc_indoor.run                  # drive the lap
```

| Env var | Default | Use |
|---|---|---|
| `START_S` | `0` | start mid-route, metres along the map (e.g. `34` = after the piano turn) |
| `STOP_AT_S` | `0` (full lap) | stop after this much progress, for staged tests |
| `USE_CONES` / `USE_FREESPACE` / `USE_TWIN` | `1` | toggle cone pause / floor-boundary avoidance / night views |
| `SAVE_FRAMES` | `2` | save every Nth frame; `0` = off (a full lap is ~50–120 MB with frames) |
| `OVERLAY_PORT` | `8090` | live annotated view at `http://<this machine>:8090/stream.mjpg` |

Logs go to `runs/indoor/run_<stamp>/`. `python -m erc_indoor.overlay runs/indoor/run_<stamp>`
renders one to video. Other tools: `erc_indoor.probe` (1 m live motion/sign check),
`erc_indoor.record` (keyboard teach-lap recorder), `erc_indoor.route_map` and
`erc_indoor.twin` (rebuild the map; they need the teach-lap recordings, which are
not in git — ask Ashish).

### Urban (`erc_urban`)

```bash
python -m erc_urban.run                                   # mission mode: start mission -> drive checkpoints -> claim
USE_MISSION=0 GOAL_LAT=.. GOAL_LON=.. python -m erc_urban.run   # free-drive to one GPS goal
```

**Mission mode needs `MISSION_SLUG=...` in `earth-rovers-sdk/.env`** (from the
FrodoBots dashboard; the committed `.env` is the indoor setup and has none), then
restart the SDK. Free-drive doesn't need it. `STAGE` picks the stack
(`0` classical dirt-path + GPS creep, `1` LogoNav + full perception — default,
`2` + pose buffer, `3` + custom planner + VLM). The common knobs are in
`.env.example`; every module reads its tuning from env vars (see `erc_urban/run.py`).
`USE_VLM=1` additionally needs `pip install google-genai`, a Vertex AI
service-account JSON and `GOOGLE_CLOUD_PROJECT`.

For safe local testing without a mission clock use
`erc_urban_test/local_run.sh sign|sensors|drive <lat> <lon>`
(set `WANT_BOT` to your bot's slug).

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest            # test/
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest erc_urban_test/tests
```

`test_live_sdk.py` needs the SDK running with a rover connected.

## Not in git

| What | Size | Where it comes from |
|---|---|---|
| Model checkpoints (`*.pth`) | ~510 MB | `setup.sh` |
| `erc_indoor/data/indoor_track_recording.h5` (day teach lap) | 509 MB | Ashish; only to rebuild the route map |
| `runs/` (run logs, night teach lap) | GBs | local output |
| `data/` (FrodoBots rides for offline eval), `repos/` (upstream paper code) | — | reference material only |

## Contributing

- **Branch names**: `<type>/<short-topic>`, e.g. `fix/urban-depth-turn-sign`,
  `feat/action-chunk-latency`, `test/rover-sim`. `<type>` matches the commit
  types below; `<short-topic>` is a few hyphenated words, no issue numbers.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/),
  `type(scope): summary` — `fix(urban): ...`, `feat(indoor): ...`,
  `test(sdk): ...`, `perf(urban): ...`, `docs: ...`, `chore: ...`,
  `refactor: ...`. Scope is the stack (`urban`, `indoor`, `offroad`, `sdk`)
  when the change is stack-specific. One logical change per commit.
- **Before opening a PR**:
  - Run both test suites (see [Tests](#tests) above) and make sure nothing
    that passed before your change now fails.
  - If you changed `erc_urban`, check whether `erc_urban_test` has the same
    code path and mirror the fix there (and vice versa) — they drift apart
    otherwise.
  - Don't commit `.env`, API tokens, model checkpoints, or `runs/` output.
  - Use the PR template (`.github/pull_request_template.md`) — it has the
    same checklist as this section.
