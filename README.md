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

- **One SDK per KEY at a time — not per bot.** Everyone shares the same
  `SDK_API_TOKEN`, and FrodoBots gives one messaging (RTM) user id per key. So two
  people running the SDK at the same time kick each other out (`REMOTE_LOGIN`),
  **even when they drive different rovers**. Our SDK logs back in after 2 s, so two
  of them fight forever: both sessions drop every ~6 s, and commands and telemetry
  are lost. On 2026-09-29 an SDK on another machine (rover `frodobot_30d2g7`) did
  this to the indoor lap, which moved 2.8 m in 60 s. Before you start one, check
  that nobody else on the team is running an SDK.
  The viewer page `http://localhost:8000/` is safe: it uses a separate spectator id.
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

### SDK sanity check (run before every drive)

```bash
curl -s localhost:8000/data | python -c "import json,sys; d=json.load(sys.stdin); print({k: d.get(k) for k in ('battery','rtm_state','rtm_kicks','rtm_conflict','rtm_foreign_bots','rtm_age_s')})"
curl -s -o /dev/null -w "front %{http_code}\n" localhost:8000/v2/front
```

| Field | Healthy | If not |
|---|---|---|
| `rtm_state` | `CONNECTED` | `ABORTED` + `rtm_reason=REMOTE_LOGIN` means another client took our login (see above) |
| `rtm_kicks` | stays the same over ~10 s | it keeps rising: someone else is running an SDK with this key. Stop theirs. Restarting ours does not help |
| `rtm_conflict` | `false` | `true` = kicked 2+ times in 60 s; the SDK log prints `ANOTHER CLIENT IS LOGGED IN WITH THIS SDK KEY` |
| `rtm_foreign_bots` | `[]` | a rover id here = someone recently used this key for that rover |
| `rtm_age_s` | < 2 | telemetry is stale; if not kicked, the SDK rebuilds its browser by itself after 6 s |
| `/v2/front` | `200` | `404` for the first ~10–40 s after SDK start is normal (video not rendered yet) |

`erc_indoor.run` does this check by itself. It watches the login for 10 s before
driving and exits with `LOGIN CONFLICT` if it gets kicked. Mid-run, every kick
prints an `[RTM] ... kicked` line.

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
| `SEND_CONTROL` | `1` | `0` = dry run: perceive, localize and log, but never move |
| `LAMP` | `0` | `1` = headlight on (dim corridors) |
| `START_S` | `0` | start mid-route, metres along the map (e.g. `34` = after the piano turn) |
| `STOP_AT_S` | `0` (full lap) | stop after this much progress, for staged tests |
| `USE_CONES` / `USE_FREESPACE` / `USE_TWIN` | `1` | toggle cone pause / floor-boundary avoidance / night views |
| `SAVE_FRAMES` | `2` | save every Nth frame; `0` = off (a full lap is ~50–120 MB with frames) |
| `OVERLAY_PORT` | `8090` | live annotated view at `http://<this machine>:8090/stream.mjpg` |
| `MAX_RUN_S` | `1500` | hard stop after this many seconds |
| `ALLOW_RTM_CONFLICT` | `0` | `1` = drive even though the login check saw kicks (it won't go well) |
| `LINK_STALE_S` / `ODOM_STALE_S` / `SDK_TIMEOUT_S` | `4` / `2.5` / `3` | link watchdog: hold still when frames/odometry stop updating. Leave at defaults |

Logs go to `runs/indoor/run_<stamp>/`. `python -m erc_indoor.overlay runs/indoor/run_<stamp>`
renders one to video. Other tools: `erc_indoor.probe` (1 m live motion/sign check),
`erc_indoor.record` (keyboard teach-lap recorder), `erc_indoor.route_map` and
`erc_indoor.twin` (rebuild the map; they need the teach-lap recordings, which are
not in git — ask Ashish).

### Finals: the runner, per slot

The API "driver" agent was removed after it flipped the Argentina rover
(2026-09-28: an auto-calibrated `V_NOM` of 0.45 plus model turns beside a
parked rover). It is kept in `archive/driver_agent/` and is not used. Finals run
the plain runner below, with the tested speed `V_NOM=0.28`.

Each slot, from the repo root in `rover_env`:
1. Set `BOT_SLUG` and `MISSION_SLUG` in `earth-rovers-sdk/.env`.
2. Kill the previous slot's SDK. It reads `.env` only at startup, so a leftover
   SDK would drive the previous site's bot:
   ```bash
   lsof -ti tcp:8000 | xargs kill -9; pkill -9 -f "pyppeteer/.dev_profile"
   ```
3. Start the SDK (terminal 1):
   ```bash
   cd earth-rovers-sdk && hypercorn main:app --bind 127.0.0.1:8000
   ```
4. Run the mission (terminal 2):
   ```bash
   # Panama (grass, leaves, trees; broken compass)
   FAN_TERRAIN=offroad HEADING_K_COMPASS=0 GYRO_AXIS=z STAGE=1 NAV=fan USE_MISSION=1 V_NOM=0.28 \
     python -m erc_urban.run 2>&1 | tee runs/panama_m3.log
   # Costa Rica (hillside garden)
   FAN_TERRAIN=offroad HEADING_K_COMPASS=0.04 GYRO_AXIS=z STAGE=1 NAV=fan USE_MISSION=1 V_NOM=0.28 \
     python -m erc_urban.run 2>&1 | tee runs/costarica_m1.log
   ```
   Paved sites: drop `FAN_TERRAIN=offroad`.
   ```bash
   # Selebi-Phikwe (town dirt roads, a loop; people on the road). Compass off:
   # other teams' headings went noisy from CP4 on. People never range-relieved.
   # CP1 and the last CP claimed within 15 m, the middle ones within 3 m.
   FAN_TERRAIN=offroad HEADING_K_COMPASS=0 FAN_NO_RELIEF_EXTRA=person STAGE=1 NAV=fan \
     USE_MISSION=1 V_NOM=0.28 CLAIM_NEAR_M=3 python -m erc_urban.run 2>&1 | tee runs/selebi_m.log
   ```
5. Live view: `http://localhost:8090/stream.mjpg`. Frames and overlays go to
   `runs/urban/run_<time>/`. Ctrl-C stops the rover.

Every checkpoint is claimed as soon as it is within 15 m. If the server rejects
a claim at the edge (its GPS reads farther), the rover keeps driving in and
retries 2 m closer or after 5 s.

#### Mexico (Tepetlaoxtoc): `erc_urban.run_mexico`

One straight dirt lane between stacks of dry grass: CP1 at the start, CP2-CP4
every 45-56 m up the lane (bearing 35-40 deg), ~130 m of driving. A lean lane
follower, separate from `erc_urban.run` (whose aim sources fought on this lane
in the trial: 16 left/right flips, 5 m in 6 min):

```bash
python -m erc_urban.run_mexico 2>&1 | tee runs/mexico_m.log
```

- Faces the lane once (committed turn; it starts facing away), then steers at
  the centre of the longest rover-wide gap in the lane mask
  (`erc_urban/modules/lane.py`), nudged toward the checkpoint line.
- Lane = earth / dirt track / water (wet) / mountain (dark wet dirt); the grass
  stacks (plant / tree / grass) are walls. Segmentation only, no depth; frames
  fetched in the background: ~5 Hz.
- Blind frames (lens drops) coast on the last command; blocked = stop and turn
  toward more lane; a short reverse only if the wheels are truly stuck.
- Claims at 15 m. Heading = compass (matched the gyro within 3 deg on the
  trial bot; its accelerometer is unusable, so no tilt checks).
- Saves `runs/mexico/run_<time>/` (frames, overlays, ticks.jsonl); live view :8090.
- Closed-loop check: `python -m erc_urban.sim_lane` (11 lane scenarios).

### Urban / outdoor missions (`erc_urban`)

This one runner does every outdoor track (Wuhan, Panama, Mexico, …). It drives
the SDK mission's checkpoints in order and claims each one. Obstacles come from
SegFormer ground segmentation (`NAV=fan`, the default); LogoNav is loaded but
does not steer.

#### Competition-day checklist (≈3 min per bot, in this order)

1. **`.env`**: set `BOT_SLUG` and `MISSION_SLUG` in `earth-rovers-sdk/.env`
   (mission slugs vary: `mission-1`, `leg-1`, …), then restart the SDK. The SDK
   only reads `.env` at start.
2. **Only one SDK Chrome.** A Chrome left over from an earlier SDK logs into
   the same account and the two kick each other out (`rtm_state` flips to
   `ABORTED` / `REMOTE_LOGIN` every few seconds; Costa Rica, 2026-09-25):
   ```bash
   ps aux | grep "pyppeteer/.dev_profile" | grep -- "--headless" | grep -vc Helper   # must be 1
   pkill -f "pyppeteer/.dev_profile"     # if not: kill them all, restart the SDK
   ```
   This matches only the SDK's headless Chrome, never your own browser.
3. **Start the mission**:
   ```bash
   curl -X POST localhost:8000/start-mission
   ```
4. **Bot check** (turns the rover ~40° left and back, so keep it clear of things):
   ```bash
   python -m erc_urban.bot_check          # or --no-move to skip the turn
   ```
   It checks the link (one Chrome, `rtm_state` CONNECTED for 5 s), GPS,
   battery and checkpoints (distance and direction of each). The turn finds
   the gyro yaw axis and tests the compass against the gyro. It ends with a
   `Run with: GYRO_AXIS=… HEADING_K_COMPASS=…` line to copy.
   Today's compasses: Machakos was 80° off and Costa Rica under-read turns by
   20%. Expect `HEADING_K_COMPASS=0`.
5. **Pick the terrain mode**:
   ```bash
   python -m erc_urban.terrain_check
   ```
   It saves `runs/terrain_check/<time>_live.jpg`, with urban on the left and
   offroad on the right. Keep `urban` if the course is green in urban and only
   real obstacles are short. Use `FAN_TERRAIN=offroad` if urban marks dirt,
   grass paths or leaf litter red or short.
6. **Launch** with the flags from steps 4–5 (see Run below), and watch
   http://localhost:8090/stream.mjpg.
   - **Look for these startup lines:** `[osm] map loaded`, `[tilt] gravity at
     rest …`, and, with the compass off, `[heading] no fix yet: driving the
     clearest way …`. That last one is expected: the rover drives up to 8 m
     straight to get a GPS heading before it turns to the checkpoint.
   - **Stop it (Ctrl-C) if:** it hits the same spot more than about 3 times, it
     is climbing a kerb or bank, or you see sky in the view.
7. **After the last checkpoint**, run `curl -X POST localhost:8000/end-mission`
   (harmless if the ride already closed). Then change `MISSION_SLUG`, restart
   the SDK, and go back to step 2.

#### Run

```bash
# paved campus (Wuhan-style): the defaults
STAGE=1 NAV=fan USE_MISSION=1 V_NOM=0.28 python -m erc_urban.run 2>&1 | tee runs/<site>_m1.log

# park / grass / roots / dirt (Panama-style)
FAN_TERRAIN=offroad STAGE=1 NAV=fan USE_MISSION=1 V_NOM=0.28 \
  python -m erc_urban.run 2>&1 | tee runs/<site>_m1.log

# end the ride (always run it; harmless if the ride already closed)
curl -X POST localhost:8000/end-mission
```

Claiming the **last** checkpoint closes the ride on the FrodoBots side. After
that, `end-mission` answers `Failed to end mission` (the API says `404 Ride not
found`), which is fine. It matters when you stop a mission partway. To run the
next mission, change `MISSION_SLUG` in `.env`, restart the SDK, and start the
runner (it POSTs start-mission itself).

Ctrl-C stops the rover at once. If the SDK dies, the runner restarts it and
re-sends start-mission (progress is kept). It gives up after `SDK_MAX_RESTARTS`.
Logs, overlays and raw frames go to `runs/urban/run_<stamp>/`.

#### The settings that matter

| Env var | Default | What it does / when to change it |
|---|---|---|
| `FAN_TERRAIN` | `urban` | `urban`: only paved classes (floor, road, sidewalk, …) are ground, and grass is avoided if paving exists. `offroad`: paved, earth, sand and grass classes are all ground (no grass penalty), and anything else must be ≥ 12 cm tall and not flat by depth to stop the rover, so leaf litter passes but bushes, toys and kerbs stop it. Use it for parks, grass, dirt and leaf litter. |
| `HEADING_K_COMPASS` | `auto` | How much the compass corrects heading each tick. `auto`: 0.04, or 0 if the startup wiggle shows the compass not following the gyro by more than `COMPASS_BAD_DEG`=20° (log line `[compass] off by N deg … IGNORED`). Panama `hinge-moto-nova` swung 60° at rest. With 0, heading comes from the gyro plus GPS-vs-wheel track alignment every ~4 m of driving, so the first few metres may head the wrong way until the first `[heading] GPS track …` line. Set a number to force it. |
| `GYRO_AXIS` | `auto` | Which `/data` gyro axis is yaw. `auto` does a ±20° wiggle at startup and picks it. The log says `yaw = gyro x [detected by wiggle …]`. Mini+ units use `z`. Wuhan `inert-eel-code` uses `x`. Set `x`/`y`/`z` (or `-z` to flip) to skip the wiggle. |
| `V_NOM` | `0.34` | Forward command. We use 0.24–0.28. Speed per unit differs by bot (0.45 → 0.37 m/s on `inert-eel-code`). |
| `CLAIM_NEAR_M` | `15.0` | Claim a checkpoint as soon as it is within this distance (the SDK accepts 15 m). |
| `CLAIM_MAX_S` | `90` | Middle checkpoints: claim anyway after this long inside 15 m without reaching `CLAIM_NEAR_M`, e.g. when GPS noise keeps the checkpoint out of reach. |
| `HEADING_STATE` | `runs/heading_state.json` | The saved heading, reused on a restart within 15 min if the rover hasn't moved. Use one file per bot when running two bots. |
| `USE_OSM` | `1` | Route each leg along OpenStreetMap footways/roads (the minimap's data), fetched once per mission and cached in `.osm_cache/`. Legs under `OSM_MIN_LEG_M`=40 m, or routes over `OSM_MAX_DETOUR`=2.5× the straight line, go straight. It falls back to straight lines if Overpass is unreachable. |
| `HEADING_BOOTSTRAP_M` | `8` | With the compass off, first drive straight along the clearest direction for up to this far to get a GPS heading. `0` disables it. |
| `FAN_TILT_DEG` / `FAN_PIVOT_TILT_DEG` | `25` / `15` | Back off when tilt (from the gravity measured at startup) holds above this while driving or pivoting, then block ±45° there for 3 m. This catches kerbs, banks and mounds the camera reads as flat. |
| `SEND_CONTROL` | `1` | `0` = dry run: everything runs and logs, nothing moves. |
| `SKIP_START_MISSION` | `0` | `1` = don't POST start-mission (resume an already-started mission). |
| `SDK_MANAGE` / `SDK_MAX_RESTARTS` | `1` / `6` | Auto-restart a dead SDK. `SDK_MANAGE=0` only waits. |
| `OVERLAY_PORT` | `8090` | Live view port. Change it if two runners share a machine. |
| `USE_MISSION=0 GOAL_LAT=.. GOAL_LON=..` | — | Free-drive to one GPS point, with no mission clock. Comment out `MISSION_SLUG` and restart the SDK first. |

Per-bot facts so far:
- `luke-notch-quirk` (Wuhan): good compass, gyro z, urban defaults.
- `hinge-moto-nova` (Panama): broken compass (`auto` catches it; we ran with `HEADING_K_COMPASS=0`), `FAN_TERRAIN=offroad`.
- `taper-pivot-valve` (Machakos, Mini+): the compass is 80° off, so use `HEADING_K_COMPASS=0 GYRO_AXIS=z`. The site has kerbs, raised beds and grassy banks: `offroad` plus OSM footways.
- `kilo-tech-r2d2` (Costa Rica, Mini+): the compass under-reads turns by 20% and is bent further by a steel gate at the start, so use `HEADING_K_COMPASS=0 GYRO_AXIS=z`. Paved and steep: `urban`.
- `inert-eel-code` (Wuhan) is an EarthRover **Zero**, not a competition bot. Gyro x (auto-detected), and the camera sits ~0.8 m high, so the Mini+ ground geometry does not fit it. Don't judge the navigation by it.

`python -m erc_urban.sim_fan` runs the obstacle logic closed-loop in 9 synthetic
scenes (walls, pockets, gaps, grass). Run it after touching `run.py` or `modules/`.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest            # test/
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest erc_urban_test/tests
python -m unittest discover -s test -t .                     # same tests, if rover_env has no pytest
RUN_E2E=1 python -m unittest test.test_e2e_runner -v         # the runner vs a fake SDK bot (~3 min)
```

`test/test_failure_modes.py` has one test per failure met on the practice bots
(tilt glitches, kerbs, flower beds and toys, compass and gyro-axis mixups,
flapping links, cut-off model replies, and more). `test/fake_sdk.py` is a
stand-in SDK server that can play any of those bots.

`test_live_sdk.py` needs the SDK running with a rover connected.

## Not in git

| What | Size | Where it comes from |
|---|---|---|
| Model checkpoints (`*.pth`) | ~510 MB | `setup.sh` |
| `erc_indoor/data/indoor_track_recording.h5` (day teach lap) | 509 MB | Ashish; only to rebuild the route map |
| `runs/` (run logs, night teach lap) | GBs | local output |
| `data/` (FrodoBots rides for offline eval), `repos/` (upstream paper code) | — | reference material only |
