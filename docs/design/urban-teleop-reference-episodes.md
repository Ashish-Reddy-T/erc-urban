# Teleoperated reference episodes for erc_urban eval

## Problem

`erc_urban/eval/oracle_gap.py` and `erc_urban/eval/selector_bench.py` both
score the planner and the VLM selector against a "reference trajectory"
reconstructed from visual odometry over a recorded episode. `eval/record.py`
is the only way to produce that episode today, and it drives the rover on a
fixed linear/angular command (or a hand-typed `--pattern` cycle of fixed
segments), not on a person's reactions to the scene.

Both scoring scripts already flag why this is a problem, in their own
comments:

- `oracle_gap.py`: "That is what the rover ACTUALLY DROVE, not what an
  expert would have... this measures whether the candidate set COVERS the
  executed trajectory... NOT expert imitation quality... teleoperated
  episodes are the fix."
- `selector_bench.py`: a reference with high turn-rate consistency triggers
  a printed warning that the comparison is "indicative only" and to "use
  scene-driven or teleoperated reference data to decide."
- `modules/scoring.py` separately notes a heuristic scorer "benchmarked
  WORSE THAN RANDOM against human-teleoperated rides," which is the kind of
  finding this harness exists to catch, and can currently only be checked
  against `frodobots.py`'s bundled offline rides, not a fresh one from the
  team's own bot and route.

So the harness that is supposed to tell us whether the planner/selector
stack helps cannot currently produce its own correct ground truth on demand.
`erc_offroad/teleop.py` already solves this exact problem for the offroad
stack (WASD burst-drive with a background recorder), but it talks to the SDK
through offroad's own raw `_get`/`_post` helpers and writes a
`telemetry.jsonl` + `frames.jsonl` layout that neither `oracle_gap.py` nor
`selector_bench.py` can read; urban's episode format is `episode.json` with
one `ticks[]` entry per frame carrying VO deltas (`dx`, `dy`, `dyaw`,
`conf`), produced by `erc_urban/modules/odometry.py`.

## Goal

Give urban a way to record a real episode.json-compatible episode driven by
a human, so `oracle_gap.py` and `selector_bench.py` can be run against a
genuine reference trajectory instead of a fixed-command arc, with the same
safety posture `eval/record.py` already has (tilt / stall / traversability
abort).

## Assumptions (stated, not asked, per brainstorming-skill rules for this
task)

- Whoever runs this has hands on a keyboard next to the rover or a monitor
  with low-latency video, the same operating conditions `erc_offroad/teleop.py`
  assumes.
- "Docs only, no code" for this deliverable; the approaches below describe
  what a follow-up bounded change would implement, not a plan document.
- The near-term use is offline benchmarking (oracle_gap, selector_bench),
  not a live safety driver override for autonomous runs — that is a
  materially different, higher-stakes feature and out of scope here.

## Approach A — port the WASD recorder into `erc_urban/eval/`, on urban's own client and episode schema (recommended)

Add `erc_urban/eval/teleop_record.py`, structurally the same two-piece
design as `erc_offroad/teleop.py` (a `Streamer` thread that keeps sending
the latched command at the watchdog rate so holding a key drives
continuously, and a keyboard loop that sets bursts), but built on
`erc_urban.sdk_client.RoverClient` (already used by `eval/record.py`) instead
of offroad's raw HTTP helpers, and on `erc_urban.modules.odometry.VisualOdometry`
to compute `dx/dy/dyaw/conf` per tick the same way `eval/record.py` does.
Output goes to the same `episode.json` shape `eval/record.py` writes
(`{"calib", "hz", "ticks": [...]}`, one JPEG per tick as `f%04d.jpg`), so
`oracle_gap.py` and `selector_bench.py` need zero changes to consume it.

Reuse, don't reimplement, the three abort guards already in
`eval/record.py` (tilt from `/data` accel, VO stall while commanded to move,
traversability veto from `SemanticTraversability`) — factor them out of
`eval/record.py` into a small shared guard function both scripts call, since
duplicating safety-critical logic between two recorders is exactly the kind
of copy that drifts.

Trade-offs: most code reused from two existing, already-hardened scripts
(offroad's input handling, urban's odometry/telemetry/guards). Main cost is
the terminal-raw-mode plumbing (`termios`/`tty`/`select`) which is new to
`erc_urban/eval/` but is a known-working ~150 lines already proven in
`erc_offroad/teleop.py`. No new dependency.

## Approach B — joystick/gamepad control via `pygame` or `inputs`

Same recorder, but read a physical gamepad instead of keyboard bursts, giving
proportional (not bursty) linear/angular control, closer to what a real
teleop pilot would use and smoother reference trajectories.

Trade-offs: better control fidelity, but adds a new third-party input
dependency the rest of the repo doesn't have (setup.sh, requirements.txt
touched), needs a physical controller at the venue, and burst-vs-proportional
control is not the thing oracle_gap/selector_bench actually need — they need
"not a fixed command," which bursty WASD already provides. Higher cost for
marginal benefit on the stated goal.

## Approach C — passive logger, driven by whatever the team already uses to manually pilot the rover (e.g. the FrodoBots phone app / web UI)

Instead of building rover control into the eval harness at all, run a
background-only recorder that polls `/data` and `/v2/screenshot` and writes
`episode.json` while a human drives through any existing control surface
(the SDK's own web UI at `localhost:8000`, or the FrodoBots app if it can
target the local SDK).

Trade-offs: least code (no `Streamer`/keyboard handling at all, just the
polling + VO + guard loop already in `eval/record.py` minus the `send_cmd`
calls), and reuses a control path the team already trusts. But it has no
command-in-episode.json field to log accurately (the recorder wouldn't know
what was actually commanded each tick unless `/data` reliably echoes it),
and it depends on an external app path this repo does not control or test
against — if the phone app's control cadence or the SDK UI's watchdog
behavior changes, this silently degrades. Also loses the tilt/stall abort
tie-in a human is executing manually since the recorder can't intervene.

## Recommendation

Approach A. It is the smallest, self-contained addition: no new
dependencies, reuses proven code on both sides (offroad's input handling,
urban's client/odometry/guards), and produces output the existing scoring
scripts already know how to read, so the very next thing anyone does with
it is run `oracle_gap.py --ep runs/teleop_ep1` and get a trustworthy number
instead of a flagged-as-confounded one. B is worth a later follow-up once A
proves the loop is useful and burst control is cited as the limiting
factor. C is not recommended: it externalizes control to a path this repo
doesn't own and can't log accurately.

## Testing (for the eventual bounded implementation)

- Offline: a unit test can exercise `Streamer`/guard logic against a fake
  `RoverClient` the way `test/` already fakes the SDK client elsewhere in
  this repo, without needing a rover.
- On-rover: one short recorded session, then `oracle_gap.py --ep <dir>` and
  `selector_bench.py --ep <dir>` run against it and the turn-consistency
  warning in `selector_bench.py` should no longer fire (or fire honestly,
  if the person genuinely mostly turned one way).

## Non-goals

- Not a live "human can override the autonomous stack mid-mission" safety
  feature.
- Not changing `oracle_gap.py` or `selector_bench.py` themselves — the
  point of matching `episode.json`'s schema is that they don't need to
  change.
- Not evaluating indoor. `erc_indoor` doesn't have GPS to begin with and its
  reference is the taught lap itself, not a VO-reconstructed drive, so this
  gap is specific to urban.
