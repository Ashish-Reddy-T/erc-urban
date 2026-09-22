# Off-road trial — measured findings (2026-08-25, mission-1 / ride 152650)

Everything here was MEASURED on the live unit, not assumed. Bot slug `quar…`.
Venue: an indoor/covered rock garden (ceiling, track lighting and an EXIT sign
are visible in `runs/offroad/172611_front.png`).

## Blocking facts

| Fact | Value | Consequence |
|---|---|---|
| GPS | `latitude=1000, longitude=1000` on every sample, ~40 min | No fix, ever. The venue is indoors. |
| Mission checkpoints | CP1/CP2 39.0 m apart, CP1==CP3 exactly | Arlington VA coords, unreachable from this room. Scoring impossible here. |
| Rear camera | `/v2/rear` -> HTTP 404 | Reversing is BLIND, despite `HAS_REAR_CAMERA=true` in .env. |
| Telemetry lag | hw_age 0.61–1.03 s | Every reaction is ~1 s late. Sets the whole control philosophy. |
| Battery | 100% throughout | Not a limiting factor. |

## Calibration measured on this unit

* **Gravity is on `accels[2]`** (rest reading `[-0.109, 0.098, 1.087]`).
  `ERC-3/src/imu_safety.py` documents index 0. That is WRONG for this unit and
  a guard built on it computes nonsense. `erc_urban/eval/record.py` is right.
* **Gyro Z bias = -1.53 deg/s while provably stationary** (rpms all zero).
  Integrating without subtracting this drifts 92 deg/min.
* **Angular sign: POSITIVE = LEFT (counter-clockwise).** Verified by compass:
  heading 47 -> 8 under angular +0.35. Gyro agrees directionally (its sign
  convention is inverted relative to the CW-from-North compass).
* **Turn rate: angular 0.35 for 2.5 s ~= 38 deg**, repeatable in both
  directions (47->8 left, 8->45 right).
* **Motion threshold:** angular 0.20 does NOT break static friction on rock
  (`wheels=False` throughout). 0.35 does. Linear 0.26 and 0.30 both move it.

## Resting attitude is NOT level and NOT constant

At rest the rover reads 6–8 deg of absolute tilt, and the POSTURE changes every
time it moves: roll went +5.2 -> +7.7 -> -7.5 -> +14.2 across four manoeuvres,
at near-identical absolute tilt. Consequences, both now implemented in
`creep.py`:

1. A tilt guard referenced to a baseline captured minutes ago silently spends
   its entire margin on posture that is perfectly fine. Every nudge
   re-references to the attitude measured immediately before it.
2. `watch` must flag on ABSOLUTE tilt and tick-to-tick RATE only. Flagging
   baseline displacement produced a screenful of false "OVER LIMIT" on a rover
   that was sitting still and safe — and a guard that cries wolf gets ignored.

## The dominant failure mode: slope rollback, not obstacles

Three forward attempts (0.30 x 1.2 s, 0.26 x 2.0 s) failed IDENTICALLY: nothing
for 4 ticks, then at tick 5 a large positive roll excursion (+10.8, +22.6 deg in
ONE tick) and the rover slid back. Operator confirmed visually: *"movement is
going on but the slope of the rock is making it come backwards."*

Lowering power 0.30 -> 0.26 changed nothing, so this is NOT a throttle problem.
The rover starts a climb it cannot finish and gravity returns it. A partial
climb followed by an uncontrolled rollback is the worst case.

Note the timing: the lurch always lands at tick 5, ~1.0 s in. With ~0.8 s of
telemetry lag the wheels actually contacted the feature ~0.2 s after moving —
the hazard is centimetres in front, inside the camera's blind zone. **The camera
sits at ~0.148 m; roughly the first 0.4 m ahead of the wheels is below frame.
The rover cannot see what it is about to climb.**

## What the safety envelope caught

`creep.py` aborted all three events correctly (LURCH on rate, TILT on
displacement). The reverse-burst recovery the operator specified worked **4 of 5
times**, returning the rover to 3–7 deg absolute tilt.

The fifth left it perched nose-up at **roll +14.2 / pitch +21.7, absolute 25.8
deg**, rock-steady (rate < 0.35 deg/tick, wheels stopped). One transient sample
read roll -101 deg mid-impact; treat single extreme samples as impact noise, but
do NOT filter them out of the abort path.

Stopped there deliberately. At 25.8 deg resting, forward climbs the slab it is
already nose-up against (backward-flip risk) and reverse goes deeper into the
pocket, blind. Correct move is a physical reposition, not another blind command.

## Recommended parameters

```
ABORT_DELTA_DEG = 12.0   # from PER-NUDGE reference, not a stored baseline
ABORT_ABS_DEG   = 28.0   # absolute; refuses to start a nudge above this
ABORT_RATE_DEG  =  7.0   # one-tick lurch; ordinary pivot on rock is ~3 deg/tick
MAX_NUDGE_S     =  3.0   # anything longer outlives what you can react to
HZ              =  5.0   # command watchdog: STREAM, never send-once
```

## Next

1. Slope-aware gating: refuse forward when pitch indicates an uphill face
   steeper than the rover can finish. Pitch is a **predictor** here, not just an
   alarm — it is knowable before committing.
2. Route selection over throttle: the wins come from picking flat slab, not from
   more power. More power mainly raises topple risk.
3. A downward-canted or wider view would remove the blind zone that caused every
   failure tonight. Worth raising with the organizers.
