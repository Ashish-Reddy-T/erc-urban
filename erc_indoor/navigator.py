"""
Teach-and-repeat navigator for the indoor lap. Pure logic, no SDK calls, so
`python -m erc_indoor.replay --nav` can drive it with a recorded lap.

Each tick:
  1. odometry progress -> localizer.predict; live frame -> localizer.update
  2. mode:
     DRIVE  follow the teach lap. Steering = heading error between the live
            view and the teach view LOOKAHEAD_M ahead (bearing.py). Large
            errors are fixed with an in-place pulse, not by arcing.
     TURN   the teach rover turned in place here. Pulse-spin toward the
            teach view just AFTER the corner until aligned. Turns are taken
            strictly in order, so a corner can never be skipped.
     PAUSE  cone in the path within ~1 m: stop briefly (operator's request),
     AVOID  then spin ~35 deg away, drive ~0.7 m past it, and hand back to
            DRIVE, whose bearing pulls the rover back onto the route.
     DONE   at the finish.

Dead time drives the design (measured on this unit): telemetry and frames are
0.6-1.3 s old, command->motion lag ~1.1 s. So spins are OPEN-LOOP bursts sized
from a measured spin rate, followed by a settle, then a fresh measurement —
never a proportional loop at tick rate, which limit-cycles against that lag.
The position estimate is pushed forward by speed x LATENCY_S before any
trigger compares it to the map.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .bearing import Bearing, bearing, dino_shift, features
from .route_map import TURN_M_PER_DEG, TURN_RATE_DPS


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# ---- tuning (env-overridable, like erc_urban/run.py) -------------------------
V_NOM = _env("V_NOM", 0.35)            # linear cmd; live probe: 0.55 peaks at 0.69 m/s. 0.45 felt too fast past the tables (2 s lag)
V_SLOW = _env("V_SLOW", 0.30)
# Live run 1: rolling yaw response ~20-25 deg/s per unit angular, seen ~2 s
# after the command (frame age + actuation). KP 0.012 oscillated +-35 deg
# with a ~10 s period; 0.006 keeps K*delay ~0.3.
KP_DEG = _env("KP_DEG", 0.006)         # angular per deg of heading error while rolling
MAX_ANG = _env("MAX_ANG", 0.25)
LIN_MIN = _env("LIN_MIN", 0.28)
# Live run 2: a steady -5 deg error persisted for 10 m because ang ~0.03 does
# not overcome the skid-steer's deadband. A small clamped integral removes it.
KI_DEG = _env("KI_DEG", 0.0015)        # angular per deg*s
I_MAX = _env("I_MAX", 0.10)        # 0.18 barely rolled (tripped the stuck watchdog)
ALIGN_DEG = _env("ALIGN_DEG", 30.0)    # above this, stop and pulse instead of arcing
LOOKAHEAD_M = _env("LOOKAHEAD_M", 1.0)
LATENCY_S = _env("LATENCY_S", 0.8)
SPIN_ANG = _env("SPIN_ANG", 0.55)      # 0.5 sometimes fails to break stiction
SPIN_RATE_PRIOR = _env("SPIN_RATE_PRIOR", 25.0)   # deg/s at SPIN_ANG; learned live
SETTLE_S = _env("SETTLE_S", 1.4)
MAX_BURST_S = _env("MAX_BURST_S", 2.5)
MIN_BURST_S = _env("MIN_BURST_S", 0.5)     # run 7: 0.4 s pulses at turn 3 often did not break stiction
SPIN_RATE_MAX = _env("SPIN_RATE_MAX", 35.0)
TURN_LEAD_M = _env("TURN_LEAD_M", 0.2)
TURN_TOL_DEG = _env("TURN_TOL_DEG", 7.0)
TURN_AIM = _env("TURN_AIM", 0.8)       # undershoot each pulse; overshoot costs a settle
TURN_FIRST_FRAC = _env("TURN_FIRST_FRAC", 0.65)
# Every corner is a right angle. Lap g6 (2026-09-26, data-stem-cell): SIFT
# right after turns 1 and 3 showed ~94 and ~100 deg were needed; the stored
# 80 / 50 come from the old lap's under-sampled gyro.
TURN_NOMINAL_MIN_DEG = _env("TURN_NOMINAL_MIN_DEG", 90.0)
# A weak (DINO-only) lock may end a turn only if it agrees with the gyro:
# turn 1 of g6 stopped at +69 of 90 on a DINO lock at margin 0.14.
TURN_WEAK_LOCK_AGREE_DEG = _env("TURN_WEAK_LOCK_AGREE_DEG", 15.0)
TURN_MAX_PULSES = int(_env("TURN_MAX_PULSES", 12))
# Run 8, turn 4: a 2.5 s open-loop pulse spun faster than the 25 deg/s prior,
# no visual lock at night, then 4 more search pulses -> ~150 deg for a ~105
# deg corner, and a false DINO lock at margin 0.09. The gyro, integrated over
# the whole turn, read -140 (turns 2/3: +121/+107), so without a lock trust
# it, stop at the nominal angle, and demand a stronger lock inside turns.
TURN_MAX_BURST_S = _env("TURN_MAX_BURST_S", 1.5)
TURN_DINO_MIN = _env("TURN_DINO_MIN", 0.12)
# Corner-sequence window for the localizer (metres of progress).
WINDOW_BACK_M = _env("WINDOW_BACK_M", 3.0)
WINDOW_AHEAD_M = _env("WINDOW_AHEAD_M", 8.0)
POST_TURN_REF_M = _env("POST_TURN_REF_M", 0.8)
FINISH_EARLY_M = _env("FINISH_EARLY_M", 0.3)
# Run 9 stopped ~1 m short of the orange cone: the teach lap ends AT the cone
# and the map has no nodes past it. So on reaching the map's finish, roll on
# by wheel distance until the cone is alongside / within reach, capped.
FINISH_EXTRA_M = _env("FINISH_EXTRA_M", 0.6)   # 1.3 overshot the cone (run 13); 0 stopped ~1 m short (run 9)
FINISH_CONE_M = _env("FINISH_CONE_M", 0.5)
# Frames are ~1 s old, so a cone "0.45 m ahead" is already under the bumper:
# react at 1 m. Then: pause (as the operator asked), spin away, drive past,
# and let DRIVE's bearing bring the rover back onto the route.
CONE_STOP_M = _env("CONE_STOP_M", 1.0)
CONE_PATH_HALF_M = _env("CONE_PATH_HALF_M", 0.25)
CONE_PAUSE_S = _env("CONE_PAUSE_S", 1.5)
CONE_SIDESTEP_DEG = _env("CONE_SIDESTEP_DEG", 35.0)
CONE_PASS_S = _env("CONE_PASS_S", 2.0)
CONE_PASS_LIN = _env("CONE_PASS_LIN", 0.35)
STUCK_S = _env("STUCK_S", 6.0)
# Progress per tick is bounded by what the rover can physically do; after a
# stalled tick odometry is ignored and vision re-localizes (see _odom_progress).
V_PHYS_MAX = _env("V_PHYS_MAX", 0.9)          # m/s
STALL_TICK_S = _env("STALL_TICK_S", 2.0)
# Wheels turning, view not moving (run 6: hung on a floor cable, wheels ran
# 11 m at 0.6 m/s while the visual position stayed at s=47.2 for 20 s).
SPIN_WINDOW_S = _env("SPIN_WINDOW_S", 4.0)
SPIN_ODO_M = _env("SPIN_ODO_M", 1.5)
SPIN_VIS_M = _env("SPIN_VIS_M", 0.25)
# ...and the image itself must be (nearly) frozen: on the cable it was, while
# the full-lap run raised two false "hung up" reverses in the dark atrium where
# the visual position merely lagged but the view was changing.
SPIN_IMG_DIFF = _env("SPIN_IMG_DIFF", 3.0)
BEARING_MIN_INLIERS = int(_env("BEARING_MIN_INLIERS", 12))
# ---- obstacle avoidance (freespace.py: where the floor ends, per heading) ----
# Frames are ~1 s old and the rover rolls ~0.35 m/s, so every distance below is
# what the camera saw ~0.35 m ago.
STOP_M = _env("STOP_M", 0.6)           # floor ends this close ahead -> stop, then escape
NEED_M = _env("NEED_M", 1.2)           # a heading must be clear this far to drive along it
SLOW_M = _env("SLOW_M", 1.6)           # below this, scale speed down
ESCAPE_SETTLE_S = _env("ESCAPE_SETTLE_S", 1.2)
ESCAPE_REVERSE_S = _env("ESCAPE_REVERSE_S", 1.5)   # 0.8 s at -0.3 never took effect (run 4)
ESCAPE_REVERSE_LIN = _env("ESCAPE_REVERSE_LIN", 0.4)
# Recovery = reverse straight back (the way it came in), then ARC forward
# toward the clear side — turning while going forward, never spinning in
# place. Resumed run: a counter bug meant the reverse never ran; four
# in-place "most open side" spins of +45 deg turned the rover 178 deg and it
# drove 5 m back the wrong way. Escapes may never swing the heading more than
# ESCAPE_MAX_SWING_DEG from where the escape began.
ESCAPE_ARC_S = _env("ESCAPE_ARC_S", 1.6)
ESCAPE_ARC_LIN = _env("ESCAPE_ARC_LIN", 0.30)
ESCAPE_ARC_ANG = _env("ESCAPE_ARC_ANG", 0.35)
ESCAPE_MAX_SWING_DEG = _env("ESCAPE_MAX_SWING_DEG", 60.0)
# After a reverse, ALWAYS arc toward the side that was clearer when it stopped.
# Lap g9: thin stool legs look like open floor once the rover has backed off,
# so "clear ahead -> resume" drove it into the same stool 11 times in a row.
ESCAPE_SIDE_DEG = _env("ESCAPE_SIDE_DEG", 25.0)
ESCAPE_RESET_M = _env("ESCAPE_RESET_M", 1.5)   # escape count resets only after this much wheel progress
MAX_ESCAPES = int(_env("MAX_ESCAPES", 4))
BOXED_WAIT_S = _env("BOXED_WAIT_S", 4.0)      # then reset and try again
BOXED_RETRIES = int(_env("BOXED_RETRIES", 3))
GAP_ALIGN_DEG = _env("GAP_ALIGN_DEG", 8.0)    # rotate to face a gap this far off before going
# ---- corridor following ------------------------------------------------------
# Live run 5 (CP2 corridor): the teach bearing was biased ~20 deg toward the
# right wall — near-wall SIFT features shift with lateral offset, not just
# heading — and four obstacle stops were needed. The direction in which the
# floor runs FARTHEST tracks the corridor's vanishing point to ~5-10 deg in
# every frame of that run. In a corridor, steer down it; the teach heading only
# chooses which opening.
CORRIDOR_FAR_M = _env("CORRIDOR_FAR_M", 3.0)      # far floor must reach this
CORRIDOR_MAX_OPEN_BINS = int(_env("CORRIDOR_MAX_OPEN_BINS", 9))   # <=~45 deg of open floor = corridor
CORRIDOR_MAX_DEV = _env("CORRIDOR_MAX_DEV", 15.0)  # far direction must be this close to the planned heading
# Lap g9: in the open piano area just after corner 1, "farthest floor" points
# diagonally back across the atrium; at 35 deg allowed deviation corridor
# following pulled the rover ~20 deg further left every tick (seen as a U-turn).
CORRIDOR_AFTER_TURN_M = _env("CORRIDOR_AFTER_TURN_M", 3.0)   # no corridor steering this soon after a corner

# ---- wheel-distance band --------------------------------------------------------
# The rover cannot be further along than its wheels turned (they over-read on
# slip and weaving, never under-read), nor far behind. Since the start / the
# last corner, progress is kept within [LO_FRAC*w - LO_M, HI_FRAC*w + HI_M] of
# wheel distance w. Run g5 (2026-09-26): folding tables seen at s~10 matched
# the new lap's views at s~20 and the estimate jumped 10 m ahead in 3 s.
# Replayed on all of that day's runs and the night lap: no corner moved.
# Map metres per wheel metre on each straight (start->corner 1, 1->2, 2->3,
# 3->4), measured on three full laps and both rovers: 0.84-0.85, 0.89-0.91,
# 0.95-0.99, 0.92-0.93. Wheel distance placed corner 1 within ~1 m on every
# lap, while the camera estimate ran 3-5 m late on three of them (the corridor
# to the piano looks the same from 26 m and 31 m). So the band is centred on
# calibrated wheel distance w*r, +-(BAND_M + BAND_FRAC*w), and a corner is
# also taken when calibrated wheel distance reaches it (_wheel_says_corner).
WHEEL_RATIO = [float(x) for x in os.environ.get("WHEEL_RATIO", "0.86,0.88,0.935,0.90").split(",")]
BAND_M = _env("BAND_M", 1.5)
BAND_FRAC = _env("BAND_FRAC", 0.05)
WHEEL_CORNER_LEAD_M = _env("WHEEL_CORNER_LEAD_M", 0.3)

# ---- keep off things beside the path -------------------------------------------
# Heading hold keeps the teach HEADING, not the teach line: started beside the
# bar stools, the rover rolled past them ~0.2 m away. Obstacle feet within
# SIDE_LOOK_M ahead and SIDE_BAND_M to a side push the heading away and slow down.
SIDE_LOOK_M = _env("SIDE_LOOK_M", 1.5)
SIDE_BAND_M = _env("SIDE_BAND_M", 0.8)
SIDE_GAP_M = _env("SIDE_GAP_M", 0.35)             # wanted gap beside the body
SIDE_MAX_DEG = _env("SIDE_MAX_DEG", 10.0)
BODY_HALF_M = _env("BODY_HALF_M", 0.10)

# ---- never drive blind ------------------------------------------------------
DINO_MIN_MARGIN = _env("DINO_MIN_MARGIN", 0.10)   # 0.08 let it follow a wrong-way view (resumed run)
BLIND_TICKS = int(_env("BLIND_TICKS", 8))          # ~2 s with no heading -> stop + search
SEARCH_STEPS_DEG = [25.0, -50.0, 75.0, -100.0, 125.0, -150.0]
APPROACH_SLOW_M = _env("APPROACH_SLOW_M", 2.5)     # slow down this far before a corner
SEARCH_TURN_LEAD_M = _env("SEARCH_TURN_LEAD_M", 2.0)

# From ERC-3's indoor SimpleLocalController (heading_filter_alpha 0.35,
# angular_rate_limit 0.10): smooth the heading error and rate-limit angular so
# ~1.2 s of dead time does not turn steering into a limit cycle.
HEADING_ALPHA = _env("HEADING_ALPHA", 0.5)
ANG_RATE_LIMIT = _env("ANG_RATE_LIMIT", 0.10)

# ---- gyro heading hold ------------------------------------------------------
# Dawn run (2026-09-26): vision matched weakly (people at the tables, grey
# light), flipped +20/-19 deg and dropped out, and the blind rule spun the
# rover off the route at s=20. Vision and gyro agreed throughout. So steer on
# the gyro against the teach lap's own yaw, and let every confident visual
# match re-anchor the offset between the two (the teach yaw drifts ~1 deg/m
# in places, so the offset is kept local).
GYRO_HEAD = _env("GYRO_HEAD", 1.0) > 0
FRAME_LAG_S = _env("FRAME_LAG_S", 1.0)            # the frame shows the heading ~this long ago
YOFF_ALPHA_SIFT = _env("YOFF_ALPHA_SIFT", 0.35)
YOFF_ALPHA_DINO = _env("YOFF_ALPHA_DINO", 0.15)
YOFF_GATE_DEG = _env("YOFF_GATE_DEG", 25.0)       # visual samples this far off the gyro are outliers...
YOFF_FLIP_N = int(_env("YOFF_FLIP_N", 6))         # ...unless this many in a row agree with each other
DINO_FIX_MAX_DEG = _env("DINO_FIX_MAX_DEG", 8.0)
GYRO_BLIND_M = _env("GYRO_BLIND_M", 40.0)         # gyro drifts only a few deg per 50 m (lap g7); DINO-only is the weaker fallback
TEACH_YAW_SMOOTH = int(_env("TEACH_YAW_SMOOTH", 15))   # half-window, nodes


# ---- steering delay predictor ---------------------------------------------------
# Morning run 2026-09-26: ang -> gyro yaw-rate peaks ~2 s later (actuation +
# telemetry), at ~65 deg/s per unit ang (night lap: ~45). Measured error is
# therefore ~2 s stale, and KP+KI weaved +-30 deg with a ~12 s period (a
# simulation of that loop reproduces it). Add the turning already commanded
# but not yet visible in the measurement (a Smith predictor).
PRED_DELAY_S = _env("PRED_DELAY_S", 2.0)
PRED_GAIN = _env("PRED_GAIN", 55.0)               # deg/s per unit ang
I_BAND_DEG = _env("I_BAND_DEG", 6.0)              # integrate only near zero (no wind-up)


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


@dataclass
class Obs:
    now: float
    frame: Optional[np.ndarray]
    s_total: float          # odometer cumulative wheel distance, m
    yaw_total: float        # odometer cumulative yaw, deg, left-positive
    v: float                # wheel speed, m/s


@dataclass
class Cmd:
    lin: float
    ang: float
    hold_s: float
    mode: str
    note: str = ""
    done: bool = False


@dataclass
class _Pulse:
    """One open-loop spin burst followed by a settle."""
    ang: float
    burst_end: float
    settle_end: float
    dur: float
    planned_deg: float      # signed, left-positive, at the spin rate believed then


@dataclass
class NavLog:
    node: int = 0
    s: float = 0.0
    p: float = 0.0
    spread: float = 0.0
    sim: float = 0.0
    ref: int = -1
    ref_src: str = "day"
    yaw_err: Optional[float] = None
    inliers: int = 0
    turn_k: int = 0
    spin_rate: float = SPIN_RATE_PRIOR
    cones: list = field(default_factory=list)
    checkpoint: str = "START"
    head_src: str = ""
    ahead_m: Optional[float] = None
    avoid_deg: Optional[float] = None
    corridor_deg: Optional[float] = None
    gyro_err: Optional[float] = None


class Navigator:
    def __init__(self, route, localizer, encoder, cone_detector=None, verbose: bool = True,
                 freespace=None, start_p: Optional[float] = None):
        self.route = route
        self.loc = localizer
        self.enc = encoder
        self.cones = cone_detector
        self.free = freespace
        self.last_fs = None          # (FreeSpace, time) for the overlay
        self.last_cones: list = []
        # Operator-stated start (START_S). Corners behind it are done, whatever
        # the first, still-uncertain estimate says (run 4 re-ran turn 1 because
        # the first estimate landed inside the turn segment).
        self.start_p = start_p
        self._live_tok = None
        self._tok_cache: dict = {}
        self._blind = 0
        self._search_i = 0
        self._escape_until = 0.0
        self._escapes = 0
        self._force_reverse = False
        self._escape_phase = "stop"
        self._boxed_since = None
        self._boxed_cycles = 0
        self._escape_end_t = -1e9
        self._escape_end_s = -1e9
        self._escape_side = 1.0
        self._escape_yaw0 = 0.0
        self._arc_ang = 0.0
        self._finish_odo0 = 0.0
        self._boost = 1.0            # grows when a pulse produces no visible rotation (stiction)
        self.verbose = verbose
        self.mode = "INIT"
        self.turn_k = 0
        self.spin_rate = SPIN_RATE_PRIOR
        self.log = NavLog()
        self._prev: Optional[Obs] = None
        self._init_ticks = 0
        self._pulse: Optional[_Pulse] = None
        self._turn_yaw0 = 0.0
        self._turn_est = 0.0          # open-loop estimate of degrees turned this corner
        self._last_pulse: Optional[_Pulse] = None
        self._last_lock: Optional[tuple[int, float]] = None   # (ref node, yaw_err) before the pulse
        self._turn_nominal = 0.0
        self._turn_pulses = 0
        self._last_ang = 0.0
        self._e_f: Optional[float] = None
        self._c_f: Optional[float] = None
        self._i_term = 0.0
        self._i_t: Optional[float] = None
        self._bearing_fail = 0
        self._pause_end = 0.0
        self._avoid_deg = 0.0
        self._avoid_phase = ""
        self._pass_end = 0.0
        self._cone_hits = 0
        self._move_ref: Optional[tuple[float, float]] = None
        self._prog_hist: list = []            # (now, odo_s, visual s, image diff) while DRIVE
        self._thumb = None
        self._passed_cps: set[str] = set()
        # gyro heading hold
        self._yaw_hist: list = []            # (now, yaw_total)
        self._yoff: Optional[float] = None   # live gyro yaw minus teach yaw
        self._yoff_n = 0
        self._yoff_s = 0.0                   # odometer s at the last visual fix
        self._yoff_out: list = []            # consecutive outlier samples
        self._teach_yaw = self._smooth_teach_yaw()
        self._cmd_hist: list = []            # (now, ang) of every command issued
        self._band: Optional[tuple[float, float]] = None   # (anchor p, odometer s there)

    # ---- helpers -------------------------------------------------------------
    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _odom_progress(self, obs: Obs) -> float:
        if self._prev is None:
            return 0.0
        dt = max(obs.now - self._prev.now, 1e-3)
        if dt > STALL_TICK_S:
            # the loop stalled (SDK stopped answering): odometry across the gap
            # is not trustworthy — let vision re-localize from a wider belief
            self._say(f"[nav] loop stalled {dt:.0f} s — ignoring odometry across the gap")
            self.loc.predict(0.0)
            self.loc.b = self.loc.b ** 0.3
            self.loc.b /= self.loc.b.sum()
            return 0.0
        ds = obs.s_total - self._prev.s_total
        dyaw = obs.yaw_total - self._prev.yaw_total
        turning = abs(dyaw) / dt > TURN_RATE_DPS
        dp = ds + (TURN_M_PER_DEG * abs(dyaw) if turning else 0.0)
        return float(np.clip(dp, -V_PHYS_MAX * dt, V_PHYS_MAX * dt + TURN_M_PER_DEG * 60.0 * dt))

    def _wheel_ratio(self) -> float:
        k = min(self.turn_k, len(WHEEL_RATIO) - 1)
        return WHEEL_RATIO[k]

    def _wheel_says_corner(self, obs: Obs, t) -> bool:
        """Calibrated wheel distance since the start / last corner has reached
        the next corner (the camera estimate may lag on corridor approaches)."""
        if self._band is None:
            return False
        p0, o0 = self._band
        p_wheel = p0 + self._wheel_ratio() * (obs.s_total - o0)
        return p_wheel >= float(self.route.p[t.start]) - WHEEL_CORNER_LEAD_M

    def _smooth_teach_yaw(self) -> np.ndarray:
        """Teach yaw, box-smoothed within each straight (not across corners)."""
        y = np.asarray(self.route.yaw, dtype=float).copy()
        out = y.copy()
        edges = [0]
        for t in self.route.turns:
            edges += [t.start, t.end]
        edges.append(len(y))
        w = TEACH_YAW_SMOOTH
        for a, b in zip(edges[0::2], edges[1::2]):
            for i in range(a, b):
                lo, hi = max(a, i - w), min(b, i + w + 1)
                out[i] = y[lo:hi].mean()
        return out

    def _yaw_at(self, t: float) -> Optional[float]:
        h = self._yaw_hist
        if not h:
            return None
        if t <= h[0][0]:
            return h[0][1]
        for (t0, y0), (t1, y1) in zip(h, h[1:]):
            if t0 <= t <= t1:
                return y0 + (y1 - y0) * (t - t0) / max(t1 - t0, 1e-6)
        return h[-1][1]

    def _gyro_fix(self, obs: Obs, ref: int, e: float, sift: bool) -> None:
        """A visual heading error to teach node `ref` corrects the gyro offset.

        Only SIFT (many agreeing matches) may move it far or re-anchor it. A
        DINO-only fix may nudge it by <= DINO_FIX_MAX_DEG: lap g7 (2026-09-26)
        re-anchored on a DINO fix at margin 0.10 right after corner 2, then held
        the wrong heading with confidence — in the plain white corridors DINO
        says "~0 deg" whichever way the rover faces — and doubled back ~150 deg."""
        y_then = self._yaw_at(obs.now - FRAME_LAG_S)
        if y_then is None:
            return
        sample = y_then - float(self._teach_yaw[ref]) - e
        if self._yoff is None:
            if sift:
                self._yoff, self._yoff_n, self._yoff_s = sample, 1, obs.s_total
            return
        r = _wrap(sample - self._yoff)
        if not sift:
            if abs(r) <= DINO_FIX_MAX_DEG:
                self._yoff += YOFF_ALPHA_DINO * r
                self._yoff_s = obs.s_total
            return
        if abs(r) > YOFF_GATE_DEG and self._yoff_n >= 3:
            self._yoff_out.append(sample)
            if len(self._yoff_out) >= YOFF_FLIP_N:
                o = np.array(self._yoff_out)
                if float(np.ptp(o)) < YOFF_GATE_DEG:
                    self._say(f"[gyro] SIFT disagrees persistently — re-anchoring offset "
                              f"{self._yoff:+.0f} -> {float(np.median(o)):+.0f}")
                    self._yoff = float(np.median(o))
                    self._yoff_n = 1
                self._yoff_out = []
            return
        a = YOFF_ALPHA_SIFT if self._yoff_n >= 3 else 0.5
        self._yoff += a * r
        self._yoff_n += 1
        self._yoff_out = []
        self._yoff_s = obs.s_total

    def _gyro_err(self, obs: Obs, ref: int) -> Optional[float]:
        """Heading error from the gyro (deg, + = live points LEFT of teach), or
        None when there is no recent visual fix to trust it by."""
        if not GYRO_HEAD or self._yoff is None:
            return None
        if obs.s_total - self._yoff_s > GYRO_BLIND_M:
            return None
        return _wrap(obs.yaw_total - self._yoff - float(self._teach_yaw[ref]))

    def _bearing_to(self, live_feat, node: int) -> Bearing:
        """Day and night picture of the node; the one with more agreeing matches
        wins. (Live run 1: at night the day view sometimes passed on ~20
        ceiling-light matches and disagreed with the night view by ~12 deg.)"""
        b = bearing(live_feat, self.route.features(node))
        self.log.ref_src = "day"
        for t in range(self.route.n_twins):
            tf = self.route.twin_features(node, t)
            if tf is None:
                continue
            bt = bearing(live_feat, tf)
            if bt.n_inliers > b.n_inliers:
                b = bt
                self.log.ref_src = self.route.twin_name(t)
        self.log.ref = node
        self.log.yaw_err = b.yaw_err_deg if b.ok else None
        self.log.inliers = b.n_inliers
        return b

    def _ref_tokens(self, node: int, t: int):
        """DINO patch tokens of node's picture: t=-1 the day map, t>=0 extra view set t."""
        key = (int(node), t)
        if key not in self._tok_cache:
            img = self.route.twin_image(node, t) if t >= 0 else self.route.image(node)
            if img is None:
                return None
            if len(self._tok_cache) > 300:
                self._tok_cache.clear()
            self._tok_cache[key] = self.enc.encode_full(img)[1]
        return self._tok_cache[key]

    def _heading(self, live_feat, node: int, dino_min: float = DINO_MIN_MARGIN) -> Optional[float]:
        """Heading error to teach node (deg, + = live points LEFT of teach).
        SIFT (day/night) first; DINOv2 patch alignment when SIFT fails
        (run 3: 3-9 SIFT inliers on the plain walls at the piano doorway,
        where the patch grid still gave a consistent answer)."""
        b = self._bearing_to(live_feat, node)
        if b.ok and b.n_inliers >= BEARING_MIN_INLIERS:
            self.log.head_src = "sift-" + self.log.ref_src
            return b.yaw_err_deg
        if self._live_tok is None:
            self.log.head_src = ""
            return None
        best = None
        for t in [-1] + list(range(self.route.n_twins)):
            rt = self._ref_tokens(node, t)
            if rt is None:
                continue
            e, m = dino_shift(self._live_tok, rt)
            if best is None or m > best[1]:
                best = (e, m, t)
        if best is not None and best[1] >= dino_min:
            name = self.route.twin_name(best[2]) if best[2] >= 0 else "day"
            self.log.head_src = f"dino-{name}({best[1]:.2f})"
            self.log.yaw_err = best[0]
            return best[0]
        self.log.head_src = ""
        self.log.yaw_err = None
        return None

    def _start_pulse(self, obs: Obs, turn_deg: float, note: str) -> Cmd:
        cap = TURN_MAX_BURST_S if self.mode == "TURN" else MAX_BURST_S
        dur = float(np.clip(self._boost * abs(turn_deg) / max(self.spin_rate, 5.0), MIN_BURST_S, cap))
        ang = SPIN_ANG * (1.0 if turn_deg > 0 else -1.0)
        planned = math.copysign(dur * self.spin_rate, turn_deg)
        self._pulse = _Pulse(ang, obs.now + dur, obs.now + dur + SETTLE_S, dur, planned)
        self._turn_est += planned
        return Cmd(0.0, ang, dur, self.mode, f"pulse {turn_deg:+.0f}deg {dur:.2f}s {note}")

    def _pulse_tick(self, obs: Obs) -> Optional[Cmd]:
        """Returns a command while a pulse is in progress, None once settled."""
        p = self._pulse
        if p is None:
            return None
        if obs.now < p.burst_end:
            return Cmd(0.0, p.ang, p.burst_end - obs.now, self.mode, "burst")
        if obs.now < p.settle_end:
            return Cmd(0.0, 0.0, 0.3, self.mode, "settle")
        # Spin rate is learned VISUALLY (_learn_rate), not from the gyro: live
        # probe, 1 s left spin: camera +23.9 deg, gyro +6.8 deg. The gyro's
        # 5-sample bursts every 0.5 s barely sample a short spin.
        self._last_pulse = p
        self._pulse = None
        # a spin covers no wheel distance; don't let it count as "stuck"
        self._move_ref = None
        return None

    def _window(self) -> tuple[float, float]:
        """Where the rover can be, given the corners already taken."""
        P, T = self.route.p, self.route.turns
        lo = float(P[T[self.turn_k - 1].end]) - WINDOW_BACK_M if self.turn_k > 0 else -1e9
        if self.turn_k < len(T):
            hi = float(P[T[self.turn_k].end]) + WINDOW_AHEAD_M
        else:
            hi = 1e9
        return lo, hi

    def _learn_rate(self, ref: int, yaw_err: float) -> None:
        """Visual spin-rate update: same reference before and after one pulse,
        so the change in heading error IS the rotation that pulse produced."""
        p, prev = self._last_pulse, self._last_lock
        self._last_lock = (ref, yaw_err)
        if p is None or prev is None or prev[0] != ref:
            return
        turned = yaw_err - prev[1]              # left-positive
        self._last_pulse = None
        if abs(turned) < 3.0:
            self._boost = min(self._boost * 1.5, 3.0)   # pulse did nothing: stiction
            return
        self._boost = 1.0
        if np.sign(turned) != np.sign(p.ang):
            return
        rate = abs(turned) / p.dur
        self.spin_rate = float(np.clip(0.5 * self.spin_rate + 0.5 * rate, 8.0, SPIN_RATE_MAX))
        self.log.spin_rate = self.spin_rate

    def _checkpoint_log(self, s: float) -> None:
        for name, cs in self.route.checkpoints.items():
            if s >= cs and name not in self._passed_cps:
                self._passed_cps.add(name)
                self._say(f"[checkpoint] passing {name} (map s={cs:.1f} m)")
        self.log.checkpoint = self.route.checkpoint_label(s)

    # ---- main ----------------------------------------------------------------
    def step(self, obs: Obs) -> Cmd:
        cmd = self._step(obs)
        self._cmd_hist.append((obs.now, float(cmd.ang)))
        self._cmd_hist = [h for h in self._cmd_hist if obs.now - h[0] <= PRED_DELAY_S + 1.0]
        return cmd

    def _pending_turn(self, now: float) -> float:
        """Degrees (left +) commanded over the last PRED_DELAY_S that the
        measurement cannot show yet."""
        h = self._cmd_hist
        tot = 0.0
        for (t0, a), t1 in zip(h, [x[0] for x in h[1:]] + [now]):
            lo = max(t0, now - PRED_DELAY_S)
            if t1 > lo:
                tot += a * (t1 - lo)
        return PRED_GAIN * tot

    def _step(self, obs: Obs) -> Cmd:
        dp = self._odom_progress(obs)
        self._prev = obs
        self._yaw_hist.append((obs.now, obs.yaw_total))
        self._yaw_hist = [h for h in self._yaw_hist if obs.now - h[0] <= 4.0]
        self.loc.predict(dp)

        in_burst = self._pulse is not None and obs.now < self._pulse.burst_end
        live_feat = None
        self._live_tok = None
        if obs.frame is not None and not in_burst:
            desc, self._live_tok = self.enc.encode_full(obs.frame)
            self.loc.update(desc)
        if self.mode != "INIT":
            self.loc.constrain(*self._window())
        if self._band is not None and self.mode != "TURN":
            p0, o0 = self._band
            w = obs.s_total - o0
            c = p0 + self._wheel_ratio() * w
            m = BAND_M + BAND_FRAC * abs(w)
            self.loc.constrain(c - m, c + m)
        est = self.loc.estimate()
        lead = max(obs.v, 0.0) * LATENCY_S
        p_now = est.p + lead
        s_now = est.s + lead
        self.log.node, self.log.s, self.log.p = est.node, est.s, est.p
        self.log.spread, self.log.sim, self.log.turn_k = est.spread_m, est.sim, self.turn_k
        self._checkpoint_log(s_now)

        # ---------------------------------------------------------------- INIT
        if self.mode == "INIT":
            self._init_ticks += 1
            if self._init_ticks < 3:
                return Cmd(0.0, 0.0, 0.3, "INIT", "localizing at start")
            self.mode = "DRIVE"
            # mid-route starts (START_S): corners already behind us are done
            p_ref = self.start_p if self.start_p is not None else est.p
            while (self.turn_k < len(self.route.turns)
                   and p_ref > self.route.p[self.route.turns[self.turn_k].end]):
                self.turn_k += 1
            self._band = (self.start_p if self.start_p is not None else float(self.route.p[0]), obs.s_total)
            if GYRO_HEAD and self.start_p is None:
                # the operator places the rover at the cone facing down the route
                self._yoff = obs.yaw_total - float(self._teach_yaw[est.node])
                self._yoff_n, self._yoff_s = 0, obs.s_total
            self._say(f"[nav] start localized at node {est.node} s={est.s:.1f} spread={est.spread_m:.1f}; "
                      f"next corner {self.turn_k + 1}/{len(self.route.turns)}")

        if self.mode == "DONE":
            return Cmd(0.0, 0.0, 0.5, "DONE", "finished", done=True)

        # ---------------------------------------------------------------- TURN
        if self.mode == "TURN":
            c = self._pulse_tick(obs)
            if c is not None:
                return c
            return self._turn_measure(obs, est)

        # ------------------------------------------------------ pulses in DRIVE
        c = self._pulse_tick(obs)
        if c is not None:
            return c

        # --------------------------------------------------------------- ESCAPE
        if self.mode == "ESCAPE":
            return self._escape(obs, est)
        if self.mode == "ESCAPE_TURN":
            c = self._pulse_tick(obs)
            if c is not None:
                return c
            self.mode = "DRIVE"
            self._move_ref = None
        # --------------------------------------------------------------- SEARCH
        if self.mode == "SEARCH":
            # lost right before a corner (full lap: s=128.3, turn 4 at 129.6,
            # held still facing the atrium windows): take the corner instead
            if self.turn_k < len(self.route.turns) and self._pulse is None:
                t = self.route.turns[self.turn_k]
                if p_now >= self.route.p[t.start] - SEARCH_TURN_LEAD_M:
                    self._say("[nav] heading lost just before the corner — taking the corner")
                    return self._turn_enter(obs, est, t)
            return self._search(obs, est)

        # ------------------------------------------------------- PAUSE / AVOID
        if self.mode == "PAUSE":
            if obs.now < self._pause_end:
                return Cmd(0.0, 0.0, 0.3, "PAUSE", "cone ahead")
            self.mode = "AVOID"
            self._avoid_phase = "spin"
            return self._start_pulse(obs, self._avoid_deg, "cone sidestep")
        if self.mode == "AVOID":
            if self._avoid_phase == "spin":
                self._avoid_phase = "pass"
                self._pass_end = obs.now + CONE_PASS_S
            if obs.now < self._pass_end:
                return Cmd(CONE_PASS_LIN, 0.0, min(0.6, self._pass_end - obs.now), "AVOID", "passing cone")
            self.mode = "DRIVE"
            self._e_f = self._c_f = None
            self._i_term, self._i_t = 0.0, None
            self._last_ang = 0.0
            self._move_ref = None

        # --------------------------------------------------------------- finish
        if self.mode == "FINISH":
            return self._finish_roll(obs)
        if self.turn_k >= len(self.route.turns) and s_now >= self.route.finish_s - FINISH_EARLY_M:
            self.mode = "FINISH"
            self._finish_odo0 = obs.s_total
            self._say(f"[nav] finish reached on the map at s={s_now:.1f} m — rolling to the cone")
            return self._finish_roll(obs)

        # ---------------------------------------------------------- turn entry
        if self.turn_k < len(self.route.turns):
            t = self.route.turns[self.turn_k]
            if p_now >= self.route.p[t.start] - TURN_LEAD_M:
                return self._turn_enter(obs, est, t)
            if self._wheel_says_corner(obs, t):
                self._say(f"[nav] wheel distance reached corner {self.turn_k + 1} "
                          f"(camera estimate s={est.s:.1f}) — turning")
                return self._turn_enter(obs, est, t)

        # --------------------------------------------------------------- cones
        if self.cones is not None and obs.frame is not None:
            cs = self.cones.detect(obs.frame)
            self.last_cones = cs
            self.log.cones = [(c.color, round(c.dist_m, 2), round(c.y_m, 2)) for c in cs[:3]]
            blocking = [c for c in cs if c.x_m < CONE_STOP_M and abs(c.y_m) < CONE_PATH_HALF_M]
            self._cone_hits = self._cone_hits + 1 if blocking else 0
            if self._cone_hits >= 2:
                c0 = blocking[0]
                self.mode = "PAUSE"
                self._pause_end = obs.now + CONE_PAUSE_S
                # pass on the side AWAY from the cone (y>0: cone is left -> go right)
                self._avoid_deg = -CONE_SIDESTEP_DEG if c0.y_m > 0.05 else CONE_SIDESTEP_DEG
                self._cone_hits = 0
                self._say(f"[cone] {c0.color} cone {c0.dist_m:.2f} m ahead (y={c0.y_m:+.2f}) — pausing, "
                          f"then passing {'right' if self._avoid_deg < 0 else 'left'}")
                return Cmd(0.0, 0.0, 0.3, "PAUSE", "cone")

        # --------------------------------------------------------------- DRIVE
        self.mode = "DRIVE"
        cap = self.route.turns[self.turn_k].start if self.turn_k < len(self.route.turns) else len(self.route) - 1
        ref = min(self.route.node_at_p(est.p + LOOKAHEAD_M), cap)
        if obs.frame is None:
            return Cmd(0.0, 0.0, 0.3, "DRIVE", "no frame")
        e_vis = self._heading(features(obs.frame), ref)
        if e_vis is not None and GYRO_HEAD:
            self._gyro_fix(obs, ref, e_vis, self.log.head_src.startswith("sift"))
        # teach yaw inside a corner is mid-rotation: after a corner, measure
        # against the straight that follows it (run g1: the estimate slid back
        # into turn 1, the mid-turn yaw said "54 deg too far left", and the
        # rover turned back toward the piano)
        gref = ref
        if self.turn_k > 0:
            gref = max(gref, self.route.turns[self.turn_k - 1].end + 3)
        e_gyro = self._gyro_err(obs, gref)
        self.log.gyro_err = e_gyro
        e = e_gyro if e_gyro is not None else e_vis
        if e_gyro is not None:
            self.log.head_src = (self.log.head_src + "+gyro") if e_vis is not None else "gyro"

        # corridor/avoid guides are per-tick; clear stale ones for the overlay
        self.log.corridor_deg = None
        # --- obstacle check first: never drive into what the camera can see
        fs = self.free.estimate(obs.frame) if self.free is not None else None
        if fs is not None:
            self.last_fs = (fs, obs.now)
        self.log.ahead_m = fs.ahead_m if fs is not None else None
        self.log.avoid_deg = None
        if fs is not None and fs.ahead_m < STOP_M:
            return self._begin_escape(obs, f"floor ends {fs.ahead_m:.2f} m ahead")

        # --- never drive blind (a clear corridor direction counts as a heading)
        corridor_seen = (fs is not None and float(fs.clear_m.max()) >= CORRIDOR_FAR_M
                         and 0 < int((fs.clear_m >= CORRIDOR_FAR_M).sum()) <= CORRIDOR_MAX_OPEN_BINS)
        if e is None and not corridor_seen:
            self._blind += 1
            if self._blind >= BLIND_TICKS:
                self._blind = 0
                self._search_i = 0
                self.mode = "SEARCH"
                self._say(f"[nav] no heading for {BLIND_TICKS} ticks at node {est.node} — stopping to search")
                return Cmd(0.0, 0.0, 0.3, "SEARCH", "lost heading")
        else:
            self._blind = 0

        if e is not None:
            e = e + self._pending_turn(obs.now)      # where the heading is about to be
        want = -e if e is not None else 0.0          # deg to turn, left-positive
        corridor = None
        if fs is not None:
            far = float(fs.clear_m.max())
            n_open = int((fs.clear_m >= CORRIDOR_FAR_M).sum())
            since_turn = obs.s_total - self._band[1] if (self._band is not None and self.turn_k > 0) else 1e9
            if far >= CORRIDOR_FAR_M and 0 < n_open <= CORRIDOR_MAX_OPEN_BINS and since_turn >= CORRIDOR_AFTER_TURN_M:
                cand = fs.bins_deg[fs.clear_m >= 0.9 * far]
                cand = cand[np.abs(cand - want) <= CORRIDOR_MAX_DEV]
                if len(cand):
                    corridor = float(np.median(cand))
        if corridor is not None:
            want = corridor
        self.log.corridor_deg = corridor
        # keep off obstacles beside the path
        side_close = False
        if fs is not None and fs.pts is not None and len(fs.pts):
            P = np.asarray(fs.pts)
            near = P[(P[:, 0] > 0.05) & (P[:, 0] < SIDE_LOOK_M) & (np.abs(P[:, 1]) < SIDE_BAND_M)
                     & (np.abs(P[:, 1]) > BODY_HALF_M + 0.05)]     # in-path points are AVOID's job
            gl = float(near[near[:, 1] > 0, 1].min()) - BODY_HALF_M if (near[:, 1] > 0).any() else 9.0
            gr = float(-near[near[:, 1] < 0, 1].max()) - BODY_HALF_M if (near[:, 1] < 0).any() else 9.0
            push = 0.0
            if gl < SIDE_GAP_M:
                push -= SIDE_MAX_DEG * (SIDE_GAP_M - gl) / SIDE_GAP_M
            if gr < SIDE_GAP_M:
                push += SIDE_MAX_DEG * (SIDE_GAP_M - gr) / SIDE_GAP_M
            push = float(np.clip(push, -SIDE_MAX_DEG, SIDE_MAX_DEG))
            if abs(push) > 0.5:
                want += push
            side_close = min(gl, gr) < SIDE_GAP_M
        avoiding = False
        if fs is not None and fs.clear_at(want) < NEED_M:
            alt = self.free.steer(fs, want, NEED_M)
            if alt is None:
                alt = self.free.steer(fs, want, STOP_M + 0.2)
            if alt is None:
                return self._begin_escape(obs, "no clear heading in view")
            want, avoiding = alt, True
            self.log.avoid_deg = alt

        if abs(want) > ALIGN_DEG or (avoiding and abs(want) >= GAP_ALIGN_DEG
                                     and fs is not None and fs.ahead_m < NEED_M):
            # a gap off to one side and the way ahead short: face the gap in
            # place first (doorways: a gentle arc kept clipping the jamb)
            self._e_f = self._c_f = None
            self._i_term, self._i_t = 0.0, None
            return self._start_pulse(obs, want * TURN_AIM,
                                     f"{'face gap' if avoiding else 'realign'} ({want:+.0f})")
        if avoiding or corridor is not None:
            # steer to the clear / corridor heading; the teach integral must not fight it
            self._c_f = want if self._c_f is None else HEADING_ALPHA * want + (1 - HEADING_ALPHA) * self._c_f
            gain = KP_DEG * (1.5 if avoiding else 1.0)
            ang = float(np.clip(gain * self._c_f, -MAX_ANG, MAX_ANG))
            self._i_t = None
        elif e is not None:
            self._c_f = None
            self._e_f = e if self._e_f is None else HEADING_ALPHA * e + (1 - HEADING_ALPHA) * self._e_f
            if self._i_t is not None and abs(self._e_f) < I_BAND_DEG:
                dt = min(obs.now - self._i_t, 0.5)
                self._i_term = float(np.clip(self._i_term - KI_DEG * self._e_f * dt, -I_MAX, I_MAX))
            self._i_t = obs.now
            ang = float(np.clip(-KP_DEG * self._e_f + self._i_term, -MAX_ANG, MAX_ANG))
        else:
            ang = 0.5 * self._last_ang

        lin = max(LIN_MIN, V_NOM * float(np.clip(1.0 - abs(want) / 45.0, 0.4, 1.0)))
        if est.spread_m > 3.0 or (e is None and corridor is None) or avoiding or side_close:
            lin = min(lin, V_SLOW)
        if self.turn_k < len(self.route.turns):
            to_corner = self.route.p[self.route.turns[self.turn_k].start] - (est.p + max(obs.v, 0) * LATENCY_S)
            if to_corner < APPROACH_SLOW_M:
                lin = min(lin, V_SLOW)
        if fs is not None and fs.ahead_m < SLOW_M:
            lin = max(LIN_MIN, lin * float(np.clip((fs.ahead_m - STOP_M) / (SLOW_M - STOP_M), 0.0, 1.0)))
        ang = float(np.clip(ang, self._last_ang - ANG_RATE_LIMIT, self._last_ang + ANG_RATE_LIMIT))

        # stuck watchdog 1: wheels spinning, view not moving
        thumb = cv2.resize(cv2.cvtColor(obs.frame, cv2.COLOR_RGB2GRAY), (64, 36)).astype(np.float32)
        img_diff = float(np.abs(thumb - self._thumb).mean()) if self._thumb is not None else 99.0
        self._thumb = thumb
        self._prog_hist.append((obs.now, obs.s_total, est.s, img_diff))
        self._prog_hist = [h for h in self._prog_hist if obs.now - h[0] <= SPIN_WINDOW_S]
        h0 = self._prog_hist[0]
        frozen = max(h[3] for h in self._prog_hist[1:]) < SPIN_IMG_DIFF if len(self._prog_hist) > 2 else False
        if (obs.now - h0[0] >= SPIN_WINDOW_S - 0.5 and obs.s_total - h0[1] > SPIN_ODO_M
                and est.s - h0[2] < SPIN_VIS_M and frozen):
            self._prog_hist = []
            return self._begin_escape(obs, f"wheels ran {obs.s_total - h0[1]:.1f} m, view moved "
                                           f"{est.s - h0[2]:.2f} m — hung up", force_reverse=True)
        # stuck watchdog 2: commanded forward, wheels not advancing
        if self._move_ref is None or obs.s_total - self._move_ref[1] > 0.15:
            self._move_ref = (obs.now, obs.s_total)
        elif obs.now - self._move_ref[0] > STUCK_S:
            self._move_ref = (obs.now, obs.s_total)
            return self._begin_escape(obs, "no wheel progress")

        self._last_ang = ang
        if obs.s_total - self._escape_end_s > ESCAPE_RESET_M:
            self._escapes = 0
        note = (f"ref {ref} err {e:+.1f} {self.log.head_src}" if e is not None else f"ref {ref} no-heading x{self._blind}")
        if e_gyro is not None and e_vis is not None:
            note += f" (vis {e_vis:+.0f})"
        if corridor is not None:
            note += f" CORRIDOR->{corridor:+.0f}"
        if avoiding:
            note += f" AVOID->{want:+.0f}"
        return Cmd(lin, ang, 0.6, "DRIVE", note)

    def _finish_roll(self, obs: Obs) -> Cmd:
        rolled = obs.s_total - self._finish_odo0
        near_cone = False
        if self.cones is not None and obs.frame is not None:
            for c in self.cones.detect(obs.frame):
                if c.color == "red" and c.dist_m < FINISH_CONE_M + 0.6 and abs(c.bearing_deg) > 35:
                    near_cone = True       # alongside: crossing the line
                if c.color == "red" and c.x_m < FINISH_CONE_M and abs(c.y_m) < 0.3:
                    near_cone = True       # right in front: stop, don't hit it
        fs = self.free.estimate(obs.frame) if (self.free is not None and obs.frame is not None) else None
        blocked = fs is not None and fs.ahead_m < STOP_M
        if rolled >= FINISH_EXTRA_M or near_cone or blocked:
            self.mode = "DONE"
            why = "cone alongside/ahead" if near_cone else ("obstacle ahead" if blocked else f"rolled {rolled:.1f} m")
            self._say(f"[nav] FINISH ({why})")
            return Cmd(0.0, 0.0, 0.5, "DONE", "finish line", done=True)
        return Cmd(V_SLOW, 0.0, 0.6, "FINISH", f"rolling to the cone ({rolled:.1f}/{FINISH_EXTRA_M:.1f} m)")

    # ---- escape / search -------------------------------------------------------
    def _begin_escape(self, obs: Obs, why: str, force_reverse: bool = False) -> Cmd:
        """Stop, then: reverse straight back -> look -> arc forward toward the
        clear side. Repeats at most MAX_ESCAPES times, then holds still."""
        if self.mode != "ESCAPE" and obs.now - self._escape_end_t > 8.0:
            # new episode (not a re-block right after arcing out)
            self._escape_yaw0 = obs.yaw_total        # swing limit is relative to here
            self._escapes = 0
        self.mode = "ESCAPE"
        self._prog_hist = []
        self._escape_phase = "stop"
        self._escape_until = obs.now + ESCAPE_SETTLE_S
        self._escapes += 1
        # which side had more room when we stopped (the frame that saw the obstacle)
        self._escape_side = 1.0
        if self.last_fs is not None and obs.now - self.last_fs[1] < 2.0:
            fs0 = self.last_fs[0]
            left = float(np.mean(fs0.clear_m[(fs0.bins_deg >= 10) & (fs0.bins_deg <= 45)]))
            right = float(np.mean(fs0.clear_m[(fs0.bins_deg <= -10) & (fs0.bins_deg >= -45)]))
            self._escape_side = 1.0 if left >= right else -1.0
        self._e_f = self._c_f = None
        self._i_term, self._i_t = 0.0, None
        self._last_ang = 0.0
        self._say(f"[obstacle] {why} — stop (escape #{self._escapes}, will arc "
                  f"{'left' if self._escape_side > 0 else 'right'})")
        return Cmd(0.0, 0.0, 0.3, "ESCAPE", why)

    def _escape(self, obs: Obs, est) -> Cmd:
        if obs.now < self._escape_until:
            if self._escape_phase == "reverse":
                return Cmd(-ESCAPE_REVERSE_LIN, 0.0, min(0.6, self._escape_until - obs.now), "ESCAPE", "reversing")
            if self._escape_phase == "arc":
                return Cmd(ESCAPE_ARC_LIN, self._arc_ang, min(0.6, self._escape_until - obs.now), "ESCAPE", "arcing out")
            return Cmd(0.0, 0.0, 0.3, "ESCAPE", "settling for a fresh frame")

        if self._escape_phase == "arc":            # arc finished: back to normal driving
            self._escape_end_t = obs.now
            self._escape_end_s = obs.s_total
            self._boxed_cycles = 0
            self.mode = "DRIVE"
            self._move_ref = None
            return Cmd(0.0, 0.0, 0.3, "DRIVE", "escape done (arced out)")
        if self._escape_phase == "reverse":        # reverse finished: settle for a fresh frame
            self._escape_phase = "look"
            self._escape_until = obs.now + ESCAPE_SETTLE_S
            return Cmd(0.0, 0.0, 0.3, "ESCAPE", "settling after reverse")

        if obs.frame is None or self.free is None:
            self.mode = "DRIVE"
            return Cmd(0.0, 0.0, 0.3, "DRIVE", "escape: no free-space estimate")
        fs = self.free.estimate(obs.frame)
        self.last_fs = (fs, obs.now)
        self.log.ahead_m = fs.ahead_m
        cap = self.route.turns[self.turn_k].start if self.turn_k < len(self.route.turns) else len(self.route) - 1
        e = self._heading(features(obs.frame), min(self.route.node_at_p(est.p + LOOKAHEAD_M), cap))
        want = -e if e is not None else 0.0
        swung = obs.yaw_total - self._escape_yaw0
        # headings that would take the total escape swing past the limit are not allowed
        lo, hi = -ESCAPE_MAX_SWING_DEG - swung, ESCAPE_MAX_SWING_DEG - swung
        want = float(np.clip(want, lo, hi))

        if self._escape_phase == "stop" and fs.ahead_m >= NEED_M and abs(want) < 10:
            # the obstacle is gone (a person walked off) — no need to back off
            self.mode = "DRIVE"
            self._move_ref = None
            return Cmd(0.0, 0.0, 0.3, "DRIVE", f"escape done, {fs.ahead_m:.1f} m clear")

        # 1) first time here (or still blocked close in): reverse straight back
        if self._escape_phase == "stop" and (self._force_reverse or fs.ahead_m < NEED_M):
            self._force_reverse = False
            if self._escapes > MAX_ESCAPES:
                return self._boxed(obs)
            self._escape_phase = "reverse"
            self._escape_until = obs.now + ESCAPE_REVERSE_S
            self._say(f"[obstacle] reversing {ESCAPE_REVERSE_S:.1f} s (ahead {fs.ahead_m:.2f} m)")
            return Cmd(-ESCAPE_REVERSE_LIN, 0.0, 0.6, "ESCAPE", "reversing")

        # 2) after reversing: arc forward to the side that had more room, never
        #    straight back at the same obstacle
        alt = None
        if self._escape_phase == "look":
            for side in (self._escape_side, -self._escape_side):
                cand = float(np.clip(side * ESCAPE_SIDE_DEG + 0.3 * want, lo, hi))
                if abs(cand) >= 10 and fs.clear_at(cand) >= STOP_M + 0.3:
                    alt = cand
                    break
        if alt is not None:
            self._arc_ang = float(np.clip(alt / 45.0, -1.0, 1.0)) * ESCAPE_ARC_ANG
            self._escape_phase = "arc"
            self._escape_until = obs.now + ESCAPE_ARC_S
            self._say(f"[obstacle] arcing forward toward {alt:+.0f} deg ({fs.clear_at(alt):.1f} m clear), "
                      f"ang {self._arc_ang:+.2f}")
            return Cmd(ESCAPE_ARC_LIN, self._arc_ang, 0.6, "ESCAPE", f"arc {alt:+.0f}")
        alt = self.free.steer(fs, want, NEED_M)
        if alt is not None and not (lo <= alt <= hi):
            alt = None
        if alt is None:
            alt = self.free.steer(fs, want, STOP_M + 0.2)
            if alt is not None and not (lo <= alt <= hi):
                alt = None
        if alt is None:
            # nothing clear within the allowed swing: back off again (bounded)
            self._escapes += 1
            if self._escapes > MAX_ESCAPES:
                return self._boxed(obs)
            self._escape_phase = "reverse"
            self._escape_until = obs.now + ESCAPE_REVERSE_S
            self._say(f"[obstacle] nothing clear within +-{ESCAPE_MAX_SWING_DEG:.0f} deg — reversing again")
            return Cmd(-ESCAPE_REVERSE_LIN, 0.0, 0.6, "ESCAPE", "reversing")
        if abs(alt) < 6:
            self.mode = "DRIVE"
            self._move_ref = None
            return Cmd(0.0, 0.0, 0.3, "DRIVE", "escape: clear ahead")
        self._arc_ang = float(np.clip(alt / 45.0, -1.0, 1.0)) * ESCAPE_ARC_ANG
        self._escape_phase = "arc"
        self._escape_until = obs.now + ESCAPE_ARC_S
        self._say(f"[obstacle] arcing forward toward {alt:+.0f} deg ({fs.clear_at(alt):.1f} m clear), "
                  f"ang {self._arc_ang:+.2f}")
        return Cmd(ESCAPE_ARC_LIN, self._arc_ang, 0.6, "ESCAPE", f"arc {alt:+.0f}")

    def _boxed(self, obs: Obs) -> Cmd:
        """Out of escapes: hold still a moment, then start over (bounded)."""
        if self._boxed_since is None:
            self._boxed_since = obs.now
            self._say(f"[obstacle] boxed in — holding {BOXED_WAIT_S:.0f} s, then retrying "
                      f"({self._boxed_cycles + 1}/{BOXED_RETRIES})")
        if obs.now - self._boxed_since < BOXED_WAIT_S or self._boxed_cycles >= BOXED_RETRIES:
            return Cmd(0.0, 0.0, 0.5, "ESCAPE", "boxed in")
        self._boxed_since = None
        self._boxed_cycles += 1
        self._escapes = 0
        self._escape_phase = "stop"
        self._escape_until = obs.now + ESCAPE_SETTLE_S
        return Cmd(0.0, 0.0, 0.3, "ESCAPE", "retrying after boxed-in")

    def _search(self, obs: Obs, est) -> Cmd:
        c = self._pulse_tick(obs)
        if c is not None:
            return c
        if obs.frame is None:
            return Cmd(0.0, 0.0, 0.3, "SEARCH", "no frame")
        cap = self.route.turns[self.turn_k].start if self.turn_k < len(self.route.turns) else len(self.route) - 1
        ref = min(self.route.node_at_p(est.p + LOOKAHEAD_M), cap)
        e = self._heading(features(obs.frame), ref)
        if e is not None and GYRO_HEAD:
            self._gyro_fix(obs, ref, e, self.log.head_src.startswith("sift"))
        if e is not None:
            self._say(f"[nav] heading found again ({self.log.head_src}, err {e:+.0f})")
            self.mode = "DRIVE"
            self._move_ref = None
            return Cmd(0.0, 0.0, 0.3, "DRIVE", "search done")
        if self._search_i >= len(SEARCH_STEPS_DEG):
            return Cmd(0.0, 0.0, 0.5, "SEARCH", "LOST: holding still")
        step = SEARCH_STEPS_DEG[self._search_i]
        self._search_i += 1
        return self._start_pulse(obs, step, f"search {self._search_i}/{len(SEARCH_STEPS_DEG)}")

    # ---- turns ---------------------------------------------------------------
    def _turn_enter(self, obs: Obs, est, t) -> Cmd:
        self.mode = "TURN"
        self._turn_yaw0 = obs.yaw_total
        self._turn_est = 0.0
        self._last_lock = None
        nominal = math.copysign(max(abs(t.angle_deg), TURN_NOMINAL_MIN_DEG), t.angle_deg)
        self._turn_nominal = nominal
        self._turn_pulses = 1
        self._say(f"[turn {self.turn_k + 1}/{len(self.route.turns)}] "
                  f"{'LEFT' if nominal > 0 else 'RIGHT'} ~{abs(nominal):.0f} deg at s={est.s:.1f} m "
                  f"(teach turned at s={t.s:.1f})")
        return self._start_pulse(obs, nominal * TURN_FIRST_FRAC, "turn open-loop")

    def _turn_finish(self, obs: Obs, why: str) -> Cmd:
        t = self.route.turns[self.turn_k]
        self._say(f"[turn {self.turn_k + 1}] done ({why}); commanded ~{self._turn_est:+.0f} deg, "
                  f"spin rate {self.spin_rate:.0f} deg/s")
        self.loc.anchor(float(self.route.p[t.end]) + 0.2, sigma_m=1.5)
        self._band = (float(self.route.p[t.end]), obs.s_total)
        # the teach yaw across a corner is unreliable (old gyro): let the first
        # SIFT fixes after the corner re-anchor the gyro offset without gating
        self._yoff_n = 0
        self._yoff_out = []
        self.turn_k += 1
        self.mode = "DRIVE"
        self._move_ref = None
        self._e_f = self._c_f = None
        self._i_term, self._i_t = 0.0, None
        self._last_ang = 0.0
        return Cmd(0.0, 0.0, 0.3, "DRIVE", f"turn done: {why}")

    def _turn_measure(self, obs: Obs, est) -> Cmd:
        t = self.route.turns[self.turn_k]
        # gyro integrated since the turn began (settled) — reliable over whole
        # turns (+121/+107/-140 on runs 6-8) though not for a single 1 s spin
        turned = obs.yaw_total - self._turn_yaw0
        remaining = self._turn_nominal - turned
        if obs.frame is None:
            return Cmd(0.0, 0.0, 0.3, "TURN", "no frame")
        # post-corner views 0.8 / 2.0 / 3.5 m into the next leg; the first one
        # that gives a heading (SIFT, else DINOv2 patch alignment) is the lock
        lf = features(obs.frame)
        e, ref = None, -1
        for ahead in (POST_TURN_REF_M, 2.0, 3.5):
            n = self.route.node_at_p(float(self.route.p[t.end]) + ahead)
            h = self._heading(lf, n, dino_min=TURN_DINO_MIN)
            if h is not None and abs(h) < 45.0:
                e, ref = h, n
                break
        if e is not None and not self.log.head_src.startswith("sift"):
            implied = turned - e             # total turn this lock says the corner is
            if abs(implied - self._turn_nominal) > TURN_WEAK_LOCK_AGREE_DEG:
                e, ref = None, -1            # weak lock disagrees with gyro + corner angle
        locked = e is not None
        self._turn_pulses += 1
        if locked:
            self._learn_rate(ref, e)
            need = -e
            if abs(need) <= TURN_TOL_DEG:
                return self._turn_finish(obs, f"aligned to teach view {ref} ({self.log.head_src}, err {e:+.1f})")
            if self._turn_pulses > TURN_MAX_PULSES:
                return self._turn_finish(obs, "pulse budget spent (visually close)")
            return self._start_pulse(obs, need * TURN_AIM, f"visual {need:+.0f} {self.log.head_src}")
        # no visual lock: turn by the gyro to the nominal corner angle, then
        # hand over to DRIVE (corridor / free-space / teach heading) rather
        # than searching past it
        if self._turn_pulses > TURN_MAX_PULSES:
            return self._turn_finish(obs, "pulse budget spent (no visual lock)")
        sign = math.copysign(1.0, self._turn_nominal)
        if remaining * sign > 10.0:
            return self._start_pulse(obs, remaining * TURN_AIM, f"gyro {turned:+.0f}/{self._turn_nominal:+.0f}")
        if remaining * sign < -15.0:
            return self._start_pulse(obs, remaining * TURN_AIM, f"gyro overturn {turned:+.0f}, back {remaining:+.0f}")
        return self._turn_finish(obs, f"no visual lock; gyro says {turned:+.0f} of {self._turn_nominal:+.0f}")
