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
V_NOM = _env("V_NOM", 0.45)            # linear cmd; live probe: 0.55 peaks at 0.69 m/s
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
TURN_NOMINAL_MIN_DEG = _env("TURN_NOMINAL_MIN_DEG", 75.0)   # every corner is ~90 deg
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
CORRIDOR_MAX_DEV = _env("CORRIDOR_MAX_DEV", 35.0)  # far direction must be this close to teach heading

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

    def _bearing_to(self, live_feat, node: int) -> Bearing:
        """Day and night picture of the node; the one with more agreeing matches
        wins. (Live run 1: at night the day view sometimes passed on ~20
        ceiling-light matches and disagreed with the night view by ~12 deg.)"""
        b = bearing(live_feat, self.route.features(node))
        self.log.ref_src = "day"
        tf = self.route.twin_features(node)
        if tf is not None:
            bt = bearing(live_feat, tf)
            if bt.n_inliers > b.n_inliers:
                b = bt
                self.log.ref_src = "night"
        self.log.ref = node
        self.log.yaw_err = b.yaw_err_deg if b.ok else None
        self.log.inliers = b.n_inliers
        return b

    def _ref_tokens(self, node: int, night: bool):
        key = (int(node), night)
        if key not in self._tok_cache:
            img = self.route.twin_image(node) if night else self.route.image(node)
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
        for night in (False, True):
            rt = self._ref_tokens(node, night)
            if rt is None:
                continue
            e, m = dino_shift(self._live_tok, rt)
            if best is None or m > best[1]:
                best = (e, m, night)
        if best is not None and best[1] >= dino_min:
            self.log.head_src = f"dino-{'night' if best[2] else 'day'}({best[1]:.2f})"
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
        dp = self._odom_progress(obs)
        self._prev = obs
        self.loc.predict(dp)

        in_burst = self._pulse is not None and obs.now < self._pulse.burst_end
        live_feat = None
        self._live_tok = None
        if obs.frame is not None and not in_burst:
            desc, self._live_tok = self.enc.encode_full(obs.frame)
            self.loc.update(desc)
        if self.mode != "INIT":
            self.loc.constrain(*self._window())
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
        e = self._heading(features(obs.frame), ref)

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

        want = -e if e is not None else 0.0          # deg to turn, left-positive
        corridor = None
        if fs is not None:
            far = float(fs.clear_m.max())
            n_open = int((fs.clear_m >= CORRIDOR_FAR_M).sum())
            if far >= CORRIDOR_FAR_M and 0 < n_open <= CORRIDOR_MAX_OPEN_BINS:
                cand = fs.bins_deg[fs.clear_m >= 0.9 * far]
                cand = cand[np.abs(cand - want) <= CORRIDOR_MAX_DEV]
                if len(cand):
                    corridor = float(np.median(cand))
        if corridor is not None:
            want = corridor
        self.log.corridor_deg = corridor
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
            if self._i_t is not None:
                dt = min(obs.now - self._i_t, 0.5)
                self._i_term = float(np.clip(self._i_term - KI_DEG * self._e_f * dt, -I_MAX, I_MAX))
            self._i_t = obs.now
            ang = float(np.clip(-KP_DEG * self._e_f + self._i_term, -MAX_ANG, MAX_ANG))
        else:
            ang = 0.5 * self._last_ang

        lin = max(LIN_MIN, V_NOM * float(np.clip(1.0 - abs(want) / 45.0, 0.4, 1.0)))
        if est.spread_m > 3.0 or (e is None and corridor is None) or avoiding:
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
        self._escapes = 0 if (fs is None or fs.ahead_m > NEED_M) else self._escapes
        note = (f"ref {ref} err {e:+.1f} {self.log.head_src}" if e is not None else f"ref {ref} no-heading x{self._blind}")
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
        self._e_f = self._c_f = None
        self._i_term, self._i_t = 0.0, None
        self._last_ang = 0.0
        self._say(f"[obstacle] {why} — stop (escape #{self._escapes})")
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

        if fs.ahead_m >= NEED_M and abs(want) < 10:
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

        # 2) after reversing: arc forward toward the clear heading nearest the route
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
        e = self._heading(features(obs.frame), min(self.route.node_at_p(est.p + LOOKAHEAD_M), cap))
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
