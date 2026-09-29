"""
Main control loop. This is the file you actually execute.

    STAGE=0 python -m erc_urban.run   # classical dirt-path + GPS creep
    STAGE=1 python -m erc_urban.run   # LogoNav + full perception stack (default)
    STAGE=2 python -m erc_urban.run   # + pose buffer
    STAGE=3 python -m erc_urban.run   # + custom CKPT planner + VLM fusion

Mission mode (default): start mission -> drive checkpoints -> /checkpoint-reached.
Free-drive: USE_MISSION=0 GOAL_LAT=.. GOAL_LON=.. python -m erc_urban.run

Older earth-rovers-sdk builds have no /status or /missions; neither is required.
"""
from __future__ import annotations
import math
import os
import signal
import time
import numpy as np

from .sdk_client import RoverClient
from .streamer import CommandStreamer, LinkWatchdog
from .sdk_guard import SdkGuard
from .interfaces import Observation
from .geo import geodesic_bearing_range, reached
from .modules.baseline import (
    BaselinePlanner, PurePursuitController, VillageGotoController,
)
from .modules.dirt_path import DirtPathGate, analyze, rescore
from .modules.logonav import LogoNavPlanner
from .modules.obstacle import analyze_obstacle, front_blocked_rear_clear
from .modules.depth_gate import DepthObstacleGate
from .modules.pivot import PulsePivot
from .modules.costmap import DepthCostmap, CostmapConfig
from .modules.semantics import SemanticTraversability
from .calib import CameraCalibration
from .modules.candidates import MultiHypothesisPlanner
from .modules.depth_scale import DepthResponseFit
from .modules.route import RouteAligner, RouteAim
from .modules.route_profile import build_route_profile
from .modules.traversability import TraversabilityVeto, SegFormerSource
from .modules.scoring import path_traversable_fractions
from .modules.odometry import VisualOdometry, Breadcrumbs
from .modules import osm_route
from .modules.wheel_odom import WheelOdometry, detect_yaw_axis, parse_gyro_axis
from erc_indoor.freespace import path_clear
from .modules.obstacle_memory import ObstacleMemory, choose_full, plan_heading
from .modules.segment_commit import SegmentCommit
from .modules.claim import ApproachClaim
from .modules.heading import HeadingFilter, load_state, save_state
from .modules.freespace import (
    FreeSpaceUrban, GRASS_NAMES, GROUND_NAMES, NEVER_LOW_NAMES, OFFROAD_OBSTACLE_NAMES,
    class_ids, open_side,
    unlabel_impossible,
)
from .modules.commit import CommitWindow
from .modules.localmap import LocalMap
from .modules.topograph import TopoGraph, BLOCKED as TOPO_BLOCKED
from .modules.explore import GapExplorer
from .modules.selector import (
    VLMSelector, GeminiVertexVLM, dump_candidates, render_overlay,
)
from .modules.upgrades import LearnedPlanner, VLMSelectorFusion, PoseBuffer

STAGE = int(os.environ.get("STAGE", "1"))
# Local planner.
#   fan:   free space per heading from where drivable ground ends (the
#          erc_indoor method; modules/freespace.py). Default since Wuhan
#          mission-3, where the K-candidate fan crashed into a planter kerb.
#   cands: the original K-candidate fan (LogoNav + arcs + vetoes + escapes).
NAV = os.environ.get("NAV", "fan").strip().lower()
CTRL_HZ = 5.0
K = 12 if STAGE < 3 else 18
USE_MISSION = os.environ.get("USE_MISSION", "1") != "0"
PATH_BLEND = float(os.environ.get("PATH_BLEND", "0.65"))
USE_DIRT_GATE = (
    True if STAGE == 0
    else os.environ.get("USE_DIRT_GATE", "0") == "1"
)
# Classical near-field contact (lamp-on-wall / dark blob).
USE_OBS_STOP = os.environ.get("USE_OBS_STOP", "1") != "0"
# Metric depth clearance. Falls back silently if the checkpoint is missing.
# Under NAV=fan depth has ONE job: the overhang check in modules/freespace.py
# (the ground boundary misses things you can see ground under, like a parked
# e-scooter). The depth costmap it used to feed stays off (USE_COSTMAP).
USE_DEPTH = os.environ.get("USE_DEPTH", "1") != "0"
LAMP = int(os.environ.get("LAMP", "1"))
# Only STOP to pivot for large errors: a stop-and-settle cycle costs ~3 s, so
# anything steerable while rolling should be steered.
ALIGN_DEG = float(os.environ.get("ALIGN_DEG", "65"))
# In-place pivots need far more throttle than arc-following — with lin=0 the
# skid-steer must break stiction on both tracks, and 0.16 does nothing. A flat
# 0.6 held for a whole slow tick overshoots >100 deg, so the pivot is
# proportional between the stiction floor and this ceiling.
ALIGN_ANG = float(os.environ.get("ALIGN_ANG", "0.60"))
ALIGN_MIN = float(os.environ.get("ALIGN_MIN", "0.30"))
ALIGN_KP = float(os.environ.get("ALIGN_KP", "0.005"))
# Telemetry is 0.8-1.4 s stale and refreshes at 2 Hz, so hold still at least
# this long after a pivot burst before believing the next heading.
SETTLE_S = float(os.environ.get("SETTLE_S", "1.8"))
# Rolling heading correction, in yaw-rate terms (see _steer_nudge).
STEER_KP = float(os.environ.get("STEER_KP", "0.30"))        # deg/s per deg err
STEER_MAX_DPS = float(os.environ.get("STEER_MAX_DPS", "10"))
STEER_MAX_ANG = float(os.environ.get("STEER_MAX_ANG", "0.15"))
# Commands streamed at STREAM_HZ (streamer.py) vs posted once per tick. Defined
# here because the steering constants below depend on it: see STEER_RATE_PRIOR.
STREAM_CMDS = os.environ.get("STREAM_CMDS", "1") != "0"
# deg/s per unit angular, until a pivot burst measures the real value.
#   * posted once per tick: measured 15 (not the 90 first assumed); below ~0.7
#     stiction wins and it does not yaw at all.
#   * streamed: measured 37 while driving (Wuhan, 2026-09-23: yaw vs commanded
#     angular over 154 ticks, peak correlation at ~2.7 s lag). Keeping 15 there
#     made the loop ~2.5x too hot: heading oscillated with a 10.6 s period
#     (~4x the loop delay) and 11.7 deg std; 37 with LOOKAHEAD_M 2.0 gave 6.1.
# Overestimating shrinks the steering gain, so the rover under-corrects and
# drifts until it trips the pivot latch.
STEER_RATE_PRIOR = float(os.environ.get("STEER_RATE_PRIOR", "37" if STREAM_CMDS else "15"))
# Hysteresis: enter align at ALIGN_DEG, stay in it until well inside the band.
ALIGN_EXIT = float(os.environ.get("ALIGN_EXIT", "22"))
# Forward speed. v_nom 0.20 with max_angular 0.14 pinned LogoNav to the
# turn_scale floor at 0.08 throttle, i.e. barely moving.
V_NOM = float(os.environ.get("V_NOM", "0.34"))
MAX_ANG = float(os.environ.get("MAX_ANG", "0.26"))
# Pure-pursuit lookahead. Its heading gain is ~2v/L per second; with ~2.5 s of
# loop delay and continuous (streamed) motion, 0.8 m oscillates. See
# STEER_RATE_PRIOR for the measurement.
LOOKAHEAD_M = float(os.environ.get("LOOKAHEAD_M", "2.0" if STREAM_CMDS else "0.8"))
# Progress watchdog: metres of range that must be closed within this window.
FREEZE_SEC = float(os.environ.get("FREEZE_SEC", "45"))
FREEZE_M = float(os.environ.get("FREEZE_M", "1.5"))


def _auto_device() -> str:
    """Apple Silicon GPU when present — 'auto' upstream only checks CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return "mps"
    except Exception:
        pass
    return "cpu"

# Ground-relative obstacle costmap and its candidate veto.
USE_COSTMAP = os.environ.get("USE_COSTMAP", "0" if NAV == "fan" else "1") != "0"
# Telemetry is ~1.1 s stale and commands take ~0.3 s to land, so the rover keeps
# moving for over a second after a decision. 0.55 m is not a stopping distance,
# it is a collision report.
COSTMAP_STOP_M = float(os.environ.get("COSTMAP_STOP_M", "1.2"))
# 1.8, not 2.5: depth saturates on this camera (hypersim caps at 19.6 m), so an
# empty courtyard measures 2.55 m and a 2.5 m threshold has the rover creeping
# everywhere. 1.8 clears that floor and still leaves 0.6 m of deceleration.
COSTMAP_SLOW_M = float(os.environ.get("COSTMAP_SLOW_M", "1.8"))
# Depth reading below which nothing else gets a vote.
DEPTH_HARD_STOP_M = float(os.environ.get("DEPTH_HARD_STOP_M", "0.45"))
OBST_HEIGHT_M = float(os.environ.get("OBST_HEIGHT_M", "0.18"))
ROVER_RADIUS_M = float(os.environ.get("ROVER_RADIUS_M", "0.35"))

# Semantic traversability: dirt track vs grass verge, plus the hazards geometry
# cannot see (water, people).
USE_SEMANTICS = os.environ.get("USE_SEMANTICS", "1") != "0"
OFFTRACK_PENALTY = float(os.environ.get("OFFTRACK_PENALTY", "0.6"))
# Below this fraction of traversable surface in the near band there is no path
# ahead. Monocular depth CANNOT see a grass wall — it reported 2.4 m with blades
# against the lens — so this is the only signal that stops the rover burying
# itself.
TRACK_MIN = float(os.environ.get("TRACK_MIN", "0.04"))
# Corridor following. On a single-track path the corridor is not a hint to blend
# into a goal-seeking controller, it IS the control law: LogoNav is
# goal-conditioned and will steer off a 1 m path toward a GPS point 34 m away,
# which is how the rover buried itself.
CORRIDOR_MIN = float(os.environ.get("CORRIDOR_MIN", "0.12"))
CORRIDOR_MAX_ANG = float(os.environ.get("CORRIDOR_MAX_ANG", "0.35"))
CORRIDOR_GOAL_BIAS = float(os.environ.get("CORRIDOR_GOAL_BIAS", "0.20"))
CORRIDOR_BIAS_MAX_DEG = float(os.environ.get("CORRIDOR_BIAS_MAX_DEG", "8"))
# Only turn around if the goal is genuinely behind us; otherwise follow the
# path, even when it points away from the checkpoint.
CORRIDOR_ALIGN_DEG = float(os.environ.get("CORRIDOR_ALIGN_DEG", "120"))
# Return leg: one open-loop spin, then drive. Do not servo live GPS/compass —
# both flip while yawing (+157 -> -179 -> +112, then STUCK after ~80 ticks).
TURNAROUND_DPS = float(os.environ.get("TURNAROUND_DPS", "25"))
TURNAROUND_BURST_CAP = int(os.environ.get("TURNAROUND_BURST_CAP", "4"))
# The post-turnaround hold ("follow the corridor, ignore a goal that still reads
# behind") only covers compass/GPS lag right after the spin. Held for the whole
# leg, it trapped mission-3 (Wuhan): an escape swung the rover south again with
# CP2 behind it, the latch forbade turning back, and it drove away along
# corridor(no-gps) into a parked rover. Released after this much driving, and
# on every escape.
TURNAROUND_HOLD_M = float(os.environ.get("TURNAROUND_HOLD_M", "4.0"))
# Consecutive "every candidate untraversable" ticks before that becomes an
# escape. The first one holds still and re-perceives.
TRAV_ALL_TICKS = int(os.environ.get("TRAV_ALL_TICKS", "2"))
# Recovery scan: rotate in steps looking for a drivable heading.
SCAN_STEPS = int(os.environ.get("SCAN_STEPS", "7"))
SCAN_STEP_DEG = float(os.environ.get("SCAN_STEP_DEG", "50"))
SCAN_RESUME = float(os.environ.get("SCAN_RESUME", "0.35"))
# Mission-route guard: on the Rware run the route prior was correct but noisy
# geometric vetoes kept launching exploratory turns while semantics still saw an
# open paved sidewalk. A strong visible sidewalk suppresses those detours; real
# close obstacles still stop through the hard floors below.
ROUTE_SIDEWALK_MIN_DRIVE = float(os.environ.get("ROUTE_SIDEWALK_MIN_DRIVE", "0.75"))
ROUTE_SIDEWALK_MIN_TRACK = float(os.environ.get("ROUTE_SIDEWALK_MIN_TRACK", "0.35"))
ROUTE_SIDEWALK_MAX_HARD = float(os.environ.get("ROUTE_SIDEWALK_MAX_HARD", "0.03"))
ROUTE_COSTMAP_HARD_STOP_M = float(os.environ.get("ROUTE_COSTMAP_HARD_STOP_M", "0.75"))
ROUTE_DEPTH_HARD_STOP_M = float(os.environ.get("ROUTE_DEPTH_HARD_STOP_M", "0.25"))
ROUTE_TURNAROUND_DEG = float(os.environ.get("ROUTE_TURNAROUND_DEG", "90"))

# Measured hardware geometry: fx/FOV from closing a 360 deg spin, camera height
# from the MINI+ dimension drawing. Everything geometric reads from here rather
# than re-deriving a 90 deg guess.
CALIB = CameraCalibration.load_or_default(
    os.path.join(os.path.dirname(__file__), "camera_front.json"), 1024, 576
)
USE_MULTI = os.environ.get("USE_MULTI", "1") != "0"
USE_TRAV_VETO = os.environ.get("USE_TRAV_VETO", "1") != "0"
USE_DEPTH_FIT = os.environ.get("USE_DEPTH_FIT", "1") != "0"
# Depth is trustworthy to ~3.5 m on this camera: hypersim saturates at 19.6 m,
# so anything beyond maps into the near field and no remap recovers it.
DEPTH_MAX_M = float(os.environ.get("DEPTH_MAX_M", "3.5"))

_DEPTH_GATE: DepthObstacleGate | None = None
_COSTMAP: DepthCostmap | None = None
_SEM: SemanticTraversability | None = None
_SEM_TRIED = False
_DEPTH_FIT: DepthResponseFit | None = None
_ROUTE: RouteAligner | None = None
_TRAV: TraversabilityVeto | None = None

# Odometry, the breadcrumb trail, and the commit window.
USE_ODOM = os.environ.get("USE_ODOM", "1") != "0"
USE_COMMIT = os.environ.get("USE_COMMIT", "1") != "0"
# "wheel": rpm + gyro from /data (erc_indoor's Odometer, proven on the Mini+),
# falling back to VO by itself if this unit's rpms never report travel.
# "vo": ground-texture visual odometry only.
ODOM_SOURCE = os.environ.get("ODOM_SOURCE", "wheel").strip().lower()
WHEEL_M_PER_REV = float(os.environ.get("WHEEL_M_PER_REV", "0.388"))
# Which /data gyro axis is yaw: "auto" (short left/right wiggle at startup),
# or "x" / "y" / "z", "-" prefix to flip. Mini+ units so far: z. Wuhan
# inert-eel-code: x (z stayed flat through a 25 deg spin).
GYRO_AXIS = os.environ.get("GYRO_AXIS", "auto")
_VO: VisualOdometry | None = None
_WHEEL: WheelOdometry | None = None
_CRUMBS = Breadcrumbs()
_COMMIT = CommitWindow()

# Spatial memory: the map remembers where things ARE, the topological graph
# remembers what has been TRIED per place in world bearings, and the explorer
# turns both into a committed way around an obstacle. Set either to 0 for
# single-frame behaviour.
USE_LOCALMAP = os.environ.get("USE_LOCALMAP", "1") != "0"
USE_EXPLORE = os.environ.get("USE_EXPLORE", "1") != "0"
# 36 bins is 10-degree resolution over the full circle — finer than the compass
# is stable to, but cheap, and the gap extractor merges bins anyway.
SCAN_BINS = int(os.environ.get("SCAN_BINS", "36"))
SCAN_RANGE_M = float(os.environ.get("SCAN_RANGE_M", "10.0"))
# Numbered candidate overlays, so a live run is inspectable.
CAND_VIZ_DIR = os.environ.get("CAND_VIZ_DIR", "/tmp/erc_cands")
CAND_VIZ_EVERY = int(os.environ.get("CAND_VIZ_EVERY", "1"))
_LOCALMAP: LocalMap | None = None
_TOPO: TopoGraph | None = None
_EXPLORER: GapExplorer | None = None


class GpsHealth:
    """
    Decide whether the GPS-derived goal bearing may steer the rover.

    Measured: lat/lon are bit-identical across 24 stationary samples AND across
    30 s of driving while the IMU updates normally. A frozen fix is not a noisy
    measurement of the right answer, it is not a measurement at all.

    Visual odometry is what makes this decidable — if we have demonstrably moved
    and the fix has not, the fix is stale. Without VO you cannot tell "frozen
    GPS" from "stationary rover".
    """

    def __init__(self, min_move_m: float = 0.35, stale_after_s: float = 20.0):
        self.min_move_m = min_move_m
        self.stale_after_s = stale_after_s
        self.last_fix = None
        self.last_change_t = None
        self.moved_since_change = 0.0
        self.stale = False

    def update(self, lat, lon, odom, now: float) -> bool:
        if np.isnan(lat) or np.isnan(lon):
            return self.stale
        fix = (round(float(lat), 7), round(float(lon), 7))
        if self.last_fix is None:
            self.last_fix, self.last_change_t = fix, now
            return self.stale
        if fix != self.last_fix:
            self.last_fix, self.last_change_t = fix, now
            self.moved_since_change = 0.0
            self.stale = False
            return self.stale
        if odom is not None and odom.confidence > 0.0:
            self.moved_since_change += odom.distance_m
        held = now - (self.last_change_t or now)
        # Both: moved far enough that the fix SHOULD have changed, and enough
        # wall time that this is not merely a slow update rate.
        self.stale = (
            self.moved_since_change >= self.min_move_m and held >= self.stale_after_s
        )
        return self.stale

    def describe(self) -> str:
        return (f"gps={'STALE' if self.stale else 'ok'}"
                f"({self.moved_since_change:.2f}m since fix change)")


_GPS = GpsHealth()


# VLM candidate selection. OFF unless USE_VLM=1, because it bills a cloud API
# per query. Runs on its own thread: 1.6 s median, and the rover's command
# watchdog means an in-tick call would physically slow it down.
USE_VLM = os.environ.get("USE_VLM", "0") == "1"
_SELECTOR: VLMSelector | None = None


def _selector() -> VLMSelector | None:
    global _SELECTOR
    if not USE_VLM:
        return None
    if _SELECTOR is None:
        vlm = GeminiVertexVLM(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            location=os.environ.get("VERTEX_LOCATION", "global"),
            model=os.environ.get("VLM_MODEL", "gemini-3.5-flash-lite"),
        )
        _SELECTOR = VLMSelector(
            CALIB, vlm_call=vlm,
            query_hz=float(os.environ.get("VLM_HZ", "1.0")),
        )
    return _SELECTOR


def _vo() -> VisualOdometry | None:
    global _VO
    if not USE_ODOM:
        return None
    if _VO is None:
        _VO = VisualOdometry(CALIB)
    return _VO


def _odom_update(front):
    """This tick's motion: wheel odometry when live, else visual odometry."""
    w = _WHEEL
    if w is not None and not w.rpm_dead:
        d = w.update(commanding_forward=_STREAMER.forward if _STREAMER else False)
        if not w.rpm_dead:
            return d
        print("[odom] wheels never reported travel under a forward command — "
              "this unit's rpms are not live; switching to visual odometry",
              flush=True)
    vo = _vo()
    return vo.update(front) if vo is not None else None


def _start_wheel_odom(client) -> None:
    """Start the rpm+gyro thread and calibrate gyro bias. Rover must be still."""
    global _WHEEL
    if not USE_ODOM or ODOM_SOURCE != "wheel":
        print(f"odometry: {'visual' if USE_ODOM else 'off'} (ODOM_SOURCE={ODOM_SOURCE})")
        return
    from erc_indoor.odometry import Odometer

    odo = Odometer(client, m_per_rev=WHEEL_M_PER_REV)
    odo.start()
    t0 = time.monotonic()
    while not odo.ready and time.monotonic() - t0 < 10.0:
        time.sleep(0.2)
    if not odo.ready:
        odo.stop()
        print("odometry: no rpm/gyro samples in /data after 10 s — using visual odometry")
        return
    fixed = parse_gyro_axis(GYRO_AXIS)
    if fixed is not None:
        odo.set_yaw_axis(*fixed)
        how = "set by GYRO_AXIS"
    elif SEND_CONTROL:
        def _compass():
            try:
                y = client.get_data().yaw
                return None if y is None else float(np.degrees(y)) % 360.0
            except Exception:
                return None
        for attempt in range(3):
            # Right after start-mission the link can deliver only a burst or
            # two (Machakos taper-pivot-valve: 5+5 samples, inconclusive, so
            # the compass check never ran and a stuck compass took over).
            w = detect_yaw_axis(odo, lambda lin, ang, hold: (
                _cmd(client, lin, ang, hold) if hold else _halt(client)), read_compass=_compass)
            if w.axis is not None and w.compass_err is not None:
                break
            print(f"[gyro] wiggle {attempt + 1}: {w.note} — retrying", flush=True)
            time.sleep(1.0)
        res, note = w.axis, w.note
        if HEADING_K_COMPASS == "auto" and w.compass_err is not None:
            if w.compass_err > COMPASS_BAD_DEG:
                _HEADING.k_compass = 0.0
            print(f"[compass] off by {w.compass_err:.0f} deg over the wiggle -> "
                  f"{'IGNORED (gyro + GPS track only)' if _HEADING.k_compass <= 0 else 'used'}"
                  f" (k_compass={_HEADING.k_compass})", flush=True)
        if res is None:
            how = f"wiggle inconclusive ({note}) -> default"
            print(f"[gyro] WARNING yaw axis not detected ({note}); keeping "
                  f"{'-' if odo.yaw_sign < 0 else ''}{'xyz'[odo.yaw_axis]}. "
                  f"Set GYRO_AXIS if turns look wrong.", flush=True)
        else:
            odo.set_yaw_axis(res[0], res[1])
            how = f"detected by wiggle, left-right {res[2]:+.0f} deg, {note}"
    else:
        how = "default (dry run, no wiggle)"
    bias = odo.calibrate_bias(3.0)
    global _GRAVITY
    try:
        acc = client.get_data().raw.get("accels") or []
        g = np.array([x[:3] for x in acc if isinstance(x, list) and len(x) >= 3], float).mean(0)
        if np.linalg.norm(g) > 0.5:
            _GRAVITY = _snap_gravity(g)
            print(f"[tilt] gravity at rest {np.round(_GRAVITY, 2).tolist()}, guard {FAN_TILT_DEG:.0f} deg",
                  flush=True)
    except Exception:
        pass
    _WHEEL = WheelOdometry(odo)
    axis = f"{'-' if odo.yaw_sign < 0 else ''}{'xyz'[odo.yaw_axis]}"
    print(f"odometry: wheel+gyro ({WHEEL_M_PER_REV} m/rev, yaw = gyro {axis} [{how}], "
          f"gyro bias {bias:+.3f} deg/s)", flush=True)


def _depth_fit() -> DepthResponseFit | None:
    global _DEPTH_FIT
    if not USE_DEPTH_FIT:
        return None
    if _DEPTH_FIT is None:
        _DEPTH_FIT = DepthResponseFit(CALIB)
    return _DEPTH_FIT


def _route() -> RouteAligner:
    global _ROUTE
    if _ROUTE is None:
        _ROUTE = RouteAligner(
            corridor_min_width=CORRIDOR_MIN,
            goal_bias_max_deg=CORRIDOR_BIAS_MAX_DEG,
            goal_bias_gain=CORRIDOR_GOAL_BIAS,
            track_min_frac=TRACK_MIN,
            turnaround_deg=CORRIDOR_ALIGN_DEG,
        )
    return _ROUTE


def _trav_veto() -> TraversabilityVeto | None:
    global _TRAV
    if not USE_TRAV_VETO:
        return None
    if _TRAV is None:
        sem = _semantics()
        if sem is None:
            return None
        _TRAV = TraversabilityVeto(source=SegFormerSource(semantics=sem), calib=CALIB)
    return _TRAV


def _semantics() -> SemanticTraversability | None:
    global _SEM, _SEM_TRIED
    if not USE_SEMANTICS:
        return None
    if not _SEM_TRIED:
        _SEM_TRIED = True
        s = SemanticTraversability(
            device=os.environ.get("SEG_DEVICE", "auto"),
            size=int(os.environ.get("SEG_SIZE", "512")),
        )
        _SEM = s if s.ready else None
    return _SEM


def _costmap() -> DepthCostmap | None:
    global _COSTMAP
    if not USE_COSTMAP:
        return None
    if _COSTMAP is None:
        _COSTMAP = DepthCostmap(
            CostmapConfig(
                obstacle_height_m=OBST_HEIGHT_M,
                rover_radius_m=ROVER_RADIUS_M,
                fov_horizontal_deg=CALIB.fov_horizontal_deg,
                max_range_m=DEPTH_MAX_M,
            )
        )
    return _COSTMAP


def _depth_gate() -> DepthObstacleGate | None:
    global _DEPTH_GATE
    if not USE_DEPTH:
        return None
    if _DEPTH_GATE is None:
        _DEPTH_GATE = DepthObstacleGate(
            model_size=os.environ.get("DEPTH_MODEL", "small"),
            device=os.environ.get("DEPTH_DEVICE", _auto_device()),
            every_n=int(os.environ.get("DEPTH_EVERY_N", "2")),
            checkpoint_domain=os.environ.get("DEPTH_DOMAIN", "indoor"),
        )
        _DEPTH_GATE.response_fit = _depth_fit()
    return _DEPTH_GATE if _DEPTH_GATE.ready else None


def _map() -> LocalMap | None:
    """The accumulated world-frame occupancy map."""
    global _LOCALMAP
    if not USE_LOCALMAP:
        return None
    if _LOCALMAP is None:
        _LOCALMAP = LocalMap(
            cell_m=float(os.environ.get("MAP_CELL_M", "0.15")),
            extent_m=float(os.environ.get("MAP_EXTENT_M", "24")),
            rover_radius_m=ROVER_RADIUS_M,
        )
    return _LOCALMAP


def _topo() -> TopoGraph | None:
    """What we have TRIED, per place, in world bearings."""
    global _TOPO
    if not USE_EXPLORE:
        return None
    if _TOPO is None:
        _TOPO = TopoGraph(
            node_spacing_m=float(os.environ.get("NODE_SPACING_M", "1.5")),
        )
    return _TOPO


def _explorer() -> GapExplorer | None:
    """Gap selection and committed circumnavigation."""
    global _EXPLORER
    if not USE_EXPLORE:
        return None
    if _EXPLORER is None:
        _EXPLORER = GapExplorer(
            rover_width_m=2.0 * ROVER_RADIUS_M,
            commit_budget_m=float(os.environ.get("DETOUR_BUDGET_M", "4.0")),
        )
    return _EXPLORER


def _note_blocked(reason: str = "") -> None:
    """
    Record that the direction we are FACING did not work here.

    World-bearing, so the memory survives the rover turning away from it, which
    is exactly what happens next.
    """
    tg, lm = _topo(), _map()
    if tg is None or lm is None or tg.current is None:
        return
    tg.mark(lm.theta, TOPO_BLOCKED)


# Command streaming (streamer.py). The loop SETS a command with an expiry and a
# thread re-sends it at STREAM_HZ, so a slow tick neither starves the rover's
# command watchdog nor leaves the last command running if the loop hangs.
# STREAM_CMDS=0 restores the old post-once-per-tick behaviour, which every
# speed and tick-count constant in this file was originally tuned against.
# (STREAM_CMDS itself is defined with the steering constants near the top.)
STREAM_HZ = float(os.environ.get("STREAM_HZ", "10"))
# How long a driving command stays live without a fresh decision. Longer than
# the ~0.7 s tick so motion is continuous; short enough that a hung loop stops.
CMD_HOLD_S = float(os.environ.get("CMD_HOLD_S", "1.0"))
# A blind reverse must not run on between ticks: cap each reverse tick.
REVERSE_HOLD_S = float(os.environ.get("REVERSE_HOLD_S", "0.5"))
# 0 = dry run: the whole stack runs and logs, nothing is sent to /control.
SEND_CONTROL = os.environ.get("SEND_CONTROL", "1") != "0"
# Once running, a hung SDK must fail the tick fast (60 s only covers Chrome
# starting on the first request).
SDK_TIMEOUT_S = float(os.environ.get("SDK_TIMEOUT_S", "3.0"))
LINK_STALE_S = float(os.environ.get("LINK_STALE_S", "4.0"))
_STREAMER: CommandStreamer | None = None
_LINK = LinkWatchdog(LINK_STALE_S)
# SDK self-recovery (sdk_guard.py). With neither a camera frame nor telemetry
# for SDK_DEAD_S, restart the SDK if it is dead, re-send /start-mission (keeps
# progress, per the SDK README), and carry on. SDK_MANAGE=0: only wait.
SDK_MANAGE = os.environ.get("SDK_MANAGE", "1") != "0"
SDK_DEAD_S = float(os.environ.get("SDK_DEAD_S", "20"))
SDK_MAX_RESTARTS = int(os.environ.get("SDK_MAX_RESTARTS", "6"))
_GUARD: SdkGuard | None = None
_LAST_ALIVE = time.monotonic()
# Mission sequence number of each remaining goal, aligned with `goals`.
_GOAL_SEQS: list[int] = []
# --- Agent hooks (erc_urban/agent.py). An API model watches _OBS and steers
# through _AIM_OVERRIDE / _AGENT_BACK; every safety layer below still applies.
# Unused (None/empty) when run.py is launched directly.
_OBS: dict = {}                      # latest fan tick: frame, freespace, pose, goal
_AIM_OVERRIDE: dict | None = None    # {"world_deg", "until_m" (odometer mark), "t_until"}
_AGENT_BACK: float = 0.0             # >0: agent asked to back up (once)
_AGENT_TAKEOVER: bool = False        # agent acts during a long escape: drop it
# Agent "push": drive a world heading for a short distance IGNORING the camera
# obstacle stop (leaf litter / tall grass the segmenter calls blocked). Tilt and
# stall guards still apply. {"world_deg", "until_m", "t_until"}
_AGENT_PUSH: dict | None = None
_EVENTS: list = []                   # (monotonic t, text): escapes, claims, tilt
_MISSION_INFO: dict = {}             # {"route": [(lat, lon)...], "seqs": [...]}


# Per-tick planner line: stdout by default; the agent sends it to a file so the
# terminal shows the driver's decisions (TICK_LOG=<path>).
_TICK_LOG = os.environ.get("TICK_LOG", "")
_TICK_FH = None


def _tick_print(line: str) -> None:
    global _TICK_FH
    if not _TICK_LOG:
        print(line, flush=True)
        return
    if _TICK_FH is None:
        _TICK_FH = open(_TICK_LOG, "a", buffering=1)
    _TICK_FH.write(time.strftime("%H:%M:%S ") + line + "\n")


_CLAIM_HOLD = None                   # (rng, t, goal_index) of the last rejected claim
CLAIM_RETRY_S = float(os.environ.get("CLAIM_RETRY_S", "5"))
_EVENT_SEQ = 0                        # events ever raised (the list keeps the last 50)


def _event(text: str) -> None:
    global _EVENT_SEQ
    _EVENT_SEQ += 1
    _EVENTS.append((time.monotonic(), text, _EVENT_SEQ))
    del _EVENTS[:-50]
# Drive in to CLAIM_NEAR_M of each checkpoint before claiming (see claim.py).
_CLAIM = ApproachClaim(
    near_m=float(os.environ.get("CLAIM_NEAR_M", "15.0")),   # user 2026-09-28: claim any CP within 15 m
    stall_s=float(os.environ.get("CLAIM_STALL_S", "45")),
    stall_m=float(os.environ.get("CLAIM_STALL_M", "4")),
    max_s=float(os.environ.get("CLAIM_MAX_S", "90")),
)
# Live annotated view (candidate fan + HUD) as MJPEG on OVERLAY_PORT, served by
# erc_indoor's OverlayServer:  http://<mac>:8090/stream.mjpg  and /latest.jpg
OVERLAY = os.environ.get("OVERLAY", "1") != "0"
OVERLAY_PORT = int(os.environ.get("OVERLAY_PORT", "8090"))
# Save every Nth published overlay to runs/urban/run_*/overlay/ for review.
OVERLAY_SAVE_EVERY = int(os.environ.get("OVERLAY_SAVE_EVERY", "2"))
_OVERLAY_SRV = None
_RUN_DIR: str | None = None
_OVERLAY_N = 0


def _tint_semantics(img, good_m, soft_m, hard_m):
    """SegFormer's verdict on the image: green track, yellow verge, red hazard."""
    import cv2

    h, w = img.shape[:2]
    out = img.astype(np.float32)
    for m, colour, a in ((good_m, (0, 200, 0), 0.28), (soft_m, (230, 200, 0), 0.30),
                         (hard_m, (230, 0, 0), 0.50)):
        if m is None:
            continue
        mm = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        out[mm] = out[mm] * (1.0 - a) + np.array(colour, np.float32) * a
    return out.astype(np.uint8)


def _draw_fan(img, fan, cal):
    """Ground-ends boundary (red), clearance per heading (bars), chosen heading
    (white), aim (cyan) and remembered obstacles (magenta)."""
    import cv2

    fs, chosen, want = fan[:3]
    mem = fan[3] if len(fan) > 3 else None
    h, w = img.shape[:2]
    for bnd, col in ((fs.drive_boundary, (255, 40, 40)), (fs.track_boundary, (255, 160, 0))):
        sx, sy = w / len(bnd), h / FAN_SEG_HW[0]
        pts = np.stack([(np.arange(len(bnd)) + 0.5) * sx, (bnd + 0.5) * sy], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, col, 2, lineType=cv2.LINE_AA)

    def col_of(deg):
        return int(round(cal.cx - cal.fx * np.tan(np.radians(np.clip(deg, -75, 75)))))

    base = h - 4
    for deg, c in zip(fs.drive.bins_deg, fs.drive.clear_m):
        colr = (0, 210, 0) if c >= FAN_NEED_M else ((255, 190, 0) if c >= FAN_STOP_M else (230, 0, 0))
        u = col_of(deg)
        cv2.line(img, (u, base), (u, base - int(12 + 14 * min(c, 5.0))), colr, 5)
    hz = int(cal.horizon_v())
    cv2.line(img, (col_of(want), h - 1), (col_of(want), hz), (0, 230, 255), 2)
    if chosen is not None:
        cv2.line(img, (col_of(chosen), h - 1), (col_of(chosen), hz), (255, 255, 255), 4)
    if mem is not None and len(mem):
        ahead = mem[mem[:, 0] > 0.15]
        if len(ahead):
            uv, ok = cal.project_ground(ahead)
            for (u, v), o in zip(uv, ok):
                if o and 0 <= u < w and 0 <= v < h:
                    cv2.circle(img, (int(u), int(v)), 3, (255, 0, 255), -1)


def _publish_overlay(front, cands=None, chosen_idx=None, lines=(),
                     good_m=None, soft_m=None, hard_m=None,
                     fan=None, ground=None, grass=None) -> None:
    """SegFormer tint + candidate fan or free-space fan + HUD. Never raises."""
    global _OVERLAY_N
    if _OVERLAY_SRV is None or front is None:
        return
    try:
        import cv2

        if ground is not None:
            base = _tint_semantics(front, ground, grass, hard_m)
        else:
            base = _tint_semantics(front, good_m, soft_m, hard_m)
        img = (render_overlay(base, cands, CALIB) if cands
               else np.ascontiguousarray(base))
        h, w = img.shape[:2]
        if fan is not None:
            _draw_fan(img, fan, CALIB.scaled_to(w, h))
        if cands and chosen_idx is not None and 0 <= chosen_idx < len(cands):
            uv, ok = CALIB.scaled_to(w, h).project_ground(
                np.asarray(cands[chosen_idx].waypoints, np.float64))
            pts = [(int(u), int(v)) for (u, v), g in zip(uv, ok)
                   if g and -w < u < 2 * w and -h < v < 2 * h]
            if len(pts) >= 2:
                cv2.polylines(img, [np.array(pts, np.int32)], False,
                              (255, 255, 255), 6, lineType=cv2.LINE_AA)
        band = 22 * max(1, len(lines)) + 8
        img[:band] = (img[:band] * 0.35).astype(np.uint8)
        for k, line in enumerate(lines):
            cv2.putText(img, str(line)[:110], (8, 22 * (k + 1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        img = cv2.resize(img, (768, int(768 * h / w)))
        data = _OVERLAY_SRV.publish(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        _OVERLAY_N += 1
        if _RUN_DIR and OVERLAY_SAVE_EVERY > 0 and _OVERLAY_N % OVERLAY_SAVE_EVERY == 0:
            with open(os.path.join(_RUN_DIR, "overlay", f"{_OVERLAY_N:06d}.jpg"), "wb") as fh:
                fh.write(data)
            # The raw frame too, so a veto or segmentation call can be replayed.
            cv2.imwrite(os.path.join(_RUN_DIR, "frames", f"{_OVERLAY_N:06d}.jpg"),
                        cv2.cvtColor(cv2.resize(front, (512, 288)), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception:
        pass


def _cmd(client: RoverClient, lin: float, ang: float,
         hold_s: float | None = None) -> None:
    if _STREAMER is not None and STREAM_CMDS:
        _STREAMER.set(lin, ang, CMD_HOLD_S if hold_s is None else hold_s)
    elif SEND_CONTROL:
        client.send_cmd(lin, ang, lamp=LAMP)


def _halt(client: RoverClient) -> None:
    if _STREAMER is not None:
        _STREAMER.halt()
    elif SEND_CONTROL:
        client.send_cmd(0.0, 0.0, lamp=LAMP)


def _align_cmd(bearing: float) -> tuple[float, float]:
    """Pivot toward goal; never drive forward while facing the wrong way."""
    err_deg = float(np.degrees(bearing))
    sign = 1.0 if err_deg >= 0.0 else -1.0
    abs_err = abs(err_deg)
    mag = min(ALIGN_ANG, max(ALIGN_MIN, ALIGN_KP * abs_err))
    return 0.0, float(np.clip(sign * mag, -1.0, 1.0))



def _rear_track_frac(rear) -> float | None:
    """
    Fraction of traversable surface behind the rover — the signal that says
    reversing is safe when the front camera is buried in vegetation.

    Needs HAS_REAR_CAMERA=true in the SDK .env; it defaults to False, so
    /v2/screenshot returns no rear_frame at all until that is set.
    """
    if rear is None:
        return None
    sem = _semantics()
    if sem is None:
        return None
    try:
        labels = sem.labels(rear, (120, 160))
        if labels is None:
            return None
        hard, soft, good = sem.categorize(labels)
        # Drivable, matching the forward check: a verge behind us is a fine
        # place to reverse onto.
        _, band, _ = sem.track((good | soft) & ~hard)
        return float(band)
    except Exception:
        return None


def _steer_nudge(stuck_state: dict | None, bearing: float) -> float:
    """
    Heading correction while rolling, as a target YAW RATE converted through the
    rover's measured authority.

    "10 deg/s toward the goal" is portable; "0.08 angular" is not — on a
    high-authority rover that is a circle, on a sluggish one it is nothing.
    """
    pivot = stuck_state.get("pivot_ctl") if stuck_state is not None else None
    if pivot is not None and pivot.calibrated and pivot.ang > 1e-6:
        rate_per_unit = pivot.rate / pivot.ang
    else:
        rate_per_unit = STEER_RATE_PRIOR
    rate_per_unit = max(rate_per_unit, 10.0)
    desired_dps = float(
        np.clip(STEER_KP * np.degrees(bearing), -STEER_MAX_DPS, STEER_MAX_DPS)
    )
    return float(np.clip(desired_dps / rate_per_unit, -STEER_MAX_ANG, STEER_MAX_ANG))


def _wrap180(deg: float) -> float:
    return (float(deg) + 180.0) % 360.0 - 180.0


def _latch_return_turn(
    stuck_state: dict,
    bearing_rad: float,
    retreat_deg: float | None = None,
) -> float:
    """
    Latch a one-shot return-leg yaw and never retarget it from live GPS.

    After a checkpoint the next one is often behind on the same trail, but
    PulsePivot re-reads bearing every idle tick and that number flips sign while
    spinning, so the rover hunts forever. Pick a direction once — the trail just
    driven, else sign of the first GPS reading x 180 deg — and grind it open
    loop.
    """
    remain = stuck_state.get("turnaround_remain")
    if remain is not None:
        return float(remain)
    if retreat_deg is not None and abs(float(retreat_deg)) >= 45.0:
        remain = _wrap180(float(retreat_deg))
        why = "crumbs"
    else:
        remain = 180.0 if float(bearing_rad) >= 0.0 else -180.0
        why = "open-loop 180"
    stuck_state["turnaround_remain"] = remain
    stuck_state["turnaround_bursts"] = 0
    stuck_state["turnaround_seen_cycles"] = 0
    print(f"  turnaround latch {remain:+.0f}° via {why}", flush=True)
    return remain


_TURNAROUND_KEYS = ("turnaround_done", "turnaround_remain", "turnaround_bursts",
                    "turnaround_seen_cycles", "turnaround_done_at")


def _clear_turnaround(stuck_state: dict, why: str) -> None:
    """Forget the one-shot turnaround so a new one can latch."""
    had = bool(stuck_state.get("turnaround_done")) or (
        stuck_state.get("turnaround_remain") is not None)
    for k in _TURNAROUND_KEYS:
        stuck_state.pop(k, None)
    if had:
        print(f"  turnaround latch released ({why})", flush=True)


def _turnaround_expired(stuck_state: dict, travelled_m: float) -> bool:
    """True once the rover has driven TURNAROUND_HOLD_M since the turnaround."""
    if not stuck_state.get("turnaround_done"):
        return False
    at = stuck_state.setdefault("turnaround_done_at", travelled_m)
    return travelled_m - at >= TURNAROUND_HOLD_M


def _credit_return_burst(stuck_state: dict, burst_s: float) -> bool:
    """Subtract an open-loop burst from the latched remain. True when done."""
    remain = float(stuck_state.get("turnaround_remain", 0.0))
    if remain == 0.0 and stuck_state.get("turnaround_done"):
        return True
    turned = TURNAROUND_DPS * max(float(burst_s), 0.35)
    sign = 1.0 if remain >= 0.0 else -1.0
    nxt = remain - sign * turned
    if nxt * remain <= 0.0:
        nxt = 0.0
    stuck_state["turnaround_remain"] = nxt
    stuck_state["turnaround_bursts"] = int(stuck_state.get("turnaround_bursts", 0)) + 1
    done = (
        abs(nxt) <= ALIGN_EXIT
        or stuck_state["turnaround_bursts"] >= TURNAROUND_BURST_CAP
    )
    if done:
        stuck_state["turnaround_done"] = True
        stuck_state["turnaround_remain"] = 0.0
    return done


def _aligning(stuck_state: dict | None, bearing: float) -> bool:
    """
    Hysteretic align latch: enter above ALIGN_DEG, leave only below ALIGN_EXIT.

    Without it a pivot that overshoots (+73 -> -53 deg in one slow tick) lands
    back outside the band with the sign flipped, and the rover oscillates
    forever instead of driving.
    """
    err = abs(float(np.degrees(bearing)))
    if stuck_state is None:
        return err > ALIGN_DEG
    latched = stuck_state.get("align_latch", False)
    if latched:
        latched = err > ALIGN_EXIT
    else:
        latched = err > ALIGN_DEG
    stuck_state["align_latch"] = latched
    return latched


def build_stack():
    """
    STAGE 0: classical dirt-path + village creep
    STAGE 1: LogoNav + pure-pursuit (default)
    STAGE 2: + PoseBuffer
    STAGE 3: + LearnedPlanner / CKPT + VLM fusion
    """
    if STAGE == 0:
        planner = BaselinePlanner()
        controller = VillageGotoController()
        gate = DirtPathGate()
        posebuf = None
        selector = None
    elif STAGE == 1:
        # LogoNav runs on the accelerator: measured under torch 2.11 at mps
        # 13 ms vs cpu 203 ms at batch 1, still 13 ms at K=18, outputs agreeing
        # to 8e-6. Override with LOGONAV_DEVICE if a machine regresses.
        base = LogoNavPlanner(
            device=os.environ.get("LOGONAV_DEVICE", _auto_device()),
        )
        # LogoNav alone emits ONE trajectory, leaving the selector and the
        # vetoes nothing to choose between. MultiHypothesisPlanner keeps it as
        # the scorer and takes coverage from a geometric anchor fan.
        planner = MultiHypothesisPlanner(base=base) if USE_MULTI else base
        controller = PurePursuitController(
            v_nom=V_NOM, 
            max_angular=MAX_ANG, 
            lookahead_m=LOOKAHEAD_M,
        )
        gate = DirtPathGate() if USE_DIRT_GATE else None
        posebuf = None
        selector = None
    else:
        if STAGE >= 3:
            planner = LearnedPlanner(checkpoint_path=os.environ["CKPT"])
            selector = VLMSelectorFusion(k=K)
        else:
            base = LogoNavPlanner(
                device=os.environ.get("LOGONAV_DEVICE", _auto_device()),
            )
            planner = MultiHypothesisPlanner(base=base) if USE_MULTI else base
            selector = None
        controller = PurePursuitController()
        gate = DirtPathGate() if USE_DIRT_GATE else None
        posebuf = PoseBuffer()
    return planner, selector, gate, posebuf, controller


def wait_for_sdk(client: RoverClient, timeout_s: float = 30.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if client.ping():
            st = None
            try:
                st = client.get_status()
            except Exception:
                pass
            if st is None:
                print("SDK HTTP up (no /status on this build — ok)")
            else:
                print(f"SDK ready: {st}")
            return
        print("waiting for SDK on", client.base, "...")
        time.sleep(1.0)
    raise RuntimeError(
        f"SDK not reachable at {client.base}. "
    )


def resolve_goals(
    client: RoverClient,
) -> tuple[list[tuple[float, float]], bool, list[tuple[float, float]], int]:
    if USE_MISSION:
        try:
            if os.environ.get("SKIP_START_MISSION", "0") == "0":
                print("starting mission...")
                started = client.start_mission()
                print("start-mission ok:", started.get("message", started))
            cps, latest = client.get_checkpoints()
            remaining = [c for c in cps if c.sequence > latest]
            route_goals = [(c.lat, c.lon) for c in cps]
            route_start_index = 0
            for j, c in enumerate(cps):
                if c.sequence > latest:
                    route_start_index = j
                    break
            if not remaining and cps:
                print(f"all {len(cps)} checkpoints already scanned (latest={latest})")
                return [], True, route_goals, route_start_index
            if not cps:
                raise RuntimeError(
                    "start-mission succeeded but checkpoints list is empty"
                )
            goals = [(c.lat, c.lon) for c in remaining]
            _GOAL_SEQS[:] = [int(c.sequence) for c in remaining]
            print(f"mission checkpoints: {len(goals)} remaining (latest_scanned={latest})")
            for c in remaining:
                print(f"  seq={c.sequence} id={c.id} ({c.lat}, {c.lon})")
            return goals, True, route_goals, route_start_index
        except Exception as e:
            msg = str(e)
            print(f"mission setup failed: {msg}")
            if "unavailable" in msg.lower():
                raise SystemExit(2) from e
            if "GOAL_LAT" not in os.environ or "GOAL_LON" not in os.environ:
                # The old fallback drove toward (0, 0): a mission that failed
                # to start must stop, not head for the Gulf of Guinea.
                print("no GOAL_LAT/GOAL_LON set — not driving")
                raise SystemExit(2) from e
            print("falling back to GOAL_LAT/GOAL_LON")

    _GOAL_SEQS[:] = []
    goals = [(
        float(os.environ.get("GOAL_LAT", "0.0")),
        float(os.environ.get("GOAL_LON", "0.0")),
    )]
    print(f"free-drive goal: {goals[0]}")
    return goals, False, goals, 0


def wait_for_gps(client: RoverClient, timeout_s: float = 120.0) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        tel = client.get_data()
        if tel.raw:
            print(f"/data keys={list(tel.raw.keys())[:12]}")
        else:
            print("waiting for GPS in /data... (null/empty telemetry)")
            time.sleep(2.0)
            continue
        if not np.isnan(tel.lat) and not np.isnan(tel.lon):
            print(f"GPS ok: lat={tel.lat:.6f} lon={tel.lon:.6f} yaw={tel.yaw}")
            return
        print("waiting for GPS in /data...")
        time.sleep(2.0)
    print("WARNING: proceeding without GPS fix yet")


# Reversing is BLIND: /v2/screenshot returns rear_frame=None even with
# HAS_REAR_CAMERA set, so a 15-tick reverse is metres travelled with no
# perception — it put this rover into a staircase. With no rear view, reverse is
# a short nudge to break contact, never a retreat; the recovery heading comes
# from the in-place scan, which the front camera can see.
REVERSE_TICKS = int(os.environ.get("REVERSE_TICKS", "15"))
REVERSE_TICKS_BLIND = int(os.environ.get("REVERSE_TICKS_BLIND", "4"))
_REAR_SEEN = False


def _wrap_rad(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _reverse_ticks() -> tuple[int, str]:
    if _REAR_SEEN:
        return REVERSE_TICKS, ""
    return REVERSE_TICKS_BLIND, " [BLIND: no rear camera — short nudge only]"


# Prefer turning around to reversing: a 180 deg turn makes a blind manoeuvre a
# sensed one, so the veto and costmap apply to the escape as well as the
# approach. Reversing has neither.
USE_TURNAROUND = os.environ.get("USE_TURNAROUND", "1") != "0"
MAX_ESCAPES = int(os.environ.get("MAX_ESCAPES", "8"))
# Travel since the last escape that counts as "that worked". Without it
# `rev_attempts` is a LIFETIME counter and a long mission that recovers eight
# separate times aborts on the ninth however well it is driving.
ESCAPE_RESET_M = float(os.environ.get("ESCAPE_RESET_M", "2.0"))


def _rough_but_driveable(drive_frac, hard_m) -> bool:
    """Dirt/roots the rover can roll, not a wall — with semantics agreeing."""
    return (
        drive_frac is not None
        and float(drive_frac) >= 0.50
        and (hard_m is None or float(hard_m.mean()) < 0.05)
    )


def _escape_budget_ok(stuck_state: dict, rng: float) -> None:
    """
    Clear the escape budget once the rover has demonstrably got going again.

    `_begin_escape` has no `rng`, so the first tick after each escape seeds the
    range baseline here.
    """
    attempts = stuck_state.get("rev_attempts", 0)
    if attempts <= 0:
        return
    if stuck_state.get("esc_seen") != attempts:
        stuck_state["esc_seen"] = attempts
        stuck_state["esc_rng"] = rng
        return
    since = _CRUMBS.travelled_m - stuck_state.get("esc_travel", 0.0)
    closed = stuck_state.get("esc_rng", rng) - rng
    if since >= ESCAPE_RESET_M or closed >= FREEZE_M:
        stuck_state["rev_attempts"] = 0
        stuck_state.pop("esc_seen", None)
        print(
            f"  escape budget cleared ({since:.1f}m driven, "
            f"{closed:+.1f}m closed since last escape)",
            flush=True,
        )


def _begin_escape(stuck_state: dict, reason: str, key=None) -> None:
    """
    Get out of a stuck state. Best strategy first:

      1. TURN toward where we came from and drive forward, watching — ground we
         have already driven over.
      2. Turn around on the goal bearing anyway, if there is no trail.
      3. Only then a short blind reverse nudge, far enough to break contact and
         no further. A blind 15-tick reverse put this rover into a staircase.
    """
    attempts = stuck_state.get("rev_attempts", 0) + 1
    stuck_state["rev_attempts"] = attempts
    stuck_state["n"] = 0
    stuck_state["key"] = None
    # Baseline for _escape_budget_ok: an escape that led somewhere is not
    # evidence of a trap.
    stuck_state["esc_travel"] = _CRUMBS.travelled_m
    # An escape turns the rover by an arbitrary amount: a goal behind us now
    # is a fact about the new heading, not compass lag from the old spin.
    _clear_turnaround(stuck_state, "escape")

    # Remember that THIS heading failed at THIS place, before turning away from
    # it. Every stop routes through here. Without it each escape is computed
    # from scratch and the rover retries headings it has already failed — which
    # it did eight times, at +116, +113, +134, +130, -160, +112, -165, +101.
    lm, tg = _map(), _topo()
    if tg is not None and lm is not None:
        tg.update(lm.x, lm.y, lm.theta, lm.travelled_m, time.monotonic(),
                  force=True)
        tg.mark(lm.theta, TOPO_BLOCKED)
    ex = _explorer()
    if ex is not None and ex.detour is not None:
        # The committed way round failed too: drop it so the next tick
        # re-decides rather than driving a detour that has already failed.
        if tg is not None:
            tg.mark(ex.detour.bearing, TOPO_BLOCKED)
        ex.detour = None
    # Escapes turn along the trail rather than reversing blind, so they are
    # cheap and a low cap only aborts runs that would have recovered — three
    # attempts once burned in ~1 s against a phantom obstacle.
    if attempts > MAX_ESCAPES:
        print(
            f"STUCK: {reason} after {attempts - 1} escape attempts "
            f"({key}) — stopping. Needs walker assist.",
            flush=True,
        )
        raise SystemExit(3)

    # "Where is there something untried" beats "which way did I come from". The
    # graph answers in WORLD bearings, so convert to the robot frame first.
    back = None
    src = "the path just driven"
    if tg is not None and lm is not None:
        wb = tg.escape_bearing()
        if wb is not None:
            back = _wrap_rad(wb - lm.theta)
            src = f"toward unexplored ground (topo n={len(tg.nodes)})"
    if back is None and USE_ODOM:
        back = _CRUMBS.retreat_bearing()
    if USE_TURNAROUND and back is not None:
        stuck_state["detour_deg"] = float(np.degrees(back))
        stuck_state.pop("detour_started", None)
        stuck_state["rev_left"] = 0
        stuck_state["pivot_left"] = 0
        print(
            f"STUCK: {reason} — turning {np.degrees(back):+.0f}° "
            f"{src} and driving forward "
            f"(attempt {attempts}/{MAX_ESCAPES})",
            flush=True,
        )
        return

    if USE_TURNAROUND and not _REAR_SEEN:
        # No usable trail: odometry lost, never moved, or the trail has no
        # spatial EXTENT because yawing laid it down rather than driving.
        # Sweeping beats turning toward a bearing derived from odometry noise.
        stuck_state["detour_deg"] = 180.0
        stuck_state.pop("detour_started", None)
        stuck_state["scan_after_reverse"] = True
        stuck_state["scan_left"] = SCAN_STEPS
        stuck_state["scan_best"] = (-1.0, 0.0)
        stuck_state["scan_turned"] = 0.0
        stuck_state["rev_left"] = 0
        print(
            f"STUCK: {reason} — no usable trail; scanning up to "
            f"{SCAN_STEPS}x{SCAN_STEP_DEG:.0f}° for a drivable heading "
            f"(attempt {attempts}/{MAX_ESCAPES})",
            flush=True,
        )
        return

    ticks, note = _reverse_ticks()
    stuck_state["rev_left"] = ticks
    stuck_state["pivot_left"] = 0
    if not _REAR_SEEN:
        stuck_state["scan_after_reverse"] = True
    print(
        f"STUCK: {reason} — reversing {ticks} ticks "
        f"(attempt {attempts}/{MAX_ESCAPES}){note}",
        flush=True,
    )


def _begin_pivot(stuck_state: dict, reason: str) -> None:
    """In-place yaw toward the goal when lighting is uncertain, not a reverse."""
    attempts = stuck_state.get("pivot_attempts", 0) + 1
    stuck_state["pivot_attempts"] = attempts
    stuck_state["n"] = 0
    stuck_state["key"] = None
    stuck_state["pivot_left"] = 12
    stuck_state["rev_left"] = 0
    print(
        f"UNCERTAIN: {reason} — pivoting toward goal ({attempts}/4)",
        flush=True,
    )


# ---- NAV=fan: free space per heading (modules/freespace.py) ----------------
FAN_SEG_HW = (288, 512)
FAN_NEED_M = float(os.environ.get("FAN_NEED_M", "1.2"))    # a heading counts as clear
FAN_MIN_M = float(os.environ.get("FAN_MIN_M", "0.8"))      # ...or at least this, if nothing is
FAN_STOP_M = float(os.environ.get("FAN_STOP_M", "0.6"))    # ground ends this close ahead: escape
FAN_SLOW_M = float(os.environ.get("FAN_SLOW_M", "1.6"))    # start slowing below this
FAN_ALIGN_DEG = float(os.environ.get("FAN_ALIGN_DEG", "35"))    # bigger: stop and pulse-turn
# Straight ahead shorter than this and the clear heading off to one side: turn
# in place first. Wuhan mission-3: 1.3 m ahead, gap at -10 deg; the gentle
# steer (~2 deg/s) plus ~2.5 s of link lag drove it into a parked scooter's
# wheel before the turn happened.
FAN_TIGHT_M = float(os.environ.get("FAN_TIGHT_M", "1.5"))
# Near a checkpoint, face it before driving: at 4 m with the goal 15-34 deg off
# (under FAN_ALIGN_DEG), steering while driving orbited it (Panama mission-2 CP1).
FAN_NEAR_M = float(os.environ.get("FAN_NEAR_M", "6.0"))
FAN_BEHIND_LATCH_DEG = float(os.environ.get("FAN_BEHIND_LATCH_DEG", "135"))
FAN_NEAR_ALIGN_DEG = float(os.environ.get("FAN_NEAR_ALIGN_DEG", "15"))
# Heading loop, same units as STEER_*: deg/s of yaw per deg of error. ~2.5 s of
# loop delay (Wuhan) keeps this well under 1/s.
FAN_KP = float(os.environ.get("FAN_KP", "0.4"))
FAN_MAX_DPS = float(os.environ.get("FAN_MAX_DPS", "20"))
FAN_MAX_ANG = float(os.environ.get("FAN_MAX_ANG", "0.35"))
FAN_TOL_DEG = float(os.environ.get("FAN_TOL_DEG", "8"))
# Forward floor: below this the skid-steer does not move at all. Mission-3,
# CP2->CP3: slow x turn factors took V_NOM 0.24 down to lin 0.08 beside a wall,
# the rover sat still for 6 s and the stuck watchdog fired. (indoor: LIN_MIN.)
FAN_LIN_MIN = float(os.environ.get("FAN_LIN_MIN", "0.18"))
# Optional climbing throttle on a clear, gently steered path. Zero preserves
# the level-ground profile; obstacle approaches and tight turns keep slowing.
FAN_CLEAR_LIN_MIN = float(os.environ.get("FAN_CLEAR_LIN_MIN", "0"))

def _clear_path_throttle(lin, clearance, ahead, angular):
    if min(clearance, ahead) >= FAN_SLOW_M and abs(angular) <= 0.20:
        return max(lin, min(max(FAN_CLEAR_LIN_MIN, 0.0), 0.45))
    return lin
# Depth-based overhang check (modules/freespace.py). 0 = boundary only.
FAN_OVERHANG = os.environ.get("FAN_OVERHANG", "1") != "0"
# Escape, erc_indoor's: stop, back off if close, look, turn toward the most
# open side (never a fixed +50 first: that turned mission-3 into the kerb).
ESC_REV_LIN = float(os.environ.get("ESC_REV_LIN", "-0.30"))
ESC_REV_S = float(os.environ.get("ESC_REV_S", "1.2"))
ESC_SETTLE_S = float(os.environ.get("ESC_SETTLE_S", "1.2"))
ESC_TURN_DEG = float(os.environ.get("ESC_TURN_DEG", "45"))
ESC_SWING_DEG = float(os.environ.get("ESC_SWING_DEG", "120"))
ESC_TRIES = int(os.environ.get("ESC_TRIES", "4"))
ESC_BOXED_MAX = int(os.environ.get("ESC_BOXED_MAX", "3"))
# Stalled forward (roots, slope, soft dirt): after FAN_BOOST_AFTER_S with no
# wheel travel, add FAN_BOOST_RATE of linear per second, up to FAN_BOOST_MAX,
# before the stuck watchdog calls it stuck.
FAN_BOOST_AFTER_S = float(os.environ.get("FAN_BOOST_AFTER_S", "2.5"))
FAN_BOOST_RATE = float(os.environ.get("FAN_BOOST_RATE", "0.06"))
FAN_BOOST_MAX = float(os.environ.get("FAN_BOOST_MAX", "0.25"))
FAN_STUCK_S = float(os.environ.get("FAN_STUCK_S", "6"))
FAN_STUCK_M = float(os.environ.get("FAN_STUCK_M", "0.15"))
# A heading that ended in an escape is off-limits (world frame, +-BLOCK_DEG)
# until the rover has driven BLOCK_M or BLOCK_S has passed.
FAN_BLOCK_DEG = float(os.environ.get("FAN_BLOCK_DEG", "15"))
FAN_BLOCK_M = float(os.environ.get("FAN_BLOCK_M", "3.0"))
FAN_BLOCK_S = float(os.environ.get("FAN_BLOCK_S", "60"))
_FSU: FreeSpaceUrban | None = None
# Places an escape proved shut, in the odometry frame (see obstacle_memory).
FAN_MEM = os.environ.get("FAN_MEM", "1") == "1"
# Heading = gyro for turns + slow, gated compass / GPS-course correction
# (see heading.py). HEADING_FUSE=0 uses the raw compass as before.
HEADING_FUSE = os.environ.get("HEADING_FUSE", "1") == "1"
# HEADING_K_COMPASS: "auto" = 0.04, or 0 if the startup wiggle shows the
# compass not following the gyro (Panama hinge-moto-nova); a number forces it.
HEADING_K_COMPASS = os.environ.get("HEADING_K_COMPASS", "auto").strip().lower()
COMPASS_BAD_DEG = float(os.environ.get("COMPASS_BAD_DEG", "20"))
_HEADING = HeadingFilter(
    k_compass=0.04 if HEADING_K_COMPASS == "auto" else float(HEADING_K_COMPASS))
HEADING_STATE = os.environ.get("HEADING_STATE", os.path.join("runs", "heading_state.json"))
HEADING_BOOTSTRAP_M = float(os.environ.get("HEADING_BOOTSTRAP_M", "8"))
FAN_MEM_HITS = int(os.environ.get("FAN_MEM_HITS", "3"))
# Off-road: remember only what is seen close (<= 1.5 m); far boundaries on a
# slope (a tree line over a crest) read ~1 m and, remembered, boxed the rover in.
# "urban" (paved allow-list, avoid grass) or "offroad" (park: everything but
# real obstacles is ground; only things >= FAN_MIN_OBS_H tall stop the rover).
FAN_TERRAIN = os.environ.get("FAN_TERRAIN", "urban")
FAN_SIMPLE = os.environ.get("FAN_SIMPLE", "0") == "1"
FAN_SEGMENT_M = float(os.environ.get("FAN_SEGMENT_M", "0"))
# Passive obstacle memory is valuable beside urban walls, but unsafe off-road:
# a stable uphill tree/grass boundary is re-added every tick and tiny odometry
# changes during pivots smear it into hundreds of magenta spokes.  Off-road
# remembers only an obstacle that actually caused an escape/stall.
FAN_MEM_OBSERVE = os.environ.get(
    "FAN_MEM_OBSERVE", "0" if FAN_TERRAIN == "offroad" else "1") == "1"
FAN_MEM_ADD_MAX_M = float(os.environ.get(
    "FAN_MEM_ADD_MAX_M", "1.5" if FAN_TERRAIN == "offroad" else "3.0"))
_MEM: ObstacleMemory | None = (
    ObstacleMemory(
        add_range_m=FAN_MEM_ADD_MAX_M,
        seen_max_m=float(os.environ.get(
            "FAN_MEM_SEEN_MAX_M", "1.5" if FAN_TERRAIN == "offroad" else "2.5")),
    ) if FAN_MEM else None)
_FAN_IDS: tuple[list[int], ...] | None = None
FAN_UNLABEL = os.environ.get("FAN_UNLABEL", "1") == "1"
_FAN_SKY: np.ndarray | None = None         # off-road: sky mask of this tick
_FAN_RANGE_RELIEF: np.ndarray | None = None  # off-road: depth-gated far boundary
# Off-road: kerb-like classes, never waved through as flat litter. Only these:
# Panama leaf litter is labelled tree/building/fence/water, a Machakos
# concrete kerb is labelled wall.
KERB_NAMES = {"wall", "stairs", "steps", "stairway"}
OFFROAD_GROUND_NAMES = GROUND_NAMES | GRASS_NAMES | {"hill", "mountain", "dirt track", "land", "sand"}
_FAN_KERB: np.ndarray | None = None
FAN_MIN_OBS_H = float(os.environ.get("FAN_MIN_OBS_H", "0.12" if FAN_TERRAIN == "offroad" else "0.07"))
FAN_RANGE_RELIEF_RATIO = float(os.environ.get(
    "FAN_RANGE_RELIEF_RATIO", "1.2" if FAN_TERRAIN == "offroad" else "0"))
FAN_RANGE_RELIEF_MIN_M = float(os.environ.get(
    "FAN_RANGE_RELIEF_MIN_M", "2.0" if FAN_TERRAIN == "offroad" else "0"))


def _fan_masks(sem, labels_hi):
    """(track, grass) masks from an allow-list of ADE20K ground classes."""
    global _FAN_IDS
    if _FAN_IDS is None:
        id2 = sem.model.config.id2label
        _FAN_IDS = (sorted(class_ids(id2, GROUND_NAMES)), sorted(class_ids(id2, GRASS_NAMES)),
                    sorted(class_ids(id2, NEVER_LOW_NAMES)))
    if FAN_TERRAIN == "offroad":
        # Lenient: only real obstacles (trees, rocks, walls, people ...) are
        # not ground; grass is plain ground (no grass tier).
        global _FAN_SKY, _FAN_RANGE_RELIEF, _FAN_KERB
        if len(_FAN_IDS) < 7:
            id2 = sem.model.config.id2label
            _FAN_IDS = _FAN_IDS[:3] + (sorted(class_ids(id2, OFFROAD_OBSTACLE_NAMES)),
                                       sorted(class_ids(id2, {"sky"})),
                                       sorted(class_ids(id2, {"tree", "palm"})),
                                       sorted(class_ids(id2, KERB_NAMES)))
        _FAN_KERB = np.isin(labels_hi, _FAN_IDS[6])
        _FAN_SKY = np.isin(labels_hi, _FAN_IDS[4])
        # SegFormer labels the same Panama/Mexico forest edge as tree, pole,
        # building or sign as the camera turns.  Let every explicit off-road
        # obstacle class use range relief; the raw-depth ratio still keeps a
        # genuinely near trunk/person/vehicle at its original blocking range.
        _FAN_RANGE_RELIEF = np.isin(labels_hi, _FAN_IDS[3])
        # Allow-list, not "everything but known obstacles": a Costa Rica yard
        # play slide read "sculpture" and the rover climbed it to 42 deg; a
        # flower bed read "plant". Ground = paved + earth + grass classes; any
        # other label is an obstacle unless the flat/depth tests wave it
        # through, which is what keeps Panama's leaf litter (labelled plant,
        # bag, tree ...) drivable: 0/260 Panama frames changed.
        if len(_FAN_IDS) < 8:
            id2 = sem.model.config.id2label
            _FAN_IDS = _FAN_IDS + (sorted(class_ids(id2, OFFROAD_GROUND_NAMES)),)
        return np.isin(labels_hi, _FAN_IDS[7]) & ~_FAN_SKY, np.zeros(labels_hi.shape, bool)
    track, grass = np.isin(labels_hi, _FAN_IDS[0]), np.isin(labels_hi, _FAN_IDS[1])
    if FAN_SIMPLE:
        # One ground class: no paved-vs-grass steering preference.
        return track | grass, np.zeros_like(grass)
    if FAN_UNLABEL:
        hz = CALIB.scaled_to(labels_hi.shape[1], labels_hi.shape[0]).horizon_v()
        track = unlabel_impossible(track, np.isin(labels_hi, _FAN_IDS[2]) & ~grass, hz)
    return track, grass


def _downsample_labels(sem, labels_hi, out_hw):
    """Nearest-neighbour label map (and confidence) at the old pipeline's size."""
    h, w = labels_hi.shape
    rows = ((np.arange(out_hw[0]) + 0.5) * h / out_hw[0]).astype(int)
    cols = ((np.arange(out_hw[1]) + 0.5) * w / out_hw[1]).astype(int)
    conf = getattr(sem, "last_conf", None)
    if conf is not None and conf.shape == labels_hi.shape:
        sem.last_conf = conf[np.ix_(rows, cols)]
    return labels_hi[np.ix_(rows, cols)]


def _hdg_note(tel) -> str:
    raw = getattr(_HEADING, "last_raw", None)
    if not HEADING_FUSE or raw is None or tel.yaw is None:
        return ""
    return f"hdg={np.degrees(tel.yaw) % 360:.0f}(c{raw:.0f}) "


def _fan_state(stuck_state: dict) -> dict:
    fan = stuck_state.get("fan")
    if fan is None:
        fan = {
            "h_lp": 0.0, "fwd_since": None, "fwd_s0": 0.0, "esc": None,
            "boxed": 0, "esc_travel": None,
            "pivot": PulsePivot(ang=ALIGN_ANG, settle_s=SETTLE_S, tol_deg=FAN_TOL_DEG,
                                exact_bursts=_STREAMER is not None and STREAM_CMDS),
        }
        stuck_state["fan"] = fan
    return fan


def _fan_forbid(fan: dict, bins_deg: np.ndarray, yaw_deg, now: float):
    """Bins pointing (in the world) at a heading that recently ended in an escape."""
    blk = [b for b in fan.get("blocked", [])
           if _CRUMBS.travelled_m < b[1] and now < b[2]]
    fan["blocked"] = blk
    if yaw_deg is None or not blk:
        return None
    world = yaw_deg - bins_deg                 # bins are left-positive, yaw is CW
    forbid = np.zeros(len(bins_deg), bool)
    for w, _, _ in blk:
        forbid |= np.abs((world - w + 180.0) % 360.0 - 180.0) <= FAN_BLOCK_DEG
    return forbid


def _fan_begin_escape(fan: dict, why: str, now: float, force_reverse: bool,
                      yaw_deg=None, failed_rel_deg: float = 0.0,
                      pts: np.ndarray | None = None, sticky: bool = False) -> None:
    # Transient only if it came out of nowhere (clear the tick before) and this
    # heading has not already been let off once. Wuhan: pressed against a
    # scooter, the overhang read 0.1 m / clear / 0.1 m, and unblocking every
    # time pushed it into the same wheel three times.
    sudden = fan.get("last_ahead", 0.0) >= FAN_SLOW_M
    fan["esc"] = {"phase": "stop", "t": now + ESC_SETTLE_S, "tries": 0,
                  "swing": 0.0, "force": force_reverse, "reversed": False, "blk": None,
                  "sudden": sudden}
    fan["fwd_since"] = None
    fan["esc_travel"] = _CRUMBS.travelled_m
    blocked = ""
    if yaw_deg is not None:
        w = (yaw_deg - failed_rel_deg) % 360.0
        entry = (w, _CRUMBS.travelled_m + FAN_BLOCK_M, now + FAN_BLOCK_S)
        fan.setdefault("blocked", []).append(entry)
        fan["esc"]["blk"] = entry
        if why.startswith("tilted"):
            k = int(FAN_TILT_BLOCK_DEG // (2 * FAN_BLOCK_DEG))
            for j in range(1, k + 1):
                for sgn in (-1, 1):
                    fan["blocked"].append(((w + sgn * j * 2 * FAN_BLOCK_DEG) % 360.0,
                                           entry[1], entry[2]))
        blocked = f" — heading {w:.0f} deg blocked for {FAN_BLOCK_M:.0f} m"
    # "Can't get through here": remember it, so turning back to the goal does
    # not mean turning back into it.
    if _MEM is not None and pts is not None and len(pts):
        n0 = len(_MEM)
        fan["esc"]["mem_batch"] = _MEM.add(pts, _MEM.yaw, now, sticky=sticky)
        blocked += f"; remembered {len(_MEM) - n0} pts (mem {len(_MEM)})"
    print(f"ESCAPE: {why}{blocked}", flush=True)
    _event(f"escape: {why}")


def _fan_escape(fan: dict, fs, want: float, tel, now: float, forbid_fn=None,
                mem_pts=None):
    """One tick of the escape episode -> (lin, ang, hold_s, note)."""
    e, piv = fan["esc"], fan["pivot"]
    if e["phase"] == "stop":
        if now < e["t"]:
            return 0.0, 0.0, None, "escape: stop+settle"
        if e["force"] or fs.drive.ahead_m < FAN_NEED_M:
            e["phase"], e["t"] = "rev", now + ESC_REV_S
        else:
            e["phase"] = "look"
            # Still frame, same heading, clear again without backing off: a
            # one-frame blip or someone who walked through. Wuhan mission-3:
            # open plaza read 5.0 m, 0.5 m, 5.0 m on consecutive ticks, and
            # the block sent the rover 15 deg off the goal for 3 m.
            w0 = e["blk"][0] if e["blk"] is not None else None
            let_off = [t for t in fan.get("let_off", [])
                       if _CRUMBS.travelled_m < t[1] and now < t[2]]
            fan["let_off"] = let_off
            again = w0 is not None and any(
                abs((w0 - t[0] + 180.0) % 360.0 - 180.0) <= FAN_BLOCK_DEG for t in let_off)
            if e["blk"] is not None and e["swing"] == 0.0 and e["sudden"] and not again \
                    and fs.drive.ahead_m >= FAN_NEED_M:
                blk = fan.get("blocked", [])
                if e["blk"] in blk:
                    blk.remove(e["blk"])
                let_off.append(e["blk"])
                e["blk"] = None
                if _MEM is not None and e.get("mem_batch"):
                    _MEM.drop_batch(e["mem_batch"])
                    # This tick's plan still saw those points: look next tick.
                    return 0.0, 0.0, None, "escape: transient, re-planning"
                print("  escape: clear again after stopping — transient, heading unblocked",
                      flush=True)
    if e["phase"] == "rev":
        if not e["reversed"]:
            e["rev_s0"] = _CRUMBS.travelled_m
        e["reversed"] = True
        if now < e["t"]:
            # Blind (no rear camera on the Mini+), hence short. A reverse that
            # barely moved last time (roots) gets a bit more.
            return (ESC_REV_LIN - fan.get("rev_boost", 0.0), 0.0, REVERSE_HOLD_S,
                    "escape: reverse")
        moved = _CRUMBS.travelled_m - e.get("rev_s0", _CRUMBS.travelled_m)
        fan["rev_boost"] = (min(0.15, fan.get("rev_boost", 0.0) + 0.075) if moved < 0.05
                            else max(0.0, fan.get("rev_boost", 0.0) - 0.075))
        e["phase"], e["t"] = "settle", now + ESC_SETTLE_S
        return 0.0, 0.0, None, "escape: reversed, settling"
    if e["phase"] == "settle":
        if now < e["t"]:
            return 0.0, 0.0, None, "escape: settle"
        e["phase"] = "look"
    # look: a fresh, still frame, plus what is remembered around the rover
    h, which, inview = choose_full(fs, want, FAN_NEED_M, None, mem_pts, forbid_fn,
                                   half_width_m=getattr(_FSU, "half_width_m", 0.17))
    if h is not None and inview and abs(e["swing"] + h) <= ESC_SWING_DEG:
        fan["esc"] = None
        print(f"  escape: clear heading {h:+.0f}deg ({which}) — resuming", flush=True)
        if abs(h) > FAN_TOL_DEG:
            lin, ang, pn = piv.command(h, tel, now)
            return lin, ang, piv.burst_remaining(now), f"escape: turn {h:+.0f} pivot[{pn}]"
        return 0.0, 0.0, None, "escape: clear ahead"
    e["tries"] += 1
    if h is not None and not inview and e["tries"] <= ESC_TRIES:
        # Nothing in view, but memory says that way (aside, or back the way
        # we came) is open: turn to look at it. At most 90 deg per look.
        turn = float(np.clip(h, -90.0, 90.0))
        lin, ang, pn = piv.command(turn, tel, now)
        return (lin, ang, piv.burst_remaining(now),
                f"escape: turn {turn:+.0f} toward {h:+.0f} (remembered clear, try "
                f"{e['tries']}/{ESC_TRIES}) pivot[{pn}]")
    if e["tries"] > ESC_TRIES:
        fan["boxed"] += 1
        if fan["boxed"] > ESC_BOXED_MAX:
            print(f"STUCK: boxed in {fan['boxed']} times — stopping. Needs walker assist.",
                  flush=True)
            raise SystemExit(3)
        print(f"  escape: boxed in ({fan['boxed']}/{ESC_BOXED_MAX}) — holding 4 s, "
              f"then backing off again", flush=True)
        # Same keys as a fresh escape: a missing "reversed" crashed every
        # tick after a boxed-in hold (Panama mission-2, "tick error: 'reversed'").
        fan["esc"] = {"phase": "stop", "t": now + 4.0, "tries": 0, "swing": 0.0,
                      "force": True, "reversed": False, "blk": None, "sudden": False}
        return 0.0, 0.0, None, "escape: boxed in"
    turn = open_side(fs) * ESC_TURN_DEG
    if abs(e["swing"] + turn) > ESC_SWING_DEG:
        turn = -turn
    if abs(e["swing"] + turn) > ESC_SWING_DEG:
        e["phase"], e["t"] = "rev", now + ESC_REV_S
        return 0.0, 0.0, None, "escape: swing limit — backing off"
    e["swing"] += turn
    lin, ang, pn = piv.command(turn, tel, now)
    return (lin, ang, piv.burst_remaining(now),
            f"escape: look {turn:+.0f} (try {e['tries']}/{ESC_TRIES}) pivot[{pn}]")


# Global route over OpenStreetMap footways (modules/osm_route.py). Machakos
# leg-1: the CP2->CP3 chord crossed bush, a kerb, gravel and grassy banks for
# 40+ min while a mapped footway ran from CP2 straight to CP3. Steer at a
# point OSM_LOOKAHEAD_M along the route; within OSM_DIRECT_M of the checkpoint,
# or with no route, steer at the checkpoint as before.
USE_OSM = os.environ.get("USE_OSM", "1") == "1"
OSM_LOOKAHEAD_M = float(os.environ.get("OSM_LOOKAHEAD_M", "8"))
OSM_DIRECT_M = float(os.environ.get("OSM_DIRECT_M", "8"))
OSM_REPLAN_M = float(os.environ.get("OSM_REPLAN_M", "20"))
# Short legs go straight: checkpoints often sit just off the road (Costa Rica
# mission-1: 20 m legs beside a junction) and snapping to road nodes there is a
# detour, not a route. Same for routes much longer than the chord.
OSM_MIN_LEG_M = float(os.environ.get("OSM_MIN_LEG_M", "40"))
OSM_MAX_DETOUR = float(os.environ.get("OSM_MAX_DETOUR", "2.5"))
_OSM: dict = {"payload": None, "goal": None, "route": None, "t": 0.0}


def _osm_load(points) -> None:
    """One Overpass payload covering the whole mission (cached on disk)."""
    if not USE_OSM or not points:
        return
    lats, lons = [p[0] for p in points], [p[1] for p in points]
    box = osm_route.bbox((min(lats), min(lons)), (max(lats), max(lons)), buffer_m=150.0)
    fetch = osm_route.CachedFetch()
    for _ in range(3):
        payload = fetch(box)
        if payload:
            _OSM["payload"] = payload
            n = sum(1 for e in payload.get("elements", []) if e.get("type") == "way")
            print(f"[osm] map loaded: {n} ways", flush=True)
            return
        time.sleep(3.0)
    print("[osm] no map (Overpass unreachable) — straight lines to checkpoints", flush=True)


def _osm_target(lat: float, lon: float, goal, rng: float):
    """(lat, lon) to steer at this tick, or None to steer at the checkpoint."""
    if not USE_OSM or _OSM["payload"] is None or rng <= OSM_DIRECT_M:
        return None
    here = (lat, lon)
    r = _OSM["route"]
    off = (min(osm_route.haversine_m(here, p) for p in r.waypoints)
           if r is not None and r.waypoints else float("inf"))
    if _OSM["goal"] != tuple(goal) or (off > OSM_REPLAN_M and time.monotonic() - _OSM["t"] > 30.0):
        chord = osm_route.haversine_m(here, tuple(goal))
        if chord < OSM_MIN_LEG_M:
            r = osm_route.Route([here, tuple(goal)], "chord", chord, 1.0, "short leg")
        else:
            r = osm_route.plan(here, tuple(goal), fetch=lambda box: _OSM["payload"])
            if r.ok and r.detour_ratio > OSM_MAX_DETOUR:
                r = osm_route.Route([here, tuple(goal)], "chord", chord, 1.0,
                                    f"detour x{r.detour_ratio:.1f}")
        _OSM.update(goal=tuple(goal), route=r, t=time.monotonic())
        print(f"[osm] route to {goal[0]:.6f},{goal[1]:.6f}: {r.describe()}", flush=True)
    if r is None or not r.ok:
        return None
    return osm_route.next_target(r, here, OSM_LOOKAHEAD_M)


# Tilt guard. Machakos leg-1: climbing a kerb/rock edge pitched the rover up
# until the camera saw only sky; off-road counts sky as ground, read "5 m
# clear" and kept driving. Gravity is measured at startup (axis differs per
# unit), so this is the angle away from how the rover sat then.
FAN_TILT_DEG = float(os.environ.get("FAN_TILT_DEG", "25"))
# Pivoting on an edge tips the rover sideways/back while it thinks it is
# standing still (Machakos: slid off a raised bed edge mid-pivot and read
# 100 deg). Any pivot is cut above FAN_PIVOT_TILT_DEG, every tick.
FAN_PIVOT_TILT_DEG = float(os.environ.get("FAN_PIVOT_TILT_DEG", "10"))   # rise during one pivot
# After a tilt escape, block this wide a sector (not FAN_BLOCK_DEG): with
# +-15 deg the rover re-tried the same Machakos kerb from 20 deg further round,
# 15 escapes and 2 flips in 8 minutes.
FAN_TILT_BLOCK_DEG = float(os.environ.get("FAN_TILT_BLOCK_DEG", "45"))
_GRAVITY: np.ndarray | None = None
_LAST_ACC = np.zeros(3)


def _snap_gravity(g) -> np.ndarray:
    """Unit gravity, snapped to the nearest axis when within 30 deg of it: tilt
    is then measured from true vertical, not from however the rover sat at
    startup. Costa Rica started on a ~9 deg driveway; pivoting there changed
    tilt relative to the start pose by 20+ deg on flat concrete."""
    g = np.asarray(g, float)
    g = g / np.linalg.norm(g)
    ax = int(np.argmax(np.abs(g)))
    if abs(g[ax]) > math.cos(math.radians(30.0)):
        g = np.eye(3)[ax] * np.sign(g[ax])
    return g


def _tilt_deg(tel) -> float | None:
    acc = (getattr(tel, "raw", None) or {}).get("accels") or []
    if _GRAVITY is None or not acc:
        return None
    # Every sample of the burst: pivot jolts read 30 deg on single samples
    # while the rover sat level (Machakos).
    a = np.array([x[:3] for x in acc if isinstance(x, list) and len(x) >= 3], float)
    if not len(a):
        return None
    v = a.mean(0)
    n = float(np.linalg.norm(v))
    # Standing still or rolling slowly the IMU reads ~1 g, whatever the tilt.
    # Costa Rica: one burst read 162 deg (upside down) with the rover level on
    # its wheels; a glitch or a knock, not a pose.
    if not 0.75 <= n <= 1.3:
        return None
    _LAST_ACC[:] = v
    t = float(np.degrees(np.arccos(np.clip(np.dot(v / n, _GRAVITY), -1.0, 1.0))))
    return t if t <= 60.0 else None          # no climb reads like that: a glitch


def _tilt_held(fan: dict, tel) -> float | None:
    """This tick's tilt, but only once it has held for 2 ticks in a row (the
    lower of the two), so one jolt cannot trigger a back-off."""
    t = _tilt_deg(tel)
    prev, fan["tilt_prev"] = fan.get("tilt_prev"), t
    if t is None or prev is None:
        return None
    return min(t, prev)


def _fan_stuck(fan: dict, lin: float, now: float) -> bool:
    """Commanded forward FAN_STUCK_S with < FAN_STUCK_M of wheel travel."""
    if lin <= 0.0:
        fan["fwd_since"] = None
        return False
    if fan["fwd_since"] is None or _CRUMBS.travelled_m - fan["fwd_s0"] >= FAN_STUCK_M:
        fan["fwd_since"], fan["fwd_s0"] = now, _CRUMBS.travelled_m
        return False
    return now - fan["fwd_since"] > FAN_STUCK_S


def _fan_push_guards(fan: dict, lin: float, ang: float, mode: str, now: float, yaw_deg):
    """Stall boost, tilt and stall guards for an agent push (same as the drive branch)."""
    global _AGENT_PUSH
    if fan["fwd_since"] is not None:
        stalled = now - fan["fwd_since"] - FAN_BOOST_AFTER_S
        if stalled > 0.0:
            boost = min(FAN_BOOST_MAX, FAN_BOOST_RATE * stalled)
            lin += boost
            mode += f" boost+{boost:.2f}"
    tilt = fan.get("tilt_now")
    bump = np.stack([np.full(5, 0.25), np.linspace(-0.2, 0.2, 5)], 1)
    if tilt is not None and tilt > FAN_TILT_DEG:
        _AGENT_PUSH = None
        _fan_begin_escape(fan, f"tilted {tilt:.0f} deg during push", now, force_reverse=True,
                          yaw_deg=yaw_deg, failed_rel_deg=0.0, pts=bump, sticky=True)
        return 0.0, 0.0, "ESCAPE tilt (push)"
    if _fan_stuck(fan, lin, now):
        _AGENT_PUSH = None
        _fan_begin_escape(fan, "stuck during push: wheels not advancing", now,
                          force_reverse=True, yaw_deg=yaw_deg, failed_rel_deg=0.0,
                          pts=bump, sticky=True)
        return 0.0, 0.0, "ESCAPE stuck (push)"
    return lin, ang, mode


def _fan_step(client, tel, front, rng, bearing, pre_aim, ground, grass,
              stuck_state, odom, sem_note, goal_index, depth=None) -> bool:
    """One NAV=fan tick: decide, command, log, publish. Never claims a goal."""
    global _FSU
    now = time.monotonic()
    if _FSU is None:
        _FSU = FreeSpaceUrban(CALIB, seg_hw=FAN_SEG_HW, min_obstacle_h_m=FAN_MIN_OBS_H,
                              # Unverified on saved frames (raw depth near-field
                              # squash defeats it); opt-in only.
                              occlusion_ratio=float(os.environ.get("FAN_OCCLUSION_RATIO", "0")),
                              range_relief_ratio=FAN_RANGE_RELIEF_RATIO,
                              range_relief_min_m=FAN_RANGE_RELIEF_MIN_M)
    fan = _fan_state(stuck_state)
    fan["tilt_now"] = _tilt_held(fan, tel)
    fs = _FSU.estimate(ground, grass, depth,
                       sky=_FAN_SKY if (FAN_TERRAIN == "offroad" and _FAN_SKY is not None
                                        and _FAN_SKY.shape == ground.shape) else None,
                       range_relief=_FAN_RANGE_RELIEF
                       if (FAN_TERRAIN == "offroad" and _FAN_RANGE_RELIEF is not None
                           and _FAN_RANGE_RELIEF.shape == ground.shape) else None,
                       # Kerbs/walls are never flat litter (see FreeSpaceUrban).
                       hard=_FAN_KERB
                       if (FAN_TERRAIN == "offroad" and _FAN_KERB is not None
                           and _FAN_KERB.shape == ground.shape) else None)
    prev_ahead = fan.get("ahead_now", fs.drive.ahead_m)
    fan["last_ahead"], fan["ahead_now"] = prev_ahead, fs.drive.ahead_m
    chunk = None
    if FAN_SEGMENT_M > 0:
        chunk = fan.setdefault("segment", SegmentCommit(now, max_m=FAN_SEGMENT_M))
        phase = chunk.update(odom, now)
        if fan["pivot"].busy() or fan["esc"] is not None:
            chunk.reset(now)
        else:
            if chunk.active and (fs.drive.ahead_m < FAN_STOP_M
                    or fs.drive.clear_at(chunk.heading) < FAN_STOP_M + 0.2
                    or abs(chunk.heading) > FAN_ALIGN_DEG):
                chunk.reset(now)
                phase = "live obstacle/heading stop"
            if not chunk.active and (now < chunk.ready_at or phase != "ready"):
                _halt(client)
                fan["fwd_since"] = None
                print(f"[segment] STOP {phase}; stationary fresh-frame planning", flush=True)
                _publish_overlay(front, lines=(f"STOP / PLAN: {phase}",),
                                 fan=(fs, None, float(np.degrees(bearing)), None),
                                 ground=ground, grass=grass)
                return False
    want = float(np.degrees(pre_aim.bearing))
    # Heading bootstrap. With the compass ignored and no GPS-track fix yet,
    # the heading is a guess, so aiming at the checkpoint is aiming anywhere:
    # Costa Rica mission-1 started beside a steel gate that bent the compass
    # seed, turned "toward" CP1 into the gate and boxed itself in before
    # driving the 4 m a fix needs. Drive the clearest way ahead first.
    boot = (HEADING_FUSE and HEADING_BOOTSTRAP_M > 0 and _HEADING.k_compass <= 0
            and _HEADING.fixes == 0 and _CRUMBS.travelled_m < HEADING_BOOTSTRAP_M)
    if boot:
        c = fs.drive.clear_m
        ok = c >= min(c.max(), FAN_NEED_M * 2)
        want = float(fs.drive.bins_deg[ok][np.argmin(np.abs(fs.drive.bins_deg[ok]))]) if ok.any() else 0.0
        if not fan.get("boot_told"):
            fan["boot_told"] = True
            print(f"[heading] no fix yet: driving the clearest way ({want:+.0f} deg) "
                  f"for up to {HEADING_BOOTSTRAP_M:.0f} m to get a GPS heading", flush=True)
    yaw_deg = float(np.degrees(tel.yaw)) if tel.yaw is not None else None
    global _AIM_OVERRIDE, _AGENT_BACK, _AGENT_TAKEOVER, _AGENT_PUSH
    if _AGENT_TAKEOVER:
        _AGENT_TAKEOVER = False
        if fan["esc"]:
            fan["esc"] = None
            if hasattr(fan.get("pivot"), "reset"):
                fan["pivot"].reset()
            _event("escape cancelled by driver")
    _OBS.update(t=now, front=front, fs=fs, yaw=yaw_deg, want_gps=want, rng=rng,
                goal_index=goal_index, lat=getattr(tel, "lat", float("nan")),
                lon=getattr(tel, "lon", float("nan")),
                tilt=fan.get("tilt_now"), escaping=bool(fan["esc"]),
                travelled=_CRUMBS.travelled_m)
    ov = _AIM_OVERRIDE
    if ov is not None and yaw_deg is not None and not boot:
        if _CRUMBS.travelled_m < ov["until_m"] and now < ov["t_until"]:
            want = (yaw_deg - ov["world_deg"] + 180.0) % 360.0 - 180.0
        else:
            _AIM_OVERRIDE = None
            _event("agent heading done")
    if _AGENT_BACK > 0.0 and not fan["esc"]:
        _AGENT_BACK = 0.0
        _fan_begin_escape(fan, "agent: back up", now, force_reverse=True)
    push = _AGENT_PUSH
    if push is not None and (yaw_deg is None or boot or _CRUMBS.travelled_m >= push["until_m"]
                             or now >= push["t_until"]):
        _AGENT_PUSH = push = None
        _event("agent push done")

    # Goal behind: choose which way to turn round ONCE. At +-180 the sign of
    # the bearing flips with every degree of noise, and each flip reversed the
    # pivot (Panama mission-2, CP3 -> CP4: right 90, left 90, right 90 ...).
    if abs(want) > FAN_BEHIND_LATCH_DEG:
        if fan.get("behind_dir") is None:
            fan["behind_dir"] = 1.0 if want >= 0.0 else -1.0
    elif abs(want) < 90.0:
        fan["behind_dir"] = None
    behind_dir = fan.get("behind_dir")

    def forbid_fn(bins):
        f = _fan_forbid(fan, bins, yaw_deg, now)
        if behind_dir is not None:
            wrong = (bins * behind_dir < 0.0) & (np.abs(bins) > 90.0)
            f = wrong if f is None else (f | wrong)
        return f

    mem_pts = None
    if _MEM is not None and (tel.yaw is not None or odom is not None):
        # Memory frame heading: wheel+gyro odometry (compass only as a fallback).
        if odom is not None and odom.reason == "wheel" and odom.confidence > 0.0:
            _MEM.step_odom(odom.dx, odom.dy, odom.dyaw)
        elif tel.yaw is not None:
            if odom is not None and odom.confidence > 0.0:
                _MEM.update_pose(odom.dx, odom.dy, tel.yaw)
            _MEM.yaw = tel.yaw
        my = _MEM.yaw
        _MEM.prune(now)
        _MEM.clear_seen(fs.drive.pts, my)
        if FAN_MEM_OBSERVE:
            _MEM.observe(fs.drive.pts, my, now)
        mem_pts = _MEM.to_robot(my, max_range_m=10.0, min_hits=FAN_MEM_HITS)
    # Around what is remembered: the shortest path's heading replaces the
    # straight bearing when they differ (goal behind a wall, inside a pocket).
    plan_note = ""
    if mem_pts is not None and len(mem_pts) and not (chunk and chunk.active):
        ph = plan_heading(mem_pts, want, rng)
        if ph is not None and abs((ph - want + 180.0) % 360.0 - 180.0) > 8.0:
            # Commit to the way round: a pillar dead on the goal line is a
            # tie, and re-deciding every tick pivoted left, right, left. Keep
            # the committed world heading until 1 m of travel, or until memory
            # says it is shut; small corrections (< 30 deg) always go through.
            prev = fan.get("plan")
            if yaw_deg is not None:
                ph_w = (yaw_deg - ph) % 360.0
                if prev is not None:
                    rel_c = (yaw_deg - prev[0] + 180.0) % 360.0 - 180.0
                    turned = abs((ph_w - prev[0] + 180.0) % 360.0 - 180.0) > 30.0
                    fresh = _CRUMBS.travelled_m - prev[1] < 1.0
                    open_c = path_clear(mem_pts, rel_c, getattr(_FSU, "half_width_m", 0.17)) \
                        >= FAN_NEED_M
                    if turned and fresh and open_c:
                        ph, ph_w = rel_c, prev[0]
                    else:
                        prev = None
                if prev is None:
                    fan["plan"] = (ph_w, _CRUMBS.travelled_m)
            plan_note = f" plan={ph:+.0f}"
            want = ph
        else:
            fan["plan"] = None
    piv = fan["pivot"]
    lin = ang = 0.0
    hold = None
    h, which = None, ""
    # Stale wheel data is not "no motion": with a laggy link (Panama, signal
    # 3, rpm 2+ s old) the stuck watchdog and the stall boost read it as
    # wheels not turning and fired escapes on open grass. Restart their clock.
    if odom is not None and odom.reason in ("wheel_stale", "wheel_not_ready"):
        fan["fwd_since"] = None
    # Same after a gap in ticks (link loss, SDK restart): the first tick back
    # otherwise sees "6 s forward, no travel" and escapes on open grass.
    if now - fan.get("last_tick", now) > 2.0:
        fan["fwd_since"] = None
    fan["last_tick"] = now
    if fan["esc_travel"] is not None and _CRUMBS.travelled_m - fan["esc_travel"] > 2.0:
        fan["boxed"], fan["esc_travel"] = 0, None      # got going again

    if piv.busy():
        lin, ang, pn = piv.command(0.0, tel, now)
        hold, mode = piv.burst_remaining(now), f"pivot[{pn}]"
    elif fan["esc"] is not None:
        lin, ang, hold, mode = _fan_escape(fan, fs, want, tel, now, forbid_fn=forbid_fn,
                                           mem_pts=mem_pts)
    elif push is not None:
        # Driver override of the camera stop: steer at the world heading, no
        # clearance checks. The tilt / stall guards below still end it.
        h = (yaw_deg - push["world_deg"] + 180.0) % 360.0 - 180.0
        if abs(h) > FAN_ALIGN_DEG:
            lin, ang, pn = piv.command(0.8 * h, tel, now)
            hold, mode = piv.burst_remaining(now), f"PUSH realign {h:+.0f}: pivot[{pn}]"
        else:
            dps = float(np.clip(FAN_KP * h, -FAN_MAX_DPS, FAN_MAX_DPS))
            ang = float(np.clip(dps / max(STEER_RATE_PRIOR, 10.0), -FAN_MAX_ANG, FAN_MAX_ANG))
            lin = max(FAN_LIN_MIN, V_NOM)
            mode = f"PUSH {h:+.0f}deg (driver override, camera stop ignored)"
            lin, ang, mode = _fan_push_guards(fan, lin, ang, mode, now, yaw_deg)
    else:
        # Full circle: goal behind or aside, and remembered walls, all in one
        # cost (see choose_full). Out of view means turn in place to look.
        if chunk and chunk.active:
            h, which, inview = chunk.heading, "track", True
            want = h
        else:
            h, which, inview = choose_full(fs, want, FAN_NEED_M, FAN_MIN_M, mem_pts, forbid_fn,
                                           half_width_m=getattr(_FSU, "half_width_m", 0.17))
        if h is not None and not inview:
            turn = float(np.clip(h, -90.0, 90.0))
            lin, ang, pn = piv.command(turn, tel, now)
            hold, mode = piv.burst_remaining(now), f"turn {h:+.0f} ({which}): pivot[{pn}]"
        elif fs.drive.ahead_m < FAN_STOP_M or h is None:
            why = (f"ground ends {fs.drive.ahead_m:.2f} m ahead"
                   if fs.drive.ahead_m < FAN_STOP_M else "no clear heading in view")
            _fan_begin_escape(fan, why, now, force_reverse=False,
                              yaw_deg=yaw_deg, failed_rel_deg=0.0, pts=fs.drive.pts)
            mode = f"ESCAPE {why}"
        elif abs(h) > (FAN_NEAR_ALIGN_DEG if rng < FAN_NEAR_M else FAN_ALIGN_DEG):
            lin, ang, pn = piv.command(0.8 * h, tel, now)
            hold, mode = piv.burst_remaining(now), f"realign {h:+.0f}: pivot[{pn}]"
        elif fs.drive.ahead_m < FAN_TIGHT_M and abs(h) > FAN_TOL_DEG:
            lin, ang, pn = piv.command(h, tel, now)
            hold = piv.burst_remaining(now)
            mode = f"tight ({fs.drive.ahead_m:.1f}m ahead) turn {h:+.0f}: pivot[{pn}]"
        else:
            fan["h_lp"] = 0.5 * fan["h_lp"] + 0.5 * h
            hl = fan["h_lp"]
            dps = float(np.clip(FAN_KP * hl, -FAN_MAX_DPS, FAN_MAX_DPS))
            ang = float(np.clip(dps / max(STEER_RATE_PRIOR, 10.0), -FAN_MAX_ANG, FAN_MAX_ANG))
            clr = (fs.track if which == "track" else fs.drive).clear_at(h)
            if chunk and not chunk.active:
                if chunk.begin(h, min(clr, fs.drive.ahead_m), FAN_STOP_M + 0.2, now):
                    print(f"[segment] COMMIT {h:+.0f}deg for {chunk.remaining:.2f}m", flush=True)
                else:
                    chunk.reset(now)
                    _halt(client)
                    fan["fwd_since"] = None
                    print("[segment] STOP insufficient segment clearance", flush=True)
                    return False
            slow = float(np.clip((clr - FAN_STOP_M) / max(FAN_SLOW_M - FAN_STOP_M, 1e-3), 0.5, 1.0))
            turn = float(np.clip(1.0 - abs(hl) / 45.0, 0.5, 1.0))
            lin = max(FAN_LIN_MIN, V_NOM * slow * turn * max(pre_aim.speed_scale, 0.6))
            lin = _clear_path_throttle(lin, clr, fs.drive.ahead_m, ang)
            mode = f"drive {h:+.0f}deg ({which}, {clr:.1f}m)"
            if chunk and chunk.active:
                mode += f" COMMIT {chunk.remaining:.2f}m left"
            if fan["fwd_since"] is not None:
                stalled = now - fan["fwd_since"] - FAN_BOOST_AFTER_S
                if stalled > 0.0:
                    boost = min(FAN_BOOST_MAX, FAN_BOOST_RATE * stalled)
                    lin += boost
                    mode += f" boost+{boost:.2f}"
            tilt = fan.get("tilt_now")
            if tilt is not None and tilt > FAN_TILT_DEG:
                bump = np.stack([np.full(5, 0.25), np.linspace(-0.2, 0.2, 5)], 1)
                _fan_begin_escape(fan, f"tilted {tilt:.0f} deg (climbing something; accel "
                                  f"{np.round(_LAST_ACC, 2).tolist()})", now,
                                  force_reverse=True, yaw_deg=yaw_deg, failed_rel_deg=0.0,
                                  pts=bump, sticky=True)
                lin = ang = 0.0
                mode = "ESCAPE tilt"
            elif _fan_stuck(fan, lin, now):
                # Something the camera cannot see (a kerb lip, a low bar) is
                # right at the bumper: remember a rover-wide strip there.
                bump = np.stack([np.full(5, 0.25), np.linspace(-0.2, 0.2, 5)], 1)
                _fan_begin_escape(fan, "stuck: wheels not advancing under a forward command",
                                  now, force_reverse=True, yaw_deg=yaw_deg, failed_rel_deg=0.0,
                                  pts=bump, sticky=True)
                lin = ang = 0.0
                mode = "ESCAPE stuck"
    if lin <= 0.0 and not (fan["esc"] and fan["esc"]["phase"] == "rev"):
        fan["fwd_since"] = None
    if chunk and (abs(ang) > 0 and lin <= 0 or fan["esc"] is not None):
        chunk.reset(now)

    odo_s = ""
    if odom is not None:
        odo_s = (f"odo({odom.distance_m*100:.1f}cm {np.degrees(odom.dyaw):+.0f}° "
                 f"{odom.reason}) trail={_CRUMBS.travelled_m:.1f}m ")
    _tick_print(
        f"  rng={rng:.1f}m bearing={np.degrees(bearing):+.0f}° want={want:+.0f}°{plan_note} "
        f"{_hdg_note(tel)}"
        f"aim={pre_aim.source} fan(track {fs.track.ahead_m:.1f}m drive "
        f"{fs.drive.ahead_m:.1f}m oh={_FSU.last_overhang_cols} ct={_FSU.last_contact_cols}"
        f"{' mem=%d/%d' % (_MEM.n_confirmed(), len(_MEM)) if _MEM is not None and len(_MEM) else ''}) "
        f"{odo_s}{sem_note}"
        f"{_GPS.describe() + ' ' if _GPS.stale else ''}src=fan[{mode}] "
        f"cmd lin={lin:.2f} ang={ang:.2f}")
    _publish_overlay(front, lines=(
        f"FAN {mode}",
        f"cmd=({lin:+.2f},{ang:+.2f})  rng={rng:.1f}m  want={want:+.0f}deg  aim={pre_aim.source}",
        f"ahead track {fs.track.ahead_m:.1f}m / drive {fs.drive.ahead_m:.1f}m   "
        f"{'OVERHANG x' + str(_FSU.last_overhang_cols) + '  ' if _FSU.last_overhang_cols else ''}"
        f"{'CONTACT x' + str(_FSU.last_contact_cols) + '  ' if _FSU.last_contact_cols else ''}"
        f"{'MEM ' + str(len(_MEM)) + '  ' if _MEM is not None and len(_MEM) else ''}"
        f"{'gps STALE  ' if _GPS.stale else ''}goal {goal_index + 1}",
    ), fan=(fs, h, want, mem_pts), ground=ground, grass=grass)
    # Tilt, whatever this tick decided (the drive branch only sees it while
    # going forward).
    tilt = fan.get("tilt_now")
    if (tilt is not None and abs(lin) < 1e-3 and abs(ang) > 1e-3
          and (tilt > FAN_TILT_DEG
               or tilt - fan.setdefault("tilt_pivot0", tilt) > FAN_PIVOT_TILT_DEG)):
        bump = np.stack([np.full(5, 0.25), np.linspace(-0.2, 0.2, 5)], 1)
        _fan_begin_escape(fan, f"tilted {tilt:.0f} deg while pivoting (edge)", now,
                          force_reverse=True, yaw_deg=yaw_deg, failed_rel_deg=0.0,
                          pts=bump, sticky=True)
        lin = ang = 0.0
    # Tilt when this pivot began: on a uniform slope tilt stays put while
    # turning; going over an edge raises it.
    # (Kept through the settle pauses between pulse bursts; reset once it
    # drives again.)
    if abs(lin) < 1e-3:
        if abs(ang) > 1e-3 and tilt is not None:
            fan.setdefault("tilt_pivot0", tilt)
    else:
        fan.pop("tilt_pivot0", None)
    if abs(lin) < 1e-3 and abs(ang) < 1e-3:
        _halt(client)
    else:
        _cmd(client, lin, ang, hold_s=hold)
    return False


def step(client, planner, selector, gate, posebuf, controller, goal,
         mission_mode: bool, stuck_state: dict | None = None,
         route_profile=None, goal_index: int = 0) -> bool:
    global _REAR_SEEN, _LAST_ALIVE
    try:
        front, rear = client.get_frames()
    except Exception:
        front, rear = None, None
    if rear is not None:
        _REAR_SEEN = True
    tel = client.get_data()
    # SDK liveness, independent of GPS: a frame and a non-empty /data payload.
    if front is not None and tel.raw:
        _LAST_ALIVE = time.monotonic()

    was_down = _LINK.down
    moving = _STREAMER.moving if _STREAMER is not None else False
    if _LINK.update(front, tel.hw_ts, moving):
        _halt(client)
        if not was_down:
            print(
                f"[LINK] no new image or IMU data for {LINK_STALE_S:.0f} s while "
                f"commanding motion (telemetry age "
                f"{tel.age_s if tel.age_s is not None else float('nan'):.0f} s) "
                f"— holding zero until data flows again",
                flush=True,
            )
        return False
    if was_down:
        print("[LINK] data flowing again — resuming", flush=True)

    if np.isnan(tel.lat) or np.isnan(tel.lon):
        return False
    if tel.yaw is None:
        return False
    if front is None:
        return False

    # Measured motion. Everything downstream that needs to know how far the
    # rover actually moved reads this, because GPS is frozen, `rpms` is
    # [0,0,0,0] and the compass drifts. Runs before any control decision.
    odom = _odom_update(front)
    if HEADING_FUSE:
        raw = float(np.degrees(tel.yaw)) if tel.yaw is not None else None
        ok = odom is not None and odom.confidence > 0.0
        if _HEADING.est is None and not getattr(_HEADING, "tried_load", False):
            _HEADING.tried_load = True
            saved = load_state(HEADING_STATE, tel.lat, tel.lon)
            if saved is not None:
                _HEADING.est, _HEADING.fixes = saved, 1
                print(f"[heading] resumed saved heading {saved:.0f} (rover has not moved)",
                      flush=True)
        fused = _HEADING.update(raw, float(np.degrees(odom.dyaw)) if ok else 0.0,
                                odom.dx if ok else 0.0, tel.lat, tel.lon)
        _HEADING.last_raw = raw
        if fused is not None:
            tel.yaw = float(np.radians(fused))
            if _HEADING.fixes:
                save_state(HEADING_STATE, fused, tel.lat, tel.lon, _HEADING.fixes)
    gps_stale = _GPS.update(tel.lat, tel.lon, odom, tel.stamp)
    if odom is not None:
        _CRUMBS.update(odom, tel.stamp)
        if posebuf is not None:
            posebuf.integrate(tel.stamp, odom)
    # Advance the map pose before folding anything into it. Translation from
    # VO, heading from the compass — see localmap.py for why that split is not
    # negotiable here.
    lmap = _map()
    if lmap is not None:
        lmap.update_pose(odom, tel.yaw)

    checkpoint_bearing, rng = geodesic_bearing_range(
        tel.lat, tel.lon, tel.yaw, goal[0], goal[1]
    )
    bearing = checkpoint_bearing
    route_target = None
    route_note = ""

    global _CLAIM_HOLD
    claim_now, claim_why = _CLAIM.update(rng, time.monotonic(), gps_stale=_GPS.stale,
                                         travelled_m=_CRUMBS.travelled_m)
    # Just rejected (the server's fix put us past 15 m while ours read inside):
    # keep driving toward the checkpoint and retry 2 m closer or after
    # CLAIM_RETRY_S. Retrying every tick halted the rover at the edge for good.
    hold = _CLAIM_HOLD
    if claim_now and hold is not None and hold[2] == goal_index \
            and rng > hold[0] - 2.0 and time.monotonic() - hold[1] < CLAIM_RETRY_S:
        claim_now = False
    if claim_now:
        _halt(client)
        print(f"claiming: {claim_why} (rng={rng:.1f}m)", flush=True)
        if not mission_mode:
            _event(f"claimed checkpoint ({claim_why})")
            return True
        resp = client.checkpoint_reached()
        if resp.get("_http_status", 200) >= 400:
            print(f"checkpoint-reached rejected (rng={rng:.1f}m): {resp} "
                  f"-> driving closer, retry in {CLAIM_RETRY_S:.0f} s or 2 m", flush=True)
            _CLAIM_HOLD = (rng, time.monotonic(), goal_index)
            _event(f"claim rejected by server at {rng:.1f} m; driving closer")
            return False
        _CLAIM_HOLD = None
        _event(f"claimed checkpoint ({claim_why})")
        print(f"checkpoint-reached ok: {resp}")
        return True

    if route_profile is not None and not (NAV == "fan" and FAN_SIMPLE):
        route_target = route_profile.target(goal_index, tel.lat, tel.lon)
        if route_target is not None:
            bearing, _ = geodesic_bearing_range(
                tel.lat, tel.lon, tel.yaw, route_target.lat, route_target.lon
            )
            route_note = (
                f"{route_target.describe()} "
                f"ckpt_bearing={np.degrees(checkpoint_bearing):+.0f}° "
            )

    if route_target is None and not (NAV == "fan" and FAN_SIMPLE):
        osm_t = _osm_target(tel.lat, tel.lon, goal, rng)
        if osm_t is not None:
            bearing, _ = geodesic_bearing_range(tel.lat, tel.lon, tel.yaw, osm_t[0], osm_t[1])
            route_note = (f"osm aim={np.degrees(bearing):+.0f}° "
                          f"ckpt_bearing={np.degrees(checkpoint_bearing):+.0f}° ")

    path = analyze(front) if USE_DIRT_GATE else None
    # Prefer front+rear logic; fall back to front-only.
    obs_view = None
    if USE_OBS_STOP:
        obs_view = front_blocked_rear_clear(front, rear)
        if obs_view is None:
            obs_view = analyze_obstacle(front)
    depth_view = None
    dg = _depth_gate()
    if dg is not None:
        depth_view = dg.evaluate(front, bearing)

    # Ground-relative costmap, reusing the depth map just computed. Unlike the
    # fixed-band clearance it sees low, close obstacles in the bottom of the
    # frame: curbs, rocks, ruts.
    cmap = _costmap()
    cm_res = None
    sem_note = ""
    track_frac: float | None = None
    track_bearing: float | None = None
    corr_bearing: float | None = None
    corr_width: float = 0.0
    # good_m must be initialised here: it is only bound inside the semantics
    # block but the rescore below reads it unconditionally.
    hard_m = soft_m = good_m = None
    labels = None
    # Near band that is DRIVABLE (track or verge), as against `track_frac`,
    # which is track only.
    drive_frac: float | None = None
    route_sidewalk_clear = False
    # Semantics runs INDEPENDENTLY of depth: it is the more trustworthy signal
    # here, so it must not be gated on the depth model succeeding — the
    # path-visible check is what stops the rover burying itself.
    sem = _semantics()
    fan_ground = fan_grass = None
    if sem is not None:
        seg_hw = (
            dg.last_depth.shape[:2]
            if (dg is not None and dg.last_depth is not None)
            else (120, 160)
        )
        if NAV == "fan":
            # One SegFormer pass at the free-space resolution; everything else
            # below keeps its low-res labels via nearest-neighbour sampling.
            labels_hi = sem.labels(front, FAN_SEG_HW)
            labels = None
            if labels_hi is not None:
                fan_ground, fan_grass = _fan_masks(sem, labels_hi)
                labels = _downsample_labels(sem, labels_hi, seg_hw)
        else:
            labels = sem.labels(front, seg_hw)
        if labels is not None:
            hard_m, soft_m, good_m = sem.categorize(labels)
            # Teach the response fit using the ground as a ruler: with measured
            # intrinsics the true distance to a ground pixel is pure geometry.
            # Fit on the RAW map — fitting on corrected depth compounds.
            fit = _depth_fit()
            if fit is not None and dg is not None and dg.last_raw_depth is not None:
                rd = dg.last_raw_depth
                if rd.shape == good_m.shape:
                    cs = CALIB.scaled_to(rd.shape[1], rd.shape[0])
                    rows = np.arange(rd.shape[0])[:, None]
                    fit.observe(rd, good_m & ~hard_m & (rows > cs.horizon_v()))
            t_all, t_band, t_bearing = sem.track(good_m)
            track_frac = t_band
            track_bearing = t_bearing
            corr_bearing, corr_width = sem.corridor(good_m)
            # "Where is the track" and "is there anywhere to drive at all" are
            # DIFFERENT questions on DIFFERENT masks. Track-following uses
            # `good` so the rover prefers a real path; the no-way-forward stop
            # uses `good | soft`, because soft IS drivable and every other
            # module already treats it that way.
            #
            # Measuring the stop on `good` alone is fatal on a grass site:
            # observed at sem(hard=0.00 off=0.81 track=0.07/near=0.08), the
            # rover declared no_path_visible and began escaping while sitting on
            # a perfectly drivable field.
            _, d_band, _ = sem.track(good_m | soft_m)
            drive_frac = d_band
            sem_note = (
                f"sem(hard={hard_m.mean():.2f} off={soft_m.mean():.2f} "
                f"track={t_all:.2f}/near={t_band:.2f} drive={d_band:.2f}"
                + (
                    f" @{np.degrees(t_bearing):+.0f}°) "
                    if t_bearing is not None else ") "
                )
            )
            route_sidewalk_clear = (
                route_profile is not None
                and d_band >= ROUTE_SIDEWALK_MIN_DRIVE
                and t_band >= ROUTE_SIDEWALK_MIN_TRACK
                and float(hard_m.mean()) <= ROUTE_SIDEWALK_MAX_HARD
            )
            if route_sidewalk_clear:
                sem_note += "route_sidewalk_clear "

    if cmap is not None and dg is not None and dg.last_depth is not None:
        try:
            zmap = dg.last_depth
            # Sky has no depth and nothing above the horizon is ground. Without
            # this the segmenter's mid-range sky value lands in the near field
            # and the costmap reports ~15,000 obstacle points in an empty
            # courtyard; with it, 0-628.
            if labels is not None and hard_m is not None and zmap.shape == hard_m.shape:
                cs = CALIB.scaled_to(zmap.shape[1], zmap.shape[0])
                rows = np.arange(zmap.shape[0])[:, None]
                sky = np.isin(labels, list(sem.notground)) if sem.notground else False
                zmap = np.where(
                    (rows <= cs.horizon_v()) | sky, np.nan, zmap
                )
            # `odom` tells the costmap whether its robot-frame occupancy
            # history is still valid: it is not, once we have yawed.
            cm_res = cmap.compute(
                zmap, hard_mask=hard_m, soft_mask=soft_m, odom=odom
            )
        except Exception as e:
            print(f"  costmap error: {e}", flush=True)
            cm_res = None

    # Fold this tick's geometry into the accumulated map. Fed from
    # `cm_res.occupied`, which has had the ground plane removed by RANSAC, never
    # from raw depth — a 148 mm camera sees mostly ground, and painting that as
    # obstacle would wall the rover in.
    scan = None
    if lmap is not None and cm_res is not None:
        cfg = cmap.cfg
        oy, ox = np.nonzero(cm_res.occupied)
        pts = (
            np.stack([(ox + 0.5) * cfg.cell_m,
                      (oy + 0.5) * cfg.cell_m - cfg.grid_half_width_m], axis=1)
            if len(ox) else None
        )
        # A carpet of near-field dirt — thousands of "obstacle" points inside
        # 1 m — is not a map of the world. Integrating it walls the rover in and
        # every 10 m candidate then dies on that memory: one tick had pts=7352,
        # painted 700 cells, and the explorer detoured -100 then -180 deg.
        carpet = (
            (not cm_res.ground_found)
            or cm_res.n_obstacle_points > 400
            or _rough_but_driveable(drive_frac, hard_m)
        )
        try:
            lmap.integrate(
                obstacles_robot=None if carpet else pts,
                clearance_by_bin=cm_res.clearance_by_bin,
                bin_centers=cm_res.bin_centers,
                max_free_m=DEPTH_MAX_M,
            )
            scan = lmap.clearance_polar(
                n_bins=SCAN_BINS, max_range_m=SCAN_RANGE_M
            )
        except Exception as e:
            print(f"  localmap error: {e}", flush=True)
            scan = None

    # Place node for this pose. `force` on stop events happens below, at the
    # sites that know a stop occurred.
    tgraph = _topo()
    if tgraph is not None and lmap is not None:
        tgraph.update(
            lmap.x, lmap.y, lmap.theta, lmap.travelled_m, tel.stamp,
            lat=None if gps_stale else tel.lat,
            lon=None if gps_stale else tel.lon,
            goal_range_m=rng,
        )
    # Depth hallucinates ~3 m at true contact, so distrust it when RGB says
    # closed.
    if (
        depth_view is not None
        and not depth_view.blocked
        and obs_view is not None
        and obs_view.blocked
    ):
        depth_view = type(depth_view)(
            True,
            depth_view.forward_m,
            f"depth_overridden({obs_view.reason})",
            0.0,
            depth_view.turn_rad,
        )

    # ---- steering intent, decided BEFORE the stop gates -------------------
    #
    # Hoisted above the stuck block on purpose: the gates below must know
    # whether THIS tick is an in-place manoeuvre, and that answer lives in the
    # route aim. Computing it afterwards let eight collision escapes fire
    # against a rover that had never commanded forward motion.
    pre_aim = (RouteAim(bearing=checkpoint_bearing, confidence=1.0,
                        source="gps-simple", speed_scale=1.0, corridor_width=0.0)
               if NAV == "fan" and FAN_SIMPLE else _route().align(
        goal_bearing=bearing,
        corridor_bearing=corr_bearing,
        corridor_width=corr_width,
        track_bearing=track_bearing,
        track_frac=track_frac,
        goal_trusted=not gps_stale,
    ))
    if (
        route_profile is not None
        and not (NAV == "fan" and FAN_SIMPLE)
        and abs(float(np.degrees(bearing))) >= ROUTE_TURNAROUND_DEG
    ):
        # After a false detour the rover can be looking cleanly along the
        # sidewalk but the wrong way. The generic corridor policy tolerates that
        # to 120 deg because a path may legitimately bend away from a fix; a
        # route-profile rail is stronger evidence, so turn back to it.
        pre_aim = RouteAim(
            bearing=bearing,
            confidence=0.0,
            source="turnaround",
            speed_scale=0.0,
            corridor_width=corr_width,
        )

    if NAV == "fan" and fan_ground is not None and stuck_state is not None:
        # Off-road range relief intentionally uses RAW depth.  The response fit
        # calibrates against a flat-ground assumption, which is exactly what is
        # false on the rising slopes this correction handles.  Urban/Wuhan
        # keeps the existing corrected-depth path unchanged.
        fan_depth = (dg.last_raw_depth if FAN_TERRAIN == "offroad" else dg.last_depth) \
            if dg is not None else None
        return _fan_step(client, tel, front, rng, bearing, pre_aim, fan_ground,
                         fan_grass, stuck_state, odom, sem_note, goal_index,
                         depth=(fan_depth if FAN_OVERHANG else None))

    # The explorer gets right of way over the corridor ONLY once it has
    # committed to going around something; otherwise the corridor stays the
    # default. The detour replaces `pre_aim` wholesale rather than blending, so
    # the align decision, the planner's ranking bearing and the steering
    # override all see one target — blending would leave two opinions with
    # whichever applied last winning.
    ex = _explorer()
    ex_aim = None
    if ex is not None and scan is not None and lmap is not None:
        ex_aim = ex.aim(
            scan, goal_bearing=bearing, theta=lmap.theta,
            travelled_m=lmap.travelled_m,
            blocked_score=tgraph.blocked_score if tgraph is not None else None,
        )
        if ex_aim.source == "detour":
            # A confident visible corridor is the way through. Letting the
            # explorer replace it yawed the rover toward a -160 deg map gap
            # while the log still printed corridor(1.00@-0 deg), with range
            # closing 39->33 m until that turn. U-turns from a dirt-stained map
            # are not circumnavigation.
            strong_vis = (
                pre_aim.source.startswith("corridor")
                and pre_aim.confidence >= 0.35
            )
            u_turn = abs(float(ex_aim.bearing)) > np.radians(70.0)
            if strong_vis or u_turn:
                ex.detour = None
            else:
                pre_aim = RouteAim(
                    bearing=ex_aim.bearing,
                    confidence=ex_aim.confidence,
                    source="detour",
                    speed_scale=ex_aim.speed_scale,
                    corridor_width=corr_width,
                )
    # With a corridor in view, do NOT stop and pivot toward the GPS goal: the
    # path is the only way through and may well point away from it.
    #
    # The GPS bearing here is close to meaningless — lat/lon frozen AND the
    # compass drifting -141 deg then back to -7 deg while commanded one way — so
    # the latch trips on drift and the rover pivots forever without converging.
    # Vision therefore gets right of way; pivoting for the goal is reserved for
    # when vision has nothing to say, or the goal is genuinely behind us.
    on_corridor = pre_aim.confidence > 0.0
    want_align = (
        pre_aim.source == "turnaround"
        if on_corridor
        else _aligning(stuck_state, bearing)
    )
    # One-shot return leg: after spinning, GPS still claims the goal is behind.
    # Follow the corridor we just turned onto and do not re-enter align, but
    # only for TURNAROUND_HOLD_M of driving.
    if stuck_state is not None and _turnaround_expired(stuck_state, _CRUMBS.travelled_m):
        _clear_turnaround(stuck_state, f"{TURNAROUND_HOLD_M:.0f} m driven since the spin")
    if stuck_state is not None and stuck_state.get("turnaround_done"):
        want_align = False
        if pre_aim.source == "turnaround":
            pre_aim = _route().align(
                goal_bearing=bearing,
                corridor_bearing=corr_bearing,
                corridor_width=corr_width,
                track_bearing=track_bearing,
                track_frac=track_frac,
                goal_trusted=False,
            )
    elif (
        stuck_state is not None
        and want_align
        and (
            pre_aim.source == "turnaround"
            or abs(float(np.degrees(bearing))) >= 90.0
        )
    ):
        retreat = None
        if USE_ODOM:
            rb = _CRUMBS.retreat_bearing(max_m=3.0, min_extent_m=1.0)
            if rb is not None:
                retreat = float(np.degrees(rb))
        _latch_return_turn(stuck_state, bearing, retreat)
    if on_corridor and stuck_state is not None and not want_align:
        stuck_state["align_latch"] = False
    pivot = stuck_state.get("pivot_ctl") if stuck_state is not None else None
    # A burst in flight is never preempted by a fresh (stale) bearing.
    pivot_busy = pivot is not None and pivot.busy()

    if stuck_state is not None:
        _escape_budget_ok(stuck_state, rng)
        if stuck_state.get("rev_left", 0) > 0:
            stuck_state["rev_left"] -= 1
            rev_ang = 0.10 if float(np.degrees(bearing)) >= 0.0 else -0.10
            if path is not None and abs(path.center_offset) > 0.15:
                rev_ang = -0.10 * np.sign(path.center_offset)
            lin, ang = -0.22, float(rev_ang)
            extra = (
                f"dirt={path.dirt_frac:.2f} grass={path.grass_frac:.2f} "
                if path else ""
            )
            print(
                f"  REVERSE rng={rng:.1f}m {extra}"
                f"cmd lin={lin:.2f} ang={ang:.2f} "
                f"({stuck_state['rev_left']} left)",
                flush=True,
            )
            _cmd(client, lin, ang, hold_s=REVERSE_HOLD_S)
            if stuck_state["rev_left"] <= 0:
                _halt(client)
                stuck_state["key"] = None
                stuck_state["n"] = 0
                if stuck_state.pop("scan_after_reverse", False):
                    stuck_state["scan_left"] = SCAN_STEPS
                    stuck_state["scan_best"] = (-1.0, 0.0)
                    stuck_state["scan_turned"] = 0.0
                    print(
                        "reverse done — scanning for a drivable heading "
                        f"(up to {SCAN_STEPS}x{SCAN_STEP_DEG:.0f}°)",
                        flush=True,
                    )
                else:
                    print("reverse done — retrying forward", flush=True)
            return False

        if stuck_state.get("pivot_left", 0) > 0:
            stuck_state["pivot_left"] -= 1
            ang = float(np.clip(0.12 * np.sign(bearing or 1e-3), -0.14, 0.14))
            print(
                f"  PIVOT rng={rng:.1f}m bearing={np.degrees(bearing):+.0f}° "
                f"ang={ang:.2f} ({stuck_state['pivot_left']} left)",
                flush=True,
            )
            _cmd(client, 0.0, ang)
            if stuck_state["pivot_left"] <= 0:
                _halt(client)
                stuck_state["force_forward"] = 25
                print("pivot done — retrying forward", flush=True)
            return False

        # Hard stop: depth clearance OR classical contact.
        hit = False
        why = ""
        # An IN-PLACE manoeuvre cannot drive into anything, and every
        # forward-looking gate below is unreliable during one: the camera sits
        # 148 mm up, so yawing sweeps it across the ground and the depth gate
        # reads the ground itself as contact (1.20 m -> 0.05 m while the costmap
        # held 3.50 m clear with ZERO obstacle points). Eight escapes fired that
        # way in one run without the rover ever moving.
        #
        # Keyed on what THIS tick commands, not the last one. Gating on "we were
        # moving" leaves the guard off for the first tick that rolls forward,
        # which is exactly the tick that drives at the step.
        maneuvering = (
            stuck_state.get("scan_left", 0) > 0
            or stuck_state.get("detour_deg") is not None
            or pivot_busy
            or want_align
        )
        if drive_frac is not None and drive_frac < TRACK_MIN and not maneuvering:
            why = f"no_path_visible(drive_near={drive_frac:.3f})"
            # Reversing alone puts us back where we were pointing the same way,
            # so the next forward run repeats the mistake. Scan afterwards.
            stuck_state["scan_after_reverse"] = True
            # If the rear camera can see the track, we drove in off a path that
            # is still there, and reversing onto it is the reliable escape.
            rear_frac = _rear_track_frac(rear)
            if rear_frac is not None:
                why += f" rear_track={rear_frac:.2f}"
            hit = True
        elif (
            cm_res is not None
            and cm_res.forward_clearance_m < COSTMAP_STOP_M
            and not maneuvering
            and not _rough_but_driveable(drive_frac, hard_m)
            and not (
                route_sidewalk_clear
                and cm_res.forward_clearance_m >= ROUTE_COSTMAP_HARD_STOP_M
            )
        ):
            # `not maneuvering` matters as much here as above: an escape sets
            # detour_deg and takes several ticks to turn, but the obstacle stays
            # in front the whole time, so without it this re-fired every tick
            # and burned the whole escape budget inside ~1 second.
            hit, why = True, (
                f"costmap_stop({cm_res.forward_clearance_m:.2f}m "
                f"pts={cm_res.n_obstacle_points})"
            )
        elif (
            depth_view is not None
            and depth_view.blocked
            and not maneuvering
            and (
                cm_res is None
                # The costmap only outranks depth while it can SEE. A step edge
                # presents almost nothing above the fitted plane, so RANSAC
                # reported pts=0 and fwd=3.50 m while the rover drove into it.
                or not cm_res.ground_found
                # No reading this low is ignorable, whatever the costmap thinks
                # (0.06 m observed immediately before impact). Deliberately NOT
                # corroborated against the costmap: the step crash read
                # depth_stop(0.79 -> 0.06 m) against cost(fwd=3.50m pts=0) with
                # ground_found TRUE, so requiring agreement would disable this
                # guard on exactly the failure that motivated it. The false
                # depth_stop during a pivot is handled by `maneuvering` above.
                or depth_view.forward_m < DEPTH_HARD_STOP_M
            )
            and not (
                route_sidewalk_clear
                and cm_res is not None
                and cm_res.forward_clearance_m >= ROUTE_COSTMAP_HARD_STOP_M
                and depth_view.forward_m >= ROUTE_DEPTH_HARD_STOP_M
            )
            # The camera sits ~15 cm up, so dirt/roots under the lens read
            # 0.04-0.36 m. That is ground, not a wall, when semantics says the
            # near band is drivable and nothing hard is in frame.
            and not _rough_but_driveable(drive_frac, hard_m)
        ):
            # Above the hard-stop floor this is only a fallback for a costmap
            # that is unavailable or blind: the depth gate scores a fixed image
            # band with no ground-plane removal, so once depth is correctly
            # scaled it reads the ground itself as an obstacle (depth_stop(0.17m)
            # on open pavement, i.e. the ground under a 0.148 m camera).
            hit, why = True, depth_view.reason
        elif obs_view is not None and obs_view.blocked and not maneuvering:
            # Guarded like the rest: without it a lamp-on-wall reading could
            # fire an escape mid-pivot.
            hit, why = True, (
                f"obstacle ahead ({obs_view.reason} "
                f"dark={obs_view.dark_frac:.2f} bright={obs_view.bright_frac:.2f})"
            )
        if hit:
            _begin_escape(stuck_state, why)
            return False

        if path is not None:
            force = stuck_state.get("force_forward", 0)
            if force > 0:
                stuck_state["force_forward"] = force - 1
            else:
                if path.blocked and path.grass_frac > 0.25:
                    _begin_escape(stuck_state, "grass/no dirt corridor in view")
                    return False
                if path.uncertain or (path.blocked and path.grass_frac <= 0.25):
                    if stuck_state.get("pivot_attempts", 0) >= 4:
                        _begin_escape(
                            stuck_state,
                            "still dim/uncertain after pivots",
                        )
                    else:
                        _begin_pivot(
                            stuck_state,
                            f"dim/uncertain road (dirt={path.dirt_frac:.2f} "
                            f"grass={path.grass_frac:.2f} "
                            f"bright={path.mean_bright:.0f})",
                        )
                    return False

        key = (round(rng, 1), round(float(np.degrees(bearing)), 0))
        if stuck_state.get("key") == key:
            stuck_state["n"] = stuck_state.get("n", 0) + 1
        else:
            stuck_state["key"] = key
            stuck_state["n"] = 0
        if stuck_state["n"] >= 40:
            _begin_escape(stuck_state, f"rng/bearing frozen at {key}", key)
            return False

        # Range not improving while we keep commanding = jammed. An in-place
        # pivot holds range constant BY DESIGN, so it must not feed the jam
        # counter or aligning always self-trips.
        aligning = _aligning(stuck_state, bearing)
        if stuck_state.get("turnaround_remain") is not None and not stuck_state.get(
            "turnaround_done"
        ):
            # Open-loop return spin: live GPS error is not a progress signal.
            aligning = False
        if aligning:
            # Bounded: if the pivot is not closing the error the rover cannot
            # yaw here, so escape rather than spin forever.
            err = abs(float(np.degrees(bearing)))
            best = stuck_state.get("align_best")
            if best is None or err < best - 3.0:
                stuck_state["align_best"] = err
                stuck_state["align_n"] = 0
            else:
                stuck_state["align_n"] = stuck_state.get("align_n", 0) + 1
            # Pulse-and-settle improves the error once per ~3 s burst cycle, so
            # this counts far more ticks than a per-tick loop would.
            if stuck_state["align_n"] >= 150:
                stuck_state["align_best"] = None
                stuck_state["align_n"] = 0
                _begin_escape(
                    stuck_state, f"pivot not converging (bearing {err:.0f}°)"
                )
                return False
            stuck_state["prog_t0"] = None
        else:
            stuck_state["align_best"] = None
            stuck_state["align_n"] = 0
            # Progress over a TIME window, not per tick: at ~1.5 Hz and a slow
            # creep the per-tick delta is a few cm and always looks frozen.
            now = time.monotonic()
            base_rng = stuck_state.get("prog_rng")
            if stuck_state.get("prog_t0") is None or base_rng is None:
                stuck_state["prog_t0"] = now
                stuck_state["prog_rng"] = rng
            elif rng < base_rng - FREEZE_M:
                stuck_state["prog_t0"] = now
                stuck_state["prog_rng"] = rng
            elif now - stuck_state["prog_t0"] >= FREEZE_SEC:
                held = now - stuck_state["prog_t0"]
                stuck_state["prog_t0"] = None
                stuck_state["prog_rng"] = None
                _begin_escape(
                    stuck_state,
                    f"no progress: {base_rng - rng:+.1f}m in {held:.0f}s "
                    f"(need {FREEZE_M}m)",
                )
                return False

    # The planner RANKS candidates by agreement with `goal_bearing`, so a dead
    # GPS bearing makes its ranking worse than chance: measured at 0.494 m ADE
    # for top-1 against 0.320 m for a uniformly RANDOM candidate, and fusion
    # then dragged the VLM's -41% advantage down to -4% by mixing it back in.
    # So when GPS is untrustworthy the planner sees the route-aligned bearing,
    # which is ranked on something real.
    planner_bearing = pre_aim.bearing if (gps_stale or pre_aim.confidence > 0) else bearing
    obs = Observation(
        front=front, rear=rear, lat=tel.lat, lon=tel.lon,
        yaw=tel.yaw, stamp=tel.stamp,
        goal_bearing=planner_bearing, goal_range_m=rng,
    )

    # --- recovery scan: rotate in steps, stop at the first drivable heading ---
    if (
        stuck_state is not None
        and stuck_state.get("scan_left", 0) > 0
        and not pivot_busy
        and stuck_state.get("detour_deg") is None
    ):
        # Drivable, not track-only: the scan looks for somewhere to GO, and on
        # grass a track-only measure reads 0 in every direction and rejects
        # seven perfectly drivable headings.
        seen = float(drive_frac) if drive_frac is not None else 0.0
        best_frac, best_turn = stuck_state.get("scan_best", (-1.0, 0.0))
        turned = float(stuck_state.get("scan_turned", 0.0))
        if seen > best_frac:
            stuck_state["scan_best"] = (seen, turned)
            best_frac, best_turn = seen, turned
        if seen >= SCAN_RESUME:
            print(
                f"  SCAN: drivable heading found after {turned:+.0f}° "
                f"(drive_near={seen:.2f}) — resuming",
                flush=True,
            )
            stuck_state["scan_left"] = 0
            # Finding the heading we already face is not progress: resetting
            # the budget there loops 180 deg scans forever.
            if abs(turned) > 15.0:
                stuck_state["rev_attempts"] = 0
        else:
            left = stuck_state["scan_left"] - 1
            stuck_state["scan_left"] = left
            if left <= 0:
                # Nothing clear anywhere: go back to the best heading seen.
                back = best_turn - turned
                print(
                    f"  SCAN: swept {turned:+.0f}°, best track_near="
                    f"{best_frac:.2f} at {best_turn:+.0f}° — returning there",
                    flush=True,
                )
                if abs(back) > 10.0:
                    stuck_state["detour_deg"] = back
                    stuck_state.pop("detour_started", None)
                stuck_state["rev_attempts"] = 0
            else:
                # Alternate sides, widening: +50, -50, +100, -100, ... so a path
                # just behind-left does not cost a near-full circle.
                k = SCAN_STEPS - left            # 1, 2, 3, ...
                mag = SCAN_STEP_DEG * ((k + 1) // 2)
                target = mag if (k % 2 == 1) else -mag
                target = max(-180.0, min(180.0, target))
                rel = target - turned
                stuck_state["scan_turned"] = target
                stuck_state["detour_deg"] = rel
                stuck_state.pop("detour_started", None)
                print(
                    f"  SCAN: drive_near={seen:.2f} at {turned:+.0f}° — "
                    f"turning {rel:+.0f}° to {target:+.0f}° ({left} left)",
                    flush=True,
                )

    detour = stuck_state.get("detour_deg") if stuck_state is not None else None
    if (
        detour is not None
        and pivot is not None
        and stuck_state.get("detour_started")
        and pivot.cycles > stuck_state.get("detour_cycles", -1)
    ):
        # One burst+settle done: drop the target and re-perceive rather than
        # grinding at a fixed angle that never decreases.
        stuck_state.pop("detour_deg", None)
        stuck_state.pop("detour_started", None)
        pivot.reset()
        pivot_busy = False
        detour = None

    # `on_corridor` / `want_align` decided above the stop gates.
    if STAGE != 0 and (pivot_busy or detour is not None or want_align):
        if stuck_state is not None:
            stuck_state["align_latch"] = True
        if pivot is None:
            pivot = PulsePivot(
                ang=ALIGN_ANG, settle_s=SETTLE_S, tol_deg=ALIGN_EXIT,
                exact_bursts=_STREAMER is not None and STREAM_CMDS,
            )
            if stuck_state is not None:
                stuck_state["pivot_ctl"] = pivot
        if detour is not None:
            err_deg = float(detour)
            if not stuck_state.get("detour_started"):
                stuck_state["detour_started"] = True
                stuck_state["detour_cycles"] = pivot.cycles
        elif stuck_state is not None and stuck_state.get("turnaround_remain") is not None:
            err_deg = float(stuck_state["turnaround_remain"])
            pivot.learn = False
        else:
            err_deg = float(np.degrees(bearing))
        prev_cycles = (
            int(stuck_state.get("turnaround_seen_cycles", 0))
            if stuck_state is not None
            else 0
        )
        lin, ang, note = pivot.command(err_deg, tel, time.monotonic())
        if (
            stuck_state is not None
            and stuck_state.get("turnaround_remain") is not None
            and not stuck_state.get("turnaround_done")
            and pivot.cycles > prev_cycles
        ):
            stuck_state["turnaround_seen_cycles"] = pivot.cycles
            if _credit_return_burst(stuck_state, getattr(pivot, "_actual", 0.0)):
                stuck_state["align_latch"] = False
                pivot.learn = True
                pivot.reset()
                _halt(client)
                print(
                    f"  turnaround done after {stuck_state.get('turnaround_bursts', 0)} "
                    f"bursts — driving the corridor",
                    flush=True,
                )
                return False
        if note == "aligned":
            # Release the latch here rather than at ALIGN_EXIT: the pivot cannot
            # resolve finer, and holding on deadlocks at zero command.
            #
            # HALT and hand off on the NEXT tick. `want_align` was true when the
            # stop gates ran, so they were all suppressed as an in-place
            # manoeuvre; falling through to the controller here would issue the
            # first forward command with every collision gate skipped, which is
            # the tick that drives at the step.
            if stuck_state is not None:
                stuck_state["align_latch"] = False
                stuck_state.pop("detour_deg", None)
                stuck_state.pop("detour_started", None)
                if stuck_state.get("turnaround_remain") is not None:
                    stuck_state["turnaround_done"] = True
                    stuck_state["turnaround_remain"] = 0.0
                    pivot.learn = True
            pivot.reset()
            _halt(client)
            return False
        else:
            age = f" age={tel.age_s:.1f}s" if tel.age_s is not None else ""
            print(
                f"  rng={rng:.1f}m bearing={np.degrees(bearing):+.0f}°{age} "
                f"src=pivot[{note}] cmd lin={lin:.2f} ang={ang:.2f}",
                flush=True,
            )
            _publish_overlay(front, lines=(
                f"PIVOT {note}",
                f"rng={rng:.1f}m bearing={np.degrees(bearing):+.0f}deg "
                f"target={err_deg:+.0f}deg cmd=({lin:+.2f},{ang:+.2f})",
            ))
            # A burst ends when PulsePivot planned it to, not when the next slow
            # tick notices: the streamer holds it for exactly the remainder.
            _cmd(client, lin, ang, hold_s=pivot.burst_remaining(time.monotonic()))
            return False
    if stuck_state is not None and stuck_state.get("pivot_ctl") is not None:
        stuck_state["pivot_ctl"].reset()

    cands = planner.propose(obs, K)
    # Re-rank with perception: candidates do not exist until propose() runs, so
    # traversability can only enter as a second pass. As a post-hoc veto only,
    # the ranking scored WORSE THAN RANDOM against human-driven rides (1.029 m
    # vs 0.757 m median ADE); feeding it into the score cut that to 0.775 m.
    if cands and good_m is not None and hard_m is not None and hasattr(planner, 'rescore'):
        try:
            tf = path_traversable_fractions(cands, good_m & ~hard_m, CALIB)
            cands = planner.rescore(cands, obs, trav_fracs=tf)
        except Exception as e:
            print(f'  rescore skipped: {e}', flush=True)
    if path is not None and STAGE <= 1:
        cands = rescore(cands, path)

    # Hard veto AFTER proposal: drop any candidate crossing occupied space.
    # Runs at every stage and does NOT depend on USE_DIRT_GATE.
    veto_note = ""
    if cm_res is not None and cmap is not None and cands:
        keep = cmap.veto(cm_res, cands)
        survivors = [c for c, ok in zip(cands, keep) if ok]
        n_vetoed = len(cands) - len(survivors)
        if survivors:
            cands = survivors
            if n_vetoed:
                veto_note = f"veto={n_vetoed}/{len(keep)} "
            # Prefer the track over the verge. A preference, not a rule, so the
            # rover can still leave the trail to get around something.
            if OFFTRACK_PENALTY > 0.0:
                pen = cmap.offtrack_cost(cm_res, cands)
                if pen.any():
                    cands = [
                        type(c)(
                            waypoints=c.waypoints,
                            score=float(c.score - OFFTRACK_PENALTY * p),
                            stamp=c.stamp,
                        )
                        for c, p in zip(cands, pen)
                    ]
                    veto_note += f"offtrack={pen.max():.2f} "
        elif stuck_state is not None:
            if (
                route_sidewalk_clear
                and cm_res.forward_clearance_m >= ROUTE_COSTMAP_HARD_STOP_M
            ):
                veto_note = (
                    f"route_keep_costmap_veto={len(keep)}/{len(keep)} "
                    f"fwd={cm_res.forward_clearance_m:.2f}m "
                )
            # Everything ahead is blocked. With room left, TURN toward the most
            # open bearing and re-perceive — reversing is for a genuinely
            # trapped rover, not a wall 3 m away.
            #
            # Exception: max range with a flat histogram means the 10 m arcs hit
            # smeared memory, not a wall. argmax then picks the FOV edge and the
            # rover spins in place (three DETOUR -53 deg in a row, pts=0).
            elif cm_res.forward_clearance_m > COSTMAP_STOP_M:
                clr = np.asarray(cm_res.clearance_by_bin, float)
                uniform = float(clr.max() - clr.min()) < 0.40
                if uniform or _rough_but_driveable(drive_frac, hard_m):
                    veto_note = (
                        f"veto_keep_all={len(keep)} "
                        f"fwd={cm_res.forward_clearance_m:.2f}m "
                    )
                else:
                    bi = int(np.argmax(clr))
                    open_deg = float(np.degrees(cm_res.bin_centers[bi]))
                    stuck_state["detour_deg"] = open_deg
                    stuck_state.pop("detour_started", None)
                    print(
                        f"  DETOUR: all {len(keep)} candidates vetoed "
                        f"(fwd {cm_res.forward_clearance_m:.2f}m) — turning "
                        f"{open_deg:+.0f}° toward open space "
                        f"({clr[bi]:.1f}m)",
                        flush=True,
                    )
                    _halt(client)
                    return False
            elif _rough_but_driveable(drive_frac, hard_m):
                veto_note = (
                    f"veto_keep_all={len(keep)} "
                    f"fwd={cm_res.forward_clearance_m:.2f}m "
                )
            else:
                _begin_escape(
                    stuck_state,
                    f"all {len(keep)} candidates vetoed by costmap "
                    f"(fwd {cm_res.forward_clearance_m:.2f}m)",
                )
                return False
        else:
            _halt(client)
            print("  BLOCKED: every candidate vetoed by costmap", flush=True)
            return False
    # Reachable only under USE_DIRT_GATE=1, since `path` is None otherwise.
    # The costmap veto above is deliberately not gated on it.
    if gate is not None and path is not None:
        m = gate.mask(obs, cands)
        safe = [c for c, ok in zip(cands, m) if ok]
        if not safe:
            if stuck_state is not None:
                if path.grass_frac > 0.25:
                    _begin_escape(stuck_state, "all arcs vetoed (grass)")
                elif stuck_state.get("force_forward", 0) > 0:
                    safe = cands[:1]
                elif stuck_state.get("pivot_attempts", 0) >= 4:
                    _begin_escape(
                        stuck_state, "arcs vetoed after pivots (dim)"
                    )
                else:
                    _begin_pivot(
                        stuck_state,
                        "all arcs vetoed but little grass — search corridor",
                    )
            if not safe:
                if stuck_state is None:
                    _halt(client)
                    print(
                        f"  BLOCKED rng={rng:.1f}m dirt={path.dirt_frac:.2f} "
                        f"grass={path.grass_frac:.2f} — holding",
                        flush=True,
                    )
                return False
        cands = safe

    chosen_i = None
    if STAGE == 0 and hasattr(controller, "command_bearing") and path is not None:
        path_bearing = path.center_offset * (35.0 * np.pi / 180.0)
        fused = (1.0 - PATH_BLEND) * bearing + PATH_BLEND * path_bearing
        lin, ang = controller.command_bearing(fused, rng, blocked=path.blocked)
        best = max(cands, key=lambda c: c.score)
        end_y = float(best.waypoints[-1, 1])
        ang = float(
            np.clip(ang + 0.04 * np.sign(end_y) * min(abs(end_y), 1.0), -1.0, 1.0)
        )
        src = "creep"
    elif STAGE == 0 and abs(np.degrees(bearing)) > ALIGN_DEG:
        # STAGE 0 ONLY. Unconditional, this discarded RouteAligner's whole
        # policy: on a clear sidewalk with cost(fwd=3.50m pts=0) and the path at
        # +10 deg, the rover stopped dead and span toward a GPS bearing 97 deg
        # off. For STAGE >= 1 the `want_align` block above owns aligning, via
        # PulsePivot.
        lin, ang = _align_cmd(bearing)
        src = "align"
    else:
        sel = _selector() or selector
        if sel is not None:
            # Gate on scene difficulty. On an empty pathway the planner is
            # already competent and a 1.6-2.1 s query only costs tick rate.
            idx = sel.select(
                obs, cands, progress_m=_CRUMBS.travelled_m,
                hazard_frac=float(hard_m.mean()) if hard_m is not None else 0.0,
                corridor_width=corr_width,
            )
        else:
            idx = int(np.argmax([c.score for c in cands]))
        # The semantic veto applies AFTER selection and falls back to the
        # best-scoring traversable candidate. Geometry catches solid things;
        # this catches surface distinctions it cannot make.
        tv = _trav_veto()
        if tv is not None:
            new_idx, tv_note = tv.apply(obs, cands, idx)
            if new_idx != idx or tv_note != "ok":
                veto_note += f"trav[{tv_note}] "
            # "every candidate is untraversable" IS "no way forward" and must
            # stop the rover, not decorate the log line. It did not: driving
            # into a raised step this reported all_untraversable for ~10 ticks
            # while depth fell 1.20 -> 0.06 m and the rover kept commanding
            # lin=0.20, because the verdict only reached a display string.
            if (
                tv_note.startswith("all_untraversable")
                and route_sidewalk_clear
            ):
                veto_note += "route_keep_trav_all "
            elif tv_note.startswith("all_untraversable") and stuck_state is not None:
                # One frame is not enough to turn around on. Mission-3: a clean
                # corridor (drive=0.85, hard=0) went 12/12 vetoed for one tick
                # and the escape swung the rover away from CP2. Stop and look
                # again; a real wall is still there next tick.
                n_all = int(stuck_state.get("trav_all_n", 0)) + 1
                stuck_state["trav_all_n"] = n_all
                if n_all < TRAV_ALL_TICKS:
                    _halt(client)
                    print(f"  traversability: {tv_note} — holding for a second "
                          f"look ({n_all}/{TRAV_ALL_TICKS})", flush=True)
                    return False
                stuck_state["trav_all_n"] = 0
                _begin_escape(stuck_state, f"traversability: {tv_note}")
                return False
            if stuck_state is not None:
                stuck_state["trav_all_n"] = 0
            idx = new_idx
        chosen = cands[idx]
        chosen_i = idx
        if CAND_VIZ_DIR and cands:
            n = 0
            if stuck_state is not None:
                n = int(stuck_state.get("_viz_n", 0)) + 1
                stuck_state["_viz_n"] = n
            if n == 0 or n % max(CAND_VIZ_EVERY, 1) == 0:
                try:
                    dump_candidates(
                        CAND_VIZ_DIR, front, cands, CALIB, chosen_idx=idx,
                        extra={"bearing_deg": float(np.degrees(bearing)),
                               "rng_m": float(rng)},
                    )
                except Exception:
                    pass
        wp = chosen.waypoints
        if posebuf is not None:
            # Pose is integrated from VO at the top of the tick, so this does
            # not degenerate to a pure rotation.
            wp = posebuf.realign(wp, chosen.stamp, tel.stamp)
        # Hold the chosen path across ticks, advanced by measured motion, so a
        # single bad perception tick cannot swing the command.
        if USE_COMMIT:
            _COMMIT.advance(odom, 1.0 / CTRL_HZ)
            wp = _COMMIT.offer(wp)
        # Depth safety: override the first waypoint if clearance is bad.
        if depth_view is not None and depth_view.turn_rad is not None and not depth_view.blocked:
            scale = max(0.25, float(depth_view.linear_scale))
            wp = wp.copy()
            wp[0, 0] = max(0.15, abs(np.cos(depth_view.turn_rad)) * scale)
            wp[0, 1] = np.sin(depth_view.turn_rad) * scale
        lin, ang = controller.command(wp)
        if path is not None and STAGE <= 1:
            ang = float(np.clip(ang + 0.10 * path.center_offset, -1.0, 1.0))
        # Reuses the `pre_aim` computed above the stop gates rather than
        # calling align() again with identical arguments: one aim per tick means
        # the gates, the pivot decision and the steering command can never
        # disagree about where the rover is trying to go.
        aim = pre_aim
        steer_bearing = aim.bearing
        ang = float(np.clip(ang + _steer_nudge(stuck_state, steer_bearing), -1.0, 1.0))

        # Corridor OVERRIDE: when a drivable corridor is visible, it replaces
        # the goal-seeking angular command rather than nudging it. The goal
        # only biases the aim point WITHIN the corridor.
        if aim.source.startswith("corridor") or aim.source == "detour":
            # A committed detour steers like a corridor and for the same reason:
            # it is a CONSTRAINT, not a hint, so it replaces the goal-seeking
            # yaw rather than nudging it. `_steer_nudge` caps at STEER_MAX_ANG
            # and could not turn the rover 60 deg around an obstacle in any
            # useful distance. RouteAligner already applied the bounded goal
            # bias; this only converts the aim into a yaw command.
            aim_rad = aim.bearing
            pivot_ref = stuck_state.get("pivot_ctl") if stuck_state else None
            rate_pu = (
                pivot_ref.rate / pivot_ref.ang
                if pivot_ref is not None and pivot_ref.calibrated
                and pivot_ref.ang > 1e-6
                else STEER_RATE_PRIOR
            )
            desired_dps = float(
                np.clip(STEER_KP * np.degrees(aim_rad), -3.0 * STEER_MAX_DPS,
                        3.0 * STEER_MAX_DPS)
            )
            ang = float(
                np.clip(
                    desired_dps / max(rate_pu, 10.0),
                    -CORRIDOR_MAX_ANG, CORRIDOR_MAX_ANG,
                )
            )
            # Narrow corridor => slow down. RouteAim decided by how much.
            lin = float(lin * aim.speed_scale)
            shown = aim.bearing
            src = (
                f"{aim.source}({corr_width:.2f}"
                f"@{np.degrees(shown):+.0f}°)"
            )
        # Same reasoning as the stop above: with no ground-plane removal and
        # correctly-scaled depth the gate reads the ground and returns
        # linear_scale=0.0, zeroing the throttle on open pavement. Let it slow
        # the rover only when the costmap is unavailable.
        if cm_res is None and depth_view is not None and depth_view.linear_scale < 1.0:
            lin = float(lin * depth_view.linear_scale)
        # Decelerate into COSTMAP_STOP_M, so the stop is reached at a speed the
        # dead time can absorb.
        if cm_res is not None and cm_res.forward_clearance_m < COSTMAP_SLOW_M:
            span = max(COSTMAP_SLOW_M - COSTMAP_STOP_M, 1e-3)
            t = (cm_res.forward_clearance_m - COSTMAP_STOP_M) / span
            lin = float(lin * float(np.clip(t, 0.25, 1.0)))
        if not (aim.source.startswith("corridor") or aim.source == "detour"):
            src = "logonav" if chosen.score > 0 else "arc"

    if path is not None:
        tag = " unc" if path.uncertain else ""
        extra = (
            f"dirt={path.dirt_frac:.2f} grass={path.grass_frac:.2f} "
            f"bright={path.mean_bright:.0f}{tag} off={path.center_offset:+.2f} "
        )
    else:
        extra = "gate=off "
    if route_note:
        extra += route_note
    if odom is not None:
        extra += (
            f"odo({odom.distance_m*100:.1f}cm {np.degrees(odom.dyaw):+.0f}° "
            f"c={odom.confidence:.2f}{'' if odom.reason == 'ok' else ' ' + odom.reason}) "
            f"trail={len(_CRUMBS.pts)}@{_CRUMBS.travelled_m:.1f}m "
        )
    if _selector() is not None:
        extra += f"{_selector().describe()} "
    if _GPS.stale:
        extra += f"{_GPS.describe()} "
    if obs_view is not None:
        extra += f"obs={obs_view.reason} "
    if depth_view is not None:
        extra += f"{depth_view.reason} "
    if cm_res is not None:
        g = "" if cm_res.ground_found else "!noground "
        extra += (
            f"cost(fwd={cm_res.forward_clearance_m:.2f}m "
            f"pts={cm_res.n_obstacle_points}) {g}{sem_note}{veto_note}"
        )
    if lmap is not None:
        extra += f"{lmap.describe()} "
    if tgraph is not None:
        extra += f"{tgraph.describe()} "
    if ex_aim is not None:
        extra += f"{ex_aim.describe()} "
    print(
        f"  rng={rng:.1f}m bearing={np.degrees(bearing):+.0f}° "
        f"{extra}src={src} cmd lin={lin:.2f} ang={ang:.2f}",
        flush=True,
    )
    _publish_overlay(front, cands, chosen_i, lines=(
        f"{src}  cmd=({lin:+.2f},{ang:+.2f})  rng={rng:.1f}m  "
        f"bearing={np.degrees(bearing):+.0f}deg",
        (f"cost fwd={cm_res.forward_clearance_m:.2f}m pts={cm_res.n_obstacle_points}  "
         if cm_res is not None else "") + f"{sem_note}{veto_note}",
        f"{_GPS.describe()}  "
        + (f"odo {odom.reason} trail {_CRUMBS.travelled_m:.1f}m" if odom is not None else "")
        + f"   goal {goal_index + 1}",
    ), good_m=good_m, soft_m=soft_m, hard_m=hard_m)
    if lin <= 0.0 and abs(ang) < 1e-3:
        _halt(client)
        return False
    _cmd(client, lin, ang)
    return False


def _sdk_recover(client, mission_mode: bool) -> None:
    """
    No frame and no telemetry for SDK_DEAD_S: bring the SDK back or stop.
    The rover is already stationary (command expiry), so this may take its time.
    """
    global _LAST_ALIVE
    _halt(client)
    print(f"[SDK] no camera frame or telemetry for {SDK_DEAD_S:.0f} s — recovering",
          flush=True)
    if _GUARD is None or not SDK_MANAGE:
        print("[SDK] SDK_MANAGE=0 — waiting for it to come back by itself", flush=True)
        _LAST_ALIVE = time.monotonic()
        return
    saved = client.timeout
    client.timeout = 60.0          # start-mission can relaunch Chrome
    try:
        while not _GUARD.recover(client, mission_mode):
            if _GUARD.restarts >= SDK_MAX_RESTARTS:
                print(f"[SDK] gave up after {_GUARD.restarts} restarts — stopping",
                      flush=True)
                raise SystemExit(4)
    finally:
        client.timeout = saved
    _LAST_ALIVE = time.monotonic()


def _scanned_through(client) -> int | None:
    """latest_scanned_checkpoint, or None if it cannot be read."""
    try:
        _, latest = client.get_checkpoints()
        return int(latest)
    except Exception:
        return None


def _sigterm(*_):
    # `kill` must stop the rover through main()'s finally, not end Python with
    # the last command still streaming.
    raise KeyboardInterrupt


def _lan_ip() -> str | None:
    """This Mac's LAN address (UDP connect sends nothing), for a phone/laptop link."""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def main():
    global _STREAMER, _OVERLAY_SRV, _GUARD, _RUN_DIR
    sdk_url = os.environ.get("SDK_URL", "http://localhost:8000")
    client = RoverClient(sdk_url)
    planner, selector, gate, posebuf, controller = build_stack()
    ready = getattr(planner, "ready", None)
    print(
        f"STAGE={STAGE} planner={type(planner).__name__} "
        f"ready={ready} gate={type(gate).__name__ if gate else 'off'} "
        f"USE_DIRT_GATE={USE_DIRT_GATE} USE_OBS_STOP={USE_OBS_STOP} "
        f"USE_DEPTH={USE_DEPTH} LAMP={'on' if LAMP else 'off'} "
        f"SEND_CONTROL={SEND_CONTROL} STREAM_CMDS={STREAM_CMDS} SDK_MANAGE={SDK_MANAGE}"
    )
    _GUARD = SdkGuard(sdk_url, manage=SDK_MANAGE)
    if SDK_MANAGE:
        print(f"[SDK] status at start: {_GUARD.ensure_healthy()}", flush=True)
    wait_for_sdk(client)
    if STREAM_CMDS:
        _STREAMER = CommandStreamer(client, send=SEND_CONTROL, hz=STREAM_HZ, lamp=LAMP)
        _STREAMER.halt()
        _STREAMER.start()
    else:
        _halt(client)
    if OVERLAY:
        try:
            from erc_indoor.overlay import OverlayServer

            port = OVERLAY_PORT
            for port in range(OVERLAY_PORT, OVERLAY_PORT + 6):
                try:
                    # A second mission in the same process keeps the first server.
                    _OVERLAY_SRV = _OVERLAY_SRV or OverlayServer(port)
                    break
                except OSError:          # port taken (a stale run): try the next
                    continue
            if _OVERLAY_SRV is None:
                raise OSError(f"ports {OVERLAY_PORT}-{OVERLAY_PORT + 5} all taken")
            port = getattr(getattr(_OVERLAY_SRV, "httpd", None), "server_address", (0, port))[1]
            ip = _lan_ip()
            print(f"LIVE VIEW  http://localhost:{port}/stream.mjpg"
                  + (f"   (same network: http://{ip}:{port}/stream.mjpg)" if ip else "")
                  + f"   still frame: /latest.jpg", flush=True)
            if OVERLAY_SAVE_EVERY > 0:
                _RUN_DIR = os.path.join("runs", "urban", time.strftime("run_%Y%m%d_%H%M%S"))
                os.makedirs(os.path.join(_RUN_DIR, "overlay"), exist_ok=True)
                os.makedirs(os.path.join(_RUN_DIR, "frames"), exist_ok=True)
                print(f"overlays + raw frames saved to {_RUN_DIR}/ (every {OVERLAY_SAVE_EVERY} ticks)")
        except OSError as e:
            print(f"overlay server not started ({e})")
    signal.signal(signal.SIGTERM, _sigterm)
    # A background launch inherits SIGINT as IGNORED, so Ctrl-C / kill -INT
    # would do nothing while the rover drives. Restore Python's handler.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        _drive(client, planner, selector, gate, posebuf, controller)
    except KeyboardInterrupt:
        print("interrupted — halting", flush=True)
    finally:
        # Every exit path, including STUCK's SystemExit(3), ends here. A second
        # Ctrl-C / kill must not interrupt the halt itself.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if _STREAMER is not None:
            _STREAMER.stop(lamp_off=True)
        elif SEND_CONTROL:
            try:
                client.send_cmd(0.0, 0.0, lamp=0)
            except Exception:
                pass
        print("stopped (lamp off)", flush=True)


def _drive(client, planner, selector, gate, posebuf, controller):
    dg = _depth_gate()
    print(f"depth_gate={'ready' if dg else 'off/unavailable'}")

    goals, mission_mode, route_goals, route_start_index = resolve_goals(client)
    if not goals:
        print("nothing to do")
        return
    _MISSION_INFO.update(route=list(route_goals or goals), seqs=list(_GOAL_SEQS),
                         start=route_start_index)
    route_profile = build_route_profile(route_goals)
    if route_profile is not None:
        print(f"route profile: {route_profile.describe()}")

    wait_for_gps(client)
    try:
        t0 = client.get_data()
        _osm_load(list(route_goals or goals) + [(t0.lat, t0.lon)])
    except Exception as e:
        print(f"[osm] load failed: {e}", flush=True)
    # After start-mission: before it, /data answers 400 in mission mode.
    _start_wheel_odom(client)
    # Startup is over (Chrome launched, mission started): from here a request
    # that hangs is a dead link, and the tick must fail fast and halt.
    client.timeout = SDK_TIMEOUT_S

    dt = 1.0 / CTRL_HZ
    for i, goal in enumerate(goals):
        print(f"driving to goal {i + 1}/{len(goals)}: {goal}")
        if hasattr(planner, "set_goal"):
            planner.set_goal(goal[0], goal[1])
        stuck_state: dict = {}
        _CLAIM.start_goal(is_last=(i == len(goals) - 1),
                          is_first=bool(mission_mode and i < len(_GOAL_SEQS)
                                        and _GOAL_SEQS[i] == 1))
        tel = client.get_data()
        rng0 = (geodesic_bearing_range(tel.lat, tel.lon, tel.yaw, *goal)[1]
                if not np.isnan(tel.lat) and tel.yaw is not None else float("nan"))
        if reached(rng0) and (_CLAIM.is_last or _CLAIM.is_first or rng0 <= _CLAIM.near_m):
            print(f"already within {rng0:.1f} m of {goal} — claiming without driving")
            _halt(client)
            if mission_mode:
                resp = client.checkpoint_reached()
                print(f"checkpoint-reached: {resp}")
                if resp.get("_http_status", 200) >= 400:
                    print("claim failed; will drive carefully")
                else:
                    _event("claimed checkpoint (already inside the radius)")
                    print(f"reached {goal}")
                    continue
            else:
                print(f"reached {goal}")
                continue

        while True:
            t0 = time.monotonic()
            try:
                if step(
                    client, planner, selector, gate, posebuf, controller,
                    goal, mission_mode, stuck_state,
                    route_profile=route_profile,
                    goal_index=route_start_index + i,
                ):
                    print(f"reached {goal}")
                    break
            except SystemExit:
                raise
            except Exception as e:
                _halt(client)
                print("tick error:", e)
            if time.monotonic() - _LAST_ALIVE > SDK_DEAD_S:
                _sdk_recover(client, mission_mode)
                # The claim may have landed just before the SDK died.
                if mission_mode and i < len(_GOAL_SEQS):
                    latest = _scanned_through(client)
                    if latest is not None and latest >= _GOAL_SEQS[i]:
                        print(f"checkpoint seq {_GOAL_SEQS[i]} already scanned "
                              f"(latest={latest}) — next goal", flush=True)
                        break
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    print("done — all goals reached")


if __name__ == "__main__":
    main()
