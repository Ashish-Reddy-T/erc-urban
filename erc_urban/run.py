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
import os
import time
import numpy as np

from .sdk_client import RoverClient
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
from .modules.commit import CommitWindow
from .modules.localmap import LocalMap
from .modules.topograph import TopoGraph, BLOCKED as TOPO_BLOCKED
from .modules.explore import GapExplorer
from .modules.selector import VLMSelector, GeminiVertexVLM, dump_candidates
from .modules.upgrades import LearnedPlanner, VLMSelectorFusion, PoseBuffer

STAGE = int(os.environ.get("STAGE", "1"))
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
# deg/s per unit angular, until a pivot burst measures the real value. Measured
# at 15, not the 90 first assumed; below ~0.7 stiction wins and it does not yaw
# at all. Overestimating here shrinks the steering gain, so the rover
# under-corrects and drifts until it trips the pivot latch.
STEER_RATE_PRIOR = float(os.environ.get("STEER_RATE_PRIOR", "15"))
# Hysteresis: enter align at ALIGN_DEG, stay in it until well inside the band.
ALIGN_EXIT = float(os.environ.get("ALIGN_EXIT", "22"))
# Forward speed. v_nom 0.20 with max_angular 0.14 pinned LogoNav to the
# turn_scale floor at 0.08 throttle, i.e. barely moving.
V_NOM = float(os.environ.get("V_NOM", "0.34"))
MAX_ANG = float(os.environ.get("MAX_ANG", "0.26"))
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
USE_COSTMAP = os.environ.get("USE_COSTMAP", "1") != "0"
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

# Visual odometry, the breadcrumb trail, and the commit window.
USE_ODOM = os.environ.get("USE_ODOM", "1") != "0"
USE_COMMIT = os.environ.get("USE_COMMIT", "1") != "0"
_VO: VisualOdometry | None = None
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


def _cmd(client: RoverClient, lin: float, ang: float) -> None:
    client.send_cmd(lin, ang, lamp=LAMP)


def _halt(client: RoverClient) -> None:
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
            lookahead_m=0.8,
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
            print(f"mission checkpoints: {len(goals)} remaining (latest_scanned={latest})")
            for c in remaining:
                print(f"  seq={c.sequence} id={c.id} ({c.lat}, {c.lon})")
            return goals, True, route_goals, route_start_index
        except Exception as e:
            msg = str(e)
            print(f"mission setup failed: {msg}")
            if "unavailable" in msg.lower():
                raise SystemExit(2) from e
            print("falling back to GOAL_LAT/GOAL_LON")

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


# Telemetry freshness. The SDK adds rtm_age_s (age of the last RTM peer
# message, on the SDK host clock, so no rover/laptop skew) and rtm_state (the
# Agora connection state) to every /data payload. When the session is ABORTED
# (REMOTE_LOGIN: someone else connected) or the last peer message is older than
# the SDK's own RTM_STALE_S (6 s, earth-rovers-sdk/main.py), lat/lon/yaw are a
# frozen copy and steering on them is driving blind. Older SDK builds omit
# both keys; the gate is then a no-op.
TELEM_STALE_S = float(os.environ.get("TELEM_STALE_S", "6"))
_RTM_DEAD_STATES = frozenset({"ABORTED", "DISCONNECTED", "TOKEN_EXPIRED"})


def _telemetry_dead(tel) -> str | None:
    """Reason string when this tick's telemetry must not be driven on."""
    raw = tel.raw if isinstance(tel.raw, dict) else {}
    state = raw.get("rtm_state")
    if state in _RTM_DEAD_STATES:
        return f"rtm_state={state} reason={raw.get('rtm_reason')}"
    try:
        age = float(raw.get("rtm_age_s"))
    except (TypeError, ValueError):
        return None
    if age > TELEM_STALE_S:
        return f"rtm_age_s={age:.1f}"
    return None


def step(client, planner, selector, gate, posebuf, controller, goal,
         mission_mode: bool, stuck_state: dict | None = None,
         route_profile=None, goal_index: int = 0) -> bool:
    global _REAR_SEEN
    try:
        front, rear = client.get_frames()
    except Exception:
        front, rear = None, None
    if rear is not None:
        _REAR_SEEN = True
    tel = client.get_data()
    dead = _telemetry_dead(tel)
    if dead is not None:
        _halt(client)
        print(f"telemetry stale ({dead}): holding still", flush=True)
        return False

    if np.isnan(tel.lat) or np.isnan(tel.lon):
        return False
    if tel.yaw is None:
        return False
    if front is None:
        return False

    # Measured motion. Everything downstream that needs to know how far the
    # rover actually moved reads this, because GPS is frozen, `rpms` is
    # [0,0,0,0] and the compass drifts. Runs before any control decision.
    vo = _vo()
    odom = vo.update(front) if vo is not None else None
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

    if reached(rng):
        _halt(client)
        if not mission_mode:
            return True
        resp = client.checkpoint_reached()
        if resp.get("_http_status", 200) >= 400:
            print(f"checkpoint-reached rejected (rng={rng:.1f}m): {resp}")
            return False
        print(f"checkpoint-reached ok: {resp}")
        return True

    if route_profile is not None:
        route_target = route_profile.target(goal_index, tel.lat, tel.lon)
        if route_target is not None:
            bearing, _ = geodesic_bearing_range(
                tel.lat, tel.lon, tel.yaw, route_target.lat, route_target.lon
            )
            route_note = (
                f"{route_target.describe()} "
                f"ckpt_bearing={np.degrees(checkpoint_bearing):+.0f}° "
            )

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
    if sem is not None:
        seg_hw = (
            dg.last_depth.shape[:2]
            if (dg is not None and dg.last_depth is not None)
            else (120, 160)
        )
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
    pre_aim = _route().align(
        goal_bearing=bearing,
        corridor_bearing=corr_bearing,
        corridor_width=corr_width,
        track_bearing=track_bearing,
        track_frac=track_frac,
        goal_trusted=not gps_stale,
    )
    if (
        route_profile is not None
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
    # Follow the corridor we just turned onto and do not re-enter align.
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
            _cmd(client, lin, ang)
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
                ang=ALIGN_ANG, settle_s=SETTLE_S, tol_deg=ALIGN_EXIT
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
            _cmd(client, lin, ang)
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
                _begin_escape(stuck_state, f"traversability: {tv_note}")
                return False
            idx = new_idx
        chosen = cands[idx]
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
            f"vo({odom.distance_m*100:.1f}cm {np.degrees(odom.dyaw):+.0f}° "
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
    if lin <= 0.0 and abs(ang) < 1e-3:
        _halt(client)
        return False
    _cmd(client, lin, ang)
    return False


def main():
    client = RoverClient(os.environ.get("SDK_URL", "http://localhost:8000"))
    planner, selector, gate, posebuf, controller = build_stack()
    ready = getattr(planner, "ready", None)
    print(
        f"STAGE={STAGE} planner={type(planner).__name__} "
        f"ready={ready} gate={type(gate).__name__ if gate else 'off'} "
        f"USE_DIRT_GATE={USE_DIRT_GATE} USE_OBS_STOP={USE_OBS_STOP} "
        f"USE_DEPTH={USE_DEPTH} LAMP={'on' if LAMP else 'off'}"
    )
    wait_for_sdk(client)
    dg = _depth_gate()
    print(f"depth_gate={'ready' if dg else 'off/unavailable'}")
    _cmd(client, 0.0, 0.0)

    goals, mission_mode, route_goals, route_start_index = resolve_goals(client)
    if not goals:
        print("nothing to do")
        return
    route_profile = build_route_profile(route_goals)
    if route_profile is not None:
        print(f"route profile: {route_profile.describe()}")

    wait_for_gps(client)

    dt = 1.0 / CTRL_HZ
    for i, goal in enumerate(goals):
        print(f"driving to goal {i + 1}/{len(goals)}: {goal}")
        if hasattr(planner, "set_goal"):
            planner.set_goal(goal[0], goal[1])
        stuck_state: dict = {}
        tel = client.get_data()
        if (
            not np.isnan(tel.lat)
            and tel.yaw is not None
            and reached(geodesic_bearing_range(tel.lat, tel.lon, tel.yaw, *goal)[1])
        ):
            print(f"already within 15 m of {goal} — claiming without driving")
            _halt(client)
            if mission_mode:
                resp = client.checkpoint_reached()
                print(f"checkpoint-reached: {resp}")
                if resp.get("_http_status", 200) >= 400:
                    print("claim failed; will drive carefully")
                else:
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
            time.sleep(max(0.0, dt - (time.monotonic() - t0)))

    client.send_cmd(0.0, 0.0, lamp=0)
    print("done (lamp off)")


if __name__ == "__main__":
    main()
