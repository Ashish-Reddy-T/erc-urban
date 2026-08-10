"""
Main control loop. Runs the Stage-0 baseline out of the box; each upgrade
flips one flag and swaps one module. This is the file you actually execute.

    STAGE=0 python -m erc_urban.run   # dirt-path + GPS creep (classical, locale-specific)
    STAGE=1 python -m erc_urban.run   # LogoNav (universal default; dirt gate OFF)
    STAGE=2 python -m erc_urban.run   # + latency buffer (SAM2 gate still stub)
    STAGE=3 python -m erc_urban.run   # + learned planner + VLM fusion

Mission mode (default): start mission → drive checkpoints → POST /checkpoint-reached
Free-drive smoke test: USE_MISSION=0 GOAL_LAT=.. GOAL_LON=.. python -m erc_urban.run

Mission-1 (any site — Mexico dirt OR Mauritius road):
    STAGE=1 USE_MISSION=1 python -m erc_urban.run
    # LogoNav + lamp + near-obstacle brake. Dirt gate off by default.
    # Optional village safety: USE_DIRT_GATE=1. Lamp off: LAMP=0.

Note: older earth-rovers-sdk builds have no /status or /missions — we don't require them.
With MISSION_SLUG set, call /start-mission BEFORE /data or the dashboard will work.
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
from .modules.route import RouteAligner
from .modules.traversability import TraversabilityVeto, SegFormerSource
from .modules.scoring import path_traversable_fractions
from .modules.odometry import VisualOdometry, Breadcrumbs
from .modules.commit import CommitWindow
from .modules.selector import VLMSelector, GeminiVertexVLM
from .modules.upgrades import (
    LearnedPlanner, VLMSelectorFusion, PoseBuffer, TraversabilityGate,
)

STAGE = int(os.environ.get("STAGE", "1"))  # default: LogoNav mission-1
CTRL_HZ = 5.0
K = 12 if STAGE < 3 else 18
USE_MISSION = os.environ.get("USE_MISSION", "1") != "0"
PATH_BLEND = float(os.environ.get("PATH_BLEND", "0.65"))
USE_DIRT_GATE = (
    True if STAGE == 0
    else os.environ.get("USE_DIRT_GATE", "0") == "1"
)
# Classical near-field contact (lamp-on-wall / dark blob). Always on for safety.
USE_OBS_STOP = os.environ.get("USE_OBS_STOP", "1") != "0"
# Real clearance: Depth Anything V2 (ERC-3). On by default; falls back if no ckpt.
USE_DEPTH = os.environ.get("USE_DEPTH", "1") != "0"
LAMP = int(os.environ.get("LAMP", "1"))
# Only STOP to pivot for large errors. A stop-and-settle cycle costs ~3 s, so
# anything the rover can steer out of while rolling should be steered, not
# pivoted. Below this, drive and correct with the proportional term.
ALIGN_DEG = float(os.environ.get("ALIGN_DEG", "65"))
# In-place pivots need far more throttle than arc-following: with lin=0 the
# skid-steer has to break stiction on both tracks. 0.16 does nothing. But a
# flat 0.6 held for a whole (slow) tick swings >100deg and overshoots, so the
# pivot is proportional between a stiction floor and this ceiling.
ALIGN_ANG = float(os.environ.get("ALIGN_ANG", "0.60"))
ALIGN_MIN = float(os.environ.get("ALIGN_MIN", "0.30"))
ALIGN_KP = float(os.environ.get("ALIGN_KP", "0.005"))
# Telemetry is 0.8-1.4 s stale and refreshes at 2 Hz (measured). Hold still at
# least this long after a pivot burst before believing the next heading.
SETTLE_S = float(os.environ.get("SETTLE_S", "1.8"))
# Rolling heading correction, in yaw-rate terms (see _steer_nudge).
STEER_KP = float(os.environ.get("STEER_KP", "0.30"))        # deg/s per deg err
STEER_MAX_DPS = float(os.environ.get("STEER_MAX_DPS", "10"))
STEER_MAX_ANG = float(os.environ.get("STEER_MAX_ANG", "0.15"))
# deg/s per unit angular, used until a pivot burst measures the real value.
# Was 90, which is ~6x too fast: measured on the live Mini+ at 15 deg/s per unit
# (0.7 command for 0.8 s -> 8.5 deg), and below ~0.7 stiction wins and the rover
# does not yaw at all. An overestimate here makes the steering gain too small,
# so the rover under-corrects and drifts until it trips the pivot latch.
STEER_RATE_PRIOR = float(os.environ.get("STEER_RATE_PRIOR", "15"))
# Hysteresis: enter align at ALIGN_DEG, stay in it until well inside the band.
ALIGN_EXIT = float(os.environ.get("ALIGN_EXIT", "22"))
# Forward speed. v_nom 0.20 with max_angular 0.14 pinned LogoNav to the 0.4
# turn_scale floor => 0.08 throttle, i.e. barely moving.
V_NOM = float(os.environ.get("V_NOM", "0.34"))
MAX_ANG = float(os.environ.get("MAX_ANG", "0.26"))
# Progress watchdog: metres of range that must be closed within this window.
FREEZE_SEC = float(os.environ.get("FREEZE_SEC", "45"))
FREEZE_M = float(os.environ.get("FREEZE_M", "1.5"))
# Beyond this the rover is going over; nothing forward-looking helps.
MAX_TILT_DEG = float(os.environ.get("MAX_TILT_DEG", "35"))
# Consecutive over-threshold ticks before believing it.
TILT_TRIPS = int(os.environ.get("TILT_TRIPS", "4"))


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

# Ground-relative obstacle costmap + GeNIE-style candidate veto (Module 6).
USE_COSTMAP = os.environ.get("USE_COSTMAP", "1") != "0"
# Telemetry is ~1.1 s stale and commands take ~0.3 s to land, so the rover
# keeps moving for well over a second after a decision. 0.55 m of clearance
# is not a stopping distance, it is a collision report.
COSTMAP_STOP_M = float(os.environ.get("COSTMAP_STOP_M", "1.2"))
# Was 2.5, which is now permanently active. Depth saturates on this camera:
# hypersim caps at 19.6 m, so distant structure compresses into ~2.5 m and an
# empty courtyard measures 2.55 m. At the old threshold the rover would creep
# everywhere. 1.8 sits clear of that floor while still leaving 0.6 m of
# deceleration above the 1.2 m stop.
COSTMAP_SLOW_M = float(os.environ.get("COSTMAP_SLOW_M", "1.8"))
OBST_HEIGHT_M = float(os.environ.get("OBST_HEIGHT_M", "0.18"))
ROVER_RADIUS_M = float(os.environ.get("ROVER_RADIUS_M", "0.35"))

# Semantic traversability: tells dirt track from grass verge, and sees the
# hazards geometry cannot (water, people).
USE_SEMANTICS = os.environ.get("USE_SEMANTICS", "1") != "0"
OFFTRACK_PENALTY = float(os.environ.get("OFFTRACK_PENALTY", "0.6"))
SEG_EVERY_N = int(os.environ.get("SEG_EVERY_N", "1"))
# Below this fraction of traversable surface in the near-field band, there is
# no path in front of the rover. Monocular depth CANNOT see a grass wall — it
# reported 2.4 m with blades against the lens — so this is the only signal
# that stops the rover burying itself. Refuse to drive, and back out.
TRACK_MIN = float(os.environ.get("TRACK_MIN", "0.04"))
# How strongly the visible track overrides the GPS bearing. On a single-track
# path the line is the constraint; the GPS goal is only a direction hint.
TRACK_BLEND = float(os.environ.get("TRACK_BLEND", "0.75"))
# Corridor following. On a single-track path the corridor is not a hint to be
# blended into a goal-seeking controller — it IS the control law. LogoNav is
# goal-conditioned and will happily steer off a 1 m path toward a GPS point
# 34 m away, which is exactly how the rover buried itself.
CORRIDOR_MIN = float(os.environ.get("CORRIDOR_MIN", "0.12"))
CORRIDOR_MAX_ANG = float(os.environ.get("CORRIDOR_MAX_ANG", "0.35"))
CORRIDOR_GOAL_BIAS = float(os.environ.get("CORRIDOR_GOAL_BIAS", "0.20"))
CORRIDOR_BIAS_MAX_DEG = float(os.environ.get("CORRIDOR_BIAS_MAX_DEG", "8"))
# Only turn around for the goal if it is genuinely behind us; otherwise follow
# the path, even when it currently points away from the checkpoint.
CORRIDOR_ALIGN_DEG = float(os.environ.get("CORRIDOR_ALIGN_DEG", "120"))
# Recovery scan: rotate in steps looking for a drivable heading.
SCAN_STEPS = int(os.environ.get("SCAN_STEPS", "7"))
SCAN_STEP_DEG = float(os.environ.get("SCAN_STEP_DEG", "50"))
SCAN_RESUME = float(os.environ.get("SCAN_RESUME", "0.35"))

# --- measured hardware geometry (erc_urban/camera_front.json) ---------------
# fx/FOV measured on the live Mini+ by closing a 360 deg spin; camera height
# from the official MINI+ dimension drawing. Everything geometric reads from
# here rather than re-deriving a 90 deg guess. See calib.py.
CALIB = CameraCalibration.load_or_default(
    os.path.join(os.path.dirname(__file__), "camera_front.json"), 1024, 576
)
# K-candidate planner (M1) and the semantic traversability veto (M6).
USE_MULTI = os.environ.get("USE_MULTI", "1") != "0"
USE_TRAV_VETO = os.environ.get("USE_TRAV_VETO", "1") != "0"
# Ground-anchored depth response correction (M6 geometric half).
USE_DEPTH_FIT = os.environ.get("USE_DEPTH_FIT", "1") != "0"
# Depth is only trustworthy to ~3.5 m on this camera: hypersim saturates at
# 19.6 m, so anything beyond maps into the near field and no remap recovers it.
# Bound the costmap to what the sensor can actually resolve.
DEPTH_MAX_M = float(os.environ.get("DEPTH_MAX_M", "3.5"))

_DEPTH_GATE: DepthObstacleGate | None = None
_COSTMAP: DepthCostmap | None = None
_SEM: SemanticTraversability | None = None
_SEM_TRIED = False
_DEPTH_FIT: DepthResponseFit | None = None
_ROUTE: RouteAligner | None = None
_TRAV: TraversabilityVeto | None = None

# M5: visual odometry, the breadcrumb trail, and the commit window.
USE_ODOM = os.environ.get("USE_ODOM", "1") != "0"
USE_COMMIT = os.environ.get("USE_COMMIT", "1") != "0"
_VO: VisualOdometry | None = None
_CRUMBS = Breadcrumbs()
_COMMIT = CommitWindow()


class GpsHealth:
    """
    Decide whether the GPS-derived goal bearing may steer the rover.

    Measured on this Mini+: `latitude`/`longitude` are bit-identical across 24
    stationary samples AND across 30 s of driving, while the IMU updates
    normally. A frozen fix is not a noisy measurement of the right answer, it is
    not a measurement at all -- and combined with a drifting compass it produced
    a bearing that sat at +-177 deg with the SIGN flipping between ticks, so the
    rover reversed its pivot every burst and never drove anywhere.

    Visual odometry is the cross-check that makes this decidable: if we have
    demonstrably moved and the fix has not changed, the fix is stale. Without
    VO you cannot distinguish "frozen GPS" from "stationary rover".
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
        # Both conditions: moved far enough that the fix SHOULD have changed,
        # and enough wall time that this is not just a slow update rate.
        self.stale = (
            self.moved_since_change >= self.min_move_m and held >= self.stale_after_s
        )
        return self.stale

    def describe(self) -> str:
        return (f"gps={'STALE' if self.stale else 'ok'}"
                f"({self.moved_since_change:.2f}m since fix change)")


_GPS = GpsHealth()


# M4: VLM candidate selection. OFF unless USE_VLM=1, because it bills a
# cloud API per query. Measured 1.6 s median on Vertex Gemini 3.5 Flash-Lite,
# which is why it runs on its own thread -- the rover has a command watchdog,
# so an in-tick call would drop the command rate and physically slow it down.
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
    Fraction of traversable surface behind the rover.

    When the front camera is buried in vegetation the rear view is usually the
    path we just came off, so this is the signal that says reversing is safe.
    Requires HAS_REAR_CAMERA=true in the SDK .env — it defaults to False, so
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
        _, _, good = sem.categorize(labels)
        _, band, _ = sem.track(good)
        return float(band)
    except Exception:
        return None


def _steer_nudge(stuck_state: dict | None, bearing: float) -> float:
    """
    Heading correction while rolling, expressed as a target YAW RATE and then
    converted to an angular command using the rover's measured authority.

    Asking for "10 deg/s toward the goal" is portable; asking for "0.08
    angular" is not — on a high-authority rover that is a circle, on a
    sluggish one it is nothing.
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


def _aligning(stuck_state: dict | None, bearing: float) -> bool:
    """
    Hysteretic align latch. Enter above ALIGN_DEG, leave only below ALIGN_EXIT.

    Without this, a pivot that overshoots (+73° -> -53° in one slow tick) lands
    back outside the band with the sign flipped and the rover oscillates
    forever instead of ever driving.
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
    STAGE 0: classical dirt-path + village creep (Mexico-style dirt)
    STAGE 1: LogoNav + pure-pursuit (universal); optional USE_DIRT_GATE=1
    STAGE 2+: PoseBuffer; SAM2 TraversabilityGate still a stub
    STAGE 3: LearnedPlanner / CKPT + VLM selector
    """
    if STAGE == 0:
        planner = BaselinePlanner()
        controller = VillageGotoController()
        gate = DirtPathGate()
        posebuf = None
        selector = None
    elif STAGE == 1:
        # LogoNav runs on the accelerator. The old note here said CPU won
        # (120 ms vs 930 ms on mps), but that was measured on an earlier torch
        # where mps fell back to CPU for these ops. Re-measured under torch
        # 2.11: mps 13 ms vs cpu 203 ms at batch 1, and still 13 ms at K=18.
        # Outputs agree to 8e-6. Override with LOGONAV_DEVICE if a machine
        # regresses.
        base = LogoNavPlanner(
            device=os.environ.get("LOGONAV_DEVICE", _auto_device()),
        )
        # M1: LogoNav alone emits ONE trajectory, so M4/M5/M6 would have nothing
        # to select over or fall back to. MultiHypothesisPlanner keeps LogoNav as
        # the scorer and takes coverage from a geometric anchor fan -- the same
        # shape as S2E's anchors, which Slow-Brain builds on. USE_MULTI=0 falls
        # back to the single-hypothesis planner.
        planner = MultiHypothesisPlanner(base=base) if USE_MULTI else base
        controller = PurePursuitController(
            v_nom=V_NOM, max_angular=MAX_ANG, lookahead_m=0.8,
        )
        gate = DirtPathGate() if USE_DIRT_GATE else None
        posebuf = None
        selector = None
    else:
        controller = PurePursuitController()
        gate = DirtPathGate() if USE_DIRT_GATE else TraversabilityGate()
        posebuf = PoseBuffer()
        if STAGE >= 3:
            planner = LearnedPlanner(checkpoint_path=os.environ["CKPT"])
            selector = VLMSelectorFusion(k=K)
        else:
            planner = LogoNavPlanner(
                device=os.environ.get("LOGONAV_DEVICE", "auto"),
            )
            selector = None
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
        "Start it first: cd earth-rovers-sdk && hypercorn main:app --reload"
    )


def resolve_goals(client: RoverClient) -> tuple[list[tuple[float, float]], bool]:
    if USE_MISSION:
        try:
            if os.environ.get("SKIP_START_MISSION", "0") == "0":
                print("starting mission...")
                started = client.start_mission()
                print("start-mission ok:", started.get("message", started))
            cps, latest = client.get_checkpoints()
            remaining = [c for c in cps if c.sequence > latest]
            if not remaining and cps:
                print(f"all {len(cps)} checkpoints already scanned (latest={latest})")
                return [], True
            if not cps:
                raise RuntimeError(
                    "start-mission succeeded but checkpoints list is empty"
                )
            goals = [(c.lat, c.lon) for c in remaining]
            print(f"mission checkpoints: {len(goals)} remaining (latest_scanned={latest})")
            for c in remaining:
                print(f"  seq={c.sequence} id={c.id} ({c.lat}, {c.lon})")
            return goals, True
        except Exception as e:
            msg = str(e)
            print(f"mission setup failed: {msg}")
            if "unavailable" in msg.lower():
                print(
                    "\n*** Bot unavailable for SDK ***\n"
                    "FrodoBots rejected the ride. Ask them to free the bot / confirm\n"
                    "BOT_SLUG + MISSION_SLUG, and that no one else is controlling it.\n"
                    "Do NOT fall back to GOAL_LAT/LON for a scored mission.\n"
                )
                raise SystemExit(2) from e
            print("falling back to GOAL_LAT/GOAL_LON")

    goals = [(
        float(os.environ.get("GOAL_LAT", "0.0")),
        float(os.environ.get("GOAL_LON", "0.0")),
    )]
    print(f"free-drive goal: {goals[0]}")
    return goals, False


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
    print("warning: proceeding without GPS fix yet")


# Reversing is BLIND on this rover. /v2/screenshot returns rear_frame=None even
# with HAS_REAR_CAMERA set, so nothing whatsoever is sensed behind. A 15-tick
# reverse at 0.22 throttle is several metres travelled with no perception, and
# it reversed this rover into a staircase. Negative obstacles (steps, kerb
# drops, pool edges) are exactly what a rear camera would be needed to see, and
# exactly what geometry ahead cannot warn about.
#
# So: with no rear view, reverse becomes a short nudge to break contact, not a
# retreat manoeuvre. Recovery heading comes from the in-place scan, which is
# sensed by the front camera.
REVERSE_TICKS = int(os.environ.get("REVERSE_TICKS", "15"))
REVERSE_TICKS_BLIND = int(os.environ.get("REVERSE_TICKS_BLIND", "4"))
_REAR_SEEN = False


def _reverse_ticks() -> tuple[int, str]:
    if _REAR_SEEN:
        return REVERSE_TICKS, ""
    return REVERSE_TICKS_BLIND, " [BLIND: no rear camera — short nudge only]"


# Prefer turning around over reversing. With no rear camera, a 180 deg turn
# converts a blind manoeuvre into a sensed one: the front camera then looks
# where the rover is going, so the traversability veto and costmap apply to the
# escape as well as to the approach. Reversing has neither.
USE_TURNAROUND = os.environ.get("USE_TURNAROUND", "1") != "0"


def _begin_escape(stuck_state: dict, reason: str, key=None) -> None:
    """
    Get out of a stuck state. Strategy, best first:

      1. TURN toward where we came from and drive forward, watching. The
         breadcrumb trail says which way that is, and it is ground we have
         already driven over.
      2. Turn around anyway, using the goal bearing, if there is no trail.
      3. Only then a short blind reverse nudge, and only far enough to break
         contact -- never as a retreat manoeuvre.

    A blind 15-tick reverse is what put this rover into a staircase.
    """
    attempts = stuck_state.get("rev_attempts", 0) + 1
    stuck_state["rev_attempts"] = attempts
    stuck_state["n"] = 0
    stuck_state["key"] = None
    if attempts > 3:
        print(
            f"STUCK: {reason} after {attempts - 1} escape attempts "
            f"({key}) — stopping. Needs walker assist.",
            flush=True,
        )
        raise SystemExit(3)

    back = _CRUMBS.retreat_bearing() if USE_ODOM else None
    if USE_TURNAROUND and back is not None:
        stuck_state["detour_deg"] = float(np.degrees(back))
        stuck_state.pop("detour_started", None)
        stuck_state["rev_left"] = 0
        stuck_state["pivot_left"] = 0
        print(
            f"STUCK: {reason} — turning {np.degrees(back):+.0f}° back along "
            f"the path just driven ({len(_CRUMBS.pts)} crumbs, "
            f"{_CRUMBS.travelled_m:.1f} m) and driving forward "
            f"(attempt {attempts}/3)",
            flush=True,
        )
        return

    if USE_TURNAROUND and not _REAR_SEEN:
        # No trail (odometry lost, or we never moved). Turning around still
        # beats reversing blind: at least the camera faces the direction of
        # travel and the scan can find a way out.
        stuck_state["detour_deg"] = 180.0
        stuck_state.pop("detour_started", None)
        stuck_state["scan_after_reverse"] = True
        stuck_state["rev_left"] = 0
        print(
            f"STUCK: {reason} — no trail; turning 180° to face the escape "
            f"direction (attempt {attempts}/3)",
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
        f"(attempt {attempts}/3){note}",
        flush=True,
    )


# Kept as the old name so the many call sites read unchanged.
_begin_reverse = _begin_escape


def _begin_pivot(stuck_state: dict, reason: str) -> None:
    """In-place yaw toward goal when lighting is uncertain — do not reverse away."""
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


def step(client, planner, selector, gate, posebuf, controller, goal,
         mission_mode: bool, stuck_state: dict | None = None) -> bool:
    global _REAR_SEEN
    try:
        front, rear = client.get_frames()
    except Exception:
        front, rear = None, None
    if rear is not None:
        _REAR_SEEN = True
    tel = client.get_data()

    # Tilt abort. The rover has gone over twice in testing, both times while a
    # loop kept commanding forward into something it had climbed. No amount of
    # forward-looking perception catches a rover already past the tipping
    # point, so this reads the IMU directly and refuses to command anything.
    acc = (tel.raw.get('accels') or [[0, 0, 1, 0]])[-1][:3]
    nrm = float(np.linalg.norm(acc)) or 1.0
    tilt_deg = float(np.degrees(np.arccos(np.clip(acc[2] / nrm, -1.0, 1.0))))
    # Debounced. A single sample is not evidence: an in-place pivot lurches the
    # rover and a bare reading tripped at 31 deg while the rover was measured
    # upright at 2.8 deg immediately afterwards. ERC-3's imu_safety.py counts
    # consecutive trips for the same reason.
    if stuck_state is not None:
        n_tilt = stuck_state.get('tilt_n', 0) + 1 if tilt_deg > MAX_TILT_DEG else 0
        stuck_state['tilt_n'] = n_tilt
        if n_tilt >= TILT_TRIPS:
            _halt(client)
            print(f'TILT {tilt_deg:.0f} deg > {MAX_TILT_DEG:.0f} for {n_tilt} '
                  'consecutive ticks — halting. Rover needs righting.', flush=True)
            raise SystemExit(4)

    if np.isnan(tel.lat) or np.isnan(tel.lon):
        return False
    if tel.yaw is None:
        return False
    if front is None:
        return False

    # M5a: measured motion. Everything downstream that needs to know how far
    # the rover actually moved reads this, because GPS is frozen, `rpms` is
    # [0,0,0,0] and the compass drifts. Run it before any control decision.
    vo = _vo()
    odom = vo.update(front) if vo is not None else None
    gps_stale = _GPS.update(tel.lat, tel.lon, odom, tel.stamp)
    if odom is not None:
        _CRUMBS.update(odom, tel.stamp)
        if posebuf is not None:
            posebuf.integrate(tel.stamp, odom)

    bearing, rng = geodesic_bearing_range(
        tel.lat, tel.lon, tel.yaw, goal[0], goal[1]
    )

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

    # Ground-relative obstacle costmap, reusing the depth map just computed.
    # Unlike the fixed-band clearance it sees low, close obstacles (curbs,
    # rocks, ruts) that live in the bottom of the frame.
    cmap = _costmap()
    cm_res = None
    sem_note = ""
    track_frac: float | None = None
    track_bearing: float | None = None
    corr_bearing: float | None = None
    corr_width: float = 0.0
    # good_m must be initialised alongside the others: it is only bound
    # inside the semantics block, and the rescore below reads it
    # unconditionally -- a NameError on every tick without semantics.
    hard_m = soft_m = good_m = None
    # Semantics runs INDEPENDENTLY of depth. It is the more trustworthy signal
    # on this platform (the vkitti metric checkpoint is out of distribution for
    # a 10 cm-high camera), so it must not be gated on the depth model
    # succeeding — the path-visible check is what stops the rover burying itself.
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
            # Teach the depth response fit using the ground as a ruler. Given
            # the measured intrinsics/height, the true distance to a ground
            # pixel is pure geometry, so traversable pixels below the horizon
            # are a dense raw->true correspondence spanning the whole visible
            # range. Fit on the RAW map: fitting on corrected depth compounds.
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
            sem_note = (
                f"sem(hard={hard_m.mean():.2f} off={soft_m.mean():.2f} "
                f"track={t_all:.2f}/near={t_band:.2f}"
                + (
                    f" @{np.degrees(t_bearing):+.0f}°) "
                    if t_bearing is not None else ") "
                )
            )

    if cmap is not None and dg is not None and dg.last_depth is not None:
        try:
            zmap = dg.last_depth
            # Sky has no depth, and nothing above the horizon can be ground.
            # Without this the segmenter's mid-range value for sky lands in the
            # near field and the costmap reports ~15,000 obstacle points in an
            # empty courtyard (measured); with it, 0-628.
            if labels is not None and hard_m is not None and zmap.shape == hard_m.shape:
                cs = CALIB.scaled_to(zmap.shape[1], zmap.shape[0])
                rows = np.arange(zmap.shape[0])[:, None]
                sky = np.isin(labels, list(sem.notground)) if sem.notground else False
                zmap = np.where(
                    (rows <= cs.horizon_v()) | sky, np.nan, zmap
                )
            cm_res = cmap.compute(zmap, hard_mask=hard_m, soft_mask=soft_m)
        except Exception as e:
            print(f"  costmap error: {e}", flush=True)
            cm_res = None
    # Depth often hallucinates ~3 m at true contact — distrust it when RGB says closed.
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

    if stuck_state is not None:
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

        # Hard stop: depth clearance OR classical contact (lamp-on-wall / blob).
        hit = False
        why = ""
        # No traversable surface in the near field. Depth is useless here
        # (grass on the lens reads as metres of clear space), so this check
        # must not be conditioned on any geometric agreement.
        # A scan already in progress IS the response to "no path" — let it run
        # instead of re-triggering a reverse every tick and burning the retry
        # budget before the rover has finished looking around.
        _pv = stuck_state.get("pivot_ctl")
        recovering = (
            stuck_state.get("scan_left", 0) > 0
            or stuck_state.get("detour_deg") is not None
            or (_pv is not None and _pv.busy())
        )
        if track_frac is not None and track_frac < TRACK_MIN and not recovering:
            why = f"no_path_visible(track_near={track_frac:.3f})"
            # Reversing alone just puts us back where we were, pointing the
            # same way, so the next forward run repeats the mistake. Scan for a
            # new heading after backing off.
            stuck_state["scan_after_reverse"] = True
            # Check behind before flailing forward: if the rear camera can see
            # the track, we drove in off a path that is still there, and
            # reversing back onto it is the reliable escape.
            rear_frac = _rear_track_frac(rear)
            if rear_frac is not None:
                why += f" rear_track={rear_frac:.2f}"
            hit = True
        elif cm_res is not None and cm_res.forward_clearance_m < COSTMAP_STOP_M:
            hit, why = True, (
                f"costmap_stop({cm_res.forward_clearance_m:.2f}m "
                f"pts={cm_res.n_obstacle_points})"
            )
        elif (
            depth_view is not None and depth_view.blocked and cm_res is None
        ):
            # Only when the costmap is unavailable. OutdoorTraversability scores
            # a fixed image band and has no ground-plane removal, so once depth
            # is correctly scaled it reads the ground itself as an obstacle:
            # observed depth_stop(0.17m) on open pavement, which is the ground
            # under a 0.148 m camera. Its 0.55 m threshold was tuned against the
            # old broken 5-6 m scale. The costmap does RANSAC ground removal and
            # is the right authority now; this stays only as a fallback.
            hit, why = True, depth_view.reason
        elif obs_view is not None and obs_view.blocked:
            hit, why = True, (
                f"obstacle ahead ({obs_view.reason} "
                f"dark={obs_view.dark_frac:.2f} bright={obs_view.bright_frac:.2f})"
            )
        if hit:
            _begin_reverse(stuck_state, why)
            return False

        if path is not None:
            force = stuck_state.get("force_forward", 0)
            if force > 0:
                stuck_state["force_forward"] = force - 1
            else:
                if path.blocked and path.grass_frac > 0.25:
                    _begin_reverse(stuck_state, "grass/no dirt corridor in view")
                    return False
                if path.uncertain or (path.blocked and path.grass_frac <= 0.25):
                    if stuck_state.get("pivot_attempts", 0) >= 4:
                        _begin_reverse(
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
            _begin_reverse(stuck_state, f"rng/bearing frozen at {key}", key)
            return False

        # GPS range not improving while we keep commanding → jammed (wall/rover).
        # An intentional in-place pivot holds range constant BY DESIGN, so it
        # must not feed the jam counter — otherwise aligning always self-trips.
        aligning = _aligning(stuck_state, bearing)
        if aligning:
            # Bounded: if the pivot isn't closing the bearing error, the rover
            # really can't yaw here — reverse instead of spinning forever.
            err = abs(float(np.degrees(bearing)))
            best = stuck_state.get("align_best")
            if best is None or err < best - 3.0:
                stuck_state["align_best"] = err
                stuck_state["align_n"] = 0
            else:
                stuck_state["align_n"] = stuck_state.get("align_n", 0) + 1
            # Pulse-and-settle only improves the error once per ~3 s burst
            # cycle, so this must count many more ticks than a per-tick loop.
            if stuck_state["align_n"] >= 150:
                stuck_state["align_best"] = None
                stuck_state["align_n"] = 0
                _begin_reverse(
                    stuck_state, f"pivot not converging (bearing {err:.0f}°)"
                )
                return False
            stuck_state["prog_t0"] = None
        else:
            stuck_state["align_best"] = None
            stuck_state["align_n"] = 0
            # Progress is measured over a TIME window, not per tick. At a
            # ~1.5 Hz loop and a slow creep the per-tick delta is a few cm and
            # always looks "frozen" — that is loop latency, not a jam.
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
                _begin_reverse(
                    stuck_state,
                    f"no progress: {base_rng - rng:+.1f}m in {held:.0f}s "
                    f"(need {FREEZE_M}m)",
                )
                return False

    pre_aim = _route().align(
        goal_bearing=bearing,
        corridor_bearing=corr_bearing,
        corridor_width=corr_width,
        track_bearing=track_bearing,
        track_frac=track_frac,
        goal_trusted=not gps_stale,
    )

    # The planner RANKS candidates by agreement with `goal_bearing`, so feeding
    # it a dead GPS bearing makes its ranking worse than chance. Measured on a
    # recorded episode: with a wrong bearing the planner's top-1 pick scored
    # 0.494 m ADE against a uniformly RANDOM candidate's 0.320 m -- i.e. its
    # ordering was actively harmful, and Probability Fusion then dragged the
    # VLM's -41% advantage down to -4% by mixing in that bad ranking.
    #
    # So the bearing the planner sees is the route-aligned one whenever GPS is
    # untrustworthy: vision is the only live signal, and a planner ranked on it
    # is ranked on something real.
    planner_bearing = pre_aim.bearing if (gps_stale or pre_aim.confidence > 0) else bearing
    obs = Observation(
        front=front, rear=rear, lat=tel.lat, lon=tel.lon,
        yaw=tel.yaw, stamp=tel.stamp,
        goal_bearing=planner_bearing, goal_range_m=rng,
    )

    # Facing the wrong way: pivot in place. Short-circuit before the planner —
    # its candidates are discarded on align ticks and the forward pass is what
    # drags the loop well below CTRL_HZ.
    pivot = stuck_state.get("pivot_ctl") if stuck_state is not None else None
    # A burst in flight is never preempted by a fresh (stale) bearing.
    pivot_busy = pivot is not None and pivot.busy()

    # A costmap detour is a ONE-SHOT turn: execute a single burst toward open
    # space, then drop it so the next tick re-plans against fresh perception
    # rather than spinning open-loop toward a stale gap.
    # --- recovery scan: rotate in steps, stop at the first drivable heading ---
    if (
        stuck_state is not None
        and stuck_state.get("scan_left", 0) > 0
        and not pivot_busy
        and stuck_state.get("detour_deg") is None
    ):
        seen = float(track_frac) if track_frac is not None else 0.0
        best_frac, best_turn = stuck_state.get("scan_best", (-1.0, 0.0))
        turned = float(stuck_state.get("scan_turned", 0.0))
        if seen > best_frac:
            stuck_state["scan_best"] = (seen, turned)
            best_frac, best_turn = seen, turned
        if seen >= SCAN_RESUME:
            print(
                f"  SCAN: drivable heading found after {turned:+.0f}° "
                f"(track_near={seen:.2f}) — resuming",
                flush=True,
            )
            stuck_state["scan_left"] = 0
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
                # Sweep alternating sides, widening: +50, -50, +100, -100, ...
                # A path just behind-left should not cost a near-full circle.
                k = SCAN_STEPS - left            # 1, 2, 3, ...
                mag = SCAN_STEP_DEG * ((k + 1) // 2)
                target = mag if (k % 2 == 1) else -mag
                target = max(-180.0, min(180.0, target))
                rel = target - turned
                stuck_state["scan_turned"] = target
                stuck_state["detour_deg"] = rel
                stuck_state.pop("detour_started", None)
                print(
                    f"  SCAN: track_near={seen:.2f} at {turned:+.0f}° — "
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
        # One burst+settle done — drop the target and re-perceive rather than
        # grinding toward a fixed angle that never decreases.
        stuck_state.pop("detour_deg", None)
        stuck_state.pop("detour_started", None)
        pivot.reset()
        pivot_busy = False
        detour = None

    # With a corridor in view, do NOT stop and pivot toward the GPS goal — the
    # path is the only way through and it may well point away from the goal.
    # Turn around only when the goal is genuinely behind us.
    # Whether to abandon driving and pivot toward the GPS goal. On this rover
    # the GPS bearing is close to meaningless: lat/lon is frozen AND the
    # compass drifts (measured -141 deg then back to -7 deg while commanded one
    # way). Left to itself the latch trips on that drift and the rover pivots
    # forever without converging -- observed live, bearing wandering -1 deg to
    # +83 deg while stationary, then 20+ ticks of non-converging pivots.
    #
    # So vision gets right of way: if the route aligner has any visual opinion,
    # follow it and do not align. Pivoting for the goal is reserved for when
    # vision has nothing to say, or the goal is genuinely behind us.
    on_corridor = pre_aim.confidence > 0.0
    want_align = (
        pre_aim.source == "turnaround"
        if on_corridor
        else _aligning(stuck_state, bearing)
    )
    if on_corridor and stuck_state is not None and not want_align:
        stuck_state["align_latch"] = False
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
        else:
            err_deg = float(np.degrees(bearing))
        lin, ang, note = pivot.command(err_deg, tel, time.monotonic())
        if note == "aligned":
            # The pivot cannot resolve any finer than this; releasing the latch
            # here (rather than at ALIGN_EXIT) is what stops it deadlocking at
            # zero command on a high-yaw-authority rover. Hand off to driving.
            if stuck_state is not None:
                stuck_state["align_latch"] = False
                stuck_state.pop("detour_deg", None)
                stuck_state.pop("detour_started", None)
            pivot.reset()
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
    # Re-rank with perception. Candidates do not exist until propose() runs,
    # so traversability can only enter as a second pass. This is the fix for a
    # measured defect: with traversability used only as a post-hoc veto, the
    # ranking scored WORSE THAN RANDOM against human-driven rides (1.029 m vs
    # 0.757 m median ADE). Feeding it into the score cut that to 0.775 m.
    if cands and good_m is not None and hard_m is not None and hasattr(planner, 'rescore'):
        try:
            tf = path_traversable_fractions(cands, good_m & ~hard_m, CALIB)
            cands = planner.rescore(cands, obs, trav_fracs=tf)
        except Exception as e:
            print(f'  rescore skipped: {e}', flush=True)
    if path is not None and STAGE <= 1:
        cands = rescore(cands, path)

    # GeNIE Module 6: hard veto AFTER proposal — drop any candidate whose path
    # crosses occupied space. Runs at every stage and does NOT depend on
    # USE_DIRT_GATE, unlike the block below (see comment there).
    veto_note = ""
    if cm_res is not None and cmap is not None and cands:
        keep = cmap.veto(cm_res, cands)
        survivors = [c for c, ok in zip(cands, keep) if ok]
        n_vetoed = len(cands) - len(survivors)
        if survivors:
            cands = survivors
            if n_vetoed:
                veto_note = f"veto={n_vetoed}/{len(keep)} "
            # Prefer the track over the verge — a preference, not a rule, so
            # the rover can still leave the trail to get around something.
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
            # Everything is blocked ahead. If there is still room, TURN toward
            # the most open bearing and re-perceive — reversing is for when the
            # rover is genuinely trapped, not for a wall 3 m away.
            if cm_res.forward_clearance_m > COSTMAP_STOP_M:
                bi = int(np.argmax(cm_res.clearance_by_bin))
                open_deg = float(np.degrees(cm_res.bin_centers[bi]))
                stuck_state["detour_deg"] = open_deg
                stuck_state.pop("detour_started", None)
                print(
                    f"  DETOUR: all {len(keep)} candidates vetoed "
                    f"(fwd {cm_res.forward_clearance_m:.2f}m) — turning "
                    f"{open_deg:+.0f}° toward open space "
                    f"({cm_res.clearance_by_bin[bi]:.1f}m)",
                    flush=True,
                )
                _halt(client)
                return False
            _begin_reverse(
                stuck_state,
                f"all {len(keep)} candidates vetoed by costmap "
                f"(fwd {cm_res.forward_clearance_m:.2f}m)",
            )
            return False
        else:
            _halt(client)
            print("  BLOCKED: every candidate vetoed by costmap", flush=True)
            return False

    # NOTE: this legacy block is reachable only when USE_DIRT_GATE=1, because
    # `path` is None otherwise — which is why the SAM2 TraversabilityGate never
    # ran at any stage. The costmap veto above is deliberately not gated on it.
    if gate is not None and path is not None:
        m = gate.mask(obs, cands)
        safe = [c for c, ok in zip(cands, m) if ok]
        if not safe:
            if stuck_state is not None:
                if path.grass_frac > 0.25:
                    _begin_reverse(stuck_state, "all arcs vetoed (grass)")
                elif stuck_state.get("force_forward", 0) > 0:
                    safe = cands[:1]
                elif stuck_state.get("pivot_attempts", 0) >= 4:
                    _begin_reverse(
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
    elif abs(np.degrees(bearing)) > ALIGN_DEG:
        lin, ang = _align_cmd(bearing)
        src = "align"
    else:
        sel = _selector() or selector
        if sel is not None:
            # Gate on scene difficulty: hazards visible, ambiguous corridor, or
            # a contested candidate set. On an empty pathway the planner is
            # already competent and a 1.6-2.1 s query would only cost tick rate.
            idx = sel.select(
                obs, cands, progress_m=_CRUMBS.travelled_m,
                hazard_frac=float(hard_m.mean()) if hard_m is not None else 0.0,
                corridor_width=corr_width,
            )
        else:
            idx = int(np.argmax([c.score for c in cands]))
        # M6 (GeNIE): the semantic veto is applied AFTER selection, and falls
        # back to the best-scoring traversable candidate. Geometry catches solid
        # things; this catches the surface distinctions geometry cannot make
        # (grass verge vs path, both perfectly flat).
        tv = _trav_veto()
        if tv is not None:
            new_idx, tv_note = tv.apply(obs, cands, idx)
            if new_idx != idx or tv_note != "ok":
                veto_note += f"trav[{tv_note}] "
            idx = new_idx
        chosen = cands[idx]
        wp = chosen.waypoints
        if posebuf is not None:
            # Pose is integrated from visual odometry at the top of the tick,
            # so this no longer degenerates to a pure rotation the way the old
            # push(stamp, 0.0, 0.0, yaw) did.
            wp = posebuf.realign(wp, chosen.stamp, tel.stamp)
        # M5b: hold the chosen path across ticks and advance it by measured
        # motion, so a single bad perception tick cannot swing the command and
        # the rover always has something to follow between plans.
        if USE_COMMIT:
            _COMMIT.advance(odom, 1.0 / CTRL_HZ)
            wp = _COMMIT.offer(wp)
        # Depth safety: override first waypoint if clearance is bad (ERC-3 style).
        if depth_view is not None and depth_view.turn_rad is not None and not depth_view.blocked:
            scale = max(0.25, float(depth_view.linear_scale))
            wp = wp.copy()
            wp[0, 0] = max(0.15, abs(np.cos(depth_view.turn_rad)) * scale)
            wp[0, 1] = np.sin(depth_view.turn_rad) * scale
        lin, ang = controller.command(wp)
        if path is not None and STAGE <= 1:
            ang = float(np.clip(ang + 0.10 * path.center_offset, -1.0, 1.0))
        # Proportional, not sign(): a bang-bang nudge applies its full value at
        # 1° of error just as at 39°, which under ~1.1 s of telemetry dead time
        # makes the heading hunt until it trips the pivot latch.
        #
        # And normalised by measured yaw authority: the same angular command
        # produces wildly different deg/s on different rovers, so a fixed gain
        # that is calm on a slow one makes a fast one circle instead of close
        # range. PulsePivot measures deg/s per unit angular; reuse it.
        # On a single-track path the LINE is the constraint and the GPS goal is
        # only a hint: the track may lead away from the goal bearing initially
        # and still be the only way through. Steer to the track when we can see
        # it, and let the goal bearing pull us along it.
        # M2 (UrbanVLA route-visual alignment): one tested policy for fusing the
        # GPS prior with what the camera sees, replacing the three ad-hoc blends
        # that used to live here. Matters more on this rover than in the paper:
        # lat/lon was measured bit-identical across 24 stationary samples AND
        # across 30 s of driving, so vision is the only live signal.
        aim = _route().align(
            goal_bearing=bearing,
            corridor_bearing=corr_bearing,
            corridor_width=corr_width,
            track_bearing=track_bearing,
            track_frac=track_frac,
            goal_trusted=not gps_stale,
        )
        steer_bearing = aim.bearing
        ang = float(np.clip(ang + _steer_nudge(stuck_state, steer_bearing), -1.0, 1.0))

        # Corridor OVERRIDE: when a drivable corridor is visible, it replaces
        # the goal-seeking angular command rather than nudging it. The goal
        # only biases the aim point WITHIN the corridor.
        if aim.source.startswith("corridor"):
            # RouteAligner already applied the bounded goal bias inside the
            # corridor; this block only converts that aim into a yaw command.
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
            # Narrow corridor => slow down; RouteAim already decided by how much.
            lin = float(lin * aim.speed_scale)
            src = f"corridor({corr_width:.2f}@{np.degrees(corr_bearing):+.0f}°)"
        # Same reasoning as the stop above: OutdoorTraversability has no
        # ground-plane removal, so with correctly-scaled depth it reads the
        # ground itself and returns linear_scale=0.0 -- which zeroed the
        # throttle on open pavement while the costmap reported 3.50 m clear
        # with ZERO obstacle points. Let it slow the rover only when the
        # costmap, which does remove the ground, is unavailable.
        if cm_res is None and depth_view is not None and depth_view.linear_scale < 1.0:
            lin = float(lin * depth_view.linear_scale)
        # Slow down as the nearest real obstacle approaches, so the stop at
        # COSTMAP_STOP_M is reached at a speed the dead time can absorb.
        if cm_res is not None and cm_res.forward_clearance_m < COSTMAP_SLOW_M:
            span = max(COSTMAP_SLOW_M - COSTMAP_STOP_M, 1e-3)
            t = (cm_res.forward_clearance_m - COSTMAP_STOP_M) / span
            lin = float(lin * float(np.clip(t, 0.25, 1.0)))
        src = "logonav" if chosen.score > 0 else "arc"

    if path is not None:
        tag = " unc" if path.uncertain else ""
        extra = (
            f"dirt={path.dirt_frac:.2f} grass={path.grass_frac:.2f} "
            f"bright={path.mean_bright:.0f}{tag} off={path.center_offset:+.2f} "
        )
    else:
        extra = "gate=off "
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
        f"USE_DEPTH={USE_DEPTH} LAMP={LAMP}"
    )

    wait_for_sdk(client)
    dg = _depth_gate()
    print(f"depth_gate={'ready' if dg else 'off/unavailable'}")
    _cmd(client, 0.0, 0.0)
    print(f"lamp={'on' if LAMP else 'off'}")

    goals, mission_mode = resolve_goals(client)
    if not goals:
        print("nothing to do")
        return

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
