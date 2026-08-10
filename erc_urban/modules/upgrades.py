"""
UPGRADE MODULES. Each class honors an interface from interfaces.py so it
drops into the Stage-0 loop with a one-line swap. Where a class wraps an
external repo whose exact API I cannot verify from here, the integration
point is marked `# >>> WIRE:` — that is the ONE line you edit after cloning
the repo. I refuse to invent function signatures around those points; a
wrong guess there costs you more than the stub saves.

Provenance per module is in each docstring.
"""
from __future__ import annotations
import time
import threading
import numpy as np
from ..interfaces import Candidate, Observation


# =====================================================================
# M1 UPGRADE — learned multi-hypothesis planner (MBRA/LogoNav or NoMaD)
#   repo: general-navigation-models (ViNT/NoMaD)  https://github.com/robodhruv/visualnav-transformer
#   repo: MBRA/LogoNav (Berkeley)  — checkpoints via the paper's project page
#   Why: native to FrodoBots-2K/7K (MBRA-labeled) and emits the K-candidate
#   set the whole downstream stack needs. This is the KEYSTONE risk: verify
#   you can get calibrated multi-hypothesis output before building M4/M5.
# =====================================================================
class LearnedPlanner:
    """
    M1 upgrade stub. Prefer STAGE=1 LogoNavPlanner (erc_urban.modules.logonav)
    which wraps the colleague ERC-3 outdoor LogoNav weights.

    For STAGE=3 / custom CKPT, set CKPT=... and wire sample() below.
    """
    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 horizon_m: float = 3.0, n_pts: int = 8):
        self.device = device
        self.horizon_m = horizon_m
        self.n_pts = n_pts
        self._ckpt = checkpoint_path
        # Prefer the working LogoNav adapter when CKPT points at logonav.pth.
        try:
            from .logonav import LogoNavPlanner
            self._logo = LogoNavPlanner(
                checkpoint_path=checkpoint_path, device=device
            )
        except Exception:
            self._logo = None
        self.model = None if self._logo is None or not self._logo.ready else True

    def set_goal(self, lat: float, lon: float) -> None:
        if self._logo is not None:
            self._logo.set_goal(lat, lon)

    def propose(self, obs: Observation, k: int) -> list[Candidate]:
        if self._logo is not None and self._logo.ready:
            return self._logo.propose(obs, k)
        raise RuntimeError(
            "LearnedPlanner not wired. Use STAGE=1 (LogoNavPlanner) or place "
            "logonav.pth and set CKPT / LOGONAV_CKPT."
        )


# =====================================================================
# M4 + M5(slow) — VLM candidate selector + latency-resilient fusion
#   paper: Slow Brain, Fast Planner (arXiv 2606.20458)
#   Why: closes the trajectory-scoring gap at junctions/pedestrians and is
#   TRAINING-FREE. Uses their two verified tricks:
#     (a) render K candidates as numbered overlays, VLM picks an index;
#     (b) Probability Fusion with exponential staleness decay so a 1-3s
#         stale pick still drives a 5-20Hz loop.
#   Their ablations, baked in as defaults below: hide goal+scores from the
#   VLM; K in [12,18]; Gemini-2.5-Flash-Lite for latency; Probability (not
#   Score) Fusion for stability.
# =====================================================================
class VLMSelectorFusion:
    def __init__(self, k: int = 18, tau_decay_s: float = 5.0, lam: float = 1.0,
                 query_hz: float = 1.0, vlm_call=None):
        self.k = k
        self.tau = tau_decay_s
        self.lam = lam
        self.query_dt = 1.0 / query_hz
        # >>> WIRE: vlm_call(image_with_overlay_np, k) -> int index.
        # Wrap your VLM API (Gemini/GPT/Qwen) here. Prompt: hide goal &
        # planner scores; ask only which numbered path is best & safe.
        self.vlm_call = vlm_call
        self._last = {"idx": None, "waypoints": None, "stamp": -1e9}
        self._lock = threading.Lock()
        self._last_query = -1e9

    # ---- async slow loop: fire-and-forget streaming queries -------------
    def maybe_query(self, obs: Observation, cands: list[Candidate]) -> None:
        now = time.monotonic()
        if self.vlm_call is None or now - self._last_query < self.query_dt:
            return
        self._last_query = now
        overlay = _render_overlay(obs.front, cands)  # numbered colored polylines
        def _run():
            try:
                idx = int(self.vlm_call(overlay, len(cands)))
                with self._lock:
                    self._last = {"idx": idx,
                                  "waypoints": cands[idx].waypoints.copy(),
                                  "stamp": obs.stamp}
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    # ---- fast loop: fuse stale VLM pick into fresh candidate scores -----
    def select(self, obs: Observation, cands: list[Candidate]) -> int:
        self.maybe_query(obs, cands)
        with self._lock:
            last = dict(self._last)
        if last["waypoints"] is None:
            return int(np.argmax([c.score for c in cands]))  # planner fallback

        dt = obs.stamp - last["stamp"]
        w = np.exp(-dt / self.tau)  # exponential staleness decay
        # motion-compensate omitted for brevity: compare in current frame via
        # horizon-truncated similarity (paper Eq.1). Odometry compensation
        # plugs in at PoseBuffer (M5 transport) below.
        planner_scores = np.array([c.score for c in cands], float)
        sims = -np.array([_traj_dist(c.waypoints, last["waypoints"])
                          for c in cands])
        # Probability Fusion (bounded influence): mix softmaxes, alpha<=1
        p_pl = _softmax(planner_scores)
        p_vlm = _softmax(sims)
        alpha = (self.lam / (self.lam + 1.0)) * w
        fused = (1 - alpha) * p_pl + alpha * p_vlm
        return int(np.argmax(fused))


# =====================================================================
# M5(transport) — SE(2) pose-buffer realignment (AsyncShield)
#   Why: de-stales the CHOSEN waypoint for the 4G round-trip using the
#   odometry delta between capture and now. Needs only IMU/odom (which we
#   have) — no onboard compute, which we don't.
# =====================================================================
class PoseBuffer:
    """
    AsyncShield's temporal pose buffer: turn "instruction lag" into a
    deterministic "spatial offset" via an SE(2) transform.

    Feed it with `integrate()` from visual odometry. run.py used to call
    `push(stamp, 0.0, 0.0, yaw)` with x and y HARDCODED TO ZERO, so the
    translation term was identically zero and `realign` did nothing but rotate
    -- the module appeared wired while performing half its job. The odometry
    the plan assumed (wheel RPM, yaw rate) does not exist on this rover: `rpms`
    reads [0,0,0,0] permanently and the compass suffers magnetometer
    interference. modules/odometry.py supplies the real thing.
    """

    def __init__(self, maxlen: int = 256):
        from collections import deque
        self.buf = deque(maxlen=maxlen)  # (stamp, x, y, yaw) in odom frame
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

    def push(self, stamp, x, y, yaw):
        self.buf.append((stamp, x, y, yaw))

    def integrate(self, stamp, delta) -> None:
        """
        Accumulate one OdomDelta into an odom-frame pose and record it.

        The delta is expressed in the robot frame at the START of the step, so
        it must be rotated into the odom frame before accumulating -- adding it
        directly would make every turn corrupt the track.
        """
        if delta is None or delta.confidence <= 0.0:
            # Still record the timestamp: a gap with no motion estimate is
            # better represented as "no movement observed" than as a hole,
            # which `_nearest` would silently bridge with a stale pose.
            self.push(stamp, self.x, self.y, self.yaw)
            return
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self.x += c * delta.dx - s * delta.dy
        self.y += s * delta.dx + c * delta.dy
        self.yaw = _wrap(self.yaw + delta.dyaw)
        self.push(stamp, self.x, self.y, self.yaw)

    def realign(self, waypoints: np.ndarray, t_capture: float,
                t_now: float) -> np.ndarray:
        """Re-express capture-time waypoints in the current robot frame."""
        p0 = self._nearest(t_capture)
        p1 = self._nearest(t_now)
        if p0 is None or p1 is None:
            return waypoints
        dx, dy = p1[1] - p0[1], p1[2] - p0[2]
        dyaw = _wrap(p1[3] - p0[3])
        c, s = np.cos(-dyaw), np.sin(-dyaw)
        R = np.array([[c, -s], [s, c]])
        shifted = waypoints - np.array([dx, dy])
        return shifted @ R.T

    def _nearest(self, t):
        if not self.buf:
            return None
        return min(self.buf, key=lambda p: abs(p[0] - t))


# =====================================================================
# M6 — traversability veto (GeNIE-style SAM2 costmap)
#   repo: SAM2 (facebookresearch/sam2). GeNIE fuses a generalizable
#   traversability head; here we expose the veto interface. Marked stub
#   because the segmentation->costmap->robot-frame projection depends on
#   your camera intrinsics/extrinsics, which I can't guess.
# =====================================================================
class TraversabilityGate:
    def __init__(self, sam2_predictor=None, intrinsics=None, extrinsics=None):
        # >>> WIRE: load SAM2 predictor + your camera calibration.
        self.seg = sam2_predictor
        self.K = intrinsics
        self.T = extrinsics

    def mask(self, obs: Observation, cands) -> np.ndarray:
        if self.seg is None:
            return np.ones(len(cands), dtype=bool)  # no-op until wired
        # >>> WIRE: segment walkable surface on obs.front, project each
        # candidate's waypoints to image, mark False if any point lands on
        # non-traversable (road/grass/obstacle). Return boolean mask.
        raise NotImplementedError


# ---------------- small helpers ----------------
def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def _traj_dist(a, b):
    n = min(len(a), len(b))
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))

def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def _render_overlay(front, cands):
    # Minimal overlay so the WIRE point has something to pass. Replace with a
    # proper camera-projected polyline renderer once intrinsics are known.
    return front