"""
LogoNav planner adapter.

Loads IL_gps / LogoNav from ERC-3-earthrover-challenge/mbra_repo, runs the same
GPS-conditioned inference, and emits robot-frame waypoint Candidates. Falls back
to BaselinePlanner arcs when weights or torch are missing.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image as PILImage

from ..interfaces import Candidate, Observation
from .baseline import BaselinePlanner

# Vendored colleague outdoor LogoNav tree (cloned into this workspace).
_ERC3 = Path(__file__).resolve().parents[2] / "ERC-3-earthrover-challenge"
_MBRA_DEPLOY = _ERC3 / "mbra_repo" / "deployment"
_MBRA_TRAIN = _ERC3 / "mbra_repo" / "train"
_DEFAULT_WEIGHTS = _MBRA_DEPLOY / "model_weights" / "logonav.pth"
_DEFAULT_CONFIG = _MBRA_TRAIN / "config" / "LogoNav.yaml"


def _ensure_mbra_path() -> None:
    for p in (_MBRA_DEPLOY, _MBRA_TRAIN):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _clip_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _rotate_local(dx: float, dy: float, heading_rad: float) -> tuple[float, float]:
    # Same as ERC-3 outdoor_logonav_controller / LogoNav_frodobot.
    rx = dx * math.cos(heading_rad) + dy * math.sin(heading_rad)
    ry = -dx * math.sin(heading_rad) + dy * math.cos(heading_rad)
    return rx, ry


class LogoNavPlanner:
    """
    GPS-conditioned LogoNav -> Candidates.

    The LogoNav trajectory scores highest; the remaining slots are
    BaselinePlanner arcs, so a gate still has grass-crossing paths to reject.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "auto",
        image_size: tuple[int, int] = (96, 96),
        metric_waypoint_spacing: float = 0.25,
        goal_distance_cap_m: float = 30.0,
        fallback: Optional[BaselinePlanner] = None,
    ):
        self.checkpoint_path = Path(
            checkpoint_path
            or os.environ.get("LOGONAV_CKPT", str(_DEFAULT_WEIGHTS))
        )
        self.config_path = Path(
            config_path
            or os.environ.get("LOGONAV_CONFIG", str(_DEFAULT_CONFIG))
        )
        self.image_size = image_size
        self.metric_waypoint_spacing = metric_waypoint_spacing
        self.goal_distance_cap_m = goal_distance_cap_m
        self.fallback = fallback or BaselinePlanner()
        self.goal_latlon: Optional[tuple[float, float]] = None
        self.context_queue: list[PILImage.Image] = []
        self.model = None
        self.context_size = 5
        self.device = None
        self._load_error: Optional[str] = None
        self._warned_slow_path = False
        self._try_load(device)

    def _try_load(self, requested: str) -> None:
        if not self.checkpoint_path.is_file():
            self._load_error = f"missing weights: {self.checkpoint_path}"
            print(f"[LogoNavPlanner] {self._load_error} — using BaselinePlanner")
            return
        if not self.config_path.is_file():
            self._load_error = f"missing config: {self.config_path}"
            print(f"[LogoNavPlanner] {self._load_error} — using BaselinePlanner")
            return
        try:
            import torch
            import yaml
            import utm  # noqa: F401 — needed at propose time
            _ensure_mbra_path()
            from utils_logonav import load_model  # type: ignore

            if requested == "auto":
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            else:
                self.device = torch.device(requested)
            params = yaml.safe_load(self.config_path.read_text())
            self.context_size = int(params["context_size"])
            self.model = load_model(
                str(self.checkpoint_path), params, self.device
            )
            self.model = self.model.to(self.device).eval()
            print(
                f"[LogoNavPlanner] loaded {self.checkpoint_path.name} "
                f"on {self.device}"
            )
        except Exception as e:
            self.model = None
            self._load_error = str(e)
            print(f"[LogoNavPlanner] load failed ({e}) — using BaselinePlanner")

    @property
    def ready(self) -> bool:
        return self.model is not None

    def set_goal(self, lat: float, lon: float) -> None:
        self.goal_latlon = (float(lat), float(lon))
        self.context_queue = []

    def reset(self) -> None:
        self.context_queue = []

    def propose(self, obs: Observation, k: int) -> list[Candidate]:
        base = self.fallback.propose(obs, k)
        logo = self._logo_candidate(obs)
        if logo is None:
            return base
        # Put LogoNav first with a high score; keep remaining baseline arcs.
        return ([logo] + base)[: max(k, 1)]

    # ---- reused by the K-candidate sampler (modules/candidates.py) ---------

    def update_context(self, obs: Observation) -> bool:
        """
        Push this frame into the context queue; True once it is deep enough to
        run the model.

        Split out of `_logo_candidate` so the multi-hypothesis sampler advances
        the SAME queue exactly once per tick — calling it twice feeds the model
        duplicate frames and corrupts the temporal context.
        """
        if self.model is None:
            return False
        image = PILImage.fromarray(obs.front.astype(np.uint8), mode="RGB")
        image = image.resize(self.image_size).convert("RGB")
        if len(self.context_queue) < self.context_size + 1:
            self.context_queue.append(image)
        else:
            self.context_queue.pop(0)
            self.context_queue.append(image)
        return len(self.context_queue) > self.context_size

    def goal_pose(
        self,
        obs: Observation,
        lateral_offset_m: float = 0.0,
        forward_offset_m: float = 0.0,
    ) -> Optional[np.ndarray]:
        """
        The model's 4-vector conditioning input, optionally aimed at a point
        offset from the true GPS goal.

        Layout, verified against upstream and the training dataloader:

            [forward_m / 0.25, left_m / 0.25, cos(dheading), sin(dheading)]

        Both the action labels and the goal position are divided by
        `metric_waypoint_spacing = 0.25` in training, which is why the same
        constant de-normalises the output in `forward_batch`.

        `lateral_offset_m` shifts the aim point sideways (left-positive) — the
        diversity axis the K-candidate scheme rides on: it asks the policy what
        it would do if the goal were a metre to the left.
        """
        if self.goal_latlon is None:
            return None
        if np.isnan(obs.lat) or np.isnan(obs.lon) or obs.yaw is None:
            return None
        try:
            import utm
        except Exception:
            return None

        cur = utm.from_latlon(float(obs.lat), float(obs.lon))
        goal = utm.from_latlon(self.goal_latlon[0], self.goal_latlon[1])

        # SDK orientation: degrees CW from North → LogoNav compass convention.
        cur_compass = -math.degrees(float(obs.yaw)) / 180.0 * math.pi

        dx, dy = float(goal[0]) - float(cur[0]), float(goal[1]) - float(cur[1])
        relative_x, relative_y = _rotate_local(dx, dy, cur_compass)
        # Upstream stores the pair rotated; name them for what they are, so the
        # offsets below are obviously in robot frame rather than a swapped one.
        fwd, left = relative_y, -relative_x

        # Cap to the trained range BEFORE perturbing. The cap is a proportional
        # rescale, so offsetting first shrinks the offset by the same factor: a
        # checkpoint 111 m out would turn a 2 m lateral offset into 0.54 m, and
        # the candidate set degenerates to near-duplicates exactly when the goal
        # is far and diversity matters most.
        dist = math.hypot(fwd, left)
        if dist > self.goal_distance_cap_m and dist > 1e-6:
            scale = self.goal_distance_cap_m / dist
            fwd, left = fwd * scale, left * scale

        # Applied to the capped point, so an offset means what it says in
        # metres. The overshoot (30 m aim + 3 m lateral = 30.1 m) is well inside
        # the policy's range.
        fwd += forward_offset_m
        left += lateral_offset_m

        s = self.metric_waypoint_spacing
        return np.array(
            [
                fwd / s,
                left / s,
                np.cos(0.0 - cur_compass),
                np.sin(0.0 - cur_compass),
            ],
            dtype=np.float32,
        )

    def forward_batch(self, goal_poses: np.ndarray) -> Optional[np.ndarray]:
        """
        Run the policy on B goal poses at once -> (B, T, 2) robot-frame metres.

        The B poses share one image context and the vision encoder is the
        expensive half, so batching the model's own `forward` would re-encode
        the same frames B times — measured 971 ms at B=11 on CPU against a
        200 ms tick. Encode once, fan out only the cheap decoder head.

        Falls back to the plain batched forward when the fast path cannot run,
        so an upstream change can only cost speed, never correctness.
        """
        if self.model is None or len(self.context_queue) <= self.context_size:
            return None
        try:
            import torch
            from utils_logonav import to_numpy, transform_images_mbra  # type: ignore
        except Exception:
            return None

        gp = np.atleast_2d(np.asarray(goal_poses, dtype=np.float32))
        b = gp.shape[0]

        obs_images = transform_images_mbra(self.context_queue)
        obs_images = torch.split(obs_images, 3, dim=1)
        obs_images = torch.cat(obs_images, dim=1).to(self.device)
        goal_t = torch.from_numpy(gp).to(self.device)

        with torch.no_grad():
            try:
                enc = self._encode_context(obs_images)          # (1, ctx+1, E)
                waypoints = self._decode(enc.expand(b, -1, -1), goal_t)
            except Exception as e:
                if not self._warned_slow_path:
                    self._warned_slow_path = True
                    print(f"[LogoNavPlanner] fast path unavailable ({e}) — "
                          "re-encoding per candidate, expect slow ticks")
                waypoints = self.model(
                    obs_images.expand(b, *obs_images.shape[1:]), goal_t
                )
        wps = to_numpy(waypoints)  # (B, T, 4): dx, dy, hx, hy
        # Robot-frame metres (x_forward, y_left) — matches ERC-3 twist decode.
        return np.asarray(wps[:, :, :2], dtype=np.float64) * self.metric_waypoint_spacing

    # The next two mirror IL_gps.forward, split at the goal-conditioning point.
    # Kept literal against upstream so a side-by-side diff is easy;
    # test_candidates asserts they reproduce model(...) exactly.
    def _encode_context(self, obs_img):
        """Vision half of IL_gps.forward — independent of the goal, so once."""
        import torch

        m = self.model
        stacked = torch.concat(torch.split(obs_img, 3, dim=1), dim=0)
        e = m.obs_encoder.extract_features(stacked)
        e = m.obs_encoder._avg_pooling(e)
        if m.obs_encoder._global_params.include_top:
            e = e.flatten(start_dim=1)
            e = m.obs_encoder._dropout(e)
        e = m.compress_obs_enc(e)
        e = e.reshape((m.context_size + 1, -1, m.obs_encoding_size))
        return torch.transpose(e, 0, 1)                        # (B, ctx+1, E)

    def _decode(self, obs_encoding, goal_pose):
        """Goal-conditioned half — the part that varies per candidate."""
        import torch
        import torch.nn.functional as F

        m = self.model
        goal_encoding = m.local_goal(goal_pose).unsqueeze(1)
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)
        action_pred = m.action_predictor(m.decoder(tokens))
        action_pred = action_pred.reshape(
            (action_pred.shape[0], m.len_trajectory_pred, m.num_action_params)
        )
        # Deltas -> waypoints, then unit-normalise the heading pair. The clone
        # avoids an in-place write on an expanded view.
        action_pred = torch.cat(
            [
                torch.cumsum(action_pred[:, :, :2], dim=1),
                F.normalize(action_pred[:, :, 2:].clone(), dim=-1),
            ],
            dim=-1,
        )
        return action_pred

    def score_waypoints(self, xy: np.ndarray, obs: Observation) -> float:
        """
        Planner confidence: how well the endpoint bearing agrees with the goal.

        Offset by 2.0 so any model path outranks the geometric arcs, which score
        <= 0 (see BaselinePlanner.propose).
        """
        end_bearing = float(np.arctan2(xy[-1, 1], max(xy[-1, 0], 1e-3)))
        return 2.0 - abs(_clip_angle(end_bearing - obs.goal_bearing))

    def _logo_candidate(self, obs: Observation) -> Optional[Candidate]:
        if not self.update_context(obs):
            return None  # no model, or still warming up — baseline arcs drive
        gp = self.goal_pose(obs)
        if gp is None:
            return None
        out = self.forward_batch(gp[None, :])
        if out is None:
            return None
        xy = out[0]
        return Candidate(
            waypoints=xy, score=self.score_waypoints(xy, obs), stamp=obs.stamp
        )
