"""
Depth-based obstacle gate — colleague ERC-3 / Depth Anything V2.

Official FrodoBots SDK exposes only RGB + GPS (+ lamp). There is no LiDAR /
ultrasonic. Real obstacle clearance in the outdoor stack is monocular metric
depth (Depth Anything V2) → polar bins → stop/turn, same as
ERC-3 outdoor_traversability + depth_estimator.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ERC3_SRC = Path(__file__).resolve().parents[2] / "ERC-3-earthrover-challenge" / "src"


@dataclass
class DepthClearance:
    blocked: bool
    forward_m: float
    reason: str
    linear_scale: float = 1.0
    turn_rad: Optional[float] = None


class DepthObstacleGate:
    """Lazy-loads DepthEstimator + OutdoorTraversability from ERC-3."""

    def __init__(
        self,
        model_size: str = "small",
        stop_m: float = 0.55,
        slow_m: float = 1.20,
        obstacle_m: float = 1.50,
        every_n: int = 2,
        device: str = "auto",
        checkpoint_domain: str = "indoor",
    ):
        self.model_size = model_size
        self.stop_m = stop_m
        self.slow_m = slow_m
        self.obstacle_m = obstacle_m
        self.every_n = max(1, every_n)
        self.device = device
        # 'indoor' (hypersim, max_depth 20) not 'outdoor' (vkitti, max_depth 80).
        # Measured on the live Mini+: the vkitti checkpoint's whole-frame minimum
        # is 5.36 m and ground directly under a 148 mm camera reads 5.62 m --
        # its output range IS the vkitti range, because vkitti is a car-mounted
        # driving set and this camera is 10x lower. With clearance never below
        # ~5.4 m, stop_m=0.55 and COSTMAP_STOP_M=1.2 could never fire: the
        # geometric safety layer was silently dead. hypersim gives 1.14-19.61 m
        # on the same frame. Residual scale error is handled by DepthResponseFit.
        self.checkpoint_domain = checkpoint_domain
        self._estimator = None
        self._trav = None
        self._tick = 0
        self._last: Optional[DepthClearance] = None
        # Most recent metric depth map, exposed so the costmap can reuse it
        # instead of running Depth Anything a second time per tick. This is the
        # CORRECTED map when a response fit is attached.
        self.last_depth = None
        # Uncorrected map, kept so the response fit can be re-observed against
        # it. Fitting on already-corrected depth would compound the correction.
        self.last_raw_depth = None
        # Optional DepthResponseFit (modules/depth_scale.py). Applied inside
        # evaluate() so this gate's own stop distance and the costmap's shared
        # copy are corrected identically -- two consumers disagreeing about
        # scale is how a "safe" clearance becomes a collision report.
        self.response_fit = None
        self._load_error: Optional[str] = None
        self._try_load()

    @property
    def ready(self) -> bool:
        return self._estimator is not None and self._trav is not None

    def _try_load(self) -> None:
        try:
            s = str(_ERC3_SRC)
            if s not in sys.path:
                sys.path.insert(0, s)
            from depth_estimator import DepthEstimator  # type: ignore
            from outdoor_traversability import (  # type: ignore
                OutdoorTraversability,
                OutdoorTraversabilityConfig,
            )

            dev = None if self.device == "auto" else self.device
            try:
                self._estimator = DepthEstimator(
                    model_size=self.model_size,
                    device=dev,
                    checkpoint_domain=self.checkpoint_domain,
                )
            except Exception as e:
                # Never lose the obstacle gate over an accelerator quirk —
                # a slow gate beats no gate.
                if dev in (None, "cpu"):
                    raise
                print(f"[DepthObstacleGate] {dev} failed ({e}) — retrying on cpu")
                self.device = "cpu"
                self._estimator = DepthEstimator(
                    model_size=self.model_size,
                    device="cpu",
                    checkpoint_domain=self.checkpoint_domain,
                )
            cfg = OutdoorTraversabilityConfig(
                obstacle_distance_m=self.obstacle_m,
                stop_distance_m=self.stop_m,
                slow_distance_m=self.slow_m,
                memory_frames=3,
            )
            self._trav = OutdoorTraversability(cfg)
            print(
                f"[DepthObstacleGate] ready ({self.model_size}) "
                f"stop<{self.stop_m}m"
            )
        except Exception as e:
            self._estimator = None
            self._trav = None
            self._load_error = str(e)
            print(f"[DepthObstacleGate] unavailable ({e})")

    def evaluate(self, front: np.ndarray, goal_bearing: float) -> DepthClearance:
        if not self.ready:
            return DepthClearance(False, 99.0, "depth_off")

        self._tick += 1
        if self._last is not None and (self._tick % self.every_n) != 0:
            return self._last

        try:
            # 16:9 to match the 1024x576 frame. The old (120, 160) is 4:3, so it
            # squashed the frame vertically by 1.33x -- and both DepthCostmap
            # and ERC-3 backproject assuming SQUARE pixels (fy = fx). Every
            # height and range computed from that map was therefore wrong,
            # manufacturing obstacles out of flat ground.
            depth = self._estimator.estimate(front, target_size=(144, 256))
            self.last_raw_depth = depth
            if self.response_fit is not None and self.response_fit.ready:
                depth = self.response_fit.apply(depth)
            self.last_depth = depth
            trav = self._trav.compute(depth, goal_bearing)
            fwd = float(trav.forward_clearance)
            if trav.all_blocked or trav.linear_scale <= 0.0 or fwd < self.stop_m:
                out = DepthClearance(
                    True, fwd, f"depth_stop({fwd:.2f}m)",
                    0.0, trav.angular_override,
                )
            elif trav.forward_blocked:
                out = DepthClearance(
                    False, fwd, f"depth_slow({fwd:.2f}m)",
                    float(trav.linear_scale), trav.angular_override,
                )
            else:
                out = DepthClearance(
                    False, fwd, f"depth_ok({fwd:.2f}m)",
                    float(trav.linear_scale), None,
                )
            self._last = out
            return out
        except Exception as e:
            return DepthClearance(False, 99.0, f"depth_err:{e}")
