"""
Metric depth scale anchoring.

WHY THIS EXISTS
---------------
Monocular "metric" depth is only metric on scenes resembling its training set.
Measured on the live Mini+:

    vkitti checkpoint : whole-frame minimum 5.36 m, bottom row 5.62 m
    hypersim checkpoint: whole-frame minimum 1.14 m, bottom row 1.18 m

The bottom row is ground directly beneath a ~17 cm camera — call it 0.2-0.3 m.
vkitti is a car-mounted driving set (camera ~1.5 m, depths 5-80 m) and simply
emits its training prior here; its output range 5.36-79.69 m IS the vkitti
range. Swapping to hypersim helps a lot but still leaves ~5-7x.

That silently disabled the entire geometric safety layer: with clearance never
below ~5.4 m, `COSTMAP_STOP_M = 1.2` and `DepthObstacleGate.stop_m = 0.55` could
never fire. run.py's comments record the symptom ("depth hallucinates ~3 m at
true contact") without naming the cause.

THE FIX
-------
We know something the depth model does not: how high the camera is. The rover
drives on a ground plane, that plane is visible in almost every frame, and
`DepthCostmap._fit_ground` already RANSACs it every tick. So

    scale = height_true / height_from_plane_fit

is a per-frame observation of the model's scale error, and multiplying depth by
it makes the ground land where the ground actually is. This is a *gauge* fix, and
it works precisely because the error is a single global factor: the relative
depth structure that Depth Anything is genuinely good at is untouched.

Deliberately conservative, because scale drives the brakes:
  * EMA across frames — a single bad plane fit must not swing the stop distance.
  * Hard clamp — if the implied correction leaves the plausible band, the fit is
    wrong (looking at a wall, a ramp, or nothing), so hold the last good scale.
  * Never trusted until enough frames agree.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..calib import extrinsics_from_ground_plane


class GroundScaleAnchor:
    """Tracks the multiplicative scale error of a monocular metric depth model."""

    def __init__(
        self,
        camera_height_m: float,
        ema: float = 0.25,
        min_scale: float = 0.01,
        max_scale: float = 2.0,
        warmup_frames: int = 3,
        max_pitch_deg: float = 35.0,
    ):
        """
        camera_height_m: the true height of the camera above the ground. This is
            the one piece of outside knowledge the whole method needs. For the
            Mini+ the body is 19.5 cm and the camera sits near the top, so
            ~0.16-0.19 m; see erc_urban/camera_front.json.
        ema: smoothing on the scale estimate. Low, because this multiplies the
            distance at which the rover decides to stop.
        min_scale/max_scale: plausibility band for the correction. vkitti needs
            ~0.05 and hypersim ~0.17, so the band is wide, but a fit implying
            10x too CLOSE is a bad fit, not a real observation.
        max_pitch_deg: a plane this tilted relative to the camera is not the
            ground — more likely a wall or a ramp — so ignore it.
        """
        self.camera_height_m = float(camera_height_m)
        self.ema = ema
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.warmup_frames = warmup_frames
        self.max_pitch_deg = max_pitch_deg

        self.scale: Optional[float] = None
        self.n_updates = 0
        self.n_rejected = 0
        self.last_reason = "no_observation"
        self.last_raw: Optional[float] = None
        self.last_pitch_deg: Optional[float] = None

    @property
    def ready(self) -> bool:
        """True once enough consistent observations have accumulated."""
        return self.scale is not None and self.n_updates >= self.warmup_frames

    def observe(self, normal_cam: np.ndarray, offset_d: float) -> Optional[float]:
        """
        Feed one ground-plane fit, in the convention `DepthCostmap._fit_ground`
        returns: unit normal oriented up (camera frame), and `offset_d` such
        that a point's height above ground is `n . p - d`.

        Returns the current scale estimate, or None while unusable.
        """
        h_meas, pitch = extrinsics_from_ground_plane(normal_cam, offset_d)
        self.last_pitch_deg = float(np.degrees(pitch))

        if not np.isfinite(h_meas) or h_meas <= 1e-6:
            self.n_rejected += 1
            self.last_reason = "degenerate_plane"
            return self.scale
        if abs(self.last_pitch_deg) > self.max_pitch_deg:
            self.n_rejected += 1
            self.last_reason = f"not_ground(pitch={self.last_pitch_deg:.0f}deg)"
            return self.scale

        raw = self.camera_height_m / h_meas
        self.last_raw = float(raw)
        if not (self.min_scale <= raw <= self.max_scale):
            self.n_rejected += 1
            self.last_reason = f"scale_out_of_band({raw:.3f})"
            return self.scale

        self.scale = raw if self.scale is None else (
            (1.0 - self.ema) * self.scale + self.ema * raw
        )
        self.n_updates += 1
        self.last_reason = "ok"
        return self.scale

    def apply(self, depth: np.ndarray) -> np.ndarray:
        """
        Scale a depth map into true metres. Returns it unchanged until the
        estimate has warmed up — an unscaled map is wrong but at least
        consistently wrong, whereas a half-converged scale moves the brakes
        around unpredictably.
        """
        if not self.ready:
            return depth
        return depth * float(self.scale)

    def describe(self) -> str:
        if self.scale is None:
            return f"scale=--- ({self.last_reason})"
        return (
            f"scale={self.scale:.4f} (1/{1.0/max(self.scale,1e-9):.1f}x) "
            f"n={self.n_updates} rej={self.n_rejected} {self.last_reason}"
        )


class DepthResponseFit:
    """
    Two-parameter depth response correction, using the ground as a ruler.

    A single gain is not enough. Measured: after anchoring the scale so the
    ground under the camera reads correctly (0.19 m, right), an open courtyard
    clear for 20+ m reported 1.73 m of forward clearance with 15,700 "obstacle"
    points -- because hypersim squeezes the whole outdoor range into 0-20 m, so
    a gain that fixes the near field drags distant buildings into it. The error
    is compressive, not multiplicative.

    The fix is to stop treating the ground plane as a single anchor point and
    start treating it as a calibration TARGET. Given the camera's height, pitch
    and fx, the true distance to the ground at image row v is pure geometry --
    and the visible ground spans roughly 0.2 m to the horizon. That is a dense
    correspondence between raw depth and true depth across the entire range, for
    free, in every frame.

    We regress in inverse depth (disparity), where monocular networks are
    close to linear and where near-field accuracy is naturally weighted:

        1/z_true  =  a * (1/z_raw)  +  b

    Only ground pixels are used, identified by a caller-supplied mask (SegFormer
    road/sidewalk works well: 32.6% road on a real frame) and further restricted
    to below the calibrated horizon, since nothing above it can be ground.
    """

    def __init__(
        self,
        calib,
        ema: float = 0.3,
        min_points: int = 400,
        max_true_m: float = 25.0,
    ):
        self.calib = calib
        self.ema = ema
        self.min_points = min_points
        self.max_true_m = max_true_m
        self.a: Optional[float] = None
        self.b: Optional[float] = None
        self.n_updates = 0
        self.last_reason = "no_observation"
        self.last_resid_m: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self.a is not None

    def observe(self, depth: np.ndarray, ground_mask: np.ndarray) -> bool:
        """
        depth, ground_mask: (H,W), same shape. ground_mask True where the pixel
        is believed to be ground. Returns True if the fit was updated.
        """
        if depth.shape != ground_mask.shape:
            self.last_reason = "shape_mismatch"
            return False

        h, w = depth.shape
        cal = self.calib.scaled_to(w, h)
        vv, uu = np.nonzero(ground_mask)
        if len(vv) < self.min_points:
            self.last_reason = f"too_few_ground_px({len(vv)})"
            return False

        # True distance to each ground pixel, from geometry alone.
        xy, ok = cal.image_to_ground(np.stack([uu, vv], axis=1).astype(np.float64))
        z_true = np.hypot(xy[:, 0], xy[:, 1])
        z_raw = depth[vv, uu]
        good = (
            ok
            & np.isfinite(z_true) & (z_true > 0.05) & (z_true < self.max_true_m)
            & np.isfinite(z_raw) & (z_raw > 1e-3)
        )
        if int(good.sum()) < self.min_points:
            self.last_reason = f"too_few_valid({int(good.sum())})"
            return False

        d_true = 1.0 / z_true[good]
        d_raw = 1.0 / z_raw[good]

        # Robust line fit: least squares, then re-fit on the inlier half.
        A = np.stack([d_raw, np.ones_like(d_raw)], axis=1)
        sol, *_ = np.linalg.lstsq(A, d_true, rcond=None)
        resid = np.abs(A @ sol - d_true)
        keep = resid <= np.median(resid) * 2.0 + 1e-9
        if int(keep.sum()) >= self.min_points:
            sol, *_ = np.linalg.lstsq(A[keep], d_true[keep], rcond=None)

        a, b = float(sol[0]), float(sol[1])
        if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
            self.last_reason = f"degenerate_fit(a={a:.3g})"
            return False

        if self.a is None:
            self.a, self.b = a, b
        else:
            self.a = (1 - self.ema) * self.a + self.ema * a
            self.b = (1 - self.ema) * self.b + self.ema * b
        self.n_updates += 1

        z_fit = self.apply(z_raw[good])
        self.last_resid_m = float(np.median(np.abs(z_fit - z_true[good])))
        self.last_reason = "ok"
        return True

    def apply(self, depth: np.ndarray) -> np.ndarray:
        """Map raw depth to true metres. Identity until fitted."""
        if not self.ready:
            return depth
        d = 1.0 / np.maximum(np.asarray(depth, dtype=np.float64), 1e-6)
        inv = self.a * d + self.b
        # Non-positive inverse depth means "beyond anything we can resolve";
        # clamp to the far limit rather than emitting negative distances.
        return np.where(inv > 1e-6, 1.0 / np.maximum(inv, 1e-6), self.max_true_m)

    def describe(self) -> str:
        if not self.ready:
            return f"response=--- ({self.last_reason})"
        r = "n/a" if self.last_resid_m is None else f"{self.last_resid_m:.2f} m"
        return (f"response: 1/z_true={self.a:.3f}/z_raw{self.b:+.4f} "
                f"n={self.n_updates} ground-residual={r}")
