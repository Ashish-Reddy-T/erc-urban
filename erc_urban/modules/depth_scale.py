"""
Metric depth response correction.

Monocular "metric" depth is only metric on scenes resembling its training set.
Measured on this camera, the vkitti checkpoint bottoms out at 5.36 m and
hypersim at 1.14 m, where the true distance to the ground under a ~15 cm camera
is 0.2-0.3 m. That silently disabled the whole geometric safety layer: with
clearance never below ~5.4 m, no stop threshold could ever fire.

The fix uses what the model does not know — the camera's height, pitch and fx,
which make the true distance to any ground pixel pure geometry. See
DepthResponseFit for why a single gain is not enough.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class DepthResponseFit:
    """
    Two-parameter depth response correction, using the ground as a ruler.

    A single gain is not enough: the error is compressive, not multiplicative.
    Measured, a gain that put the ground under the camera at the right 0.19 m
    still reported 1.73 m of clearance in a courtyard clear for 20+ m, because
    hypersim squeezes the whole outdoor range into 0-20 m and dragged distant
    buildings into the near field.

    So treat the ground plane as a calibration TARGET, not one anchor point. It
    spans 0.2 m to the horizon, giving a dense raw->true correspondence in every
    frame. Regress in inverse depth, where monocular nets are near-linear and
    near-field accuracy is naturally weighted:

        1/z_true  =  a * (1/z_raw)  +  b

    Ground pixels come from a caller-supplied mask, restricted to below the
    calibrated horizon since nothing above it can be ground.
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
