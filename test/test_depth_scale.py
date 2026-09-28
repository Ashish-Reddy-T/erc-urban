"""Tests for the metric depth response correction (erc_urban.modules.depth_scale)."""
from __future__ import annotations

import unittest

import numpy as np

from erc_urban.calib import CameraCalibration
from erc_urban.modules.depth_scale import DepthResponseFit


def _make_calib(width=64, height=48):
    return CameraCalibration.from_fov(
        width=width, height=height, fov_horizontal_deg=90.0, height_m=0.15
    )


def _synthetic_scene(cal, a_true, b_true, min_v_frac=0.55):
    """
    Build a (depth, ground_mask) pair whose true distance-per-pixel comes from
    the calibration's own geometry, and whose raw depth is the KNOWN
    compressive response of that true distance:

        1/z_true = a_true * (1/z_raw) + b_true
        =>  z_raw = a_true / (1/z_true - b_true)

    Because both the fixture and DepthResponseFit.observe() derive z_true from
    the same `cal.image_to_ground`, this is an exact round trip: fitting must
    recover a_true/b_true (no noise).
    """
    w, h = cal.width, cal.height
    depth = np.zeros((h, w), dtype=np.float64)
    mask = np.zeros((h, w), dtype=bool)
    v0 = int(h * min_v_frac)
    uu, vv = np.meshgrid(np.arange(w), np.arange(v0, h))
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
    xy, ok = cal.image_to_ground(uv)
    z_true = np.hypot(xy[:, 0], xy[:, 1])
    inv_true = 1.0 / z_true
    good = ok & np.isfinite(inv_true) & (inv_true - b_true > 1e-6) & (z_true < 25.0)
    inv_raw = (inv_true - b_true) / a_true
    z_raw = np.where(good, 1.0 / np.maximum(inv_raw, 1e-9), 1.0)
    depth[v0:h, :] = z_raw.reshape(h - v0, w)
    mask[v0:h, :] = good.reshape(h - v0, w)
    return depth, mask, z_true.reshape(h - v0, w), v0


class TestDepthResponseFitBeforeFit(unittest.TestCase):
    def test_not_ready_initially(self):
        fit = DepthResponseFit(_make_calib())
        self.assertFalse(fit.ready)

    def test_apply_is_identity_before_fit(self):
        fit = DepthResponseFit(_make_calib())
        depth = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = fit.apply(depth)
        self.assertTrue(np.array_equal(out, depth))

    def test_describe_before_fit_reports_reason(self):
        fit = DepthResponseFit(_make_calib())
        self.assertIn("no_observation", fit.describe())


class TestDepthResponseFitRejects(unittest.TestCase):
    def test_shape_mismatch_is_rejected(self):
        fit = DepthResponseFit(_make_calib())
        depth = np.ones((10, 10))
        mask = np.ones((5, 5), dtype=bool)
        self.assertFalse(fit.observe(depth, mask))
        self.assertEqual(fit.last_reason, "shape_mismatch")
        self.assertFalse(fit.ready)

    def test_too_few_ground_pixels_is_rejected(self):
        cal = _make_calib()
        fit = DepthResponseFit(cal, min_points=400)
        depth = np.ones((cal.height, cal.width))
        mask = np.zeros((cal.height, cal.width), dtype=bool)
        mask[cal.height - 1, :5] = True  # far fewer than min_points
        self.assertFalse(fit.observe(depth, mask))
        self.assertTrue(fit.last_reason.startswith("too_few_ground_px"))
        self.assertFalse(fit.ready)


class TestDepthResponseFitRecoversKnownResponse(unittest.TestCase):
    def test_observe_recovers_a_and_b_from_a_noiseless_scene(self):
        cal = _make_calib()
        a_true, b_true = 0.6, 0.02
        depth, mask, _, _ = _synthetic_scene(cal, a_true, b_true)
        fit = DepthResponseFit(cal, min_points=200)
        updated = fit.observe(depth, mask)
        self.assertTrue(updated)
        self.assertTrue(fit.ready)
        self.assertAlmostEqual(fit.a, a_true, places=3)
        self.assertAlmostEqual(fit.b, b_true, places=3)
        self.assertEqual(fit.n_updates, 1)
        self.assertEqual(fit.last_reason, "ok")

    def test_apply_recovers_true_distance_after_fit(self):
        cal = _make_calib()
        a_true, b_true = 0.6, 0.02
        depth, mask, z_true, v0 = _synthetic_scene(cal, a_true, b_true)
        fit = DepthResponseFit(cal, min_points=200)
        fit.observe(depth, mask)
        corrected = fit.apply(depth[v0:, :])
        # Only compare where the fixture actually marked ground pixels valid.
        self.assertTrue(np.allclose(corrected[mask[v0:, :]], z_true[mask[v0:, :]], atol=1e-2))

    def test_second_observation_blends_with_ema(self):
        cal = _make_calib()
        fit = DepthResponseFit(cal, min_points=200, ema=0.3)

        depth1, mask1, _, _ = _synthetic_scene(cal, 0.6, 0.02)
        fit.observe(depth1, mask1)
        a1, b1 = fit.a, fit.b

        depth2, mask2, _, _ = _synthetic_scene(cal, 0.9, 0.05)
        # Compute what a lone fit on scene 2 would produce, for the expected blend.
        fit2 = DepthResponseFit(cal, min_points=200)
        fit2.observe(depth2, mask2)
        a2_solo, b2_solo = fit2.a, fit2.b

        fit.observe(depth2, mask2)
        self.assertEqual(fit.n_updates, 2)
        self.assertAlmostEqual(fit.a, 0.7 * a1 + 0.3 * a2_solo, places=3)
        self.assertAlmostEqual(fit.b, 0.7 * b1 + 0.3 * b2_solo, places=3)


if __name__ == "__main__":
    unittest.main()
