"""
Tests for M5a, visual odometry and the breadcrumb trail.

VO is tested against SYNTHETIC ground imagery warped by a known motion, so the
ground truth is exact and the test runs without hardware. The warp is built from
the same calibration the estimator uses, which makes this a test of the
estimator rather than of the projection -- the projection itself is pinned
separately in test_calib.
"""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban.calib import CameraCalibration
from erc_urban.modules.odometry import (
    Breadcrumbs, OdomDelta, VisualOdometry, _kabsch_2d,
)

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _cal() -> CameraCalibration:
    return CameraCalibration(width=512, height=288, fx=170.0, fy=170.0,
                             cx=256.0, cy=144.0, height_m=0.148,
                             pitch_rad=math.radians(3.0), source="test")


def _render_ground(cal, dx: float = 0.0, dy: float = 0.0, dyaw: float = 0.0,
                   cell_m: float = 0.06, seed: int = 0) -> np.ndarray:
    """
    Render the camera view of a fixed ground pattern after the rover has moved
    by (dx, dy, dyaw) from the origin.

    Rendering beats warping an image: a warp of screen-space noise turns to
    aliased mush under perspective and LK loses it (measured: 1 surviving
    track). Rendering keeps the pattern sharp at every range because the
    texture lives in GROUND space, which is also what the real paving does.

    The pattern is a randomised checkerboard -- strong, uniquely-identifiable
    corners rather than the repeating grid that would alias into false matches.
    """
    h, w = cal.height, cal.width
    vv, uu = np.mgrid[0:h, 0:w]
    uv = np.stack([uu.ravel(), vv.ravel()], axis=1).astype(np.float64)
    xy, ok = cal.image_to_ground(uv)

    # Ground point as seen from the moved rover -> its position in world frame.
    c, s = math.cos(dyaw), math.sin(dyaw)
    R = np.array([[c, -s], [s, c]])
    world = xy @ R.T + np.array([dx, dy])

    gi = np.floor(world[:, 0] / cell_m).astype(np.int64)
    gj = np.floor(world[:, 1] / cell_m).astype(np.int64)
    rng = np.random.default_rng(seed)
    lut = rng.integers(30, 226, 4096, dtype=np.int64)
    val = lut[(gi * 7919 + gj * 104729) % 4096]

    img = np.where(ok, val, 0).reshape(h, w).astype(np.uint8)
    return np.stack([img] * 3, axis=2)


def _texture(cal, seed: int = 0) -> np.ndarray:
    return _render_ground(cal, seed=seed)


class TestKabsch(unittest.TestCase):
    def test_recovers_pure_rotation(self):
        P = np.array([[1.0, 0.0], [2.0, 0.5], [1.5, -0.4], [3.0, 0.2]])
        th = math.radians(12.0)
        R0 = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        R, t = _kabsch_2d(P, P @ R0.T)
        self.assertAlmostEqual(math.degrees(math.atan2(R[1, 0], R[0, 0])), 12.0, places=6)
        np.testing.assert_allclose(t, [0, 0], atol=1e-9)

    def test_recovers_pure_translation(self):
        P = np.array([[1.0, 0.0], [2.0, 0.5], [1.5, -0.4], [3.0, 0.2]])
        R, t = _kabsch_2d(P, P + np.array([0.3, -0.1]))
        np.testing.assert_allclose(t, [0.3, -0.1], atol=1e-9)


@unittest.skipUnless(_HAS_CV2, "requires opencv")
class TestVisualOdometry(unittest.TestCase):
    def test_stationary_reports_zero(self):
        """The property that matters most: no phantom motion. Verified on the
        real rover too -- 0.01 cm drift over 5 stationary frames."""
        cal = _cal()
        vo = VisualOdometry(cal)
        img = _render_ground(cal)
        vo.update(img)
        d = vo.update(img.copy())
        self.assertLess(d.distance_m, 0.005)
        self.assertLess(abs(math.degrees(d.dyaw)), 0.5)

    def test_first_frame_has_no_estimate(self):
        vo = VisualOdometry(_cal())
        d = vo.update(_texture(_cal()))
        self.assertEqual(d.reason, "first_frame")
        self.assertEqual(d.confidence, 0.0)

    def test_recovers_forward_translation(self):
        """
        Motions here are 0.02-0.06 m, the real per-tick range: the rover was
        measured at ~0.01-0.03 m per tick, and 0.3 m/s at 5 Hz is 0.06 m.

        Above ~0.1 m per frame the synthetic case loses tracking (near-field
        ground changes scale drastically under a 148 mm camera). Real frames are
        far more forgiving -- 749 tracks on live paving -- but the limit is real
        and is why `confidence` exists; see the note in odometry.py.
        """
        cal = _cal()
        img = _render_ground(cal)
        for truth in (0.02, 0.04, 0.06):
            vo = VisualOdometry(cal)
            vo.update(img)
            d = vo.update(_render_ground(cal, truth, 0.0, 0.0))
            self.assertGreater(d.confidence, 0.0, msg=f"no estimate at {truth} m")
            self.assertAlmostEqual(d.dx, truth, delta=0.3 * truth + 0.01)
            self.assertLess(abs(d.dy), 0.03)

    def test_forward_motion_has_correct_sign(self):
        """A sign flip here would make the breadcrumb trail point forwards, and
        a 'retreat' would drive into whatever the rover just stopped for."""
        cal = _cal()
        img = _render_ground(cal)
        vo = VisualOdometry(cal)
        vo.update(img)
        d = vo.update(_render_ground(cal, 0.05, 0.0, 0.0))
        self.assertGreater(d.confidence, 0.0)
        self.assertGreater(d.dx, 0.0)

    def test_large_motion_degrades_to_no_confidence_not_a_wrong_answer(self):
        """Failing loudly matters more than failing rarely: a confident wrong
        displacement would poison both the pose buffer and the trail."""
        cal = _cal()
        vo = VisualOdometry(cal)
        vo.update(_render_ground(cal))
        d = vo.update(_render_ground(cal, 0.40, 0.0, 0.0))
        self.assertTrue(d.confidence == 0.0 or abs(d.dx - 0.40) < 0.15)

    def test_rejects_implausible_jump(self):
        cal = _cal()
        vo = VisualOdometry(cal, max_step_m=0.25)
        vo.update(_texture(cal, seed=1))
        d = vo.update(_texture(cal, seed=99))     # unrelated scene
        self.assertEqual(d.confidence, 0.0)

    def test_featureless_ground_reports_no_confidence(self):
        cal = _cal()
        flat = np.full((cal.height, cal.width, 3), 128, np.uint8)
        vo = VisualOdometry(cal)
        vo.update(flat)
        d = vo.update(flat)
        self.assertEqual(d.confidence, 0.0)
        self.assertIn(d.reason, ("no_features", "tracking_lost", "few_ground_pts"))


class TestBreadcrumbs(unittest.TestCase):
    def _fwd(self, d=0.2, conf=1.0):
        return OdomDelta(dx=d, dy=0.0, dyaw=0.0, confidence=conf)

    def test_trail_falls_behind_after_driving_forward(self):
        bc = Breadcrumbs()
        for i in range(6):
            bc.update(self._fwd(), stamp=float(i))
        pts = np.array([[x, y] for x, y, _ in bc.pts])
        self.assertTrue((pts[:, 0] < 0).all(), "crumbs must be BEHIND the rover")

    def test_retreat_bearing_points_backwards(self):
        """Driving straight forward means the way back is straight behind."""
        bc = Breadcrumbs()
        for i in range(6):
            bc.update(self._fwd(), stamp=float(i))
        b = bc.retreat_bearing()
        self.assertIsNotNone(b)
        self.assertGreater(abs(math.degrees(b)), 150.0)

    def test_turning_then_reversing_recovers_the_curve(self):
        """After an arc, the way back is not straight behind -- which is the
        whole reason a trail beats a blind straight reverse."""
        bc = Breadcrumbs()
        for i in range(8):
            bc.update(OdomDelta(dx=0.2, dy=0.0, dyaw=math.radians(15.0),
                                confidence=1.0), stamp=float(i))
        b = bc.retreat_bearing()
        self.assertIsNotNone(b)
        self.assertLess(abs(math.degrees(b)), 179.0)

    def test_low_confidence_is_not_recorded(self):
        bc = Breadcrumbs()
        for i in range(6):
            bc.update(self._fwd(conf=0.0), stamp=float(i))
        self.assertEqual(len(bc.pts), 0)

    def test_no_trail_returns_none_rather_than_inventing_one(self):
        bc = Breadcrumbs()
        self.assertIsNone(bc.retreat_path())
        self.assertIsNone(bc.retreat_bearing())

    def test_trail_is_distance_bounded(self):
        bc = Breadcrumbs(max_distance_m=1.0)
        for i in range(30):
            bc.update(self._fwd(), stamp=float(i))
        for x, y, _ in bc.pts:
            self.assertLessEqual(math.hypot(x, y), 1.0 + 1e-9)

    def test_trail_is_age_bounded(self):
        bc = Breadcrumbs(max_age_s=3.0)
        for i in range(10):
            bc.update(self._fwd(d=0.02), stamp=float(i))
        self.assertTrue(all(9.0 - t <= 3.0 for _, _, t in bc.pts))

    def test_reset_clears(self):
        bc = Breadcrumbs()
        for i in range(5):
            bc.update(self._fwd(), stamp=float(i))
        bc.reset()
        self.assertEqual(len(bc.pts), 0)
        self.assertEqual(bc.travelled_m, 0.0)


if __name__ == "__main__":
    unittest.main()
