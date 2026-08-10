"""
Tests for camera calibration and projection.

Projection is the classic place where a sign error produces output that looks
entirely reasonable — paths drawn on the image, numbers in the right range —
while being mirrored or upside down. Every convention is therefore pinned by a
test that would fail loudly rather than degrade quietly.
"""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban.calib import (
    CameraCalibration,
    DEFAULT_FOV_DEG,
    extrinsics_from_ground_plane,
)


def _cam(**kw) -> CameraCalibration:
    base = dict(width=320, height=240, fov_horizontal_deg=DEFAULT_FOV_DEG,
                height_m=0.10, pitch_rad=0.0)
    base.update(kw)
    return CameraCalibration.from_fov(**base)


class TestIntrinsics(unittest.TestCase):
    def test_from_fov_matches_existing_costmap_arithmetic(self):
        """Adopting this class must not change what costmap.py already does
        (costmap.py:92 / ERC-3 depth_estimator.py:270)."""
        c = _cam()
        expected = 320 / (2.0 * math.tan(math.radians(90.0) / 2.0))
        self.assertAlmostEqual(c.fx, expected)
        self.assertAlmostEqual(c.fy, expected)     # square pixels
        self.assertAlmostEqual(c.cx, 160.0)
        self.assertAlmostEqual(c.cy, 120.0)

    def test_fov_round_trips(self):
        for fov in (60.0, 90.0, 120.0):
            self.assertAlmostEqual(_cam(fov_horizontal_deg=fov).fov_horizontal_deg,
                                   fov, places=6)

    def test_assumed_calibration_is_not_marked_measured(self):
        """Provenance must not lie: nothing downstream should trust a guess."""
        self.assertFalse(_cam().measured)
        c = _cam()
        c.source = "checkerboard"
        self.assertTrue(c.measured)

    def test_K_matrix_shape_and_entries(self):
        c = _cam()
        K = c.K()
        self.assertEqual(K.shape, (3, 3))
        self.assertAlmostEqual(K[0, 0], c.fx)
        self.assertAlmostEqual(K[1, 2], c.cy)
        self.assertAlmostEqual(K[2, 2], 1.0)

    def test_scaled_to_preserves_geometry(self):
        """Depth runs at 160x120, semantics at 512x512. A projection computed
        at one size must survive being moved to another."""
        c = _cam()
        big = c.scaled_to(640, 480)
        self.assertAlmostEqual(big.fov_horizontal_deg, c.fov_horizontal_deg, places=6)
        self.assertAlmostEqual(big.cx, 320.0)
        # Same ground point lands at the same RELATIVE image position.
        (uv0, _), (uv1, _) = c.project_ground([[3.0, 0.5]]), big.project_ground([[3.0, 0.5]])
        self.assertAlmostEqual(uv0[0, 0] / c.width, uv1[0, 0] / big.width, places=6)
        self.assertAlmostEqual(uv0[0, 1] / c.height, uv1[0, 1] / big.height, places=6)


class TestProjectionConventions(unittest.TestCase):
    def test_straight_ahead_projects_to_horizontal_centre(self):
        uv, valid = _cam().project_ground([[3.0, 0.0]])
        self.assertTrue(valid[0])
        self.assertAlmostEqual(float(uv[0, 0]), 160.0, places=6)

    def test_ground_is_below_the_optical_axis(self):
        """Level camera: the ground must appear in the lower half."""
        uv, _ = _cam().project_ground([[3.0, 0.0]])
        self.assertGreater(float(uv[0, 1]), 120.0)

    def test_left_in_robot_frame_is_left_in_image(self):
        """y is left-positive; image u increases rightward. Getting this
        backwards mirrors every overlay the VLM ever sees."""
        left, _ = _cam().project_ground([[3.0, 1.0]])
        right, _ = _cam().project_ground([[3.0, -1.0]])
        self.assertLess(float(left[0, 0]), 160.0)
        self.assertGreater(float(right[0, 0]), 160.0)

    def test_nearer_ground_is_lower_in_image(self):
        uv, _ = _cam().project_ground([[1.0, 0.0], [10.0, 0.0]])
        self.assertGreater(float(uv[0, 1]), float(uv[1, 1]))

    def test_points_behind_camera_are_invalid(self):
        """A backwards candidate must not draw as a confident forward path."""
        _, valid = _cam().project_ground([[-3.0, 0.0]])
        self.assertFalse(bool(valid[0]))

    def test_downward_pitch_raises_distant_ground(self):
        """Pinning the pitch sign. Tilting the camera down moves the horizon
        and distant ground UP the image (smaller v)."""
        level = _cam(pitch_rad=0.0)
        tilted = _cam(pitch_rad=math.radians(20.0))
        v_level, _ = level.project_ground([[8.0, 0.0]])
        v_tilt, _ = tilted.project_ground([[8.0, 0.0]])
        self.assertLess(float(v_tilt[0, 1]), float(v_level[0, 1]))

    def test_horizon_row_matches_a_very_distant_point(self):
        c = _cam(pitch_rad=math.radians(15.0))
        uv, _ = c.project_ground([[1e6, 0.0]])
        self.assertAlmostEqual(float(uv[0, 1]), c.horizon_v(), places=3)


class TestInverseProjection(unittest.TestCase):
    def test_ground_round_trip(self):
        for pitch in (0.0, math.radians(10.0), math.radians(25.0)):
            c = _cam(pitch_rad=pitch)
            pts = np.array([[2.0, 0.0], [5.0, 1.5], [3.0, -2.0], [1.0, 0.3]])
            uv, ok = c.project_ground(pts)
            back, ok2 = c.image_to_ground(uv)
            self.assertTrue(ok.all() and ok2.all(), msg=f"pitch={pitch}")
            np.testing.assert_allclose(back, pts, atol=1e-6)

    def test_above_horizon_is_invalid_not_a_huge_distance(self):
        """The failure that puts phantom drivable ground at 400 m and drives
        the rover into a wall."""
        c = _cam(pitch_rad=math.radians(10.0))
        above = c.horizon_v() - 5.0
        xy, valid = c.image_to_ground([[160.0, above]])
        self.assertFalse(bool(valid[0]))

    def test_below_horizon_is_valid_and_finite(self):
        c = _cam(pitch_rad=math.radians(10.0))
        xy, valid = c.image_to_ground([[160.0, c.horizon_v() + 40.0]])
        self.assertTrue(bool(valid[0]))
        self.assertGreater(float(xy[0, 0]), 0.0)
        self.assertTrue(np.isfinite(xy).all())


class TestGroundPlaneSelfCalibration(unittest.TestCase):
    """
    Recovering extrinsics from the plane DepthCostmap already fits, in the same
    convention `_fit_ground` returns (normal oriented up, height = n.p - d).
    """

    def test_level_camera(self):
        h, pitch = extrinsics_from_ground_plane(np.array([0.0, -1.0, 0.0]), -0.10)
        self.assertAlmostEqual(h, 0.10, places=9)
        self.assertAlmostEqual(pitch, 0.0, places=9)

    def test_pitched_camera_recovers_angle(self):
        for deg in (5.0, 15.0, 30.0):
            t = math.radians(deg)
            n = np.array([0.0, -math.cos(t), -math.sin(t)])
            h, pitch = extrinsics_from_ground_plane(n, -0.12)
            self.assertAlmostEqual(math.degrees(pitch), deg, places=6)
            self.assertAlmostEqual(h, 0.12, places=9)

    def test_unnormalised_normal_is_handled(self):
        h, pitch = extrinsics_from_ground_plane(np.array([0.0, -5.0, 0.0]), -0.10)
        self.assertAlmostEqual(pitch, 0.0, places=9)

    def test_round_trip_against_projection(self):
        """Self-calibrated extrinsics must reproduce the projection they came
        from — otherwise the two halves disagree and BEV drifts."""
        t = math.radians(18.0)
        c = _cam(pitch_rad=t, height_m=0.14)
        n = np.array([0.0, -math.cos(t), -math.sin(t)])
        h, pitch = extrinsics_from_ground_plane(n, -0.14)
        self.assertAlmostEqual(h, c.height_m, places=9)
        self.assertAlmostEqual(pitch, c.pitch_rad, places=9)


class TestPersistence(unittest.TestCase):
    def test_save_load_round_trip(self):
        import tempfile, os

        c = _cam(pitch_rad=0.2, height_m=0.13)
        c.source = "checkerboard"
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cam.json")
            c.save(p)
            back = CameraCalibration.load(p)
            self.assertEqual(back.source, "checkerboard")
            self.assertAlmostEqual(back.pitch_rad, 0.2)
            self.assertAlmostEqual(back.fx, c.fx)

    def test_load_or_default_falls_back_cleanly(self):
        c = CameraCalibration.load_or_default("/nonexistent/cam.json", 320, 240)
        self.assertFalse(c.measured)
        self.assertEqual((c.width, c.height), (320, 240))


if __name__ == "__main__":
    unittest.main()
