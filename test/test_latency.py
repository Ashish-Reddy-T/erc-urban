"""
Tests for M5: SE(2) pose buffer, commit window, staleness decay.

The pose-buffer tests exist mostly to stop a specific regression: run.py fed it
`(stamp, 0.0, 0.0, yaw)` for its whole life, so the translation term was
identically zero and the module looked wired while doing half its job.
"""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban.modules.commit import CommitWindow
from erc_urban.modules.odometry import OdomDelta
from erc_urban.modules.upgrades import PoseBuffer, VLMSelectorFusion


def _d(dx=0.0, dy=0.0, dyaw=0.0, conf=1.0):
    return OdomDelta(dx=dx, dy=dy, dyaw=dyaw, confidence=conf)


class TestPoseBufferIntegration(unittest.TestCase):
    def test_straight_line_accumulates_forward(self):
        pb = PoseBuffer()
        for i in range(5):
            pb.integrate(float(i), _d(dx=0.2))
        self.assertAlmostEqual(pb.x, 1.0, places=9)
        self.assertAlmostEqual(pb.y, 0.0, places=9)

    def test_translation_term_is_not_zero(self):
        """The regression. If x,y stay 0, realign() degenerates to a rotation
        and AsyncShield's whole contribution is silently lost."""
        pb = PoseBuffer()
        for i in range(4):
            pb.integrate(float(i), _d(dx=0.25))
        self.assertGreater(abs(pb.x) + abs(pb.y), 0.5)

    def test_delta_is_rotated_into_the_odom_frame(self):
        """Drive forward, turn 90 deg left, drive forward. The second leg must
        go +y, not +x. Adding deltas without rotating corrupts every turn."""
        pb = PoseBuffer()
        pb.integrate(0.0, _d(dx=1.0))
        pb.integrate(1.0, _d(dyaw=math.radians(90.0)))
        pb.integrate(2.0, _d(dx=1.0))
        self.assertAlmostEqual(pb.x, 1.0, places=6)
        self.assertAlmostEqual(pb.y, 1.0, places=6)

    def test_zero_confidence_records_no_movement_not_a_hole(self):
        pb = PoseBuffer()
        pb.integrate(0.0, _d(dx=0.3))
        before = (pb.x, pb.y)
        pb.integrate(1.0, _d(dx=0.3, conf=0.0))
        self.assertEqual((pb.x, pb.y), before)
        self.assertEqual(len(pb.buf), 2)     # timestamp still recorded

    def test_realign_shifts_waypoints_by_travel(self):
        pb = PoseBuffer()
        pb.integrate(0.0, _d())
        pb.integrate(1.0, _d(dx=0.5))
        wp = np.array([[2.0, 0.0], [3.0, 0.0]])
        out = pb.realign(wp, 0.0, 1.0)
        # Having driven 0.5 m forward, a point 2 m ahead is now 1.5 m ahead.
        self.assertLess(out[0, 0], wp[0, 0])
        self.assertAlmostEqual(out[0, 0], 1.5, delta=0.05)

    def test_realign_is_identity_without_motion(self):
        pb = PoseBuffer()
        pb.integrate(0.0, _d())
        pb.integrate(1.0, _d())
        wp = np.array([[2.0, 0.3], [3.0, 0.4]])
        np.testing.assert_allclose(pb.realign(wp, 0.0, 1.0), wp, atol=1e-9)


class TestCommitWindow(unittest.TestCase):
    def _path(self, x0=0.5, n=6):
        s = np.linspace(x0, x0 + 2.0, n)
        return np.stack([s, np.zeros_like(s)], axis=1)

    def test_first_offer_is_adopted(self):
        cw = CommitWindow()
        p = self._path()
        np.testing.assert_allclose(cw.offer(p), p)
        self.assertEqual(cw.last_reason, "adopted")

    def test_held_plan_resists_a_single_bad_tick(self):
        """A one-frame perception glitch must not swing the command."""
        cw = CommitWindow(blend=0.35)
        good = self._path()
        cw.offer(good)
        cw.advance(_d(dx=0.05), dt=0.2)
        wild = good.copy()
        wild[:, 1] += 2.0
        out = cw.offer(wild)
        self.assertLess(abs(out[0, 1]), 1.0)     # nowhere near the 2 m swing
        self.assertEqual(cw.last_reason, "held")

    def test_persistent_change_is_eventually_adopted(self):
        """Resisting a glitch must not mean ignoring the world."""
        cw = CommitWindow(blend=0.35)
        cw.offer(self._path())
        moved = self._path()
        moved[:, 1] += 1.0
        for _ in range(12):
            cw.advance(_d(dx=0.02), dt=0.05)
            out = cw.offer(moved)
        self.assertGreater(abs(out[0, 1]), 0.7)

    def test_expires_on_age(self):
        cw = CommitWindow(max_age_s=0.5)
        cw.offer(self._path())
        cw.advance(_d(dx=0.01), dt=0.6)
        self.assertTrue(cw.expired())

    def test_expires_on_distance(self):
        cw = CommitWindow(max_age_s=99.0, max_distance_m=0.3)
        cw.offer(self._path())
        for _ in range(5):
            cw.advance(_d(dx=0.1), dt=0.01)
        self.assertTrue(cw.expired())

    def test_advance_moves_path_toward_the_rover(self):
        cw = CommitWindow()
        cw.offer(self._path(x0=1.0))
        before = cw.path[0, 0]
        cw.advance(_d(dx=0.3), dt=0.2)
        self.assertLess(cw.path[0, 0], before)

    def test_advance_applies_rotation(self):
        cw = CommitWindow()
        cw.offer(self._path(x0=1.0))
        cw.advance(_d(dyaw=math.radians(30.0)), dt=0.2)
        self.assertGreater(abs(cw.path[0, 1]), 0.1)

    def test_no_odometry_expires_rather_than_guessing(self):
        """Without a motion estimate the held path is in an unknown frame.
        Following it would steer using a stale geometry."""
        cw = CommitWindow()
        cw.offer(self._path())
        cw.advance(_d(conf=0.0), dt=0.2)
        self.assertTrue(cw.expired())
        self.assertEqual(cw.last_reason, "no_odom")

    def test_passed_waypoints_are_dropped(self):
        cw = CommitWindow()
        cw.offer(self._path(x0=0.1, n=6))
        n0 = len(cw.path)
        cw.advance(_d(dx=0.5), dt=0.2)
        self.assertLess(len(cw.path), n0)


class TestStalenessDecay(unittest.TestCase):
    def test_tau_matches_the_papers_deployed_value(self):
        """Slow-Brain deploys tau = 5 s; our default was 4.0."""
        self.assertAlmostEqual(VLMSelectorFusion().tau, 5.0)

    def test_alpha_is_bounded_by_one(self):
        """Probability Fusion's stability claim: alpha = (lam/(lam+1))*w <= 1,
        which is why the paper prefers it over unbounded Score Fusion."""
        for lam in (0.5, 1.0, 4.0, 100.0):
            f = VLMSelectorFusion(lam=lam)
            for dt in (0.0, 1.0, 5.0, 30.0):
                w = math.exp(-dt / f.tau)
                self.assertLessEqual((f.lam / (f.lam + 1.0)) * w, 1.0)

    def test_decay_halves_influence_over_time(self):
        f = VLMSelectorFusion()
        self.assertAlmostEqual(math.exp(-0.0 / f.tau), 1.0)
        self.assertLess(math.exp(-5.0 / f.tau), 0.4)
        self.assertLess(math.exp(-15.0 / f.tau), 0.06)


if __name__ == "__main__":
    unittest.main()
