"""
Tests for M6, the semantic traversability veto.

Uses synthetic masks with known geometry so the veto logic is testable without
the SegFormer weights, plus one real-fixture test that runs when they are
present. The point of most of these is the FAIL-OPEN vs FAIL-CLOSED boundary:
a veto that fails closed strands the rover, and a veto that fails open is the
no-op stub this module replaces.
"""
from __future__ import annotations
import unittest

import numpy as np

from erc_urban.calib import CameraCalibration
from erc_urban.interfaces import Candidate, Observation
from erc_urban.modules.traversability import TraversabilityVeto


MASK_HW = (120, 160)


def _obs(w: int = 320, h: int = 240) -> Observation:
    return Observation(
        front=np.zeros((h, w, 3), dtype=np.uint8), rear=None,
        lat=37.0, lon=-122.0, yaw=0.0, stamp=1.0,
        goal_bearing=0.0, goal_range_m=30.0,
    )


def _straight(lateral: float, n: int = 8, length: float = 3.0) -> Candidate:
    s = np.linspace(0.2, length, n)
    return Candidate(
        waypoints=np.stack([s, np.full_like(s, lateral)], axis=1),
        score=-abs(lateral), stamp=1.0,
    )


class FakeSource:
    """Traversability mask driven by a predicate on (u, v) in mask pixels."""

    ready = True

    def __init__(self, fn=None):
        self.fn = fn
        self.calls = 0

    def traversable(self, rgb, out_hw):
        self.calls += 1
        if self.fn is None:
            return None
        h, w = out_hw
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        return self.fn(u, v)


def _all_good(u, v):
    return np.ones_like(u, dtype=bool)


def _all_bad(u, v):
    return np.zeros_like(u, dtype=bool)


def _robot_left_untraversable(u, v):
    """
    Returns the TRAVERSABLE mask (True = drivable), untraversable on image-left.

    Image-left (small u) is robot-LEFT (+y) — see
    test_calib.test_left_in_robot_frame_is_left_in_image. So a candidate with
    positive lateral offset crosses the bad surface and must be vetoed.
    """
    return u >= 80


class TestVetoLogic(unittest.TestCase):
    def test_all_traversable_keeps_everything(self):
        veto = TraversabilityVeto(source=FakeSource(_all_good), mask_hw=MASK_HW)
        cands = [_straight(x) for x in (-1.0, 0.0, 1.0)]
        self.assertTrue(veto.mask(_obs(), cands).all())

    def test_all_untraversable_vetoes_everything(self):
        veto = TraversabilityVeto(
            source=FakeSource(_all_bad), mask_hw=MASK_HW,
            min_traversable_frac=0.0,     # disable the OOD guard for this test
        )
        cands = [_straight(x) for x in (-1.0, 0.0, 1.0)]
        self.assertFalse(veto.mask(_obs(), cands).any())

    def test_undiscriminating_source_fails_open(self):
        """Measured need: SegFormer calls `synth_dirt_path` 100% sky and
        `mauritius_rover_block` 84% wall. An out-of-distribution frame yields a
        confident nonsense mask, not a cautious one. Acting on it would veto
        every candidate and hand run.py a spurious 'no way forward'."""
        veto = TraversabilityVeto(
            source=FakeSource(_all_bad), mask_hw=MASK_HW,
            min_traversable_frac=0.02,
        )
        cands = [_straight(x) for x in (-1.0, 0.0, 1.0)]
        self.assertTrue(veto.mask(_obs(), cands).all())
        self.assertIn("undiscriminating", veto.last_reason)

    def test_guard_does_not_suppress_a_normal_scene(self):
        """A scene with real drivable surface must still be vetoed normally."""
        veto = TraversabilityVeto(
            source=FakeSource(_robot_left_untraversable), mask_hw=MASK_HW,
            min_traversable_frac=0.02,
        )
        keep = veto.mask(_obs(), [_straight(+1.5), _straight(-1.5)])
        self.assertFalse(bool(keep[0]))
        self.assertTrue(bool(keep[1]))

    def test_vetoes_the_side_that_is_untraversable(self):
        """The whole point: grass on one side must kill only the paths that
        cross it, and leave the others alone."""
        veto = TraversabilityVeto(
            source=FakeSource(_robot_left_untraversable), mask_hw=MASK_HW
        )
        cands = [_straight(+1.5), _straight(-1.5)]
        keep = veto.mask(_obs(), cands)
        self.assertFalse(bool(keep[0]), "robot-left path crosses bad surface")
        self.assertTrue(bool(keep[1]), "robot-right path should survive")

    def test_source_unavailable_fails_open(self):
        """No evidence is not a veto — the geometric costmap still applies.
        Failing closed here would strand the rover whenever the segmenter
        hiccups."""
        veto = TraversabilityVeto(source=FakeSource(None), mask_hw=MASK_HW)
        cands = [_straight(0.0)]
        self.assertTrue(veto.mask(_obs(), cands).all())
        self.assertEqual(veto.last_reason, "source_unavailable")

    def test_a_few_stray_bad_pixels_do_not_veto(self):
        """Segmentation is noisy at boundaries; zero-tolerance fails closed."""
        def speckle(u, v):
            m = np.ones_like(u, dtype=bool)
            m[::40, ::40] = False
            return m

        veto = TraversabilityVeto(source=FakeSource(speckle), mask_hw=MASK_HW)
        self.assertTrue(veto.mask(_obs(), [_straight(0.0)]).all())

    def test_empty_candidates_is_safe(self):
        veto = TraversabilityVeto(source=FakeSource(_all_good), mask_hw=MASK_HW)
        self.assertEqual(len(veto.mask(_obs(), [])), 0)

    def test_paths_outside_the_frame_are_not_vetoed(self):
        """Below min_samples there is no evidence either way."""
        veto = TraversabilityVeto(source=FakeSource(_all_bad), mask_hw=MASK_HW)
        behind = Candidate(
            waypoints=np.stack(
                [np.linspace(-3.0, -0.5, 8), np.zeros(8)], axis=1
            ), score=0.0, stamp=1.0,
        )
        self.assertTrue(bool(veto.mask(_obs(), [behind])[0]))

    def test_densify_catches_a_curb_between_waypoints(self):
        """Waypoints are ~0.4 m apart; a thin bad stripe between two of them
        must still be seen, or the rover drives over the curb it stepped over
        in the sampling."""
        def stripe(u, v):
            m = np.ones_like(u, dtype=bool)
            m[:, 78:82] = False        # narrow vertical band at image centre
            return m

        sparse = TraversabilityVeto(
            source=FakeSource(stripe), mask_hw=MASK_HW,
            samples_per_segment=1, max_bad_frac=0.05,
        )
        dense = TraversabilityVeto(
            source=FakeSource(stripe), mask_hw=MASK_HW,
            samples_per_segment=16, max_bad_frac=0.05,
        )
        cand = [_straight(0.0)]
        self.assertGreaterEqual(
            int((~dense.mask(_obs(), cand)).sum()),
            int((~sparse.mask(_obs(), cand)).sum()),
        )


class TestApplyAfterSelection(unittest.TestCase):
    """plan.md's contract: veto AFTER selection, fall back to next-best."""

    def test_passes_through_when_choice_is_safe(self):
        veto = TraversabilityVeto(source=FakeSource(_all_good), mask_hw=MASK_HW)
        cands = [_straight(0.0), _straight(1.0)]
        idx, note = veto.apply(_obs(), cands, 0)
        self.assertEqual(idx, 0)
        self.assertEqual(note, "ok")

    def test_falls_back_to_next_best_traversable(self):
        veto = TraversabilityVeto(
            source=FakeSource(_robot_left_untraversable), mask_hw=MASK_HW
        )
        # index 0 goes robot-left across the bad surface; index 1 does not.
        cands = [_straight(+1.5), _straight(-1.5)]
        idx, note = veto.apply(_obs(), cands, 0)
        self.assertEqual(idx, 1)
        self.assertIn("vetoed", note)

    def test_reports_when_nothing_is_traversable(self):
        """Must not silently hand back an unsafe path as if it were fine —
        run.py's reverse/scan recovery needs to know."""
        veto = TraversabilityVeto(
            source=FakeSource(_all_bad), mask_hw=MASK_HW,
            min_traversable_frac=0.0,     # a genuinely blocked scene, not OOD
        )
        cands = [_straight(0.0), _straight(1.0)]
        idx, note = veto.apply(_obs(), cands, 0)
        self.assertEqual(idx, 0)
        self.assertIn("all_untraversable", note)

    def test_fallback_respects_score_order(self):
        """Fall back to the best-scoring survivor, not merely the first."""
        veto = TraversabilityVeto(
            source=FakeSource(_robot_left_untraversable), mask_hw=MASK_HW
        )
        bad = _straight(+1.5)                  # robot-left, untraversable
        good_low = _straight(-2.0)             # score -2.0
        good_high = _straight(-0.5)            # score -0.5, the better one
        cands = [bad, good_low, good_high]
        idx, _ = veto.apply(_obs(), cands, 0)
        self.assertEqual(idx, 2)


class TestSegFormerSource(unittest.TestCase):
    def test_unavailable_source_reports_not_ready_rather_than_vetoing(self):
        """The bug this module replaces: upgrades.TraversabilityGate returned
        all-True forever because its predictor was always None."""
        from erc_urban.modules.traversability import SAMTPSource

        s = SAMTPSource(predictor=None)
        self.assertFalse(s.ready)
        self.assertIsNone(s.traversable(np.zeros((8, 8, 3), np.uint8), MASK_HW))


if __name__ == "__main__":
    unittest.main()
