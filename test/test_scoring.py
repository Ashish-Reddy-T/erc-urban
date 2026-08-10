"""
Tests for candidate scoring.

This module exists because the previous ranking measured WORSE THAN RANDOM
against human-driven rides (1.029 m median ADE vs 0.757 m). These tests pin the
properties that failure came from, so it cannot come back silently.
"""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban.interfaces import Candidate
from erc_urban.modules.scoring import (
    CandidateScorer, ScoreWeights, path_traversable_fractions,
)


def _arc(lateral: float, n: int = 10, length: float = 2.0) -> Candidate:
    s = np.linspace(0.0, length, n)
    return Candidate(
        waypoints=np.stack([s, lateral * (s / max(length, 1e-9)) ** 2], axis=1),
        score=0.0, stamp=0.0,
    )


class TestGoalTerm(unittest.TestCase):
    def test_path_pointing_at_the_goal_scores_highest(self):
        sc = CandidateScorer()
        cands = [_arc(-1.0), _arc(0.0), _arc(1.0)]
        s = sc.score(cands, goal_bearing=0.0)
        self.assertEqual(int(np.argmax(s)), 1)

    def test_goal_term_uses_the_whole_path_not_just_the_endpoint(self):
        """
        Two paths reaching the same endpoint by different routes must not tie.
        That equivalence let a kerb-crossing candidate rank level with a clean
        one, and was part of why the ranking was sub-random.
        """
        s = np.linspace(0.0, 2.0, 11)
        straight = Candidate(np.stack([s, np.zeros_like(s)], axis=1), 0.0, 0.0)
        detour = np.stack([s, 0.9 * np.sin(np.pi * s / 2.0)], axis=1)
        detour[-1, 1] = 0.0                     # same endpoint
        sc = CandidateScorer()
        vals = sc.score([straight, Candidate(detour, 0.0, 0.0)], goal_bearing=0.0)
        self.assertGreater(vals[0], vals[1])


class TestSmoothTerm(unittest.TestCase):
    def test_constant_curvature_arcs_are_equally_smooth(self):
        """
        The regression. The old term penalised curvature itself, making it a
        second vote for 'go straight' -- Spearman -0.32, helping on 0% of
        ticks. Anchor arcs all have constant curvature and must tie.
        """
        sc = CandidateScorer(weights=ScoreWeights(goal=0.0, prior=0.0,
                                                  traversable=0.0, clearance=0.0,
                                                  smooth=1.0))
        vals = sc.score([_arc(0.0), _arc(1.0), _arc(-1.5)], goal_bearing=0.0)
        self.assertLess(float(np.ptp(vals)), 0.05)

    def test_a_path_that_changes_its_mind_is_penalised(self):
        sc = CandidateScorer(weights=ScoreWeights(goal=0.0, prior=0.0,
                                                  traversable=0.0, clearance=0.0,
                                                  smooth=1.0))
        s = np.linspace(0.0, 2.0, 12)
        smooth = Candidate(np.stack([s, 0.3 * s], axis=1), 0.0, 0.0)
        jerky = Candidate(
            np.stack([s, 0.3 * s + 0.15 * np.sin(6.0 * s)], axis=1), 0.0, 0.0)
        vals = sc.score([smooth, jerky], goal_bearing=0.0)
        self.assertGreater(vals[0], vals[1])


class TestPerceptionTerms(unittest.TestCase):
    def test_traversable_path_outranks_untraversable_one(self):
        """The only term measured to genuinely help (rho +0.23, 68% of ticks)."""
        sc = CandidateScorer()
        cands = [_arc(0.0), _arc(0.0)]
        vals = sc.score(cands, goal_bearing=0.0, trav_fracs=np.array([0.2, 1.0]))
        self.assertGreater(vals[1], vals[0])

    def test_more_clearance_outranks_less(self):
        sc = CandidateScorer(clearance_ref_m=2.0)
        cands = [_arc(0.0), _arc(0.0)]
        vals = sc.score(cands, goal_bearing=0.0, clearances=np.array([0.2, 1.9]))
        self.assertGreater(vals[1], vals[0])

    def test_clearance_saturates(self):
        """Beyond the reference distance, more room is not more better."""
        sc = CandidateScorer(clearance_ref_m=2.0)
        vals = sc.score([_arc(0.0), _arc(0.0)], goal_bearing=0.0,
                        clearances=np.array([5.0, 50.0]))
        self.assertAlmostEqual(vals[0], vals[1], places=9)

    def test_traversability_outweighs_goal_alignment(self):
        """Driving on the wrong surface is the dominant sidewalk-violation
        failure mode, so it must beat pointing the right way."""
        sc = CandidateScorer()
        on_goal_off_path = _arc(0.0)
        off_goal_on_path = _arc(1.0)
        vals = sc.score([on_goal_off_path, off_goal_on_path], goal_bearing=0.0,
                        trav_fracs=np.array([0.0, 1.0]))
        self.assertGreater(vals[1], vals[0])


class TestDegradation(unittest.TestCase):
    def test_missing_perception_terms_contribute_nothing(self):
        """Must degrade to available information, not invent values."""
        sc = CandidateScorer()
        cands = [_arc(0.0), _arc(1.0)]
        sc.score(cands, goal_bearing=0.0)
        for t in sc.last:
            self.assertEqual(t.traversable, 0.0)
            self.assertEqual(t.clearance, 0.0)

    def test_terms_are_recorded_for_debugging(self):
        sc = CandidateScorer()
        sc.score([_arc(0.0)], goal_bearing=0.0, trav_fracs=np.array([0.5]))
        self.assertEqual(len(sc.last), 1)
        self.assertIn("trav=", sc.last[0].describe())

    def test_empty_set_is_safe(self):
        self.assertEqual(len(CandidateScorer().score([], 0.0)), 0)


class TestTraversableFractions(unittest.TestCase):
    def test_all_drivable_gives_one(self):
        from erc_urban.calib import CameraCalibration

        cal = CameraCalibration(width=256, height=144, fx=85.0, fy=85.0,
                                cx=128.0, cy=72.0, height_m=0.148,
                                pitch_rad=math.radians(3.0), source="test")
        mask = np.ones((144, 256), bool)
        f = path_traversable_fractions([_arc(0.0)], mask, cal)
        self.assertAlmostEqual(float(f[0]), 1.0, places=6)

    def test_none_drivable_gives_zero(self):
        from erc_urban.calib import CameraCalibration

        cal = CameraCalibration(width=256, height=144, fx=85.0, fy=85.0,
                                cx=128.0, cy=72.0, height_m=0.148,
                                pitch_rad=math.radians(3.0), source="test")
        f = path_traversable_fractions([_arc(0.0)], np.zeros((144, 256), bool), cal)
        self.assertAlmostEqual(float(f[0]), 0.0, places=6)

    def test_is_continuous_not_binary(self):
        """The veto needs a threshold; the ranking needs the gradient, so it can
        prefer the middle of a path over its edge."""
        from erc_urban.calib import CameraCalibration

        cal = CameraCalibration(width=256, height=144, fx=85.0, fy=85.0,
                                cx=128.0, cy=72.0, height_m=0.148,
                                pitch_rad=math.radians(3.0), source="test")
        mask = np.zeros((144, 256), bool)
        mask[100:, :] = True
        f = path_traversable_fractions([_arc(0.0)], mask, cal)
        self.assertGreater(float(f[0]), 0.0)
        self.assertLess(float(f[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
