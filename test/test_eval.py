"""
Tests for M9, the evaluation harness.

The metric itself needs testing as much as the code it measures. The first run
of the oracle gap reported a median top-1 ADE of 1.28 m, which was not a finding
about the planner at all -- it was index-wise pairing of a ~2.5 m candidate
against a 0.30 m reference, i.e. the metric measuring its own length mismatch.
These pin the corrected behaviour.
"""
from __future__ import annotations
import math
import unittest

import numpy as np

from erc_urban.eval.oracle_gap import ade, future_path


def _line(n=10, length=2.0, lateral=0.0):
    s = np.linspace(0.0, length, n)
    return np.stack([s, lateral * s / max(length, 1e-9)], axis=1)


def _tick(dx=0.05, dy=0.0, dyaw=0.0, conf=1.0):
    return {"dx": dx, "dy": dy, "dyaw": dyaw, "conf": conf}


class TestArcLengthADE(unittest.TestCase):
    def test_identical_paths_score_zero(self):
        p = _line()
        self.assertAlmostEqual(ade(p, p), 0.0, places=9)

    def test_length_mismatch_is_not_penalised(self):
        """The regression. A candidate that follows the reference exactly but
        extends further must score ~0, not be punished for being longer."""
        short = _line(n=6, length=0.3)
        long = _line(n=20, length=2.5)
        self.assertLess(ade(long, short), 0.02)

    def test_shape_difference_is_penalised(self):
        ref = _line(length=1.0, lateral=0.0)
        straight = _line(length=2.5, lateral=0.0)
        veering = _line(length=2.5, lateral=2.0)
        self.assertLess(ade(straight, ref), ade(veering, ref))

    def test_ordering_is_by_shape_not_by_length(self):
        ref = _line(length=1.0, lateral=0.5)
        near = _line(length=3.0, lateral=1.5)      # same curvature, longer
        far = _line(length=1.0, lateral=-0.5)      # same length, wrong side
        self.assertLess(ade(near, ref), ade(far, ref))

    def test_degenerate_inputs_are_nan_not_zero(self):
        """A silent 0.0 would look like a perfect match and win the oracle."""
        self.assertTrue(math.isnan(ade(_line(), np.zeros((1, 2)))))
        self.assertTrue(math.isnan(ade(_line(), _line(n=4, length=0.001))))


class TestFuturePath(unittest.TestCase):
    def test_straight_chain_accumulates_forward(self):
        ticks = [_tick(dx=0.1) for _ in range(6)]
        p = future_path(ticks, 0, 6)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(float(p[-1][0]), 0.6, places=6)
        self.assertAlmostEqual(float(p[-1][1]), 0.0, places=6)

    def test_rotation_is_applied_along_the_chain(self):
        """Deltas are in the frame at the start of each step, so a turn must
        rotate everything after it. Summing them raw corrupts every arc."""
        ticks = [_tick(dx=1.0),
                 _tick(dx=0.0, dyaw=math.radians(90.0)),   # pure rotation
                 _tick(dx=1.0)]
        p = future_path(ticks, 0, 3)
        self.assertAlmostEqual(float(p[-1][0]), 1.0, places=6)
        self.assertAlmostEqual(float(p[-1][1]), 1.0, places=6)

    def test_low_confidence_breaks_the_chain(self):
        """An unmeasured reference silently rewards candidates that match
        nothing, so it must be refused rather than guessed."""
        ticks = [_tick(), _tick(conf=0.0), _tick()]
        self.assertIsNone(future_path(ticks, 0, 3))

    def test_too_short_returns_none(self):
        self.assertIsNone(future_path([_tick(), _tick()], 0, 2))


if __name__ == "__main__":
    unittest.main()
