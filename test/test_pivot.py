"""Tests for PulsePivot (erc_urban.modules.pivot): the open-loop pivot
that issues in-place turn commands. Covers the safety-relevant contracts the
caller relies on: sign convention (left-positive), the commanded magnitude
never exceeding the configured max, burst duration never exceeding the
configured max_burst or falling under min_burst, and the pivot going silent
(zero command) once aligned instead of hunting forever.
"""
from __future__ import annotations

import unittest

from erc_urban.modules.pivot import PulsePivot, wrap180


class _Tel:
    def __init__(self, yaw: float = 0.0, hw_ts: float = 0.0):
        self.yaw = yaw
        self.hw_ts = hw_ts


class TestWrap180(unittest.TestCase):
    def test_wraps_into_minus_180_to_180(self):
        self.assertAlmostEqual(wrap180(370.0), 10.0)
        self.assertAlmostEqual(wrap180(-370.0), -10.0)
        self.assertAlmostEqual(wrap180(0.0), 0.0)


class TestPulsePivotDirection(unittest.TestCase):
    def test_positive_error_turns_left_positive_angular(self):
        pivot = PulsePivot()
        _, ang, _ = pivot.command(err_deg=60.0, tel=_Tel(), now=0.0)
        self.assertGreater(ang, 0.0)

    def test_negative_error_turns_right_negative_angular(self):
        pivot = PulsePivot()
        _, ang, _ = pivot.command(err_deg=-60.0, tel=_Tel(), now=0.0)
        self.assertLess(ang, 0.0)

    def test_linear_component_is_always_zero(self):
        """PulsePivot only pivots in place; it must never command forward motion."""
        pivot = PulsePivot()
        lin, _, _ = pivot.command(err_deg=60.0, tel=_Tel(), now=0.0)
        self.assertEqual(lin, 0.0)


class TestPulsePivotMagnitudeCap(unittest.TestCase):
    def test_commanded_angular_never_exceeds_configured_max(self):
        pivot = PulsePivot(ang=0.45)
        for err in (10.0, 45.0, 90.0, 170.0):
            p = PulsePivot(ang=0.45)
            _, ang, _ = p.command(err_deg=err, tel=_Tel(), now=0.0)
            self.assertLessEqual(abs(ang), pivot.ang + 1e-9)

    def test_large_error_commands_full_configured_magnitude(self):
        pivot = PulsePivot(ang=0.45)
        _, ang, _ = pivot.command(err_deg=170.0, tel=_Tel(), now=0.0)
        self.assertAlmostEqual(abs(ang), 0.45, places=6)


class TestPulsePivotBurstDuration(unittest.TestCase):
    def test_burst_never_runs_longer_than_max_burst(self):
        pivot = PulsePivot(max_burst=1.0, ang=0.45, rate_prior=5.0)
        pivot.command(err_deg=170.0, tel=_Tel(), now=0.0)
        self.assertTrue(pivot.busy())

        # Just before the cap: still mid-burst.
        _, ang, note = pivot.command(err_deg=170.0, tel=_Tel(), now=0.99)
        self.assertNotEqual(note, "burst done")

        # At/after the cap: the burst must have ended.
        _, ang, note = pivot.command(err_deg=170.0, tel=_Tel(), now=1.01)
        self.assertEqual(note, "burst done")
        self.assertEqual(ang, 0.0)

    def test_burst_never_shorter_than_min_burst(self):
        # A small error, weak enough that the target turn is under one
        # full-command min_burst sweep, still gets the full min_burst (with
        # a weakened magnitude) rather than a shorter burst.
        pivot = PulsePivot(
            tol_deg=5.0, min_burst=0.5, ang=0.45, ang_floor=0.05, rate_prior=30.0
        )
        pivot.command(err_deg=10.0, tel=_Tel(), now=0.0)
        self.assertTrue(pivot.busy())

        _, _, note = pivot.command(err_deg=10.0, tel=_Tel(), now=0.49)
        self.assertNotEqual(note, "burst done")


class TestPulsePivotAligned(unittest.TestCase):
    def test_within_tolerance_issues_zero_command_and_stays_idle(self):
        pivot = PulsePivot(tol_deg=10.0)
        lin, ang, note = pivot.command(err_deg=2.0, tel=_Tel(), now=0.0)
        self.assertEqual((lin, ang), (0.0, 0.0))
        self.assertEqual(note, "aligned")
        self.assertFalse(pivot.busy())

    def test_repeated_calls_while_aligned_stay_silent(self):
        """Once aligned, the pivot must not resume spinning on later ticks."""
        pivot = PulsePivot(tol_deg=10.0)
        for now in (0.0, 1.0, 2.0):
            lin, ang, note = pivot.command(err_deg=2.0, tel=_Tel(), now=now)
            self.assertEqual((lin, ang, note), (0.0, 0.0, "aligned"))


if __name__ == "__main__":
    unittest.main()
