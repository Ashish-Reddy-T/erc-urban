"""Tests for action-chunk latency index selection (NS / RTC style)."""
from __future__ import annotations
import unittest

import numpy as np

from erc_urban_test.modules.action_chunk import (
    ActionChunkExecutor,
    chunk_index,
    select_action,
)


class TestChunkIndex(unittest.TestCase):
    def test_elapsed_zero_is_first_action(self):
        self.assertEqual(chunk_index(0.0), 0)

    def test_quarter_second_at_ten_hz(self):
        self.assertEqual(chunk_index(0.25, dt_s=0.1), 2)

    def test_past_horizon_clamps(self):
        self.assertEqual(chunk_index(5.0, dt_s=0.1, horizon=10), 9)

    def test_negative_elapsed_clamps_to_zero(self):
        self.assertEqual(chunk_index(-0.5), 0)

    def test_non_finite_elapsed_clamps_to_zero(self):
        self.assertEqual(chunk_index(float("nan")), 0)
        self.assertEqual(chunk_index(float("inf")), 0)

    def test_exact_step_boundaries_select_that_step(self):
        # 0.3 / 0.1 == 2.9999999999999996 in binary floating point.
        got = {e: chunk_index(e, dt_s=0.1) for e in (0.3, 0.6, 0.7, 0.9)}
        self.assertEqual(got, {0.3: 3, 0.6: 6, 0.7: 7, 0.9: 9})

    def test_every_step_boundary_selects_that_step(self):
        horizon = 10
        for dt in (0.1, 0.05, 0.2, 1.0 / 15.0):
            for k in range(horizon):
                # Rounded so elapsed is the decimal boundary (0.3), not 3 * 0.1.
                elapsed = round(k * dt, 12)
                with self.subTest(dt=dt, k=k, elapsed=elapsed):
                    self.assertEqual(
                        chunk_index(elapsed, dt_s=dt, horizon=horizon), k
                    )

    def test_just_below_boundary_stays_on_previous_step(self):
        self.assertEqual(chunk_index(0.2999, dt_s=0.1), 2)

    def test_ratio_overflow_clamps_to_last_step(self):
        self.assertEqual(chunk_index(1e308, dt_s=1e-10, horizon=10), 9)


class TestSelectAction(unittest.TestCase):
    def test_selects_row_from_matrix_chunk(self):
        chunk = np.arange(30, dtype=float).reshape(10, 3)
        np.testing.assert_array_equal(select_action(chunk, 0.25, dt_s=0.1), chunk[2])

    def test_selects_scalar_from_vector_chunk(self):
        chunk = np.linspace(0.0, 0.9, 10)
        self.assertAlmostEqual(float(select_action(chunk, 0.0)), 0.0)

    def test_mutating_returned_row_leaves_chunk_unchanged(self):
        chunk = np.arange(20, dtype=float).reshape(10, 2)
        row = select_action(chunk, 0.0)
        row *= 2
        np.testing.assert_array_equal(chunk[0], np.array([0.0, 1.0]))

    def test_empty_chunk_raises_value_error(self):
        with self.assertRaises(ValueError):
            select_action(np.zeros((0, 2)), 0.0)


class TestActionChunkExecutor(unittest.TestCase):
    def test_step_before_push_returns_none(self):
        ex = ActionChunkExecutor()
        self.assertIsNone(ex.step(now=100.0))

    def test_push_then_step_uses_elapsed_since_t0(self):
        chunk = np.stack([np.array([1.0, 0.0]), np.array([2.0, 0.0])])
        ex = ActionChunkExecutor()
        ex.push(chunk, t0=10.0)
        out = ex.step(now=10.25)
        np.testing.assert_allclose(out, np.array([2.0, 0.0]))

    def test_mutating_step_output_leaves_stored_chunk_and_caller_unchanged(self):
        chunk = np.array([[0.5, 0.2], [0.6, 0.3]])
        ex = ActionChunkExecutor()
        ex.push(chunk, t0=0.0)
        out = ex.step(now=0.0)
        out *= 2
        np.testing.assert_allclose(ex.step(now=0.0), np.array([0.5, 0.2]))
        np.testing.assert_allclose(chunk[0], np.array([0.5, 0.2]))

    def test_caller_mutating_pushed_array_leaves_stored_chunk_unchanged(self):
        chunk = np.array([[0.5, 0.2], [0.6, 0.3]])
        ex = ActionChunkExecutor()
        ex.push(chunk, t0=0.0)
        chunk[0] = [9.0, 9.0]
        np.testing.assert_allclose(ex.step(now=0.0), np.array([0.5, 0.2]))

    def test_push_rejects_empty_scalar_or_none_chunk(self):
        bad = {
            "empty_list": [],
            "zeros_0x2": np.zeros((0, 2)),
            "zero_d": np.array(1.0),
            "none": None,
        }
        for name, chunk in bad.items():
            with self.subTest(name=name):
                ex = ActionChunkExecutor()
                with self.assertRaises(ValueError):
                    ex.push(chunk, t0=0.0)


if __name__ == "__main__":
    unittest.main()
