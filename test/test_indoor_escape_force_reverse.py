"""Hung-up escape must honor force_reverse even when free space looks clear."""
from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock

import numpy as np

from erc_indoor.navigator import ESCAPE_SETTLE_S, Navigator, Obs


def _fake_route():
    route = MagicMock()
    route.turns = []
    route.__len__ = MagicMock(return_value=10)
    route.node_at_p = MagicMock(return_value=3)
    return route


def _fake_freespace(ahead_m: float = 5.0):
    bins = np.linspace(-40.0, 40.0, 17)
    clear = np.full_like(bins, ahead_m, dtype=float)
    fs = types.SimpleNamespace(
        bins_deg=bins,
        clear_m=clear,
        ahead_m=ahead_m,
        pts=None,
    )
    fs.clear_at = lambda deg: ahead_m
    free = MagicMock()
    free.estimate = MagicMock(return_value=fs)
    # Without force_reverse, clear ahead + a steer target would arc, not reverse.
    free.steer = MagicMock(return_value=20.0)
    return free


def _escape_after_settle(heading_err: float, force_reverse: bool):
    """Begin an escape with clear floor ahead, settle, and return the first escape decision."""
    nav = Navigator(
        _fake_route(),
        MagicMock(),
        MagicMock(),
        freespace=_fake_freespace(ahead_m=5.0),
        verbose=False,
    )
    # Avoid real bearing / SIFT: want = -heading_err.
    nav._heading = MagicMock(return_value=heading_err)

    t0 = 100.0
    obs = Obs(
        now=t0,
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        s_total=1.0,
        yaw_total=0.0,
        v=0.0,
    )
    nav._begin_escape(obs, "hung", force_reverse=force_reverse)
    began_forced = nav._force_reverse

    obs2 = Obs(
        now=t0 + ESCAPE_SETTLE_S + 0.05,
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        s_total=1.0,
        yaw_total=0.0,
        v=0.0,
    )
    est = types.SimpleNamespace(p=0.0, s=0.0)
    return nav, began_forced, nav._escape(obs2, est)


class TestIndoorEscapeForceReverse(unittest.TestCase):
    def _assert_reversed(self, nav, cmd):
        self.assertTrue(
            cmd.lin < 0 and nav._escape_phase == "reverse",
            f"expected reverse after hung-up escape; got lin={cmd.lin} "
            f"phase={nav._escape_phase!r} mode={nav.mode!r} note={cmd.note!r}",
        )

    def test_begin_escape_force_reverse_triggers_reverse(self):
        nav, began_forced, cmd = _escape_after_settle(25.0, force_reverse=True)
        self.assertTrue(
            began_forced,
            "_begin_escape(force_reverse=True) must set self._force_reverse",
        )
        self._assert_reversed(nav, cmd)

    def test_force_reverse_wins_over_clear_aligned_exit(self):
        # Hung up while already facing the route with clear floor: the
        # "escape done, clear" shortcut must not skip the forced reverse.
        nav, _, cmd = _escape_after_settle(0.0, force_reverse=True)
        self._assert_reversed(nav, cmd)
        self.assertFalse(nav._force_reverse, "force_reverse is one-shot")

    def test_unforced_clear_aligned_still_returns_to_drive(self):
        nav, _, cmd = _escape_after_settle(0.0, force_reverse=False)
        self.assertEqual(nav.mode, "DRIVE")
        self.assertEqual(cmd.lin, 0.0)


if __name__ == "__main__":
    unittest.main()
