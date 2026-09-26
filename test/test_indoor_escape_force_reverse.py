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


class TestIndoorEscapeForceReverse(unittest.TestCase):
    def test_begin_escape_force_reverse_triggers_reverse(self):
        nav = Navigator(
            _fake_route(),
            MagicMock(),
            MagicMock(),
            freespace=_fake_freespace(ahead_m=5.0),
            verbose=False,
        )
        # Avoid real bearing / SIFT; keep |want| >= 10 so clear-ahead early exit
        # does not fire before the force_reverse check.
        nav._heading = MagicMock(return_value=25.0)

        t0 = 100.0
        obs = Obs(
            now=t0,
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            s_total=1.0,
            yaw_total=0.0,
            v=0.0,
        )
        nav._begin_escape(obs, "hung", force_reverse=True)

        self.assertTrue(
            nav._force_reverse,
            "_begin_escape(force_reverse=True) must set self._force_reverse",
        )

        obs2 = Obs(
            now=t0 + ESCAPE_SETTLE_S + 0.05,
            frame=np.zeros((64, 64, 3), dtype=np.uint8),
            s_total=1.0,
            yaw_total=0.0,
            v=0.0,
        )
        est = types.SimpleNamespace(p=0.0, s=0.0)
        cmd = nav._escape(obs2, est)

        reversed_ = cmd.lin < 0 or nav._escape_phase == "reverse"
        self.assertTrue(
            reversed_,
            f"expected reverse after hung-up escape; got lin={cmd.lin} "
            f"phase={nav._escape_phase!r} mode={nav.mode!r} note={cmd.note!r}",
        )


if __name__ == "__main__":
    unittest.main()
