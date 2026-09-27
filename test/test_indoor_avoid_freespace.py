"""The cone AVOID pass must respect the free-space stop, like DRIVE does."""
from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock

import numpy as np

from erc_indoor.freespace import CAP_M, FreeSpace, path_clear
from erc_indoor.navigator import CONE_PASS_LIN, STOP_M, Navigator, Obs


def _free_with_pts(pts: np.ndarray) -> MagicMock:
    """FreeSpaceEstimator stand-in that builds FreeSpace the way estimate() does."""
    bins = np.arange(-50.0, 50.0 + 1e-6, 5.0)
    fs = FreeSpace(bins, np.array([path_clear(pts, float(b)) for b in bins]),
                   path_clear(pts, 0.0), pts)
    free = MagicMock()
    free.estimate = MagicMock(return_value=fs)
    free.steer = MagicMock(return_value=None)
    return free


def _nav_mid_pass(free) -> Navigator:
    route = MagicMock()
    route.turns = []
    route.checkpoints = {}
    route.checkpoint_label = MagicMock(return_value="")
    route.finish_s = 1e9
    route.__len__ = MagicMock(return_value=10)
    enc = MagicMock()
    enc.encode_full = MagicMock(return_value=(None, None))
    loc = MagicMock()
    loc.estimate = MagicMock(return_value=types.SimpleNamespace(
        p=1.0, s=1.0, node=1, spread_m=0.5, sim=0.9))
    nav = Navigator(route, loc, enc, freespace=free, verbose=False)
    nav._window = MagicMock(return_value=(0, 10))
    nav.mode = "AVOID"
    nav._avoid_phase = "pass"
    nav._pass_end = 101.0
    return nav


def _obs(now: float = 100.0) -> Obs:
    return Obs(now=now, frame=np.zeros((64, 64, 3), np.uint8),
               s_total=1.0, yaw_total=0.0, v=0.0)


class TestIndoorAvoidFreespace(unittest.TestCase):
    def test_wall_ahead_mid_pass_stops_forward_motion(self):
        # obstacle feet 0.3 m straight ahead, across the whole body strip
        pts = np.array([[0.3, y] for y in np.linspace(-0.3, 0.3, 13)])
        self.assertLess(path_clear(pts, 0.0), STOP_M)
        nav = _nav_mid_pass(_free_with_pts(pts))
        cmd = nav.step(_obs())
        self.assertLessEqual(cmd.lin, 0.0, f"drove on: mode={cmd.mode} note={cmd.note}")
        self.assertEqual(nav.mode, "ESCAPE")

    def test_open_floor_mid_pass_keeps_passing(self):
        # estimate() drops boundary points at/above the horizon: open floor is
        # an EMPTY pts array, which path_clear reads as CAP_M clear.
        nav = _nav_mid_pass(_free_with_pts(np.zeros((0, 2))))
        cmd = nav.step(_obs())
        self.assertEqual(cmd.mode, "AVOID")
        self.assertAlmostEqual(cmd.lin, CONE_PASS_LIN)
        self.assertEqual(path_clear(np.zeros((0, 2)), 0.0), CAP_M)

    def test_no_freespace_estimator_keeps_old_behaviour(self):
        nav = _nav_mid_pass(None)
        cmd = nav.step(_obs())
        self.assertEqual(cmd.mode, "AVOID")
        self.assertAlmostEqual(cmd.lin, CONE_PASS_LIN)


if __name__ == "__main__":
    unittest.main()
