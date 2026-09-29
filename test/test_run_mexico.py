"""Mexico lane runner: pursuit prior, lane rays, decision rules (no rover, no models)."""
import math
import os
import types
import unittest

import numpy as np

from erc_urban import run_mexico as M
from erc_urban.modules.lane import LaneFinder

P = types.SimpleNamespace


def _ll(e, n, lat0=19.570217, lon0=-98.814691):
    return lat0 + n / 110540.0, lon0 + e / (111320.0 * math.cos(math.radians(lat0)))


class TestPursuit(unittest.TestCase):
    def setUp(self):
        a, b = _ll(0, 0), _ll(0, 50)                    # line due north, 50 m
        self.a, self.b = P(lat=a[0], lon=a[1]), P(lat=b[0], lon=b[1])

    def test_on_line_points_along_it(self):
        lat, lon = _ll(0, 10)
        self.assertAlmostEqual(M.wrap(M.pursuit_bearing(lat, lon, self.a, self.b, 10) - 0), 0, delta=1)

    def test_drifted_right_points_back_left(self):
        lat, lon = _ll(4, 10)                           # 4 m east of the line
        brg = M.pursuit_bearing(lat, lon, self.a, self.b, 10)
        self.assertLess(M.wrap(brg), -15)               # heads NW, back to the line

    def test_overshoot_turns_back_to_goal(self):
        lat, lon = _ll(0, 60)                           # 10 m past b
        brg = M.pursuit_bearing(lat, lon, self.a, self.b, 10)
        self.assertAlmostEqual(abs(M.wrap(brg)), 180, delta=1)


class TestLaneRays(unittest.TestCase):
    def setUp(self):
        self.lf = LaneFinder(M.CAL, hw=M.SEG_HW)
        self.h, self.w = M.SEG_HW

    def _lane(self, centre_col, half):
        g = np.zeros((self.h, self.w), bool)
        hz = int(self.lf.cal.horizon_v()) + 2
        for v in range(hz, self.h):
            f = (v - hz) / max(1, self.h - hz)          # widen toward the bottom (perspective)
            hw = int(half * (0.15 + f))
            g[v, max(0, centre_col - hw):min(self.w, centre_col + hw)] = True
        return g

    def test_straight_lane(self):
        e = self.lf.estimate(self._lane(self.w // 2, 60))
        self.assertIsNotNone(e.steer_deg)
        self.assertLessEqual(abs(e.steer_deg), 10)
        self.assertGreater(e.best_m, 2.0)

    def test_lane_to_the_right_steers_right(self):
        e = self.lf.estimate(self._lane(int(self.w * 0.75), 40))
        self.assertLess(e.steer_deg, -5)

    def test_nothing_is_blocked(self):
        e = self.lf.estimate(np.zeros((self.h, self.w), bool))
        self.assertIsNone(e.steer_deg)


class TestDecide(unittest.TestCase):
    def _m(self):
        m = M.Mexico.__new__(M.Mexico)
        m.turning, m.fwd_since, m.last_rev = 0.0, None, 0.0
        return m

    def _e(self, steer, best=4.0, left=1.0, right=1.0):
        return types.SimpleNamespace(steer_deg=steer, best_m=best, ahead_m=best, left_m=left, right_m=right)

    def test_facing_away_turns_once_one_way(self):
        m = self._m()
        lin, ang, mode = m.decide(self._e(0), +130, None, 0.0)
        self.assertEqual((lin, ang > 0), (0.0, True))
        # still > exit threshold on the other side of 0? stays committed to the same direction
        lin, ang, mode = m.decide(self._e(0), +70, None, 0.2)
        self.assertTrue(ang > 0 and "TURN" in mode)
        lin, ang, mode = m.decide(self._e(0), +10, None, 0.4)
        self.assertIn("LANE", mode)

    def test_blocked_turns_toward_more_lane_never_reverses(self):
        m = self._m()
        lin, ang, mode = m.decide(self._e(None, best=0.0, left=0.0, right=2.0), 0, None, 0.0)
        self.assertEqual(lin, 0.0)
        self.assertLess(ang, 0)                           # right has more lane

    def test_small_steer_still_turns(self):
        m = self._m()
        lin, ang, mode = m.decide(self._e(5.0), 0, None, 0.0)
        self.assertGreaterEqual(abs(ang), M.ANG_MIN)
        self.assertGreater(lin, 0.0)

    def test_stuck_reverses_briefly_once(self):
        m = self._m()
        tel = types.SimpleNamespace(raw={"rpms": [[0, 0, 0, 0, 0]]})
        m.decide(self._e(0), 0, tel, 100.0)
        lin, ang, mode = m.decide(self._e(0), 0, tel, 100.0 + M.STUCK_S + 0.1)
        self.assertLess(lin, 0.0)
        lin, ang, mode = m.decide(self._e(0), 0, tel, 100.0 + 2 * M.STUCK_S + 0.3)
        self.assertGreater(lin, 0.0)                      # not again within 10 s


if __name__ == "__main__":
    unittest.main()


class TestLaneSim(unittest.TestCase):
    """Closed loop (erc_urban/sim_lane.py): every scenario finishes all checkpoints
    with the wheels never in the grass."""

    def test_all_scenarios(self):
        from erc_urban import sim_lane as S
        for w in S.SCENARIOS:
            r = S.LaneSim(w).run()
            self.assertTrue(r["done"], r)
            self.assertEqual(r["wheels_off_lane_s"], 0.0, r)
            self.assertLess(r["t"], 480, r)
