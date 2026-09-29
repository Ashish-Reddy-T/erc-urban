"""One-shot return-leg turn: do not servo a flipping compass."""
import math
import unittest

from erc_urban.modules.pivot import PulsePivot
from erc_urban.run import (
    ALIGN_EXIT,
    TURNAROUND_BURST_CAP,
    TURNAROUND_HOLD_M,
    _clear_turnaround,
    _credit_return_burst,
    _latch_return_turn,
    _turnaround_expired,
)


class TestLatchReturnTurn(unittest.TestCase):
    def test_latches_gps_sign_180_once(self):
        st = {}
        a = _latch_return_turn(st, math.radians(157.0))
        b = _latch_return_turn(st, math.radians(-179.0))
        self.assertEqual(a, 180.0)
        self.assertEqual(b, 180.0)
        self.assertEqual(st["turnaround_remain"], 180.0)

    def test_negative_gps_latches_minus_180(self):
        st = {}
        self.assertEqual(_latch_return_turn(st, math.radians(-140.0)), -180.0)

    def test_crumbs_win_when_they_point_back(self):
        st = {}
        remain = _latch_return_turn(st, math.radians(157.0), retreat_deg=168.0)
        self.assertAlmostEqual(remain, 168.0, places=1)

    def test_flipping_compass_cannot_relatch(self):
        st = {}
        _latch_return_turn(st, math.radians(157.0), retreat_deg=170.0)
        for flip in (131.0, 102.0, -179.0, 112.0, -133.0, 130.0):
            _latch_return_turn(st, math.radians(flip), retreat_deg=None)
        self.assertAlmostEqual(st["turnaround_remain"], 170.0, places=1)


class TestCreditReturnBurst(unittest.TestCase):
    def test_cap_releases_even_if_compass_never_converges(self):
        st = {}
        _latch_return_turn(st, math.radians(157.0))
        done = False
        for _ in range(TURNAROUND_BURST_CAP):
            done = _credit_return_burst(st, 1.60)
        self.assertTrue(done)
        self.assertTrue(st["turnaround_done"])
        self.assertEqual(st["turnaround_remain"], 0.0)

    def test_small_remain_releases_before_cap(self):
        st = {"turnaround_remain": ALIGN_EXIT + 5.0, "turnaround_bursts": 0}
        self.assertTrue(_credit_return_burst(st, 1.60))
        self.assertTrue(st["turnaround_done"])


class TestPulsePivotSkipsLearn(unittest.TestCase):
    def test_learn_flag_skips_rate_update(self):
        p = PulsePivot(rate_prior=30.0)
        p.learn = False
        p.rate = 30.0
        p._yaw0 = 0.0
        p._actual = 1.0
        p._mag = p.ang
        p._phase = "settle"
        p._settle_end = 0.0
        p._ts0 = None

        class _Tel:
            yaw = math.radians(80.0)
            hw_ts = 2.0

        p.command(180.0, _Tel(), now=10.0)
        self.assertEqual(p.rate, 30.0)


class TestPivotBoost(unittest.TestCase):
    """Panama: roots and a slope; a burst that does not turn must push harder."""

    def _burst(self, p, yaw_after_deg):
        class _T:
            yaw = 0.0
            hw_ts = 1.0
        p.command(60.0, _T(), now=0.0)                  # plan a burst
        p.command(60.0, _T(), now=5.0)                  # burst done
        _T.yaw = math.radians(yaw_after_deg)
        _T.hw_ts = 2.0
        p.command(60.0, _T(), now=10.0)                 # settle over -> learn
        return p

    def test_stalled_burst_raises_command(self):
        p = PulsePivot(ang=0.6, rate_prior=30.0)
        self._burst(p, 0.5)
        self.assertGreater(p.boost, 1.0)
        lin, ang, _ = p.command(60.0, type("T", (), {"yaw": 0.0, "hw_ts": 3.0})(), now=11.0)
        self.assertGreater(abs(ang), 0.6)
        self.assertLessEqual(abs(ang), 1.0)

    def test_responsive_burst_does_not_boost(self):
        p = PulsePivot(ang=0.6, rate_prior=30.0)
        self._burst(p, 30.0)
        self.assertEqual(p.boost, 1.0)


class TestTurnaroundRelease(unittest.TestCase):
    """Mission-3 (Wuhan): the post-spin hold lasted the whole leg, so after an
    escape left the goal behind, the rover could never turn back to it."""

    def _done(self):
        st = {}
        _latch_return_turn(st, math.radians(170.0))
        st["turnaround_done"] = True
        st["turnaround_remain"] = 0.0
        return st

    def test_hold_expires_after_driving(self):
        st = self._done()
        self.assertFalse(_turnaround_expired(st, 10.0))      # stamps the start
        self.assertFalse(_turnaround_expired(st, 10.0 + TURNAROUND_HOLD_M - 0.1))
        self.assertTrue(_turnaround_expired(st, 10.0 + TURNAROUND_HOLD_M + 0.1))

    def test_clear_allows_a_fresh_latch(self):
        st = self._done()
        _clear_turnaround(st, "escape")
        self.assertNotIn("turnaround_done", st)
        self.assertEqual(_latch_return_turn(st, math.radians(-150.0)), -180.0)

    def test_not_done_never_expires(self):
        self.assertFalse(_turnaround_expired({}, 100.0))


if __name__ == "__main__":
    unittest.main()
