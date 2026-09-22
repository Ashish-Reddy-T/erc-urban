"""One-shot return-leg turn: do not servo a flipping compass."""
import math
import unittest

from erc_urban.modules.pivot import PulsePivot
from erc_urban.run import (
    ALIGN_EXIT,
    TURNAROUND_BURST_CAP,
    _credit_return_burst,
    _latch_return_turn,
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


if __name__ == "__main__":
    unittest.main()
