"""creep._stream safety envelope: which aborts survive allow_tilt_abort=False.

Recovery reverses (back-and-fill, auto-recover) pass allow_tilt_abort=False
because backing off a high point is SUPPOSED to change pitch a lot, so the
delta-from-baseline and lurch aborts would fire on every successful recovery.
Absolute tilt is different: past ABORT_ABS_DEG the rover is going over no
matter which way it is driving.
"""
from __future__ import annotations
import unittest
from unittest.mock import patch

from erc_offroad import creep


def _att(roll, pitch):
    tilt = max(abs(roll), abs(pitch))
    return creep.Attitude(roll=roll, pitch=pitch, tilt=tilt, gz=0.0,
                          age_s=0.1, moving=True, orientation=None)


class TestStreamAborts(unittest.TestCase):
    BASE = {"roll": 0.0, "pitch": 0.0}

    def _run(self, attitudes, allow_tilt_abort):
        posts = []
        seq = iter(attitudes)
        with patch.object(creep, "_post", side_effect=lambda p, body, **kw: posts.append(body)), \
                patch.object(creep, "_get", return_value={}), \
                patch.object(creep, "attitude", side_effect=lambda d: next(seq)), \
                patch.object(creep.time, "sleep"):
            ok, reason = creep._stream(-0.3, 0.0, len(attitudes) / creep.HZ,
                                       self.BASE, None, allow_tilt_abort=allow_tilt_abort)
        return ok, reason, posts

    def _stopped(self, posts):
        return posts[-1] == {"command": {"linear": 0.0, "angular": 0.0}}

    def test_recovery_reverse_still_aborts_on_absolute_tilt(self):
        over = creep.ABORT_ABS_DEG + 2.0
        ok, reason, posts = self._run([_att(0, 5), _att(0, 15), _att(0, over), _att(0, over)],
                                      allow_tilt_abort=False)
        self.assertFalse(ok)
        self.assertIn("ABS", reason.upper())
        self.assertTrue(self._stopped(posts))
        # Stopped on the tick that crossed the limit, not after the full burst.
        self.assertEqual(len([p for p in posts if p["command"]["linear"] != 0.0]), 3)

    def test_recovery_reverse_ignores_delta_and_lurch(self):
        # 20 deg off baseline and a 10 deg single-tick jump, but under the
        # absolute limit: exactly what reversing off a hump looks like.
        ok, reason, _ = self._run([_att(0, 10), _att(0, 20), _att(0, 20)],
                                  allow_tilt_abort=False)
        self.assertTrue(ok, reason)

    def test_normal_stream_aborts_on_delta(self):
        ok, reason, posts = self._run([_att(0, 2), _att(0, creep.ABORT_DELTA_DEG + 1)],
                                      allow_tilt_abort=True)
        self.assertFalse(ok)
        self.assertIn("TILT ABORT", reason)
        self.assertTrue(self._stopped(posts))

    def test_normal_stream_aborts_on_lurch(self):
        ok, reason, _ = self._run([_att(0, 0), _att(0, creep.ABORT_RATE_DEG + 1)],
                                  allow_tilt_abort=True)
        self.assertFalse(ok)
        self.assertIn("LURCH", reason)


class TestRecoveryStopsOnAbort(unittest.TestCase):
    """A recovery reverse that hit the absolute limit must end the recovery,
    not carry on into a pivot or a second reverse while tipped over."""

    ABORT = (False, "ABS TILT ABORT abs=31.0 d_roll=+0.0 d_pitch=+31.0")

    def test_back_and_fill_skips_turn_after_abort(self):
        from erc_offroad import auto
        calls = []

        def fake_stream(lin, ang, secs, ref, log, **kw):
            calls.append((lin, ang))
            return self.ABORT

        with patch.object(auto, "_ref_attitude", return_value={"roll": 0.0, "pitch": 0.0}), \
                patch.object(auto, "_stream", side_effect=fake_stream), \
                patch.object(auto.time, "sleep"):
            out = auto.back_and_fill(None, side=1.0)
        self.assertEqual(len(calls), 1, "pivoted after the reverse aborted")
        self.assertIn("ABORT", out.upper())

    def test_nudge_auto_recover_stops_after_abort(self):
        import argparse
        import tempfile
        calls = []

        def fake_stream(lin, ang, secs, base, log, **kw):
            calls.append(lin)
            # First call is the nudge itself failing; recovery reverse then aborts.
            return (False, "TILT ABORT d_pitch=+13.0") if len(calls) == 1 else self.ABORT

        with tempfile.TemporaryDirectory() as out:
            args = argparse.Namespace(seconds=1.0, linear=0.3, angular=0.0, out=out,
                                      auto_recover=True, recover_linear=0.4)
            with patch.object(creep, "_ref_attitude", return_value={"roll": 0.0, "pitch": 0.0}), \
                    patch.object(creep, "_get", return_value={}), \
                    patch.object(creep, "attitude",
                                 side_effect=[_att(0, 5), _att(0, 31), _att(0, 31)]), \
                    patch.object(creep, "_stream", side_effect=fake_stream), \
                    patch.object(creep.time, "sleep"):
                rc = creep.cmd_nudge(args)
        self.assertEqual(rc, 2)
        self.assertEqual(len(calls), 2, "second recovery reverse ran after an abort")


if __name__ == "__main__":
    unittest.main()
