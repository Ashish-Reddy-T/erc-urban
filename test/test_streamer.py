"""Command streaming, the link watchdog, and exact pivot bursts (no live SDK)."""
from __future__ import annotations
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from erc_urban.modules.pivot import PulsePivot
from erc_urban.streamer import CommandStreamer, LinkWatchdog


class FakeClient:
    def __init__(self):
        self.sent: list[tuple[float, float, object]] = []
        self._lock = threading.Lock()

    def send_cmd(self, lin, ang, lamp=None):
        with self._lock:
            self.sent.append((lin, ang, lamp))


class TestCommandStreamer(unittest.TestCase):
    def test_streams_until_expiry_then_zeros(self):
        c = FakeClient()
        s = CommandStreamer(c, send=True, hz=50.0, lamp=1)
        s.set(0.3, -0.1, hold_s=0.15)
        s.start()
        time.sleep(0.4)
        s.stop()
        moving = [x for x in c.sent if x[0] == 0.3]
        self.assertGreater(len(moving), 3, "must repeat the command, not post once")
        # After expiry it posts zeros, never the stale command again.
        last_move = max(i for i, x in enumerate(c.sent) if x[0] == 0.3)
        self.assertTrue(all(x[:2] == (0.0, 0.0) for x in c.sent[last_move + 1:]))

    def test_dry_run_sends_nothing(self):
        c = FakeClient()
        s = CommandStreamer(c, send=False, hz=50.0)
        s.set(0.3, 0.0, hold_s=1.0)
        s.start()
        time.sleep(0.1)
        s.stop(lamp_off=True)
        self.assertEqual(c.sent, [])

    def test_stop_turns_lamp_off(self):
        c = FakeClient()
        s = CommandStreamer(c, send=True, hz=50.0, lamp=1)
        s.stop(lamp_off=True)
        self.assertEqual(c.sent[-1], (0.0, 0.0, 0))

    def test_moving_reflects_live_nonzero_command(self):
        s = CommandStreamer(FakeClient(), send=True)
        self.assertFalse(s.moving)
        s.set(0.2, 0.0, hold_s=5.0)
        self.assertTrue(s.moving)
        self.assertTrue(s.forward)
        s.set(0.0, 0.0, hold_s=5.0)
        self.assertFalse(s.moving)

    def test_dry_run_never_reports_motion(self):
        """Wuhan dry run: a 'forward' flag with nothing sent tripped the
        dead-rpm check and switched a healthy unit to visual odometry."""
        s = CommandStreamer(FakeClient(), send=False)
        s.set(0.2, 0.1, hold_s=5.0)
        self.assertFalse(s.moving)
        self.assertFalse(s.forward)


def _frame(v):
    return np.full((36, 64, 3), v, np.uint8)


class TestLinkWatchdog(unittest.TestCase):
    def test_frozen_while_moving_goes_stale(self):
        w = LinkWatchdog(stale_s=4.0)
        self.assertFalse(w.update(_frame(10), 1.0, moving=True, now=0.0))
        self.assertFalse(w.update(_frame(10), 1.0, moving=True, now=3.0))
        self.assertTrue(w.update(_frame(10), 1.0, moving=True, now=5.0))

    def test_stays_down_after_halt_until_data_changes(self):
        """Halting makes `moving` False; that alone must not count as recovery."""
        w = LinkWatchdog(stale_s=4.0)
        w.update(_frame(10), 1.0, moving=True, now=0.0)
        self.assertTrue(w.update(_frame(10), 1.0, moving=True, now=5.0))
        self.assertTrue(w.update(_frame(10), 1.0, moving=False, now=6.0))
        self.assertFalse(w.update(_frame(10), 2.0, moving=False, now=7.0))
        self.assertFalse(w.down)

    def test_changing_image_is_not_stale(self):
        w = LinkWatchdog(stale_s=4.0)
        for t in range(10):
            self.assertFalse(w.update(_frame(10 * t), 1.0, moving=True, now=float(t)))

    def test_stationary_is_never_stale(self):
        w = LinkWatchdog(stale_s=4.0)
        for t in range(10):
            self.assertFalse(w.update(_frame(10), 1.0, moving=False, now=float(t)))


class TestExactBursts(unittest.TestCase):
    def _tel(self, ts):
        return SimpleNamespace(yaw=0.0, hw_ts=ts)

    def test_burst_remaining_counts_down(self):
        p = PulsePivot(rate_prior=30.0)
        p.command(60.0, self._tel(1.0), now=0.0)
        self.assertGreater(p.burst_remaining(0.0), 0.3)
        self.assertEqual(p.burst_remaining(100.0), 0.0)

    def test_late_tick_not_counted_as_rotation_when_streamed(self):
        for exact, expect_capped in ((True, True), (False, False)):
            p = PulsePivot(rate_prior=30.0, exact_bursts=exact)
            p.command(60.0, self._tel(1.0), now=0.0)
            planned = p._dur
            p.command(60.0, self._tel(1.0), now=planned + 0.7)   # slow tick
            if expect_capped:
                self.assertAlmostEqual(p._actual, planned)
            else:
                self.assertAlmostEqual(p._actual, planned + 0.7)


if __name__ == "__main__":
    unittest.main()
