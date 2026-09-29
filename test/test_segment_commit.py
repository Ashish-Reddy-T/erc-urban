import math
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from erc_urban.modules.segment_commit import SegmentCommit


def motion(dx=0, yaw=0, confidence=1):
    return SimpleNamespace(dx=dx, dy=0, dyaw=math.radians(yaw), confidence=confidence)


class SegmentCommitTest(unittest.TestCase):
    def test_live_obstacle_interrupts_controller_commit(self):
        import numpy as np
        import erc_urban.run as R
        from erc_urban.modules.odometry import OdomDelta
        from test.test_freespace_urban import _open, _row_at, CAL, HW
        from erc_urban.modules.freespace import FreeSpaceUrban
        state = {}
        chunk = SegmentCommit(0)
        chunk.begin(0, 5, .8, 2)
        R._fan_state(state)["segment"] = chunk
        ground = _open()
        ground[:_row_at(.4), :] = False
        sent = []
        with patch.object(R, "FAN_SEGMENT_M", 5), \
                patch.object(R, "_FSU", FreeSpaceUrban(CAL, seg_hw=HW)), \
                patch.object(R, "_halt", lambda client: sent.append(0)), \
                patch.object(R, "_cmd", lambda *a, **k: self.fail("must not drive")), \
                patch.object(R, "_publish_overlay", lambda *a, **k: None), \
                patch.object(R.time, "monotonic", lambda: 3):
            R._fan_step(None, SimpleNamespace(yaw=0, raw={}), None, 20, 0,
                        SimpleNamespace(bearing=0), ground, None, state,
                        OdomDelta(0, 0, 0, 1, reason="wheel"), "", 0)
        self.assertEqual(sent, [0])
        self.assertFalse(chunk.active)

    def test_waits_for_stationary_settling(self):
        c = SegmentCommit(0)
        self.assertEqual(c.update(motion(), 1), "settling")
        c.update(motion(.1), 2)
        self.assertEqual(c.update(motion(), 3), "settling")
        self.assertEqual(c.update(motion(), 4), "ready")

    def test_clearance_cap_and_gyro_following(self):
        c = SegmentCommit(0)
        self.assertTrue(c.begin(20, 3, .8, 2))
        self.assertAlmostEqual(c.remaining, 2.2)
        c.update(motion(.5, 10), 3)
        self.assertAlmostEqual(c.heading, 10)
        self.assertAlmostEqual(c.remaining, 1.7)
        self.assertEqual(c.update(motion(1.8), 4), "segment complete")
        self.assertFalse(c.active)

    def test_stale_and_timeout_cancel(self):
        c = SegmentCommit(0)
        c.begin(0, 10, .8, 2)
        self.assertEqual(c.remaining, 5)
        self.assertEqual(c.update(motion(confidence=0), 3), "waiting for fresh odometry")
        self.assertFalse(c.active)
        c.begin(0, 10, .8, 5)
        self.assertEqual(c.update(motion(), 36), "segment complete")

    def test_short_clearance_never_commits(self):
        c = SegmentCommit(0)
        self.assertFalse(c.begin(0, .9, .8, 2))
        self.assertFalse(c.active)
