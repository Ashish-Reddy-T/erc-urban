"""
Regression test: CommitWindow must expire when the whole committed path has
been driven past in a single tick, not keep holding a stale behind-the-robot
path.

advance() is supposed to "drop waypoints the rover has already driven past"
(see modules/commit.py docstring), by filtering to `path[:, 0] > 0`. But the
filter is only applied `if keep.any()`: when EVERY waypoint has fallen behind
(keep is all False), the filter is skipped and self.path is left holding the
stale, now-fully-behind array. CommitWindow.active only checks length, so the
window stays active with backward-pointing waypoints. The next offer() call
then blends those backward points into the executed path instead of adopting
the fresh plan outright.
"""
from __future__ import annotations
import unittest
from dataclasses import dataclass

import numpy as np

from erc_urban_test.modules.commit import CommitWindow


@dataclass
class FakeDelta:
    dx: float
    dy: float
    dyaw: float
    confidence: float
    distance_m: float


class TestCommitWindowFullyBehind(unittest.TestCase):
    def test_advance_past_all_waypoints_expires_the_window(self):
        cw = CommitWindow(max_age_s=5.0, max_distance_m=5.0)
        cw.path = np.array([[0.1, 0.0], [0.2, 0.0]])
        cw.age_s = 0.0
        cw.distance_m = 0.0

        # Drive forward 1.0 m: both committed waypoints (0.1 m, 0.2 m ahead)
        # are now behind the rover.
        delta = FakeDelta(dx=1.0, dy=0.0, dyaw=0.0, confidence=1.0, distance_m=1.0)
        cw.advance(delta, 0.2)

        self.assertFalse(
            cw.active,
            f"CommitWindow stayed active with a fully-behind path: {cw.path!r}",
        )

    def test_offer_after_fully_behind_adopts_new_plan_outright(self):
        cw = CommitWindow(max_age_s=5.0, max_distance_m=5.0)
        cw.path = np.array([[0.1, 0.0], [0.2, 0.0]])
        cw.age_s = 0.0
        cw.distance_m = 0.0

        delta = FakeDelta(dx=1.0, dy=0.0, dyaw=0.0, confidence=1.0, distance_m=1.0)
        cw.advance(delta, 0.2)

        new_path = np.array([[0.5, 0.3], [1.0, 0.3]])
        out = cw.offer(new_path)

        np.testing.assert_allclose(out, new_path)
        self.assertEqual(cw.last_reason, "adopted")


if __name__ == "__main__":
    unittest.main()
