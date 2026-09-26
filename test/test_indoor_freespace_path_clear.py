"""Empty/None free-space feet must be UNKNOWN (blocked), not fail-open clear."""
from __future__ import annotations

import unittest

import numpy as np

from erc_indoor.freespace import CAP_M, path_clear


class TestPathClearEmptyUnknown(unittest.TestCase):
    def test_empty_pts_returns_blocked(self):
        """No obstacle feet means unknown - treat as blocked (0), not CAP_M clear."""
        got = path_clear(np.zeros((0, 2)), 0.0)
        self.assertEqual(got, 0.0)
        self.assertNotEqual(got, CAP_M)

    def test_none_pts_returns_blocked(self):
        got = path_clear(None, 0.0)
        self.assertEqual(got, 0.0)
        self.assertNotEqual(got, CAP_M)

    def test_empty_strip_with_pts_elsewhere_still_cap(self):
        """Points exist but none in this heading's strip — that heading is clear to CAP."""
        # Obstacle foot straight ahead; heading 90 deg (left) has an empty strip.
        pts = np.array([[1.0, 0.0]], dtype=float)
        got = path_clear(pts, 90.0)
        self.assertEqual(got, CAP_M)


if __name__ == "__main__":
    unittest.main()
