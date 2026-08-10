"""Tests for classical near-field obstacle stop + front/rear logic."""
from __future__ import annotations
import unittest

import numpy as np

from erc_urban.modules.obstacle import (
    analyze_obstacle,
    front_blocked_rear_clear,
)


class TestObstacle(unittest.TestCase):
    def test_open_road_clear(self):
        img = np.full((240, 320, 3), 40, np.uint8)
        img[120:, :] = (90, 90, 88)
        img[:80, :] = (30, 30, 50)
        # peripheral streetlights (left + right upper)
        img[10:30, 10:35] = 220
        img[15:35, 285:310] = 230
        img[5:25, 5:25] = 200
        pv = analyze_obstacle(img)
        self.assertFalse(pv.blocked)

    def test_dark_blob_blocked(self):
        img = np.full((240, 320, 3), 80, np.uint8)
        img[100:230, 70:250] = (15, 15, 15)
        pv = analyze_obstacle(img)
        self.assertTrue(pv.blocked)

    def test_lamp_on_wall_blocked(self):
        rng = np.random.default_rng(1)
        img = np.full((240, 320, 3), 35, np.uint8)
        tex = rng.integers(25, 60, size=(240, 320, 3), dtype=np.uint8)
        img[:] = tex
        img[50:170, 80:240] = np.clip(
            img[50:170, 80:240].astype(int) + 130, 0, 255
        ).astype(np.uint8)
        pv = analyze_obstacle(img)
        self.assertTrue(pv.blocked)

    def test_front_closed_rear_open(self):
        rng = np.random.default_rng(2)
        front = rng.integers(30, 70, size=(240, 320, 3), dtype=np.uint8)
        front[60:160, 100:220] = np.clip(front[60:160, 100:220].astype(int) + 120, 0, 255)
        rear = np.full((240, 320, 3), 35, np.uint8)
        rear[140:, :] = (70, 70, 68)
        rear[8:20, 50:60] = 210
        rear[12:28, 180:200] = 230
        rear[5:18, 260:280] = 200
        hit = front_blocked_rear_clear(front, rear)
        self.assertIsNotNone(hit)
        self.assertTrue(hit.blocked)
        self.assertTrue(
            "rear_open" in hit.reason or hit.reason in ("lamp_wall", "fov_filled")
        )


if __name__ == "__main__":
    unittest.main()
