"""Offline tests for classical dirt-path perception + veto."""
from __future__ import annotations
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from erc_urban.interfaces import Observation
from erc_urban.modules.baseline import BaselinePlanner
from erc_urban.modules.dirt_path import DirtPathGate, analyze, rescore

FIX = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> np.ndarray:
    return np.asarray(Image.open(FIX / name).convert("RGB"))


def _obs(front: np.ndarray, goal_bearing: float = 0.0) -> Observation:
    return Observation(
        front=front,
        rear=None,
        lat=0.0,
        lon=0.0,
        yaw=0.0,
        stamp=1.0,
        goal_bearing=goal_bearing,
        goal_range_m=40.0,
    )


class TestDirtAnalyze(unittest.TestCase):
    def test_synth_dirt_path_not_blocked(self):
        pv = analyze(_load("synth_dirt_path.png"))
        self.assertGreater(pv.dirt_frac, 0.15)
        self.assertFalse(pv.blocked)
        self.assertLess(abs(pv.center_offset), 0.35)

    def test_synth_grass_wall_blocked(self):
        pv = analyze(_load("synth_grass_wall.png"))
        self.assertTrue(pv.blocked)
        self.assertGreater(pv.grass_frac, 0.5)

    def test_dirt_left_positive_offset(self):
        pv = analyze(_load("synth_dirt_left.png"))
        self.assertFalse(pv.blocked)
        self.assertGreater(pv.center_offset, 0.2)

    def test_grass_field_panel_blocked(self):
        pv = analyze(_load("grass_field.png"))
        self.assertTrue(pv.blocked)

    def test_synth_buried_blocked(self):
        pv = analyze(_load("synth_buried.png"))
        self.assertTrue(pv.blocked)

    def test_buried_lens_frame_blocked(self):
        pv = analyze(_load("buried_lens.png"))
        self.assertTrue(pv.blocked)

    def test_night_asphalt_drivable(self):
        pv = analyze(_load("synth_night_asphalt.png"))
        self.assertFalse(pv.blocked)
        self.assertGreater(pv.dirt_frac, 0.25)
        self.assertLess(pv.mean_bright, 70.0)

    def test_day_asphalt_drivable(self):
        pv = analyze(_load("synth_day_asphalt.png"))
        self.assertFalse(pv.blocked)
        self.assertGreater(pv.dirt_frac, 0.20)


class TestDirtGateAndRescore(unittest.TestCase):
    def test_gate_keeps_straight_on_dirt(self):
        front = _load("synth_dirt_path.png")
        obs = _obs(front)
        cands = BaselinePlanner().propose(obs, k=11)
        cands = rescore(cands, analyze(front))
        mask = DirtPathGate().mask(obs, cands)
        self.assertTrue(mask.any())
        mid = len(cands) // 2
        self.assertTrue(mask[mid])

    def test_gate_rejects_all_on_grass_wall(self):
        front = _load("synth_grass_wall.png")
        obs = _obs(front)
        cands = BaselinePlanner().propose(obs, k=11)
        mask = DirtPathGate().mask(obs, cands)
        self.assertFalse(mask.any())

    def test_rescore_prefers_left_when_path_left(self):
        front = _load("synth_dirt_left.png")
        path = analyze(front)
        obs = _obs(front, goal_bearing=0.0)
        cands = rescore(BaselinePlanner().propose(obs, k=11), path)
        best = max(cands, key=lambda c: c.score)
        self.assertGreater(best.waypoints[-1, 1], 0.0)

    def test_gate_rejects_buried_frame(self):
        front = _load("buried_lens.png")
        obs = _obs(front)
        cands = BaselinePlanner().propose(obs, k=11)
        mask = DirtPathGate().mask(obs, cands)
        self.assertFalse(mask.any())

    def test_gate_keeps_arcs_on_night_asphalt(self):
        front = _load("synth_night_asphalt.png")
        obs = _obs(front)
        cands = BaselinePlanner().propose(obs, k=11)
        mask = DirtPathGate().mask(obs, cands)
        self.assertTrue(mask.any())


class TestVillageBlocked(unittest.TestCase):
    def test_controller_zeros_when_blocked(self):
        from erc_urban.modules.baseline import VillageGotoController
        lin, ang = VillageGotoController().command_bearing(0.0, 40.0, blocked=True)
        self.assertEqual(lin, 0.0)
        self.assertEqual(ang, 0.0)


if __name__ == "__main__":
    unittest.main()
