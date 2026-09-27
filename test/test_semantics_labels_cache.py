"""
One SegFormer forward pass per frame.

run.step calls sem.labels(front, depth_hw) and TraversabilityVeto then calls
SegFormerSource.traversable(front, mask_hw) -> sem.labels(front, mask_hw) on the
same frame object at a different size. On CPU the forward pass dominates the tick.
"""
from __future__ import annotations

import unittest

import numpy as np

from erc_urban.modules.semantics import SemanticTraversability
from erc_urban.modules.traversability import SegFormerSource

DEPTH_HW = (144, 256)
MASK_HW = (120, 160)


class CountingModel:
    """Stub SegFormer whose logits depend on the pixels, so reuse is visible."""

    def __init__(self):
        self.calls = 0

    def __call__(self, pixel_values=None):
        import torch

        self.calls += 1
        _, _, h, w = pixel_values.shape
        base = pixel_values[:, :1].mean()
        ramp = torch.linspace(0.0, 1.0, w).expand(1, 1, h, w)
        logits = torch.cat([ramp, 1.0 - ramp, base.expand(1, 1, h, w),
                            torch.zeros(1, 1, h, w)], dim=1)
        return type("Out", (), {"logits": logits})()


class FakeProc:
    def __call__(self, images=None, return_tensors=None, size=None):
        import torch

        px = torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0)
        px = px.permute(2, 0, 1).unsqueeze(0)
        px = torch.nn.functional.interpolate(
            px, size=(size["height"], size["width"]), mode="bilinear",
            align_corners=False)
        return {"pixel_values": px}


def _sem():
    import torch

    sem = SemanticTraversability.__new__(SemanticTraversability)
    sem.ready = True
    sem.size = 32
    sem.device = "cpu"
    sem._torch = torch
    sem.proc = FakeProc()
    sem.model = CountingModel()
    sem.hard, sem.soft, sem.good, sem.notground = {3}, {1}, {0}, set()
    sem.conf = 0.0
    sem.min_blob_frac = 0.0
    sem._logits_rgb = None
    sem._logits = None
    return sem


def _frame(v: int) -> np.ndarray:
    return np.full((72, 128, 3), v, dtype=np.uint8)


class TestOneForwardPassPerFrame(unittest.TestCase):
    def test_tick_then_veto_on_same_frame_runs_model_once(self):
        sem = _sem()
        front = _frame(40)
        labels = sem.labels(front, DEPTH_HW)
        trav = SegFormerSource(semantics=sem).traversable(front, MASK_HW)
        self.assertEqual(sem.model.calls, 1)
        self.assertEqual(labels.shape, DEPTH_HW)
        self.assertEqual(trav.shape, MASK_HW)

    def test_cached_outputs_match_uncached(self):
        front = _frame(200)
        warm = _sem()
        warm.labels(front, DEPTH_HW)
        got = warm.labels(front, MASK_HW)
        got_conf = warm.last_conf
        fresh = _sem()
        want = fresh.labels(front, MASK_HW)
        np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(got_conf, fresh.last_conf)

    def test_new_frame_object_runs_model_again(self):
        sem = _sem()
        sem.labels(_frame(10), DEPTH_HW)
        sem.labels(_frame(10), DEPTH_HW)  # equal pixels, new object
        self.assertEqual(sem.model.calls, 2)

    def test_failed_forward_is_not_cached(self):
        sem = _sem()
        front = _frame(10)
        good_model = sem.model

        def boom(pixel_values=None):
            raise RuntimeError("oom")

        sem.model = boom
        self.assertIsNone(sem.labels(front, DEPTH_HW))
        sem.model = good_model
        self.assertIsNotNone(sem.labels(front, MASK_HW))
        self.assertEqual(good_model.calls, 1)


if __name__ == "__main__":
    unittest.main()
