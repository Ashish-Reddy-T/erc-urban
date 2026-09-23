"""
Global place descriptors for localizing a live frame against the teach lap.

DINOv2 ViT-S/14 (already in the torch hub cache, no download at the venue):
CLS token concatenated with GeM-pooled patch tokens, L2-normalised. One
forward pass per tick; the teach lap's descriptors are computed once by
`build_map` and stored in the route map.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

HUB_DIR = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
# Multiple of the 14 px patch; keeps the 16:9 frame aspect.
IN_H, IN_W = 224, 392
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


class PlaceEncoder:
    def __init__(self, device: str | None = None):
        self.device = device or auto_device()
        if os.path.isdir(HUB_DIR):
            m = torch.hub.load(HUB_DIR, "dinov2_vits14", source="local")
        else:
            m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.model = m.eval().to(self.device)

    @torch.no_grad()
    def encode_full(self, rgb: np.ndarray) -> tuple[np.ndarray, torch.Tensor]:
        """Place descriptor AND the normalised patch-token grid (16 x 28 x 384,
        on device) from ONE forward pass. The grid gives heading (bearing.
        dino_shift) where SIFT finds nothing: textureless walls, night vs day."""
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255.0)
        x = F.interpolate(x, size=(IN_H, IN_W), mode="bilinear", align_corners=False)
        x = ((x - _MEAN) / _STD).to(self.device)
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]
        patch = out["x_norm_patchtokens"]
        gem = patch.clamp(min=1e-6).pow(3).mean(dim=1).pow(1.0 / 3.0)
        d = torch.cat([F.normalize(cls, dim=-1), F.normalize(gem, dim=-1)], dim=-1)
        desc = F.normalize(d, dim=-1)[0].cpu().numpy().astype(np.float32)
        tok = F.normalize(patch[0], dim=-1).reshape(IN_H // 14, IN_W // 14, -1)
        return desc, tok

    @torch.no_grad()
    def encode(self, rgb_batch: list[np.ndarray]) -> np.ndarray:
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1) for im in rgb_batch])
        x = x.float().div(255.0)
        x = F.interpolate(x, size=(IN_H, IN_W), mode="bilinear", align_corners=False)
        x = ((x - _MEAN) / _STD).to(self.device)
        out = self.model.forward_features(x)
        cls = out["x_norm_clstoken"]
        patch = out["x_norm_patchtokens"].clamp(min=1e-6)
        gem = patch.pow(3).mean(dim=1).pow(1.0 / 3.0)
        d = torch.cat([F.normalize(cls, dim=-1), F.normalize(gem, dim=-1)], dim=-1)
        return F.normalize(d, dim=-1).cpu().numpy().astype(np.float32)

    def encode_one(self, rgb: np.ndarray) -> np.ndarray:
        return self.encode([rgb])[0]
