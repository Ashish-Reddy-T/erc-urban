"""
Semantic traversability — the surface judgement geometry cannot make.

A dirt track and the flat grass verge beside it are geometrically identical, and
so are a puddle and solid ground. SegFormer-B2 on ADE20K gives NAMED classes
(road, path, earth, grass, person, car, water), which is what the veto needs;
class-agnostic masks would still need a heuristic to say which region is
drivable.

Three categories:
  HARD  — never drive here, whatever the geometry says: dynamic actors, and the
          negative obstacles geometry cannot see (water, pools, stairwells).
  SOFT  — drivable but off-track (grass, plants). Penalised, never vetoed.
  GOOD  — the track itself. Confirms the path, never forces it.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

_MODEL_ID = os.environ.get(
    "SEG_MODEL", "nvidia/segformer-b2-finetuned-ade-512-512"
)

# Matched by NAME against the checkpoint's id2label, so a different ADE20K
# checkpoint with reordered ids still works.
_HARD_NAMES = {
    "person", "car", "truck", "bus", "van", "minibike", "motorbike",
    "bicycle", "animal", "boat", "airplane",
    "water", "river", "sea", "lake", "swimming pool", "pool",
    "fountain", "waterfall", "stairs", "stairway", "escalator",
}
_SOFT_NAMES = {
    "grass", "plant", "flower", "field", "tree",
}
_GOOD_NAMES = {
    "road", "sidewalk", "pavement", "earth", "ground", "path",
    "dirt track", "land", "floor", "sand", "runway",
}
# Never ground, whatever else they are — excluded from the traversable set
# so they cannot be mistaken for path in the near-field band.
_NOT_GROUND_NAMES = {
    "sky", "building", "house", "skyscraper", "wall", "ceiling",
    "mountain", "hill", "tower", "bridge",
}


def _match(id2label: dict, names: set[str]) -> set[int]:
    out = set()
    for i, lbl in id2label.items():
        # ADE20K labels are comma-separated synonym lists ("earth, ground").
        parts = {p.strip().lower() for p in str(lbl).split(",")}
        if parts & names:
            out.add(int(i))
    return out


class SemanticTraversability:
    def __init__(
        self,
        device: str = "auto",
        size: int = 512,
        conf: float = 0.60,
        min_blob_frac: float = 0.004,
    ):
        self.size = size
        # A HARD class can stop the mission, so it clears two bars: confident
        # per-pixel, and part of a region big enough to be real. Sunlit or wet
        # dirt reads as "water" to ADE20K models.
        self.conf = conf
        self.min_blob_frac = min_blob_frac
        self.ready = False
        self.model = None
        self.proc = None
        self.hard: set[int] = set()
        self.soft: set[int] = set()
        self.good: set[int] = set()
        self.notground: set[int] = set()
        self.device = device
        # The urban tick segments the same frame twice at different out_hw
        # (depth grid, then TraversabilityVeto.mask_hw). Holding the frame
        # itself, not id(), keeps a freed frame's id from matching a new one.
        self._logits_rgb: Optional[np.ndarray] = None
        self._logits = None
        self._load(device)

    def _load(self, device: str) -> None:
        try:
            import torch
            from transformers import (
                AutoImageProcessor,
                AutoModelForSemanticSegmentation,
            )

            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
                    device = "mps"
                else:
                    device = "cpu"
            self.device = device
            self.proc = AutoImageProcessor.from_pretrained(_MODEL_ID)
            self.model = (
                AutoModelForSemanticSegmentation.from_pretrained(_MODEL_ID)
                .to(device)
                .eval()
            )
            id2label = self.model.config.id2label
            self.hard = _match(id2label, _HARD_NAMES)
            self.soft = _match(id2label, _SOFT_NAMES)
            self.good = _match(id2label, _GOOD_NAMES)
            self.notground = _match(id2label, _NOT_GROUND_NAMES)
            self._torch = torch
            self.ready = True
            print(
                f"[Semantics] SegFormer on {device} "
                f"(hard={len(self.hard)} soft={len(self.soft)} "
                f"good={len(self.good)} classes)"
            )
        except Exception as e:
            print(f"[Semantics] unavailable ({e})")
            self.ready = False

    def labels(self, rgb: np.ndarray, out_hw: tuple[int, int]) -> Optional[np.ndarray]:
        """
        Per-pixel ADE20K class ids, resampled to out_hw so it aligns with the
        depth map. The processor does a plain resize (no crop), so resizing the
        logits back to out_hw undoes the distortion exactly.
        """
        if not self.ready:
            return None
        torch = self._torch
        try:
            if rgb is self._logits_rgb and self._logits is not None:
                logits = self._logits
            else:
                inp = self.proc(
                    images=rgb,
                    return_tensors="pt",
                    size={"height": self.size, "width": self.size},
                )
                px = inp["pixel_values"].to(self.device)
                with torch.no_grad():
                    logits = self.model(pixel_values=px).logits
                self._logits_rgb, self._logits = rgb, logits
            up = torch.nn.functional.interpolate(
                logits, size=out_hw, mode="bilinear", align_corners=False
            )
            prob = up.softmax(dim=1)[0]
            conf, idx = prob.max(dim=0)
            self.last_conf = conf.to("cpu").numpy().astype(np.float32)
            return idx.to("cpu").numpy().astype(np.int16)
        except Exception as e:
            print(f"  semantics error: {e}", flush=True)
            return None

    def categorize(self, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(hard, soft, good) boolean masks over the label map."""
        hard = np.isin(labels, list(self.hard)) if self.hard else np.zeros_like(labels, bool)
        soft = np.isin(labels, list(self.soft)) if self.soft else np.zeros_like(labels, bool)
        good = np.isin(labels, list(self.good)) if self.good else np.zeros_like(labels, bool)
        hard = self._filter_hard(hard)
        # Traversable BY EXCLUSION, not allow-list. A gravel track segments as
        # "rock", a leaf-littered one as any of a dozen classes; enumerating
        # them fails closed, i.e. refuses to drive on a good path. Vegetation
        # and hazards are the small well-defined sets.
        notground = (
            np.isin(labels, list(self.notground))
            if self.notground else np.zeros_like(labels, bool)
        )
        good = good | (~hard & ~soft & ~notground)
        return hard, soft, good

    def track(
        self, good: np.ndarray, fov_deg: float = 90.0, lower_frac: float = 0.45
    ) -> tuple[float, float, Optional[float]]:
        """
        Where is the path, and how much of it can we see?

        Returns (fraction_of_frame, fraction_of_lower_band, bearing_rad), the
        bearing left-positive from the centroid of traversable pixels in the
        near-field band. None when nothing is traversable.

        Monocular depth cannot see a grass wall — it reported 2.4 m with blades
        against the lens — but this can see the track has run out.
        """
        h, w = good.shape
        band = good[int(h * (1.0 - lower_frac)):, :]
        frac_all = float(good.mean())
        frac_band = float(band.mean()) if band.size else 0.0
        if not band.any():
            return frac_all, frac_band, None
        cols = band.sum(axis=0).astype(np.float32)
        if cols.sum() <= 0:
            return frac_all, frac_band, None
        u = np.arange(w, dtype=np.float32)
        cx = w / 2.0
        fx = w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
        centroid = float((cols * u).sum() / cols.sum())
        # Left-positive to match the goal-bearing convention.
        bearing = -float(np.arctan((centroid - cx) / fx))
        return frac_all, frac_band, bearing

    def corridor(
        self,
        good: np.ndarray,
        fov_deg: float = 90.0,
        lower_frac: float = 0.45,
        col_thresh: float = 0.5,
    ) -> tuple[Optional[float], float]:
        """
        Widest continuous drivable corridor in the near field, as
        (bearing_rad_left_positive, width_as_fraction_of_frame).

        A centroid is the wrong statistic here: grass both sides with a gap in
        the middle averages to the middle, but so does grass in the middle with
        gaps both sides. The widest contiguous run is what the rover fits
        through, and its width measures confidence directly.
        """
        h, w = good.shape
        band = good[int(h * (1.0 - lower_frac)):, :]
        if band.size == 0:
            return None, 0.0
        colfrac = band.mean(axis=0)
        openc = colfrac >= col_thresh
        if not openc.any():
            return None, 0.0
        # Longest contiguous run of open columns.
        best_len = best_start = 0
        cur_len = cur_start = 0
        for i, v in enumerate(openc):
            if v:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                if cur_len > best_len:
                    best_len, best_start = cur_len, cur_start
            else:
                cur_len = 0
        if best_len <= 0:
            return None, 0.0
        center = best_start + best_len / 2.0
        cx = w / 2.0
        fx = w / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
        bearing = -float(np.arctan((center - cx) / fx))
        return bearing, float(best_len) / float(w)

    def _filter_hard(self, hard: np.ndarray) -> np.ndarray:
        """Confidence gate + small-blob removal on mission-stopping classes."""
        if not hard.any():
            return hard
        conf = getattr(self, "last_conf", None)
        if conf is not None and conf.shape == hard.shape:
            hard = hard & (conf >= self.conf)
            if not hard.any():
                return hard
        try:
            import cv2

            n, lab = cv2.connectedComponents(hard.astype(np.uint8))
            min_px = max(8, int(self.min_blob_frac * hard.size))
            out = np.zeros_like(hard)
            for i in range(1, n):
                blob = lab == i
                if blob.sum() >= min_px:
                    out |= blob
            return out
        except Exception:
            return hard
