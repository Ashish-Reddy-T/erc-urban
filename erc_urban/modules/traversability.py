"""
The traversability veto (semantic half): a HARD veto applied AFTER selection.
If the chosen candidate crosses untraversable surface, drop it and take the
next-best that does not.

Project each candidate's waypoints to image space through `calib`, sample the
mask at those pixels, veto if too much of the path lands on bad surface. Image
space rather than BEV because we already HAVE the candidates — we only need to
test paths, not plan in BEV.

The mask source is pluggable: SegFormerSource works today off the model
semantics.py already loads; SAMTPSource needs a manual weights download, so it
stays behind an explicit constructor argument rather than silently failing to
None and vetoing nothing.

Complementary to DepthCostmap.veto, which catches solid things whatever the
segmenter calls them. A candidate must pass both, and semantics may only ADD
hazards, never clear a geometric one.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..calib import CameraCalibration
from ..interfaces import Candidate, Observation


class TraversabilitySource(Protocol):
    """Anything that can label pixels drivable."""

    def traversable(
        self, rgb: np.ndarray, out_hw: tuple[int, int]
    ) -> Optional[np.ndarray]:
        """(H,W) bool at out_hw, True = drivable. None if unavailable."""
        ...


class SegFormerSource:
    """
    Adapter over the SegFormer-B2 ADE20K model semantics.py already loads.

    The default because it needs no extra download. Reuses
    `SemanticTraversability.categorize` so it inherits the same
    traversable-by-exclusion policy rather than duplicating a divergent list.
    """

    def __init__(self, semantics=None, device: str = "auto", size: int = 512):
        if semantics is None:
            from .semantics import SemanticTraversability

            semantics = SemanticTraversability(device=device, size=size)
        self.sem = semantics

    @property
    def ready(self) -> bool:
        return bool(getattr(self.sem, "ready", False))

    def traversable(self, rgb, out_hw):
        if not self.ready:
            return None
        labels = self.sem.labels(rgb, out_hw)
        if labels is None:
            return None
        hard, soft, good = self.sem.categorize(labels)
        # `soft` (grass, plants) is drivable-but-off-track: costed by
        # costmap.offtrack_cost, NOT vetoed here. A rover that refuses to touch
        # a verge cannot get around a parked car — and on a grass site,
        # excluding soft reported all_untraversable 12/12 on ground with zero
        # hazard and a fully drivable near field.
        return np.asarray((good | soft) & ~hard, dtype=bool)


class SAMTPSource:
    """
    SAM-TP: SAM2 fine-tuned on annotated road scenes for pixel-wise
    traversability. Trained for exactly this label rather than repurposed from a
    general scene parser, but not the default because the checkpoint is a manual
    download.

    UNWIRED: point `predictor` at a loaded model exposing
    `predict(rgb, K, pose) -> (H, W) logits`.
    """

    def __init__(self, predictor=None, calib: Optional[CameraCalibration] = None,
                 threshold: float = 0.0):
        self.predictor = predictor
        self.calib = calib
        self.threshold = threshold

    @property
    def ready(self) -> bool:
        return self.predictor is not None

    def traversable(self, rgb, out_hw):
        if self.predictor is None:
            return None
        cal = (self.calib or CameraCalibration.from_fov(rgb.shape[1], rgb.shape[0]))
        logits = self.predictor.predict(rgb, cal.K(), cal.T_robot_to_cam())
        m = np.asarray(logits) > self.threshold
        if m.shape != tuple(out_hw):
            from PIL import Image

            m = np.asarray(
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (out_hw[1], out_hw[0]), Image.NEAREST
                )
            ) > 127
        return m


class TraversabilityVeto:
    """Semantic veto. `mask` returns True per candidate that is safe to keep."""

    def __init__(
        self,
        source: Optional[TraversabilitySource] = None,
        calib: Optional[CameraCalibration] = None,
        max_bad_frac: float = 0.34,
        min_samples: int = 3,
        mask_hw: tuple[int, int] = (120, 160),
        samples_per_segment: int = 4,
        min_traversable_frac: float = 0.02,
        max_range_m: Optional[float] = 4.0,
    ):
        """
        max_bad_frac: veto once this fraction of sampled points land on bad
            surface. Not zero-tolerance — monocular segmentation is noisy at
            boundaries and one stray pixel on a curb edge must not kill a good
            path, which fails closed and strands the rover.
        min_samples: below this many in-frame points there is no evidence, so
            the candidate passes. The geometric costmap is still watching.
        samples_per_segment: waypoints are metres apart and can straddle a whole
            curb, so interpolate along each segment.
        min_traversable_frac: below this the segmenter is not discriminating and
            its output is ignored. An out-of-distribution frame produces a
            confident nonsense mask, not a cautious one (SegFormer labels one
            fixture 100% "sky", another 84% "wall"), and vetoing everything on
            that hands run.py a spurious "no way forward".
        max_range_m: only the part of each path within this range is tested.
            Under a 148 mm camera, ground past ~4 m projects within ~2 rows of
            the horizon in a 120-row mask, onto whatever stands at the far edge
            of the visible ground, so a 10 m anchor was vetoed by a wall 9 m
            away and "all untraversable" fired escapes long before anything
            was near. The veto re-runs every tick; the far part gets tested
            once the rover is there. None = the whole path.
        """
        self.max_range_m = max_range_m
        self.source = source if source is not None else SegFormerSource()
        self.calib = calib
        self.max_bad_frac = max_bad_frac
        self.min_samples = min_samples
        self.mask_hw = mask_hw
        self.samples_per_segment = max(1, samples_per_segment)
        self.min_traversable_frac = min_traversable_frac
        self.last_mask: Optional[np.ndarray] = None
        self.last_reason: str = ""

    @property
    def ready(self) -> bool:
        return bool(getattr(self.source, "ready", False))

    def _calib_for(self, rgb: np.ndarray) -> CameraCalibration:
        h, w = rgb.shape[:2]
        cal = self.calib or CameraCalibration.from_fov(w, h)
        # The mask is computed at mask_hw, so project into THAT pixel grid.
        return cal.scaled_to(self.mask_hw[1], self.mask_hw[0])

    @staticmethod
    def _densify(wp: np.ndarray, n: int) -> np.ndarray:
        """Interpolate along each segment so a path cannot step over a curb."""
        if len(wp) < 2 or n <= 1:
            return wp
        out = []
        for a, b in zip(wp[:-1], wp[1:]):
            for t in np.linspace(0.0, 1.0, n, endpoint=False):
                out.append(a + t * (b - a))
        out.append(wp[-1])
        return np.asarray(out)

    def mask(self, obs: Observation, cands: list[Candidate]) -> np.ndarray:
        keep = np.ones(len(cands), dtype=bool)
        if not cands or obs.front is None:
            return keep

        trav = self.source.traversable(obs.front, self.mask_hw)
        self.last_mask = trav
        if trav is None:
            self.last_reason = "source_unavailable"
            return keep  # no evidence: the geometric veto still applies

        frac = float(trav.mean())
        if frac < self.min_traversable_frac:
            # Not "the world is impassable" — almost certainly an
            # out-of-distribution frame the segmenter has confidently
            # mislabelled. Vetoing everything here would be acting on that
            # nonsense. Defer to geometry and say so.
            self.last_reason = f"source_undiscriminating(trav={frac:.3f})"
            return keep

        cal = self._calib_for(obs.front)
        mh, mw = trav.shape
        n_vetoed = 0
        for i, c in enumerate(cands):
            pts = self._densify(np.asarray(c.waypoints, dtype=np.float64),
                                self.samples_per_segment)
            if self.max_range_m is not None:
                pts = pts[np.linalg.norm(pts, axis=1) <= self.max_range_m]
            uv, ok = cal.project_ground(pts)
            u = np.rint(uv[:, 0]).astype(int)
            v = np.rint(uv[:, 1]).astype(int)
            inb = ok & (u >= 0) & (u < mw) & (v >= 0) & (v < mh)
            if int(inb.sum()) < self.min_samples:
                continue
            bad = float((~trav[v[inb], u[inb]]).mean())
            if bad > self.max_bad_frac:
                keep[i] = False
                n_vetoed += 1

        self.last_reason = f"vetoed {n_vetoed}/{len(cands)}"
        return keep

    def apply(
        self, obs: Observation, cands: list[Candidate], chosen: int
    ) -> tuple[int, str]:
        """
        The veto proper: applied AFTER selection, returns (index, note).

        If the chosen candidate is untraversable, fall back to the next-best
        traversable one by score. If nothing is traversable, return the original
        choice and say so — run.py owns the "no way forward" recovery, and
        silently substituting an unsafe path would be worse than reporting it.
        """
        if not cands:
            return chosen, "no_candidates"
        keep = self.mask(obs, cands)
        if keep[chosen]:
            return chosen, "ok"
        order = sorted(range(len(cands)), key=lambda i: cands[i].score, reverse=True)
        for i in order:
            if keep[i]:
                return i, f"vetoed {chosen} -> {i} ({self.last_reason})"
        return chosen, f"all_untraversable ({self.last_reason})"
