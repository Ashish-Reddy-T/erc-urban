"""
Module 6 — the traversability veto (semantic half).

plan.md: a HARD veto applied AFTER selection. If the chosen candidate crosses
untraversable surface, drop it and take the next-best that doesn't. GeNIE won
ERC ICRA'25 with exactly this (79% of max, zero interventions), and Sorokin's
sidewalk study is why: curb cuts, driveways and grass-vs-pavement confusion are
the dominant sidewalk-violation failure mode — all of them *semantic*
distinctions that geometry cannot make. A grass verge and the path beside it
are the same flat plane.

WHAT THIS REPLACES
------------------
`upgrades.TraversabilityGate.mask` returns `np.ones(len(cands), bool)` when its
predictor is None, and its predictor is always None. It is a no-op that has
never vetoed anything at any stage. Worse, run.py:898 only reaches the gate at
all when `path is not None`, i.e. only under USE_DIRT_GATE=1 — so the SAM2 veto
was doubly unreachable.

HOW IT WORKS
------------
Project each candidate's waypoints to image space through `calib`, sample the
traversability mask at those pixels, veto the candidate if too much of its path
lands on untraversable surface. Image-space sampling rather than GeNIE's BEV
rasterisation because we already HAVE the candidates — we do not need to plan in
BEV, only to test paths — and it avoids the resolution loss of a second
rasterisation step.

The mask source is pluggable on purpose. `SegFormerSource` works today off the
SegFormer-B2 model `semantics.py` already loads. `SAMTPSource` is the GeNIE
model proper and needs weights that only come from a manual Google Drive
download, so it stays behind an explicit constructor argument rather than a
silent import that fails to None and vetoes nothing — which is the exact bug
this module exists to fix.

RELATION TO THE GEOMETRIC VETO
------------------------------
`DepthCostmap.veto` stays: geometry catches solid things regardless of what the
segmenter calls them. This adds the surface judgement geometry cannot make.
Semantics may only ADD hazards, never clear a geometric one — the invariant
already stated at costmap.py:223. Both are vetoes; a candidate must pass both.
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
    Adapter over the SegFormer-B2 ADE20K model `semantics.py` already loads.

    Available with no extra download, which makes it the default: a working
    veto today beats a better veto that needs weights nobody has fetched.
    Reuses `SemanticTraversability.categorize`, so it inherits that module's
    traversable-by-exclusion policy (semantics.py:174) — vegetation and hazards
    are the small well-defined sets, everything else at ground level is
    drivable — rather than duplicating a second, divergent class list.
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
        # `soft` (grass, plants) is drivable-but-off-track. It is costed by
        # costmap.offtrack_cost, NOT vetoed here: a rover that refuses to touch
        # a verge cannot get around a parked car.
        return np.asarray(good & ~hard, dtype=bool)


class SAMTPSource:
    """
    GeNIE's SAM-TP (jiaming-ai/GENIE-SAMTP): SAM2 fine-tuned on 15,347
    annotated road scenes for pixel-wise traversability.

    Better than ADE20K semantics for this job — it was trained for exactly this
    label, on outdoor road scenes, rather than repurposed from a general scene
    parser. Not the default only because the checkpoint is a manual Google Drive
    download.

    >>> WIRE: point `predictor` at a loaded SAM-TP model exposing
    `predict(rgb, K, pose) -> (H, W) logits`. Its documented inputs are an
    (H,W,3) uint8 array, a 3x3 intrinsics matrix and a 4x4 camera pose — which
    is why `calib.py` exists and produces both.
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
    """
    M6 semantic veto. Honors the `SafetyGate` protocol.

    Returns a boolean mask over candidates, True = safe to keep.
    """

    def __init__(
        self,
        source: Optional[TraversabilitySource] = None,
        calib: Optional[CameraCalibration] = None,
        max_bad_frac: float = 0.34,
        min_samples: int = 3,
        mask_hw: tuple[int, int] = (120, 160),
        samples_per_segment: int = 4,
        min_traversable_frac: float = 0.02,
    ):
        """
        max_bad_frac: veto once this fraction of a path's sampled points land on
            untraversable surface. Not zero-tolerance: monocular segmentation is
            noisy at boundaries, and a single stray pixel on a curb edge must
            not kill an otherwise good path — that fails closed and strands the
            rover, the failure mode `semantics.py` already warns about.
        min_samples: below this many in-frame points we have no real evidence,
            so the candidate passes. Absence of evidence is not a veto; the
            geometric costmap is still watching.
        samples_per_segment: waypoints are metres apart, so consecutive points
            can straddle an entire curb. Interpolate along each segment.
        min_traversable_frac: below this the segmenter is not discriminating
            and its output is ignored. Measured need: SegFormer labels
            `synth_dirt_path` 100% "sky" and `mauritius_rover_block` 84% "wall"
            — an out-of-distribution frame does not produce a cautious mask, it
            produces a confident nonsense one, and this veto would then kill
            every candidate and hand run.py a spurious "no way forward". A
            genuinely impassable scene is not this module's call: run.py's
            TRACK_MIN check already owns "no path visible", and the geometric
            costmap is watching independently.
        """
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
        plan.md's actual M6 contract: a veto applied AFTER selection.

        Returns (index, note). If the selected candidate is untraversable, fall
        back to the next-best traversable one by score. If nothing is
        traversable we return the original choice and say so — the caller
        (run.py) already has reverse/scan recovery for "no way forward", and
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
