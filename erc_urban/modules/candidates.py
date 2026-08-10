"""
Module 1 — the K-candidate generator (the integration keystone).

plan.md calls a scored candidate SET "non-negotiable": Modules 4 (VLM
selection), 5 (latency fusion) and 6 (traversability veto) all operate on sets,
and with a single waypoint they have nothing to select over, fuse across, or
retreat to. plan.md also names getting calibrated multi-hypothesis output as
risk #1. That risk is real. This module is the answer, and the answer is not
the one plan.md assumed.

WHY NOT SAMPLE THE POLICY
-------------------------
plan.md says to take LogoNav and "run it in NoMaD-style multi-hypothesis
sampling". You cannot. Two measured facts:

  1. LogoNav has no sampling dimension. `IL_gps.forward(obs_img, goal_pose)`
     (mbra_repo il.py:213) is a deterministic regressor — no noise input, no
     sampling head. The `num_diffusion_iters: 10` in LogoNav.yaml is inherited
     NoMaD template config that IL_gps never reads, so the config *looks* like
     it supports sampling when it cannot. Verified byte-identical to upstream.
     Its successor OmniVLA is deterministic too: an L1 regression action head
     (`prismatic/models/action_heads.py:108`), diffusion scaffolding unused.

  2. Its one varying input barely moves it. Perturbing the conditioning goal
     laterally — the obvious substitute for sampling — was measured across five
     real fixtures at 0.02-0.21 m of endpoint spread, against 0.41-3.89 m for
     plain geometric arcs over the same horizon: 9x to 254x less. Even rotating
     the TRUE goal a full +/-90 deg swings the output bearing only ~10 deg.
     That is not a bug to tune away; LogoNav is deliberately damped against its
     GPS goal, which is the entire point of the MBRA paper (robustness to noisy
     GPS), and its horizon is only 0.7-2.7 m.

WHAT WE DO INSTEAD
------------------
Separate the two jobs that "emit K scored candidates" conflates:

    COVERAGE comes from a geometric anchor fan (BaselinePlanner arcs).
    JUDGEMENT comes from the learned policy, used as a scorer.

This is not a workaround, it is what the papers we are integrating actually do.
Slow-Brain's base planner is S2E, whose candidates come from "embodiment-
agnostic anchors" — a fixed geometric basis ranked by a learned model. And it
is the OmniVLA standard for a GPS/pose goal: generate a diverse set, then keep
the trajectory whose endpoint lands nearest the target pose. We follow that
rule exactly, with arcs as the basis instead of NoMaD samples.

The anchor basis is deliberately swappable: `anchor_source` takes anything with
BaselinePlanner's `.propose(obs, k)` signature, so dropping in NoMaD-exploration
samples (a real diffusion policy, the one genuinely multi-hypothesis option in
this lineage) is a constructor argument, not a rewrite.

LogoNav's own trajectory is always in the set, and the goal-perturbed variants
ride along because after the encode-once fix they cost ~12 ms for 17 extra
hypotheses — cheap enough to keep for the little coverage they do add.
"""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..interfaces import Candidate, Observation
from .baseline import BaselinePlanner
from .logonav import LogoNavPlanner
from .scoring import CandidateScorer, ScoreWeights


class AnchorSource(Protocol):
    """Anything that emits a spread of geometric candidates (BaselinePlanner,
    or a NoMaD-exploration sampler once weights are available)."""
    def propose(self, obs: Observation, k: int) -> list[Candidate]: ...


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _end_bearing(xy: np.ndarray) -> float:
    return float(np.arctan2(xy[-1, 1], max(xy[-1, 0], 1e-3)))


def _mean_l2(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


class MultiHypothesisPlanner:
    """
    M1: an anchor fan scored by LogoNav → K scored Candidates.

    Honors the `Planner` protocol, so it drops into run.py's loop as a one-line
    swap for LogoNavPlanner.
    """

    def __init__(
        self,
        base: Optional[LogoNavPlanner] = None,
        anchor_source: Optional[AnchorSource] = None,
        n_anchors: int = 12,
        n_perturbed: int = 6,
        lateral_span_m: float = 3.0,
        prior_weight: float = 0.6,
        prior_scale_m: float = 1.0,
        dedup_m: float = 0.15,
    ):
        """
        n_anchors: width of the geometric fan. This is where diversity actually
            comes from, so it dominates the set.
        n_perturbed: goal-perturbed LogoNav variants. Small, because they add
            little spread — kept only because they are nearly free.
        prior_weight / prior_scale_m: how hard LogoNav's own path pulls the
            ranking. 0 makes this a pure goal-seeking geometric planner (the
            learned priors stop mattering); very large makes the anchors
            decorative because only the arc nearest LogoNav can ever win. 0.6
            over a 1 m scale keeps both terms comparable at our ~2 m horizon.
        dedup_m: paths closer than this in mean L2 are the same path. Without
            it a saturated policy's duplicate outputs would win a softmax vote
            by sheer multiplicity in M4/M5.
        """
        self.base = base if base is not None else LogoNavPlanner()
        self.anchors = anchor_source or BaselinePlanner()
        # 0 means genuinely no anchors. Clamping to 1 would hand back a single
        # extreme arc from the end of the fan -- a hard turn presented as if it
        # were "no geometric candidates", which is worse than none.
        self.n_anchors = max(0, n_anchors)
        self.n_perturbed = max(0, n_perturbed)
        self.lateral_span_m = lateral_span_m
        self.prior_weight = prior_weight
        self.prior_scale_m = max(prior_scale_m, 1e-6)
        self.dedup_m = dedup_m
        # prior_weight must reach the scorer, or the constructor argument is
        # silently dead and callers think they are tuning something they are not.
        self.scorer = CandidateScorer(
            weights=ScoreWeights(prior=prior_weight), prior_scale_m=prior_scale_m
        )
        self._trav_fracs = None
        self._clearances = None

    # LogoNavPlanner's goal/reset API, forwarded so this is a drop-in swap.
    def set_goal(self, lat: float, lon: float) -> None:
        self.base.set_goal(lat, lon)

    def reset(self) -> None:
        self.base.reset()

    @property
    def ready(self) -> bool:
        return self.base.ready

    def offsets(self) -> np.ndarray:
        """
        Lateral aim-point offsets for the perturbed variants, left-positive.

        Built in +/- pairs and sized to a whole number of pairs so the set stays
        mirror-symmetric. Truncating a flat list to n would drop one arm of the
        last pair and bias every tick toward that side — a steering bias
        disguised as a sampling detail. Index 0 is always 0.0, so LogoNav's
        unperturbed path is always present.
        """
        if self.n_perturbed <= 1:
            return np.zeros(1)
        n_pairs = (self.n_perturbed - 1) // 2
        if n_pairs < 1:
            return np.zeros(1)
        half = np.linspace(0.0, self.lateral_span_m, n_pairs + 1)[1:]
        out = [0.0]
        for h in half:
            out.extend((+float(h), -float(h)))
        return np.asarray(out, dtype=np.float64)

    def propose(self, obs: Observation, k: int) -> list[Candidate]:
        model_paths = self._model_paths(obs)
        # model_paths[0] is the unperturbed LogoNav trajectory: our prior.
        prior = model_paths[0] if len(model_paths) else None

        # Match the anchor fan to LogoNav's own horizon. The policy shortens
        # its path when the way ahead is blocked (measured 2.69 m on open dirt,
        # 0.69 m facing another rover), and comparing a 3 m arc against a 0.7 m
        # prediction would make every anchor look equally unlike the prior.
        anchors = self._anchor_paths(obs, prior)

        paths = self._dedup(list(model_paths) + anchors)
        if not paths:
            return self.anchors.propose(obs, k)
        cands = [Candidate(waypoints=xy, score=0.0, stamp=obs.stamp) for xy in paths]
        # Rank with everything available. trav/clearance are per-candidate and
        # therefore only present when a caller supplied them via
        # set_perception() from the PREVIOUS tick's masks -- the candidates do
        # not exist before this call.
        cands = self.rescore(
            cands, obs, prior=prior,
            trav_fracs=self._trav_fracs
            if self._trav_fracs is not None and len(self._trav_fracs) == len(cands)
            else None,
            clearances=self._clearances
            if self._clearances is not None and len(self._clearances) == len(cands)
            else None,
        )

        # Top-K by score. Slow-Brain filters 64 raw down to the 18 it shows the
        # VLM; same idea, so the selector sees a short, ranked set.
        return cands[: max(k, 1)]

    def set_perception(self, trav_fracs=None, clearances=None) -> None:
        """
        Hand the scorer what perception knows about THIS tick's candidates,
        to be consumed by the next `propose`.

        Kept as a separate call rather than an `Observation` field because the
        values are per-CANDIDATE, and the candidates do not exist until the
        planner has run. run.py therefore scores once, then re-ranks with
        perception attached. See modules/scoring.py for why this matters: with
        traversability and clearance used only as a post-hoc veto, ranking was
        measured worse than random.
        """
        self._trav_fracs = trav_fracs
        self._clearances = clearances

    def _score(self, xy: np.ndarray, obs: Observation, prior) -> float:
        """Single-candidate fallback; the set is normally scored by `rescore`."""
        goal_term = -abs(_wrap(_end_bearing(xy) - obs.goal_bearing))
        if prior is None:
            return goal_term
        return goal_term - self.prior_weight * (_mean_l2(xy, prior) / self.prior_scale_m)

    def rescore(self, cands: list[Candidate], obs: Observation,
                prior=None, trav_fracs=None, clearances=None) -> list[Candidate]:
        """
        Re-rank an existing candidate set using perception.

        Returns a NEW sorted list; candidates are immutable elsewhere in the
        stack and quietly mutating them would desynchronise anything holding a
        reference (the commit window, the VLM's snapshot).
        """
        if not cands:
            return cands
        scores = self.scorer.score(
            cands, obs.goal_bearing, prior=prior,
            trav_fracs=trav_fracs, clearances=clearances,
        )
        out = [
            Candidate(waypoints=c.waypoints, score=float(s), stamp=c.stamp)
            for c, s in zip(cands, scores)
        ]
        out.sort(key=lambda c: c.score, reverse=True)
        return out

    def _model_paths(self, obs: Observation) -> np.ndarray | list:
        # Exactly once per tick: the context queue is temporal state, and
        # pushing the same frame twice would corrupt the model's history.
        if not self.base.update_context(obs):
            return []
        poses = []
        for d in self.offsets():
            gp = self.base.goal_pose(obs, lateral_offset_m=float(d))
            if gp is None:
                return []
            poses.append(gp)
        if not poses:
            return []
        out = self.base.forward_batch(np.stack(poses))
        return out if out is not None else []

    def _anchor_paths(self, obs: Observation, prior) -> list[np.ndarray]:
        if self.n_anchors <= 0:
            return []
        horizon = float(np.linalg.norm(prior[-1])) if prior is not None else 0.0
        source = self.anchors
        if prior is not None and isinstance(source, BaselinePlanner):
            source = BaselinePlanner(
                horizon_m=max(horizon, 0.5),
                n_pts=len(prior),
                max_curv=source.max_curv,
            )
        return [c.waypoints for c in source.propose(obs, self.n_anchors)]

    def _dedup(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        kept: list[np.ndarray] = []
        for xy in paths:
            if not any(_mean_l2(xy, k) < self.dedup_m for k in kept):
                kept.append(xy)
        return kept
