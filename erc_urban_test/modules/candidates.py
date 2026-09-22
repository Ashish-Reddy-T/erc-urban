"""
The K-candidate generator. Selection, latency fusion and the vetoes all operate
on SETS; with a single waypoint they have nothing to choose between.

LogoNav cannot supply that set. It is a deterministic regressor with no sampling
head, and its one varying input barely moves it: perturbing the conditioning
goal laterally gave 0.02-0.21 m of endpoint spread across five real fixtures
against 0.41-3.89 m for plain geometric arcs. That is by design — MBRA damps the
policy against a noisy GPS goal.

So split the two jobs that "emit K scored candidates" conflates:

    COVERAGE  from a geometric anchor fan (BaselinePlanner arcs)
    JUDGEMENT from the learned policy, used as a scorer

`anchor_source` takes anything with BaselinePlanner's `.propose(obs, k)`
signature, so swapping in a real sampling policy is a constructor argument.
LogoNav's own trajectory is always in the set; the goal-perturbed variants ride
along because they cost ~12 ms for 17 extra hypotheses.
"""
from __future__ import annotations

import os
from typing import Optional, Protocol

import numpy as np

from ..interfaces import Candidate, Observation
from .baseline import BaselinePlanner
from .logonav import LogoNavPlanner
from .scoring import CandidateScorer, ScoreWeights


class AnchorSource(Protocol):
    """Anything that emits a spread of geometric candidates."""
    def propose(self, obs: Observation, k: int) -> list[Candidate]: ...


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _end_bearing(xy: np.ndarray) -> float:
    return float(np.arctan2(xy[-1, 1], max(xy[-1, 0], 1e-3)))


def _index_l2(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


def _prefix_to_range(xy: np.ndarray, r: float) -> np.ndarray:
    d = np.linalg.norm(np.asarray(xy, float), axis=1)
    if len(d) == 0 or d[-1] <= r:
        return xy
    i = int(np.searchsorted(d, r))
    return xy[: max(i, 2)]


def _mean_l2(a: np.ndarray, b: np.ndarray) -> float:
    # Compare over the overlapping forward distance. Anchors are 10 m so the
    # VLM can see past bumper roots; LogoNav is 0.7-2.7 m. Index-wise L2
    # would make every long arc equally unlike the prior.
    r = min(float(np.linalg.norm(a[-1])), float(np.linalg.norm(b[-1])))
    a = _prefix_to_range(a, r)
    b = _prefix_to_range(b, r)
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


class MultiHypothesisPlanner:
    """An anchor fan scored by LogoNav -> K scored Candidates."""

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
        n_anchors: width of the geometric fan — where diversity comes from, so
            it dominates the set.
        n_perturbed: goal-perturbed LogoNav variants. Small, because they add
            little spread; kept because they are nearly free.
        prior_weight / prior_scale_m: how hard LogoNav's own path pulls the
            ranking. 0 makes this purely geometric; very large makes the anchors
            decorative. 0.6 over 1 m keeps both terms comparable at a ~2 m
            horizon.
        dedup_m: paths closer than this in mean L2 are the same path. Without
            it a saturated policy's duplicates win a softmax vote by
            multiplicity.
        """
        self.base = base if base is not None else LogoNavPlanner()
        self.anchors = anchor_source or BaselinePlanner()
        # 0 means genuinely no anchors. Clamping to 1 would hand back one
        # extreme arc from the end of the fan, i.e. a hard turn dressed up as
        # "no geometric candidates".
        self.n_anchors = max(0, n_anchors)
        self.n_perturbed = max(0, n_perturbed)
        self.lateral_span_m = lateral_span_m
        self.prior_weight = prior_weight
        self.prior_scale_m = max(prior_scale_m, 1e-6)
        self.dedup_m = dedup_m
        # prior_weight must reach the scorer or the constructor argument is
        # silently dead.
        self.scorer = CandidateScorer(
            weights=ScoreWeights(prior=prior_weight), prior_scale_m=prior_scale_m
        )

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

        Built in +/- pairs so the set stays mirror-symmetric — truncating a flat
        list would drop one arm of the last pair and bias every tick toward that
        side. Index 0 is always 0.0, so LogoNav's own path is always present.
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

        # Match the fan to LogoNav's horizon. The policy shortens its path when
        # blocked (2.69 m on open dirt, 0.69 m facing another rover), and a 3 m
        # arc against a 0.7 m prediction makes every anchor look equally unlike
        # the prior.
        anchors = self._anchor_paths(obs, prior)

        paths = self._dedup(list(model_paths) + anchors)
        if not paths:
            return self.anchors.propose(obs, k)
        cands = [Candidate(waypoints=xy, score=0.0, stamp=obs.stamp) for xy in paths]
        # Ranked on goal + prior only. Perception is per-CANDIDATE and the
        # candidates do not exist yet, so run.py re-ranks via rescore() once it
        # has projected the traversability mask onto them.
        cands = self.rescore(cands, obs, prior=prior)

        # Top-K by score, so the selector sees a short ranked set.
        return cands[: max(k, 1)]

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

        Returns a NEW sorted list: candidates are immutable elsewhere, and
        mutating them would desynchronise anything holding a reference (the
        commit window, the VLM's snapshot).
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
        # Once per tick: the context queue is temporal state, and pushing the
        # same frame twice corrupts the model's history.
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
        # LogoNav's path is 0.7-2.7 m (training horizon). The fan must be
        # longer so the veto can see past the roots under the bumper — every
        # 2 m arc died in the near field. 10 m matches the explore radius.
        want = float(os.environ.get("ANCHOR_HORIZON_M", "10.0"))
        model_h = float(np.linalg.norm(prior[-1])) if prior is not None else 0.0
        horizon = max(want, model_h, 0.5)
        n_pts = max(12, len(prior) if prior is not None else 8)
        source = self.anchors
        if isinstance(source, BaselinePlanner):
            # Keep endpoints inside a ~70 deg cone: max_curv=0.9 at 10 m wraps
            # the arc onto a 1.1 m circle and every "long" path ends 1 m away.
            max_curv = min(float(source.max_curv), 1.2 / horizon)
            source = BaselinePlanner(
                horizon_m=horizon,
                n_pts=n_pts,
                max_curv=max_curv,
            )
        return [c.waypoints for c in source.propose(obs, self.n_anchors)]

    def _dedup(self, paths: list[np.ndarray]) -> list[np.ndarray]:
        kept: list[np.ndarray] = []
        for xy in paths:
            # Full-path L2: prefix L2 would collapse a 10 m fan, since every
            # arc shares the first metre.
            if not any(_index_l2(xy, k) < self.dedup_m for k in kept):
                kept.append(xy)
        return kept
