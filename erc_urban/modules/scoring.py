"""
Candidate scoring — how the K hypotheses are ranked.

WHY THIS IS ITS OWN MODULE
--------------------------
Ranking was previously two lines inside MultiHypothesisPlanner: endpoint-bearing
agreement with the goal, plus similarity to LogoNav's prediction. Benchmarked
against human-teleoperated FrodoBots rides, that ranking came out WORSE THAN
RANDOM -- 1.029 m median ADE against a uniformly random candidate's 0.757 m,
reproduced on a second ride. A ranking worse than chance is not a small tuning
problem; it actively misleads everything downstream, and it dragged the VLM
selector's contribution from -29% to -3% once Probability Fusion mixed it back in.

The cause was not a bad weight. It was that the score ignored almost everything
the stack already knows:

  * only the ENDPOINT bearing entered, so two paths reaching the same point by
    completely different routes scored identically -- including one crossing a
    kerb and one staying on pavement.
  * traversability was consulted only as a binary veto AFTER ranking, so among
    surviving candidates the one hugging a grass edge ranked as well as the one
    down the middle of the path.
  * obstacle clearance never entered the ranking at all, for the same reason.

Perception was computing rich, per-candidate information and the ranking threw
all of it away. This module puts it back, as an explicit weighted sum with each
term normalised to a comparable scale, so the weights mean something and can be
tuned against the benchmark rather than by feel.

Every term is optional. With only a goal bearing it reduces to the old
behaviour, which keeps it usable when semantics or depth are unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _mean_l2(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


@dataclass
class ScoreTerms:
    """Per-candidate breakdown, kept so a bad ranking can be debugged."""

    goal: float = 0.0
    prior: float = 0.0
    traversable: float = 0.0
    clearance: float = 0.0
    smooth: float = 0.0
    total: float = 0.0

    def describe(self) -> str:
        return (f"goal={self.goal:+.2f} prior={self.prior:+.2f} "
                f"trav={self.traversable:+.2f} clear={self.clearance:+.2f} "
                f"smooth={self.smooth:+.2f} -> {self.total:+.2f}")


@dataclass
class ScoreWeights:
    """
    All terms are normalised to roughly [-1, 0], so weights are directly
    comparable and a weight of 1.0 means "as important as goal agreement".
    """

    # Weights set from measurement, not taste. `eval.score_diag` reports the
    # rank correlation of each term with "how close this candidate is to what a
    # human actually drove", over teleoperated FrodoBots rides:
    #
    #     traversable  rho +0.23, helps on 68% of ticks   -> trust it
    #     goal         rho -0.36, helps on  2% of ticks   -> see below
    #     smooth       rho -0.32, helps on  0% of ticks   -> was measuring the
    #                                                        wrong thing; fixed
    #
    # The goal term is NOT deleted despite scoring worst, and the reason
    # matters: that benchmark measures imitation of a human whose destination we
    # do not know, so a goal-seeking term is penalised for pursuing OUR goal
    # rather than theirs. It cannot validate goal-seeking, only safety. Its
    # weight is reduced so perception outranks it, not removed -- a rover with
    # no goal term wanders.
    goal: float = 0.5
    prior: float = 0.6
    traversable: float = 1.5      # highest, and the only term measured to help:
                                  # wrong-surface driving is also the dominant
                                  # sidewalk-violation failure mode (Sorokin)
    clearance: float = 0.8
    smooth: float = 0.15


class CandidateScorer:
    """
    Ranks candidates from everything currently known about the scene.

    Deliberately separate from the veto. The veto answers "is this candidate
    allowed"; this answers "which allowed candidate is best". Collapsing them
    loses ordering information among the survivors, which is exactly what made
    the old ranking sub-random.
    """

    def __init__(self, weights: Optional[ScoreWeights] = None,
                 prior_scale_m: float = 1.0, clearance_ref_m: float = 2.0):
        self.w = weights or ScoreWeights()
        self.prior_scale_m = max(prior_scale_m, 1e-6)
        self.clearance_ref_m = max(clearance_ref_m, 1e-6)
        self.last: list[ScoreTerms] = []

    # ---- individual terms (each returns <= 0, 0 = perfect) --------------
    def _goal(self, xy: np.ndarray, goal_bearing: float) -> float:
        """
        Agreement with the desired direction, integrated along the path rather
        than read off the endpoint. Two paths ending at the same place by
        different routes must not score the same -- that equivalence is what
        let a kerb-crossing candidate tie with a clean one.
        """
        d = np.linalg.norm(xy, axis=1)
        use = d > 0.05
        if not use.any():
            return -np.pi
        bearings = np.arctan2(xy[use, 1], np.maximum(xy[use, 0], 1e-3))
        err = np.abs((bearings - goal_bearing + np.pi) % (2 * np.pi) - np.pi)
        # Weight by distance: where the path ends up matters more than the
        # first few centimetres, which are near-identical for every candidate.
        wts = d[use] / max(float(d[use].sum()), 1e-9)
        return -float(np.sum(err * wts))

    def _prior(self, xy: np.ndarray, prior: np.ndarray) -> float:
        return -_mean_l2(xy, prior) / self.prior_scale_m

    def _traversable(self, xy: np.ndarray, trav_frac: float) -> float:
        """`trav_frac` in [0,1] is the fraction of the path on drivable surface."""
        return -(1.0 - float(np.clip(trav_frac, 0.0, 1.0)))

    def _clearance(self, clearance_m: float) -> float:
        """Nearest obstacle along this path, saturating at clearance_ref_m."""
        return -(1.0 - float(np.clip(clearance_m / self.clearance_ref_m, 0.0, 1.0)))

    def _smooth(self, xy: np.ndarray) -> float:
        """
        Penalise CHANGE of curvature, not curvature.

        The first version penalised the second difference, i.e. curvature
        itself, which made it a second vote for "go straight" on top of the
        goal term. Measured against human-driven rides it scored Spearman
        -0.316 and helped on 0% of ticks -- because humans turn, and a term
        that dislikes turning ranks every turn last.

        Constant-curvature arcs (which is what the anchor fan is) should all be
        equally smooth; only a path that changes its mind mid-way is jerky.
        """
        if len(xy) < 4:
            return 0.0
        d3 = np.diff(xy, n=3, axis=0)
        return -float(np.mean(np.linalg.norm(d3, axis=1))) * 10.0

    # ---- combined -------------------------------------------------------
    def score(
        self,
        candidates: list,
        goal_bearing: float,
        prior: Optional[np.ndarray] = None,
        trav_fracs: Optional[np.ndarray] = None,
        clearances: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns one score per candidate, higher is better.

        trav_fracs / clearances are per-candidate and optional; when absent
        those terms simply do not contribute, so the scorer degrades to the
        information actually available rather than inventing values.
        """
        self.last = []
        out = np.zeros(len(candidates), dtype=np.float64)
        for i, c in enumerate(candidates):
            xy = np.asarray(c.waypoints, dtype=np.float64)
            t = ScoreTerms()
            t.goal = self.w.goal * self._goal(xy, goal_bearing)
            if prior is not None and len(prior) >= 2:
                t.prior = self.w.prior * self._prior(xy, prior)
            if trav_fracs is not None:
                t.traversable = self.w.traversable * self._traversable(
                    xy, float(trav_fracs[i]))
            if clearances is not None:
                t.clearance = self.w.clearance * self._clearance(float(clearances[i]))
            t.smooth = self.w.smooth * self._smooth(xy)
            t.total = t.goal + t.prior + t.traversable + t.clearance + t.smooth
            self.last.append(t)
            out[i] = t.total
        return out


def path_traversable_fractions(
    candidates: list, trav_mask: np.ndarray, calib, samples_per_segment: int = 4
) -> np.ndarray:
    """
    Fraction of each candidate's path landing on drivable surface.

    Same projection and densification the veto uses, but returned as a
    CONTINUOUS value rather than a boolean. The veto needs a threshold; the
    ranking needs the gradient, so it can prefer the middle of the path over
    its edge instead of treating both as merely "allowed".
    """
    h, w = trav_mask.shape
    cal = calib.scaled_to(w, h)
    out = np.ones(len(candidates), dtype=np.float64)
    for i, c in enumerate(candidates):
        wp = np.asarray(c.waypoints, dtype=np.float64)
        if len(wp) >= 2 and samples_per_segment > 1:
            dense = []
            for a, b in zip(wp[:-1], wp[1:]):
                for tt in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
                    dense.append(a + tt * (b - a))
            dense.append(wp[-1])
            wp = np.asarray(dense)
        uv, ok = cal.project_ground(wp)
        u = np.rint(uv[:, 0]).astype(int)
        v = np.rint(uv[:, 1]).astype(int)
        inb = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if int(inb.sum()) >= 3:
            out[i] = float(trav_mask[v[inb], u[inb]].mean())
    return out


def path_clearances(candidates: list, costmap_result, cell_m: float,
                    half_width_m: float) -> np.ndarray:
    """
    Distance from each candidate to the nearest occupied cell.

    Again continuous where the veto is binary: among candidates that all avoid
    obstacles, the one that passes a metre clear should outrank the one that
    shaves past at the inflation radius.
    """
    occ = costmap_result.occupied
    n = len(candidates)
    far = float(occ.shape[1] * cell_m)
    out = np.full(n, far, dtype=np.float64)
    if not occ.any():
        return out
    oy, ox = np.nonzero(occ)
    obs = np.stack([ox * cell_m, oy * cell_m - half_width_m], axis=1)
    for i, c in enumerate(candidates):
        wp = np.asarray(c.waypoints, dtype=np.float64)
        if not len(wp):
            continue
        d = np.linalg.norm(obs[None, :, :] - wp[:, None, :], axis=2)
        out[i] = float(d.min())
    return out
