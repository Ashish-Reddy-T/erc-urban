"""
Gap selection and committed circumnavigation: "the goal is ahead, something is
in the way — do I go left or right, and how do I keep going that way long enough
to get around it?"

Neither half had an answer before. Directions could not be ranked over anything
but the current frame, and no decision survived a tick, so the rover re-decided
every tick and oscillated through eight escapes without moving.

VFH+ supplies the first half: reduce the surroundings to a polar histogram, find
the sectors wide enough to drive through, pick one by a cost function, and add
hysteresis so near-equal gaps do not swap the lead on sensor noise.

Bug2 supplies the second: when you leave the goal line to get around something,
follow the boundary on ONE side until the goal line is reacquired. That is what
turns "steer around a bit" into "go around", and why `Detour` latches a side
rather than a bearing.

Scoring includes a frontier term — how much of the ray is unobserved — or a
gap-follower prefers space it has already mapped and grinds along known dead
ends.

THE INPUT THAT MATTERS
----------------------
The histogram comes from `LocalMap.clearance_polar`, the ACCUMULATED map, not
the current frame. Against a single-frame costmap the commitment logic is
worthless: the moment the rover turns to go around something, the thing stops
existing and the goal term pulls it straight back.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .localmap import PolarScan


@dataclass
class Gap:
    """A contiguous run of bearings the rover could actually fit through."""

    bearing: float          # robot frame, left-positive, radians
    width_rad: float        # stored, not derived from lo/hi: a gap spanning
                            # +/-pi has hi < lo
    depth_m: float          # shallowest clearance across the gap
    frontier: float         # 0..1 unobserved fraction beyond it
    score: float = 0.0


@dataclass
class Option:
    """One concrete direction the rover could aim at, drawn from a gap."""

    bearing: float          # robot frame, left-positive
    gap: Gap
    depth_m: float          # clearance measured AT this bearing
    frontier: float
    score: float = 0.0


@dataclass
class Detour:
    """A latched decision to go around something on a chosen side."""

    side: int               # +1 = left, -1 = right
    bearing: float          # world frame — survives the rover turning
    started_travel_m: float
    budget_m: float
    reason: str = ""

    def describe(self) -> str:
        return (f"detour[{'L' if self.side > 0 else 'R'} "
                f"{np.degrees(self.bearing):+.0f}° budget={self.budget_m:.1f}m]")


@dataclass
class ExploreAim:
    """
    The explorer's steering opinion for this tick.

    Shaped like `route.RouteAim` on purpose, so it composes with the RouteAligner
    contract instead of adding a fourth place where GPS and vision get blended.
    """

    bearing: float
    confidence: float
    source: str
    speed_scale: float = 1.0
    gap: Optional[Gap] = None
    detour: Optional[Detour] = None
    depth_m: float = 0.0
    frontier: float = 0.0

    def describe(self) -> str:
        if self.gap is None:
            return f"gap[{self.source}]"
        return (f"gap({np.degrees(self.bearing):+.0f}° d={self.depth_m:.1f}m "
                f"w={np.degrees(self.gap.width_rad):.0f}° "
                f"frontier={self.frontier:.2f} {self.source})")


class GapExplorer:
    def __init__(
        self,
        min_gap_m: float = 1.2,
        gap_full_m: float = 3.5,
        rover_width_m: float = 0.7,
        wide_deg: float = 60.0,
        goal_weight: float = 1.0,
        depth_weight: float = 0.45,
        frontier_weight: float = 0.35,
        blocked_weight: float = 1.4,
        hysteresis: float = 0.5,
        commit_budget_m: float = 4.0,
        clear_ahead_m: float = 2.5,
    ):
        """
        blocked_weight is the largest weight on purpose: a direction that has
            already failed here should lose to a merely-worse-aligned one. That
            ordering IS the anti-livelock property, and it is a preference not a
            ban, so the rover still has a move when everything has failed once.
        hysteresis: bonus for the gap already committed to, or two near-equal
            gaps swap the lead on sensor noise.
        commit_budget_m: how far a detour runs before being re-decided. Bounded
            so a wrong commitment costs metres, not the mission.
        clear_ahead_m: clearance on the goal bearing at which the goal line
            counts as reacquired and the detour is dropped (the Bug2 rejoin).
        """
        self.min_gap_m = min_gap_m
        self.gap_full_m = gap_full_m
        self.rover_width_m = rover_width_m
        # Above this a gap is a "wide valley" and we aim at its border rather
        # than its centre. See `_options`.
        self.wide_rad = math.radians(wide_deg)
        self.goal_weight = goal_weight
        self.depth_weight = depth_weight
        self.frontier_weight = frontier_weight
        self.blocked_weight = blocked_weight
        self.hysteresis = hysteresis
        self.commit_budget_m = commit_budget_m
        self.clear_ahead_m = clear_ahead_m
        self.detour: Optional[Detour] = None
        self.last_gaps: list[Gap] = []

    # ---- gaps -------------------------------------------------------------
    def open_threshold(self, scan: PolarScan) -> float:
        """
        How much clearance a bearing needs to count as a way THROUGH rather than
        merely a bit of room.

        Adaptive by necessity: a fixed `min_gap_m` conflates the two, so with a
        wall 2 m ahead and a 1.2 m threshold the wall's own bearings pass, the
        "gap" found IS the wall, and the rover drives into it.

        So the bar scales with the best clearance on offer, bounded at both
        ends — open terrain demands a direction lead somewhere, a tight corridor
        falls back to `min_gap_m` rather than declaring the world impassable.
        """
        best = float(np.max(scan.clearance_m)) if len(scan.clearance_m) else 0.0
        return float(min(max(self.min_gap_m, 0.6 * best), self.gap_full_m))

    def gaps(self, scan: PolarScan) -> list[Gap]:
        """Contiguous bearing runs that lead somewhere and fit the rover."""
        out: list[Gap] = []
        n = len(scan.clearance_m)
        if n == 0:
            return out
        openb = scan.clearance_m >= self.open_threshold(scan)
        if not openb.any():
            return out
        step = (
            abs(float(scan.bin_centers[1] - scan.bin_centers[0]))
            if n > 1 else 2 * math.pi
        )
        full_circle = n > 1 and (n * step) > (2 * math.pi - 1e-6)

        # Start at a CLOSED bin so a run spanning the +/-pi seam is traversed
        # as one run, not split into two half-gaps.
        if full_circle and not openb.all():
            start = int(np.argmin(openb))
            order = [(start + k) % n for k in range(n)]
        else:
            order = list(range(n))

        runs: list[list[int]] = []
        cur: list[int] = []
        for k in order:
            if openb[k]:
                cur.append(k)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)

        for idx in runs:
            depth = float(np.min(scan.clearance_m[idx]))
            width = len(idx) * step
            # Must fit AT the depth it would be used: a 10-degree slot is
            # 0.35 m wide at 2 m and the rover is 0.7 m.
            if 2.0 * depth * math.sin(min(width, math.pi) / 2.0) < self.rover_width_m:
                continue
            # Unwrap before averaging, or a gap straddling +/-pi averages to 0,
            # i.e. straight ahead — the worst possible error.
            angs = np.unwrap(scan.bin_centers[idx])
            out.append(Gap(
                bearing=_wrap(float(angs.mean())),
                width_rad=float(width),
                depth_m=depth,
                frontier=float(np.mean(scan.unknown_frac[idx])),
            ))
        return out

    # ---- the decision -----------------------------------------------------
    def aim(
        self,
        scan: PolarScan,
        goal_bearing: float,
        theta: float,
        travelled_m: float,
        blocked_score=None,
    ) -> ExploreAim:
        """
        goal_bearing: robot frame, left-positive.
        theta: the rover's WORLD heading, so decisions survive the rover
            turning — same reason TopoGraph bins outcomes by world bearing.
        blocked_score: callable(world_bearing) -> 0..1, normally
            `TopoGraph.blocked_score`. Optional; without it the explorer works
            but has no memory of past failures.
        """
        gaps = self.gaps(scan)
        self.last_gaps = gaps
        self._expire_detour(scan, goal_bearing, travelled_m)

        if not gaps:
            # Nothing wide enough anywhere: say so and let the caller's escape
            # ladder run. Inventing a bearing here produced eight contradictory
            # turns.
            return ExploreAim(bearing=goal_bearing, confidence=0.0,
                              source="no_gap", speed_scale=0.0)

        options = self._options(scan, gaps, goal_bearing)
        for o in options:
            o.score = self._score(o, goal_bearing, theta, blocked_score)
        best = max(options, key=lambda o: o.score)

        # A 180 deg "gap" has its goal-nearest border almost behind the rover
        # (logged gap(+130 deg w=180 deg) while the dirt ahead was open). If the
        # hole in front is clear, do not latch a side/rear border.
        k0 = int(np.argmin(np.abs(_wrap_arr(scan.bin_centers))))
        if (
            float(scan.clearance_m[k0]) >= self.clear_ahead_m
            and abs(best.bearing) > math.radians(60.0)
        ):
            near = [o for o in options if abs(o.bearing) <= math.radians(60.0)]
            if near:
                best = max(near, key=lambda o: o.score)

        # Goal is clear ahead: no detour needed, and any running one is done.
        if self._goal_is_clear(scan, goal_bearing):
            if self.detour is not None:
                self.detour = None
            return ExploreAim(bearing=goal_bearing, confidence=0.4,
                              source="goal_clear", gap=best.gap,
                              depth_m=best.depth_m, frontier=best.frontier)

        if self.detour is None:
            side = 1 if _wrap(best.bearing) >= 0 else -1
            self.detour = Detour(
                side=side,
                bearing=_wrap(theta + best.bearing),
                started_travel_m=travelled_m,
                budget_m=self.commit_budget_m,
                reason=f"gap {np.degrees(best.bearing):+.0f}°",
            )

        # Follow the committed side. The stored bearing is in world terms, so
        # convert back against the CURRENT heading — that is what makes the
        # commitment hold while the rover turns.
        want = _wrap(self.detour.bearing - theta)
        # Re-aim within the committed side rather than grinding at a stale
        # angle.
        same_side = [
            o for o in options
            if (1 if o.bearing >= 0 else -1) == self.detour.side
        ]
        chosen = max(same_side, key=lambda o: o.score) if same_side else best
        if same_side:
            want = chosen.bearing
            self.detour.bearing = _wrap(theta + want)
        # A U-turn is not getting around something when the hole in front is
        # open: a map stained with near-field dirt committed -160/-180 deg and
        # yawed the rover off a closing GPS line.
        if (
            abs(want) > math.radians(70.0)
            and float(scan.clearance_m[k0]) >= self.clear_ahead_m
        ):
            self.detour = None
            return ExploreAim(
                bearing=0.0, confidence=0.5, source="forward_clear",
                gap=chosen.gap, depth_m=float(scan.clearance_m[k0]),
                frontier=float(scan.unknown_frac[k0]), speed_scale=1.0,
            )
        speed = 1.0 if chosen.depth_m > 2.0 * self.min_gap_m else 0.6
        return ExploreAim(bearing=want, confidence=0.8, source="detour",
                          speed_scale=speed, gap=chosen.gap, detour=self.detour,
                          depth_m=chosen.depth_m, frontier=chosen.frontier)

    def _options(self, scan: PolarScan, gaps: list[Gap],
                 goal_bearing: float) -> list["Option"]:
        """
        Turn gaps into concrete aim directions. A gap is not a decision.

        Narrow valleys are aimed at their centre; WIDE ones at the border
        nearest the target, stepped inward by half the required width. That
        distinction is load-bearing: with a wall ahead, every bearing from ~+35
        deg round the back to ~-35 deg is one contiguous open run whose centre
        is 180 deg — straight backwards. Aiming there turned the rover round
        instead of stepping past the wall's edge.

        Stepping inward from BOTH borders yields the two ways round an obstacle
        as separately-scored options, so left-vs-right is a real comparison
        rather than an artefact of where a run was averaged.
        """
        out: list[Option] = []
        for g in gaps:
            half = g.width_rad / 2.0
            if g.width_rad <= self.wide_rad:
                bearings = [g.bearing]
            else:
                inner = max(0.0, half - self.wide_rad / 2.0)
                bearings = [_wrap(g.bearing - inner), _wrap(g.bearing + inner)]
                # The goal itself, when it lies inside the valley: no reason to
                # hug an edge if the target direction is already drivable.
                if abs(_wrap(goal_bearing - g.bearing)) < inner:
                    bearings.append(_wrap(goal_bearing))
            for b in bearings:
                k = int(np.argmin(np.abs(_wrap_arr(scan.bin_centers - b))))
                out.append(Option(
                    bearing=b, gap=g,
                    depth_m=float(scan.clearance_m[k]),
                    frontier=float(scan.unknown_frac[k]),
                ))
        return out

    def _score(self, o: "Option", goal_bearing: float, theta: float,
               blocked_score) -> float:
        align = 1.0 - abs(_wrap(o.bearing - goal_bearing)) / math.pi
        depth = min(o.depth_m, 6.0) / 6.0
        s = (self.goal_weight * align
             + self.depth_weight * depth
             + self.frontier_weight * o.frontier)
        if blocked_score is not None:
            s -= self.blocked_weight * float(blocked_score(_wrap(theta + o.bearing)))
        if self.detour is not None and (1 if o.bearing >= 0 else -1) == self.detour.side:
            s += self.hysteresis
        return float(s)

    def _goal_is_clear(self, scan: PolarScan, goal_bearing: float) -> bool:
        """
        Bug2's rejoin test: is the goal direction itself drivable?

        Deliberately NOT gated on the goal being near straight ahead. Gating it
        conflates "is something in the way" with "am I already pointing at the
        goal", so on open ground with the goal 97 degrees to the side the
        explorer invented a detour around nothing.

        Unobserved space reads as clear, which is the conservative answer: with
        no evidence of an obstacle there is nothing to circumnavigate, and the
        corridor policy keeps the rover moving until there is.
        """
        k = int(np.argmin(np.abs(_wrap_arr(scan.bin_centers - goal_bearing))))
        return bool(scan.clearance_m[k] >= self.clear_ahead_m)

    def _expire_detour(self, scan, goal_bearing, travelled_m) -> None:
        d = self.detour
        if d is None:
            return
        if travelled_m - d.started_travel_m > d.budget_m:
            self.detour = None
            return
        if self._goal_is_clear(scan, goal_bearing):
            self.detour = None

    def describe(self) -> str:
        if self.detour is None:
            return f"gaps={len(self.last_gaps)}"
        return f"gaps={len(self.last_gaps)} {self.detour.describe()}"


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def _wrap_arr(a: np.ndarray) -> np.ndarray:
    return (a + math.pi) % (2 * math.pi) - math.pi
