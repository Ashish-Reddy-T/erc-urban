"""
The topological map — the stack's memory of what it has TRIED.

`LocalMap` remembers where things ARE; it does not remember what happened. The
failure this fixes is a livelock on open ground: eight escapes at +116, +113,
+134, +130, -160, +112, -165, +101 degrees, each computed from scratch and each
undone by the next tick's align, because nothing recorded that a heading had
already been tried and failed.

An incrementally-built graph over places, with GPS as a hint rather than a
coordinate. Geometric, so it costs no extra network forward pass.

BEARINGS ARE WORLD-RELATIVE
---------------------------
Every outcome is binned by WORLD heading. This is the load-bearing detail:
"blocked at +40 deg" means nothing once the rover has turned 40 degrees, while
"blocked toward world-north" survives precisely the manoeuvres that create it.

LOOP CLOSURE
------------
By proximity in the map frame at `merge_radius_m` — no visual place recognition,
because over the 10-30 m horizons this rover works at, odometry drift stays well
inside 1.2 m. The honest limitation: after a long excursion it may fail to
recognise a place it has returned to and will re-explore it. That is a wasted
detour, not a wrong turn.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DRIVEN = "driven"
OPEN = "open"
BLOCKED = "blocked"


@dataclass
class Node:
    """One place the rover has been, and what it learned there."""

    id: int
    x: float
    y: float
    stamp: float
    lat: Optional[float] = None
    lon: Optional[float] = None
    goal_range_m: Optional[float] = None
    visits: int = 1
    # World bearing bin -> outcome. Blocked counts are separate because one
    # failed attempt is a hint and three is a fact.
    attempts: dict = field(default_factory=dict)
    blocked_n: dict = field(default_factory=dict)
    # Bin -> the actual world bearing last recorded blocked. Kept alongside the
    # bin because scoring measures angular distance from what was observed;
    # from the bin CENTRE it would be off by up to half a bin.
    blocked_at: dict = field(default_factory=dict)

    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], np.float64)


class TopoGraph:
    def __init__(
        self,
        node_spacing_m: float = 1.5,
        merge_radius_m: float = 1.2,
        heading_step_deg: float = 45.0,
        n_bins: int = 12,
        blocked_hard_n: int = 2,
        max_nodes: int = 400,
    ):
        """
        n_bins: 12 gives 30-degree resolution. Finer is false precision — the
            compass drifts by more than one bin.
        blocked_hard_n: attempts in the same bin before it counts as genuinely
            impassable rather than one bad look. The world is dynamic; a single
            failure must not permanently wall off a direction.
        """
        self.node_spacing_m = node_spacing_m
        self.merge_radius_m = merge_radius_m
        self.heading_step = math.radians(heading_step_deg)
        self.n_bins = n_bins
        self.blocked_hard_n = blocked_hard_n
        self.max_nodes = max_nodes
        self.nodes: list[Node] = []
        self.edges: dict[tuple[int, int], str] = {}
        self.current: Optional[Node] = None
        self._last_node_travel = 0.0
        self._last_node_theta: Optional[float] = None
        self._next_id = 0

    # ---- bins -------------------------------------------------------------
    def bin_of(self, world_bearing: float) -> int:
        return int(
            math.floor((_wrap(world_bearing) + math.pi)
                       / (2 * math.pi) * self.n_bins)
        ) % self.n_bins

    def bin_bearing(self, b: int) -> float:
        return _wrap(-math.pi + (b + 0.5) * 2 * math.pi / self.n_bins)

    # ---- construction -----------------------------------------------------
    def update(
        self,
        x: float,
        y: float,
        theta: float,
        travelled_m: float,
        stamp: float,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        goal_range_m: Optional[float] = None,
        force: bool = False,
    ) -> Node:
        """
        Drop or revisit a node for the rover's current pose.

        `force` is for stop events: wherever a stop happened is worth
        remembering however recently the last node was placed, because that is
        the place whose blocked bearings matter most on the next pass.
        """
        near = self._nearest(x, y)
        if near is not None and _dist(near, x, y) <= self.merge_radius_m:
            if near is not self.current:
                self._link(self.current, near, DRIVEN)
                near.visits += 1
                self.current = near
                self._last_node_travel = travelled_m
                self._last_node_theta = theta
            near.stamp = stamp
            if lat is not None:
                near.lat, near.lon = lat, lon
            if goal_range_m is not None:
                near.goal_range_m = goal_range_m
            return near

        moved = travelled_m - self._last_node_travel
        turned = (
            abs(_wrap(theta - self._last_node_theta))
            if self._last_node_theta is not None else 0.0
        )
        if (
            self.current is not None
            and not force
            and moved < self.node_spacing_m
            and turned < self.heading_step
        ):
            return self.current

        node = Node(id=self._next_id, x=float(x), y=float(y), stamp=float(stamp),
                    lat=lat, lon=lon, goal_range_m=goal_range_m)
        self._next_id += 1
        self.nodes.append(node)
        self._link(self.current, node, DRIVEN)
        if self.current is not None:
            # We got here from the previous node, so that direction is proven
            # drivable. Free evidence, and it stops an escape routing the rover
            # back through a wall.
            self.mark(_bearing(self.current, node), DRIVEN, at=self.current)
        self.current = node
        self._last_node_travel = travelled_m
        self._last_node_theta = theta
        if len(self.nodes) > self.max_nodes:
            self._evict()
        return node

    def _link(self, a: Optional[Node], b: Optional[Node], kind: str) -> None:
        if a is None or b is None or a.id == b.id:
            return
        self.edges[(a.id, b.id)] = kind
        self.edges[(b.id, a.id)] = kind

    def _nearest(self, x: float, y: float) -> Optional[Node]:
        if not self.nodes:
            return None
        d = [(_dist(n, x, y), n) for n in self.nodes]
        return min(d, key=lambda t: t[0])[1]

    def _evict(self) -> None:
        """Drop the oldest node. Bounded memory: this is not a SLAM map."""
        old = min(self.nodes, key=lambda n: n.stamp)
        self.nodes.remove(old)
        for k in [k for k in self.edges if old.id in k]:
            self.edges.pop(k, None)

    # ---- outcomes ---------------------------------------------------------
    def mark(self, world_bearing: float, outcome: str,
             at: Optional[Node] = None) -> None:
        node = at or self.current
        if node is None:
            return
        b = self.bin_of(world_bearing)
        if outcome == BLOCKED:
            node.blocked_n[b] = node.blocked_n.get(b, 0) + 1
            node.blocked_at[b] = _wrap(world_bearing)
            # A direction we have DRIVEN is not reclassified by one bad look.
            if node.attempts.get(b) != DRIVEN:
                node.attempts[b] = BLOCKED
        else:
            node.attempts[b] = outcome
            if outcome == DRIVEN:
                node.blocked_n.pop(b, None)
                node.blocked_at.pop(b, None)

    def blocked_score(self, world_bearing: float,
                      at: Optional[Node] = None) -> float:
        """
        0 = never failed here, 1 = failed enough times to believe it.

        Continuous in confidence, so the explorer can DOWNWEIGHT a direction
        rather than forbid it — forbidding fails closed, and a rover that has
        walled off every bearing has no move left.

        Continuous in angle too, falling off over ~1.5 bins, because exact bin
        matching is too sharp: the compass drifts by more than a bin width, and
        bin edges fall on the round numbers where recorded bearings cluster
        (30 deg floors to bin 6 via 6.999999999999999 while 60 and 90 land
        cleanly), so a contiguous arc of failures was recorded gap-toothed.
        """
        node = at or self.current
        if node is None or not node.blocked_n:
            return 0.0
        width = 2 * math.pi / self.n_bins
        reach = 1.5 * width
        best = 0.0
        for b, n in node.blocked_n.items():
            ref = node.blocked_at.get(b, self.bin_bearing(b))
            d = abs(_wrap(world_bearing - ref))
            if d >= reach:
                continue
            conf = min(1.0, n / max(1, self.blocked_hard_n))
            best = max(best, conf * (1.0 - d / reach))
        return float(best)

    def blocked_bins(self, at: Optional[Node] = None) -> set[int]:
        node = at or self.current
        if node is None:
            return set()
        return {b for b, n in node.blocked_n.items() if n >= self.blocked_hard_n}

    # ---- escape -----------------------------------------------------------
    def escape_bearing(self, max_hops: int = 6) -> Optional[float]:
        """
        World bearing toward the nearest place with something untried.

        The primary escape source, ahead of `Breadcrumbs.retreat_bearing`: that
        answers "which way did I come from" from a trail a pure pivot fills with
        noise, this answers "where is there something I have not tried", which
        is what an escape is actually asking.

        None when the graph knows nowhere better, so the caller falls through to
        the sweep rather than being handed a fabricated heading.
        """
        if self.current is None or len(self.nodes) < 2:
            return None
        seen = {self.current.id}
        frontier = [(self.current, 0)]
        by_id = {n.id: n for n in self.nodes}
        while frontier:
            node, hops = frontier.pop(0)
            if hops > max_hops:
                break
            if node.id != self.current.id and self._has_untried(node):
                v = node.xy() - self.current.xy()
                if float(np.hypot(*v)) > 1e-3:
                    return float(math.atan2(v[1], v[0]))
            for (a, b), kind in self.edges.items():
                if a != node.id or b in seen or kind == BLOCKED:
                    continue
                seen.add(b)
                if b in by_id:
                    frontier.append((by_id[b], hops + 1))
        return None

    def _has_untried(self, node: Node) -> bool:
        hard = {b for b, n in node.blocked_n.items() if n >= self.blocked_hard_n}
        return len(hard) + len(node.attempts) < self.n_bins

    def describe(self) -> str:
        blocked = len(self.blocked_bins())
        return (f"topo(n={len(self.nodes)} visits="
                f"{self.current.visits if self.current else 0} "
                f"blocked={blocked}/{self.n_bins})")


def _wrap(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def _dist(n: Node, x: float, y: float) -> float:
    return float(math.hypot(n.x - x, n.y - y))


def _bearing(a: Node, b: Node) -> float:
    return float(math.atan2(b.y - a.y, b.x - a.x))
