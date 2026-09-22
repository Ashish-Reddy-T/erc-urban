"""
Global routing over OpenStreetMap. The layer the stack never had.

WHY
---
Between checkpoints the rover drove the straight chord, because that is all a
GPS bearing knows. A chord ignores where the walkable surface actually is, and
on a real site it points through buildings, across roads and over walls — the
local planner then spends its whole budget discovering that, one 3.5 m costmap
at a time.

The dashboard minimap is the standing proof that better data exists: it is a
Leaflet render of OSM, with the rover and the checkpoints drawn on it. That
raster is downstream of vector data anyone can query. So query the vectors.

RELATION TO THE MINIMAP AND TO THE VLM
--------------------------------------
`sdk_client.get_map_frame()` returns the same information as a PICTURE. Asking a
VLM to recover topology from it means OCR-ing a render of a database we could
have queried directly — lossy, 1-3 s per query, and able to hallucinate a path
that is not there. This module is the other arm of that comparison
(`eval/map_compare.py`); it is the one that returns geometry.

The VLM's edge is on the FRONT camera, judging surfaces OSM cannot know about:
whether the gate is open, whether that is a ramp or a kerb, whether the mapped
"footway" is currently a building site.

DESIGN, ADAPTED FROM ERC-3's `src/osm_router.py`
------------------------------------------------
Same shape, same tuned cost multipliers (footway 1.0 ... primary 6.0, so the
router prefers pedestrian ways and only touches roads when it must), A* over a
node graph built from the Overpass payload.

Two things differ, both because this runs inside a live control loop:

  * `fetch` is injectable. Overpass is a public, rate-limited, sometimes-down
    service, and a navigation stack that hard-depends on a third-party HTTP
    endpoint has swapped one single point of failure for another. Tests inject
    a fixture; the rover injects a cached fetcher.
  * Every failure returns the straight chord rather than raising. Degrading to
    the behaviour we already had is always better than stopping.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

_R_EARTH = 6_371_000.0

# Pedestrian-first traversal costs, from ERC-3's tuning. A multiplier is a
# preference, not a ban: a rover that refuses to touch a road cannot cross one.
HIGHWAY_COST = {
    "footway": 1.0,
    "pedestrian": 1.0,
    "path": 1.05,
    "living_street": 1.20,
    "cycleway": 1.60,
    "service": 1.75,
    "track": 1.90,
    "residential": 2.20,
    "unclassified": 2.40,
    "tertiary": 2.80,
    "road": 3.00,
    "secondary": 4.00,
    "primary": 6.00,
}

OVERPASS_URL = os.environ.get(
    "OVERPASS_URL", "https://overpass-api.de/api/interpreter")


@dataclass
class Route:
    """A global route, or the honest absence of one."""

    waypoints: list[tuple[float, float]]      # (lat, lon), start -> goal
    source: str                               # "osm" | "chord" | "cached"
    distance_m: float = 0.0
    detour_ratio: float = 1.0                 # route length / chord length
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.source != "chord" and len(self.waypoints) > 2

    def describe(self) -> str:
        return (f"osm[{self.source} n={len(self.waypoints)} "
                f"{self.distance_m:.0f}m x{self.detour_ratio:.2f}"
                f"{' ' + self.note if self.note else ''}]")


def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    p0, p1 = math.radians(a[0]), math.radians(b[0])
    dlat, dlon = p1 - p0, math.radians(b[1] - a[1])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(p0) * math.cos(p1) * math.sin(dlon / 2) ** 2)
    return 2 * _R_EARTH * math.asin(min(1.0, math.sqrt(h)))


def bbox(start, goal, buffer_m: float = 300.0):
    """Lat/lon box covering both points plus a margin, for the Overpass query."""
    dlat = buffer_m / 111_320.0
    mid = math.radians((start[0] + goal[0]) / 2.0)
    dlon = buffer_m / max(111_320.0 * math.cos(mid), 1.0)
    return (min(start[0], goal[0]) - dlat, min(start[1], goal[1]) - dlon,
            max(start[0], goal[0]) + dlat, max(start[1], goal[1]) + dlon)


def overpass_query(box, timeout_s: int = 25) -> str:
    s, w, n, e = box
    kinds = "|".join(HIGHWAY_COST)
    return (f'[out:json][timeout:{timeout_s}];\n'
            f'way["highway"~"{kinds}"]({s},{w},{n},{e});\n'
            f'(._;>;);\nout body;')


# Overpass etiquette REQUIRES a descriptive User-Agent, and the mirrors enforce
# it: `python-requests/2.x` is filtered and the server answers 406 Not
# Acceptable with an HTML body. Measured — with the default header every query
# for Wuhan, Kisumu and Gaborone returned nothing and the router silently fell
# back to the chord, which looked exactly like "OSM has no coverage there".
HTTP_HEADERS = {
    "User-Agent": os.environ.get(
        "OSM_USER_AGENT", "erc-urban-rover/1.0 (IROS Earth Rover Challenge)"),
    "Accept": "application/json",
}

# Last transport failure, for callers that want to say WHY there is no route.
# Swallowing this is how the VLM selector ran 45 queries against a 403 while
# reporting only `a=0`; the error was recorded and never surfaced.
last_error: Optional[str] = None


def http_fetch(box, timeout_s: float = 30.0) -> Optional[dict]:
    """Real Overpass call. Returns None on any failure — never raises."""
    global last_error
    try:
        import requests

        r = requests.post(OVERPASS_URL,
                          data={"data": overpass_query(box, int(timeout_s))},
                          headers=HTTP_HEADERS, timeout=timeout_s)
        if r.status_code != 200:
            last_error = f"HTTP {r.status_code}: {r.text[:120]}"
            return None
        last_error = None
        return r.json()
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        return None


class CachedFetch:
    """
    Overpass wrapper that queries rarely and remembers what it got.

    Overpass is public and rate-limited; a control loop that queried it per tick
    would be throttled within a minute and would deserve to be. The road network
    does not change during a mission, so one payload per area is enough.
    """

    def __init__(self, cache_dir: str = ".osm_cache", ttl_s: float = 86400.0,
                 fetch: Callable = http_fetch):
        self.cache_dir = cache_dir
        self.ttl_s = ttl_s
        self.fetch = fetch
        self.n_http = 0
        self.n_hits = 0

    def _key(self, box) -> str:
        return "osm_" + "_".join(f"{v:.4f}" for v in box) + ".json"

    def __call__(self, box) -> Optional[dict]:
        path = os.path.join(self.cache_dir, self._key(box))
        if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < self.ttl_s:
            try:
                with open(path) as fh:
                    self.n_hits += 1
                    return json.load(fh)
            except Exception:
                pass
        payload = self.fetch(box)
        self.n_http += 1
        if payload is not None:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(path, "w") as fh:
                    json.dump(payload, fh)
            except Exception:
                pass
        return payload


def build_graph(payload: dict):
    """Overpass payload -> (nodes {id: (lat,lon)}, adjacency {id: [(id, cost)]})."""
    nodes: dict[int, tuple[float, float]] = {}
    for el in payload.get("elements", []):
        if el.get("type") == "node":
            nodes[int(el["id"])] = (float(el["lat"]), float(el["lon"]))

    adj: dict[int, list[tuple[int, float]]] = {}
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        tags = el.get("tags", {}) or {}
        mult = HIGHWAY_COST.get(str(tags.get("highway", "")).lower())
        if mult is None:
            continue
        ids = [int(n) for n in el.get("nodes", []) if int(n) in nodes]
        for a, b in zip(ids[:-1], ids[1:]):
            d = haversine_m(nodes[a], nodes[b])
            # Undirected on purpose. `oneway` constrains vehicles; a 0.35 m
            # rover on a footway is a pedestrian, and routing it the long way
            # round a one-way street would be a bug, not compliance.
            adj.setdefault(a, []).append((b, d * mult))
            adj.setdefault(b, []).append((a, d * mult))
    return nodes, adj


def nearest_node(pt, nodes) -> tuple[Optional[int], float]:
    best, best_d = None, float("inf")
    for nid, ll in nodes.items():
        d = haversine_m(pt, ll)
        if d < best_d:
            best, best_d = nid, d
    return best, best_d


def astar(start_id: int, goal_id: int, nodes, adj) -> list[int]:
    import heapq

    if start_id == goal_id:
        return [start_id]
    g = {start_id: 0.0}
    came: dict[int, int] = {}
    seen: set[int] = set()
    h0 = haversine_m(nodes[start_id], nodes[goal_id])
    q = [(h0, start_id)]
    while q:
        _, cur = heapq.heappop(q)
        if cur == goal_id:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        if cur in seen:
            continue
        seen.add(cur)
        for nb, cost in adj.get(cur, []):
            t = g[cur] + cost
            if t < g.get(nb, float("inf")):
                g[nb], came[nb] = t, cur
                heapq.heappush(q, (t + haversine_m(nodes[nb], nodes[goal_id]), nb))
    return []


def thin(points, min_spacing_m: float = 6.0):
    """Drop waypoints closer than the rover's own sensing horizon."""
    if len(points) < 3:
        return list(points)
    out = [points[0]]
    for p in points[1:-1]:
        if haversine_m(out[-1], p) >= min_spacing_m:
            out.append(p)
    out.append(points[-1])
    return out


def plan(start, goal, fetch: Optional[Callable] = None,
         max_snap_m: float = 60.0) -> Route:
    """
    Route from `start` to `goal` over OSM pedestrian ways.

    Always returns a Route. Anything that goes wrong — no network, no coverage,
    nothing within `max_snap_m` of the rover, no connected path — degrades to
    the straight chord, which is exactly the behaviour the stack has today. A
    router that can fail closed is worse than no router.
    """
    chord = haversine_m(start, goal)
    fallback = Route([start, goal], "chord", chord, 1.0, "")

    payload = (fetch or http_fetch)(bbox(start, goal))
    if not payload:
        return Route([start, goal], "chord", chord, 1.0, "no_payload")

    nodes, adj = build_graph(payload)
    if not nodes:
        return Route([start, goal], "chord", chord, 1.0, "no_ways")

    s_id, s_d = nearest_node(start, nodes)
    g_id, g_d = nearest_node(goal, nodes)
    if s_id is None or g_id is None:
        return Route([start, goal], "chord", chord, 1.0, "no_snap")
    if s_d > max_snap_m or g_d > max_snap_m:
        # Off-network. Common on open ground, and the honest answer is that OSM
        # has nothing to say here — not that the rover should drive to the
        # nearest mapped way just because one exists.
        return Route([start, goal], "chord", chord, 1.0,
                     f"off_network({max(s_d, g_d):.0f}m)")

    path_ids = astar(s_id, g_id, nodes, adj)
    if len(path_ids) < 2:
        return Route([start, goal], "chord", chord, 1.0, "disconnected")

    pts = [nodes[i] for i in path_ids]
    dist = sum(haversine_m(a, b) for a, b in zip(pts[:-1], pts[1:]))
    ratio = dist / max(chord, 1e-6)
    wps = thin([start] + pts + [goal])
    return Route(wps, "osm", dist, ratio,
                 f"snap={s_d:.0f}/{g_d:.0f}m")


def next_target(route: Route, here, lookahead_m: float = 12.0):
    """
    The (lat, lon) the local planner should currently steer at.

    A lookahead point along the route rather than its far end: the local stack
    is a 3.5 m costmap and a ~2 m planning horizon, so handing it a target 200 m
    away is the same as handing it the chord and losing everything the route
    knew.
    """
    if not route.waypoints:
        return here
    best_i, best_d = 0, float("inf")
    for i, p in enumerate(route.waypoints):
        d = haversine_m(here, p)
        if d < best_d:
            best_i, best_d = i, d
    for p in route.waypoints[best_i:]:
        if haversine_m(here, p) >= lookahead_m:
            return p
    return route.waypoints[-1]
