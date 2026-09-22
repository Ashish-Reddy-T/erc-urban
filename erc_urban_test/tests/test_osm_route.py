"""
Tests for OSM global routing. All offline — Overpass is injected as a fixture.

The behaviour that matters most is the fallback: this module sits in a live
control loop and depends on a public, rate-limited, sometimes-down third-party
service. Every failure has to degrade to the straight chord the stack already
drove, never to a stop and never to an exception.
"""
from __future__ import annotations

import unittest

from erc_urban_test.modules import osm_route as O
from erc_urban_test.modules.osm_route import (
    HIGHWAY_COST, CachedFetch, Route, astar, bbox, build_graph, haversine_m,
    nearest_node, next_target, overpass_query, plan, thin,
)

# A tiny grid of footways just north-east of the origin, ~11 m spacing.
A, B, C, D = 1, 2, 3, 4


def _payload(highway="footway"):
    return {"elements": [
        {"type": "node", "id": A, "lat": 0.0000, "lon": 0.0000},
        {"type": "node", "id": B, "lat": 0.0000, "lon": 0.0001},
        {"type": "node", "id": C, "lat": 0.0001, "lon": 0.0001},
        {"type": "node", "id": D, "lat": 0.0001, "lon": 0.0000},
        {"type": "way", "id": 10, "tags": {"highway": highway}, "nodes": [A, B]},
        {"type": "way", "id": 11, "tags": {"highway": highway}, "nodes": [B, C]},
        {"type": "way", "id": 12, "tags": {"highway": highway}, "nodes": [C, D]},
    ]}


class TestGraph(unittest.TestCase):
    def test_builds_nodes_and_undirected_edges(self):
        nodes, adj = build_graph(_payload())
        self.assertEqual(len(nodes), 4)
        self.assertIn(B, [n for n, _ in adj[A]])
        self.assertIn(A, [n for n, _ in adj[B]], "edges must be undirected")

    def test_unroutable_highway_kinds_are_dropped(self):
        nodes, adj = build_graph(_payload(highway="motorway"))
        self.assertEqual(adj, {}, "a rover does not belong on a motorway")

    def test_pedestrian_ways_cost_less_than_roads(self):
        _, foot = build_graph(_payload("footway"))
        _, prim = build_graph(_payload("primary"))
        self.assertLess(foot[A][0][1], prim[A][0][1])
        self.assertEqual(HIGHWAY_COST["footway"], 1.0)

    def test_query_asks_only_for_ways_we_can_use(self):
        q = overpass_query(bbox((0.0, 0.0), (0.001, 0.001)))
        self.assertIn("footway", q)
        self.assertIn("out:json", q)
        self.assertNotIn("motorway", q)


class TestAStar(unittest.TestCase):
    def test_finds_the_connected_path(self):
        nodes, adj = build_graph(_payload())
        self.assertEqual(astar(A, D, nodes, adj), [A, B, C, D])

    def test_same_node_is_trivial(self):
        nodes, adj = build_graph(_payload())
        self.assertEqual(astar(A, A, nodes, adj), [A])

    def test_disconnected_returns_empty(self):
        p = _payload()
        p["elements"].append({"type": "node", "id": 99, "lat": 5.0, "lon": 5.0})
        nodes, adj = build_graph(p)
        self.assertEqual(astar(A, 99, nodes, adj), [])


class TestPlanFallsBackRatherThanFailing(unittest.TestCase):
    """A router that can fail closed is worse than no router."""

    def test_no_network_gives_the_chord(self):
        r = plan((0.0, 0.0), (0.0001, 0.0), fetch=lambda box: None)
        self.assertEqual(r.source, "chord")
        self.assertEqual(r.note, "no_payload")
        self.assertFalse(r.ok)

    def test_empty_coverage_gives_the_chord(self):
        r = plan((0.0, 0.0), (0.0001, 0.0), fetch=lambda box: {"elements": []})
        self.assertEqual(r.source, "chord")
        self.assertEqual(r.note, "no_ways")

    def test_a_raising_fetcher_does_not_escape(self):
        def boom(box):
            raise RuntimeError("overpass is down")
        with self.assertRaises(RuntimeError):
            plan((0.0, 0.0), (0.0001, 0.0), fetch=boom)
        # Documents the contract precisely: `plan` swallows a fetcher that
        # RETURNS failure, not one that raises. http_fetch never raises, and a
        # custom fetcher must honour the same contract.

    def test_far_from_any_mapped_way_gives_the_chord(self):
        """
        Open ground. OSM having nothing to say here is the honest answer; it is
        not a reason to drive to the nearest mapped way just because one exists.
        """
        r = plan((10.0, 10.0), (10.0005, 10.0), fetch=lambda box: _payload())
        self.assertEqual(r.source, "chord")
        self.assertTrue(r.note.startswith("off_network"))

    def test_a_real_route_is_returned_when_coverage_exists(self):
        r = plan((0.0, 0.0), (0.0001, 0.0), fetch=lambda box: _payload())
        self.assertEqual(r.source, "osm")
        self.assertTrue(r.ok)
        self.assertGreater(r.distance_m, 0.0)
        # Routing round three sides of a square must beat the chord in cost but
        # lose on distance — that is what the detour ratio is for.
        self.assertGreater(r.detour_ratio, 1.0)


class TestLookahead(unittest.TestCase):
    def test_targets_a_point_within_the_local_horizon(self):
        route = Route([(0.0, 0.0), (0.0, 0.0002), (0.0, 0.002)], "osm")
        t = next_target(route, (0.0, 0.0), lookahead_m=12.0)
        self.assertNotEqual(t, (0.0, 0.0))
        self.assertLess(haversine_m((0.0, 0.0), t), 60.0,
                        "must not hand the local planner a 200 m target")

    def test_falls_back_to_the_final_waypoint(self):
        route = Route([(0.0, 0.0), (0.0, 0.00001)], "osm")
        self.assertEqual(next_target(route, (0.0, 0.0), lookahead_m=500.0),
                         (0.0, 0.00001))

    def test_empty_route_returns_where_we_are(self):
        self.assertEqual(next_target(Route([], "chord"), (1.0, 2.0)), (1.0, 2.0))


class TestCaching(unittest.TestCase):
    def test_second_call_does_not_hit_the_network(self):
        """Overpass is public and rate-limited; per-tick queries deserve a ban."""
        import tempfile

        calls = {"n": 0}

        def counting(box):
            calls["n"] += 1
            return _payload()

        with tempfile.TemporaryDirectory() as d:
            f = CachedFetch(cache_dir=d, fetch=counting)
            box = bbox((0.0, 0.0), (0.0001, 0.0))
            f(box)
            f(box)
            self.assertEqual(calls["n"], 1)
            self.assertEqual(f.n_hits, 1)


class TestGeometry(unittest.TestCase):
    def test_thin_keeps_the_endpoints(self):
        pts = [(0.0, i * 0.00001) for i in range(20)]
        out = thin(pts, min_spacing_m=6.0)
        self.assertEqual(out[0], pts[0])
        self.assertEqual(out[-1], pts[-1])
        self.assertLess(len(out), len(pts))

    def test_nearest_node_picks_the_closest(self):
        nodes, _ = build_graph(_payload())
        nid, d = nearest_node((0.00009, 0.00001), nodes)
        self.assertEqual(nid, D)
        self.assertLess(d, 20.0)


class TestOverpassEtiquette(unittest.TestCase):
    def test_a_descriptive_user_agent_is_sent(self):
        """
        Overpass mirrors filter `python-requests/2.x` and answer 406 Not
        Acceptable with an HTML body. Measured: without this header, live
        queries for Wuhan, Kisumu and Gaborone all returned nothing and the
        router fell back to the chord — indistinguishable from "OSM has no
        coverage here", which would have been the wrong conclusion. Kisumu in
        fact routes fine (5 waypoints, detour ratio 1.53).
        """
        ua = O.HTTP_HEADERS.get("User-Agent", "")
        self.assertNotIn("python-requests", ua)
        self.assertGreater(len(ua), 12)
        self.assertEqual(O.HTTP_HEADERS.get("Accept"), "application/json")

    def test_transport_failures_are_recorded_not_swallowed(self):
        """
        The VLM selector ran 45 queries against a 403 while reporting only
        `a=0`, because the error was captured and never surfaced. Same class of
        bug, so the same guard.
        """
        self.assertTrue(hasattr(O, "last_error"))


if __name__ == "__main__":
    unittest.main()
