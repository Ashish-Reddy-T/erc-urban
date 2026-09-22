"""
Regression tests for the topological graph (M8b).

This module exists to stop one specific failure: eight escapes in a row, each
computed from scratch, each turning a fresh amount (+116, +113, +134, +130,
-160, +112, -165, +101 degrees) while the rover never left one spot. Nothing
recorded that a heading had already been tried and failed, so it was always
available to try again.

The property that makes the memory useful is that outcomes are stored in WORLD
bearings. A robot-relative record is worthless here, because turning away from
a failed direction is exactly what happens immediately after recording it.
"""
from __future__ import annotations

import math
import unittest

from erc_urban.modules.topograph import TopoGraph, BLOCKED, DRIVEN


class TestNodes(unittest.TestCase):
    def test_first_update_creates_a_node(self):
        g = TopoGraph()
        n = g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertIsNotNone(n)
        self.assertEqual(len(g.nodes), 1)

    def test_small_moves_do_not_spawn_nodes(self):
        g = TopoGraph(node_spacing_m=1.5)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        for k in range(1, 6):
            g.update(0.1 * k, 0.0, 0.0, 0.1 * k, float(k))
        self.assertEqual(len(g.nodes), 1)

    def test_travelling_far_enough_spawns_a_node(self):
        g = TopoGraph(node_spacing_m=1.5)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.update(3.0, 0.0, 0.0, 3.0, 1.0)
        self.assertEqual(len(g.nodes), 2)

    def test_a_big_turn_spawns_a_node_even_without_travel(self):
        """A place seen from a different heading is a different observation,
        and stop events cluster around turns."""
        g = TopoGraph(heading_step_deg=45.0, merge_radius_m=0.05)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.update(0.1, 0.0, math.radians(90.0), 0.1, 1.0)
        self.assertEqual(len(g.nodes), 2)

    def test_force_spawns_a_node_regardless(self):
        g = TopoGraph(merge_radius_m=0.05)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.update(0.1, 0.0, 0.0, 0.1, 1.0, force=True)
        self.assertEqual(len(g.nodes), 2)

    def test_returning_to_a_place_revisits_rather_than_duplicates(self):
        g = TopoGraph(merge_radius_m=1.2)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.update(5.0, 0.0, 0.0, 5.0, 1.0)
        g.update(0.3, 0.2, 0.0, 10.0, 2.0)
        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(g.current.visits, 2)

    def test_node_count_is_bounded(self):
        g = TopoGraph(node_spacing_m=0.5, merge_radius_m=0.05, max_nodes=20)
        for k in range(200):
            g.update(k * 1.0, 0.0, 0.0, k * 1.0, float(k))
        self.assertLessEqual(len(g.nodes), 20)


class TestBlockedMemory(unittest.TestCase):
    def test_one_failure_is_a_hint_not_a_fact(self):
        """The world is dynamic. A parked car moves; a pedestrian moves. One
        failure must not permanently wall off a direction."""
        g = TopoGraph(blocked_hard_n=2)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.mark(0.0, BLOCKED)
        self.assertEqual(g.blocked_score(0.0), 0.5)
        self.assertNotIn(g.bin_of(0.0), g.blocked_bins())

    def test_repeated_failure_is_believed(self):
        g = TopoGraph(blocked_hard_n=2)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.mark(0.0, BLOCKED)
        g.mark(0.0, BLOCKED)
        self.assertEqual(g.blocked_score(0.0), 1.0)
        self.assertIn(g.bin_of(0.0), g.blocked_bins())

    def test_an_unrelated_bearing_is_unaffected(self):
        g = TopoGraph()
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.mark(0.0, BLOCKED)
        g.mark(0.0, BLOCKED)
        self.assertEqual(g.blocked_score(math.pi / 2), 0.0)

    def test_memory_survives_the_rover_turning(self):
        """
        THE regression. Outcomes are binned by world bearing, so a heading that
        failed stays failed no matter which way the rover is now pointing.
        Store these robot-relative and the memory evaporates on the very
        manoeuvre that created it.
        """
        g = TopoGraph(blocked_hard_n=2)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.mark(0.0, BLOCKED)          # world-north failed
        g.mark(0.0, BLOCKED)

        theta = math.radians(90.0)    # rover turns 90 degrees
        g.update(0.0, 0.0, theta, 0.0, 1.0)
        # Robot-relative -90 now points at world-north.
        self.assertEqual(g.blocked_score(theta + math.radians(-90.0)), 1.0)
        # And straight ahead is now a different, untried direction.
        self.assertEqual(g.blocked_score(theta), 0.0)

    def test_a_driven_direction_is_not_demoted_by_one_bad_look(self):
        g = TopoGraph()
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.mark(0.0, DRIVEN)
        g.mark(0.0, BLOCKED)
        self.assertEqual(g.current.attempts[g.bin_of(0.0)], DRIVEN)

    def test_blocked_memory_is_per_place(self):
        g = TopoGraph(node_spacing_m=1.0, merge_radius_m=0.5)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.mark(0.0, BLOCKED)
        g.mark(0.0, BLOCKED)
        g.update(10.0, 0.0, 0.0, 10.0, 1.0)
        # Somewhere else entirely: north has not failed here.
        self.assertEqual(g.blocked_score(0.0), 0.0)


class TestEscapeRouting(unittest.TestCase):
    def test_no_graph_means_no_answer(self):
        """Must return None rather than fabricate a heading — the caller then
        falls through to a sensed sweep instead of turning on faith."""
        g = TopoGraph()
        self.assertIsNone(g.escape_bearing())
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        self.assertIsNone(g.escape_bearing())

    def test_points_toward_a_place_with_something_untried(self):
        g = TopoGraph(node_spacing_m=1.0, merge_radius_m=0.5)
        g.update(0.0, 0.0, 0.0, 0.0, 0.0)
        g.update(5.0, 0.0, 0.0, 5.0, 1.0)       # node 1, largely untried
        # Exhaust every direction at the current node.
        for b in range(g.n_bins):
            g.mark(g.bin_bearing(b), BLOCKED)
        b = g.escape_bearing()
        self.assertIsNotNone(b)
        # Node 0 sits at world -X from here.
        self.assertAlmostEqual(abs(b), math.pi, places=3)


if __name__ == "__main__":
    unittest.main()
