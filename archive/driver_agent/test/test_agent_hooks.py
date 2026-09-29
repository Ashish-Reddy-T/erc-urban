"""run.py agent hooks: the heading override steers the fan; it expires."""
import math
import time
import unittest

import erc_urban.run as R
import erc_urban.sim_fan as SF


class TestAimOverride(unittest.TestCase):
    def tearDown(self):
        R._AIM_OVERRIDE = None
        R._AGENT_BACK = 0.0

    def test_override_steers_east(self):
        w = SF.World([], (0, 0, 0), (0, 30), 60, "open, goal north")
        R._CRUMBS.travelled_m = 0.0
        R._AIM_OVERRIDE = {"world_deg": 90.0, "until_m": 1e9, "t_until": 1e12}
        r = SF.run(w)
        e, n = r.trail[-1]
        self.assertGreater(e, 2.0)            # went east, not north to the goal
        self.assertGreater(e, abs(n))
        self.assertTrue(R._OBS)               # observation published

    def test_override_expires_on_distance(self):
        w = SF.World([], (0, 0, 0), (0, 10), 400, "open, goal north")
        R._AIM_OVERRIDE = {"world_deg": 90.0, "until_m": 1.5, "t_until": 1e12}
        r = SF.run(w)                           # sim resets travelled_m to 0
        self.assertTrue(r.reached)              # after 1.5 m east it goes for the goal
        self.assertIsNone(R._AIM_OVERRIDE)


class TestParseAction(unittest.TestCase):
    def test_valid_drive(self):
        from erc_urban.driver_llm import parse_action
        a = parse_action({"action": "drive", "candidate": 2, "turn_deg": 0,
                          "distance_m": 4, "reason": "path bends left"})
        self.assertEqual((a.action, a.candidate, a.distance_m), ("drive", 2, 4.0))

    def test_clamps_and_defaults(self):
        from erc_urban.driver_llm import parse_action
        a = parse_action({"action": "drive", "candidate": 12, "distance_m": 40})
        self.assertEqual((a.candidate, a.distance_m), (4, 6.0))
        self.assertEqual(parse_action({"action": "turn", "turn_deg": 400}).turn_deg, 180)
        self.assertEqual(parse_action({"action": "turn", "turn_side": "right", "turn_deg": 45}).turn_deg, -45)
        self.assertEqual(parse_action({"action": "turn", "turn_side": "left", "turn_deg": -60}).turn_deg, 60)

    def test_garbage_is_auto(self):
        from erc_urban.driver_llm import parse_action
        self.assertEqual(parse_action("sure! I think left").action, "auto")
        self.assertTrue(parse_action({"action": "fly"}).fallback)
        a = parse_action('text {"action": "back", "distance_m": 1.0, "reason": "wall"} more')
        self.assertEqual(a.action, "back")


class TestTakeover(unittest.TestCase):
    def test_takeover_cancels_escape(self):
        w = SF.World([((-3, 1.0), (3, 1.0))], (0, 0, 0), (0, 9), 30, "wall 1 m ahead")
        R._EVENTS.clear()
        SF.run(w)                                  # hits the wall -> escapes
        R._AGENT_TAKEOVER = True
        SF.run(SF.World([], (0, 0, 0), (0, 9), 3, "open"))
        self.assertFalse(R._AGENT_TAKEOVER)       # consumed on the next tick


if __name__ == "__main__":
    unittest.main()
