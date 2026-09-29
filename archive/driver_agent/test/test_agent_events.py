"""The driver must see new runner events after the runner's list is trimmed to 50."""
import unittest

import erc_urban.run as R
from erc_urban import agent


class TestEventsPastCap(unittest.TestCase):
    def test_new_events_seen_after_60(self):
        R._EVENTS.clear()
        for k in range(60):
            R._event(f"escape {k}")
        self.assertEqual(len(R._EVENTS), 50)
        seq = agent._ev_seq(R)
        self.assertEqual(agent._events_since(R, seq), [])
        R._event("claimed checkpoint (within 15.0 m)")
        self.assertGreater(agent._ev_seq(R), seq)            # len() would not have grown
        self.assertEqual(agent._events_since(R, seq), ["claimed checkpoint (within 15.0 m)"])
        self.assertEqual(agent._events_since(R, seq - 2)[-1], "claimed checkpoint (within 15.0 m)")
        self.assertEqual(len(agent._events_since(R, seq - 2)), 3)


if __name__ == "__main__":
    unittest.main()
