"""agent.driver_loop against a fake runner: actions execute, results journal, no crash."""
import threading
import time
import types
import unittest

import numpy as np

from erc_urban import agent
from erc_urban.driver_llm import Action
from erc_urban.modules import vlm_nav as V


class FakeFS:
    class _D:
        ahead_m = 5.0
        def clear_at(self, deg):
            return 5.0
    drive = _D()


def fake_runner():
    import erc_urban.run as real
    R = types.SimpleNamespace()
    R.CALIB = real.CALIB
    R._OSM = {"payload": None}
    R._MISSION_INFO = {"route": [(0.0, 0.0), (0.0002, 0.0)], "seqs": [1, 2]}
    R._EVENTS = []
    R._AIM_OVERRIDE = None
    R._AGENT_BACK = 0.0
    R._AGENT_TAKEOVER = False
    R._OBS = {"t": time.monotonic(), "front": np.zeros((576, 1024, 3), np.uint8), "fs": FakeFS(),
              "yaw": 0.0, "want_gps": 20.0, "rng": 22.0, "goal_index": 1, "lat": 0.0, "lon": 0.0,
              "tilt": 2.0, "escaping": False, "travelled": 0.0}
    return R


class ScriptDriver:
    def __init__(self, acts):
        self.acts = list(acts)
        self.errors = 0
        self.seen = []

    def decide(self, images, text):
        self.seen.append(text)
        return self.acts.pop(0) if self.acts else Action("auto", distance_m=2.0)


class TestDriverLoop(unittest.TestCase):
    def test_actions_execute_and_journal(self):
        R = fake_runner()
        drv = ScriptDriver([Action("drive", candidate=2, distance_m=1.0, reason="left path"),
                            Action("turn", turn_deg=-90, reason="look right"),
                            Action("auto", distance_m=2.0, reason="open")])
        stop = threading.Event()
        agent.ACTION_TIMEOUT_S = {"auto": 2.0, "drive": 2.0, "turn": 2.0, "back": 2.0}

        def world():                      # the "planner": moves while commanded
            while not stop.is_set():
                R._OBS["t"] = time.monotonic()
                R._OBS["travelled"] += 0.2
                if R._AIM_OVERRIDE and R._OBS["travelled"] >= R._AIM_OVERRIDE["until_m"]:
                    R._AIM_OVERRIDE = None
                time.sleep(0.05)
        threading.Thread(target=world, daemon=True).start()
        th = threading.Thread(target=agent.driver_loop, args=(R, V, drv, stop), daemon=True)
        th.start()
        time.sleep(6.0)
        stop.set()
        th.join(3.0)
        self.assertGreaterEqual(len(drv.seen), 3)
        self.assertIn("Your recent actions", drv.seen[2])    # journal fed back
        self.assertIn("drive #2", drv.seen[2])

    def test_long_escape_triggers_takeover(self):
        R = fake_runner()
        R._OBS["escaping"] = True
        drv = ScriptDriver([Action("back", distance_m=1.0, reason="boxed")])
        stop = threading.Event()
        agent.ESCAPE_PATIENCE_S = 0.5
        th = threading.Thread(target=agent.driver_loop, args=(R, V, drv, stop), daemon=True)

        def tick():
            while not stop.is_set():
                R._OBS["t"] = time.monotonic()
                time.sleep(0.05)
        threading.Thread(target=tick, daemon=True).start()
        th.start()
        time.sleep(2.5)
        stop.set()
        th.join(3.0)
        self.assertTrue(drv.seen)                  # it decided despite the escape
        self.assertTrue(R._AGENT_TAKEOVER)          # escape handed to the driver
        self.assertGreater(R._AGENT_BACK, 0.0)       # and its "back" was queued


if __name__ == "__main__":
    unittest.main()
