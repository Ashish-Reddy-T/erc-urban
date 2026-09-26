"""SEND_CONTROL=0 must turn erc_urban_test.run into a no-motion dry run."""
from __future__ import annotations
import importlib
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import erc_urban_test.run as run


class TestSendControlGuard(unittest.TestCase):
    def test_env_zero_disables_control(self):
        with patch.dict(os.environ, {"SEND_CONTROL": "0"}):
            mod = importlib.reload(run)
        try:
            self.assertFalse(mod.SEND_CONTROL)
        finally:
            importlib.reload(run)

    def test_default_enables_control(self):
        env = {k: v for k, v in os.environ.items() if k != "SEND_CONTROL"}
        with patch.dict(os.environ, env, clear=True):
            mod = importlib.reload(run)
        try:
            self.assertTrue(mod.SEND_CONTROL)
        finally:
            importlib.reload(run)

    def test_dry_run_cmd_and_halt_send_nothing(self):
        client = MagicMock()
        with patch.object(run, "SEND_CONTROL", False):
            run._cmd(client, 0.3, 0.5)
            run._halt(client)
        client.send_cmd.assert_not_called()

    def test_live_cmd_and_halt_send(self):
        client = MagicMock()
        with patch.object(run, "SEND_CONTROL", True):
            run._cmd(client, 0.3, 0.5)
            run._halt(client)
        self.assertEqual(client.send_cmd.call_count, 2)
        client.send_cmd.assert_any_call(0.3, 0.5, lamp=run.LAMP)
        client.send_cmd.assert_any_call(0.0, 0.0, lamp=run.LAMP)

    def test_dry_run_skips_mission_posts(self):
        client = MagicMock()
        with patch.object(run, "SEND_CONTROL", False):
            started = run._start_mission(client)
            claimed = run._claim(client)
        client.start_mission.assert_not_called()
        client.checkpoint_reached.assert_not_called()
        self.assertIn("dry run", started["message"])
        self.assertIn("dry run", claimed["message"])

    def test_live_mission_posts_go_through(self):
        client = MagicMock()
        client.start_mission.return_value = {"message": "ok"}
        client.checkpoint_reached.return_value = {"_http_status": 200}
        with patch.object(run, "SEND_CONTROL", True):
            self.assertEqual(run._start_mission(client), {"message": "ok"})
            self.assertEqual(run._claim(client), {"_http_status": 200})
        client.start_mission.assert_called_once_with()
        client.checkpoint_reached.assert_called_once_with()

    def test_mission_mode_dry_run_reads_checkpoints_without_starting(self):
        cp = SimpleNamespace(sequence=1, id="cp1", lat=40.0, lon=-73.0)
        client = MagicMock()
        client.get_checkpoints.return_value = ([cp], 0)
        env = {k: v for k, v in os.environ.items() if k != "SKIP_START_MISSION"}
        with patch.dict(os.environ, env, clear=True), \
                patch.object(run, "USE_MISSION", True), \
                patch.object(run, "SEND_CONTROL", False):
            goals, mission_mode, _, _ = run.resolve_goals(client)
        client.start_mission.assert_not_called()
        client.get_checkpoints.assert_called_once_with()
        self.assertTrue(mission_mode)
        self.assertEqual(goals, [(40.0, -73.0)])


if __name__ == "__main__":
    unittest.main()
