"""SEND_CONTROL=0 must turn erc_urban.run into a no-motion dry run."""
from __future__ import annotations
import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

import erc_urban.run as run


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
        self.assertTrue(mod.SEND_CONTROL)

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


if __name__ == "__main__":
    unittest.main()
