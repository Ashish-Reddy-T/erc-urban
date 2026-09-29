"""SdkGuard: classify SDK health, and recover without restarting needlessly."""
from __future__ import annotations
import unittest
from unittest.mock import MagicMock, patch

import requests

from erc_urban.sdk_guard import SdkGuard


def _resp(code, body=None, text=""):
    r = MagicMock(status_code=code, text=text)
    if isinstance(body, Exception):
        r.json.side_effect = body
    else:
        r.json.return_value = body
    return r


class TestStatus(unittest.TestCase):
    def setUp(self):
        self.g = SdkGuard("http://localhost:8000", manage=False)

    @patch("erc_urban.sdk_guard.requests.get")
    def test_telemetry_is_ok(self, get):
        get.return_value = _resp(200, {"latitude": 30.4, "rpms": []})
        self.assertEqual(self.g.status(), "ok")

    @patch("erc_urban.sdk_guard.requests.get")
    def test_null_payload_is_a_dead_session(self, get):
        """Wuhan: /data hung ~23 s and returned null while video still worked."""
        get.return_value = _resp(200, None, "null")
        self.assertEqual(self.g.status(), "null")

    @patch("erc_urban.sdk_guard.requests.get")
    def test_waiting_for_mission(self, get):
        get.return_value = _resp(400, None,
                                 '{"detail":"Call /start-mission endpoint to start a mission"}')
        self.assertEqual(self.g.status(), "mission")

    @patch("erc_urban.sdk_guard.requests.get")
    def test_hang_is_down(self, get):
        get.side_effect = requests.Timeout()
        self.assertEqual(self.g.status(), "down")

    def test_port_parsed_from_url(self):
        self.assertEqual(SdkGuard("http://127.0.0.1:8123").port, 8123)

class TestRecover(unittest.TestCase):
    def _guard(self, statuses):
        g = SdkGuard("http://localhost:8000", manage=True)
        g.status = MagicMock(side_effect=list(statuses))
        g.restart = MagicMock(return_value=True)
        return g

    @patch("erc_urban.sdk_guard.time.sleep")
    def test_mission_gate_gets_start_mission_not_a_restart(self, _):
        g = self._guard(["mission", "ok"])
        client = MagicMock()
        client.start_mission.return_value = {"message": "Mission started successfully"}
        self.assertTrue(g.recover(client, mission_mode=True))
        client.start_mission.assert_called_once()
        g.restart.assert_not_called()

    @patch("erc_urban.sdk_guard.time.sleep")
    def test_dead_sdk_is_restarted_then_mission_refreshed(self, _):
        g = self._guard(["null", "mission", "ok"])
        client = MagicMock()
        client.start_mission.return_value = {"message": "ok"}
        self.assertTrue(g.recover(client, mission_mode=True))
        g.restart.assert_called_once()
        client.start_mission.assert_called_once()

    @patch("erc_urban.sdk_guard.time.sleep")
    def test_free_drive_never_starts_a_mission(self, _):
        g = self._guard(["down", "ok"])
        client = MagicMock()
        self.assertTrue(g.recover(client, mission_mode=False))
        client.start_mission.assert_not_called()

    def test_unmanaged_guard_never_restarts(self):
        g = SdkGuard("http://localhost:8000", manage=False)
        self.assertFalse(g.restart())


if __name__ == "__main__":
    unittest.main()
