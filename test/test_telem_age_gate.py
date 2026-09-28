"""step() must hold still on stale telemetry or a dead RTM session."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import erc_urban.run as run
from erc_urban.sdk_client import Telemetry


class _DroveOn(Exception):
    """Raised when step() gets past the telemetry checks into perception."""


def _client(raw: dict) -> MagicMock:
    client = MagicMock()
    client.get_frames.return_value = (np.zeros((8, 8, 3), np.uint8), None)
    client.get_data.return_value = Telemetry(
        lat=40.0, lon=-73.0, yaw=0.0, stamp=0.0, raw=raw)
    return client


def _step(client):
    return run.step(client, MagicMock(), None, None, None, MagicMock(),
                    (40.001, -73.0), False, {})


class TestTelemAgeGate(unittest.TestCase):
    def test_stale_rtm_age_halts_before_perception(self):
        client = _client({"rtm_age_s": 10.0, "rtm_state": "CONNECTED"})
        with patch.object(run, "_vo", side_effect=_DroveOn):
            self.assertFalse(_step(client))
        client.send_cmd.assert_called_once_with(0.0, 0.0, lamp=run.LAMP)

    def test_aborted_session_halts_before_perception(self):
        client = _client({"rtm_age_s": 0.5, "rtm_state": "ABORTED",
                          "rtm_reason": "REMOTE_LOGIN"})
        with patch.object(run, "_vo", side_effect=_DroveOn):
            self.assertFalse(_step(client))
        client.send_cmd.assert_called_once_with(0.0, 0.0, lamp=run.LAMP)

    def test_fresh_telemetry_proceeds(self):
        client = _client({"rtm_age_s": 1.0, "rtm_state": "CONNECTED"})
        with patch.object(run, "_vo", side_effect=_DroveOn):
            with self.assertRaises(_DroveOn):
                _step(client)

    def test_old_sdk_without_rtm_keys_proceeds(self):
        client = _client({})
        with patch.object(run, "_vo", side_effect=_DroveOn):
            with self.assertRaises(_DroveOn):
                _step(client)

    def test_bad_age_value_is_ignored(self):
        for age in (None, "x", float("nan")):
            self.assertIsNone(run._telemetry_dead(
                Telemetry(40.0, -73.0, 0.0, 0.0, {"rtm_age_s": age})))


if __name__ == "__main__":
    unittest.main()
