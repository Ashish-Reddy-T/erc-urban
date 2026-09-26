"""Tests for RoverClient parsing (mocked HTTP — no live SDK)."""
from __future__ import annotations
import base64
import io
import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from erc_urban.sdk_client import RoverClient


def _png_b64(color=(255, 0, 0), size=(4, 4)) -> str:
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestSdkClientDefaults(unittest.TestCase):
    def test_default_http_timeout_is_short(self):
        """Hung HTTP must not block a whole urban control tick (indoor uses ~3s)."""
        client = RoverClient("http://localhost:8000")
        self.assertLessEqual(client.timeout, 5.0)
        self.assertEqual(client.timeout, 3.0)

    def test_constructor_timeout_override(self):
        client = RoverClient("http://localhost:8000", timeout=1.0)
        self.assertEqual(client.timeout, 1.0)


class TestSdkClientParse(unittest.TestCase):
    def setUp(self):
        self.client = RoverClient("http://localhost:8000", timeout=1.0)

    @patch("erc_urban.sdk_client.requests.get")
    def test_orientation_degrees_to_radians(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "latitude": 30.48,
                "longitude": 114.30,
                "orientation": 90,  # degrees east
            },
        )
        mock_get.return_value.raise_for_status = lambda: None
        tel = self.client.get_data()
        self.assertAlmostEqual(tel.lat, 30.48)
        self.assertAlmostEqual(tel.lon, 114.30)
        self.assertIsNotNone(tel.yaw)
        self.assertAlmostEqual(tel.yaw, math.pi / 2, places=5)

    @patch("erc_urban.sdk_client.requests.get")
    def test_missing_gps_becomes_nan(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {})
        mock_get.return_value.raise_for_status = lambda: None
        tel = self.client.get_data()
        self.assertTrue(np.isnan(tel.lat))
        self.assertTrue(np.isnan(tel.lon))
        self.assertIsNone(tel.yaw)

    @patch("erc_urban.sdk_client.requests.get")
    def test_get_frames_decodes(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "front_frame": _png_b64((0, 255, 0)),
                "rear_frame": _png_b64((0, 0, 255)),
            },
        )
        mock_get.return_value.raise_for_status = lambda: None
        front, rear = self.client.get_frames()
        self.assertIsNotNone(front)
        self.assertIsNotNone(rear)
        self.assertEqual(front.shape[2], 3)

    @patch("erc_urban.sdk_client.requests.get")
    def test_checkpoints_parse_string_coords(self, mock_get):
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "checkpoints_list": [
                    {"id": 2, "sequence": 2, "latitude": "30.1", "longitude": "114.2"},
                    {"id": 1, "sequence": 1, "latitude": "30.0", "longitude": "114.1"},
                ],
                "latest_scanned_checkpoint": 0,
            },
        )
        mock_get.return_value.raise_for_status = lambda: None
        cps, latest = self.client.get_checkpoints()
        self.assertEqual(latest, 0)
        self.assertEqual(len(cps), 2)
        # sorted by sequence
        self.assertEqual(cps[0].sequence, 1)
        self.assertAlmostEqual(cps[0].lat, 30.0)
        self.assertAlmostEqual(cps[1].lon, 114.2)

    @patch("erc_urban.sdk_client.requests.post")
    def test_send_cmd_clamps(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)
        self.client.send_cmd(2.0, -3.0)
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(body["command"]["linear"], 1.0)
        self.assertEqual(body["command"]["angular"], -1.0)

    @patch("erc_urban.sdk_client.requests.post")
    def test_checkpoint_reached_keeps_400_body(self, mock_post):
        mock_post.return_value = MagicMock(
            status_code=400,
            json=lambda: {
                "detail": {
                    "error": "Bot is not within XX meters from the checkpoint",
                    "proximate_distance_to_checkpoint": 16.87,
                }
            },
        )
        body = self.client.checkpoint_reached()
        self.assertEqual(body["_http_status"], 400)
        self.assertIn("detail", body)


if __name__ == "__main__":
    unittest.main()
