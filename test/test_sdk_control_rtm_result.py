"""POST /control must not claim success when the RTM send fails."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

# earth-rovers-sdk/ is the import root for main.py / browser_service.py
REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = REPO_ROOT / "earth-rovers-sdk"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

# Avoid needing a real .env / Chrome for import-time side effects.
os.environ.setdefault("BOT_SLUG", "test-bot")
os.environ.setdefault("MISSION_SLUG", "test-mission")
os.environ.setdefault("SDK_API_TOKEN", "test-token")

try:
    from fastapi.testclient import TestClient
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@unittest.skipUnless(_HAS_FASTAPI, "fastapi not installed in this env")
class TestControlRtmResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # main.py mounts ./static relative to CWD
        cls._prev_cwd = os.getcwd()
        os.chdir(SDK_ROOT)
        import main as sdk_main

        cls.sdk_main = sdk_main
        cls.client = TestClient(sdk_main.app)

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._prev_cwd)

    def test_control_returns_503_when_send_message_fails(self):
        """Bug on main: /control ignores send_message return and always 200."""
        with patch.object(self.sdk_main, "need_start_mission", new=AsyncMock()), \
                patch.object(self.sdk_main, "auth_response_data", {"ok": True}), \
                patch.object(
                    self.sdk_main.browser_service,
                    "send_message",
                    new=AsyncMock(return_value=False),
                ):
            resp = self.client.post(
                "/control",
                json={"command": {"linear": 0.2, "angular": 0.0}},
            )
        self.assertEqual(resp.status_code, 503, resp.text)
        self.assertIn("RTM", resp.json()["detail"])

    def test_control_returns_200_when_send_message_ok(self):
        with patch.object(self.sdk_main, "need_start_mission", new=AsyncMock()), \
                patch.object(self.sdk_main, "auth_response_data", {"ok": True}), \
                patch.object(
                    self.sdk_main.browser_service,
                    "send_message",
                    new=AsyncMock(return_value=True),
                ):
            resp = self.client.post(
                "/control",
                json={"command": {"linear": 0.2, "angular": 0.0}},
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertIn("success", resp.json()["message"].lower())


if __name__ == "__main__":
    unittest.main()
