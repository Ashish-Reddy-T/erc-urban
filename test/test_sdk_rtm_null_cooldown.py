"""GET /data must not close headless Chrome on every null rtm_data poll."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = REPO_ROOT / "earth-rovers-sdk"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

os.environ.setdefault("BOT_SLUG", "test-bot")
os.environ.setdefault("MISSION_SLUG", "test-mission")
os.environ.setdefault("SDK_API_TOKEN", "test-token")

try:
    from fastapi.testclient import TestClient
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False


@unittest.skipUnless(_HAS_FASTAPI, "fastapi not installed in this env")
class TestRtmNullCooldown(unittest.TestCase):
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

    def _poll(self, n: int, close: AsyncMock) -> list[int]:
        bs = self.sdk_main.browser_service
        with patch.object(self.sdk_main, "need_start_mission", new=AsyncMock()), \
                patch.object(self.sdk_main, "_rtm_last_reinit", 0.0), \
                patch.object(bs, "data", new=AsyncMock(return_value=None)), \
                patch.object(bs, "rtm_status", new=AsyncMock(return_value={})), \
                patch.object(bs, "close_browser", new=close):
            return [self.client.get("/data").status_code for _ in range(n)]

    def test_repeated_null_closes_browser_once_within_cooldown(self):
        """Bug on main: every null poll closes Chrome (no cooldown)."""
        close = AsyncMock()
        codes = self._poll(5, close)
        self.assertEqual(codes, [200] * 5)
        self.assertEqual(close.await_count, 1)

    def test_null_after_cooldown_closes_again(self):
        close = AsyncMock()
        with patch.object(self.sdk_main, "RTM_REINIT_COOLDOWN_S", 0.0):
            self._poll(3, close)
        self.assertEqual(close.await_count, 3)


if __name__ == "__main__":
    unittest.main()
