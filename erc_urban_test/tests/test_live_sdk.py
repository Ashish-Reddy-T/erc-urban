"""
Optional live SDK smoke test. Skipped unless LIVE_SDK=1 and server is up.

    LIVE_SDK=1 python -m unittest test.test_live_sdk -v
"""
from __future__ import annotations
import os
import unittest

import requests

LIVE = os.environ.get("LIVE_SDK", "0") == "1"
BASE = os.environ.get("SDK_URL", "http://localhost:8000")


@unittest.skipUnless(LIVE, "set LIVE_SDK=1 to hit a running earth-rovers-sdk")
class TestLiveSdk(unittest.TestCase):
    def test_status(self):
        r = requests.get(f"{BASE}/status", timeout=5)
        r.raise_for_status()
        body = r.json()
        self.assertIn("browser_ready", body)

    def test_data_has_expected_keys(self):
        from erc_urban_test.sdk_client import RoverClient

        tel = RoverClient(BASE).get_data()
        # Soft asserts: bot may be offline; just ensure parse didn't crash.
        self.assertTrue(hasattr(tel, "lat"))
        self.assertTrue(hasattr(tel, "yaw"))
        print("live telemetry:", tel.lat, tel.lon, tel.yaw, list(tel.raw)[:8])


if __name__ == "__main__":
    unittest.main()
