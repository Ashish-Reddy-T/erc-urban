"""
End to end: the finals command itself, `python -m erc_urban.run` with the
Panama settings, against a fake SDK server (test/fake_sdk.py). Loads the real
models, so it takes a few minutes; runs only with RUN_E2E=1. Use rover_env's
python (the one the rover runs with).

    RUN_E2E=1 python -m unittest test.test_e2e_runner -v
"""
import os
import subprocess
import sys
import tempfile
import time
import unittest

from test import fake_sdk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CPS = [(0.0, 3.0), (0.0, 28.0), (20.0, 28.0)]      # east, north (m) from the start
PANAMA = {"FAN_TERRAIN": "offroad", "HEADING_K_COMPASS": "0", "GYRO_AXIS": "z", "STAGE": "1",
          "NAV": "fan", "USE_MISSION": "1", "V_NOM": "0.28"}


@unittest.skipUnless(os.environ.get("RUN_E2E") == "1", "set RUN_E2E=1 (loads models, ~3 min)")
class TestRunnerEndToEnd(unittest.TestCase):
    def test_panama_command_with_claim_rejections(self):
        """The server accepts claims only within 11 m while we claim at 15 m (its GPS
        disagrees with ours): rejected claims must not park the rover at the edge."""
        rover = fake_sdk.FakeRover(CPS, gyro_axis="z", front_jpeg=fake_sdk.front_fixture(),
                                   claim_radius=11.0)
        port = 8781
        srv = fake_sdk.serve(rover, port)
        env = dict(os.environ)
        for k in ("GYRO_AXIS", "HEADING_K_COMPASS", "V_NOM", "FAN_TERRAIN", "CLAIM_NEAR_M"):
            env.pop(k, None)
        tmp = tempfile.mkdtemp()
        env.update(PANAMA)
        env.update({"SDK_URL": f"http://127.0.0.1:{port}", "SDK_MANAGE": "0", "OVERLAY_PORT": "8191",
                    "HEADING_STATE": os.path.join(tmp, "heading.json"), "PYTHONUNBUFFERED": "1",
                    "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-m", "erc_urban.run"], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=480)
        finally:
            srv.shutdown()
            rover._running = False
        out = p.stdout + p.stderr
        with open(os.path.join(tmp, "run_out.log"), "w") as fh:
            fh.write(out)
        self.assertNotIn("Traceback", out, f"see {tmp}/run_out.log")
        self.assertIn("[Semantics] SegFormer on", out)
        self.assertIn("overlays + raw frames saved to", out)
        self.assertEqual(rover.latest, len(CPS), f"claimed {rover.latest}/{len(CPS)}; see {tmp}")
        self.assertIn("done — all goals reached", out)
        self.assertIn("checkpoint-reached rejected", out)
        for seq, d in rover.claims:
            self.assertLessEqual(d, 11.0)
        self.assertLessEqual(rover.max_ang_cmd, 1.0)
        print(f"\n[e2e] runner (panama settings): {rover.claims} in {time.time() - t0:.0f}s, "
              f"path {rover.path_m:.0f} m; log {tmp}/run_out.log")


    def test_mexico_lane_runner(self):
        """erc_urban.run_mexico on the Mexico geometry, starting facing away: one
        committed turn, then the lane; all 4 checkpoints, no reverse-first."""
        cps = [(-8.0, 1.0), (23.0, 36.0), (49.0, 73.0), (85.0, 116.0)]
        rover = fake_sdk.FakeRover(cps, gyro_axis="z", start_yaw=175.0,
                                   front_jpeg=fake_sdk.front_fixture())
        port = 8782
        srv = fake_sdk.serve(rover, port)
        env = dict(os.environ, SDK_URL=f"http://127.0.0.1:{port}", OVERLAY_PORT="8192",
                   PYTHONUNBUFFERED="1", PYTORCH_ENABLE_MPS_FALLBACK="1")
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, "-m", "erc_urban.run_mexico"], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=900)
        finally:
            srv.shutdown()
            rover._running = False
        out = p.stdout + p.stderr
        self.assertNotIn("Traceback", out)
        self.assertEqual(rover.latest, 4, f"claimed {rover.latest}/4")
        self.assertIn("done - all checkpoints claimed", out)
        self.assertIn("TURN left to lane", out)
        print(f"\n[e2e] run_mexico: {rover.claims} in {time.time() - t0:.0f}s, path {rover.path_m:.0f} m")


if __name__ == "__main__":
    unittest.main()
