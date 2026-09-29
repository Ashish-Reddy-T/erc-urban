"""
End to end: the real `python -m erc_urban.agent` against a fake SDK server
(test/fake_sdk.py) playing different bots. Loads the real models, so it takes
a few minutes per scenario; runs only with RUN_E2E=1.

    RUN_E2E=1 python -m unittest test.test_e2e_agent -v
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


def run_agent(rover, port, site, script, timeout_s=420, extra_env=None):
    srv = fake_sdk.serve(rover, port)
    tmp = tempfile.mkdtemp()
    env = dict(os.environ)
    for k in ("GYRO_AXIS", "HEADING_K_COMPASS", "V_NOM", "FAN_TERRAIN", "CLAIM_NEAR_M"):
        env.pop(k, None)
    env.update({
        "SITE": site, "SDK_URL": f"http://127.0.0.1:{port}", "SDK_MANAGE": "0", "OVERLAY": "0",
        "BOT_CHECK_FIX": "0", "DRIVER_BACKEND": "fake", "DRIVER_FAKE_SCRIPT": script,
        "HEADING_STATE": os.path.join(tmp, "heading.json"), "TICK_LOG": os.path.join(tmp, "ticks.log"),
        "PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    })
    env.update(extra_env or {})
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "erc_urban.agent"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout_s)
    srv.shutdown()
    rover._running = False
    out = p.stdout + p.stderr
    with open(os.path.join(tmp, "agent_out.log"), "w") as fh:
        fh.write(out)
    return p.returncode, out, time.time() - t0, tmp


@unittest.skipUnless(os.environ.get("RUN_E2E") == "1", "set RUN_E2E=1 (loads models, ~3 min each)")
class TestAgentEndToEnd(unittest.TestCase):
    def _check_common(self, rc, out, rover, tmp):
        self.assertNotIn("Traceback", out, f"see {tmp}/agent_out.log")
        self.assertEqual(rover.latest, len(CPS), f"claimed {rover.latest}/{len(CPS)}; see {tmp}")
        self.assertIn("done — all goals reached", out)
        self.assertIn("[driver] step", out)
        for seq, d in rover.claims:
            self.assertLessEqual(d, 15.0)

    def test_mini_plus_good_compass(self):
        """Mini+ (gyro z), honest compass: checks pick z + compass on; mission 3/3."""
        rover = fake_sdk.FakeRover(CPS, gyro_axis="z", front_jpeg=fake_sdk.front_fixture())
        rc, out, dt, tmp = run_agent(rover, 8771, "wuhan", "auto:6,drive:3:3,auto:8,turn:left:45,auto:8")
        self._check_common(rc, out, rover, tmp)
        self.assertIn("gyro: yaw = z", out)
        self.assertIn("compass: ", out)
        self.assertIn("USED", out)
        print(f"\n[e2e] mini+ good compass: {rover.claims} in {dt:.0f}s, path {rover.path_m:.0f} m")

    def test_zero_axis_bad_compass_flaky_model(self):
        """Yaw on gyro x, compass 80 deg off, model errors and junk replies:
        checks pick x + compass off, bootstrap drive, fallbacks to auto; mission 3/3."""
        rover = fake_sdk.FakeRover(CPS, gyro_axis="x", compass_bias=80.0, compass_scale=0.8,
                                   front_jpeg=fake_sdk.front_fixture())
        rc, out, dt, tmp = run_agent(rover, 8772, "unlp", "error,junk,auto:6,error,drive:5:3,auto:8,back:1,auto:8")
        self._check_common(rc, out, rover, tmp)
        self.assertIn("gyro: yaw = x", out)
        self.assertIn("ignored", out)                     # compass rejected
        self.assertIn("model error", out)                 # error surfaced, fell back
        print(f"\n[e2e] zero-axis bad compass: {rover.claims} in {dt:.0f}s, path {rover.path_m:.0f} m")

    def test_push_override(self):
        """Model pushes through (camera stop ignored); pushes move the rover and end
        cleanly, planner resumes; mission 3/3 on the offroad (panama) profile."""
        # The server only accepts within 11 m while we claim at 15: claims are
        # rejected at the edge and the rover must keep driving in, not park.
        rover = fake_sdk.FakeRover(CPS, gyro_axis="z", front_jpeg=fake_sdk.front_fixture(),
                                   claim_radius=11.0)
        rc, out, dt, tmp = run_agent(rover, 8773, "panama",
                                     "push:4:3,auto:6,push:3:2,auto:8,push:4:3,auto:8")
        self._check_common(rc, out, rover, tmp)
        self.assertIn("-> PUSH #4", out)
        self.assertIn("result: done", out)
        ticks = open(os.path.join(tmp, "ticks.log")).read()
        self.assertIn("PUSH", ticks)
        self.assertIn("claim rejected", out)
        for seq, d in rover.claims:
            self.assertLessEqual(d, 11.0)
        print(f"\n[e2e] push: {rover.claims} in {dt:.0f}s, path {rover.path_m:.0f} m")


if __name__ == "__main__":
    unittest.main()
