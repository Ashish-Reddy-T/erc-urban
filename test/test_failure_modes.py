"""
One test per failure mode met on the practice bots (2026-09-22..25), so none
comes back unnoticed. Each docstring names where it happened.
"""
import io
import contextlib
import math
import os
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

import erc_urban.run as R
import erc_urban.sim_fan as SF
from erc_urban.modules.freespace import FreeSpaceUrban


def tel(acc):
    return types.SimpleNamespace(raw={"accels": [list(a) + [1.0] for a in acc]})


# --------------------------------------------------------------------------- tilt
class TestTilt(unittest.TestCase):
    def setUp(self):
        self.g0 = R._GRAVITY
        R._GRAVITY = np.array([0.0, 0.0, 1.0])

    def tearDown(self):
        R._GRAVITY = self.g0

    def test_imu_glitch_162_deg_is_ignored(self):
        """Costa Rica: one burst read 162 deg (upside down) with the rover level."""
        self.assertIsNone(R._tilt_deg(tel([[0.3, 0.0, -0.95]])))

    def test_knock_outside_one_g_is_ignored(self):
        self.assertIsNone(R._tilt_deg(tel([[0.1, 0.0, 0.4]])))
        self.assertIsNone(R._tilt_deg(tel([[0.9, 0.0, 1.6]])))

    def test_real_climb_reads(self):
        """Machakos bank: 25-37 deg while climbing."""
        t = R._tilt_deg(tel([[0.5, 0.0, 0.87]]))
        self.assertAlmostEqual(t, 30.0, delta=1.5)

    def test_whole_burst_is_averaged(self):
        """Machakos: single pivot-jolt samples read 30 deg on level ground."""
        t = R._tilt_deg(tel([[0.0, 0.0, 1.0]] * 4 + [[0.5, 0.0, 0.87]]))
        self.assertLess(t, 8.0)

    def test_gravity_snaps_to_axis(self):
        """Costa Rica: started on a 9 deg driveway; tilt must be from vertical."""
        np.testing.assert_allclose(R._snap_gravity([-0.16, 0.03, 0.99]), [0, 0, 1])
        np.testing.assert_allclose(R._snap_gravity([0.99, -0.01, 0.04]), [1, 0, 0])   # Zero
        g = R._snap_gravity([0.6, 0.0, 0.8])                                          # >30 deg
        self.assertAlmostEqual(float(np.linalg.norm(g)), 1.0)
        self.assertGreater(abs(g[0]), 0.1)

    def test_debounce_needs_two_ticks(self):
        fan = {}
        with patch.object(R, "_tilt_deg", side_effect=[30.0, 3.0, 30.0, 31.0]):
            self.assertIsNone(R._tilt_held(fan, None))       # no previous yet
            self.assertEqual(R._tilt_held(fan, None), 3.0)   # min(30, 3)
            self.assertEqual(R._tilt_held(fan, None), 3.0)
            self.assertEqual(R._tilt_held(fan, None), 30.0)  # held twice


class TestSlopeMode(unittest.TestCase):
    """Costa Rica finals: a steady 25-30 deg grass slope must be climbed with
    sustained power, not escaped as 'tilted'; roll and blocked views still escape."""

    def setUp(self):
        self.saved = (R._GRAVITY, R._LAST_ACC.copy())
        R._GRAVITY = np.array([0.0, 0.0, 1.0])

    def tearDown(self):
        R._GRAVITY, R._LAST_ACC[:] = self.saved[0], self.saved[1]

    def _fs(self, ahead):
        return types.SimpleNamespace(drive=types.SimpleNamespace(ahead_m=ahead))

    def _acc(self, pitch, roll):
        p, r = np.radians(pitch), np.radians(roll)
        R._LAST_ACC[:] = [np.sin(r) * np.cos(p), np.sin(p), np.cos(p) * np.cos(r)]

    def test_pitch_roll(self):
        self._acc(30, 0)
        p, r = R._pitch_roll()
        self.assertAlmostEqual(p, 30, delta=1); self.assertAlmostEqual(r, 0, delta=1)
        self._acc(0, 20)
        p, r = R._pitch_roll()
        self.assertAlmostEqual(p, 0, delta=1); self.assertAlmostEqual(r, 20, delta=1)

    def test_slope_up_with_ground_ahead(self):
        self._acc(28, 3)
        self.assertAlmostEqual(R._slope({"tilt_now": 28.0}, self._fs(4.0)), 28, delta=1)

    def test_not_a_slope(self):
        self._acc(28, 3)
        self.assertIsNone(R._slope({"tilt_now": 28.0}, self._fs(0.4)))    # bench leg / wall ahead
        self._acc(20, 18)
        self.assertIsNone(R._slope({"tilt_now": 27.0}, self._fs(4.0)))    # rolling: flip risk
        self._acc(40, 0)
        self.assertIsNone(R._slope({"tilt_now": 40.0}, self._fs(4.0)))    # too steep
        self._acc(5, 0)
        self.assertIsNone(R._slope({"tilt_now": 5.0}, self._fs(4.0)))     # flat


class TestTiltInTheLoop(unittest.TestCase):
    def _run(self, fn, ticks=40):
        w = SF.SCENARIOS["behind"]
        w = SF.World(w.walls, w.start, w.goal, ticks, w.name)
        buf = io.StringIO()
        with patch.object(R, "_tilt_deg", fn), contextlib.redirect_stdout(buf):
            SF.run(w, verbose=True)
        return buf.getvalue()

    def test_uniform_steep_slope_does_not_block_pivots(self):
        """Costa Rica driveway: turning on a steady 18 deg slope is fine."""
        self.assertEqual(self._run(lambda t: 18.0).count("while pivoting"), 0)

    def _slope_run(self, pitch, roll, ticks=40):
        p, r = np.radians(pitch), np.radians(roll)
        acc = np.array([np.sin(r) * np.cos(p), np.sin(p), np.cos(p) * np.cos(r)])
        tilt = float(np.degrees(np.arccos(acc[2])))

        def fn(t):
            R._LAST_ACC[:] = acc
            return tilt
        w = SF.SCENARIOS["open"]
        w = SF.World(w.walls, w.start, w.goal, ticks, w.name)
        saved = R._GRAVITY
        R._GRAVITY = np.array([0.0, 0.0, 1.0])
        buf = io.StringIO()
        try:
            with patch.object(R, "_tilt_deg", fn), contextlib.redirect_stdout(buf):
                SF.run(w, verbose=True)
        finally:
            R._GRAVITY = saved
        return buf.getvalue()

    def test_costa_rica_slope_climbs_with_power(self):
        """Finals: steady 28 deg grass slope, ground ahead -> no tilt escape, lin >= 0.55."""
        out = self._slope_run(28, 3)
        self.assertEqual(out.count("ESCAPE: tilted"), 0)
        self.assertIn("SLOPE +28deg", out)
        lins = [float(x) for x in __import__("re").findall(r"SLOPE \+28deg\S*\] cmd lin=([\d.]+)", out)]
        self.assertTrue(lins and min(lins) >= 0.55, lins[:5])

    def test_downhill_bank_still_escapes(self):
        """Costa Rica: the bank down to the road must never be driven."""
        self.assertGreater(self._slope_run(-28, 2).count("ESCAPE: tilted"), 0)

    def test_rolled_slope_still_escapes(self):
        """Side slope (roll 26 deg) is how rovers flip: never waived."""
        self.assertGreater(self._slope_run(10, 26).count("ESCAPE: tilted"), 0)

    def test_tilt_rising_during_pivot_backs_off(self):
        """Machakos: slid off a raised bed edge mid-pivot and went over."""
        seq = iter([5.0] * 6 + [20.0] * 500)
        self.assertGreater(self._run(lambda t: next(seq)).count("while pivoting"), 0)

    def test_tilt_escape_blocks_wide_sector(self):
        """Machakos: +-15 deg let it retry the same kerb 20 deg round: 15 escapes, 2 flips."""
        fan = R._fan_state({})
        fan["blocked"] = []
        with contextlib.redirect_stdout(io.StringIO()):
            R._fan_begin_escape(fan, "tilted 30 deg (climbing something)", 0.0, force_reverse=True,
                                yaw_deg=100.0, failed_rel_deg=0.0,
                                pts=np.array([[0.25, 0.0]]), sticky=True)
        hs = sorted(round(b[0]) for b in fan["blocked"])
        self.assertIn(100, hs)
        self.assertTrue(any(abs(h - 70) <= 1 for h in hs) and any(abs(h - 130) <= 1 for h in hs))

    def test_no_flip_recovery_code(self):
        """User: flip recovery 'not required' - it never rolled a Mini+ back."""
        for name in ("FAN_FLIP_DEG", "FLIP_TRIES", "FLIP_GAP_S"):
            self.assertFalse(hasattr(R, name), name)


# ----------------------------------------------------------------- segmentation
class _Sem:
    NAMES = ["grass", "earth", "sculpture", "plant", "sky", "wall", "road", "tree", "flower"]

    def __init__(self):
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(
            id2label={i: n for i, n in enumerate(self.NAMES)}))

    def id(self, n):
        return self.NAMES.index(n)


class TestOffroadMasks(unittest.TestCase):
    def setUp(self):
        self.saved = (R.FAN_TERRAIN, R._FAN_IDS)
        R.FAN_TERRAIN, R._FAN_IDS = "offroad", None

    def tearDown(self):
        R.FAN_TERRAIN, R._FAN_IDS = self.saved

    def test_allow_list(self):
        """Costa Rica: a play slide read 'sculpture' and a flower bed 'plant'; both
        were driven onto when offroad meant 'everything but known obstacles'."""
        sem = _Sem()
        lab = np.array([[sem.id(n) for n in _Sem.NAMES]])
        g, _ = R._fan_masks(sem, lab)
        ground = {n for n, ok in zip(_Sem.NAMES, g[0]) if ok}
        self.assertEqual(ground, {"grass", "earth", "road"})
        self.assertTrue(R._FAN_KERB[0][_Sem.NAMES.index("wall")])

    def test_vehicles_never_range_relieved(self):
        """Argentina finals: a parked EarthRover 0.25 m away read 'truck'; range
        relief pushed it to 2 m and the rover clipped it and flipped. Trees still
        get relief (Panama slope tree lines)."""
        sem = _Sem()
        sem.NAMES = _Sem.NAMES + ["truck", "car", "person"]
        sem.model.config.id2label = {i: n for i, n in enumerate(sem.NAMES)}
        lab = np.array([[sem.NAMES.index(n) for n in ("truck", "car", "tree", "plant", "person")]])
        R._fan_masks(sem, lab)
        self.assertEqual(R._FAN_RANGE_RELIEF[0].tolist(), [False, False, True, True, True])


class TestKerbGuard(unittest.TestCase):
    """Machakos: a ~15 cm kerb labelled wall implied 0.10-0.115 m (< 0.12 off-road)
    and the rover drove into it 10+ times."""

    def _scene(self, vertical: bool, hard: bool):
        fsu = FreeSpaceUrban(R.CALIB, seg_hw=(288, 512), min_obstacle_h_m=0.12)
        H, W = fsu.h, fsu.w
        u = W // 2
        gx = fsu._gx[:, u]
        rows = np.arange(H)
        below = rows > fsu.horizon + 2
        r0 = int(np.argmin(np.abs(np.where(below, gx, 99) - 0.40)))     # face bottom 0.40 m
        r1 = int(np.argmin(np.abs(np.where(below, gx, 99) - 1.25)))     # ground again at 1.25 m
        g = np.tile(below[:, None], (1, W))
        g[r1 + 1: r0 + 1, :] = False
        kerb = np.zeros_like(g)
        kerb[r1 + 1: r0 + 1, :] = True
        depth = np.sqrt(np.clip(np.where(below, gx, 20.0), 0.05, 20.0))[:, None].repeat(W, 1)
        if vertical:
            depth[r1 + 1: r0 + 1, :] = depth[r0, u]          # a face: depth stands still
        fs = fsu.estimate(g, None, depth.astype(np.float32), hard=kerb if hard else None)
        return fs.drive.ahead_m

    def test_kerb_face_stops_the_rover(self):
        self.assertLess(self._scene(vertical=True, hard=True), 1.0)

    def test_without_the_guard_it_was_driven_into(self):
        self.assertGreater(self._scene(vertical=True, hard=False), 1.0)

    def test_flat_litter_labelled_wall_still_passes(self):
        """Panama: leaf drifts labelled wall recede like ground: keep driving."""
        self.assertGreater(self._scene(vertical=False, hard=True), 1.0)


# ------------------------------------------------------------------ heading / OSM
class TestBootstrap(unittest.TestCase):
    def test_no_fix_drives_clearest_first(self):
        """Costa Rica: a steel gate bent the compass seed; aiming at CP1 on a guessed
        heading drove into the gate. With the compass off and no GPS fix yet, drive the
        clearest way first."""
        saved = (R._HEADING.k_compass, R._HEADING.fixes)
        R._HEADING.k_compass, R._HEADING.fixes = 0.0, 0
        try:
            w = SF.World([], (0, 0, 0), (-20, 0), 25, "goal west")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                r = SF.run(w, verbose=True)
            self.assertIn("no fix yet", buf.getvalue())
            e, n = r.trail[-1]
            self.assertGreater(n, abs(e))            # went straight (north), not E or W
        finally:
            R._HEADING.k_compass, R._HEADING.fixes = saved


def _payload():
    """A straight footway 0..200 m north of (0,0), plus a spur."""
    lat0, lon0 = 0.0, 0.0
    nodes = [{"type": "node", "id": i, "lat": lat0 + i * 20 / 110540.0, "lon": lon0} for i in range(11)]
    nodes.append({"type": "node", "id": 99, "lat": 0.0, "lon": 60 / 111320.0})
    way = {"type": "way", "id": 1, "nodes": list(range(11)), "tags": {"highway": "footway"}}
    spur = {"type": "way", "id": 2, "nodes": [0, 99], "tags": {"highway": "footway"}}
    return {"elements": nodes + [way, spur]}


class TestOsm(unittest.TestCase):
    def setUp(self):
        self.saved = dict(R._OSM)
        R._OSM.update(payload=_payload(), goal=None, route=None, t=0.0)

    def tearDown(self):
        R._OSM.clear()
        R._OSM.update(self.saved)

    def test_short_leg_goes_straight(self):
        """Costa Rica: 18-36 m legs beside a junction; road-node routing is a detour."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIsNone(R._osm_target(0.0, 0.0, (30 / 110540.0, 0.0), 30.0))
        self.assertIn("short leg", R._OSM["route"].note)

    def test_long_leg_follows_path(self):
        """Machakos: the CP2->CP3 chord crossed bush while a footway joined them."""
        with contextlib.redirect_stdout(io.StringIO()):
            t = R._osm_target(0.0, 0.0, (150 / 110540.0, 0.0), 150.0)
        self.assertIsNotNone(t)
        self.assertEqual(R._OSM["route"].source, "osm")

    def test_near_checkpoint_aims_direct(self):
        self.assertIsNone(R._osm_target(0.0, 0.0, (150 / 110540.0, 0.0), R.OSM_DIRECT_M - 1))


# ---------------------------------------------------------------- claims, logs
class TestClaimAndLogs(unittest.TestCase):
    def test_every_checkpoint_claims_within_15m(self):
        """User, finals: claim any checkpoint at <= 15 m."""
        self.assertEqual(R._CLAIM.near_m, 15.0)
        from erc_urban.modules.claim import ApproachClaim
        c = ApproachClaim(near_m=15.0)
        c.start_goal(is_last=False)
        self.assertTrue(c.update(14.9, 0.0)[0])
        self.assertFalse(c.update(15.2, 1.0)[0])


class TestBotCheck(unittest.TestCase):
    """bot_check against the fake SDK, in process."""

    def _run(self, rover, port, **env):
        from test import fake_sdk
        from erc_urban import bot_check as B
        srv = fake_sdk.serve(rover, port)
        try:
            with patch.object(B, "SDK", f"http://127.0.0.1:{port}"), \
                    patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()) as out:
                restarts = []
                with patch.object(B, "_restart_sdk", lambda: restarts.append(1)), \
                        patch.object(B, "_headless_chromes", lambda: 1):
                    flags = B.run_checks(move=True, fresh=False)
            return flags, restarts, out.getvalue()
        finally:
            srv.shutdown()
            rover._running = False

    def test_flapping_link_restarts_sdk(self):
        """Costa Rica: a stale Chrome kicked ours off (REMOTE_LOGIN) every few seconds."""
        from test import fake_sdk
        rover = fake_sdk.FakeRover([(0, 30)], flap=True)
        flags, restarts, out = self._run(rover, 8781)
        self.assertTrue(restarts, out)

    def test_wheels_barely_moving_pushes_harder(self):
        """Machakos/Costa Rica grass and slopes: 0.3 barely moved the rover."""
        from test import fake_sdk
        rover = fake_sdk.FakeRover([(0, 30)], speed_per_unit=0.15)
        flags, _, out = self._run(rover, 8782)
        self.assertEqual(flags.get("V_NOM"), "0.34", out)

    def test_fast_bot_is_slowed(self):
        from test import fake_sdk
        rover = fake_sdk.FakeRover([(0, 30)], speed_per_unit=2.4)
        flags, _, out = self._run(rover, 8783)
        self.assertLess(float(flags["V_NOM"]), 0.2 + 0.03, out)

    def test_no_turn_measured_defaults_safe(self):
        """Machakos: the wiggle got no clear turn -> z (Mini+) and compass off."""
        from test import fake_sdk
        rover = fake_sdk.FakeRover([(0, 30)], yaw_per_unit=2.0)
        flags, _, out = self._run(rover, 8784)
        self.assertEqual((flags["GYRO_AXIS"], flags["HEADING_K_COMPASS"]), ("z", "0"), out)


if __name__ == "__main__":
    unittest.main()
