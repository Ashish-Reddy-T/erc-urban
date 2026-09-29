"""Free space per heading (NAV=fan): synthetic ground masks with known geometry."""
from __future__ import annotations
import os
import unittest

import numpy as np

from erc_urban.calib import CameraCalibration
from erc_urban.modules.freespace import FreeSpaceUrban, choose_heading, open_side

HW = (288, 512)
CAL = CameraCalibration.load_or_default(
    os.path.join(os.path.dirname(__file__), "..", "erc_urban", "camera_front.json"), 1024, 576)


def _row_at(dist_m: float) -> int:
    """Image row (at 288) of flat ground `dist_m` straight ahead."""
    uv, ok = CAL.scaled_to(HW[1], HW[0]).project_ground(np.array([[dist_m, 0.0]]))
    assert ok[0]
    return int(round(uv[0, 1]))


def _open() -> np.ndarray:
    return np.ones(HW, bool)


class TestFreeSpaceUrban(unittest.TestCase):
    def setUp(self):
        self.fs = FreeSpaceUrban(CAL, seg_hw=HW)

    def test_open_ground_goes_where_wanted(self):
        r = self.fs.estimate(_open())
        self.assertGreaterEqual(r.drive.ahead_m, 4.9)
        h, which = choose_heading(r, 10.0, 1.2)
        self.assertEqual((h, which), (10.0, "track"))

    def test_wall_ahead_left_turns_right(self):
        """Mission-3: angled into a planter wall, open plaza to the right."""
        g = _open()
        g[: _row_at(1.0), :300] = False        # ground ends 1 m out, left + centre
        r = self.fs.estimate(g)
        self.assertLess(r.drive.ahead_m, 1.2)
        h, which = choose_heading(r, 0.0, 1.2)
        self.assertIsNotNone(h)
        self.assertLess(h, 0.0, "must turn RIGHT, away from the wall")
        self.assertEqual(open_side(r), -1.0)

    def test_boxed_in_has_no_heading(self):
        g = _open()
        g[: _row_at(0.5), :] = False
        r = self.fs.estimate(g)
        self.assertLess(r.drive.ahead_m, 0.6)
        self.assertEqual(choose_heading(r, 0.0, 1.2, 0.8), (None, "none"))

    def test_prefers_track_over_grass(self):
        """Avoid grass, allow to pass: the goal is ahead-left across grass, the
        path carries on straight/right."""
        track = _open()
        grass = np.zeros(HW, bool)
        # Grass from 0.6 m: even at 50 deg the strip meets it within 1.2 m.
        track[: _row_at(0.6), :230] = False
        grass[: _row_at(0.6), :230] = True
        r = self.fs.estimate(track, grass)
        h, which = choose_heading(r, 30.0, 1.2)
        self.assertEqual(which, "track")
        self.assertLess(h, 30.0)
        # Grass alone would have allowed the wanted heading.
        self.assertGreaterEqual(r.drive.clear_at(30.0), 1.2)

    def test_grass_used_when_it_is_the_only_way(self):
        track = np.zeros(HW, bool)
        track[_row_at(0.6):, :] = True           # paving ends right away
        grass = ~track
        r = self.fs.estimate(track, grass)
        h, which = choose_heading(r, 0.0, 1.2)
        self.assertEqual((h, which), (0.0, "grass"))

    def test_seam_is_not_an_obstacle_but_a_kerb_is(self):
        seam = _open()
        v = _row_at(2.0)
        seam[v - 1: v + 2, :] = False            # 3 rows: grout / leaf line
        self.assertGreaterEqual(self.fs.estimate(seam).drive.ahead_m, 4.9)
        # A 10 cm kerb face 2 m out: its top edge sits where flat ground would
        # be 0.148*2/(0.148-0.10) = 6.2 m away.
        kerb = _open()
        kerb[_row_at(6.2): _row_at(2.0) + 1, :] = False
        ahead = self.fs.estimate(kerb).drive.ahead_m
        self.assertLess(ahead, 2.3)
        self.assertGreater(ahead, 1.7)

    def test_leaf_under_the_bumper_is_not_an_obstacle(self):
        """Wuhan: a ginkgo leaf lying 0.19-0.3 m ahead (<= ~5 cm implied)."""
        leaf = _open()
        leaf[_row_at(0.3):, 200:300] = False
        self.assertGreaterEqual(self.fs.estimate(leaf).drive.ahead_m, 4.9)

    def test_low_box_close_up_is_an_obstacle(self):
        """Same rows as the leaf but reaching further: 0.19 m .. 1.0 m implies
        ~12 cm, i.e. something standing there."""
        box = _open()
        box[_row_at(1.0):, 200:300] = False
        self.assertLess(self.fs.estimate(box).drive.ahead_m, 0.6)

    def test_forbidden_heading_is_not_retried(self):
        """Wuhan loop: the nearest-to-goal gap kept reading clear from afar."""
        r = self.fs.estimate(_open())
        forbid = np.abs(r.drive.bins_deg - 0.0) <= 15.0
        h, _ = choose_heading(r, 0.0, 1.2, forbid=forbid)
        self.assertGreater(abs(h), 15.0)

    def test_overhang_is_an_obstacle(self):
        """Wuhan: ground visible under a parked e-scooter almost to the
        horizon; the floorboard 0.2 m away must block, not read 5 m clear."""
        g = _open()
        under = _row_at(8.0)
        g[:under, :] = False                     # underside starts near the horizon
        depth = np.full(HW, 6.0, np.float32)     # ground: far
        depth[:under, :] = 20.0                  # background
        depth[under - 20: under, 150: 360] = 0.2  # the scooter's underside
        r = self.fs.estimate(g, None, depth)
        self.assertLess(r.drive.ahead_m, 0.4)
        self.assertGreaterEqual(self.fs.estimate(g).drive.ahead_m, 4.9)

    def test_far_background_is_not_an_overhang(self):
        """Open plaza: the depth fit squashes far things to ~1 m, ground and
        buildings alike. Nearer-than-the-ground-it-hides must not fire."""
        g = _open()
        g[: _row_at(8.0), :] = False
        depth = np.full(HW, 0.9, np.float32)     # everything far reads ~0.9 m
        r = self.fs.estimate(g, None, depth)
        self.assertEqual(self.fs.last_overhang_cols, 0)
        self.assertGreaterEqual(r.drive.ahead_m, 4.9)

    def _ground_depth(self):
        """Depth of flat ground per pixel (forward distance, as a depth model
        reports it), background far above it."""
        fs = self.fs
        d = np.where(fs._gok, fs._gx, 30.0).astype(np.float32)
        return np.minimum(d, 30.0)

    def test_scooter_body_between_wheels_blocks(self):
        """Wuhan: parked scooter side-on 1.5 m ahead; ground visible under its
        body to 6 m, so the boundary alone reads the way clear."""
        g = _open()
        near, under = _row_at(1.5), _row_at(6.0)
        depth = self._ground_depth()
        for c0, c1 in ((170, 200), (312, 342)):          # wheels: touch the ground
            g[:near, c0:c1] = False
            depth[:near, c0:c1] = 1.5
        g[:under, 200:312] = False                       # body: ground runs under it
        depth[:under, 200:312] = 1.5
        self.assertGreaterEqual(self.fs.estimate(g).drive.ahead_m, 4.9)
        r = self.fs.estimate(g, None, depth)
        self.assertGreater(self.fs.last_contact_cols, 50)
        self.assertLess(r.drive.ahead_m, 1.7)

    def test_receding_wall_is_not_a_contact(self):
        """A kerb/planter running away on the left: every column has its own
        base, and depth agrees with the ground there. No contact points."""
        g = _open()
        depth = self._ground_depth()
        for u in range(0, 200):
            # base gets nearer toward the image edge: 6 m at u=200 .. 1.2 m at u=0
            row = _row_at(1.2 + 4.8 * u / 200.0)
            g[:row, u] = False
            depth[:row, u] = depth[row, u]
        self.fs.estimate(g, None, depth)
        self.assertEqual(self.fs.last_contact_cols, 0)

    def test_flat_leaf_drift_passes_with_depth(self):
        """Panama: a dry-leaf patch 0.2..0.8 m (implied ~11 cm) that is as deep
        as the ground around it is flat, not a box."""
        g = _open()
        g[_row_at(0.8):, 200:320] = False
        depth = self._ground_depth()                 # patch depth = ground depth
        self.assertLess(self.fs.estimate(g).drive.ahead_m, 0.5)
        self.assertGreaterEqual(self.fs.estimate(g, None, depth).drive.ahead_m, 4.9)

    def test_kerb_face_stays_with_depth(self):
        """10 cm kerb 2 m out: the face keeps one (near) depth up its rows."""
        g = _open()
        top, base = _row_at(6.2), _row_at(2.0)
        g[top: base + 1, :] = False
        depth = self._ground_depth()
        depth[top: base + 1, :] = 2.0                # vertical face at 2 m
        ahead = self.fs.estimate(g, None, depth).drive.ahead_m
        self.assertLess(ahead, 2.3)

    def test_offroad_tree_line_uses_raw_depth_range(self):
        """A rising green slope must not project a distant tree line onto the
        bumper.  Only tree/palm pixels explicitly marked by the caller qualify
        for this off-road correction."""
        g = _open()
        base = _row_at(0.55)
        g[:base, :] = False
        tree = ~g
        depth = np.full(HW, 2.2, np.float32)
        raw = FreeSpaceUrban(CAL, seg_hw=HW, min_obstacle_h_m=0.12)
        fixed = FreeSpaceUrban(
            CAL, seg_hw=HW, min_obstacle_h_m=0.12, range_relief_ratio=1.5)
        self.assertLess(raw.estimate(g, None, depth).drive.ahead_m, 0.7)
        self.assertGreater(fixed.estimate(g, None, depth, range_relief=tree).drive.ahead_m,
                           1.2)

    def test_offroad_near_tree_is_not_range_relaxed(self):
        """A genuinely near tree has depth consistent with its projected base
        and remains an obstacle."""
        g = _open()
        base = _row_at(0.55)
        g[:base, :] = False
        tree = ~g
        depth = np.full(HW, 0.55, np.float32)
        fixed = FreeSpaceUrban(
            CAL, seg_hw=HW, min_obstacle_h_m=0.12, range_relief_ratio=1.5)
        self.assertLess(fixed.estimate(g, None, depth, range_relief=tree).drive.ahead_m,
                        0.7)

    def test_offroad_compressed_far_tree_gets_clearance_floor(self):
        """The live Panama model can compress a distant tree line to ~1.1 m;
        after it still clears the relative-depth gate, keep enough room to
        drive on the visibly open grass below it."""
        g = _open()
        base = _row_at(0.80)
        g[:base, :] = False
        tree = ~g
        depth = np.full(HW, 1.10, np.float32)
        fixed = FreeSpaceUrban(
            CAL, seg_hw=HW, min_obstacle_h_m=0.12,
            range_relief_ratio=1.2, range_relief_min_m=2.0)
        self.assertGreater(
            fixed.estimate(g, None, depth, range_relief=tree).drive.ahead_m,
            1.5,
        )

    def test_range_relief_never_applies_without_eligible_semantics(self):
        """Pixels not selected by the caller keep the original geometry."""
        g = _open()
        base = _row_at(0.55)
        g[:base, :] = False
        depth = np.full(HW, 2.2, np.float32)
        fixed = FreeSpaceUrban(
            CAL, seg_hw=HW, min_obstacle_h_m=0.12, range_relief_ratio=1.5)
        self.assertLess(
            fixed.estimate(g, None, depth, range_relief=np.zeros(HW, bool)).drive.ahead_m,
            0.7,
        )

    def test_mask_shape_is_checked(self):
        with self.assertRaises(ValueError):
            self.fs.estimate(np.ones((144, 256), bool))



class TestBlockedHeadings(unittest.TestCase):
    """run.py's world-frame memory of headings that ended in an escape."""

    def test_block_is_world_frame_and_expires_with_travel(self):
        import erc_urban.run as R
        bins = np.arange(-50.0, 50.1, 5.0)
        fan = {}
        R._CRUMBS.travelled_m = 10.0
        R._fan_begin_escape(fan, "test", now=100.0, force_reverse=False,
                            yaw_deg=350.0, failed_rel_deg=0.0)
        # Rover since turned 20 deg left (yaw 330): the blocked world heading
        # 350 is now 20 deg to its RIGHT, i.e. bin -20.
        f = R._fan_forbid(fan, bins, 330.0, now=101.0)
        self.assertTrue(f[bins == -20.0][0])
        self.assertFalse(f[bins == 0.0][0])
        R._CRUMBS.travelled_m = 10.0 + R.FAN_BLOCK_M + 0.1
        self.assertIsNone(R._fan_forbid(fan, bins, 330.0, now=102.0))
        R._CRUMBS.travelled_m = 0.0

    def test_transient_blip_unblocks_the_heading(self):
        """Wuhan: open plaza read 5.0, 0.5, 5.0 m on consecutive ticks; the
        still frame after stopping is clear, so the heading must not stay
        blocked (it sent the rover 15 deg off the goal for 3 m)."""
        import erc_urban.run as R
        from types import SimpleNamespace
        fs = FreeSpaceUrban(CAL, seg_hw=HW).estimate(_open())
        tel = SimpleNamespace(yaw=np.radians(175.0), hw_ts=1.0)
        fan = R._fan_state({})
        fan["last_ahead"] = 5.0                  # clear the tick before
        R._fan_begin_escape(fan, "blip", now=0.0, force_reverse=False, yaw_deg=175.0)
        self.assertEqual(len(fan["blocked"]), 1)
        lin, ang, _, note = R._fan_escape(
            fan, fs, 0.0, tel, now=5.0, forbid_fn=lambda b: R._fan_forbid(fan, b, 175.0, 5.0))
        self.assertEqual(fan["blocked"], [])
        self.assertIsNone(fan["esc"])
        self.assertEqual((lin, ang, note), (0.0, 0.0, "escape: clear ahead"))
        # Same heading fails again: no second let-off (scooter-wheel loop).
        fan["last_ahead"] = 5.0
        R._fan_begin_escape(fan, "again", now=6.0, force_reverse=False, yaw_deg=176.0)
        R._fan_escape(fan, fs, 0.0, tel, now=11.0)
        self.assertEqual(len(fan["blocked"]), 1)

    def test_creeping_up_is_not_transient(self):
        """Clearance was already short before the escape: keep the block even
        if the still frame flickers clear (overhang flicker at the bumper)."""
        import erc_urban.run as R
        from types import SimpleNamespace
        fs = FreeSpaceUrban(CAL, seg_hw=HW).estimate(_open())
        tel = SimpleNamespace(yaw=np.radians(85.0), hw_ts=1.0)
        fan = R._fan_state({})
        fan["last_ahead"] = 0.7
        R._fan_begin_escape(fan, "scooter", now=0.0, force_reverse=False, yaw_deg=85.0)
        R._fan_escape(fan, fs, 0.0, tel, now=5.0)
        self.assertEqual(len(fan["blocked"]), 1)

    def test_real_obstacle_keeps_the_block(self):
        """Still blocked after stopping -> reverse, and the heading stays out."""
        import erc_urban.run as R
        from types import SimpleNamespace
        g = _open()
        g[: _row_at(0.5), :] = False
        fs = FreeSpaceUrban(CAL, seg_hw=HW).estimate(g)
        tel = SimpleNamespace(yaw=np.radians(175.0), hw_ts=1.0)
        fan = R._fan_state({})
        R._fan_begin_escape(fan, "wall", now=0.0, force_reverse=False, yaw_deg=175.0)
        _, _, _, note = R._fan_escape(fan, fs, 0.0, tel, now=5.0)
        self.assertEqual(note, "escape: reverse")
        self.assertEqual(len(fan["blocked"]), 1)


class TestNeverLow(unittest.TestCase):
    """Wuhan: pink granite paving read "building" right at the bumper."""

    def setUp(self):
        from erc_urban.modules.freespace import unlabel_impossible
        self.fix = unlabel_impossible
        self.fs = FreeSpaceUrban(CAL, seg_hw=HW)

    def test_building_patch_with_ground_beyond_is_paving(self):
        g = _open()
        bld = np.zeros(HW, bool)
        bld[_row_at(1.5):, 150:350] = True          # "building" 0.2..1.5 m, plaza beyond
        g &= ~bld
        self.assertLess(self.fs.estimate(g).drive.ahead_m, 0.5)
        self.assertGreaterEqual(self.fs.estimate(self.fix(g, bld, self.fs.horizon)).drive.ahead_m, 4.9)

    def test_real_building_face_stays(self):
        g = _open()
        bld = np.zeros(HW, bool)
        bld[: _row_at(1.5), 150:350] = True          # facade from 1.5 m up to the top
        g &= ~bld
        out = self.fix(g, bld, self.fs.horizon)
        self.assertLess(self.fs.estimate(out).drive.ahead_m, 1.7)


class TestTightTurn(unittest.TestCase):
    """Wuhan mission-3: 1.3 m straight ahead, gap at -10 deg. Driving on while
    steering gently ran it into a parked scooter's wheel; turn in place first."""

    def _step(self, g, want_deg):
        import erc_urban.run as R
        from types import SimpleNamespace
        from unittest.mock import patch
        sent = []
        tel = SimpleNamespace(yaw=0.0, hw_ts=1.0)
        aim = SimpleNamespace(bearing=np.radians(want_deg), speed_scale=1.0, source="gps")
        with patch.object(R, "_cmd", lambda c, lin, ang, hold_s=None: sent.append((lin, ang))), \
                patch.object(R, "_halt", lambda c: sent.append((0.0, 0.0))), \
                patch.object(R, "_publish_overlay", lambda *a, **k: None):
            R._fan_step(None, tel, None, 20.0, 0.0, aim, g, None, {}, None, "", 0)
        return sent[-1]

    def test_short_ahead_with_side_gap_pivots(self):
        g = _open()
        g[: _row_at(1.1), 150:300] = False       # something 1.1 m out, centre-left
        g[: _row_at(1.3), 300:330] = False       # ...and its edge, 1.3 m out
        lin, ang = self._step(g, 0.0)
        self.assertEqual(lin, 0.0)
        self.assertLess(ang, 0.0, "pivot right, toward the gap")

    def test_open_ahead_drives(self):
        lin, ang = self._step(_open(), 5.0)
        self.assertGreater(lin, 0.0)

    def test_near_goal_faces_it_first(self):
        """Panama CP1: 4 m out, goal 25 deg off -> pivot, do not orbit."""
        import erc_urban.run as R
        from types import SimpleNamespace
        from unittest.mock import patch
        sent = []
        tel = SimpleNamespace(yaw=0.0, hw_ts=1.0)
        aim = SimpleNamespace(bearing=np.radians(25.0), speed_scale=1.0, source="gps")
        with patch.object(R, "_cmd", lambda c, lin, ang, hold_s=None: sent.append((lin, ang))), \
                patch.object(R, "_halt", lambda c: sent.append((0.0, 0.0))), \
                patch.object(R, "_publish_overlay", lambda *a, **k: None):
            R._fan_step(None, tel, None, 4.0, np.radians(25.0), aim, _open(), None, {}, None, "", 0)
            far = []
            R._fan_step(None, tel, None, 20.0, np.radians(25.0), aim, _open(), None, {}, None, "", 0)
        self.assertEqual(sent[0][0], 0.0)
        self.assertGreater(sent[0][1], 0.0)
        self.assertGreater(sent[1][0], 0.0)                 # far: drive and steer

    def test_goal_behind_keeps_one_turn_direction(self):
        """Panama CP3->CP4: bearing +179 / -172 / +165 must not reverse the pivot."""
        import erc_urban.run as R
        from types import SimpleNamespace
        from unittest.mock import patch
        st, angs, clock = {}, [], [0.0]
        tel = SimpleNamespace(yaw=0.0, hw_ts=1.0)
        with patch.object(R, "_cmd", lambda c, lin, ang, hold_s=None: angs.append(ang)), \
                patch.object(R, "_halt", lambda c: angs.append(0.0)), \
                patch.object(R, "_publish_overlay", lambda *a, **k: None), \
                patch.object(R.time, "monotonic", lambda: clock[0]):
            for b in (179.0, -172.0, 165.0, -176.0):
                aim = SimpleNamespace(bearing=np.radians(b), speed_scale=1.0, source="gps")
                R._fan_state(st)["pivot"].reset()
                R._fan_step(None, tel, None, 24.0, np.radians(b), aim, _open(), None, st, None, "", 0)
                clock[0] += 10.0
        turns = [a for a in angs if abs(a) > 1e-6]
        self.assertTrue(turns)
        self.assertTrue(all(a > 0 for a in turns) or all(a < 0 for a in turns), turns)

    def test_stale_wheel_data_is_not_a_stall(self):
        """Laggy link: odometry says 'wheel_stale' -> no boost, no stuck escape."""
        import erc_urban.run as R
        from types import SimpleNamespace
        from unittest.mock import patch
        from erc_urban.modules.odometry import OdomDelta
        sent, clock, esc = [], [100.0], []
        tel = SimpleNamespace(yaw=0.0, hw_ts=1.0)
        aim = SimpleNamespace(bearing=0.0, speed_scale=1.0, source="gps")
        st = {}
        stale = OdomDelta(0.0, 0.0, 0.0, 0.0, reason="wheel_stale")
        R._CRUMBS.travelled_m = 0.0
        with patch.object(R, "_cmd", lambda c, lin, ang, hold_s=None: sent.append(lin)), \
                patch.object(R, "_halt", lambda c: sent.append(0.0)), \
                patch.object(R, "_publish_overlay", lambda *a, **k: None), \
                patch.object(R.time, "monotonic", lambda: clock[0]):
            for _ in range(20):                          # 10 s of stale data
                R._fan_step(None, tel, None, 20.0, 0.0, aim, _open(), None, st, stale, "", 0)
                clock[0] += 0.5
        self.assertIsNone(R._fan_state(st)["esc"])
        self.assertLess(max(sent), sent[0] + 0.01)

    def test_stalled_forward_is_boosted(self):
        """Panama roots: wheels not turning under a forward command -> more power."""
        import erc_urban.run as R
        from types import SimpleNamespace
        from unittest.mock import patch
        sent, clock = [], [100.0]
        tel = SimpleNamespace(yaw=0.0, hw_ts=1.0)
        aim = SimpleNamespace(bearing=0.0, speed_scale=1.0, source="gps")
        st = {}
        R._CRUMBS.travelled_m = 0.0
        with patch.object(R, "_cmd", lambda c, lin, ang, hold_s=None: sent.append(lin)), \
                patch.object(R, "_halt", lambda c: sent.append(0.0)), \
                patch.object(R, "_publish_overlay", lambda *a, **k: None), \
                patch.object(R.time, "monotonic", lambda: clock[0]):
            for _ in range(12):                          # 5.5 s, no wheel travel
                R._fan_step(None, tel, None, 20.0, 0.0, aim, _open(), None, st, None, "", 0)
                clock[0] += 0.5
        self.assertGreater(sent[-1], sent[0] + 0.1)


if __name__ == "__main__":
    unittest.main()
