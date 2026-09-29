"""Checkpoint claim policy: drive in to ~2 m, but never circle chasing GPS."""
import unittest

from erc_urban.modules.claim import ApproachClaim


class TestApproachClaim(unittest.TestCase):
    def test_outside_radius_never_claims(self):
        c = ApproachClaim()
        c.start_goal(is_last=False)
        self.assertEqual(c.update(15.5, 0.0), (False, ""))
        self.assertFalse(c.update(float("nan"), 1.0)[0])

    def test_intermediate_drives_in_to_near(self):
        c = ApproachClaim(near_m=2.0)
        c.start_goal(is_last=False)
        for t, r in enumerate((14.0, 11.0, 8.0, 5.0, 2.6)):
            self.assertFalse(c.update(r, float(t))[0], r)
        ok, why = c.update(1.9, 6.0)
        self.assertTrue(ok)
        self.assertIn("within 2.0", why)

    def test_first_checkpoint_claims_at_radius(self):
        c = ApproachClaim(near_m=5.0)
        c.start_goal(is_last=False, is_first=True)
        ok, why = c.update(14.9, 0.0)
        self.assertTrue(ok)
        self.assertIn("first checkpoint", why)

    def test_middle_checkpoint_claims_at_5m(self):
        c = ApproachClaim(near_m=5.0)
        c.start_goal(is_last=False)
        self.assertFalse(c.update(6.0, 0.0)[0])
        self.assertTrue(c.update(4.9, 1.0)[0])

    def test_last_checkpoint_claims_at_radius(self):
        c = ApproachClaim()
        c.start_goal(is_last=True)
        self.assertTrue(c.update(14.9, 0.0)[0])

    def test_pivoting_round_is_not_a_stall(self):
        """Turning to face the checkpoint: 30 s of pivots, no distance."""
        c = ApproachClaim()
        c.start_goal(is_last=False)
        for t in range(30):
            self.assertFalse(c.update(5.1, float(t), travelled_m=0.0)[0], t)

    def test_circling_by_distance_claims(self):
        c = ApproachClaim(stall_m=4.0)
        c.start_goal(is_last=False)
        d, claimed = 0.0, None
        for t in range(60):
            d += 0.25
            ok, why = c.update(3.5 + 0.3 * ((t % 3) - 1), float(t), travelled_m=d)
            if ok:
                claimed = (d, why)
                break
        self.assertIsNotNone(claimed)
        self.assertLess(claimed[0], 6.0)
        self.assertIn("drove", claimed[1])

    def test_gps_noise_stall_claims(self):
        """Hovering 3-4 m out (GPS noise / blocked) must not circle forever."""
        c = ApproachClaim(stall_s=20.0)
        c.start_goal(is_last=False)
        t = 0.0  # time-only fallback (no travel reported)
        self.assertFalse(c.update(4.0, t)[0])
        claimed = None
        while t < 60.0:
            t += 1.0
            ok, why = c.update(3.6 + 0.3 * ((int(t) % 3) - 1), t)
            if ok:
                claimed = (t, why)
                break
        self.assertIsNotNone(claimed)
        self.assertLessEqual(claimed[0], 25.0)        # best 3.3 m at t=3, then 20 s
        self.assertIn("no closer", claimed[1])

    def test_steady_progress_is_not_a_stall(self):
        c = ApproachClaim(stall_s=20.0, max_s=90.0)
        c.start_goal(is_last=False)
        r = 14.0
        for t in range(0, 60):
            ok, _ = c.update(r, float(t))
            self.assertFalse(ok, (t, r))
            r -= 0.2

    def test_leaving_the_radius_resets(self):
        c = ApproachClaim(stall_s=20.0)
        c.start_goal(is_last=False)
        c.update(10.0, 0.0)
        c.update(16.0, 15.0)                  # escaped back out
        self.assertFalse(c.update(10.0, 25.0)[0])

    def test_stale_gps_inside_radius_claims(self):
        c = ApproachClaim()
        c.start_goal(is_last=False)
        self.assertTrue(c.update(8.0, 0.0, gps_stale=True)[0])


if __name__ == "__main__":
    unittest.main()
