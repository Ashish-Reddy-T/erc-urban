"""Fused heading: gyro carries turns, compass noise is averaged out."""
import math
import random
import unittest

from erc_urban.modules.heading import HeadingFilter, wrap180


class TestHeadingFilter(unittest.TestCase):
    def test_pending_track_is_not_a_validated_fix(self):
        f = HeadingFilter(k_compass=0.04)
        f.est = 0.0
        lat, lon = 9.0, -79.5
        f.update(None, 0, 0, lat, lon)
        for _ in range(12):
            lat -= 0.5 / 110540.0
            f.update(None, 0, 0.5, lat, lon)
        self.assertEqual(f.fixes, 0)
        self.assertIsNone(f.last_gps_fix)
        self.assertAlmostEqual(f.est, 0.0)

    def test_noisy_compass_at_rest_is_smoothed(self):
        """Panama: 265/305/262/301... at rest (std ~19 deg)."""
        rnd = random.Random(0)
        f = HeadingFilter()
        out = []
        for k in range(200):
            c = 285.0 + (20.0 if k % 2 else -20.0) + rnd.gauss(0, 5)
            out.append(f.update(c, 0.0))
        tail = out[-50:]
        self.assertLess(max(abs(wrap180(h - 285.0)) for h in tail), 6.0)

    def test_gyro_carries_a_turn_the_compass_misses(self):
        f = HeadingFilter()
        for _ in range(10):
            f.update(90.0, 0.0)
        # Turn 90 deg LEFT over 10 ticks; compass stuck/noisy at 90.
        for _ in range(10):
            h = f.update(90.0, 9.0)
        self.assertLess(abs(wrap180(h - 0.0)), 15.0)    # ~90 - 90 = 0, minus a little pull

    def test_good_compass_is_followed(self):
        f = HeadingFilter()
        for _ in range(10):
            f.update(10.0, 0.0)
        for _ in range(300):                             # gyro says nothing, compass says 40
            h = f.update(40.0, 0.0)
        self.assertLess(abs(wrap180(h - 40.0)), 1.0)

    def test_gps_course_corrects(self):
        """Compass biased 30 deg; a straight 5 m run north fixes most of it."""
        f = HeadingFilter(k_compass=0.0)
        for _ in range(10):
            f.update(30.0, 0.0, 0.0, 9.0, -79.5)
        lat = 9.0
        for _ in range(12):
            lat += 0.5 / 110540.0
            h = f.update(30.0, 0.0, 0.5, lat, -79.5)
        self.assertLess(abs(wrap180(h - 0.0)), 22.0)
        self.assertIsNotNone(f.last_gps_fix)


    def test_untrusted_compass_snaps_to_gps_course(self):
        """Panama: compass seed 170 deg off; one straight 5 m run fixes it."""
        f = HeadingFilter(k_compass=0.0)
        for _ in range(10):
            f.update(190.0, 0.0, 0.0, 9.0, -79.5)          # true heading ~north
        lat, h = 9.0, None
        for _ in range(12):
            lat += 0.5 / 110540.0
            h = f.update(73.0, 0.0, 0.5, lat, -79.5)       # compass garbage
        self.assertLess(abs(wrap180(h - 0.0)), 5.0)


    def test_curved_path_still_fixes_a_backwards_seed(self):
        """Panama: seeded ~180 deg off; the rover steers a little every tick."""
        f = HeadingFilter(k_compass=0.0)
        for _ in range(10):
            f.update(0.0, 0.0, 0.0, 9.0, -79.5)          # seed says north; truly south
        true_h, lat, lon = 180.0, 9.0, -79.5
        h = None
        for k in range(30):
            turn = 3.0 if (k // 5) % 2 == 0 else -3.0     # weave +-3 deg per tick
            true_h -= turn
            lat += 0.2 * math.cos(math.radians(true_h)) / 110540.0
            lon += 0.2 * math.sin(math.radians(true_h)) / (111320.0 * math.cos(math.radians(9.0)))
            h = f.update(0.0, turn, 0.2, lat, lon)
        self.assertLess(abs(wrap180(h - true_h)), 25.0)


    def test_later_big_jump_needs_confirmation(self):
        """One odd GPS track after a good fix must not swing the heading."""
        f = HeadingFilter(k_compass=0.0)
        f.est, f.fixes = 0.0, 3                              # already corrected
        lat, lon = 9.0, -79.5
        f.update(0.0, 0.0, 0.0, lat, lon)
        for _ in range(10):                                  # odometry says north,
            lon += 0.45 / (111320.0 * math.cos(math.radians(9.0)))   # GPS says east
            h = f.update(0.0, 0.0, 0.45, lat, lon)
        self.assertLess(abs(wrap180(h - 0.0)), 5.0)


if __name__ == "__main__":
    unittest.main()
