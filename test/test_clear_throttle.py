import unittest
from unittest.mock import patch
import erc_urban.run as R


class ClearThrottleTest(unittest.TestCase):
    def test_climb_floor_only_with_clearance_and_gentle_turn(self):
        with patch.object(R, "FAN_CLEAR_LIN_MIN", 0.40):
            self.assertEqual(R._clear_path_throttle(.22, 5, 5, .15), .40)
            self.assertEqual(R._clear_path_throttle(.22, .8, 5, .15), .22)
            self.assertEqual(R._clear_path_throttle(.22, 5, .8, .15), .22)
            self.assertEqual(R._clear_path_throttle(.22, 5, 5, .3), .22)

    def test_default_and_limit(self):
        with patch.object(R, "FAN_CLEAR_LIN_MIN", 0):
            self.assertEqual(R._clear_path_throttle(.22, 5, 5, 0), .22)
        with patch.object(R, "FAN_CLEAR_LIN_MIN", 1):
            self.assertEqual(R._clear_path_throttle(.22, 5, 5, 0), .45)
