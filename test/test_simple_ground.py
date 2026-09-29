import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import erc_urban.run as R


class SimpleGroundTest(unittest.TestCase):
    def test_grass_equals_road_but_solid_objects_remain_blocked(self):
        labels = np.array([[0, 1, 2, 3, 4, 5]])
        sem = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(
            id2label={0: "road", 1: "grass", 2: "tree", 3: "wall",
                      4: "person", 5: "car"})))
        with patch.object(R, "FAN_SIMPLE", True), \
                patch.object(R, "FAN_TERRAIN", "urban"), \
                patch.object(R, "_FAN_IDS", None):
            ground, grass = R._fan_masks(sem, labels)
        np.testing.assert_array_equal(ground, [[True, True, False, False, False, False]])
        self.assertFalse(grass.any())
