import unittest

import cv2
import numpy as np

from point_estimator import estimate_wall_points


class PointEstimatorTests(unittest.TestCase):
    def test_trapezoid_uses_mask_and_image_lines(self):
        height, width = 400, 600
        image = np.full((height, width, 3), 210, dtype=np.uint8)
        expected = np.array([[110, 80], [500, 55], [545, 330], [65, 345]], dtype=np.int32)
        cv2.polylines(image, [expected], True, (20, 20, 20), 5)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, expected, 255)
        # Simulate furniture cutting into the segmentation.
        cv2.rectangle(mask, (235, 235), (390, 350), 0, -1)

        result = estimate_wall_points(image, mask, segmentation_score=0.9)

        self.assertEqual(len(result["estimated_points"]), 4)
        self.assertGreaterEqual(result["detected_line_count"], 2)
        self.assertGreater(result["confidence"], 0.6)
        for point in result["estimated_points"]:
            self.assertGreaterEqual(point["x"], 0)
            self.assertLess(point["x"], width)
            self.assertGreaterEqual(point["y"], 0)
            self.assertLess(point["y"], height)

    def test_empty_mask_is_rejected(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "foreground"):
            estimate_wall_points(image, mask)


if __name__ == "__main__":
    unittest.main()
