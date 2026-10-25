import os
import unittest
import cv2
import numpy as np
from pathlib import Path

from backend.segmentation import (
    Crop,
    validate_image,
    remove_horizontal_rules,
    binarize_image,
    segment_lines,
    _split_wide_component,
    _resolve_overlaps,
    extract_crops,
)
from backend.config import CHECKPOINT_PATH
from backend.pipeline import AnalysisPipeline


class TestSegmentation(unittest.TestCase):
    def test_validate_image_valid(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        validate_image(img)

        gray = np.zeros((100, 100), dtype=np.uint8)
        validate_image(gray)

    def test_validate_image_invalid(self):
        with self.assertRaises(ValueError):
            validate_image(None)

        with self.assertRaises(ValueError):
            validate_image("not an image")

        with self.assertRaises(ValueError):
            validate_image(np.zeros((10, 10), dtype=np.uint8))

    def test_remove_horizontal_rules(self):
        h, w = 100, 400
        ink = np.zeros((h, w), dtype=np.uint8)
        ink[50, :] = 255
        ink[51, :] = 255
        ink[40:60, 200:203] = 255

        cleaned, rules = remove_horizontal_rules(ink)
        self.assertGreater(np.sum(rules == 255), 0, "Horizontal rules should be detected.")
        self.assertTrue(np.any(cleaned[40:48, 200:203] == 255), "Vertical stroke should be preserved.")

    def test_binarize_polarity(self):
        # Dark text on light background
        light_bg = np.ones((100, 100), dtype=np.uint8) * 240
        light_bg[40:60, 40:60] = 20
        cleaned, raw = binarize_image(light_bg)
        self.assertEqual(cleaned[50, 50], 255, "Dark text on light background should binarize to 255 (ink).")
        self.assertEqual(cleaned[10, 10], 0, "Light background should binarize to 0.")

        # Light text on dark background
        dark_bg = np.zeros((100, 100), dtype=np.uint8)
        dark_bg[40:60, 40:60] = 230
        cleaned_dark, _ = binarize_image(dark_bg)
        self.assertEqual(cleaned_dark[50, 50], 255, "Light text on dark background should binarize to 255 (ink).")

    def test_split_wide_touching_characters(self):
        canvas = np.zeros((30, 60), dtype=np.uint8)
        cv2.putText(canvas, "on", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        pieces = _split_wide_component(canvas, min_height=5)
        self.assertEqual(len(pieces), 2, f"Touching characters 'on' should be split into 2 pieces, got {len(pieces)}")

    def test_preserve_single_wide_letter_m(self):
        canvas_m = np.zeros((30, 50), dtype=np.uint8)
        cv2.putText(canvas_m, "m", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
        pieces = _split_wide_component(canvas_m, min_height=5)
        self.assertEqual(len(pieces), 1, f"Single letter 'm' should not be split, got {len(pieces)}")

    def test_resolve_overlaps(self):
        # Duplicate boxes
        boxes = [(10, 10, 30, 30), (10, 10, 30, 30)]
        resolved = _resolve_overlaps(boxes)
        self.assertEqual(len(resolved), 1, f"Duplicate boxes should be deduplicated, got {len(resolved)}")

        # Box A containing Box B
        boxes_contained = [(10, 10, 40, 40), (15, 15, 35, 35)]
        resolved_contained = _resolve_overlaps(boxes_contained)
        self.assertEqual(len(resolved_contained), 1, "Contained box should be suppressed.")

        # Two separate boxes
        boxes_sep = [(10, 10, 30, 30), (50, 10, 70, 30)]
        resolved_sep = _resolve_overlaps(boxes_sep)
        self.assertEqual(len(resolved_sep), 2, "Separate boxes should both be retained.")

    def test_extract_crops_synthetic(self):
        canvas = np.ones((300, 600, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "hello world", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
        cv2.putText(canvas, "dyslexia test", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

        crops, line_boxes, stats = extract_crops(canvas)
        self.assertGreaterEqual(len(line_boxes), 2, f"Should detect at least 2 lines, got {len(line_boxes)}")
        self.assertGreaterEqual(len(crops), 10, f"Should extract at least 10 character crops, got {len(crops)}")
        self.assertEqual(stats["lines_detected"], len(line_boxes))
        self.assertEqual(stats["final_crops_count"], len(crops))

        # Check reading order
        for idx in range(len(crops) - 1):
            c1, c2 = crops[idx], crops[idx + 1]
            if c1.line_id == c2.line_id:
                self.assertLessEqual(c1.x1, c2.x1 + 10, f"Crops within line {c1.line_id} should be sorted left-to-right")
            else:
                self.assertLess(c1.line_id, c2.line_id, "Lines should be ordered top-to-bottom")

    def test_pipeline_integration(self):
        img = cv2.imread("test_samples/sample_cell_14_0.jpeg")
        pipeline = AnalysisPipeline(CHECKPOINT_PATH)
        result = pipeline.analyze(img, "test_job_1234567890abcdef12345678")

        self.assertEqual(result["status"], "screening_complete")
        self.assertGreater(result["crop_count"], 50)
        self.assertGreater(result["line_count"], 10)
        self.assertIsNotNone(result["screening_score"])
        self.assertIn("annotated_image", result["artifacts"])
        self.assertIn("segmentation_metadata", result)
        self.assertTrue(Path("artifacts/test_job_1234567890abcdef12345678/annotated.png").is_file())
        self.assertTrue(Path("artifacts/test_job_1234567890abcdef12345678/result.json").is_file())


if __name__ == "__main__":
    unittest.main()
