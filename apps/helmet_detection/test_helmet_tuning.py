"""Unit tests for Helmet Detection tuning, spatial association, and calibration."""

import os
import sys
import unittest
import numpy as np

# Ensure path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import Box, PersonStatus, HelmetDetector, draw_helmet_detections


class TestHelmetDetectionTuning(unittest.TestCase):
    def test_box_contains_standard(self):
        person = Box(100, 100, 300, 600, 0.90, "person")
        helmet = Box(170, 110, 230, 180, 0.88, "helmet")
        self.assertTrue(person.contains(helmet))

    def test_box_contains_tilted_head(self):
        # Person box from x=100 to 300, but head tilted slightly left (x=90 to 140, cx=115 or slightly outside)
        person = Box(100, 100, 300, 600, 0.90, "person")
        # Helmet slightly outside left border due to head tilt
        tilted_helmet = Box(85, 110, 135, 170, 0.85, "helmet")
        self.assertTrue(person.contains(tilted_helmet, margin_ratio=0.15))

    def test_box_contains_helmet_above_top_boundary(self):
        # Person detected from shoulders down (y1=120), helmet sits above at y=90..130
        person = Box(100, 120, 300, 600, 0.90, "person")
        top_helmet = Box(170, 80, 230, 130, 0.89, "helmet")
        self.assertTrue(person.contains(top_helmet))

    def test_box_overlaps_iou(self):
        box1 = Box(100, 100, 200, 200, 0.9, "head")
        box2 = Box(110, 90, 190, 180, 0.85, "helmet")
        self.assertTrue(box1.overlaps(box2, min_iou=0.10))

        far_box = Box(500, 500, 600, 600, 0.9, "helmet")
        self.assertFalse(box1.overlaps(far_box, min_iou=0.10))

    def test_person_status_violation(self):
        ps = PersonStatus([100, 100, 300, 600], 0.92, "no_helmet")
        d = ps.to_dict()
        self.assertEqual(d["status"], "no_helmet")
        self.assertEqual(d["confidence"], 0.92)


if __name__ == "__main__":
    unittest.main()
