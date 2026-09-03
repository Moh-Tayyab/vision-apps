import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../apps/top_camera_counter")))

import pytest
import numpy as np
from detector import compute_iou, compute_ios, non_max_suppression
from live_counter import filter_top_row_cartons, LiveCartonCounter
from tracker import ByteTracker
from state_machine import CartonStateMachine, CartonState


def test_iou_and_ios():
    # Identical boxes
    b1 = (100.0, 100.0, 200.0, 200.0)
    b2 = (100.0, 100.0, 200.0, 200.0)
    assert compute_iou(b1, b2) == pytest.approx(1.0)
    assert compute_ios(b1, b2) == pytest.approx(1.0)

    # Box inside another box (Nested)
    outer = (100.0, 100.0, 300.0, 300.0) # area 40000
    inner = (150.0, 150.0, 250.0, 250.0) # area 10000 (100% inside outer)
    assert compute_ios(inner, outer) == pytest.approx(1.0) # 100% contained
    assert compute_iou(inner, outer) == pytest.approx(10000 / 40000)

    # Disjoint boxes
    disjoint = (500.0, 500.0, 600.0, 600.0)
    assert compute_iou(b1, disjoint) == 0.0
    assert compute_ios(b1, disjoint) == 0.0


def test_non_max_suppression():
    # 3 overlapping boxes on same carton with different confidences
    boxes = [
        {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.85},
        {"x1": 105, "y1": 95, "x2": 205, "y2": 195, "confidence": 0.60},
        {"x1": 110, "y1": 110, "x2": 190, "y2": 190, "confidence": 0.45}, # nested
        {"x1": 400, "y1": 100, "x2": 500, "y2": 200, "confidence": 0.90}, # separate carton
    ]

    kept = non_max_suppression(boxes, iou_thresh=0.35, ios_thresh=0.60)
    # Should keep only highest conf for first carton (0.85) and the separate carton (0.90)
    assert len(kept) == 2
    confs = [k["confidence"] for k in kept]
    assert 0.90 in confs
    assert 0.85 in confs


def test_filter_top_row_cartons():
    # Simulate a pallet with 2 stacks (Left stack X: 100-220, Right stack X: 350-480)
    # Each stack has 3 stacked cartons (Top, Middle, Bottom)
    # Left stack: Top (y: 100-200), Middle (y: 220-320), Bottom (y: 340-440)
    # Right stack: Top (y: 110-210), Middle (y: 230-330), Bottom (y: 350-450)
    detections = [
        (100.0, 340.0, 220.0, 440.0, 0.70), # Left Bottom
        (100.0, 100.0, 220.0, 200.0, 0.85), # Left Top
        (100.0, 220.0, 220.0, 320.0, 0.75), # Left Middle
        (350.0, 110.0, 480.0, 210.0, 0.90), # Right Top
        (350.0, 230.0, 480.0, 330.0, 0.80), # Right Middle
        (350.0, 350.0, 480.0, 450.0, 0.72), # Right Bottom
    ]

    # When top_row_only is True, ONLY the 2 Top cartons should be returned!
    top_row, layer_map = filter_top_row_cartons(
        detections, img_height=600, img_width=800, top_row_only=True
    )
    assert len(top_row) == 2
    top_y1s = sorted([d[1] for d in top_row])
    assert top_y1s == [100.0, 110.0]


def test_layer_lifecycle_and_removal_formula():
    class DummyDetector:
        def __init__(self, detections_seq):
            self.seq = detections_seq
            self.idx = 0
            self.confidence = 0.36
        def detect(self, img, apply_nms=True):
            dets = self.seq[min(self.idx, len(self.seq) - 1)]
            self.idx += 1
            return dets, 10.0

    b1 = {"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.9}
    b2 = {"x1": 250, "y1": 100, "x2": 350, "y2": 200, "confidence": 0.9}
    b3 = {"x1": 400, "y1": 100, "x2": 500, "y2": 200, "confidence": 0.9}

    seq = [
        [b1, b2, b3], # Frame 1 (hit=1, candidate)
        [b1, b2, b3], # Frame 2 (hit=2, active, stable=1)
        [b1, b2, b3], # Frame 3 (stable=2)
        [b1, b2, b3], # Frame 4 (stable=3 -> locked initial_row_cartons=3)
        [b1, b2],     # Frame 5 (1 removed, remaining=2)
        [b1, b2],     # Frame 6 (remaining=2, removed=1)
    ]

    dummy = DummyDetector(seq)
    counter = LiveCartonCounter(dummy, top_row_only=True)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    for _ in range(4):
        res = counter.process_frame(dummy_img)

    assert counter.layer_locked is True
    assert res.initial_row_cartons == 3
    assert res.cartons_remaining == 3
    assert res.cartons_removed_from_row == 0
    assert res.current_layer == 1

    # Now process frames with 1 carton removed
    res = counter.process_frame(dummy_img)
    res = counter.process_frame(dummy_img)

    # Verify formula: Removed = Initial - Visible
    assert res.initial_row_cartons == 3
    assert res.cartons_remaining == 2
    assert res.cartons_removed_from_row == 1 # 3 - 2 = 1
    assert res.total_picked == 1
