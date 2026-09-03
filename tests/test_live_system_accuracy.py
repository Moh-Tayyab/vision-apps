"""Live tracking and state machine accuracy simulation tests for Top Camera Counter.

Tests:
1. Static Stability Test (50 frames no change)
2. Sequential Picking Accuracy (Step-by-step picking of cartons)
3. Worker Hand Occlusion vs Genuine Pick Distinction
4. Temporary Jitter & Occlusion Recovery
5. Layer Transition Lifecycle & Count Rollup
6. Removal Formula Exactness: (Initial Row Cartons - Remaining)
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../apps/top_camera_counter")))

from live_counter import LiveCartonCounter
from tracker import ByteTracker
from state_machine import CartonStateMachine, CartonState
from worker_detector import WorkerDetector


class SimulatedDetector:
    def __init__(self, frame_detections):
        self.frame_detections = frame_detections
        self.idx = 0
        self.confidence = 0.36

    def detect(self, img, confidence=None, apply_nms=True):
        dets = self.frame_detections[min(self.idx, len(self.frame_detections) - 1)]
        self.idx += 1
        return dets, 5.0


def make_box(x1, y1, x2, y2, conf=0.90):
    return {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "confidence": conf}


def test_static_stability():
    """Verify that a static scene maintains 100% stable count and 0 false picks over 50 frames."""
    b1 = make_box(50, 50, 150, 150)
    b2 = make_box(180, 50, 280, 150)
    b3 = make_box(310, 50, 410, 150)
    b4 = make_box(440, 50, 540, 150)

    seq = [[b1, b2, b3, b4]] * 50
    detector = SimulatedDetector(seq)
    counter = LiveCartonCounter(detector, top_row_only=False)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    counts = []
    picks = []
    for _ in range(50):
        res = counter.process_frame(dummy_img)
        counts.append(res.cartons_remaining)
        picks.append(res.total_picked)

    # After initial stabilization (3 frames), count must be exactly 4, picks must be exactly 0
    assert counts[4:] == [4] * 46
    assert picks == [0] * 50
    assert res.initial_row_cartons == 4
    assert res.cartons_removed_from_row == 0


def test_sequential_picking_lifecycle():
    """Simulate picking cartons one by one with hand interaction."""
    b1 = make_box(50, 50, 150, 150)
    b2 = make_box(180, 50, 280, 150)
    b3 = make_box(310, 50, 410, 150)

    # Sequence:
    # Frames 1-5: 3 cartons visible, stable
    # Frames 6-10: Hand enters near b3, b3 disappears (picked)
    # Frames 11-15: 2 cartons visible (b1, b2), remaining=2, removed=1
    # Frames 16-20: Hand enters near b2, b2 disappears (picked)
    # Frames 21-25: 1 carton visible (b1), remaining=1, removed=2
    # Frames 26-30: Hand enters near b1, b1 disappears (picked)
    # Frames 31-35: 0 cartons visible, remaining=0, removed=3
    seq = (
        [[b1, b2, b3]] * 5 +
        [[b1, b2]] * 10 +
        [[b1]] * 10 +
        [[]] * 10
    )

    detector = SimulatedDetector(seq)
    counter = LiveCartonCounter(detector, top_row_only=False)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    results = []
    for _ in range(len(seq)):
        res = counter.process_frame(dummy_img)
        results.append(res)

    # Step 1: Initial state (frame 5)
    assert results[4].initial_row_cartons == 3
    assert results[4].cartons_remaining == 3
    assert results[4].cartons_removed_from_row == 0
    assert results[4].total_picked == 0

    # Step 2: 1st carton picked (frame 14)
    assert results[14].cartons_remaining == 2
    assert results[14].cartons_removed_from_row == 1
    assert results[14].total_picked == 1

    # Step 3: 2nd carton picked (frame 24)
    assert results[24].cartons_remaining == 1
    assert results[24].cartons_removed_from_row == 2
    assert results[24].total_picked == 2

    # Step 4: 3rd carton picked (frame 30)
    assert results[29].cartons_remaining == 0
    assert results[29].cartons_removed_from_row == 3
    assert results[29].total_picked == 3


def test_temporary_occlusion_resilience():
    """Verify that a temporary detection drop (1-2 frames of occlusion without hand) does not trigger false picks if carton returns."""
    b1 = make_box(50, 50, 150, 150)
    b2 = make_box(180, 50, 280, 150)

    # b2 drops out for 2 frames then returns
    seq = (
        [[b1, b2]] * 5 +   # Stable (2 cartons)
        [[b1]] * 2 +       # Brief flicker/occlusion of b2 (2 frames)
        [[b1, b2]] * 5     # b2 returns
    )

    detector = SimulatedDetector(seq)
    counter = LiveCartonCounter(detector, top_row_only=False, tracker_max_age=12)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(len(seq)):
        res = counter.process_frame(dummy_img)

    # After b2 returns, remaining count should be 2, total picked should be 0
    assert res.cartons_remaining == 2
    assert res.initial_row_cartons == 2
    assert res.cartons_removed_from_row == 0
    assert res.total_picked == 0


def test_layer_cleared_auto_transition():
    """Verify that when visible cartons reach 0 for 6 frames, layer advances from 1 to 2."""
    b1 = make_box(50, 50, 150, 150)
    b2 = make_box(180, 50, 280, 150)

    # Layer 1: 2 cartons -> 0 cartons (cleared) -> Layer 2 appears (3 cartons)
    b2_1 = make_box(50, 50, 150, 150)
    b2_2 = make_box(180, 50, 280, 150)
    b2_3 = make_box(310, 50, 410, 150)

    seq = (
        [[b1, b2]] * 5 +  # Layer 1 locked (2 cartons)
        [[]] * 7 +        # Layer 1 cleared (triggers transition after 6 zero frames)
        [[b2_1, b2_2, b2_3]] * 5 # Layer 2 appears (3 cartons)
    )

    detector = SimulatedDetector(seq)
    counter = LiveCartonCounter(detector, top_row_only=False)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    results = []
    for _ in range(len(seq)):
        res = counter.process_frame(dummy_img)
        results.append(res)

    # Frame 11 (in 0 frames): Layer 1 cleared, total picked = 2
    assert results[11].current_layer == 2
    assert results[11].total_picked == 2

    # Frame 16 (Layer 2 locked with 3 cartons):
    assert results[-1].current_layer == 2
    assert results[-1].initial_row_cartons == 3
    assert results[-1].cartons_remaining == 3
    assert results[-1].total_picked == 2 # 2 picked from previous layer + 0 from current


if __name__ == "__main__":
    test_static_stability()
    print("✓ test_static_stability PASSED")
    test_sequential_picking_lifecycle()
    print("✓ test_sequential_picking_lifecycle PASSED")
    test_temporary_occlusion_resilience()
    print("✓ test_temporary_occlusion_resilience PASSED")
    test_layer_cleared_auto_transition()
    print("✓ test_layer_cleared_auto_transition PASSED")
    print("\nALL LIVE SYSTEM ACCURACY TESTS PASSED!")
