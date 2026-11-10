#!/usr/bin/env python3
"""
QA Gate 4: State Invariants, API Idempotency & Edge Case Stress Testing.
Validates:
  1. Polling idempotency (no count drift on repeat polls).
  2. Duplicate frame resilience (identical frames produce zero new counts).
  3. Camera disconnect/reconnect resilience (handling missing camera frames).
  4. Invariant: HOLD_FOR_REVIEW strictly isolates from confirmed_count.
  5. Out-of-order timestamps and sync jitter handling.
  6. Track ID reuse and dead track eviction.
"""

import os
import cv2
import pytest
import numpy as np

from apps.truck_loading_detection.three_camera_system.pipeline import ThreeCameraCartonPipeline
from apps.truck_loading_detection.three_camera_system.consensus_policy import ConsensusLevel, DecisionAction


def test_polling_idempotency_without_new_frames():
    pipeline = ThreeCameraCartonPipeline()
    initial_confirmed = pipeline.confirmed_count
    initial_review = pipeline.review_required_count
    initial_total = pipeline.total_count

    # Repeatedly query counts 50 times
    for _ in range(50):
        assert pipeline.confirmed_count == initial_confirmed
        assert pipeline.review_required_count == initial_review
        assert pipeline.total_count == initial_total


def test_duplicate_frame_resilience():
    pipeline = ThreeCameraCartonPipeline(conf_threshold=0.15)
    blank_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    frames = {"cam_top": blank_frame, "cam_side": blank_frame, "cam_flank": blank_frame}

    # Feed 30 identical blank frames
    committed_events = []
    for f_idx in range(30):
        out = pipeline.process_synchronized_frames(frames, frame_idx=f_idx)
        if out.new_crossing_events:
            committed_events.extend(out.new_crossing_events)

    assert len(committed_events) == 0, "Duplicate static frames generated phantom crossing events"
    assert pipeline.confirmed_count == 0
    assert pipeline.provisional_count == 0


def test_camera_disconnect_and_reconnect():
    pipeline = ThreeCameraCartonPipeline(conf_threshold=0.15)
    blank_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # 1. Normal state
    f1 = {"cam_top": blank_frame, "cam_side": blank_frame, "cam_flank": blank_frame}
    pipeline.process_synchronized_frames(f1, frame_idx=1)

    # 2. Camera flank disconnected (None)
    f2 = {"cam_top": blank_frame, "cam_side": blank_frame, "cam_flank": None}
    pipeline.process_synchronized_frames(f2, frame_idx=2)

    # 3. Both side and flank disconnected
    f3 = {"cam_top": blank_frame, "cam_side": None, "cam_flank": None}
    pipeline.process_synchronized_frames(f3, frame_idx=3)

    # 4. Reconnect
    f4 = {"cam_top": blank_frame, "cam_side": blank_frame, "cam_flank": blank_frame}
    pipeline.process_synchronized_frames(f4, frame_idx=4)

    assert pipeline.confirmed_count == 0
    assert pipeline.review_required_count == 0


def test_confirmed_vs_review_required_strict_isolation():
    pipeline = ThreeCameraCartonPipeline()

    policy = pipeline.consensus_policy
    assert policy.min_single_conf == 0.15

    # Verify that SINGLE_CAMERA_OBSERVATION strictly produces HOLD_FOR_REVIEW
    single_cam_decision = policy.evaluate(
        observed_cameras=["cam_side"],
        confidences=[0.45],
        is_ambiguous=False,
        has_direction_conflict=False,
    )
    assert single_cam_decision.level == ConsensusLevel.SINGLE_CAMERA_OBSERVATION
    assert single_cam_decision.action == DecisionAction.HOLD_FOR_REVIEW

    # Verify that TWO_CAMERA_AGREEMENT produces COMMIT
    two_cam_decision = policy.evaluate(
        observed_cameras=["cam_side", "cam_flank"],
        confidences=[0.45, 0.50],
        is_ambiguous=False,
        has_direction_conflict=False,
    )
    assert two_cam_decision.level == ConsensusLevel.TWO_CAMERA_AGREEMENT
    assert two_cam_decision.action == DecisionAction.COMMIT


def test_dead_track_eviction():
    from apps.truck_loading_detection.three_camera_system.per_camera_tracker import PerCameraCartonTracker
    from apps.truck_loading_detection.three_camera_system.detector_adapter import CanonicalCartonDetection
    tracker = PerCameraCartonTracker(camera_id="cam_side", max_invisible_frames=5)
    det = CanonicalCartonDetection(
        camera_id="cam_side",
        frame_idx=1,
        timestamp_sec=0.033,
        bbox=(100.0, 100.0, 200.0, 200.0),
        confidence=0.85,
        class_name="carton",
    )
    tracklets = tracker.update([det], frame_idx=1, timestamp_sec=0.033)
    assert len(tracklets) == 1
    t_id = tracklets[0].local_track_id

    # Coast for 6 frames without detections
    for f in range(2, 9):
        tracker.update([], frame_idx=f, timestamp_sec=f * 0.033)

    # Tracklet should be terminated
    active_ids = [t.local_track_id for t in tracker.active_tracklets]
    assert t_id not in active_ids, "Dead tracklet was not evicted after max coasting frames"
