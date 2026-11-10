"""
Unit & Integration Tests for Phase 2: Per-Camera Processing.

Covers:
1. Independent carton detection and tracking across cameras (top, side, flank).
2. Tracklet IDs, timestamps, bounding boxes, confidence, motion history, and track lifecycle.
3. Three-camera frame alignment and synchronized processing.
4. Crossing-corridor zones (approach, threshold, commit) and direction reversal detection.
5. 6-second circular ring buffer, post-roll timing, and synchronized review clip export.
6. GPU/CPU processing profiles, failover behavior, dropped frames, latency, and memory telemetry.
"""

import os
import time
import pytest
import numpy as np

from apps.truck_loading_detection.three_camera_system.detector_adapter import (
    CanonicalCartonDetection,
    CartonDetectorAdapter,
)
from apps.truck_loading_detection.three_camera_system.per_camera_tracker import (
    CartonTracklet,
    PerCameraCartonTracker,
    TrackStatus,
)
from apps.truck_loading_detection.three_camera_system.timeline_aligner import (
    CameraSyncConfig,
    TimelineAligner,
)
from apps.truck_loading_detection.three_camera_system.corridor import (
    CorridorZone,
    CrossingCorridor,
    PerCameraCorridorEvidence,
)
from apps.truck_loading_detection.three_camera_system.ring_buffer import (
    BufferedFrame,
    SynchronizedRingBuffer,
)
from apps.truck_loading_detection.model_pipeline.engine_manager import (
    InferenceEngineManager,
    EngineBenchmarkResult,
)


# ===========================================================================
# 1 & 2: Independent Carton Detection & Tracklet Lifecycle
# ===========================================================================
def test_independent_three_camera_tracklet_isolation():
    """Verify each camera assigns scoped local track IDs and tracks independently."""
    trackers = {
        cid: PerCameraCartonTracker(camera_id=cid)
        for cid in ("cam_top", "cam_side", "cam_flank")
    }

    # Simulate 1 carton appearing in all 3 cameras at frame 1
    for cid, tracker in trackers.items():
        det = CanonicalCartonDetection(
            camera_id=cid,
            frame_idx=1,
            timestamp_sec=0.033,
            bbox=(100, 100, 200, 200),
            confidence=0.85,
        )
        active = tracker.update([det], frame_idx=1, timestamp_sec=0.033)
        assert len(active) == 1
        assert active[0].camera_id == cid
        assert active[0].local_track_id == f"{cid}_t0001"
        assert active[0].current_confidence == 0.85
        assert len(active[0].trajectory) == 1


def test_tracklet_lifecycle_and_motion_history():
    """Verify TENTATIVE -> CONFIRMED -> OCCLUDED -> TERMINATED transitions and velocity."""
    tracker = PerCameraCartonTracker(camera_id="cam_side", min_hits_to_confirm=2, max_invisible_frames=2)

    # Frame 1: Initial detection -> TENTATIVE
    d1 = CanonicalCartonDetection("cam_side", 1, 0.0, (100, 100, 150, 150), 0.9)
    trks = tracker.update([d1], frame_idx=1, timestamp_sec=0.0)
    assert trks[0].status == TrackStatus.TENTATIVE
    assert trks[0].displacement == 0.0

    # Frame 2: Moved 60px right in 0.1s -> CONFIRMED & positive velocity
    d2 = CanonicalCartonDetection("cam_side", 2, 0.10, (160, 100, 210, 150), 0.9)
    trks = tracker.update([d2], frame_idx=2, timestamp_sec=0.10)
    assert trks[0].status == TrackStatus.CONFIRMED
    vx, vy = trks[0].velocity
    assert vx == pytest.approx(600.0, abs=10.0)  # 60px / 0.1s = 600 px/s
    assert trks[0].displacement == pytest.approx(60.0, abs=1.0)

    # Frame 3: Disappeared -> OCCLUDED (coasting)
    trks = tracker.update([], frame_idx=3, timestamp_sec=0.15)
    assert len(trks) == 1
    assert trks[0].status == TrackStatus.OCCLUDED

    # Frame 4 & 5: Beyond max_invisible_frames -> TERMINATED
    tracker.update([], frame_idx=4, timestamp_sec=0.20)
    trks_final = tracker.update([], frame_idx=5, timestamp_sec=0.25)
    assert len(trks_final) == 0


# ===========================================================================
# 3: Three-Camera Frame Alignment
# ===========================================================================
def test_three_camera_frame_alignment_and_synchronization():
    """Verify common timeline alignment across 3 cameras with start offsets."""
    sync_configs = {
        "cam_top": CameraSyncConfig(camera_id="cam_top", fps=30.0, start_offset_sec=0.0),
        "cam_side": CameraSyncConfig(camera_id="cam_side", fps=30.0, start_offset_sec=0.10),
        "cam_flank": CameraSyncConfig(camera_id="cam_flank", fps=30.0, start_offset_sec=0.10),
    }
    aligner = TimelineAligner(sync_configs, max_inter_camera_delta_sec=0.15)

    ts_top = aligner.compute_timestamp("cam_top", frame_idx=30)
    ts_side = aligner.compute_timestamp("cam_side", frame_idx=30)
    ts_flank = aligner.compute_timestamp("cam_flank", frame_idx=30)

    assert ts_top.global_time_sec == pytest.approx(1.0, abs=1e-3)
    assert ts_side.global_time_sec == pytest.approx(1.10, abs=1e-3)
    assert ts_flank.global_time_sec == pytest.approx(1.10, abs=1e-3)

    is_aligned, skew, reasons = aligner.validate_multi_view_alignment({
        "cam_top": ts_top, "cam_side": ts_side, "cam_flank": ts_flank
    })
    assert is_aligned
    assert skew == pytest.approx(0.10, abs=1e-3)


# ===========================================================================
# 4: Crossing-Corridor Zones and Evidence
# ===========================================================================
def test_crossing_corridor_progression_and_reversal():
    """Verify spatial corridor progression and direction reversal detection."""
    corridor = CrossingCorridor(
        camera_id="cam_side",
        line_position_px=500.0,
        threshold_half_width_px=30.0,
        normal_direction=(1.0, 0.0),
    )

    # 1. Approach Zone (x = 400 < 470)
    ev1 = corridor.update_track_evidence("trk_01", (400.0, 300.0), (100.0, 0.0), timestamp_sec=1.0)
    assert ev1.initial_zone == CorridorZone.OUTSIDE_APPROACH
    assert ev1.current_zone == CorridorZone.OUTSIDE_APPROACH
    assert not ev1.is_valid_transit

    # 2. Enters Threshold Zone (x = 510 in [470, 530])
    ev2 = corridor.update_track_evidence("trk_01", (510.0, 300.0), (100.0, 0.0), timestamp_sec=1.5)
    assert ev2.current_zone == CorridorZone.THRESHOLD_BAND
    assert ev2.threshold_timestamp == 1.5

    # 3. Enters Commit Zone (x = 560 > 530)
    ev3 = corridor.update_track_evidence("trk_01", (560.0, 300.0), (100.0, 0.0), timestamp_sec=2.0)
    assert ev3.current_zone == CorridorZone.INSIDE_COMMIT
    assert ev3.commit_timestamp == 2.0
    assert ev3.transit_duration_sec == pytest.approx(1.0, abs=1e-3)
    assert ev3.is_valid_transit is True
    assert ev3.reversal_detected is False

    # 4. Separate carton reverses from threshold back to approach
    corridor.update_track_evidence("trk_rev", (450.0, 300.0), (50.0, 0.0), timestamp_sec=3.0)
    corridor.update_track_evidence("trk_rev", (500.0, 300.0), (50.0, 0.0), timestamp_sec=3.5)
    ev_rev = corridor.update_track_evidence("trk_rev", (420.0, 300.0), (-80.0, 0.0), timestamp_sec=4.0)
    assert ev_rev.reversal_detected is True
    assert ev_rev.is_valid_transit is False


# ===========================================================================
# 5: 6-Second Ring Buffer & Synchronized Review Clip
# ===========================================================================
def test_ring_buffer_capacity_and_review_clip_export(tmp_path):
    """Verify circular ring buffer keeps 6 seconds, handles post-roll, and renders MP4."""
    ring_buf = SynchronizedRingBuffer(
        camera_ids=("cam_top", "cam_side", "cam_flank"),
        buffer_duration_sec=6.0,
        fps=10.0,  # 60 frames capacity
    )
    assert ring_buf.capacity_frames == 60

    # Push 30 frames (3.0 seconds) for each camera
    dummy_frame = np.full((120, 160, 3), 100, dtype=np.uint8)
    for f_idx in range(30):
        t = f_idx / 10.0
        for cid in ("cam_top", "cam_side", "cam_flank"):
            ring_buf.push_frame(cid, f_idx, t, dummy_frame)

    assert ring_buf.buffer_sizes["cam_top"] == 30

    # Trigger review clip at t = 2.0s with 1.0s pre-roll and 0.5s post-roll
    out_clip = str(tmp_path / "test_review_clip.mp4")
    ring_buf.trigger_review_clip(
        event_id="test_event_001",
        trigger_time_sec=2.0,
        output_path=out_clip,
        pre_roll_sec=1.0,
        post_roll_sec=0.5,
    )

    # At t = 2.2s, post-roll (2.0 + 0.5 = 2.5s) has NOT elapsed -> should not export yet
    exported_early = ring_buf.poll_and_export_ready_clips(current_time_sec=2.2)
    assert len(exported_early) == 0

    # At t = 2.6s, post-roll has elapsed -> should export clip!
    exported_ready = ring_buf.poll_and_export_ready_clips(current_time_sec=2.6)
    assert len(exported_ready) == 1
    assert os.path.exists(out_clip)
    assert os.path.getsize(out_clip) > 0


# ===========================================================================
# 6: Hardware Profiles, Failover, Dropped Frames & Telemetry
# ===========================================================================
def test_hardware_profile_failover_and_telemetry():
    """Verify engine manager failover behavior and detector adapter telemetry."""
    # Test CPU fallback in InferenceEngineManager
    engine_mgr = InferenceEngineManager(
        preferred_profile="gpu_tensorrt",
        trt_engine_path="nonexistent_profile_a.engine",
        pytorch_model_path="apps/truck_loading_detection/best.pt",
    )
    assert engine_mgr.active_profile == "pytorch_cpu"
    assert engine_mgr.device == "cpu"

    # Test CartonDetectorAdapter telemetry
    adapter = CartonDetectorAdapter(
        model_path="apps/truck_loading_detection/best.pt",
        conf_threshold=0.20,
    )
    # Feed an invalid / empty frame -> should increment dropped_frames_count
    empty_det = adapter.detect(None, "cam_top", 1, 0.0)
    assert len(empty_det) == 0

    # Feed a valid dummy frame
    valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = adapter.detect(valid_frame, "cam_top", 2, 0.033)

    telemetry = adapter.get_telemetry()
    assert telemetry["total_frames_received"] == 2
    assert telemetry["total_frames_processed"] == 1
    assert telemetry["dropped_frames_count"] == 1
    assert telemetry["last_latency_ms"] > 0.0
    assert telemetry["memory_usage_mb"] > 0.0
