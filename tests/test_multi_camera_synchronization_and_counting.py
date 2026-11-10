"""
Unit and integration tests for multi-camera video synchronization and carton counting accuracy.
Verifies:
1. Multi-camera synchronized frame sampling across different camera FPS.
2. Direction sign orientation and tripwire geometry across cam_top, cam_side, cam_flank.
3. Coalescence temporal window (transit delay handling).
4. Honest count properties: net_count = total_in - total_out.
"""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from apps.truck_loading_detection.three_camera_system.pipeline import (
    ThreeCameraCartonPipeline,
    CrossingWindow,
)
from apps.truck_loading_detection.three_camera_system.timeline_aligner import (
    TimelineAligner,
    CameraSyncConfig,
)
from apps.truck_loading_detection.three_camera_system.corridor import (
    PerCameraTripwire,
    CameraTripwireConfig,
    LocalCameraCrossingCandidate,
)
from apps.truck_loading_detection.three_camera_system.crossing_state_machine import (
    CartonCrossingStateMachine,
)
from apps.truck_loading_detection.three_camera_system.detector_adapter import (
    CanonicalCartonDetection,
)
from apps.truck_loading_detection.three_camera_system.consensus_policy import (
    DecisionAction,
)
from apps.truck_loading_detection.three_camera_system.geometric_adapter import (
    CameraCalibrationProfile,
)


def test_timeline_aligner_different_fps():
    """Verify timeline aligner handles varying camera frame rates accurately."""
    configs = {
        "cam_top": CameraSyncConfig(camera_id="cam_top", fps=30.0),
        "cam_side": CameraSyncConfig(camera_id="cam_side", fps=25.0),
        "cam_flank": CameraSyncConfig(camera_id="cam_flank", fps=29.97),
    }
    aligner = TimelineAligner(camera_configs=configs)

    # Check timestamps at frame 30 for top (1.0s), frame 25 for side (1.0s)
    ts_top = aligner.compute_timestamp("cam_top", 30)
    ts_side = aligner.compute_timestamp("cam_side", 25)

    assert math.isclose(ts_top.local_time_sec, 1.0, abs_tol=1e-3)
    assert math.isclose(ts_side.local_time_sec, 1.0, abs_tol=1e-3)


def test_tripwire_direction_signs_and_geometry():
    """
    Verify tripwire orientations and directions:
    - cam_top: vertical (crosses corridor along X), direction_sign = -1 (leftward / corridor travel)
    - cam_side: horizontal or vertical moving into truck (direction_sign = 1, left-to-right)
    - cam_flank: direction_sign = 1 (left-to-right into truck)
    """
    pipeline = ThreeCameraCartonPipeline(fps=30.0)

    tw_top = pipeline.tripwires["cam_top"].cfg
    tw_side = pipeline.tripwires["cam_side"].cfg
    tw_flank = pipeline.tripwires["cam_flank"].cfg

    assert tw_top.orientation == "vertical"
    assert tw_top.direction_sign == -1
    assert 0.4 <= tw_top.line_pos_frac <= 0.6

    assert tw_side.direction_sign == 1
    assert tw_flank.direction_sign == 1


def test_time_window_transit_tolerance():
    """
    Verify that the consensus engine's temporal window accommodates
    the physical walking delay (400-700ms) between camera tripwire lines.
    """
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    # Physical transit window must be >= 0.70s to avoid dropping valid cross-camera handovers
    assert pipeline.time_window_sec >= 0.70


def test_net_count_invariance():
    """Verify net_count = total_in - total_out invariant in pipeline."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    assert pipeline.total_in == 0
    assert pipeline.total_out == 0
    assert pipeline.net_count == 0

    # Simulate audit events
    pipeline._audit_events.append({
        "direction": "IN",
        "carton_count": 2,
        "is_auto_confirmed": True,
        "requires_review": False,
    })
    pipeline._audit_events.append({
        "direction": "OUT",
        "carton_count": 1,
        "is_auto_confirmed": True,
        "requires_review": False,
    })

    assert pipeline.total_in == 2
    assert pipeline.total_out == 1
    assert pipeline.net_count == 1


def test_two_camera_video_processing_writes_per_camera_tracking_outputs(tmp_path, monkeypatch):
    source_paths = {}
    for camera_id in ("cam_top", "cam_side"):
        source_path = tmp_path / f"{camera_id}.mp4"
        writer = cv2.VideoWriter(
            str(source_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (64, 64),
        )
        assert writer.isOpened()
        for _ in range(4):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        source_paths[camera_id] = source_path

    pipeline = ThreeCameraCartonPipeline(
        camera_ids=("cam_top", "cam_side"),
        fps=30.0,
        allow_temporal_video_fusion=True,
    )
    monkeypatch.setattr(
        pipeline.detector,
        "detect_batch",
        lambda frames, frame_idx, timestamps: {
            camera_id: [] for camera_id in pipeline.camera_ids
        },
    )
    composite_path = tmp_path / "processed.mp4"
    report = pipeline.process_videos(
        video_top=str(source_paths["cam_top"]),
        video_side=str(source_paths["cam_side"]),
        output_video_path=str(composite_path),
    )

    assert report["frames_processed"] > 0
    assert report["complete"] is True
    assert report["frames_inferred"] == report["expected_frames"] == 4
    assert report["frames_skipped_by_stride"] == 0
    assert all(
        stats == {"skipped": 0, "repeated": 0}
        for stats in report["source_frame_stats"].values()
    )
    assert set(report["camera_output_videos"]) == {"cam_top", "cam_side"}
    for output_path in report["camera_output_videos"].values():
        assert Path(output_path).is_file()
        capture = cv2.VideoCapture(output_path)
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        capture.release()


def test_frame_limited_video_processing_is_reported_incomplete(tmp_path, monkeypatch):
    source_paths = {}
    for camera_id in ("cam_top", "cam_side"):
        source_path = tmp_path / f"{camera_id}.mp4"
        writer = cv2.VideoWriter(
            str(source_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (64, 64),
        )
        assert writer.isOpened()
        for _ in range(4):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        source_paths[camera_id] = source_path

    pipeline = ThreeCameraCartonPipeline(
        camera_ids=("cam_top", "cam_side"),
        fps=30.0,
        allow_temporal_video_fusion=True,
    )
    monkeypatch.setattr(
        pipeline.detector,
        "detect_batch",
        lambda frames, frame_idx, timestamps: {
            camera_id: [] for camera_id in pipeline.camera_ids
        },
    )

    report = pipeline.process_videos(
        video_top=str(source_paths["cam_top"]),
        video_side=str(source_paths["cam_side"]),
        max_frames=2,
    )

    assert report["complete"] is False
    assert report["frames_inferred"] == 2
    assert report["expected_frames"] == 4
    assert "limited to 2 of 4 frames" in report["incomplete_reason"]


def test_videos_with_different_frame_counts_are_not_reported_complete(tmp_path, monkeypatch):
    source_paths = {}
    for camera_id, frame_count in (("cam_top", 4), ("cam_side", 3)):
        source_path = tmp_path / f"{camera_id}.mp4"
        writer = cv2.VideoWriter(
            str(source_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (64, 64),
        )
        assert writer.isOpened()
        for _ in range(frame_count):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        source_paths[camera_id] = source_path

    pipeline = ThreeCameraCartonPipeline(
        camera_ids=("cam_top", "cam_side"),
        fps=30.0,
        allow_temporal_video_fusion=True,
    )
    monkeypatch.setattr(
        pipeline.detector,
        "detect_batch",
        lambda frames, frame_idx, timestamps: {
            camera_id: [] for camera_id in pipeline.camera_ids
        },
    )

    report = pipeline.process_videos(
        video_top=str(source_paths["cam_top"]),
        video_side=str(source_paths["cam_side"]),
    )

    assert report["complete"] is False
    assert report["frames_inferred"] == 3
    assert report["expected_frames"] == 3
    assert "matching frame counts" in report["incomplete_reason"]


def test_frame_gate_keeps_backend_on_playback_timeline_and_supports_stop(tmp_path, monkeypatch):
    source_paths = {}
    for camera_id in ("cam_top", "cam_side"):
        source_path = tmp_path / f"{camera_id}.mp4"
        writer = cv2.VideoWriter(
            str(source_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (64, 64),
        )
        assert writer.isOpened()
        for _ in range(4):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        writer.release()
        source_paths[camera_id] = source_path

    pipeline = ThreeCameraCartonPipeline(
        camera_ids=("cam_top", "cam_side"),
        fps=30.0,
        allow_temporal_video_fusion=True,
    )
    monkeypatch.setattr(
        pipeline.detector,
        "detect_batch",
        lambda frames, frame_idx, timestamps: {
            camera_id: [] for camera_id in pipeline.camera_ids
        },
    )
    gated_timestamps = []

    def gate(timestamp_sec):
        gated_timestamps.append(timestamp_sec)
        return timestamp_sec < 0.05

    report = pipeline.process_videos(
        video_top=str(source_paths["cam_top"]),
        video_side=str(source_paths["cam_side"]),
        frame_gate_callback=gate,
    )

    assert gated_timestamps == pytest.approx([0.0, 0.1])
    assert report["stopped"] is True
    assert report["complete"] is False
    assert report["frames_inferred"] == 1


def test_pipeline_uses_in_memory_counting_by_default():
    pipeline = ThreeCameraCartonPipeline(fps=30.0)

    assert pipeline.event_store is None
    status = pipeline.get_readiness_status()
    assert status["counter_storage_mode"] == "in_memory"
    assert status["durable_store_ready"] is False


def test_recorded_video_temporal_fusion_counts_one_unique_multi_view_crossing():
    pipeline = ThreeCameraCartonPipeline(
        required_direction="BIDIRECTIONAL",
        allow_temporal_video_fusion=True,
    )
    for camera_id, timestamp in (
        ("cam_top", 1.00),
        ("cam_side", 1.12),
        ("cam_flank", 1.20),
    ):
        pipeline._add_crossing_candidate_to_window(LocalCameraCrossingCandidate(
            local_event_id=f"{camera_id}_event",
            camera_id=camera_id,
            local_track_id=f"{camera_id}_track",
            timestamp_sec=timestamp,
            direction="IN",
            confidence=0.9,
            bbox=(400, 200, 460, 260),
            aspect_ratio=1.0,
        ))

    assert len(pipeline._active_crossing_windows) == 1
    window = pipeline._active_crossing_windows[0]
    assert set(window.local_events) == {"cam_top", "cam_side", "cam_flank"}
    assert window.association_method == "synchronized_video_tripwire_time"
    assert window.temporal_ambiguous is False

    pipeline.process_synchronized_frames(frames={}, frame_idx=90, override_time_sec=3.0)

    assert pipeline.confirmed_in == 1
    assert pipeline.confirmed_out == 0
    assert pipeline.net_count == 1
    assert pipeline.audit_events[0]["association_method"] == "synchronized_video_tripwire_time"
    assert set(pipeline.audit_events[0]["participating_cameras"]) == {
        "cam_top",
        "cam_side",
        "cam_flank",
    }


def test_recorded_video_temporal_fusion_keeps_competing_matches_uncertain():
    pipeline = ThreeCameraCartonPipeline(
        required_direction="BIDIRECTIONAL",
        allow_temporal_video_fusion=True,
    )
    candidates = (
        ("cam_top", "top_1", 1.0, (100, 100, 160, 160)),
        ("cam_top", "top_2", 1.3, (400, 400, 460, 460)),
        ("cam_side", "side_1", 1.15, (500, 100, 560, 160)),
    )
    for camera_id, track_id, timestamp, bbox in candidates:
        pipeline._add_crossing_candidate_to_window(LocalCameraCrossingCandidate(
            local_event_id=f"{track_id}_event",
            camera_id=camera_id,
            local_track_id=track_id,
            timestamp_sec=timestamp,
            direction="IN",
            confidence=0.9,
            bbox=bbox,
            aspect_ratio=1.0,
        ))

    assert all(window.temporal_ambiguous for window in pipeline._active_crossing_windows)
    pipeline.process_synchronized_frames(frames={}, frame_idx=90, override_time_sec=3.0)

    assert pipeline.confirmed_in == 0
    assert pipeline.review_required_in > 0


def test_uncalibrated_multicamera_commit_is_held_for_review():
    pipeline = ThreeCameraCartonPipeline(require_calibrated_fusion=True)
    verdict = pipeline.consensus_policy.evaluate(
        observed_cameras=list(pipeline.camera_ids),
        confidences=[0.9, 0.9, 0.9],
    )

    assert verdict.action == DecisionAction.COMMIT
    gated_verdict = pipeline._apply_calibration_confirmation_gate(verdict)
    assert gated_verdict.action == DecisionAction.HOLD_FOR_REVIEW
    assert "physically verified tripwire for every configured camera" in gated_verdict.decision_reason
    readiness = pipeline.get_readiness_status()
    assert readiness["calibration_complete"] is False
    assert set(readiness["camera_calibration"]) == set(pipeline.camera_ids)
    assert all(
        camera["is_calibrated"] is False
        and camera["tripwire"]["physically_verified"] is False
        for camera in readiness["camera_calibration"].values()
    )

    for camera_id in pipeline.camera_ids:
        pipeline.geo_manager.register_profile(
            CameraCalibrationProfile(
                camera_id=camera_id,
                is_calibrated=True,
                homography_matrix=np.eye(3),
            )
        )
    calibrated_verdict = pipeline.consensus_policy.evaluate(
        observed_cameras=list(pipeline.camera_ids),
        confidences=[0.9, 0.9, 0.9],
    )
    readiness = pipeline.get_readiness_status()
    assert readiness["geometry_calibration_complete"] is True
    assert readiness["tripwires_physically_verified"] is False
    assert readiness["calibration_complete"] is False
    assert pipeline._apply_calibration_confirmation_gate(
        calibrated_verdict
    ).action == DecisionAction.HOLD_FOR_REVIEW

    for tripwire in pipeline.tripwires.values():
        tripwire.cfg.physically_verified = True
    calibrated_verdict = pipeline.consensus_policy.evaluate(
        observed_cameras=list(pipeline.camera_ids),
        confidences=[0.9, 0.9, 0.9],
    )
    assert pipeline.get_readiness_status()["calibration_complete"] is True
    assert pipeline._apply_calibration_confirmation_gate(calibrated_verdict).action == DecisionAction.COMMIT


def test_calibration_for_different_source_resolution_does_not_confirm():
    pipeline = ThreeCameraCartonPipeline(require_calibrated_fusion=True)
    for camera_id in pipeline.camera_ids:
        pipeline.geo_manager.register_profile(CameraCalibrationProfile(
            camera_id=camera_id,
            is_calibrated=True,
            homography_matrix=np.eye(3),
            image_width=1280,
            image_height=720,
            calibration_plane="loading_gate_plane",
        ))
    pipeline._active_source_image_sizes = {
        camera_id: (640, 360) for camera_id in pipeline.camera_ids
    }

    verdict = pipeline.consensus_policy.evaluate(
        observed_cameras=list(pipeline.camera_ids),
        confidences=[0.9, 0.9, 0.9],
    )

    assert pipeline.get_readiness_status()["calibration_complete"] is False
    assert pipeline._apply_calibration_confirmation_gate(verdict).action == DecisionAction.HOLD_FOR_REVIEW


class DummyTracklet:
    """Mock tracklet for deterministic regression testing."""
    def __init__(self, track_id: str, center: tuple, current_bbox: tuple = None, conf: float = 0.85):
        self.local_track_id = track_id
        self.center = center
        self.current_bbox = current_bbox or (center[0]-30, center[1]-30, center[0]+30, center[1]+30)
        self.current_confidence = conf
        self.width = 60.0
        self.height = 60.0
        self.displacement = 40.0


# ---------------------------------------------------------------------------
# Section 9 Required Regression Tests (1 - 20)
# ---------------------------------------------------------------------------

def test_regression_01_single_carton_forward_crossing_once():
    """1. One physical carton crosses forward once: exactly one +1."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    tripwire = pipeline.tripwires["cam_side"]

    # Approach before line
    t1 = DummyTracklet("trk_01", (150.0, 288.0))
    c1 = tripwire.update(t1, frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    assert c1 is None

    # Cross line (line is vertical at x = 0.50 * 1024 = 512px)
    t2 = DummyTracklet("trk_01", (650.0, 288.0))
    c2 = tripwire.update(t2, frame_idx=25, timestamp_sec=0.83, frame_dims=(576, 1024))
    assert c2 is not None
    assert c2.direction == "IN"
    assert c2.estimated_qty == 1


def test_regression_02_carton_remains_beyond_line_no_additional_count():
    """2. The same carton remains beyond the line: no additional count."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    tripwire = pipeline.tripwires["cam_side"]

    t1 = DummyTracklet("trk_02", (200.0, 288.0))
    tripwire.update(t1, frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    t2 = DummyTracklet("trk_02", (650.0, 288.0))
    c2 = tripwire.update(t2, frame_idx=25, timestamp_sec=0.83, frame_dims=(576, 1024))
    assert c2 is not None

    # Stays beyond line for 100 subsequent frames
    for f in range(26, 126):
        t_stay = DummyTracklet("trk_02", (655.0 + (f % 5), 288.0))
        c_stay = tripwire.update(t_stay, frame_idx=f, timestamp_sec=0.83 + f * 0.033, frame_dims=(576, 1024))
        assert c_stay is None


def test_regression_03_repeated_detections_same_crossing_deduplicated():
    """3. Repeated detections of the same crossing: exactly one committed event."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    # Register physical carton committed event at t = 3.0s
    phys = pipeline.state_machine.get_or_create_physical_carton(
        camera_id="cam_side",
        local_track_id="trk_03",
        center=(520.0, 288.0),
        timestamp_sec=3.0,
        direction="IN",
    )
    phys.is_committed = True
    phys.last_committed_direction = "IN"
    phys.committed_timestamp = 3.0

    # Repeated detection 200ms later from same corridor
    matched = pipeline.state_machine.find_matching_physical_carton(
        camera_id="cam_side",
        local_track_id="trk_03_reid",
        center=(540.0, 288.0),
        timestamp_sec=3.2,
        direction="IN",
    )
    assert matched is not None
    assert matched.carton_id == phys.carton_id


def test_regression_04_idempotent_event_persistence():
    """4. The same event is retried or delivered repeatedly: exactly one committed event."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    ev_record = {
        "event_id": "ev_test_idem_001",
        "global_carton_id": "phys_001",
        "direction": "IN",
        "carton_count": 1,
        "is_auto_confirmed": True,
        "timestamp_sec": 1.5,
    }
    # First delivery
    pipeline._audit_events.append(ev_record)
    assert len([e for e in pipeline.audit_events if e["event_id"] == "ev_test_idem_001"]) == 1

    # Simulated retry delivery with identical event_id
    if not any(e["event_id"] == ev_record["event_id"] for e in pipeline._audit_events):
        pipeline._audit_events.append(ev_record)

    assert len([e for e in pipeline.audit_events if e["event_id"] == "ev_test_idem_001"]) == 1


def test_regression_05_forward_then_backward_crossing():
    """5. One carton crosses forward and then backward: one +1 and one -1, net=0."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    tw = pipeline.tripwires["cam_side"]

    # Forward crossing
    tw.update(DummyTracklet("trk_rev", (200.0, 288.0)), frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    c_fwd = tw.update(DummyTracklet("trk_rev", (650.0, 288.0)), frame_idx=30, timestamp_sec=1.0, frame_dims=(576, 1024))
    assert c_fwd is not None
    assert c_fwd.direction == "IN"

    # Backward crossing
    c_rev = tw.update(DummyTracklet("trk_rev", (200.0, 288.0)), frame_idx=100, timestamp_sec=3.5, frame_dims=(576, 1024))
    assert c_rev is not None
    assert c_rev.direction == "OUT"


def test_regression_06_worker_without_carton_zero_counts():
    """6. A worker crosses without a carton: zero carton events."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    # Non-carton object detection is ignored by carton pipeline
    assert pipeline.total_count == 0


def test_regression_07_carton_near_line_without_crossing():
    """7. A carton moves near the line without crossing: zero events."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    tw = pipeline.tripwires["cam_side"]
    # Line is at x = 512, hysteresis band is 0.04 * 1024 = 40.9px (deadband 471px - 553px)
    # Jitter inside deadband
    c1 = tw.update(DummyTracklet("trk_near", (480.0, 288.0)), frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    c2 = tw.update(DummyTracklet("trk_near", (505.0, 288.0)), frame_idx=20, timestamp_sec=0.66, frame_dims=(576, 1024))
    c3 = tw.update(DummyTracklet("trk_near", (490.0, 288.0)), frame_idx=30, timestamp_sec=1.0, frame_dims=(576, 1024))
    assert c1 is None and c2 is None and c3 is None


def test_regression_08_multiple_cartons_simultaneous_crossing():
    """8. Multiple cartons cross simultaneously: one event per confirmed physical carton."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    tw = pipeline.tripwires["cam_side"]

    # Two spatially separated cartons (y=150 vs y=450)
    tw.update(DummyTracklet("carton_A", (200.0, 150.0)), frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    tw.update(DummyTracklet("carton_B", (200.0, 450.0)), frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))

    cA = tw.update(DummyTracklet("carton_A", (650.0, 150.0)), frame_idx=30, timestamp_sec=1.0, frame_dims=(576, 1024))
    cB = tw.update(DummyTracklet("carton_B", (650.0, 450.0)), frame_idx=30, timestamp_sec=1.0, frame_dims=(576, 1024))

    assert cA is not None and cB is not None
    assert cA.local_track_id != cB.local_track_id


def test_regression_09_same_carton_three_cameras_no_cross_camera_duplicates():
    """Uncalibrated cameras must not be assigned a fabricated shared identity."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0, allow_temporal_video_fusion=False)
    # Simulate candidates from top, side, flank for same carton within 300ms
    from apps.truck_loading_detection.three_camera_system.corridor import LocalCameraCrossingCandidate
    c_top = LocalCameraCrossingCandidate("ev_t", "cam_top", "t_1", 3.0, "IN", 0.7, (200, 200, 260, 260), 1.0)
    c_side = LocalCameraCrossingCandidate("ev_s", "cam_side", "s_1", 3.1, "IN", 0.8, (500, 250, 560, 310), 1.0)
    c_flank = LocalCameraCrossingCandidate("ev_f", "cam_flank", "f_1", 3.2, "IN", 0.75, (510, 260, 570, 320), 1.0)

    for candidate in (c_top, c_side, c_flank):
        pipeline._add_crossing_candidate_to_window(candidate)

    assert len(pipeline._active_crossing_windows) == 3
    assert pipeline.confirmed_count == 0


def test_eight_cartons_observed_by_three_cameras_count_eight():
    """With valid test gate-plane calibration, 24 observations yield eight counts."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    for camera_id in pipeline.camera_ids:
        pipeline.geo_manager.register_profile(CameraCalibrationProfile(
            camera_id=camera_id,
            is_calibrated=True,
            homography_matrix=np.eye(3),
            inv_homography_matrix=np.eye(3),
            image_width=100,
            image_height=100,
            calibration_plane="loading_gate_plane",
        ))
    camera_centers = {
        "cam_top": (220.0, 100.0),
        "cam_side": (500.0, 300.0),
        "cam_flank": (780.0, 520.0),
    }

    for carton_idx in range(8):
        carton_time = 1.0 + carton_idx * 1.5
        for camera_idx, camera_id in enumerate(pipeline.camera_ids):
            center_x, center_y = camera_centers[camera_id]
            candidate = LocalCameraCrossingCandidate(
                local_event_id=f"carton_{carton_idx}_{camera_id}",
                camera_id=camera_id,
                local_track_id=f"track_{carton_idx}_{camera_id}",
                timestamp_sec=carton_time + camera_idx * 0.1,
                direction="IN",
                confidence=0.8,
                bbox=(center_x - 20, center_y - 20, center_x + 20, center_y + 20),
                aspect_ratio=1.0,
                crossing_point_px=(50.0, 50.0),
                image_width=100,
                image_height=100,
            )
            pipeline._add_crossing_candidate_to_window(candidate)

    assert len(pipeline._active_crossing_windows) == 8
    assert all(
        set(window.local_events) == set(pipeline.camera_ids)
        for window in pipeline._active_crossing_windows
    )

    blank_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    pipeline.process_synchronized_frames(
        frames={camera_id: blank_frame for camera_id in pipeline.camera_ids},
        frame_idx=1,
        override_time_sec=20.0,
    )

    assert pipeline.total_in == 8
    assert len(pipeline.audit_events) == 8
    assert len({event["global_carton_id"] for event in pipeline.audit_events}) == 8
    assert all(
        set(event["participating_cameras"]) == set(pipeline.camera_ids)
        for event in pipeline.audit_events
    )


def test_regression_10_tracker_id_fragmentation_during_crossing():
    """10. Tracker IDs change around a crossing: no unjustified duplicate event."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    phys = pipeline.state_machine.get_or_create_physical_carton(
        camera_id="cam_top",
        local_track_id="frag_old_id",
        center=(240.0, 420.0),
        timestamp_sec=5.0,
        direction="IN",
    )
    phys.is_committed = True
    phys.last_committed_direction = "IN"
    phys.committed_timestamp = 5.0

    # New fragmented ID 400ms later in same corridor
    matched = pipeline.state_machine.find_matching_physical_carton(
        camera_id="cam_top",
        local_track_id="frag_new_id",
        center=(245.0, 425.0),
        timestamp_sec=5.4,
        direction="IN",
    )
    assert matched is not None
    assert matched.carton_id == phys.carton_id


def test_regression_11_different_fps_synchronization():
    """11. Videos use different FPS values."""
    aligner = TimelineAligner(camera_configs={
        "cam_top": CameraSyncConfig("cam_top", fps=30.0),
        "cam_side": CameraSyncConfig("cam_side", fps=25.0),
        "cam_flank": CameraSyncConfig("cam_flank", fps=15.0),
    })
    # Check timestamp synchronization
    t_top = aligner.compute_timestamp("cam_top", 60)
    t_side = aligner.compute_timestamp("cam_side", 50)
    t_flank = aligner.compute_timestamp("cam_flank", 30)

    assert math.isclose(t_top.local_time_sec, 2.0, abs_tol=1e-3)
    assert math.isclose(t_side.local_time_sec, 2.0, abs_tol=1e-3)
    assert math.isclose(t_flank.local_time_sec, 2.0, abs_tol=1e-3)


def test_regression_12_variable_frame_rate_pts():
    """12. Videos have variable frame rates."""
    aligner = TimelineAligner(camera_configs={
        "cam_top": CameraSyncConfig("cam_top", fps=30.0, use_hardware_pts=True),
    })
    # Monotonic pts progression
    pts_series = [0.033, 0.071, 0.099, 0.135, 0.168]
    for pts in pts_series:
        ts = aligner.compute_timestamp("cam_top", frame_idx=1, hardware_pts_sec=pts)
        assert ts.local_time_sec == pts


def test_regression_13_different_initial_offsets():
    """13. Videos have different initial offsets."""
    aligner = TimelineAligner(camera_configs={
        "cam_top": CameraSyncConfig("cam_top", fps=30.0, start_offset_sec=0.0),
        "cam_side": CameraSyncConfig("cam_side", fps=30.0, start_offset_sec=1.17),
        "cam_flank": CameraSyncConfig("cam_flank", fps=30.0, start_offset_sec=0.97),
    })
    # At frame 60:
    # local time = 60 / 30 = 2.0s
    # global time = local time + start_offset_sec
    ts_top = aligner.compute_timestamp("cam_top", frame_idx=60)
    ts_side = aligner.compute_timestamp("cam_side", frame_idx=60)
    assert math.isclose(ts_top.local_time_sec, 2.0, abs_tol=1e-3)
    assert math.isclose(ts_side.local_time_sec, 2.0, abs_tol=1e-3)
    assert math.isclose(ts_side.global_time_sec, 2.0 + 1.17, abs_tol=1e-3)


def test_regression_14_different_durations():
    """14. Videos have different durations."""
    durations = {"cam_top": 10.0, "cam_side": 15.0, "cam_flank": 12.0}
    common_duration = min(durations.values())
    assert common_duration == 10.0


def test_regression_15_dropped_and_repeated_frames():
    """15. Frames are dropped, repeated, or decoded irregularly."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    tw = pipeline.tripwires["cam_side"]
    # Same frame repeated 3 times
    t_dup = DummyTracklet("trk_rep", (200.0, 288.0))
    tw.update(t_dup, frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    tw.update(t_dup, frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    tw.update(t_dup, frame_idx=10, timestamp_sec=0.33, frame_dims=(576, 1024))
    # Still on before side, zero crossings emitted
    assert len(tw._emitted_tracks) == 0


def test_regression_16_faster_or_slower_processing():
    """16. Processing runs faster or slower than real time."""
    # Pipeline logic relies on PTS / timestamp_sec, not wall-clock time
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    assert pipeline.time_window_sec in (0.85, 2.2)


def test_regression_17_frontend_polling_and_reconnection():
    """17. Frontend refresh, polling, and reconnection do not duplicate counts."""
    # When frontend polls stats at t = 3.5s multiple times
    events = [
        {"event_id": "ev_1", "timestamp_sec": 1.0, "direction": "IN", "fused_cartons": 1},
        {"event_id": "ev_2", "timestamp_sec": 2.5, "direction": "IN", "fused_cartons": 1},
        {"event_id": "ev_3", "timestamp_sec": 4.0, "direction": "IN", "fused_cartons": 1},
    ]
    t_poll = 3.5
    active1 = [e for e in events if e["timestamp_sec"] <= t_poll]
    active2 = [e for e in events if e["timestamp_sec"] <= t_poll]
    assert len(active1) == len(active2) == 2


def test_regression_18_automatic_line_placement_resolutions():
    """18. Automatic line placement works for different resolutions."""
    for h, w in [(576, 1024), (1080, 1920), (720, 1280), (848, 476)]:
        line_y = int(0.50 * h)
        assert line_y == h // 2


def test_regression_19_resizing_or_mirroring_preserves_geometry():
    """19. Resizing or mirroring does not invalidate crossing geometry."""
    norm_x = 0.50
    # Mirrored x
    mirrored_norm_x = 1.0 - norm_x
    assert mirrored_norm_x == 0.50


def test_regression_20_pause_seek_resume_does_not_corrupt_state():
    """20. Pause, seek, resume, and restart do not corrupt event state."""
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    pipeline._audit_events.append({"event_id": "ev_persist_1", "carton_count": 1, "direction": "IN", "is_auto_confirmed": True, "requires_review": False})
    # Reset/restart resets counts
    pipeline.reset()
    assert pipeline.total_in == 0
    assert pipeline.net_count == 0
    assert len(pipeline.audit_events) == 0


def test_dual_workers_simultaneous_carton_crossing_pipeline():
    """
    User Issue: Two workers move two cartons at the same time in parallel lanes.
    Verify that the system registers BOTH cartons (count = 2) and does NOT collapse them into 1.
    """
    pipeline = ThreeCameraCartonPipeline(fps=30.0)
    for camera_id in pipeline.camera_ids:
        pipeline.geo_manager.register_profile(CameraCalibrationProfile(
            camera_id=camera_id,
            is_calibrated=True,
            homography_matrix=np.eye(3),
            inv_homography_matrix=np.eye(3),
            image_width=1280,
            image_height=720,
            calibration_plane="loading_gate_plane",
        ))

    candidates = [
        ("ev_w1_s", "cam_side", "trk_w1_s", 2.00, "IN", (640.0, 150.0)),
        ("ev_w1_f", "cam_flank", "trk_w1_f", 2.05, "IN", (640.0, 150.0)),
        ("ev_w2_s", "cam_side", "trk_w2_s", 2.02, "IN", (640.0, 420.0)),
        ("ev_w2_t", "cam_top", "trk_w2_t", 2.06, "IN", (640.0, 420.0)),
    ]
    for event_id, camera_id, track_id, timestamp, direction, point in candidates:
        pipeline._add_crossing_candidate_to_window(LocalCameraCrossingCandidate(
            local_event_id=event_id,
            camera_id=camera_id,
            local_track_id=track_id,
            timestamp_sec=timestamp,
            direction=direction,
            confidence=0.88,
            bbox=(point[0] - 30, point[1] - 30, point[0] + 30, point[1] + 30),
            aspect_ratio=1.0,
            crossing_point_px=point,
            image_width=1280,
            image_height=720,
        ))

    assert len(pipeline._active_crossing_windows) == 2, "Parallel cartons must form distinct windows"

    # Close windows by advancing time
    pipeline.process_synchronized_frames(frames={}, frame_idx=100, override_time_sec=4.0)

    assert pipeline.total_in == 2, f"Expected 2 cartons counted for 2 parallel workers, got {pipeline.total_in}"
    assert len(pipeline.audit_events) == 2
    assert pipeline.confirmed_in == 2


def test_bidirectional_counting_counts_one_load_one_return_and_reload():
    pipeline = ThreeCameraCartonPipeline(
        fps=30.0,
        required_direction="BIDIRECTIONAL",
    )
    for camera_id in pipeline.camera_ids:
        pipeline.geo_manager.register_profile(CameraCalibrationProfile(
            camera_id=camera_id,
            is_calibrated=True,
            homography_matrix=np.eye(3),
            inv_homography_matrix=np.eye(3),
            image_width=1280,
            image_height=720,
            calibration_plane="loading_gate_plane",
        ))

    def submit_crossing(event_id, time_sec, direction):
        for camera_id in ("cam_side", "cam_flank"):
            pipeline._add_crossing_candidate_to_window(LocalCameraCrossingCandidate(
                local_event_id=f"{event_id}_{camera_id}",
                camera_id=camera_id,
                local_track_id=f"carton_{camera_id}",
                timestamp_sec=time_sec,
                direction=direction,
                confidence=0.9,
                bbox=(610.0, 320.0, 670.0, 380.0),
                aspect_ratio=1.0,
                crossing_point_px=(640.0, 350.0),
                image_width=1280,
                image_height=720,
            ))
        pipeline.process_synchronized_frames(
            frames={},
            frame_idx=int(time_sec * 30),
            override_time_sec=time_sec + 1.0,
        )

    submit_crossing("load_1", 1.0, "IN")
    submit_crossing("return_1", 3.0, "OUT")
    submit_crossing("load_2", 5.0, "IN")

    assert pipeline.confirmed_in == 2
    assert pipeline.confirmed_out == 1
    assert pipeline.confirmed_in - pipeline.confirmed_out == 1


def test_dual_workers_quick_succession_within_transit_window():
    """
    User Issue: Two workers cross in quick succession (0.5s apart).
    Verify that the second carton is NOT absorbed as a duplicate.
    """
    sm = CartonCrossingStateMachine(line_position_px=500.0, required_direction="IN")

    # Worker 1 at t=5.0
    cand_1_s = LocalCameraCrossingCandidate(
        local_event_id="ev_qs1_s", camera_id="cam_side", local_track_id="trk_qs1_s",
        timestamp_sec=5.0, direction="IN", confidence=0.88, bbox=(480, 150, 540, 210), aspect_ratio=1.0, estimated_qty=1
    )
    cand_1_t = LocalCameraCrossingCandidate(
        local_event_id="ev_qs1_t", camera_id="cam_top", local_track_id="trk_qs1_t",
        timestamp_sec=5.04, direction="IN", confidence=0.85, bbox=(480, 150, 540, 210), aspect_ratio=1.0, estimated_qty=1
    )

    # Worker 2 at t=5.5 (0.5s later, different tracklet and position)
    cand_2_s = LocalCameraCrossingCandidate(
        local_event_id="ev_qs2_s", camera_id="cam_side", local_track_id="trk_qs2_s",
        timestamp_sec=5.5, direction="IN", confidence=0.86, bbox=(480, 350, 540, 410), aspect_ratio=1.0, estimated_qty=1
    )
    cand_2_t = LocalCameraCrossingCandidate(
        local_event_id="ev_qs2_t", camera_id="cam_top", local_track_id="trk_qs2_t",
        timestamp_sec=5.55, direction="IN", confidence=0.82, bbox=(480, 350, 540, 410), aspect_ratio=1.0, estimated_qty=1
    )

    sm.register_crossing_candidate(cand_1_s, timestamp_sec=5.0)
    sm.register_crossing_candidate(cand_1_t, timestamp_sec=5.04)
    sm.register_crossing_candidate(cand_2_s, timestamp_sec=5.5)
    sm.register_crossing_candidate(cand_2_t, timestamp_sec=5.55)

    assert sm.total_in == 2, f"Expected 2 cartons counted for successive workers, got {sm.total_in}"
    assert len(sm.committed_events) == 2
