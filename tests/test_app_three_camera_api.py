import json
import io
import os
import sys
import time
from threading import Event
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_app_dir = Path(__file__).resolve().parent.parent / "apps" / "truck_loading_detection"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import app as app_module
from app import StreamState, app, multi_angle_jobs, state
from three_camera_system.pipeline import ThreeCameraCartonPipeline


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_available_videos_route(client):
    """Verify /api/3angle/available_videos returns recorded videos and defaults."""
    resp = client.get("/api/3angle/available_videos")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "recorded_videos" in data
    assert "top_candidates" in data
    assert "side_candidates" in data
    assert "flank_candidates" in data
    assert "defaults" in data

    # Check that scenario1 synced videos or standard aligned videos are present
    rec_vids = data["recorded_videos"]
    has_scenario = any("scenario1" in v for v in rec_vids)
    has_aligned = any("aligned" in v for v in rec_vids)
    assert has_scenario or has_aligned, f"No expected videos in recorded_videos: {rec_vids}"
    assert any(video["camera"] == "cam_top" for video in data["available_videos"])
    assert any(video["camera"] == "cam_side" for video in data["available_videos"])
    assert any(video["camera"] == "cam_flank" for video in data["available_videos"])


def test_uploaded_video_is_listed_for_its_assigned_camera(client, monkeypatch, tmp_path):
    monkeypatch.setitem(app.config, "UPLOAD_FOLDER", str(tmp_path))

    upload_response = client.post(
        "/api/3angle/upload",
        data={"slot": "top", "video": (io.BytesIO(b"video"), "warehouse_take.mp4")},
        content_type="multipart/form-data",
    )
    assert upload_response.status_code == 200
    filename = upload_response.get_json()["filename"]

    available = client.get("/api/3angle/available_videos").get_json()
    uploaded = next(video for video in available["available_videos"] if video["name"] == filename)
    assert uploaded["camera"] == "cam_top"
    assert uploaded["path"] == filename


def test_upload_rejects_unknown_camera_slot(client):
    response = client.post(
        "/api/3angle/upload",
        data={"slot": "../other", "video": (io.BytesIO(b"video"), "warehouse_take.mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "slot must be top, side, or flank" in response.get_json()["error"]


def test_dashboard_keeps_camera_previews_and_simple_recorded_video_controls(client):
    response = client.get("/")

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="valLoaded">0</span>' in html
    assert 'id="valReturned">0</span>' in html
    assert 'id="valNet">0</span>' in html
    assert 'id="livePreviewTop"' in html
    assert 'id="livePreviewSide"' in html
    assert 'id="livePreviewFlank"' in html
    assert 'id="countingStatus"' in html
    assert "Use at least two videos from the same loading session" in html
    assert "Start counting" in html
    assert "Live preview only. Automatic live counting is not available yet." in html
    assert "Final counts appear after processing completes." in html
    assert "Video processing was incomplete. No final count is available." in html
    assert "Flagged Events Review Queue" not in html
    assert "Pallet Scale Ground-Truth Cross-Check" not in html
    assert "Review Required" not in html
    assert 'value="review"' not in html
    assert "pending_review_count" not in html
    assert "Analyzing &amp; Correlating 3 Angles" not in html
    assert "Loop: ON" not in html
    assert "Resync" not in html
    assert 'id="recordedPlaybackControls"' in html
    assert "playbackControls.hidden = true" in html
    assert "playbackControls.hidden = false" in html
    assert "Start live counting (unavailable)" in html
    assert 'id="btnStopCounting"' in html
    assert 'id="btnResetCounting"' in html
    assert "Preview only. Select Start counting to count cartons." in html
    assert "frame_gate_callback" not in html
    assert "function drawTrackingOverlays" in html
    assert "function stopRecordedCounting" in html
    assert "Collapse Tray" not in html
    assert "Upload Mode" not in html
    assert "job.phase" not in html
    assert "Crossing Events" not in html
    assert "Net</span>" in html
    assert "94.5%" not in html


def test_stats_explains_zero_confirmed_counts_when_calibration_is_missing(client, monkeypatch):
    monkeypatch.setattr(state, "is_multi_angle", True)
    monkeypatch.setattr(
        state,
        "multi_angle_report",
        {
            "result": {"events": [], "calibration": {"calibration_complete": False}},
            "calibration": {"calibration_complete": False},
        },
    )
    monkeypatch.setattr(
        app_module.live_camera_manager,
        "status",
        lambda: {"active": False},
    )

    response = client.get("/api/stats?t=0&is_paused=1")

    assert response.status_code == 200
    data = response.get_json()
    assert data["total_in"] == data["total_out"] == data["net_count"] == 0
    assert data["counting_enabled"] is False
    assert "calibration" in data["counting_blocker"].lower()


def test_stats_accepts_only_associated_recorded_video_events_without_physical_calibration(client, monkeypatch):
    monkeypatch.setattr(state, "is_multi_angle", True)
    monkeypatch.setattr(state, "source", "recorded_output.mp4")
    monkeypatch.setattr(
        state,
        "multi_angle_report",
        {
            "processing_mode": "recorded_video",
            "calibration": {"calibration_complete": False},
            "result": {
                "processing_mode": "recorded_video",
                "calibration": {"calibration_complete": False},
                "events": [{
                    "fused_cartons": 1,
                    "direction": "IN",
                    "timestamp_sec": 1.0,
                    "is_auto_confirmed": True,
                    "requires_review": False,
                    "association_method": "synchronized_video_tripwire_time",
                    "participating_cameras": ["cam_top", "cam_side", "cam_flank"],
                    "consensus_status": "THREE_CAMERA_UNANIMOUS",
                }],
            },
        },
    )
    monkeypatch.setattr(
        app_module.live_camera_manager,
        "status",
        lambda: {
            "active": True,
            "ready": False,
            "counting_blocker": "Live preview camera is unavailable",
        },
    )

    response = client.get("/api/stats?t=2")

    assert response.status_code == 200
    data = response.get_json()
    assert data["recorded_video_mode"] is True
    assert data["counting_enabled"] is True
    assert data["counting_blocker"] is None
    assert data["total_in"] == data["net_count"] == 1


def test_app_tripwire_direction_defaults_match_pipeline(client):
    pipeline = ThreeCameraCartonPipeline(fps=30.0)

    assert state.tripwire_configs["cam_top"]["direction_sign"] == pipeline.tripwires["cam_top"].cfg.direction_sign
    assert state.tripwire_configs["cam_side"]["direction_sign"] == pipeline.tripwires["cam_side"].cfg.direction_sign
    assert state.tripwire_configs["cam_flank"]["direction_sign"] == pipeline.tripwires["cam_flank"].cfg.direction_sign


def test_set_live_cameras_accepts_frontend_camera_objects(client, monkeypatch):
    """The live-camera UI starts previews and locks automatic counts."""
    monkeypatch.setattr(state, "source", state.source)
    monkeypatch.setattr(state, "connection_status", state.connection_status)
    monkeypatch.setattr(state, "live_cameras", state.live_cameras.copy())
    monkeypatch.setattr(state, "cap", None)
    monkeypatch.setattr(state, "is_multi_angle", False)
    monkeypatch.setattr(state, "multi_angle_report", None)
    started_sources = {}
    monkeypatch.setattr(
        app_module.live_camera_manager,
        "start",
        lambda sources: started_sources.update(sources),
    )
    monkeypatch.setattr(
        app_module.live_camera_manager,
        "status",
        lambda: {
            "active": True,
            "ready": False,
            "connected_cameras": 0,
            "required_cameras": 3,
            "cameras": {},
            "counting_enabled": False,
            "counting_blocker": "Verified calibration is required for all three cameras",
        },
    )

    resp = client.post("/api/3angle/set_live_cameras", json={
        "cam_top": {"source": "rtsp://top/stream"},
        "cam_side": {"source": "0"},
        "cam_flank": {"source": "rtsp://flank/stream"},
    })

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["processing_mode"] == "three_camera_preview"
    assert "counting is not implemented" in data["message"]
    assert started_sources == {
        "cam_top": "rtsp://top/stream",
        "cam_side": 0,
        "cam_flank": "rtsp://flank/stream",
    }
    assert state.is_multi_angle is True
    assert state.multi_angle_report["calibration"]["calibration_complete"] is False
    assert state.source == ""


def test_set_live_cameras_rejects_missing_sources(client):
    resp = client.post("/api/3angle/set_live_cameras", json={
        "cam_top": {"source": "rtsp://top/stream"},
        "cam_side": {"source": "rtsp://side/stream"},
    })

    assert resp.status_code == 400
    assert "cam_flank" in resp.get_json()["error"]


def test_live_camera_feed_streams_jpeg_for_known_camera(client, monkeypatch):
    monkeypatch.setattr(
        app_module.live_camera_manager,
        "latest_jpeg",
        lambda camera_id: b"\xff\xd8test-jpeg",
    )

    response = client.get("/api/live_cameras/feed/cam_top", buffered=False)
    try:
        first_chunk = next(response.response)
    finally:
        response.close()

    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"
    assert b"Content-Type: image/jpeg" in first_chunk
    assert b"\xff\xd8test-jpeg" in first_chunk


def test_live_camera_feed_rejects_unknown_camera(client):
    response = client.get("/api/live_cameras/feed/cam_unknown")

    assert response.status_code == 404


def test_live_camera_status_route_does_not_expose_sources(client):
    response = client.get("/api/live_cameras/status")

    assert response.status_code == 200
    data = response.get_json()
    assert set(data["cameras"]) == {"cam_top", "cam_side", "cam_flank"}
    assert "source" not in data


def test_health_marks_live_cameras_degraded_until_all_connected(client, monkeypatch):
    monkeypatch.setattr(
        app_module.live_camera_manager,
        "status",
        lambda: {
            "active": True,
            "ready": False,
            "connected_cameras": 2,
            "required_cameras": 3,
            "cameras": {},
            "counting_enabled": False,
            "counting_blocker": "Live counting is not implemented",
        },
    )
    monkeypatch.setattr(state, "connection_status", "live")

    response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "degraded"
    assert data["counting_enabled"] is False
    assert data["live_cameras"]["connected_cameras"] == 2
    assert "Live counting is not implemented" in data["counting_blocker"]


def test_carton_model_override_rejects_unpinned_checkpoint(monkeypatch, tmp_path):
    unpinned_model = tmp_path / "unrelated.pt"
    unpinned_model.touch()
    monkeypatch.setenv("CARTON_MODEL_PATH", str(unpinned_model))

    with pytest.raises(ValueError, match="must point to the canonical"):
        StreamState()


def test_carton_model_override_accepts_only_canonical_path(monkeypatch):
    canonical_model = _app_dir / "best.pt"
    monkeypatch.setenv("CARTON_MODEL_PATH", str(canonical_model))

    stream_state = StreamState()
    assert Path(stream_state.model_name).resolve() == canonical_model.resolve()
    assert stream_state.carton_model.task == "segment"
    assert stream_state.carton_model.names == {0: "carton"}
    stream_state.cleanup()


def test_stats_without_playback_time_returns_standard_json(client, monkeypatch):
    monkeypatch.setattr(state, "is_multi_angle", True)
    monkeypatch.setattr(state, "source", "recorded_output.mp4")
    monkeypatch.setattr(state, "frame_idx", 0)
    monkeypatch.setattr(state, "multi_angle_report", {
        "processing_mode": "recorded_video",
        "calibration": {"calibration_complete": False},
        "result": {
            "processing_mode": "recorded_video",
            "calibration": {"calibration_complete": False},
            "events": [{
                "direction": "IN",
                "timestamp_sec": 1.25,
                "is_auto_confirmed": True,
                "requires_review": False,
                "association_method": "synchronized_video_tripwire_time",
                "participating_cameras": ["cam_top", "cam_side"],
            }],
        },
    })
    monkeypatch.setattr(app_module.live_camera_manager, "status", lambda: {"active": False})

    response = client.get("/api/stats")

    assert response.status_code == 200
    data = json.loads(
        response.get_data(as_text=True),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    assert data["timestamp_sec"] == 1.25


def test_stats_excludes_review_candidates_and_exposes_camera_evidence(client, monkeypatch):
    evidence = {
        "cam_side": [{
            "local_track_id": "cam_side_t0007",
            "crossing_point_px": [512.0, 250.0],
            "crossing_line_px": 512.0,
        }],
        "cam_top": [{
            "local_track_id": "cam_top_t0012",
            "crossing_point_px": [640.0, 320.0],
            "crossing_line_px": 640.0,
        }],
    }
    monkeypatch.setattr(state, "is_multi_angle", True)
    monkeypatch.setattr(state, "source", "recorded_output.mp4")
    monkeypatch.setattr(state, "multi_angle_report", {
        "calibration": {
            "calibration_complete": False,
            "require_calibrated_fusion": True,
        },
        "result": {
            "events": [
                {
                    "fused_cartons": 1,
                    "direction": "IN",
                    "timestamp_sec": 1.0,
                    "is_auto_confirmed": True,
                    "requires_review": False,
                    "consensus_status": "THREE_CAMERA_UNANIMOUS",
                    "details": {
                        "event_id": "event-001",
                        "global_carton_id": "global-001",
                        "camera_evidence": evidence,
                    },
                },
                {
                    "fused_cartons": 3,
                    "direction": "IN",
                    "timestamp_sec": 3.0,
                    "is_auto_confirmed": False,
                    "requires_review": True,
                    "consensus_status": "TWO_CAMERA_AGREEMENT",
                    "details": {
                        "event_id": "event-review",
                        "global_carton_id": "global-review",
                    },
                },
            ],
        },
    })

    response = client.get("/api/stats?t=4")
    assert response.status_code == 200
    stats = response.get_json()
    assert stats["total_in"] == 0
    assert stats["net_count"] == 0
    assert stats["observed_in"] == 4
    assert stats["pending_review_in"] == 4
    assert stats["consensus_status"] == "CALIBRATION_REQUIRED"
    assert stats["events"][1]["global_carton_id"] == "global-001"
    assert stats["events"][1]["camera_evidence"] == evidence
    assert stats["events"][1]["is_confirmed"] is False


def test_stats_does_not_confirm_legacy_events_without_explicit_flags(client, monkeypatch):
    monkeypatch.setattr(state, "is_multi_angle", True)
    monkeypatch.setattr(state, "source", "recorded_output.mp4")
    monkeypatch.setattr(state, "multi_angle_report", {
        "calibration": {
            "calibration_complete": True,
            "require_calibrated_fusion": True,
        },
        "result": {
            "events": [{
                "carton_count": 1,
                "direction": "IN",
                "timestamp_sec": 1.0,
                "consensus_status": "THREE_CAMERA_UNANIMOUS",
            }],
        },
    })

    response = client.get("/api/stats?t=2")

    assert response.status_code == 200
    stats = response.get_json()
    assert stats["total_in"] == 0
    assert stats["observed_in"] == 1
    assert stats["pending_review_in"] == 1


def test_stats_does_not_trust_events_without_calibration_metadata(client, monkeypatch):
    monkeypatch.setattr(state, "is_multi_angle", True)
    monkeypatch.setattr(state, "source", "legacy_output.mp4")
    monkeypatch.setattr(state, "multi_angle_report", {
        "result": {
            "events": [{
                "carton_count": 1,
                "direction": "IN",
                "timestamp_sec": 1.0,
                "is_auto_confirmed": True,
                "requires_review": False,
                "consensus_status": "THREE_CAMERA_UNANIMOUS",
            }],
        },
    })

    response = client.get("/api/stats?t=2")

    assert response.status_code == 200
    stats = response.get_json()
    assert stats["total_in"] == 0
    assert stats["observed_in"] == 1
    assert stats["pending_review_in"] == 1
    assert stats["consensus_status"] == "CALIBRATION_REQUIRED"


def test_process_3angle_missing_inputs_returns_400(client):
    """Verify missing inputs return 400 Bad Request on both /api/3angle/process and /process_3angle."""
    # 1. Missing body
    resp = client.post("/api/3angle/process", json={})
    assert resp.status_code == 400
    assert "Missing videos" in resp.get_json()["error"]

    # 2. Incomplete angles
    resp = client.post("/api/3angle/process", json={
        "video_top": "scenario1_top_synced.mp4",
        "video_side": "scenario1_side_synced.mp4",
        "video_flank": "non_existent_flank_file_xyz.mp4",
    })
    assert resp.status_code == 400
    assert "cam_flank" in resp.get_json()["error"]

    # 3. Test route alias /process_3angle
    resp_alias = client.post("/process_3angle", json={
        "video_top": "non_existent_top_file_xyz.mp4",
        "video_side": "scenario1_side_synced.mp4",
        "video_flank": "scenario1_flank_synced.mp4",
    })
    assert resp_alias.status_code == 400
    assert "cam_top" in resp_alias.get_json()["error"]


def test_process_rejects_assigning_same_video_to_multiple_cameras(client):
    response = client.post("/api/3angle/process", json={
        "cam_top": {"source": "scenario1_top_synced.mp4"},
        "cam_side": {"source": "scenario1_top_synced.mp4"},
    })
    assert response.status_code == 400
    assert "different video" in response.get_json()["error"]


def test_process_rejects_video_assigned_to_wrong_camera(client):
    response = client.post("/api/3angle/process", json={
        "cam_top": {"source": "scenario1_side_synced.mp4"},
        "cam_side": {"source": "scenario1_top_synced.mp4"},
    })
    assert response.status_code == 400
    assert "assigned to cam_top" in response.get_json()["error"]


def test_process_rejects_scenario_videos_from_different_sessions(client):
    response = client.post("/api/3angle/process", json={
        "cam_top": {"source": "scenario1_top_synced.mp4"},
        "cam_side": {"source": "scenario2_side_synced.mp4"},
        "cam_flank": {"source": "scenario1_flank_synced.mp4"},
    })
    assert response.status_code == 400
    assert "same loading session" in response.get_json()["error"]


def test_process_3angle_job_lifecycle_mocked(client, monkeypatch):
    """Verify full job creation, background processing, status polling, and stats update."""
    dummy_report = {
        "complete": True,
        "incomplete_reason": None,
        "evaluation_timestamp": "2026-10-28T12:00:00Z",
        "calibration": {
            "calibration_complete": True,
            "require_calibrated_fusion": True,
        },
        "total_loaded_cartons": 7,
        "total_returned_cartons": 0,
        "net_cartons": 7,
        "total_worker_trips": 7,
        "consensus_breakdown": {
            "THREE_CAMERA_UNANIMOUS": 7,
            "TWO_CAMERA_AGREEMENT": 0,
            "SINGLE_CAMERA_OBSERVATION": 0,
        },
        "cameras": [
            {"camera_id": "cam_top", "video": "scenario1_top_synced.mp4", "role": "width"},
            {"camera_id": "cam_side", "video": "scenario1_side_synced.mp4", "role": "height"},
            {"camera_id": "cam_flank", "video": "scenario1_flank_synced.mp4", "role": "corroborator"},
        ],
        "result": {
            "total_loaded_cartons": 7,
            "total_returned_cartons": 0,
            "net_cartons": 7,
            "review_required_cartons": 0,
            "total_worker_trips": 7,
            "consensus_breakdown": {
                "THREE_CAMERA_UNANIMOUS": 7,
                "TWO_CAMERA_AGREEMENT": 0,
                "SINGLE_CAMERA_OBSERVATION": 0,
            },
            "events": [
                {
                    "fused_cartons": 7,
                    "direction": "IN",
                    "timestamp_sec": 1.5,
                    "frame_idx": 45,
                    "consensus_status": "THREE_CAMERA_UNANIMOUS",
                    "is_auto_confirmed": True,
                    "requires_review": False,
                }
            ],
        },
        "frames_processed": 100,
        "frames_inferred": 100,
        "expected_frames": 100,
        "elapsed_seconds": 1.2,
        "processing_fps": 83.3,
    }

    mock_process = MagicMock(return_value=dummy_report)
    monkeypatch.setattr(ThreeCameraCartonPipeline, "process_videos", mock_process)

    # Submit job
    resp = client.post("/api/3angle/process", json={
        "video_top": "scenario1_top_synced.mp4",
        "video_side": "scenario1_side_synced.mp4",
        "video_flank": "scenario1_flank_synced.mp4",
    })
    assert resp.status_code == 200
    res_data = resp.get_json()
    assert res_data["success"] is True
    job_id = res_data["job_id"]
    assert job_id in multi_angle_jobs

    # Wait for thread completion
    for _ in range(50):
        time.sleep(0.05)
        st_resp = client.get(f"/api/3angle/job_status/{job_id}")
        assert st_resp.status_code == 200
        job_data = st_resp.get_json()
        if job_data["status"] == "completed":
            break

    assert job_data["status"] == "completed"
    assert job_data["complete"] is True
    assert job_data["progress_pct"] == 100
    assert job_data["result"]["total_loaded_cartons"] == 7

    # Verify /api/stats reflects the new 3-camera report
    stats_resp = client.get("/api/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.get_json()
    assert stats["is_multi_angle"] is True
    assert stats["consensus_status"] == "THREE_CAMERA_VERIFIED"
    assert len(stats["cameras"]) == 3

    # When video stream playback reaches or passes the event frame (frame 45)
    with app.app_context():
        state.frame_idx = 50
    stats_active = client.get("/api/stats").get_json()
    assert stats_active["net_count"] == 7
    assert stats_active["total_in"] == 7


def test_incomplete_video_job_withholds_partial_counts(client, monkeypatch):
    class IncompletePipeline:
        def __init__(self, **kwargs):
            self.last_processed_time_sec = 2.0

        def process_videos(self, **kwargs):
            return {
                "complete": False,
                "stopped": False,
                "incomplete_reason": "Unexpected end of cam_side",
                "frames_inferred": 60,
                "expected_frames": 100,
                "result": {
                    "total_loaded_cartons": 3,
                    "total_returned_cartons": 1,
                    "net_cartons": 2,
                    "review_required_cartons": 4,
                },
            }

    monkeypatch.setattr(app_module, "ThreeCameraCartonPipeline", IncompletePipeline)
    response = client.post("/api/3angle/process", json={
        "cam_top": {"source": "scenario1_top_synced.mp4"},
        "cam_side": {"source": "scenario1_side_synced.mp4"},
        "cam_flank": {"source": "scenario1_flank_synced.mp4"},
    })
    assert response.status_code == 200
    job_id = response.get_json()["job_id"]

    for _ in range(50):
        time.sleep(0.01)
        job = client.get(f"/api/3angle/job_status/{job_id}").get_json()
        if job["status"] != "processing":
            break

    assert job["status"] == "incomplete"
    assert job["complete"] is False
    assert job["incomplete_reason"] == "Unexpected end of cam_side"
    assert job["counts"]["loaded"] == 0
    assert job["counts"]["returned"] == 0
    assert job["counts"]["net"] == 0


def test_uncalibrated_complete_video_job_returns_image_space_counts(client, monkeypatch):
    class UncalibratedPipeline:
        def __init__(self, **kwargs):
            self.last_processed_time_sec = 2.0

        def process_videos(self, **kwargs):
            return {
                "complete": True,
                "stopped": False,
                "calibration": {
                    "calibration_complete": False,
                    "camera_calibration": {
                        "cam_top": {"is_calibrated": False},
                        "cam_side": {"is_calibrated": False},
                        "cam_flank": {"is_calibrated": False},
                    },
                },
                "result": {
                    "total_loaded_cartons": 3,
                    "total_returned_cartons": 1,
                    "net_cartons": 2,
                    "review_required_cartons": 0,
                },
            }

    monkeypatch.setattr(app_module, "ThreeCameraCartonPipeline", UncalibratedPipeline)
    response = client.post("/api/3angle/process", json={
        "cam_top": {"source": "scenario1_top_synced.mp4"},
        "cam_side": {"source": "scenario1_side_synced.mp4"},
        "cam_flank": {"source": "scenario1_flank_synced.mp4"},
    })
    assert response.status_code == 200
    job_id = response.get_json()["job_id"]

    for _ in range(50):
        time.sleep(0.01)
        job = client.get(f"/api/3angle/job_status/{job_id}").get_json()
        if job["status"] != "processing":
            break

    assert job["status"] == "completed"
    assert job["complete"] is True
    assert job["counts"]["loaded"] == 3
    assert job["counts"]["returned"] == 1
    assert job["counts"]["net"] == 2
    assert job["counting_blocker"] is None


def test_process_accepts_two_selected_recorded_cameras(client, monkeypatch):
    observed_kwargs = {}

    def fake_init(self, **kwargs):
        observed_kwargs["tripwire_configs"] = kwargs["tripwire_configs"]
        self.last_processed_time_sec = 0.0

    def fake_process(self, **kwargs):
        observed_kwargs.update(kwargs)
        return {
            "complete": True,
            "calibration": {"calibration_complete": True},
            "processing_mode": "recorded_video",
            "result": {
                "processing_mode": "recorded_video",
                "total_loaded_cartons": 0,
                "total_returned_cartons": 0,
                "net_cartons": 0,
                "review_required_cartons": 0,
                "events": [],
                "calibration": {"calibration_complete": False},
            },
            "output_video": None,
            "camera_output_videos": {
                "cam_top": "/tmp/processed_cam_top.mp4",
                "cam_side": "/tmp/processed_cam_side.mp4",
            },
        }

    monkeypatch.setattr(ThreeCameraCartonPipeline, "__init__", fake_init)
    monkeypatch.setattr(ThreeCameraCartonPipeline, "process_videos", fake_process)
    response = client.post("/api/3angle/process", json={
        "cam_top": {"source": "scenario1_top_synced.mp4"},
        "cam_side": {"source": "scenario1_side_synced.mp4"},
        "tripwire_configs": {
            "cam_top": {"physically_verified": True},
            "cam_side": {"physically_verified": True},
        },
    })

    assert response.status_code == 200
    job_id = response.get_json()["job_id"]
    for _ in range(50):
        job = client.get(f"/api/3angle/job_status/{job_id}").get_json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)

    assert job["status"] == "completed", job.get("error")
    assert observed_kwargs["video_top"]
    assert observed_kwargs["video_side"]
    assert "video_flank" not in observed_kwargs
    assert observed_kwargs["tripwire_configs"]["cam_top"]["physically_verified"] is False
    assert observed_kwargs["tripwire_configs"]["cam_side"]["physically_verified"] is False
    assert job["camera_output_videos"] == {
        "cam_top": "/api/output_video/processed_cam_top.mp4",
        "cam_side": "/api/output_video/processed_cam_side.mp4",
    }


def test_job_status_reports_confirmed_counts_during_processing(client, monkeypatch):
    progress_ready = Event()
    allow_completion = Event()

    def fake_process(self, **kwargs):
        self._audit_events.append({
            "carton_count": 1,
            "direction": "IN",
            "is_auto_confirmed": True,
            "requires_review": False,
        })
        kwargs["progress_callback"](40, "Analyzing synchronized frames")
        progress_ready.set()
        assert allow_completion.wait(timeout=5)
        return {
            "complete": True,
            "calibration": {"calibration_complete": True},
            "processing_mode": "recorded_video",
            "result": {
                "processing_mode": "recorded_video",
                "total_loaded_cartons": 1,
                "total_returned_cartons": 0,
                "net_cartons": 1,
                "review_required_cartons": 0,
                "events": [],
                "calibration": {"calibration_complete": False},
            },
            "output_video": None,
        }

    monkeypatch.setattr(ThreeCameraCartonPipeline, "process_videos", fake_process)
    response = client.post("/api/3angle/process", json={
        "video_top": "scenario1_top_synced.mp4",
        "video_side": "scenario1_side_synced.mp4",
    })
    assert response.status_code == 200
    job_id = response.get_json()["job_id"]

    try:
        assert progress_ready.wait(timeout=5)
        job = client.get(f"/api/3angle/job_status/{job_id}").get_json()
        assert job["status"] == "processing"
        assert job["counts"]["loaded"] == 1
        assert job["counts"]["returned"] == 0
        assert job["counts"]["net"] == 1
    finally:
        allow_completion.set()

    for _ in range(50):
        job = client.get(f"/api/3angle/job_status/{job_id}").get_json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert job["status"] == "completed", job.get("error")
    assert job["counts"]["loaded"] == 1
    assert job["counts"]["net"] == 1


def test_recorded_job_control_updates_playback_and_stops_worker(client):
    job_id = f"control_test_{time.time_ns()}"
    multi_angle_jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "playback_state": "paused",
        "playback_time_sec": 0.0,
        "counts": {"loaded": 0, "returned": 0, "net": 0, "timestamp_sec": 0.0},
        "_control_event": Event(),
        "_stop_event": Event(),
    }

    play = client.post(
        f"/api/3angle/job_control/{job_id}",
        json={"action": "play", "current_time": 2.5},
    )
    assert play.status_code == 200
    assert play.get_json()["playback_state"] == "playing"
    assert multi_angle_jobs[job_id]["playback_time_sec"] == 2.5

    pause = client.post(f"/api/3angle/job_control/{job_id}", json={"action": "pause"})
    assert pause.status_code == 200
    assert multi_angle_jobs[job_id]["playback_state"] == "paused"

    stop = client.post(f"/api/3angle/job_control/{job_id}", json={"action": "stop"})
    assert stop.status_code == 200
    assert multi_angle_jobs[job_id]["status"] == "stopping"
    assert multi_angle_jobs[job_id]["_stop_event"].is_set()

    status = client.get(f"/api/3angle/job_status/{job_id}").get_json()
    assert "_stop_event" not in status


def test_process_3angle_error_handling(client, monkeypatch):
    """Verify pipeline failures are gracefully caught and reported in job_status."""
    def _failing_pipeline(*args, **kwargs):
        raise RuntimeError("Simulated pipeline engine failure")

    monkeypatch.setattr(ThreeCameraCartonPipeline, "process_videos", _failing_pipeline)

    resp = client.post("/process_3angle", json={
        "video_top": "scenario1_top_synced.mp4",
        "video_side": "scenario1_side_synced.mp4",
        "video_flank": "scenario1_flank_synced.mp4",
    })
    assert resp.status_code == 200
    job_id = resp.get_json()["job_id"]

    for _ in range(50):
        time.sleep(0.05)
        st_resp = client.get(f"/api/3angle/job_status/{job_id}")
        job_data = st_resp.get_json()
        if job_data["status"] == "failed":
            break

    assert job_data["status"] == "failed"
    assert "Simulated pipeline engine failure" in job_data["error"]


def test_three_camera_pipeline_short_end_to_end():
    """Verify ThreeCameraCartonPipeline runs on real scenario1 synced videos without error for 10 frames."""
    rec_dir = _app_dir / "recorded_video"
    sync_dir = rec_dir / "synchronized"

    top_p = sync_dir / "scenario1_top_synced.mp4"
    side_p = sync_dir / "scenario1_side_synced.mp4"
    flank_p = sync_dir / "scenario1_flank_synced.mp4"

    if not (top_p.exists() and side_p.exists() and flank_p.exists()):
        pytest.skip("Synchronized scenario 1 videos not found on disk")

    pipeline = ThreeCameraCartonPipeline(fps=30.0, tripwire_x_px=500.0)
    report = pipeline.process_videos(
        video_top=str(top_p),
        video_side=str(side_p),
        video_flank=str(flank_p),
        max_frames=10,
        frame_stride=1,
    )

    assert "result" in report
    assert "total_loaded_cartons" in report["result"]
    assert "consensus_breakdown" in report["result"]
    assert report["frames_processed"] == 10
    assert report["frames_inferred"] == 10
    assert report["expected_frames"] == 817
    assert report["complete"] is False
    assert report["result"]["total_loaded_cartons"] == 0
    assert report["result"]["calibration"]["calibration_complete"] is False
    assert report["result"]["calibration"]["calibrated_cameras"] == []
    assert all(
        camera["tripwire"]["physically_verified"] is False
        for camera in report["result"]["calibration"]["camera_calibration"].values()
    )
    assert all(
        stats == {"skipped": 0, "repeated": 0}
        for stats in report["source_frame_stats"].values()
    )
    assert "processing_fps" in report
