import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "truck_loading_detection"))

from multi_angle_fusion_engine import (
    CameraAngleConfig,
    AngleTripwireTracker,
    MultiAngleFusionEngine
)


def test_top_camera_horizontal_tripwire_vertical_movement():
    """
    Test Top-Down camera with horizontal line (Y=400) and up_to_down direction.
    Worker moves from Y=350 to Y=450.
    """
    cfg = CameraAngleConfig(
        camera_id="cam_top",
        angle_type="top_overhead",
        line_orientation="horizontal",
        line_pos=400,
        direction="up_to_down",
        hysteresis=15,
        cooldown_frames=5
    )
    tracker = AngleTripwireTracker(cfg)

    # Frame 1: Worker above line (Y=350)
    obj_f1 = [(1, (200.0, 320.0, 260.0, 380.0), "person", 0.90, 1)]
    events_f1 = tracker.update(obj_f1, frame_idx=1, timestamp_sec=0.033)
    assert len(events_f1) == 0
    assert tracker.total_in == 0

    # Frame 2: Worker crosses line to below (Y=450)
    obj_f2 = [(1, (200.0, 420.0, 260.0, 480.0), "person", 0.90, 1)]
    events_f2 = tracker.update(obj_f2, frame_idx=2, timestamp_sec=0.066)
    assert len(events_f2) == 1
    assert events_f2[0]["direction"] == "IN"
    assert tracker.total_in == 1


def test_side_camera_vertical_tripwire_horizontal_movement():
    """
    Test Side Profile camera with vertical line (X=500) and left_to_right direction.
    Worker carries 2 stacked cartons from X=450 to X=550.
    """
    cfg = CameraAngleConfig(
        camera_id="cam_side",
        angle_type="side_profile",
        line_orientation="vertical",
        line_pos=500,
        direction="left_to_right",
        hysteresis=15,
        cooldown_frames=5
    )
    tracker = AngleTripwireTracker(cfg)

    # Frame 1: Worker on left (X=440)
    obj_f1 = [(1, (400.0, 200.0, 480.0, 400.0), "person", 0.92, 2)]
    events_f1 = tracker.update(obj_f1, frame_idx=1, timestamp_sec=0.033)
    assert len(events_f1) == 0

    # Frame 2: Worker on right (X=560)
    obj_f2 = [(1, (520.0, 200.0, 600.0, 400.0), "person", 0.92, 2)]
    events_f2 = tracker.update(obj_f2, frame_idx=2, timestamp_sec=0.066)
    assert len(events_f2) == 1
    assert events_f2[0]["direction"] == "IN"
    assert events_f2[0]["carton_qty"] == 2
    assert tracker.total_in == 2


def test_multi_angle_fusion_consensus():
    """
    Test that Top Camera (Width = 1) and Side Camera (Height = 2 stacked cartons)
    fused within the time window produce Fused Count = 1 * 2 = 2 Cartons.
    """
    engine = MultiAngleFusionEngine(time_window_sec=0.5)

    cam_top_cfg = CameraAngleConfig(
        camera_id="cam_top",
        angle_type="top_overhead",
        line_orientation="horizontal",
        line_pos=400,
        direction="up_to_down"
    )
    cam_side_cfg = CameraAngleConfig(
        camera_id="cam_side",
        angle_type="side_profile",
        line_orientation="vertical",
        line_pos=500,
        direction="left_to_right"
    )
    engine.add_camera(cam_top_cfg)
    engine.add_camera(cam_side_cfg)

    # Step 1: Top camera crossing at t=1.0s (Width = 1)
    engine.update_camera_frame("cam_top", [(10, (200.0, 300.0, 280.0, 360.0), "person", 0.9, 1)], 1, 0.9)
    engine.update_camera_frame("cam_top", [(10, (200.0, 440.0, 280.0, 500.0), "person", 0.9, 1)], 2, 1.0)

    # Step 2: Side camera crossing at t=1.15s (Height = 2 stacked cartons)
    engine.update_camera_frame("cam_side", [(20, (400.0, 200.0, 480.0, 400.0), "person", 0.9, 2)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(20, (540.0, 200.0, 620.0, 400.0), "person", 0.9, 2)], 2, 1.15)

    # Trigger consensus processing
    fused = engine.process_time_window_consensus(current_time_sec=1.2)

    assert len(fused) == 1
    f_event = fused[0]
    assert f_event["consensus_status"] == "DUAL_CAMERA_VERIFIED"
    assert f_event["top_camera_width"] == 1
    assert f_event["side_camera_height"] == 2
    assert f_event["fused_cartons"] == 2
    assert engine.total_loaded_cartons == 2
    assert engine.total_worker_trips == 1


def test_dual_worker_concurrent_fusion():
    """
    Test two workers crossing simultaneously:
    Worker A: 1 carton wide, 1 carton high -> 1 carton
    Worker B: 1 carton wide, 2 cartons high -> 2 cartons
    Total Loaded = 3 cartons.
    """
    engine = MultiAngleFusionEngine(time_window_sec=0.5)

    cam_top_cfg = CameraAngleConfig("cam_top", "top_overhead", "horizontal", 400, "up_to_down")
    cam_side_cfg = CameraAngleConfig("cam_side", "side_profile", "vertical", 500, "left_to_right")
    engine.add_camera(cam_top_cfg)
    engine.add_camera(cam_side_cfg)

    # Pass 1: Worker A (1 box)
    engine.update_camera_frame("cam_top", [(1, (100.0, 300.0, 160.0, 360.0), "person", 0.9, 1)], 1, 1.0)
    engine.update_camera_frame("cam_top", [(1, (100.0, 450.0, 160.0, 510.0), "person", 0.9, 1)], 2, 1.1)

    engine.update_camera_frame("cam_side", [(101, (400.0, 200.0, 480.0, 400.0), "person", 0.9, 1)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(101, (540.0, 200.0, 620.0, 400.0), "person", 0.9, 1)], 2, 1.15)

    # Pass 2: Worker B (2 boxes)
    engine.update_camera_frame("cam_top", [(2, (300.0, 300.0, 360.0, 360.0), "person", 0.9, 1)], 1, 1.0)
    engine.update_camera_frame("cam_top", [(2, (300.0, 450.0, 360.0, 510.0), "person", 0.9, 1)], 2, 1.12)

    engine.update_camera_frame("cam_side", [(102, (400.0, 100.0, 480.0, 300.0), "person", 0.9, 2)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(102, (540.0, 100.0, 620.0, 300.0), "person", 0.9, 2)], 2, 1.18)

    fused = engine.process_time_window_consensus(current_time_sec=1.3)
    assert len(fused) == 2
    total = sum(e["fused_cartons"] for e in fused)
    assert total == 3
    assert engine.total_loaded_cartons == 3


def test_three_cameras_detect_same_carton_count_is_one_not_three():
    """
    Test Scenario:
    Worker picks 1 carton from pallet and crosses red line into truck.
    Right Camera, Left Camera, and Top Camera ALL detect the crossing.
    Result MUST be Count = 1, NOT Count = 3!
    """
    engine = MultiAngleFusionEngine(time_window_sec=0.5)

    cam_top_cfg = CameraAngleConfig("cam_top", angle_type="top_overhead", line_orientation="horizontal", line_pos=400, direction="up_to_down", role="width")
    cam_right_cfg = CameraAngleConfig("cam_right", angle_type="side_profile", line_orientation="vertical", line_pos=500, direction="left_to_right", role="height")
    cam_left_cfg = CameraAngleConfig("cam_left", angle_type="side_profile", line_orientation="vertical", line_pos=500, direction="left_to_right", role="corroborator")

    engine.add_camera(cam_top_cfg)
    engine.add_camera(cam_right_cfg)
    engine.add_camera(cam_left_cfg)

    # Top Camera sees 1 carton (width 1)
    engine.update_camera_frame("cam_top", [(1, (100.0, 320.0, 180.0, 380.0), "person", 0.9, 1)], 1, 1.0)
    engine.update_camera_frame("cam_top", [(1, (100.0, 430.0, 180.0, 490.0), "person", 0.9, 1)], 2, 1.08)

    # Right Camera sees 1 carton (height 1)
    engine.update_camera_frame("cam_right", [(2, (440.0, 200.0, 520.0, 400.0), "person", 0.9, 1)], 1, 1.0)
    engine.update_camera_frame("cam_right", [(2, (550.0, 200.0, 630.0, 400.0), "person", 0.9, 1)], 2, 1.12)

    # Left Camera (Flank Corroborator) also sees 1 carton
    engine.update_camera_frame("cam_left", [(3, (440.0, 200.0, 520.0, 400.0), "person", 0.9, 1)], 1, 1.0)
    engine.update_camera_frame("cam_left", [(3, (550.0, 200.0, 630.0, 400.0), "person", 0.9, 1)], 2, 1.14)

    # Run consensus fusion
    fused = engine.process_time_window_consensus(current_time_sec=1.3)

    assert len(fused) == 1, "Must produce exactly 1 fused event, not 3!"
    assert fused[0]["consensus_status"] == "THREE_CAMERA_VERIFIED"
    assert fused[0]["fused_cartons"] == 1, "Must count exactly 1 carton!"
    assert engine.total_loaded_cartons == 1, "Total loaded count must be 1, NOT 3!"
    assert engine.total_worker_trips == 1


def test_empty_handed_worker_crosses_tripwire_count_is_zero():
    """
    Test Scenario:
    Worker walks across red line into truck WITHOUT holding any carton (carton_qty = 0).
    Count MUST be 0! Only cartons must be counted, not workers.
    """
    engine = MultiAngleFusionEngine(time_window_sec=0.5)

    cam_top_cfg = CameraAngleConfig("cam_top", "top_overhead", "horizontal", 400, "up_to_down", role="width")
    cam_side_cfg = CameraAngleConfig("cam_side", "side_profile", "vertical", 500, "left_to_right", role="height")
    cam_flank_cfg = CameraAngleConfig("cam_flank", "side_profile", "vertical", 500, "left_to_right", role="corroborator")

    engine.add_camera(cam_top_cfg)
    engine.add_camera(cam_side_cfg)
    engine.add_camera(cam_flank_cfg)

    # Top Camera detects person, but carton_qty = 0
    engine.update_camera_frame("cam_top", [(1, (100.0, 320.0, 180.0, 380.0), "person", 0.9, 0)], 1, 1.0)
    engine.update_camera_frame("cam_top", [(1, (100.0, 430.0, 180.0, 490.0), "person", 0.9, 0)], 2, 1.08)

    # Side Camera detects person, but carton_qty = 0
    engine.update_camera_frame("cam_side", [(2, (440.0, 200.0, 520.0, 400.0), "person", 0.9, 0)], 1, 1.0)
    engine.update_camera_frame("cam_side", [(2, (550.0, 200.0, 630.0, 400.0), "person", 0.9, 0)], 2, 1.12)

    # Flank Camera detects person, carton_qty = 0
    engine.update_camera_frame("cam_flank", [(3, (440.0, 200.0, 520.0, 400.0), "person", 0.9, 0)], 1, 1.0)
    engine.update_camera_frame("cam_flank", [(3, (550.0, 200.0, 630.0, 400.0), "person", 0.9, 0)], 2, 1.14)

    fused = engine.process_time_window_consensus(current_time_sec=1.3)

    assert len(fused) == 1
    assert fused[0]["fused_cartons"] == 0, "Empty-handed worker contributes 0 cartons!"
    assert engine.total_loaded_cartons == 0, "Total loaded cartons must remain 0!"
    assert engine.total_worker_trips == 1, "Worker trip is tracked, but 0 cartons counted."


def test_right_camera_dropout_covered_by_left_and_top():
    """
    Test Scenario:
    Right camera fails to detect (blind spot/occlusion).
    Left camera and Top camera successfully detect the worker carrying 1 carton.
    Count MUST be exactly 1!
    """
    engine = MultiAngleFusionEngine(time_window_sec=0.5)

    cam_top_cfg = CameraAngleConfig("cam_top", "top_overhead", "horizontal", 400, "up_to_down", role="width")
    cam_side_cfg = CameraAngleConfig("cam_side", "side_profile", "vertical", 500, "left_to_right", role="height")
    cam_flank_cfg = CameraAngleConfig("cam_flank", "side_profile", "vertical", 500, "left_to_right", role="corroborator")

    engine.add_camera(cam_top_cfg)
    engine.add_camera(cam_side_cfg)
    engine.add_camera(cam_flank_cfg)

    # Top Camera detects (carton_qty = 1)
    engine.update_camera_frame("cam_top", [(1, (100.0, 320.0, 180.0, 380.0), "person", 0.9, 1)], 1, 1.0)
    engine.update_camera_frame("cam_top", [(1, (100.0, 430.0, 180.0, 490.0), "person", 0.9, 1)], 2, 1.08)

    # Right Camera DROPS OUT (no crossing detected)

    # Left Camera (Flank) detects crossing (carton_qty = 1)
    engine.update_camera_frame("cam_flank", [(3, (440.0, 200.0, 520.0, 400.0), "person", 0.9, 1)], 1, 1.0)
    engine.update_camera_frame("cam_flank", [(3, (550.0, 200.0, 630.0, 400.0), "person", 0.9, 1)], 2, 1.14)

    # When window expires, fallback triggers without losing the carton
    fused = engine.process_time_window_consensus(current_time_sec=1.8)

    assert len(fused) == 1
    assert fused[0]["fused_cartons"] == 1
    assert engine.total_loaded_cartons == 1, "Carton successfully recovered via fallback!"

