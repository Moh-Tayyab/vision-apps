import os
import pytest
import numpy as np
import cv2
from ultralytics import YOLO

from apps.truck_loading_detection.model_pipeline.mask_refinement import CartonMaskRefiner
from apps.truck_loading_detection.three_camera_system.per_camera_tracker import (
    PerCameraCartonTracker,
    CanonicalCartonDetection,
    TrackStatus,
)
from apps.truck_loading_detection.three_camera_system.detector_adapter import CartonDetectorAdapter


def test_checkpoint_integrity_and_classes():
    """Phase 7: Verify that the production checkpoint loads as a valid single-class carton segmentation model."""
    ckpt_path = "apps/truck_loading_detection/best.pt"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    model = YOLO(ckpt_path)
    assert getattr(model, "task", None) == "segment", f"Expected segment task, got {model.task}"
    assert getattr(model, "names", {}) == {0: "carton"}, f"Expected carton class 0, got {model.names}"


def test_worker_mask_subtraction_margin():
    """Phase 4 & 5: Verify that worker polygon subtraction cleanly excises worker pixels from carton mask."""
    refiner = CartonMaskRefiner(worker_subtraction_margin_px=2)
    h, w = 480, 640

    carton_poly = [(100, 100), (200, 100), (200, 200), (100, 200)]  # 100x100 box
    worker_poly = [(180, 80), (240, 80), (240, 220), (180, 220)]   # 20px overlap on right edge

    ref = refiner.refine_carton_mask(
        raw_polygon=carton_poly,
        image_shape=(h, w),
        worker_polygons=[worker_poly],
        confidence=0.9,
    )
    assert ref is not None
    assert ref.is_occluded_by_worker is True
    assert 0.15 <= ref.worker_overlap_ratio <= 0.25
    assert ref.bbox[2] <= 182.0
    assert ref.raw_polygon is not None


def test_excessive_occlusion_preservation_threshold():
    """Phase 4 & 5: Verify that when worker occludes >=70% of carton, refiner does not obliterate carton completely."""
    refiner = CartonMaskRefiner()
    h, w = 480, 640
    carton_poly = [(100, 100), (200, 100), (200, 200), (100, 200)]
    worker_heavy = [(105, 95), (205, 95), (205, 205), (105, 205)]  # covers 95% of carton

    ref = refiner.refine_carton_mask(
        raw_polygon=carton_poly,
        image_shape=(h, w),
        worker_polygons=[worker_heavy],
        confidence=0.9,
    )
    assert ref is not None
    assert ref.is_occluded_by_worker is True
    assert ref.area_px >= refiner.min_carton_area_px


def test_temporal_smoothing_with_vertex_resampling():
    """Phase 5: Verify that perimeter resampling allows temporal EMA smoothing even when vertex counts differ."""
    refiner = CartonMaskRefiner(temporal_blend_alpha=0.60)
    h, w = 480, 640

    poly_4pts = [(100, 100), (200, 100), (200, 200), (100, 200)]
    poly_5pts = [(100, 100), (150, 95), (200, 100), (200, 200), (100, 200)]

    refiner.reset()
    r1 = refiner.refine_carton_mask(poly_4pts, (h, w), track_id="t1", timestamp_sec=0.0)
    r2 = refiner.refine_carton_mask(poly_5pts, (h, w), track_id="t1", timestamp_sec=0.04)

    assert r1 is not None and r2 is not None
    assert "t1" in refiner._previous_masks
    assert len(r2.polygon_pts) >= 4


def test_stationary_carton_occlusion_never_marked_carried():
    """
    Phase 3: Verify that a stationary carton with worker overlap is classified as OCCLUDED or AT_REST,
    and NEVER incorrectly classified as BEING_CARRIED.
    """
    refiner = CartonMaskRefiner()
    h, w = 480, 640

    carton_poly = [(200, 200), (300, 200), (300, 300), (200, 300)]
    worker_poly = [(280, 190), (350, 190), (350, 310), (280, 310)]  # overlaps 20%

    refiner.reset()
    _ = refiner.refine_carton_mask(carton_poly, (h, w), worker_polygons=[worker_poly], track_id="c1", timestamp_sec=0.0)
    r1 = refiner.refine_carton_mask(carton_poly, (h, w), worker_polygons=[worker_poly], track_id="c1", timestamp_sec=0.2)
    r2 = refiner.refine_carton_mask(carton_poly, (h, w), worker_polygons=[worker_poly], track_id="c1", timestamp_sec=0.4)

    assert r2 is not None
    assert r2.is_occluded_by_worker is True
    assert r2.handling_state in ("OCCLUDED", "AT_REST")
    assert r2.handling_state != "BEING_CARRIED", "Stationary carton MUST NEVER be classified as BEING_CARRIED"


def test_actual_carrying_and_placement_kinematics():
    """
    Phase 3: Verify handling state progression: AT_REST -> PICKUP_LIFTING -> BEING_CARRIED -> PLACEMENT_RELEASE/AT_REST.
    """
    refiner = CartonMaskRefiner()
    h, w = 480, 640

    # 1. At rest
    refiner.reset()
    c_p0 = [(200, 200), (300, 200), (300, 300), (200, 300)]
    r0 = refiner.refine_carton_mask(c_p0, (h, w), track_id="c2", timestamp_sec=0.0)
    assert r0.handling_state == "AT_REST"

    # 2. Lifted / moving across frames
    states = []
    for f in range(1, 6):
        t = f * 0.1
        x = 200.0 + f * 10.0  # speed 100 px/s
        c_pf = [(x, 200), (x + 100, 200), (x + 100, 300), (x, 300)]
        w_pf = [(x + 80, 200), (x + 180, 200), (x + 180, 350), (x + 80, 350)]
        rf = refiner.refine_carton_mask(c_pf, (h, w), worker_polygons=[w_pf], track_id="c2", timestamp_sec=t)
        states.append(rf.handling_state)

    assert "PICKUP_LIFTING" in states
    assert states[-1] == "BEING_CARRIED"


def test_per_camera_tracker_coasting_drift_bounded():
    """
    Phase 2: Verify that during 1, 3, 5, 15, and 30 frames of occlusion,
    velocity projection remains linear and bounded without runaway acceleration.
    """
    tracker = PerCameraCartonTracker(camera_id="cam_audit", max_invisible_frames=35)

    # Establish track with vx=50 px/s, vy=0 across frames 0..4
    for f in range(5):
        t = f * 0.04
        x = 100.0 + f * 2.0
        det = CanonicalCartonDetection(
            camera_id="cam_audit", frame_idx=f, timestamp_sec=t,
            bbox=(x, 50.0, x + 40.0, 90.0), confidence=0.9, class_name="carton"
        )
        tracker.update([det], f, t)

    # Occlusion across frames 5..34 (30 consecutive occluded frames)
    drifts = {}
    for f in range(5, 35):
        t = f * 0.04
        true_x = 100.0 + f * 2.0
        trks = tracker.update([], f, t)
        assert len(trks) == 1
        n_occ = f - 4
        if n_occ in (1, 3, 5, 15, 30):
            pred_x = trks[0].current_bbox[0]
            drifts[n_occ] = abs(pred_x - true_x)

    # Drifts must remain within realistic momentum decay bounds, NOT thousands of pixels
    assert drifts[1] < 2.0, f"Drift at frame 1 too high: {drifts[1]}"
    assert drifts[5] < 10.0, f"Drift at frame 5 too high: {drifts[5]}"
    assert drifts[15] < 25.0, f"Drift at frame 15 too high: {drifts[15]}"
    assert drifts[30] < 50.0, f"Drift at frame 30 too high: {drifts[30]}"


def test_tracker_reappearance_and_id_preservation():
    """
    Phase 2: Verify that when a carton reappears after 15 frames of occlusion,
    it re-matches the existing track ID without creating a duplicate.
    """
    tracker = PerCameraCartonTracker(camera_id="cam_audit2", max_invisible_frames=25)

    # Visible frames 0..3
    for f in range(4):
        t = f * 0.04
        x = 100.0 + f * 3.0
        det = CanonicalCartonDetection(
            camera_id="cam_audit2", frame_idx=f, timestamp_sec=t,
            bbox=(x, 50.0, x + 40.0, 90.0), confidence=0.9, class_name="carton"
        )
        tracker.update([det], f, t)

    # Occluded frames 4..18 (15 frames)
    for f in range(4, 19):
        tracker.update([], f, f * 0.04)

    # Reappearance at frame 19
    t19 = 19 * 0.04
    x19 = 100.0 + 19 * 3.0
    det_reappear = CanonicalCartonDetection(
        camera_id="cam_audit2", frame_idx=19, timestamp_sec=t19,
        bbox=(x19, 50.0, x19 + 40.0, 90.0), confidence=0.88, class_name="carton"
    )
    active = tracker.update([det_reappear], 19, t19)

    assert len(active) == 1
    assert active[0].local_track_id == "cam_audit2_t0001", "Must preserve original track ID"
    assert active[0].status == TrackStatus.CONFIRMED


def test_detector_adapter_preserves_raw_and_refined_masks():
    """
    Phase 4: Verify that CartonDetectorAdapter passes worker polygons to refiner
    and stores both refined mask_polygon and raw_mask_polygon.
    """
    adapter = CartonDetectorAdapter(model_path="apps/truck_loading_detection/best.pt")
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
    # Draw simulated carton
    cv2.rectangle(dummy_frame, (100, 100), (200, 200), (40, 40, 40), -1)

    worker_polys = [[(180, 80), (250, 80), (250, 220), (180, 220)]]
    dets = adapter.detect(dummy_frame, camera_id="top", frame_idx=0, timestamp_sec=0.0, worker_polygons=worker_polys)

    assert isinstance(dets, list)
    for d in dets:
        assert d.class_name == "carton"
        assert hasattr(d, "mask_polygon")
        assert hasattr(d, "raw_mask_polygon")
        assert hasattr(d, "handling_state")
