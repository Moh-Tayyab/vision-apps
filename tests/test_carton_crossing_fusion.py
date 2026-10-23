"""Verification of carton-level counting and cross-camera de-duplication.

The user requirement under test:

    A carton counts as 1 ONLY when it crosses the red line into the truck.
    The same carton seen by 2 or 3 cameras must still count as 1.
    N different cartons must count as N, not 2N or 3N.

Every test states the ground truth and asserts the exact number.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "truck_loading_detection"))

from carton_crossing_fusion import (  # noqa: E402
    CartonCrossing,
    CartonCrossingFusion,
    CartonLineTracker,
    run_scenario,
)
from carton_dedup_sim import (  # noqa: E402
    CameraGeometry,
    DEFAULT_CAMERAS,
    evenly_spaced_cartons,
    simulate_camera,
    simulate_scenario,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(detections, cameras=DEFAULT_CAMERAS, **kwargs):
    configs = {c.camera_id: c.tracker_config() for c in cameras}
    return run_scenario(
        detections,
        configs,
        fps=kwargs.pop("fps", 30.0),
        time_window_sec=kwargs.pop("time_window_sec", 0.8),
        min_cameras=kwargs.pop("min_cameras", 1),
        anchor_camera=kwargs.pop("anchor_camera", None),
    )


def _tracker(orientation="vertical", direction="left_to_right", line=640, hyst=10,
             cooldown=15):
    return CartonLineTracker(
        "cam_test",
        line_pos=line,
        orientation=orientation,
        direction=direction,
        hysteresis=hyst,
        cooldown_frames=cooldown,
    )


# ---------------------------------------------------------------------------
# 1. The core requirement: 1 carton seen by 3 cameras = 1
# ---------------------------------------------------------------------------

def test_net_balance_loaded_minus_returned():
    """3 cartons loaded, 2 brought back -> net 1, while both are visible."""
    engine = CartonCrossingFusion(time_window_sec=0.8)

    for i in range(3):
        t = 1.0 + i * 1.5
        engine.ingest(CartonCrossing("cam_top", 10 + i, t, "IN", (640, 300, 700, 360)))
        engine.ingest(CartonCrossing("cam_right", 20 + i, t + 0.06, "IN", (700, 300, 760, 360)))
        engine.ingest(CartonCrossing("cam_left", 30 + i, t + 0.12, "IN", (520, 300, 580, 360)))

    for i in range(2):
        t = 6.0 + i * 1.5
        engine.ingest(CartonCrossing("cam_top", 40 + i, t, "OUT", (640, 300, 700, 360)))
        engine.ingest(CartonCrossing("cam_right", 50 + i, t + 0.06, "OUT", (700, 300, 760, 360)))
        engine.ingest(CartonCrossing("cam_left", 60 + i, t + 0.12, "OUT", (520, 300, 580, 360)))

    summary = engine.fuse()

    assert summary.loaded_cartons == 3
    assert summary.returned_cartons == 2
    assert summary.net_cartons == 1
    # Default count_directions=("IN",) -> total stays the loaded count.
    assert summary.total_unique_cartons == 3


def test_large_per_camera_delay_does_not_create_phantom_carton():
    """A camera lagging by most of the window must not invent a carton.

    Greedy nearest-time matching attaches the delayed event to the NEXT anchor
    carton, leaving the later event unmatched (a phantom third carton). The
    order-preserving matcher keeps the sequence intact.
    """
    engine = CartonCrossingFusion(time_window_sec=0.8, anchor_camera="cam_top")
    # Anchor: two cartons 0.5 s apart.
    engine.ingest(CartonCrossing("cam_top", 1, 1.0, "IN", (640, 300, 700, 360)))
    engine.ingest(CartonCrossing("cam_top", 2, 1.5, "IN", (640, 300, 700, 360)))
    # Right camera lags by 0.7 s, so each event is nearer the WRONG anchor.
    engine.ingest(CartonCrossing("cam_right", 11, 1.7, "IN", (700, 300, 760, 360)))
    engine.ingest(CartonCrossing("cam_right", 12, 2.2, "IN", (700, 300, 760, 360)))

    summary = engine.fuse()

    assert summary.raw_crossings_total == 4
    assert summary.total_unique_cartons == 2, "late camera must not create a phantom carton"
    assert summary.duplicates_removed == 2
    assert all(len(c.cameras_seen) == 2 for c in summary.cartons)


def test_one_carton_seen_by_three_cameras_counts_once():
    cartons = [{"carton_id": 1, "cross_time": 3.0, "lane": 0.0}]
    dets = simulate_scenario(cartons, duration_sec=6.0)

    summary = _run(dets)

    assert summary.raw_crossings_total == 3, "all three cameras should see the crossing"
    assert summary.total_unique_cartons == 1, "same carton must count ONCE, not 3 times"
    assert summary.duplicates_removed == 2
    assert summary.cartons[0].cameras_seen == ["cam_top", "cam_right", "cam_left"]
    assert summary.cartons[0].evidence_count == 3


# ---------------------------------------------------------------------------
# 2. N cartons must stay N even though every camera sees every carton
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_cartons", [1, 2, 3, 6, 10])
def test_n_cartons_seen_by_three_cameras_count_n(n_cartons):
    cartons = evenly_spaced_cartons(n_cartons, first_cross_time=1.0, interval=1.0)
    duration = 1.0 + n_cartons + 2.0
    dets = simulate_scenario(cartons, duration_sec=duration)

    summary = _run(dets)

    assert summary.raw_crossings_total == n_cartons * 3
    assert summary.total_unique_cartons == n_cartons
    assert summary.duplicates_removed == n_cartons * 2
    assert len(summary.cartons) == n_cartons
    assert all(len(c.cameras_seen) == 3 for c in summary.cartons)


def test_six_cartons_raw_eighteen_fused_six():
    """The headline number for the client: 18 camera detections -> 6 cartons."""
    cartons = evenly_spaced_cartons(6, first_cross_time=1.0, interval=1.0)
    dets = simulate_scenario(cartons, duration_sec=9.0)

    summary = _run(dets)

    assert summary.raw_crossings_by_camera == {"cam_top": 6, "cam_right": 6, "cam_left": 6}
    assert summary.raw_crossings_total == 18
    assert summary.total_unique_cartons == 6
    assert summary.duplicates_removed == 12


# ---------------------------------------------------------------------------
# 3. A carton that never crosses the red line is not counted
# ---------------------------------------------------------------------------

def test_carton_approaching_but_not_crossing_counts_zero():
    cartons = [{"carton_id": 1, "cross_time": None, "lane": 0.0}]
    dets = simulate_scenario(cartons, duration_sec=6.0)

    summary = _run(dets)

    assert summary.raw_crossings_total == 0
    assert summary.total_unique_cartons == 0


# ---------------------------------------------------------------------------
# 4. Only the truck direction counts
# ---------------------------------------------------------------------------

def test_outbound_crossing_is_not_counted_as_loaded():
    tracker = _tracker(direction="left_to_right")

    # Carton starts AFTER the line (on the truck side) and moves back out.
    tracker.update([(1, (720.0, 300.0, 780.0, 360.0), 0.9)], 1, 0.0)
    tracker.update([(1, (620.0, 300.0, 680.0, 360.0), 0.9)], 2, 0.033)
    tracker.update([(1, (540.0, 300.0, 600.0, 360.0), 0.9)], 3, 0.066)

    assert len(tracker.crossings) == 1
    assert tracker.crossings[0].direction == "OUT"

    fusion = CartonCrossingFusion(count_directions=("IN",))
    fusion.ingest_many(tracker.crossings)
    summary = fusion.fuse()
    assert summary.total_unique_cartons == 0
    assert summary.returned_cartons == 1
    assert summary.net_cartons == -1


# ---------------------------------------------------------------------------
# 5. Jitter on the line is one crossing, not many
# ---------------------------------------------------------------------------

def test_jitter_on_the_line_counts_once():
    tracker = _tracker(hyst=10, cooldown=15)

    positions = [600, 620, 635, 645, 655, 645, 655, 645, 655, 645, 655]
    for frame, x in enumerate(positions, start=1):
        tracker.update([(1, (x - 30, 300.0, x + 30, 360.0), 0.9)], frame, frame / 30.0)

    ins = [c for c in tracker.crossings if c.direction == "IN"]
    outs = [c for c in tracker.crossings if c.direction == "OUT"]
    assert len(ins) == 1, "vibration around the line must not create repeated counts"
    assert len(outs) == 0


# ---------------------------------------------------------------------------
# 6. Two cartons crossing at the same instant stay two
# ---------------------------------------------------------------------------

def test_two_cartons_side_by_side_count_two():
    cartons = [
        {"carton_id": 1, "cross_time": 3.0, "lane": -80.0},
        {"carton_id": 2, "cross_time": 3.0, "lane": +80.0},
    ]
    dets = simulate_scenario(cartons, duration_sec=6.0)

    summary = _run(dets)

    assert summary.total_unique_cartons == 2, "one-per-camera matching must not merge lanes"
    assert summary.raw_crossings_total == 6
    assert all(len(c.cameras_seen) == 3 for c in summary.cartons)


# ---------------------------------------------------------------------------
# 7. Camera dropout must not lose cartons
# ---------------------------------------------------------------------------

def test_camera_dropout_still_counts_all_cartons():
    cartons = evenly_spaced_cartons(6, first_cross_time=1.0, interval=1.0)
    # The right camera sees nothing at all.
    dets = simulate_scenario(cartons, duration_sec=9.0,
                             skip_by_camera={"cam_right": [1, 2, 3, 4, 5, 6]})

    summary = _run(dets)

    assert summary.raw_crossings_by_camera.get("cam_right", 0) == 0
    assert summary.total_unique_cartons == 6, "two working cameras must still count all 6"


def test_anchor_camera_missing_one_carton_is_recovered():
    cartons = evenly_spaced_cartons(6, first_cross_time=1.0, interval=1.0)
    # The top camera (forced anchor) misses carton 3.
    dets = simulate_scenario(cartons, duration_sec=9.0,
                             skip_by_camera={"cam_top": [3]})

    summary = _run(dets, anchor_camera="cam_top")

    assert summary.raw_crossings_by_camera["cam_top"] == 5
    assert summary.total_unique_cartons == 6, "the two other angles recover the missed carton"


# ---------------------------------------------------------------------------
# 8. Same camera re-identifying one carton must not double count
# ---------------------------------------------------------------------------

def test_duplicate_track_from_same_camera_is_deduped():
    cam = CameraGeometry("cam_top", "horizontal", line_pos=300.0,
                         direction="up_to_down", lane_axis=640.0)
    frames = simulate_camera(
        [{"carton_id": 1, "cross_time": 2.0, "lane": 0.0},
         {"carton_id": 2, "cross_time": 2.1, "lane": 0.0}],
        cam, duration_sec=5.0,
    )
    # Make both detections look like the SAME physical carton: same track id
    # would already be one crossing, so instead use two ids at nearly the same
    # place/time, which is what a tracker re-ID produces.
    summary_without = _run(
        {"cam_top": frames},
        cameras=[cam],
        time_window_sec=0.8,
    )

    fusion = CartonCrossingFusion(time_window_sec=0.8, same_camera_dedup_sec=0.5)
    tracker = CartonLineTracker("cam_top", **cam.tracker_config())  # type: ignore[arg-type]
    for i, dets in enumerate(frames):
        fusion.ingest_many(tracker.update(dets, i, i / 30.0))
    summary_with = fusion.fuse()

    assert summary_without.total_unique_cartons == 2
    assert summary_with.total_unique_cartons == 1, "re-identified carton must count once"


# ---------------------------------------------------------------------------
# 9. min_cameras rejects single-camera noise
# ---------------------------------------------------------------------------

def test_min_cameras_rejects_uncorroborated_false_positive():
    cartons = evenly_spaced_cartons(3, first_cross_time=1.0, interval=1.0)
    dets = simulate_scenario(cartons, duration_sec=6.0,
                             skip_by_camera={"cam_right": [1, 2, 3], "cam_left": [1, 2, 3]})

    summary = _run(dets, min_cameras=2)

    assert summary.raw_crossings_total == 3
    assert summary.total_unique_cartons == 0
    assert summary.uncorroborated_cartons == 3


# ---------------------------------------------------------------------------
# 10. Explicit worked example (the exact user scenario)
# ---------------------------------------------------------------------------

def test_user_scenario_three_cameras_one_carton_is_one():
    """Right + Left + Top all see the same carton cross the red line."""
    engine = CartonCrossingFusion(time_window_sec=0.8)

    # Each camera reports the SAME physical carton at a slightly different time.
    engine.ingest(CartonCrossing("cam_right", 11, 10.00, "IN", (700, 300, 760, 360)))
    engine.ingest(CartonCrossing("cam_left", 22, 10.08, "IN", (520, 300, 580, 360)))
    engine.ingest(CartonCrossing("cam_top", 33, 10.14, "IN", (640, 360, 700, 420)))

    summary = engine.fuse()

    assert summary.total_unique_cartons == 1
    assert summary.raw_crossings_total == 3
    assert summary.duplicates_removed == 2
    assert summary.cartons[0].cameras_seen == ["cam_right", "cam_left", "cam_top"]


def test_user_scenario_three_separate_cartons_is_three():
    """Three cartons cross in sequence; each camera sees all three."""
    engine = CartonCrossingFusion(time_window_sec=0.8)

    for i in range(3):
        t = 10.0 + i * 1.5
        engine.ingest(CartonCrossing("cam_right", 100 + i, t + 0.00, "IN", (700, 300, 760, 360)))
        engine.ingest(CartonCrossing("cam_left", 200 + i, t + 0.08, "IN", (520, 300, 580, 360)))
        engine.ingest(CartonCrossing("cam_top", 300 + i, t + 0.14, "IN", (640, 360, 700, 420)))

    summary = engine.fuse()

    assert summary.raw_crossings_total == 9
    assert summary.total_unique_cartons == 3
    assert summary.duplicates_removed == 6
