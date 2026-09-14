import pytest
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "truck_loading_detection"))

from counter_engine import TripwireCounter, is_cargo_held_by_person


def test_is_cargo_held_by_person_detection():
    # Worker box: [100, 50, 300, 500] (width 200, height 450)
    worker_box = (100.0, 50.0, 300.0, 500.0)
    
    # Carton box held in worker's chest/arms: [150, 200, 250, 300] (fully inside worker box)
    carton_box_held = (150.0, 200.0, 250.0, 300.0)
    is_held, ioca = is_cargo_held_by_person(carton_box_held, worker_box)
    assert is_held is True
    assert ioca > 0.90

    # Carton box far away: [600, 200, 700, 300]
    carton_box_away = (600.0, 200.0, 700.0, 300.0)
    is_held_away, ioca_away = is_cargo_held_by_person(carton_box_away, worker_box)
    assert is_held_away is False
    assert ioca_away == 0.0


def test_worker_carrying_carton_no_double_count():
    """
    Test that when a worker carries a carton across the tripwire (line_x=500),
    the system increments cargo count by strictly +1 (NOT +2).
    """
    counter = TripwireCounter(
        line_x=500,
        hysteresis=15,
        cooldown_frames=10,
        min_displacement_px=15,
    )

    # Frame 1: Worker (ID 1) & Carton (ID 2) on LEFT side (X ~ 400)
    # Worker: [300, 100, 480, 500] -> center_x = 390
    # Carton: [370, 250, 470, 370] -> center_x = 420
    tracked_f1 = [
        (1, (300.0, 100.0, 480.0, 500.0), "person", 0.90),
        (2, (370.0, 250.0, 470.0, 370.0), "carton", 0.85),
    ]
    events_f1 = counter.update(tracked_f1, frame_idx=1)
    assert len(events_f1) == 0
    assert counter.total_in == 0
    assert counter.get_associated_person(2, frame_idx=1) == 1

    # Frame 2: Both cross over to RIGHT side (X ~ 580)
    # Worker: [500, 100, 680, 500] -> center_x = 590 (>= 500 + 15)
    # Carton: [570, 250, 670, 370] -> center_x = 620 (>= 500 + 15)
    tracked_f2 = [
        (1, (500.0, 100.0, 680.0, 500.0), "person", 0.90),
        (2, (570.0, 250.0, 670.0, 370.0), "carton", 0.85),
    ]
    events_f2 = counter.update(tracked_f2, frame_idx=2)

    # STRICT ASSERTION: Exactly +1 cargo counted, NOT +2
    assert counter.total_in == 1
    assert counter.total_out == 0
    assert counter.net_count == 1
    assert len(events_f2) == 1
    assert events_f2[0]["track_id"] == 2
    assert events_f2[0]["is_cargo"] is True
    assert events_f2[0]["carried_by"] == 1


def test_empty_handed_worker_return_does_not_decrement_cargo():
    """
    Test that an empty-handed worker walking back (Right -> Left)
    does NOT decrement the cargo count.
    """
    counter = TripwireCounter(
        line_x=500,
        hysteresis=15,
        cooldown_frames=10,
        min_displacement_px=15,
    )

    # Initial state: 1 carton loaded
    counter.total_in = 1

    # Frame 1: Empty-handed worker on RIGHT side
    # Worker: [550, 100, 730, 500] -> center_x = 640
    tracked_f1 = [
        (1, (550.0, 100.0, 730.0, 500.0), "person", 0.90),
    ]
    counter.update(tracked_f1, frame_idx=1)

    # Frame 2: Worker crosses to LEFT side without cargo
    # Worker: [300, 100, 480, 500] -> center_x = 390
    tracked_f2 = [
        (1, (300.0, 100.0, 480.0, 500.0), "person", 0.90),
    ]
    events_f2 = counter.update(tracked_f2, frame_idx=2)

    # Total out must NOT increment (empty worker is not cargo)
    assert counter.total_out == 0
    assert counter.net_count == 1
    assert counter.worker_trips_out == 1


def test_multiple_workers_carrying_separate_cartons():
    """
    Test 2 distinct workers carrying 2 separate cartons across line.
    Net count should be exactly 2 (not 4).
    """
    counter = TripwireCounter(
        line_x=500,
        hysteresis=15,
        cooldown_frames=5,
        min_displacement_px=15,
    )

    # Frame 1: Worker 1 with Carton 1, Worker 2 with Carton 2 on LEFT
    tracked_f1 = [
        (1, (200.0, 50.0, 380.0, 350.0), "person", 0.92),
        (10, (260.0, 150.0, 350.0, 250.0), "carton", 0.88),  # Held by Worker 1
        (2, (200.0, 400.0, 380.0, 700.0), "person", 0.91),
        (20, (260.0, 500.0, 350.0, 600.0), "carton", 0.86),  # Held by Worker 2
    ]
    counter.update(tracked_f1, frame_idx=1)
    assert counter.current_associations.get(10) == 1
    assert counter.current_associations.get(20) == 2

    # Frame 2: Both cross to RIGHT side
    tracked_f2 = [
        (1, (550.0, 50.0, 730.0, 350.0), "person", 0.92),
        (10, (610.0, 150.0, 700.0, 250.0), "carton", 0.88),
        (2, (550.0, 400.0, 730.0, 700.0), "person", 0.91),
        (20, (610.0, 500.0, 700.0, 600.0), "carton", 0.86),
    ]
    events = counter.update(tracked_f2, frame_idx=2)

    assert counter.total_in == 2
    assert counter.net_count == 2
    assert len(events) == 2
    cargos_counted = {ev["track_id"] for ev in events}
    assert cargos_counted == {10, 20}


def test_holding_temporal_memory_through_occlusion():
    """
    Test holding memory: even if carton detection is momentarily occluded in 1 frame,
    the worker-carton association persists.
    """
    counter = TripwireCounter(
        line_x=500,
        hysteresis=15,
        holding_memory_frames=10,
    )

    # Frame 1: Worker 1 holds Carton 5
    tracked_f1 = [
        (1, (200.0, 100.0, 380.0, 450.0), "person", 0.9),
        (5, (250.0, 200.0, 340.0, 300.0), "carton", 0.85),
    ]
    counter.update(tracked_f1, frame_idx=1)
    assert counter.get_associated_person(5, frame_idx=1) == 1

    # Frame 5: Carton temporarily lost by detector, only worker tracked
    tracked_f5 = [
        (1, (250.0, 100.0, 430.0, 450.0), "person", 0.9),
    ]
    counter.update(tracked_f5, frame_idx=5)
    # Memory should remember worker 1 was holding carton 5
    assert counter.get_held_cargos(1, frame_idx=5) == [5]
