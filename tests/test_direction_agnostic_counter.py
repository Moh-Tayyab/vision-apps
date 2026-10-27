import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "truck_loading_detection"))
from counter_engine import TripwireCounter, FlightCartonTracker

def test_direction_agnostic_cargo_left_to_right():
    """Verify loading_direction='any' counts carton crossing left-to-right."""
    counter = TripwireCounter(line_x=300, hysteresis=10, loading_direction="any")
    
    # Track 100 is a carton moving left to right: starts at x=250, crosses to x=350
    events1 = counter.update([(100, (240, 200, 260, 240), "carton", 0.9)], frame_idx=1)
    assert counter.net_count == 0
    
    events2 = counter.update([(100, (340, 200, 360, 240), "carton", 0.9)], frame_idx=2)
    assert len(events2) == 1
    assert events2[0]["is_cargo"] is True
    assert events2[0]["direction"] == "IN"
    assert counter.net_count == 1

def test_direction_agnostic_cargo_right_to_left():
    """Verify loading_direction='any' counts carton crossing right-to-left."""
    counter = TripwireCounter(line_x=300, hysteresis=10, loading_direction="any")
    
    # Track 101 is a carton moving right to left: starts at x=350, crosses to x=250
    events1 = counter.update([(101, (340, 200, 360, 240), "carton", 0.9)], frame_idx=1)
    assert counter.net_count == 0
    
    events2 = counter.update([(101, (240, 200, 260, 240), "carton", 0.9)], frame_idx=2)
    assert len(events2) == 1
    assert events2[0]["is_cargo"] is True
    assert events2[0]["direction"] == "IN"
    assert counter.net_count == 1

def test_direction_agnostic_worker_holding_vs_empty():
    """Verify worker holding carton counts +1, but empty-handed return triggers 0 count."""
    counter = TripwireCounter(line_x=300, hysteresis=10, loading_direction="any")
    
    # Worker 1 carries carton 200 left-to-right (starts at x=250, moves to x=350)
    w_box_start = (230, 100, 270, 300)
    c_box_start = (240, 150, 260, 180)  # held in chest
    counter.update([(1, w_box_start, "person", 0.9), (200, c_box_start, "carton", 0.85)], frame_idx=1)
    
    w_box_end = (330, 100, 370, 300)
    c_box_end = (340, 150, 360, 180)
    events = counter.update([(1, w_box_end, "person", 0.9), (200, c_box_end, "carton", 0.85)], frame_idx=5)
    
    # Carton was counted
    assert counter.net_count == 1
    
    # Now worker 1 returns right-to-left EMPTY-HANDED (carton stays in truck)
    w_box_return_start = (330, 100, 370, 300)
    counter.update([(1, w_box_return_start, "person", 0.9)], frame_idx=30)
    
    w_box_return_cross = (230, 100, 270, 300)
    return_events = counter.update([(1, w_box_return_cross, "person", 0.9)], frame_idx=35)
    
    # Net count must STILL BE 1 (return does not decrement or double-count!)
    assert counter.net_count == 1

def test_horizontal_orientation_with_line_y():
    """Verify horizontal tripwire orientation tracking along Y axis."""
    counter = TripwireCounter(line_y=400, line_orientation="horizontal", hysteresis=15, loading_direction="any")
    assert counter.line_orientation == "horizontal"
    assert counter.line_y == 400
    
    # Carton moving from top (y=300) to bottom (y=500)
    counter.update([(50, (100, 290, 140, 310), "carton", 0.9)], frame_idx=1)
    events = counter.update([(50, (100, 490, 140, 510), "carton", 0.9)], frame_idx=3)
    assert len(events) == 1
    assert counter.net_count == 1
    
    # Another carton moving from bottom (y=500) to top (y=300)
    counter.update([(51, (100, 490, 140, 510), "carton", 0.9)], frame_idx=10)
    events2 = counter.update([(51, (100, 290, 140, 310), "carton", 0.9)], frame_idx=13)
    assert len(events2) == 1
    assert counter.net_count == 2
