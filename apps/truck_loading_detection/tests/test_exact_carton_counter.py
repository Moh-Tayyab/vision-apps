"""
Unit tests for ThreeCameraCartonCounter (exact_carton_counter.py).

Verifies:
1. Per-camera truck_side derives correct loading_direction (IN direction).
2. IN crossing (+1) when carton moves toward truck side.
3. OUT crossing (-1) when carton returns from truck to pallet.
4. 2-of-3 camera fusion: same carton seen by 2 cameras counted once (+1).
5. Single camera miss recovered by other 2 cameras (redundancy).
6. All 3 cameras see same carton: still counted once (no double count).
7. Cooldown suppresses rapid back-and-forth oscillation.
8. Per-camera line position via set_line_frac.
9. get_counts returns per-camera + fused breakdown.
10. reset clears all counters.
"""

import pytest
from three_camera_system.exact_carton_counter import (
    ThreeCameraCartonCounter,
    CameraTripwireCounter,
    CAMERA_IDS,
)


def _make_track(track_id, cx, cy, w=60, h=60, conf=0.9):
    """(track_id, (x1,y1,x2,y2), cls, conf)"""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return (track_id, (x1, y1, x2, y2), "carton", conf)


class TestTruckSideDirection:
    """truck_side determines which way is IN (+1)."""

    def test_truck_side_right_loads_left_to_right(self):
        """Truck on right: carton moves left->right across line = IN (+1)."""
        c = CameraTripwireCounter("cam_side", line_x=500, truck_side="right")
        # Start left of line, move right across it
        evs = c.update([_make_track(1, 450, 300)], frame_idx=1)  # left side
        assert len(evs) == 0
        evs = c.update([_make_track(1, 550, 300)], frame_idx=2)  # crossed to right
        assert len(evs) == 1
        assert evs[0].direction == "IN"
        assert c.total_in == 1

    def test_truck_side_left_loads_right_to_left(self):
        """Truck on left: carton moves right->left across line = IN (+1)."""
        c = CameraTripwireCounter("cam_top", line_x=500, truck_side="left")
        # Start right of line, move left across it
        evs = c.update([_make_track(1, 550, 300)], frame_idx=1)  # right side
        assert len(evs) == 0
        evs = c.update([_make_track(1, 450, 300)], frame_idx=2)  # crossed to left
        assert len(evs) == 1
        assert evs[0].direction == "IN"
        assert c.total_in == 1

    def test_truck_side_right_return_is_out(self):
        """Truck on right: carton moves right->left = OUT (-1)."""
        c = CameraTripwireCounter("cam_side", line_x=500, truck_side="right")
        evs = c.update([_make_track(1, 550, 300)], frame_idx=1)
        assert len(evs) == 0
        evs = c.update([_make_track(1, 450, 300)], frame_idx=2)
        assert len(evs) == 1
        assert evs[0].direction == "OUT"
        assert c.total_out == 1

    def test_set_truck_side_changes_direction(self):
        """set_truck_side flips loading_direction."""
        c = CameraTripwireCounter("cam_side", line_x=500, truck_side="right")
        assert c.loading_direction == "left_to_right"
        c.set_truck_side("left")
        assert c.loading_direction == "right_to_left"
        assert c.truck_side == "left"


class TestThreeCameraFusion:
    """Cross-camera fusion dedupes same physical carton."""

    def test_two_cameras_same_carton_counted_once(self):
        """2 cameras detect same carton -> fused count = +1, not +2."""
        cc = ThreeCameraCartonCounter(
            truck_sides={"cam_top": "left", "cam_side": "left", "cam_flank": "left"},
            line_x=500,
        )
        # truck_side=left: IN = right->left (decreasing X)
        # Top camera sees crossing
        cc.update_camera("cam_top", [_make_track(1, 550, 300)], frame_idx=10)
        evs = cc.update_camera("cam_top", [_make_track(1, 450, 300)], frame_idx=11)
        assert cc.total_in == 1
        # Side camera sees same carton (within fusion window) -> absorbed, no new count
        cc.update_camera("cam_side", [_make_track(2, 550, 300)], frame_idx=10)
        evs2 = cc.update_camera("cam_side", [_make_track(2, 450, 300)], frame_idx=11)
        assert cc.total_in == 1, f"Expected total_in=1, got {cc.total_in}"

    def test_two_cameras_in_direction_counted_once(self):
        """2 cameras see same carton moving IN -> fused +1 total."""
        cc = ThreeCameraCartonCounter(
            truck_sides={"cam_top": "left", "cam_side": "left", "cam_flank": "left"},
            line_x=500,
        )
        # truck_side=left: IN = right->left (moving toward decreasing X)
        # Top camera: carton crosses from right (550) to left (450)
        evs1 = cc.update_camera("cam_top", [_make_track(1, 550, 300)], frame_idx=10)
        assert len(evs1) == 0  # First frame, establishes side
        evs2 = cc.update_camera("cam_top", [_make_track(1, 450, 300)], frame_idx=11)
        assert len(evs2) == 1
        assert evs2[0]["direction"] == "IN"
        # Side camera sees same carton at slightly different time (within fusion window)
        evs3 = cc.update_camera("cam_side", [_make_track(2, 550, 300)], frame_idx=10)
        assert len(evs3) == 0
        evs4 = cc.update_camera("cam_side", [_make_track(2, 450, 300)], frame_idx=11)
        assert len(evs4) == 1  # Same fused event, no new count
        # Total fused count should be 1, not 2
        assert cc.total_in == 1, f"Expected total_in=1, got {cc.total_in}"
        assert cc.total_out == 0
        assert cc.net_count == 1

    def test_three_cameras_same_carton_counted_once(self):
        """All 3 cameras see same carton -> fused count = +1."""
        cc = ThreeCameraCartonCounter(
            truck_sides={"cam_top": "left", "cam_side": "left", "cam_flank": "left"},
            line_x=500,
        )
        for cid in CAMERA_IDS:
            cc.update_camera(cid, [_make_track(1, 550, 300)], frame_idx=10)
            evs = cc.update_camera(cid, [_make_track(1, 450, 300)], frame_idx=11)
            # First camera emits, others are absorbed
        assert cc.total_in == 1, f"Expected total_in=1, got {cc.total_in}"
        assert cc.net_count == 1

    def test_single_camera_miss_recovered_by_others(self):
        """If one camera misses, the other two still fuse +1."""
        cc = ThreeCameraCartonCounter(
            truck_sides={"cam_top": "left", "cam_side": "left", "cam_flank": "left"},
            line_x=500,
        )
        # Top and side detect, flank misses entirely
        for cid in ["cam_top", "cam_side"]:
            cc.update_camera(cid, [_make_track(1, 550, 300)], frame_idx=10)
            evs = cc.update_camera(cid, [_make_track(1, 450, 300)], frame_idx=11)
        assert cc.total_in == 1
        assert cc.net_count == 1

    def test_out_direction_decrements_net(self):
        """OUT crossing (-1) reduces net count."""
        cc = ThreeCameraCartonCounter(
            truck_sides={"cam_top": "right", "cam_side": "right", "cam_flank": "right"},
            line_x=500,
        )
        # truck_side=right: IN = left->right, OUT = right->left
        # Load a carton (IN)
        cc.update_camera("cam_top", [_make_track(1, 450, 300)], frame_idx=10)
        cc.update_camera("cam_top", [_make_track(1, 550, 300)], frame_idx=11)
        assert cc.total_in == 1
        assert cc.net_count == 1
        # Return it (OUT)
        cc.update_camera("cam_side", [_make_track(2, 550, 300)], frame_idx=20)
        evs = cc.update_camera("cam_side", [_make_track(2, 450, 300)], frame_idx=21)
        assert cc.total_out == 1
        assert cc.net_count == 0

    def test_cooldown_suppresses_oscillation(self):
        """Rapid back-and-forth within cooldown produces exactly one event."""
        cc = ThreeCameraCartonCounter(
            truck_sides={"cam_top": "right", "cam_side": "right", "cam_flank": "right"},
            line_x=500,
            cooldown_frames=15,
        )
        # Cross IN (left->right)
        cc.update_camera("cam_top", [_make_track(1, 450, 300)], frame_idx=10)
        evs = cc.update_camera("cam_top", [_make_track(1, 550, 300)], frame_idx=11)
        assert cc.total_in == 1
        # Cross back OUT within cooldown (frame 12 < 11+15) -> suppressed
        evs = cc.update_camera("cam_top", [_make_track(1, 450, 300)], frame_idx=12)
        assert cc.total_out == 0
        # Cross IN again within cooldown -> suppressed (no duplicate IN)
        evs = cc.update_camera("cam_top", [_make_track(1, 550, 300)], frame_idx=13)
        assert cc.total_in == 1, f"Expected total_in=1, got {cc.total_in}"
        assert cc.net_count == 1

    def test_per_camera_counts_breakdown(self):
        """get_counts returns per-camera + fused totals."""
        cc = ThreeCameraCartonCounter(
            truck_sides={"cam_top": "left", "cam_side": "left", "cam_flank": "left"},
            line_x=500,
        )
        # Top camera: one IN crossing
        cc.update_camera("cam_top", [_make_track(1, 550, 300)], frame_idx=10)
        cc.update_camera("cam_top", [_make_track(1, 450, 300)], frame_idx=11)
        counts = cc.get_counts()
        assert counts["total_in"] == 1
        assert counts["total_out"] == 0
        assert counts["net_count"] == 1
        assert "cameras" in counts
        assert "cam_top" in counts["cameras"]
        assert "cam_side" in counts["cameras"]
        assert "cam_flank" in counts["cameras"]

    def test_reset_clears_all(self):
        """reset() clears counters and events."""
        cc = ThreeCameraCartonCounter(line_x=500, truck_sides={cid: "right" for cid in CAMERA_IDS})
        cc.update_camera("cam_top", [_make_track(1, 450, 300)], frame_idx=10)
        cc.update_camera("cam_top", [_make_track(1, 550, 300)], frame_idx=11)
        assert cc.total_in == 1
        cc.reset()
        assert cc.total_in == 0
        assert cc.total_out == 0
        assert cc.net_count == 0
        assert len(cc.fused_events) == 0

    def test_set_line_frac_updates_line_position(self):
        """set_line_frac moves the tripwire line."""
        cc = ThreeCameraCartonCounter(
            truck_sides={"cam_top": "right", "cam_side": "right", "cam_flank": "right"},
            line_x=500,
        )
        cc.set_line_frac("cam_top", 0.25, 1000, 720, "vertical")
        assert cc.counters["cam_top"].line_x == 250
        cc.set_line_frac("cam_top", 0.75, 1000, 720, "vertical")
        assert cc.counters["cam_top"].line_x == 750
