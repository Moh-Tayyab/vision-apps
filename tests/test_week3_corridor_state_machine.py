"""
Unit and Integration Tests for Week 3: Corridor State Machine (Replacing 1D Tripwires).

Covers:
1. CrossingCorridor per-camera 3-zone spatial classification:
   - Entry Zone: Worker + cartons cross dock threshold (approach)
   - Transit Zone: Bundle active; multi-camera association corridor
   - Exit Zone: Bundle enters truck interior (destination)
2. CorridorStateMachine management of TransitBundle:
   - State machine holds worker_id, carton_bundle, start_t, direction.
   - TimelineAligner 6.0s window coalescence across cam_top, cam_side, cam_flank.
3. Hysteresis Guarantees:
   - Worker pausing or stepping back inside Transit Zone DOES NOT increment/decrement.
   - Zero count change until bundle fully exits corridor into interior.
4. Reversal / Retreat Cancellation:
   - Worker retreating back through Entry Zone cancels bundle (REVERSED_CANCELLED) with 0 count.
5. Exactly-Once Commit Guarantee:
   - Bundle entering Exit Zone fires EXACTLY ONE commit event with stack disambiguation.
   - Subsequent lingering frames in Exit Zone do not produce duplicate commit events.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Tuple

from apps.truck_loading_detection.three_camera_system.corridor import (
    BundleState,
    CorridorStateMachine,
    CorridorZone,
    CrossingCorridor,
    TransitBundle,
)
from apps.truck_loading_detection.three_camera_system.consensus_policy import (
    ConsensusLevel,
    DecisionAction,
    ThreeCameraConsensusPolicy,
)
from apps.truck_loading_detection.three_camera_system.stack_disambiguator import (
    StackDisambiguator,
)


@dataclass
class MockTracklet:
    """Mock carton tracklet for testing corridor state transitions."""
    local_track_id: str
    center: Tuple[float, float]
    current_bbox: Tuple[float, float, float, float] = (400.0, 200.0, 500.0, 300.0)
    current_confidence: float = 0.88
    estimated_qty: int = 1
    worker_id: str = "worker_alpha"


# ===========================================================================
# 1. CrossingCorridor 3-Zone Spatial Classification
# ===========================================================================
def test_crossing_corridor_zone_classification_negative_direction():
    """
    Test 3-zone classification for cam_side and cam_top (direction_sign = -1).
    Dock threshold is on right (X > line + hw), truck interior on left (X < line - hw).
    """
    corridor = CrossingCorridor(
        camera_id="cam_side",
        line_position_px=500.0,
        threshold_half_width_px=40.0,
        orientation="vertical",
        direction_sign=-1,
    )

    # 1. Entry Zone (Dock threshold / approach: X > 540)
    zone_entry = corridor.classify_zone((600.0, 300.0))
    assert zone_entry == CorridorZone.ENTRY_ZONE
    assert zone_entry.is_entry
    assert not zone_entry.is_transit
    assert not zone_entry.is_exit

    # 2. Transit Zone (Corridor body: 460 <= X <= 540)
    zone_transit = corridor.classify_zone((500.0, 300.0))
    assert zone_transit == CorridorZone.TRANSIT_ZONE
    assert zone_transit.is_transit
    assert not zone_transit.is_entry

    # 3. Exit Zone (Truck interior: X < 460)
    zone_exit = corridor.classify_zone((400.0, 300.0))
    assert zone_exit == CorridorZone.EXIT_ZONE
    assert zone_exit.is_exit
    assert not zone_exit.is_entry
    assert not zone_exit.is_transit


def test_crossing_corridor_zone_classification_positive_direction():
    """
    Test 3-zone classification for cam_flank (direction_sign = +1).
    Dock threshold is on left (X < line - hw), truck interior on right (X > line + hw).
    """
    corridor = CrossingCorridor(
        camera_id="cam_flank",
        line_position_px=550.0,
        threshold_half_width_px=40.0,
        orientation="vertical",
        direction_sign=1,
    )

    # 1. Entry Zone (Dock threshold / approach: X < 510)
    zone_entry = corridor.classify_zone((450.0, 300.0))
    assert zone_entry == CorridorZone.ENTRY_ZONE
    assert zone_entry.is_entry

    # 2. Transit Zone (Corridor body: 510 <= X <= 590)
    zone_transit = corridor.classify_zone((550.0, 300.0))
    assert zone_transit == CorridorZone.TRANSIT_ZONE
    assert zone_transit.is_transit

    # 3. Exit Zone (Truck interior: X > 590)
    zone_exit = corridor.classify_zone((650.0, 300.0))
    assert zone_exit == CorridorZone.EXIT_ZONE
    assert zone_exit.is_exit


# ===========================================================================
# 2. Bundle Coalescence across cam_top, cam_side, cam_flank (6s Window)
# ===========================================================================
def test_corridor_state_machine_multi_camera_bundle_coalescence():
    """
    Verify tracklets from cam_side, cam_top, and cam_flank are coalesced
    into a single TransitBundle within the 6.0s TimelineAligner window.
    """
    sm = CorridorStateMachine(coalesce_window_sec=6.0)
    disambiguator = StackDisambiguator()
    consensus = ThreeCameraConsensusPolicy()

    # Time t = 1.0s: cam_side detects worker + carton at dock entry (x = 650)
    trk_side = MockTracklet(local_track_id="side_01", center=(650.0, 300.0), estimated_qty=2, worker_id="worker_01")
    sm.update(
        camera_tracklets={"cam_side": [trk_side]},
        timestamp_sec=1.0,
        disambiguator=disambiguator,
        consensus_policy=consensus,
    )

    assert len(sm.active_bundles) == 1
    bundle = sm.active_bundles[0]
    assert bundle.worker_id == "worker_01"
    assert bundle.state == BundleState.ENTERED
    assert "cam_side" in bundle.associated_tracks
    assert bundle.associated_tracks["cam_side"] == {"side_01"}

    # Time t = 1.5s: cam_top and cam_flank detect the same worker moving into transit
    # cam_top: x = 500 (transit zone, sign = -1)
    # cam_flank: x = 550 (transit zone, sign = +1)
    trk_top = MockTracklet(local_track_id="top_01", center=(500.0, 250.0), estimated_qty=2, worker_id="worker_01")
    trk_flank = MockTracklet(local_track_id="flank_01", center=(550.0, 260.0), estimated_qty=2, worker_id="worker_01")

    sm.update(
        camera_tracklets={"cam_top": [trk_top], "cam_flank": [trk_flank]},
        timestamp_sec=1.5,
        disambiguator=disambiguator,
        consensus_policy=consensus,
    )

    # Should coalesce into the EXACT same bundle!
    assert len(sm.active_bundles) == 1
    bundle = sm.active_bundles[0]
    assert bundle.state == BundleState.IN_TRANSIT
    assert set(bundle.associated_tracks.keys()) == {"cam_side", "cam_top", "cam_flank"}
    assert bundle.start_t == 1.0
    assert bundle.last_t == 1.5
    assert bundle.view_carton_counts["cam_side"] == 2
    assert bundle.view_carton_counts["cam_top"] == 2
    assert bundle.view_carton_counts["cam_flank"] == 2


# ===========================================================================
# 3. Hysteresis: Worker Pauses or Steps Back in Transit Corridor
# ===========================================================================
def test_corridor_state_machine_hysteresis_pausing_and_stepback():
    """
    Hysteresis Guarantee:
    If a worker enters the Transit Zone and hesitates, pauses, or steps back slightly,
    DO NOT increment or decrement until the bundle fully exits into the truck interior.
    """
    sm = CorridorStateMachine(coalesce_window_sec=6.0)

    # Step 1: Worker enters Entry Zone at t = 2.0s
    trk_side = MockTracklet(local_track_id="trk_h1", center=(620.0, 300.0))
    sm.update({"cam_side": [trk_side]}, timestamp_sec=2.0)
    assert len(sm.active_bundles) == 1
    assert sm.active_bundles[0].state == BundleState.ENTERED
    assert sm.net_count == 0

    # Step 2: Moves into Transit Zone at t = 2.4s (x = 510)
    trk_side.center = (510.0, 300.0)
    sm.update({"cam_side": [trk_side]}, timestamp_sec=2.4)
    assert sm.active_bundles[0].state == BundleState.IN_TRANSIT
    assert sm.net_count == 0
    assert len(sm.committed_events) == 0

    # Step 3: Worker pauses in transit zone at t = 2.8s, 3.2s, 3.6s
    for t_step in (2.8, 3.2, 3.6):
        trk_side.center = (505.0, 300.0)
        newly_committed = sm.update({"cam_side": [trk_side]}, timestamp_sec=t_step)
        assert len(newly_committed) == 0
        assert sm.net_count == 0
        assert sm.active_bundles[0].state == BundleState.IN_TRANSIT
        assert sm.active_bundles[0].hesitation_count > 0

    # Step 4: Worker micro-steps backward towards dock boundary but stays in transit (x = 535)
    trk_side.center = (535.0, 300.0)
    newly_committed = sm.update({"cam_side": [trk_side]}, timestamp_sec=4.0)
    assert len(newly_committed) == 0
    assert sm.net_count == 0
    assert sm.active_bundles[0].state == BundleState.IN_TRANSIT

    # Step 5: Worker resumes and fully exits into truck interior at t = 4.8s (x = 400 < 460)
    trk_side.center = (400.0, 300.0)
    newly_committed = sm.update({"cam_side": [trk_side]}, timestamp_sec=4.8)

    # NOW AND ONLY NOW fires commit event!
    assert len(newly_committed) == 1
    assert sm.net_count == 1
    assert newly_committed[0]["carton_count"] == 1
    assert len(sm.committed_bundles) == 1
    assert sm.committed_bundles[0].state == BundleState.COMMITTED


# ===========================================================================
# 4. Reversal / Retreat: Worker Returns to Dock -> Cancelled with 0 Count
# ===========================================================================
def test_corridor_state_machine_worker_reversal_cancels_with_zero_count():
    """
    If worker steps into transit corridor, then turns around and retreats back to dock,
    the bundle must be cancelled (REVERSED_CANCELLED) with exactly 0 count change.
    """
    sm = CorridorStateMachine(coalesce_window_sec=6.0)

    # Worker enters corridor at t = 1.0s (x = 650)
    trk = MockTracklet(local_track_id="trk_rev", center=(650.0, 300.0))
    sm.update({"cam_side": [trk]}, timestamp_sec=1.0)

    # Worker reaches transit zone at t = 1.5s (x = 500)
    trk.center = (500.0, 300.0)
    sm.update({"cam_side": [trk]}, timestamp_sec=1.5)
    assert sm.active_bundles[0].state == BundleState.IN_TRANSIT

    # Worker turns around and retreats back to dock Entry Zone at t = 2.2s (x = 640)
    trk.center = (640.0, 300.0)
    newly_committed = sm.update({"cam_side": [trk]}, timestamp_sec=2.2)

    assert len(newly_committed) == 0
    assert sm.net_count == 0
    bundle = sm._active_bundles[0]
    assert bundle.state == BundleState.REVERSED_CANCELLED
    assert bundle.reversal_detected is True
    assert len(sm.committed_events) == 0


# ===========================================================================
# 5. Exactly-Once Commit Guarantee & Stack Disambiguation
# ===========================================================================
def test_corridor_state_machine_exactly_once_commit_and_stack_resolution():
    """
    Verify bundle fires EXACTLY ONE commit event upon Exit Zone crossing,
    resolves stack count via StackDisambiguator, and ignores subsequent lingering frames.
    """
    sm = CorridorStateMachine(coalesce_window_sec=6.0)
    disambiguator = StackDisambiguator()
    consensus = ThreeCameraConsensusPolicy()

    # Side camera sees stack of 3 cartons; Top camera sees 1 column; Flank sees 3 cartons
    trk_side = MockTracklet(local_track_id="side_stk", center=(620.0, 300.0), estimated_qty=3)
    trk_top = MockTracklet(local_track_id="top_stk", center=(620.0, 300.0), estimated_qty=1)
    trk_flank = MockTracklet(local_track_id="flank_stk", center=(480.0, 300.0), estimated_qty=3)

    # Entry at t = 1.0s
    sm.update(
        {"cam_side": [trk_side], "cam_top": [trk_top], "cam_flank": [trk_flank]},
        timestamp_sec=1.0,
        disambiguator=disambiguator,
        consensus_policy=consensus,
    )

    # Transit at t = 1.6s
    trk_side.center = (500.0, 300.0)
    trk_top.center = (500.0, 300.0)
    trk_flank.center = (550.0, 300.0)
    sm.update(
        {"cam_side": [trk_side], "cam_top": [trk_top], "cam_flank": [trk_flank]},
        timestamp_sec=1.6,
        disambiguator=disambiguator,
        consensus_policy=consensus,
    )

    # Exit into truck interior at t = 2.4s
    # cam_side: x = 380 (< 460 exit)
    # cam_top: x = 380 (< 460 exit)
    # cam_flank: x = 680 (> 590 exit)
    trk_side.center = (380.0, 300.0)
    trk_top.center = (380.0, 300.0)
    trk_flank.center = (680.0, 300.0)
    commits_1 = sm.update(
        {"cam_side": [trk_side], "cam_top": [trk_top], "cam_flank": [trk_flank]},
        timestamp_sec=2.4,
        disambiguator=disambiguator,
        consensus_policy=consensus,
    )

    assert len(commits_1) == 1
    ev = commits_1[0]
    assert ev["carton_count"] == 3  # Stack disambiguated to 3 cartons (1 col x 3 high)
    assert set(ev["participating_cameras"]) == {"cam_side", "cam_top", "cam_flank"}
    assert ev["is_auto_confirmed"] is True
    assert sm.total_in == 3

    # Consecutive lingering frames inside truck interior at t = 2.8s, 3.2s, 3.6s
    for t_step in (2.8, 3.2, 3.6):
        commits_lingering = sm.update(
            {"cam_side": [trk_side], "cam_top": [trk_top], "cam_flank": [trk_flank]},
            timestamp_sec=t_step,
            disambiguator=disambiguator,
            consensus_policy=consensus,
        )
        # CRITICAL EXACTLY-ONCE INVARIANT: No duplicate commits!
        assert len(commits_lingering) == 0
        assert sm.total_in == 3
        assert len(sm.committed_events) == 1


# ===========================================================================
# 6. Re-entry After Cancellation
# ===========================================================================
def test_corridor_state_machine_reentry_after_cancellation():
    """
    Worker enters, retreats back to dock (bundle 1 cancelled with 0 count),
    then re-enters with fresh cartons (bundle 2) and completes transit into truck interior.
    Result: Net count matches bundle 2 exactly.
    """
    sm = CorridorStateMachine(coalesce_window_sec=6.0)

    # Bundle 1 enters at t = 1.0s
    trk1 = MockTracklet(local_track_id="trk_first", center=(650.0, 300.0), worker_id="worker_01")
    sm.update({"cam_side": [trk1]}, timestamp_sec=1.0)
    trk1.center = (500.0, 300.0)
    sm.update({"cam_side": [trk1]}, timestamp_sec=1.4)
    # Retreats back to dock
    trk1.center = (660.0, 300.0)
    sm.update({"cam_side": [trk1]}, timestamp_sec=2.0)
    assert sm.net_count == 0

    # Bundle 2 enters at t = 3.0s with 2 cartons
    trk2 = MockTracklet(local_track_id="trk_second", center=(650.0, 300.0), estimated_qty=2, worker_id="worker_01")
    sm.update({"cam_side": [trk2]}, timestamp_sec=3.0)
    assert len(sm.active_bundles) == 1
    assert sm.active_bundles[0].state == BundleState.ENTERED

    # Bundle 2 transits corridor at t = 3.6s
    trk2.center = (500.0, 300.0)
    sm.update({"cam_side": [trk2]}, timestamp_sec=3.6)
    assert sm.active_bundles[0].state == BundleState.IN_TRANSIT
    assert sm.net_count == 0

    # Bundle 2 exits into interior at t = 4.2s
    trk2.center = (400.0, 300.0)
    commits = sm.update({"cam_side": [trk2]}, timestamp_sec=4.2)
    assert len(commits) == 1
    assert commits[0]["carton_count"] == 2
    assert sm.net_count == 2
    assert len(sm.committed_bundles) == 1
    assert sm.committed_bundles[0].bundle_id == commits[0]["bundle_id"]


# ===========================================================================
# 7. Unloading Direction ("OUT")
# ===========================================================================
def test_corridor_state_machine_unloading_direction_out():
    """
    Corridor configured for unloading ("OUT"):
    Worker carries cartons from truck interior (entry) through transit to dock (exit).
    Verify total_out and net_count are updated accurately.
    """
    corridor_cfgs = {
        "cam_side": {"orientation": "vertical", "line_pos_frac": 0.50, "direction_sign": 1},  # For OUT, direction is reversed
    }
    sm = CorridorStateMachine(
        corridor_configs=corridor_cfgs,
        camera_ids=("cam_side",),
        required_direction="OUT",
        coalesce_window_sec=6.0,
    )

    # Worker starts in truck interior at t = 1.0s (x = 400 < 460)
    trk = MockTracklet(local_track_id="trk_unload", center=(400.0, 300.0), estimated_qty=1)
    sm.update({"cam_side": [trk]}, timestamp_sec=1.0)
    assert len(sm.active_bundles) == 1
    assert sm.active_bundles[0].direction == "OUT"

    # Moves into transit at t = 1.5s (x = 500)
    trk.center = (500.0, 300.0)
    sm.update({"cam_side": [trk]}, timestamp_sec=1.5)
    assert sm.active_bundles[0].state == BundleState.IN_TRANSIT

    # Exits to dock pallet at t = 2.0s (x = 620 > 540)
    trk.center = (620.0, 300.0)
    commits = sm.update({"cam_side": [trk]}, timestamp_sec=2.0)
    assert len(commits) == 1
    assert commits[0]["direction"] == "OUT"
    assert sm.total_out == 1
    assert sm.net_count == -1
