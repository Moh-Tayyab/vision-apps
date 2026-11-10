"""
Carton Crossing Event State Machine & Physical Carton Association (Phases 3 & 4).

Implements deterministic physical-carton tracking and doorway crossing transitions:
Lifecycle:
  NOT_SEEN -> APPROACHING -> CROSSING -> CROSSED -> COUNT_COMMITTED
  (Also supports REVIEW_REQUIRED, REVERSED_CANCELLED, and UNCERTAIN)

Guarantees & Invariants:
1. Exactly-Once Counting: One physical carton can transition to COUNT_COMMITTED at most once.
2. Multi-Camera Association: Multiple cameras observing the same carton coalesce into 1 physical identity.
3. Tracklet Splitting Resilience: If worker bodies occlude a carton and it reappears with a new local track ID,
   spatial-temporal continuity stitches the tracklet to the existing physical carton without duplicating counts.
4. Distinct Simultaneous Separation: Multiple workers carrying separate cartons simultaneously maintain distinct
   spatial identities and are each counted (+1 per physical carton).
5. Direction Enforced: Cartons moving opposite to required_direction or reversing from threshold are cancelled.
6. Review Isolation: Ambiguous or single-camera observations are strictly flagged as REVIEW_REQUIRED.
7. Transactional Deduplication: Committed crossing events have persistent, globally unique IDs and cannot be double-committed.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CrossingState(str, enum.Enum):
    NOT_SEEN = "NOT_SEEN"
    APPROACHING = "APPROACHING"
    CROSSING = "CROSSING"
    CROSSED = "CROSSED"
    COUNT_COMMITTED = "COUNT_COMMITTED"

    # Backward compatibility / aliases
    UNCONFIRMED = "UNCONFIRMED"
    OBSERVED = "OBSERVED"
    CROSSING_CANDIDATE = "CROSSING_CANDIDATE"
    CROSSING_CONFIRMED = "CROSSING_CONFIRMED"
    COUNTED = "COUNTED"
    UNCERTAIN = "UNCERTAIN"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REVERSED_CANCELLED = "REVERSED_CANCELLED"


class PhysicalAssociationOutcome(str, enum.Enum):
    """Explicit association outcomes for physical carton tracking."""
    MATCHED_EXISTING_PHYSICAL_CARTON = "MATCHED_EXISTING_PHYSICAL_CARTON"
    NEW_PHYSICAL_CARTON = "NEW_PHYSICAL_CARTON"
    AMBIGUOUS_ASSOCIATION = "AMBIGUOUS_ASSOCIATION"
    TEMPORARY_OCCLUSION = "TEMPORARY_OCCLUSION"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass
class CommittedCrossingEvent:
    """An immutable audit record of a committed carton crossing event."""

    event_id: str
    global_carton_id: str
    timestamp_sec: float
    direction: str  # "IN" (into truck) or "OUT" (out of truck)
    carton_count: int = 1
    participating_cameras: List[str] = field(default_factory=list)
    state: CrossingState = CrossingState.COUNT_COMMITTED
    is_ambiguous: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "global_carton_id": self.global_carton_id,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "direction": self.direction,
            "carton_count": self.carton_count,
            "participating_cameras": list(self.participating_cameras),
            "state": self.state.value,
            "is_ambiguous": self.is_ambiguous,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommittedCrossingEvent":
        st_val = data.get("state", "COUNT_COMMITTED")
        state_enum = CrossingState(st_val) if st_val in [s.value for s in CrossingState] else CrossingState.COUNT_COMMITTED
        return cls(
            event_id=data["event_id"],
            global_carton_id=data["global_carton_id"],
            timestamp_sec=float(data["timestamp_sec"]),
            direction=data["direction"],
            carton_count=int(data.get("carton_count", 1)),
            participating_cameras=list(data.get("participating_cameras", [])),
            state=state_enum,
            is_ambiguous=bool(data.get("is_ambiguous", False)),
            notes=str(data.get("notes", "")),
        )



@dataclass
class PhysicalCarton:
    """
    Represents a unique physical carton entity tracked across time and cameras.
    """

    carton_id: str
    state: CrossingState = CrossingState.NOT_SEEN
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    initial_zone: str = "before_line"
    current_zone: str = "before_line"

    # Associated tracklets across cameras: camera_id -> set of local_track_ids
    associated_tracks: Dict[str, Set[str]] = field(default_factory=dict)

    # Spatial trajectory: list of (timestamp, x, y)
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    last_center: Tuple[float, float] = (0.0, 0.0)
    last_camera_centers: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    last_dock_xy: Optional[Tuple[float, float]] = None
    displacement: float = 0.0

    # Multi-camera evidence
    contributing_cameras: Set[str] = field(default_factory=set)
    confidence_history: List[float] = field(default_factory=list)

    # Commit state
    is_committed: bool = False
    committed_event_id: Optional[str] = None
    committed_timestamp: Optional[float] = None
    resolved_count: int = 1
    last_committed_direction: Optional[str] = None
    committed_directions: List[str] = field(default_factory=list)

    # Ambiguity / Review flags
    is_ambiguous: bool = False
    review_required: bool = False
    notes: str = ""


@dataclass
class CartonCrossingEntity:
    """Legacy state tracking for a single carton crossing candidate."""

    global_id: str
    current_state: CrossingState = CrossingState.UNCONFIRMED
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    initial_side: Optional[str] = None  # "before_line" or "after_line"
    current_side: Optional[str] = None
    crossing_timestamp: Optional[float] = None
    detected_direction: Optional[str] = None
    is_counted: bool = False
    committed_event_id: Optional[str] = None
    observation_cameras: Set[str] = field(default_factory=set)


class CartonCrossingStateMachine:
    """
    Deterministic Red-Line Crossing State Machine & Physical Carton Tracker.
    
    Manages transitions:
      NOT_SEEN -> APPROACHING -> CROSSING -> CROSSED -> COUNT_COMMITTED
      (Also supports directional tracking: total_in, total_out, and net_count)
    """

    def __init__(
        self,
        line_orientation: str = "vertical",
        line_position_px: float = 500.0,
        hysteresis_margin_px: float = 25.0,
        required_direction: str = "IN",
        min_displacement_px: float = 20.0,
        max_reid_spatial_dist_px: float = 140.0,
        max_occlusion_coasting_sec: float = 2.2,
    ) -> None:
        self.orientation = str(line_orientation)
        self.line_pos = float(line_position_px)
        self.hysteresis = float(hysteresis_margin_px)
        self.required_direction = str(required_direction)
        self.min_displacement = float(min_displacement_px)
        self.max_reid_spatial_dist = float(max_reid_spatial_dist_px)
        self.max_occlusion_coasting_sec = float(max_occlusion_coasting_sec)

        # Legacy entity store
        self._entities: Dict[str, CartonCrossingEntity] = {}

        # Physical carton instances
        self._physical_cartons: Dict[str, PhysicalCarton] = {}
        self._next_carton_idx: int = 1

        # Transactional committed events registry
        self._committed_events: List[CommittedCrossingEvent] = []
        self._committed_event_ids: Set[str] = set()
        self._next_event_idx: int = 1

    @property
    def total_in(self) -> int:
        """Total unique cartons confirmed loaded into the truck (+1)."""
        return sum(
            e.carton_count
            for e in self._committed_events
            if e.direction == "IN" and e.state in (CrossingState.COUNTED, CrossingState.COUNT_COMMITTED)
        )

    @property
    def total_out(self) -> int:
        """Total unique cartons confirmed returning toward pallet / unloaded (-1)."""
        return sum(
            e.carton_count
            for e in self._committed_events
            if e.direction == "OUT" and e.state in (CrossingState.COUNTED, CrossingState.COUNT_COMMITTED)
        )

    @property
    def net_count(self) -> int:
        """Net cartons loaded inside truck (total_in - total_out)."""
        return self.total_in - self.total_out

    @property
    def total_counted(self) -> int:
        return sum(
            e.carton_count
            for e in self._committed_events
            if e.state in (CrossingState.COUNTED, CrossingState.COUNT_COMMITTED)
        )

    @property
    def confirmed_count(self) -> int:
        return sum(
            e.carton_count
            for e in self._committed_events
            if e.state in (CrossingState.COUNTED, CrossingState.COUNT_COMMITTED)
        )

    @property
    def confirmed_in(self) -> int:
        return self.total_in

    @property
    def confirmed_out(self) -> int:
        return self.total_out

    @property
    def review_required_count(self) -> int:
        return sum(
            e.carton_count
            for e in self._committed_events
            if e.state == CrossingState.REVIEW_REQUIRED
        )

    @property
    def review_required_in(self) -> int:
        return sum(
            e.carton_count
            for e in self._committed_events
            if e.direction == "IN" and e.state == CrossingState.REVIEW_REQUIRED
        )

    @property
    def review_required_out(self) -> int:
        return sum(
            e.carton_count
            for e in self._committed_events
            if e.direction == "OUT" and e.state == CrossingState.REVIEW_REQUIRED
        )

    @property
    def provisional_count(self) -> int:
        return self.confirmed_count + self.review_required_count

    @property
    def provisional_in(self) -> int:
        return self.confirmed_in + self.review_required_in

    @property
    def provisional_out(self) -> int:
        return self.confirmed_out + self.review_required_out

    @property
    def committed_events(self) -> List[CommittedCrossingEvent]:
        return list(self._committed_events)

    def determine_side(self, coord: float) -> Optional[str]:
        """Classifies coordinate relative to hysteresis boundary."""
        if coord <= (self.line_pos - self.hysteresis):
            return "before_line"
        elif coord >= (self.line_pos + self.hysteresis):
            return "after_line"
        return None  # In hysteresis buffer zone

    def update_carton_position(
        self,
        global_carton_id: str,
        coord: float,
        displacement: float,
        timestamp_sec: float,
        contributing_cameras: List[str],
        is_ambiguous: bool = False,
    ) -> Optional[CommittedCrossingEvent]:
        """
        Legacy direct-coordinate update method for carton tripwire crossings.
        Maintains backward compatibility with unit tests.
        """
        entity = self._entities.get(global_carton_id)
        if entity is None:
            entity = CartonCrossingEntity(
                global_id=global_carton_id,
                current_state=CrossingState.UNCONFIRMED,
                first_seen_time=timestamp_sec,
                last_seen_time=timestamp_sec,
            )
            self._entities[global_carton_id] = entity

        entity.last_seen_time = timestamp_sec
        entity.observation_cameras.update(contributing_cameras)

        curr_side = self.determine_side(coord)
        if curr_side is None:
            return None  # In indeterminate hysteresis band

        # 1. State: UNCONFIRMED / NOT_SEEN -> OBSERVED / APPROACHING
        if entity.current_state in (CrossingState.UNCONFIRMED, CrossingState.NOT_SEEN):
            entity.initial_side = curr_side
            entity.current_side = curr_side
            entity.current_state = CrossingState.OBSERVED
            return None

        # 2. Check for side change
        if entity.current_side is not None and curr_side != entity.current_side:
            if entity.initial_side == "before_line" and curr_side == "after_line":
                crossing_dir = "IN"
            elif entity.initial_side == "after_line" and curr_side == "before_line":
                crossing_dir = "OUT"
            else:
                crossing_dir = "UNKNOWN"

            entity.current_side = curr_side
            entity.detected_direction = crossing_dir
            entity.crossing_timestamp = timestamp_sec

            # Validate displacement
            if displacement < self.min_displacement:
                entity.current_state = CrossingState.UNCERTAIN
                return None

            # Check if this matches required counting direction
            if crossing_dir == self.required_direction:
                entity.current_state = CrossingState.CROSSING_CONFIRMED

                # EXACTLY-ONCE GUARANTEE: Check if already counted
                if not entity.is_counted:
                    event_id = f"event_{self._next_event_idx:05d}"
                    self._next_event_idx += 1

                    state_val = (
                        CrossingState.COUNTED if not is_ambiguous else CrossingState.REVIEW_REQUIRED
                    )
                    event = CommittedCrossingEvent(
                        event_id=event_id,
                        global_carton_id=global_carton_id,
                        timestamp_sec=timestamp_sec,
                        direction=crossing_dir,
                        carton_count=1,
                        participating_cameras=sorted(list(entity.observation_cameras)),
                        state=state_val,
                        is_ambiguous=is_ambiguous,
                        notes=(
                            "Normal crossing commit"
                            if not is_ambiguous
                            else "Flagged for review due to cluster ambiguity"
                        ),
                    )
                    entity.is_counted = True
                    entity.committed_event_id = event_id
                    entity.current_state = state_val
                    self._committed_events.append(event)
                    self._committed_event_ids.add(event_id)
                    return event
            else:
                entity.current_state = CrossingState.REVERSED_CANCELLED

        return None

    # -----------------------------------------------------------------------
    # Advanced Physical Carton Management (Phases 3, 4, 5)
    # -----------------------------------------------------------------------
    def find_matching_physical_carton(
        self,
        camera_id: str,
        local_track_id: str,
        center: Tuple[float, float],
        timestamp_sec: float,
        dock_xy: Optional[Tuple[float, float]] = None,
        direction: Optional[str] = None,
        return_outcome: bool = False,
    ) -> Any:
        """
        Matches an observation to an existing physical carton using:
        1. Direct tracklet association (exact match on camera and local ID)
        2. Per-camera spatial continuity if previously seen in this camera
        3. Multi-camera corridor transit temporal continuity for active/committed crossings
        """
        # 1. Exact tracklet association
        for phys in self._physical_cartons.values():
            if local_track_id in phys.associated_tracks.get(camera_id, set()):
                if return_outcome:
                    return phys, PhysicalAssociationOutcome.MATCHED_EXISTING_PHYSICAL_CARTON, "Exact local track ID match"
                return phys

        # 2. Spatio-temporal continuity
        candidates = []
        for phys in self._physical_cartons.values():
            dt = timestamp_sec - phys.last_seen_time
            if dt < -0.1 or dt > self.max_occlusion_coasting_sec:
                continue

            # Prioritize calibrated dock coordinates if available
            if dock_xy is not None and phys.last_dock_xy is not None:
                dist = math.hypot(dock_xy[0] - phys.last_dock_xy[0], dock_xy[1] - phys.last_dock_xy[1])
                if dist <= self.max_reid_spatial_dist:
                    candidates.append((dist, dt, phys))
            elif camera_id in phys.last_camera_centers:
                # Same camera: compare pixel distance within this camera's frame
                prev_c = phys.last_camera_centers[camera_id]
                y_offset = abs(center[1] - prev_c[1])
                dist = math.hypot(center[0] - prev_c[0], center[1] - prev_c[1])
                # Enforce lateral/height lane separation (<= 60px) to prevent merging parallel workers
                if y_offset <= 60.0 and dist <= self.max_reid_spatial_dist:
                    candidates.append((dist, dt, phys))

        if candidates:
            # Sort by distance first, then dt
            candidates.sort(key=lambda c: (c[0], c[1]))
            # Ambiguity check: if two physical cartons are equidistant (within 15px)
            if len(candidates) >= 2 and abs(candidates[0][0] - candidates[1][0]) < 15.0:
                if return_outcome:
                    return candidates[0][2], PhysicalAssociationOutcome.AMBIGUOUS_ASSOCIATION, "Multiple equidistant physical carton candidates"
            best_phys = candidates[0][2]
            if return_outcome:
                return best_phys, PhysicalAssociationOutcome.TEMPORARY_OCCLUSION, "Spatio-temporal re-id across occlusion"
            return best_phys

        # 3. Active committed crossing corridor protection:
        # Protects the same physical crossing from duplicate multi-camera echoes and fragmented tracklet echoes
        # within the physical transit window (2.0s), while strictly preserving separation of distinct cartons.
        for phys in self._physical_cartons.values():
            if phys.is_committed and phys.committed_timestamp is not None:
                transit_dt = timestamp_sec - phys.committed_timestamp
                if 0.0 <= transit_dt <= 1.2:
                    if direction is not None and phys.last_committed_direction != direction:
                        continue

                    # Same camera tracklet continuation check:
                    if camera_id in phys.associated_tracks and local_track_id not in phys.associated_tracks[camera_id]:
                        if camera_id in phys.last_camera_centers:
                            prev_c = phys.last_camera_centers[camera_id]
                            y_offset = abs(center[1] - prev_c[1])
                            # If vertical lane offset > 60px, distinct lane/worker (NEVER merge!)
                            if y_offset > 60.0:
                                continue
                            dist = math.hypot(center[0] - prev_c[0], center[1] - prev_c[1])
                            if dist > self.max_reid_spatial_dist:
                                continue
                        else:
                            continue

                    # Spatial proximity check within frame
                    if camera_id in phys.last_camera_centers:
                        prev_c = phys.last_camera_centers[camera_id]
                        y_diff = abs(center[1] - prev_c[1])
                        if y_diff > 60.0:
                            continue
                        dist = math.hypot(center[0] - prev_c[0], center[1] - prev_c[1])
                        if dist > self.max_reid_spatial_dist:
                            continue

                    # Global / cross-camera spatial separation check:
                    if dock_xy is not None and phys.last_dock_xy is not None:
                        d_dist = math.hypot(dock_xy[0] - phys.last_dock_xy[0], dock_xy[1] - phys.last_dock_xy[1])
                        if d_dist > self.max_reid_spatial_dist:
                            continue
                    elif phys.last_center is not None:
                        # If cross-camera, check vertical separation (different lanes)
                        y_diff = abs(center[1] - phys.last_center[1])
                        if y_diff > 120.0:
                            continue

                    if return_outcome:
                        return phys, PhysicalAssociationOutcome.MATCHED_EXISTING_PHYSICAL_CARTON, "Corridor transit continuation"
                    return phys

        if return_outcome:
            return None, PhysicalAssociationOutcome.NEW_PHYSICAL_CARTON, "No existing physical carton matched"
        return None

    def get_or_create_physical_carton(
        self,
        camera_id: str,
        local_track_id: str,
        center: Tuple[float, float],
        timestamp_sec: float,
        initial_side: str = "before_line",
        dock_xy: Optional[Tuple[float, float]] = None,
        direction: Optional[str] = None,
        allow_spatiotemporal_match: bool = True,
    ) -> PhysicalCarton:
        """Retrieves existing matching physical carton or creates a new one."""
        if allow_spatiotemporal_match:
            matched = self.find_matching_physical_carton(
                camera_id=camera_id,
                local_track_id=local_track_id,
                center=center,
                timestamp_sec=timestamp_sec,
                dock_xy=dock_xy,
                direction=direction,
            )
        else:
            matched = next(
                (
                    phys
                    for phys in self._physical_cartons.values()
                    if local_track_id in phys.associated_tracks.get(camera_id, set())
                ),
                None,
            )
        if matched is not None:
            # Register local tracklet in matched physical carton
            matched.associated_tracks.setdefault(camera_id, set()).add(local_track_id)
            matched.contributing_cameras.add(camera_id)
            matched.last_seen_time = max(matched.last_seen_time, timestamp_sec)
            matched.last_center = center
            matched.last_camera_centers[camera_id] = center
            if dock_xy is not None:
                matched.last_dock_xy = dock_xy
            return matched

        # Create new PhysicalCarton
        cid = f"phys_carton_{self._next_carton_idx:05d}"
        self._next_carton_idx += 1

        state = CrossingState.APPROACHING if initial_side == "before_line" else CrossingState.CROSSING
        phys = PhysicalCarton(
            carton_id=cid,
            state=state,
            first_seen_time=timestamp_sec,
            last_seen_time=timestamp_sec,
            initial_zone=initial_side,
            current_zone=initial_side,
            associated_tracks={camera_id: {local_track_id}},
            last_center=center,
            last_camera_centers={camera_id: center},
            last_dock_xy=dock_xy,
            contributing_cameras={camera_id},
        )
        self._physical_cartons[cid] = phys
        return phys

    def register_crossing_candidate(
        self,
        candidate: Any,
        timestamp_sec: float,
        consensus_verdict: Optional[Any] = None,
        stack_quantity: int = 1,
        is_ambiguous: bool = False,
    ) -> Optional[CommittedCrossingEvent]:
        """
        Processes a LocalCameraCrossingCandidate with deterministic state transitions.
        Guarantees EXACTLY-ONCE count per unique physical carton.
        """
        center = (
            (candidate.bbox[0] + candidate.bbox[2]) / 2.0,
            (candidate.bbox[1] + candidate.bbox[3]) / 2.0,
        )

        phys = self.get_or_create_physical_carton(
            camera_id=candidate.camera_id,
            local_track_id=candidate.local_track_id,
            center=center,
            timestamp_sec=timestamp_sec,
            initial_side="before_line",
        )

        phys.contributing_cameras.add(candidate.camera_id)
        phys.confidence_history.append(candidate.confidence)
        phys.last_seen_time = max(phys.last_seen_time, timestamp_sec)
        phys.displacement = max(phys.displacement, getattr(candidate, "displacement", 0.0))

        # Check direction policy
        direction = getattr(candidate, "direction", "IN")
        if self.required_direction not in ("BOTH", "BIDIRECTIONAL") and direction != self.required_direction:
            phys.state = CrossingState.REVERSED_CANCELLED
            return None

        # Transition: APPROACHING -> CROSSING -> CROSSED -> COUNT_COMMITTED
        phys.state = CrossingState.CROSSED

        # EXACTLY-ONCE CHECK:
        # If this physical carton was already committed in this exact direction, absorb candidate as corroboration.
        if phys.is_committed and phys.last_committed_direction == direction:
            # Update participating cameras on existing committed event
            if phys.committed_event_id:
                for ev in self._committed_events:
                    if ev.event_id == phys.committed_event_id:
                        if candidate.camera_id not in ev.participating_cameras:
                            ev.participating_cameras.append(candidate.camera_id)
                            ev.participating_cameras.sort()
                        # If previously review-required due to single view and now corroborated by multi-view
                        if ev.state == CrossingState.REVIEW_REQUIRED and len(ev.participating_cameras) >= 2:
                            ev.state = CrossingState.COUNT_COMMITTED
                            ev.is_ambiguous = False
                            ev.notes = "Upgraded to multi-camera confirmation on corroboration"
            return None

        # Determine consensus action
        is_auto_confirmed = True
        notes = "Multi-camera confirmed crossing"
        if consensus_verdict is not None:
            # Check decision action
            action_name = getattr(consensus_verdict, "action", None)
            action_str = getattr(action_name, "value", str(action_name)) if action_name else "COMMIT"
            is_auto_confirmed = (action_str == "COMMIT" and not is_ambiguous)
            notes = getattr(consensus_verdict, "decision_reason", notes)
        elif len(phys.contributing_cameras) < 2:
            is_auto_confirmed = False
            notes = f"Single camera observation on '{candidate.camera_id}' held for review"

        state_val = CrossingState.COUNT_COMMITTED if is_auto_confirmed else CrossingState.REVIEW_REQUIRED
        event_id = f"fused_ev_{self._next_event_idx:05d}"
        self._next_event_idx += 1

        resolved_qty = max(1, stack_quantity)
        ev = CommittedCrossingEvent(
            event_id=event_id,
            global_carton_id=phys.carton_id,
            timestamp_sec=timestamp_sec,
            direction=direction,
            carton_count=resolved_qty,
            participating_cameras=sorted(list(phys.contributing_cameras)),
            state=state_val,
            is_ambiguous=(not is_auto_confirmed or is_ambiguous),
            notes=notes,
        )

        phys.is_committed = True
        phys.last_committed_direction = direction
        phys.committed_directions.append(direction)
        phys.committed_event_id = event_id
        phys.committed_timestamp = timestamp_sec
        phys.state = state_val
        phys.resolved_count = resolved_qty

        self._committed_events.append(ev)
        self._committed_event_ids.add(event_id)
        return ev

    def export_state(self, file_path: str) -> None:
        """Serializes current crossing state machine state to a persistent JSON file."""
        import json
        import os

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        data = {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "net_count": self.net_count,
            "confirmed_count": self.confirmed_count,
            "review_required_count": self.review_required_count,
            "committed_event_ids": sorted(list(self._committed_event_ids)),
            "committed_events": [e.to_dict() for e in self._committed_events],
            "physical_cartons": {
                cid: {
                    "carton_id": p.carton_id,
                    "state": p.state.value if hasattr(p.state, "value") else str(p.state),
                    "first_seen_time": p.first_seen_time,
                    "last_seen_time": p.last_seen_time,
                    "is_committed": p.is_committed,
                    "committed_event_id": p.committed_event_id,
                    "last_committed_direction": p.last_committed_direction,
                    "resolved_count": p.resolved_count,
                    "contributing_cameras": sorted(list(p.contributing_cameras)),
                    "associated_tracks": {k: sorted(list(v)) for k, v in p.associated_tracks.items()},
                    "last_center": list(p.last_center),
                }
                for cid, p in self._physical_cartons.items()
            },
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_state(self, file_path: str) -> None:
        """Restores state from a persistent JSON ledger for crash-recovery and restart idempotency."""
        import json
        import os

        if not os.path.exists(file_path):
            return

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._committed_event_ids = set(data.get("committed_event_ids", []))
        self._committed_events = []
        for e_dict in data.get("committed_events", []):
            ev = CommittedCrossingEvent.from_dict(e_dict)
            self._committed_events.append(ev)

        self._physical_cartons = {}
        for cid, p_dict in data.get("physical_cartons", {}).items():
            st_val = p_dict.get("state", "COUNT_COMMITTED")
            state_enum = CrossingState(st_val) if st_val in [s.value for s in CrossingState] else CrossingState.COUNT_COMMITTED
            p = PhysicalCarton(
                carton_id=cid,
                state=state_enum,
                first_seen_time=p_dict.get("first_seen_time", 0.0),
                last_seen_time=p_dict.get("last_seen_time", 0.0),
                is_committed=p_dict.get("is_committed", True),
                committed_event_id=p_dict.get("committed_event_id"),
                last_committed_direction=p_dict.get("last_committed_direction"),
                resolved_count=p_dict.get("resolved_count", 1),
                contributing_cameras=set(p_dict.get("contributing_cameras", [])),
                associated_tracks={k: set(v) for k, v in p_dict.get("associated_tracks", {}).items()},
                last_center=tuple(p_dict.get("last_center", [0.0, 0.0])),
            )
            self._physical_cartons[cid] = p
