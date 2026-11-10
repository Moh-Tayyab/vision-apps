"""
Crossing-Corridor Zones and Spatial Tracking Evidence (Phase 2 Requirement 4).

Models the 2D spatial loading corridor at the doorway threshold:
  1. APPROACH_ZONE: Pre-crossing staging area where cartons approach the doorway.
  2. THRESHOLD_ZONE: The critical doorway boundary zone.
  3. COMMIT_ZONE: The loaded area inside the truck.

Collects per-camera corridor transit evidence:
- Entry, threshold, and exit timestamps.
- Transit duration across the corridor.
- Velocity vector alignment with the doorway normal vector.
- Validates genuine crossing vs boundary hovering / reversals.
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


class CorridorZone(str, enum.Enum):
    """
    Corridor Zones for Volumetric Tracking (Week 3 Architecture).
    
    1. ENTRY_ZONE    -> Worker+cartons cross dock threshold (approach)
    2. TRANSIT_ZONE  -> Bundle active; associate detections across cam_top/side/flank
    3. EXIT_ZONE     -> Bundle enters truck interior -> fire ONE commit event
    """
    ENTRY_ZONE = "ENTRY_ZONE"
    TRANSIT_ZONE = "TRANSIT_ZONE"
    EXIT_ZONE = "EXIT_ZONE"

    # Backward-compatible aliases for Phase 2 tests
    OUTSIDE_APPROACH = "OUTSIDE_APPROACH"
    THRESHOLD_BAND = "THRESHOLD_BAND"
    INSIDE_COMMIT = "INSIDE_COMMIT"

    def __eq__(self, other: Any) -> bool:
        if super().__eq__(other):
            return True
        pairs = {
            "ENTRY_ZONE": "OUTSIDE_APPROACH",
            "OUTSIDE_APPROACH": "ENTRY_ZONE",
            "TRANSIT_ZONE": "THRESHOLD_BAND",
            "THRESHOLD_BAND": "TRANSIT_ZONE",
            "EXIT_ZONE": "INSIDE_COMMIT",
            "INSIDE_COMMIT": "EXIT_ZONE",
        }
        if isinstance(other, CorridorZone):
            return pairs.get(self.value) == other.value
        if isinstance(other, str):
            return self.value == other or pairs.get(self.value) == other
        return False

    def __hash__(self) -> int:
        canon = {
            "OUTSIDE_APPROACH": "ENTRY_ZONE",
            "THRESHOLD_BAND": "TRANSIT_ZONE",
            "INSIDE_COMMIT": "EXIT_ZONE",
        }
        return hash(canon.get(self.value, self.value))

    @property
    def is_entry(self) -> bool:
        return self in (CorridorZone.ENTRY_ZONE, CorridorZone.OUTSIDE_APPROACH)

    @property
    def is_transit(self) -> bool:
        return self in (CorridorZone.TRANSIT_ZONE, CorridorZone.THRESHOLD_BAND)

    @property
    def is_exit(self) -> bool:
        return self in (CorridorZone.EXIT_ZONE, CorridorZone.INSIDE_COMMIT)


class BundleState(str, enum.Enum):
    """Lifecycle states for an active Transit Bundle in the loading corridor."""
    ENTERED = "ENTERED"                    # Worker+cartons entered dock threshold
    IN_TRANSIT = "IN_TRANSIT"              # Bundle active in corridor body
    COMMITTED = "COMMITTED"                # Exited corridor into truck; exactly 1 commit fired
    REVERSED_CANCELLED = "REVERSED_CANCELLED" # Worker retreated back to dock; cancelled with 0 count


@dataclass
class TransitBundle:
    """
    A unified multi-view transit bundle representing a worker and cartons in the loading corridor (Week 3).
    
    Attributes:
    - bundle_id: Unique bundle identifier.
    - worker_id: Identifier of the carrying worker (if detected/associated).
    - carton_bundle: Structured metadata of cartons being transported (stack counts, bboxes).
    - start_t: Timestamp when bundle entered the corridor threshold.
    - last_t: Timestamp of the most recent observation.
    - direction: Progression direction ("IN" or "OUT").
    - state: Current lifecycle state (ENTERED, IN_TRANSIT, COMMITTED, REVERSED_CANCELLED).
    - associated_tracks: Camera-to-tracklet IDs mapping.
    - view_carton_counts: Per-camera carton quantity estimates.
    """
    bundle_id: str
    worker_id: Optional[str] = None
    start_t: float = 0.0
    last_t: float = 0.0
    direction: str = "IN"
    state: BundleState = BundleState.ENTERED

    # Multi-camera associated tracklets: camera_id -> Set of track_ids
    associated_tracks: Dict[str, Set[str]] = field(default_factory=dict)
    camera_bboxes: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    camera_centers: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    view_carton_counts: Dict[str, int] = field(default_factory=dict)
    confidences: List[float] = field(default_factory=list)
    carton_bundle: Dict[str, Any] = field(default_factory=dict)

    # Zone history: camera_id -> list of (timestamp_sec, CorridorZone)
    zone_history: Dict[str, List[Tuple[float, CorridorZone]]] = field(default_factory=dict)

    # Resolution and commit details
    resolved_count: int = 1
    is_committed: bool = False
    committed_event_id: Optional[str] = None
    commit_t: Optional[float] = None
    reversal_detected: bool = False
    hesitation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "worker_id": self.worker_id,
            "start_t": round(self.start_t, 3),
            "last_t": round(self.last_t, 3),
            "direction": self.direction,
            "state": self.state.value,
            "associated_tracks": {k: sorted(list(v)) for k, v in self.associated_tracks.items()},
            "contributing_cameras": sorted(list(self.associated_tracks.keys())),
            "view_carton_counts": dict(self.view_carton_counts),
            "resolved_count": self.resolved_count,
            "is_committed": self.is_committed,
            "committed_event_id": self.committed_event_id,
            "reversal_detected": self.reversal_detected,
        }


@dataclass
class PerCameraCorridorEvidence:
    """Audit evidence for a single carton tracklet traversing the corridor."""

    camera_id: str
    track_id: str
    initial_zone: CorridorZone
    current_zone: CorridorZone
    entry_timestamp: float
    threshold_timestamp: Optional[float] = None
    commit_timestamp: Optional[float] = None
    transit_duration_sec: float = 0.0
    velocity_alignment: float = 0.0  # Cosine similarity to doorway vector [ -1.0 to +1.0 ]
    is_valid_transit: bool = False
    reversal_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "initial_zone": self.initial_zone.value,
            "current_zone": self.current_zone.value,
            "entry_timestamp": round(self.entry_timestamp, 3),
            "threshold_timestamp": round(self.threshold_timestamp, 3) if self.threshold_timestamp else None,
            "commit_timestamp": round(self.commit_timestamp, 3) if self.commit_timestamp else None,
            "transit_duration_sec": round(self.transit_duration_sec, 3),
            "velocity_alignment": round(self.velocity_alignment, 3),
            "is_valid_transit": self.is_valid_transit,
            "reversal_detected": self.reversal_detected,
        }


class CrossingCorridor:
    """
    Manages spatial corridor zones for a single camera perspective (Week 3).
    
    Provides 3-zone spatial state evaluation:
    - ENTRY_ZONE   (Dock Threshold): Worker + cartons approach doorway.
    - TRANSIT_ZONE (Corridor Body): Active multi-camera association tunnel.
    - EXIT_ZONE    (Truck Interior): Bundle enters destination; ready to commit.
    """

    def __init__(
        self,
        camera_id: str,
        line_position_px: float = 500.0,
        threshold_half_width_px: float = 30.0,
        normal_direction: Tuple[float, float] = (1.0, 0.0),  # (dx, dy) towards truck
        min_transit_duration_sec: float = 0.15,
        max_transit_duration_sec: float = 6.0,
        orientation: str = "vertical",
        direction_sign: int = 1,
        line_pos_frac: Optional[float] = None,
        transit_width_frac: float = 0.08,
    ) -> None:
        self.camera_id = str(camera_id)
        self.line_x = float(line_position_px)
        self.half_width = float(threshold_half_width_px)
        self.normal = normal_direction
        self.min_duration = float(min_transit_duration_sec)
        self.max_duration = float(max_transit_duration_sec)
        self.orientation = str(orientation).lower()
        self.direction_sign = int(direction_sign)
        self.line_pos_frac = float(line_pos_frac) if line_pos_frac is not None else None
        self.transit_width_frac = float(transit_width_frac)

        self._track_evidence: Dict[str, PerCameraCorridorEvidence] = {}

    def classify_zone(
        self,
        point_xy: Tuple[float, float],
        frame_dims: Optional[Tuple[int, int]] = None,
    ) -> CorridorZone:
        """Determines which corridor zone a point occupies."""
        if frame_dims is not None:
            h, w = frame_dims
            dim = w if self.orientation == "vertical" else h
            coord = point_xy[0] if self.orientation == "vertical" else point_xy[1]
            line_px = (self.line_pos_frac * dim) if self.line_pos_frac is not None else self.line_x
            hw = self.transit_width_frac * dim if self.transit_width_frac is not None else self.half_width
        else:
            coord = point_xy[0] if self.orientation == "vertical" else point_xy[1]
            line_px = self.line_x
            hw = self.half_width

        # Direction-aware zone bounds:
        if self.direction_sign > 0:
            # Positive direction: increasing coordinate moves into truck
            if coord < (line_px - hw):
                return CorridorZone.ENTRY_ZONE
            elif coord > (line_px + hw):
                return CorridorZone.EXIT_ZONE
            else:
                return CorridorZone.TRANSIT_ZONE
        else:
            # Negative direction: decreasing coordinate moves into truck (dock is right, truck is left)
            if coord > (line_px + hw):
                return CorridorZone.ENTRY_ZONE
            elif coord < (line_px - hw):
                return CorridorZone.EXIT_ZONE
            else:
                return CorridorZone.TRANSIT_ZONE

    def update_track_evidence(
        self,
        track_id: str,
        point_xy: Tuple[float, float],
        velocity_xy: Tuple[float, float],
        timestamp_sec: float,
        frame_dims: Optional[Tuple[int, int]] = None,
    ) -> PerCameraCorridorEvidence:
        """
        Updates corridor progression for a tracked carton.
        """
        curr_zone = self.classify_zone(point_xy, frame_dims=frame_dims)
        evidence = self._track_evidence.get(track_id)

        # Compute velocity vector alignment (cosine similarity with normal)
        v_mag = math.hypot(velocity_xy[0], velocity_xy[1])
        n_mag = math.hypot(self.normal[0], self.normal[1])
        if v_mag > 1e-3 and n_mag > 1e-3:
            cos_sim = (velocity_xy[0] * self.normal[0] + velocity_xy[1] * self.normal[1]) / (v_mag * n_mag)
        else:
            cos_sim = 0.0

        if evidence is None:
            evidence = PerCameraCorridorEvidence(
                camera_id=self.camera_id,
                track_id=track_id,
                initial_zone=curr_zone,
                current_zone=curr_zone,
                entry_timestamp=timestamp_sec,
                velocity_alignment=cos_sim,
            )
            self._track_evidence[track_id] = evidence
            return evidence

        evidence.current_zone = curr_zone
        evidence.velocity_alignment = cos_sim

        # Progression: check if entering THRESHOLD_BAND / TRANSIT_ZONE
        if curr_zone.is_transit and evidence.threshold_timestamp is None:
            evidence.threshold_timestamp = timestamp_sec

        # Progression: check if entering INSIDE_COMMIT / EXIT_ZONE
        if curr_zone.is_exit and evidence.commit_timestamp is None:
            if evidence.initial_zone.is_entry or evidence.initial_zone.is_transit:
                evidence.commit_timestamp = timestamp_sec
                evidence.transit_duration_sec = timestamp_sec - evidence.entry_timestamp

                # Validate genuine transit
                if self.min_duration <= evidence.transit_duration_sec <= self.max_duration and cos_sim > 0.1:
                    evidence.is_valid_transit = True

        # Check for direction reversal (was in threshold or commit, but moved back to approach)
        if evidence.threshold_timestamp is not None and curr_zone.is_entry:
            evidence.reversal_detected = True
            evidence.is_valid_transit = False

        return evidence

    def get_evidence(self, track_id: str) -> Optional[PerCameraCorridorEvidence]:
        return self._track_evidence.get(track_id)


class CorridorStateMachine:
    """
    Multi-Camera Corridor State Machine (Week 3 Architecture).
    
    Replaces 1D tripwires with volumetric 3-zone corridors:
    - Entry Zone    -> Worker + cartons cross dock threshold (initiates/activates bundle)
    - Transit Zone  -> Bundle active; associate detections across cam_top/side/flank
    - Exit Zone     -> Bundle enters truck interior -> fire EXACTLY ONE commit event
    
    Guarantees:
    1. Holds worker_id, carton_bundle, start_t, direction.
    2. Hysteresis: if worker steps back, pauses, or lingers inside the transit corridor,
       DO NOT increment/decrement until bundle fully exits into the truck interior.
    3. If worker retreats back through Entry Zone, cancel bundle (REVERSED_CANCELLED) with 0 count change.
    4. Coalesces per-camera tracklets into a single Transit Bundle using TimelineAligner 6s window.
    5. Exactly-Once: Fires ONE commit event per physical bundle traversal.
    """

    def __init__(
        self,
        corridor_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        camera_ids: Sequence[str] = ("cam_top", "cam_side", "cam_flank"),
        required_direction: str = "IN",
        coalesce_window_sec: float = 6.0,
    ) -> None:
        self.camera_ids = tuple(camera_ids)
        self.required_direction = str(required_direction)
        self.window_sec = float(coalesce_window_sec)

        # Build per-camera CrossingCorridor instances
        cfgs = corridor_configs or {
            "cam_side": {"orientation": "vertical", "line_pos_frac": 0.50, "direction_sign": -1},
            "cam_top": {"orientation": "vertical", "line_pos_frac": 0.50, "direction_sign": -1},
            "cam_flank": {"orientation": "vertical", "line_pos_frac": 0.55, "direction_sign": 1},
        }

        self.corridors: Dict[str, CrossingCorridor] = {}
        for cid in self.camera_ids:
            c_cfg = cfgs.get(cid, {})
            self.corridors[cid] = CrossingCorridor(
                camera_id=cid,
                line_position_px=float(c_cfg.get("line_position_px", 500.0)),
                threshold_half_width_px=float(c_cfg.get("threshold_half_width_px", 35.0)),
                orientation=str(c_cfg.get("orientation", "vertical")),
                direction_sign=int(c_cfg.get("direction_sign", -1 if cid in ("cam_top", "cam_side") else 1)),
                line_pos_frac=float(c_cfg.get("line_pos_frac", 0.50)),
                transit_width_frac=float(c_cfg.get("transit_width_frac", 0.08)),
            )

        self._active_bundles: List[TransitBundle] = []
        self._committed_bundles: List[TransitBundle] = []
        self._committed_events: List[Dict[str, Any]] = []
        self._next_bundle_id: int = 1
        self._total_in: int = 0
        self._total_out: int = 0

    @property
    def total_in(self) -> int:
        return self._total_in

    @property
    def total_out(self) -> int:
        return self._total_out

    @property
    def net_count(self) -> int:
        return self._total_in - self._total_out

    @property
    def active_bundles(self) -> List[TransitBundle]:
        return [b for b in self._active_bundles if b.state in (BundleState.ENTERED, BundleState.IN_TRANSIT)]

    @property
    def committed_bundles(self) -> List[TransitBundle]:
        return list(self._committed_bundles)

    @property
    def committed_events(self) -> List[Dict[str, Any]]:
        return list(self._committed_events)

    def update(
        self,
        camera_tracklets: Dict[str, List[Any]],
        timestamp_sec: float,
        frame_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        disambiguator: Optional[Any] = None,
        consensus_policy: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes one time tick of the Multi-Camera Corridor State Machine.
        """
        f_dims = frame_dims or {}
        newly_committed: List[Dict[str, Any]] = []

        # Step 1: Evaluate per-camera tracklet zones
        observed_tracklets: List[Tuple[str, Any, CorridorZone]] = []
        for cid in self.camera_ids:
            dims = f_dims.get(cid, (576, 1024))
            corridor = self.corridors.get(cid)
            if corridor is None:
                continue
            for trk in camera_tracklets.get(cid, []):
                zone = corridor.classify_zone(trk.center, frame_dims=dims)
                observed_tracklets.append((cid, trk, zone))

        # Step 2: Coalesce tracklets into Transit Bundles (using TimelineAligner 6s window)
        for cid, trk, zone in observed_tracklets:
            matched_bundle: Optional[TransitBundle] = None

            # 2a. Check active uncommitted bundles
            for b in self._active_bundles:
                if b.state in (BundleState.COMMITTED, BundleState.REVERSED_CANCELLED):
                    continue
                # Temporal proximity check (within 6s window)
                if abs(timestamp_sec - b.last_t) <= self.window_sec:
                    # Spatial association check
                    if cid in b.associated_tracks:
                        # Same camera: check if track matches or within bundle spatial envelope (220px)
                        if trk.local_track_id in b.associated_tracks[cid]:
                            matched_bundle = b
                            break
                        last_c = b.camera_centers.get(cid)
                        if last_c is not None:
                            dist = math.hypot(trk.center[0] - last_c[0], trk.center[1] - last_c[1])
                            if dist < 220.0:  # Stack / bundle envelope
                                matched_bundle = b
                                break
                    else:
                        # Across cameras: coalesce into active transit bundle
                        matched_bundle = b
                        break

            # 2b. Check recently committed bundles (post-commit refractory absorption to prevent duplicate bundles)
            if matched_bundle is None:
                for cb in self._committed_bundles:
                    c_time = cb.commit_t if cb.commit_t is not None else cb.last_t
                    if (timestamp_sec - c_time) <= 2.5:
                        if cid in cb.associated_tracks and trk.local_track_id in cb.associated_tracks[cid]:
                            matched_bundle = cb
                            break
                        last_c = cb.camera_centers.get(cid)
                        if last_c is not None:
                            dist = math.hypot(trk.center[0] - last_c[0], trk.center[1] - last_c[1])
                            if dist < 220.0 or zone.is_exit:
                                matched_bundle = cb
                                break

            if matched_bundle is not None:
                # Absorb tracklet into bundle
                matched_bundle.associated_tracks.setdefault(cid, set()).add(trk.local_track_id)
                matched_bundle.camera_bboxes[cid] = getattr(trk, "current_bbox", (0, 0, 0, 0))
                matched_bundle.camera_centers[cid] = trk.center
                # Count simultaneous concurrent tracklets observed in this view for this frame
                simultaneous_in_frame = len([
                    t for c, t, z in observed_tracklets
                    if c == cid and (
                        t.local_track_id in matched_bundle.associated_tracks[cid]
                        or math.hypot(t.center[0] - trk.center[0], t.center[1] - trk.center[1]) < 220.0
                    )
                ])
                matched_bundle.view_carton_counts[cid] = max(
                    matched_bundle.view_carton_counts.get(cid, 1),
                    simultaneous_in_frame,
                    getattr(trk, "estimated_qty", 1)
                )
                matched_bundle.confidences.append(getattr(trk, "current_confidence", 0.85))
                matched_bundle.last_t = timestamp_sec
                matched_bundle.zone_history.setdefault(cid, []).append((timestamp_sec, zone))
            else:
                # Check if this detection is within refractory window of a recently committed bundle
                is_near_recent_commit = False
                for cb in self._committed_bundles:
                    c_time = cb.commit_t if cb.commit_t is not None else cb.last_t
                    if (timestamp_sec - c_time) <= 2.0:
                        is_near_recent_commit = True
                        break

                # Only initiate a new bundle when worker + cartons cross dock threshold (ENTRY_ZONE)
                can_initiate = not is_near_recent_commit and zone.is_entry

                if can_initiate:
                    b_id = f"bundle_{self._next_bundle_id:05d}"
                    self._next_bundle_id += 1
                    new_b = TransitBundle(
                        bundle_id=b_id,
                        worker_id=getattr(trk, "worker_id", None),
                        start_t=timestamp_sec,
                        last_t=timestamp_sec,
                        direction=self.required_direction,
                        state=BundleState.ENTERED if zone.is_entry else BundleState.IN_TRANSIT,
                        associated_tracks={cid: {trk.local_track_id}},
                        camera_bboxes={cid: getattr(trk, "current_bbox", (0, 0, 0, 0))},
                        camera_centers={cid: trk.center},
                        view_carton_counts={cid: max(1, getattr(trk, "estimated_qty", 1))},
                        confidences=[getattr(trk, "current_confidence", 0.85)],
                        zone_history={cid: [(timestamp_sec, zone)]},
                    )
                    self._active_bundles.append(new_b)

        # Step 3: Evaluate State Machine Transitions & Hysteresis
        for bundle in self._active_bundles:
            if bundle.state in (BundleState.COMMITTED, BundleState.REVERSED_CANCELLED):
                continue

            # Gather recent zones across all contributing cameras
            recent_zones = []
            for cid, history in bundle.zone_history.items():
                if history:
                    recent_zones.append(history[-1][1])

            has_entry = any(z.is_entry for z in recent_zones)
            has_transit = any(z.is_transit for z in recent_zones)
            has_exit = any(z.is_exit for z in recent_zones)

            # Transition 1: Entering Transit Zone
            if bundle.state == BundleState.ENTERED and has_transit:
                bundle.state = BundleState.IN_TRANSIT

            # Transition 2: Hysteresis / Hesitation
            # If bundle is in transit, but worker steps back slightly or pauses:
            # DO NOT increment/decrement! Bundle remains IN_TRANSIT.
            if bundle.state == BundleState.IN_TRANSIT and (has_transit or has_entry):
                bundle.hesitation_count += 1
                # Remains IN_TRANSIT, count is unchanged

            # Transition 3: Reversal Cancellation
            # If bundle was in transit, but retreated completely back into ENTRY_ZONE
            if bundle.state == BundleState.IN_TRANSIT and has_entry and not has_transit and not has_exit:
                bundle.state = BundleState.REVERSED_CANCELLED
                bundle.reversal_detected = True
                logger.info("Bundle %s cancelled due to worker retreat/reversal", bundle.bundle_id)

            # Transition 4: Exit Zone Commit (Entering truck interior)
            # Guarantees bundle has fully exited the threshold corridor into the interior (not has_entry)
            if bundle.state == BundleState.IN_TRANSIT and has_exit and not has_entry and not bundle.is_committed:
                # Invariant: Must have transited for minimum duration (0.20s) to reject flicker
                if (timestamp_sec - bundle.start_t) < 0.20:
                    continue
                bundle.state = BundleState.COMMITTED
                bundle.is_committed = True
                bundle.commit_t = timestamp_sec

                # Disambiguate stack count (N_cols x H_layers)
                resolved_qty = 1
                stack_res_dict = {}
                if disambiguator is not None:
                    stack_res = disambiguator.disambiguate(bundle.view_carton_counts)
                    resolved_qty = stack_res.resolved_count
                    stack_res_dict = stack_res.to_dict()
                else:
                    resolved_qty = max(bundle.view_carton_counts.values()) if bundle.view_carton_counts else 1

                bundle.resolved_count = resolved_qty

                # Consensus verdict
                consensus_dict = {}
                supporting_cams = sorted(list(bundle.associated_tracks.keys()))
                if consensus_policy is not None:
                    verdict = consensus_policy.evaluate(
                        observed_cameras=supporting_cams,
                        confidences=bundle.confidences,
                    )
                    consensus_dict = verdict.to_dict()
                else:
                    consensus_dict = {
                        "level": "THREE_CAMERA_UNANIMOUS" if len(supporting_cams) >= 3 else ("TWO_CAMERA_AGREEMENT" if len(supporting_cams) == 2 else "SINGLE_CAMERA_OBSERVATION"),
                        "action": "COMMIT",
                    }

                ev_id = f"corridor_ev_{len(self._committed_events) + 1:05d}"
                bundle.committed_event_id = ev_id

                stack_action = stack_res_dict.get("decision_action", "COMMIT")
                is_auto_conf = (consensus_dict.get("action") == "COMMIT") and (stack_action == "COMMIT")
                req_review = not is_auto_conf

                ev_record = {
                    "event_id": ev_id,
                    "bundle_id": bundle.bundle_id,
                    "worker_id": bundle.worker_id,
                    "timestamp_sec": round(bundle.start_t, 3),
                    "commit_timestamp_sec": round(timestamp_sec, 3),
                    "direction": bundle.direction,
                    "carton_count": resolved_qty,
                    "participating_cameras": supporting_cams,
                    "state": "COUNT_COMMITTED" if is_auto_conf else ("AMBIGUOUS_STACK" if stack_action == "AMBIGUOUS_STACK" else "REVIEW_REQUIRED"),
                    "decision_action": stack_action,
                    "consensus": consensus_dict,
                    "stack_resolution": stack_res_dict,
                    "is_auto_confirmed": is_auto_conf,
                    "requires_review": req_review,
                    "human_audit_required": stack_res_dict.get("audit_recommended", False),
                    "save_clip_sec": stack_res_dict.get("save_clip_sec", 3.0),
                }

                self._committed_events.append(ev_record)
                self._committed_bundles.append(bundle)
                newly_committed.append(ev_record)
                if bundle.direction == "IN":
                    self._total_in += resolved_qty
                else:
                    self._total_out += resolved_qty

                logger.info("Corridor State Machine fired ONE commit event: %s (qty=%d, cams=%s)",
                            ev_id, resolved_qty, supporting_cams)

        # Step 4: Prune aged inactive bundles (older than 6s)
        self._active_bundles = [
            b for b in self._active_bundles
            if (timestamp_sec - b.last_t) <= self.window_sec
        ]

        return newly_committed


# ---------------------------------------------------------------------------
# Independent Per-Camera Tripwire Engine (True Three-Camera Architecture)
# ---------------------------------------------------------------------------
@dataclass
class CameraTripwireConfig:
    """Configures independent crossing boundary geometry for a single camera view."""

    camera_id: str
    orientation: str = "vertical"  # "vertical" (evaluates X) or "horizontal" (evaluates Y)
    line_pos_frac: float = 0.50  # normalized position across frame dimension [0.0 - 1.0]
    direction_sign: int = 1  # +1 (increasing coordinate) or -1 (decreasing coordinate)
    hysteresis_frac: float = 0.01  # buffer band fraction
    min_displacement_px: float = 10.0  # minimum motion required
    cooldown_frames: int = 20  # cooldown per tracklet
    support_occlusion_stitching: bool = True
    min_transit_sec: float = 0.08
    max_transit_sec: float = 3.2
    physically_verified: bool = False


@dataclass
class LocalCameraCrossingCandidate:
    """An independent crossing event candidate emitted by a single camera view."""

    local_event_id: str
    camera_id: str
    local_track_id: str
    timestamp_sec: float
    direction: str  # "IN" or "OUT"
    confidence: float
    bbox: Tuple[float, float, float, float]
    aspect_ratio: float
    estimated_qty: int = 1
    displacement: float = 0.0
    occlusion_stitched: bool = False
    estimated_stack_height: Optional[int] = None
    estimated_width_qty: Optional[int] = None
    crossing_point_px: Optional[Tuple[float, float]] = None
    crossing_line_px: Optional[float] = None
    observed_frame_idx: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


LocalCrossingCandidate = LocalCameraCrossingCandidate


class PerCameraTripwire:
    """
    Independent crossing boundary monitor for one camera view.
    Operates independently without relying on other camera views.
    Handles both continuous single-track crossings and worker-occluded corridor handoffs.
    """

    def __init__(self, config: CameraTripwireConfig) -> None:
        self.cfg = config
        self._track_sides: Dict[str, str] = {}  # local_track_id -> "before" or "after"
        self._initial_sides: Dict[str, str] = {}
        self._last_event_frame: Dict[str, int] = {}
        self._emitted_tracks: set = set()
        self._recent_approaches: List[Dict[str, Any]] = []
        self._event_counter: int = 1
        self._prev_point: Dict[str, Tuple[float, float, float]] = {}

    def determine_side(self, center: Tuple[float, float], frame_dims: Tuple[int, int]) -> Optional[str]:
        h, w = frame_dims
        dim = w if self.cfg.orientation == "vertical" else h
        coord = center[0] if self.cfg.orientation == "vertical" else center[1]
        line_px = self.cfg.line_pos_frac * dim
        hyst_px = self.cfg.hysteresis_frac * dim

        if coord <= (line_px - hyst_px):
            return "before" if self.cfg.direction_sign > 0 else "after"
        elif coord >= (line_px + hyst_px):
            return "after" if self.cfg.direction_sign > 0 else "before"
        return None  # inside hysteresis deadband

    def _crossing_timestamp(
        self,
        trajectory: Sequence[Tuple[float, float, float]],
        frame_dims: Tuple[int, int],
        direction: str,
        fallback_timestamp: float,
    ) -> float:
        """Interpolate the carton-center crossing time at the configured line."""
        _, crossing_time = self._crossing_point(
            trajectory, frame_dims, direction, fallback_timestamp
        )
        return crossing_time

    def _crossing_point(
        self,
        trajectory: Sequence[Tuple[float, float, float]],
        frame_dims: Tuple[int, int],
        direction: str,
        fallback_timestamp: float,
    ) -> Tuple[Optional[Tuple[float, float]], float]:
        """Interpolate the carton-center position and time at the configured line."""
        if len(trajectory) < 2:
            return None, fallback_timestamp

        h, w = frame_dims
        line_px = self.cfg.line_pos_frac * (w if self.cfg.orientation == "vertical" else h)
        coord_idx = 0 if self.cfg.orientation == "vertical" else 1
        is_loading = direction == "IN"
        increasing = (self.cfg.direction_sign > 0) == is_loading

        points = list(trajectory)
        for previous, current in zip(points, points[1:]):
            previous_coord = float(previous[coord_idx])
            current_coord = float(current[coord_idx])
            crossed = (
                previous_coord <= line_px <= current_coord
                if increasing
                else previous_coord >= line_px >= current_coord
            )
            if not crossed or current_coord == previous_coord:
                continue

            fraction = (line_px - previous_coord) / (current_coord - previous_coord)
            previous_time = float(previous[2])
            current_time = float(current[2])
            point = (
                float(previous[0]) + fraction * (float(current[0]) - float(previous[0])),
                float(previous[1]) + fraction * (float(current[1]) - float(previous[1])),
            )
            return point, previous_time + fraction * (current_time - previous_time)

        return None, fallback_timestamp

    def update(
        self,
        tracklet: Any,
        frame_idx: int,
        timestamp_sec: float,
        frame_dims: Tuple[int, int],
    ) -> Optional[LocalCameraCrossingCandidate]:
        tid = tracklet.local_track_id
        pts = list(getattr(tracklet, "trajectory", ()))
        if len(pts) < 2 and tid in self._prev_point:
            pts = [self._prev_point[tid], (tracklet.center[0], tracklet.center[1], timestamp_sec)]
        self._prev_point[tid] = (tracklet.center[0], tracklet.center[1], timestamp_sec)

        side = self.determine_side(tracklet.center, frame_dims)
        if side is None:
            return None

        # Clean old approach records
        self._recent_approaches = [
            a for a in self._recent_approaches
            if (timestamp_sec - a["last_time"]) <= self.cfg.max_transit_sec
        ]

        coord = tracklet.center[0] if self.cfg.orientation == "vertical" else tracklet.center[1]
        dim_px = frame_dims[1] if self.cfg.orientation == "vertical" else frame_dims[0]
        line_px = self.cfg.line_pos_frac * dim_px
        dist_to_line = abs(coord - line_px)
        max_corridor_px = 0.22 * dim_px

        if tid not in self._track_sides:
            self._initial_sides[tid] = side
            self._track_sides[tid] = side
            if side == "before" and dist_to_line <= max_corridor_px:
                self._recent_approaches.append({
                    "track_id": tid,
                    "last_time": timestamp_sec,
                    "conf": getattr(tracklet, "current_confidence", 0.5),
                    "center": tracklet.center,
                    "paired": False,
                })
            elif side == "after" and self.cfg.support_occlusion_stitching and dist_to_line <= max_corridor_px and tid not in self._emitted_tracks:
                # Occlusion recovery: Carton appeared in after zone.
                # Refractory guard: Only block if within 1.2s AND in same vertical lane
                for c_time, c_center in getattr(self, "_recent_crossings", []):
                    if (timestamp_sec - c_time) < 1.2 and abs(tracklet.center[1] - c_center[1]) < 80.0:
                        return None
                # Check directional movement consistency: an object retreating backwards away from truck
                # must NEVER be stitched as an IN crossing!
                trk_coord = tracklet.center[0] if self.cfg.orientation == "vertical" else tracklet.center[1]
                trk_traj = getattr(tracklet, "trajectory", [])
                if len(trk_traj) > 1:
                    init_coord = trk_traj[0][0] if self.cfg.orientation == "vertical" else trk_traj[0][1]
                    delta_coord = trk_coord - init_coord
                    # If moving against direction sign by > 10px, object is retreating, NOT loading!
                    if (self.cfg.direction_sign > 0 and delta_coord < -10.0) or (self.cfg.direction_sign < 0 and delta_coord > 10.0):
                        return None

                # Require approach to be in reasonable vertical lane proximity (<= 140px)
                valid = [
                    a for a in self._recent_approaches
                    if self.cfg.min_transit_sec <= (timestamp_sec - a["last_time"]) <= self.cfg.max_transit_sec
                    and not a["paired"]
                    and abs(tracklet.center[1] - a.get("center", tracklet.center)[1]) <= 140.0
                ]

                if valid:
                    conf = getattr(tracklet, "current_confidence", 0.5)
                    crossing_point = None
                    if valid:
                        best = min(valid, key=lambda a: abs(timestamp_sec - a["last_time"] - 1.0))
                        best["paired"] = True
                        for a in self._recent_approaches:
                            if abs(a["last_time"] - best["last_time"]) < 1.0 and abs(tracklet.center[1] - a.get("center", tracklet.center)[1]) <= 80.0:
                                a["paired"] = True
                        conf = max(best["conf"], conf)
                        crossing_point, crossing_time = self._crossing_point(
                            [
                                (best["center"][0], best["center"][1], best["last_time"]),
                                (tracklet.center[0], tracklet.center[1], timestamp_sec),
                            ],
                            frame_dims,
                            "IN",
                            timestamp_sec,
                        )

                    if crossing_point is None:
                        return None

                    self._emitted_tracks.add(tid)
                    self._last_event_frame[tid] = frame_idx
                    self._last_crossing_time = timestamp_sec
                    if not hasattr(self, "_recent_crossings"):
                        self._recent_crossings = []
                    self._recent_crossings.append((timestamp_sec, tracklet.center))
                    self._recent_crossings = [
                        c for c in self._recent_crossings if (timestamp_sec - c[0]) <= 5.0
                    ]

                    ev_id = f"{self.cfg.camera_id}_ev_{self._event_counter:04d}"
                    self._event_counter += 1
                    bw = getattr(tracklet, "width", 1.0)
                    bh = getattr(tracklet, "height", 1.0)
                    ar = bh / max(1.0, bw)
                    est_qty = max(1, int(round(ar))) if ar >= 1.6 else 1

                    return LocalCameraCrossingCandidate(
                        local_event_id=ev_id,
                        camera_id=self.cfg.camera_id,
                        local_track_id=tid,
                        timestamp_sec=crossing_time,
                        direction="IN",
                        confidence=conf,
                        bbox=getattr(tracklet, "current_bbox", (0.0, 0.0, 0.0, 0.0)),
                        aspect_ratio=round(ar, 2),
                        estimated_qty=est_qty,
                        displacement=getattr(tracklet, "displacement", 0.0),
                        occlusion_stitched=True,
                        crossing_point_px=crossing_point,
                        crossing_line_px=self.cfg.line_pos_frac * (
                            frame_dims[1] if self.cfg.orientation == "vertical" else frame_dims[0]
                        ),
                        observed_frame_idx=frame_idx,
                        image_width=frame_dims[1],
                        image_height=frame_dims[0],
                    )
            return None

        # Update approach record if in before zone
        if side == "before":
            found = False
            for a in self._recent_approaches:
                if a["track_id"] == tid:
                    a["last_time"] = timestamp_sec
                    a["center"] = tracklet.center
                    a["conf"] = max(a["conf"], getattr(tracklet, "current_confidence", 0.5))
                    found = True
            if not found and dist_to_line <= max_corridor_px:
                self._recent_approaches.append({
                    "track_id": tid,
                    "last_time": timestamp_sec,
                    "conf": getattr(tracklet, "current_confidence", 0.5),
                    "center": tracklet.center,
                    "paired": False,
                })

        prev_side = self._track_sides[tid]
        if side != prev_side:
            self._track_sides[tid] = side
            init_side = self._initial_sides.get(tid, prev_side)

            # Check direction
            direction = "IN" if (init_side == "before" and side == "after") else "OUT"

            # Check cooldown
            last_f = self._last_event_frame.get(tid, -999)
            if frame_idx - last_f < self.cfg.cooldown_frames:
                return None

            # Check displacement
            disp = getattr(tracklet, "displacement", 0.0)
            if disp < self.cfg.min_displacement_px:
                return None

            # Mark any approaches for this track as paired
            for a in self._recent_approaches:
                if a["track_id"] == tid:
                    a["paired"] = True

            self._last_crossing_time = timestamp_sec
            if not hasattr(self, "_recent_crossings"):
                self._recent_crossings = []
            self._recent_crossings.append((timestamp_sec, tracklet.center))
            self._recent_crossings = [
                c for c in self._recent_crossings if (timestamp_sec - c[0]) <= 5.0
            ]
            self._last_event_frame[tid] = frame_idx
            self._emitted_tracks.add(tid)
            ev_id = f"{self.cfg.camera_id}_ev_{self._event_counter:04d}"
            self._event_counter += 1

            bw = getattr(tracklet, "width", 1.0)
            bh = getattr(tracklet, "height", 1.0)
            ar = bh / max(1.0, bw)
            est_qty = max(1, int(round(ar))) if ar >= 1.6 else 1

            crossing_point, crossing_time = self._crossing_point(
                pts,
                frame_dims,
                direction,
                timestamp_sec,
            )

            # Require verified crossing of configured tripwire line
            if crossing_point is None:
                return None

            # For OUT (return) events, require genuine displacement
            if direction == "OUT" and disp < max(25.0, self.cfg.min_displacement_px):
                return None

            return LocalCameraCrossingCandidate(
                local_event_id=ev_id,
                camera_id=self.cfg.camera_id,
                local_track_id=tid,
                timestamp_sec=crossing_time,
                direction=direction,
                confidence=getattr(tracklet, "current_confidence", 0.5),
                bbox=getattr(tracklet, "current_bbox", (0.0, 0.0, 0.0, 0.0)),
                aspect_ratio=round(ar, 2),
                estimated_qty=est_qty,
                displacement=disp,
                occlusion_stitched=False,
                crossing_point_px=crossing_point,
                crossing_line_px=self.cfg.line_pos_frac * (
                    frame_dims[1] if self.cfg.orientation == "vertical" else frame_dims[0]
                ),
                observed_frame_idx=frame_idx,
                image_width=frame_dims[1],
                image_height=frame_dims[0],
            )
        return None
