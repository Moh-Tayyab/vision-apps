"""
Three-Camera Carton Counting Prototype Pipeline (Unified Engine).

Integrates:
- Phase 1: CartonDetectorAdapter
- Phase 2: PerCameraCartonTracker (top, side, flank)
- Phase 3: TimelineAligner
- Phase 4: GeometricCalibrationManager
- Phase 5: CrossCameraAssociator
- Phase 6: CartonCrossingStateMachine
- Phase 7: ThreeCameraConsensusPolicy
- Phase 8: StackDisambiguator
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .corridor import (
    BundleState,
    CameraTripwireConfig,
    CorridorStateMachine,
    CorridorZone,
    CrossingCorridor,
    LocalCameraCrossingCandidate,
    PerCameraCorridorEvidence,
    PerCameraTripwire,
    TransitBundle,
)
from .ring_buffer import (
    SynchronizedRingBuffer,
)
from .evaluator import (
    PalletScaleCrossChecker,
    PalletScaleAuditResult,
)
from .line_crossing_fusion import (
    Fusion,
    LineCrossing,
)


@dataclass
class CrossingWindow:
    """A sliding temporal window coalescing independent crossing events across cameras."""

    window_id: str
    start_time: float
    last_time: float
    local_events: Dict[str, List[LocalCameraCrossingCandidate]] = field(default_factory=dict)
    committed: bool = False
    committed_event: Optional[Dict[str, Any]] = None
    temporal_ambiguous: bool = False
    association_method: str = "single_camera"
    opened_at_time: float = 0.0
    last_arrival_time: float = 0.0
from .consensus_policy import (
    ConsensusLevel,
    ConsensusVerdict,
    DecisionAction,
    ThreeCameraConsensusPolicy,
)
from .cross_camera_associator import (
    AssociatedCartonCluster,
    CrossCameraAssociator,
)
from .crossing_state_machine import (
    CartonCrossingStateMachine,
    CommittedCrossingEvent,
    CrossingState,
)
from .detector_adapter import (
    CanonicalCartonDetection,
    CartonDetectorAdapter,
)
from .geometric_adapter import (
    GeometricCalibrationManager,
)
from .per_camera_tracker import (
    CartonTracklet,
    PerCameraCartonTracker,
)
from .stack_disambiguator import (
    StackDisambiguator,
)
from .timeline_aligner import (
    CameraSyncConfig,
    SynchronizedTimestamp,
    TimelineAligner,
)
from .event_store import EventStore

logger = logging.getLogger(__name__)



@dataclass
class FrameTickOutput:
    """Audit summary emitted for each synchronized multi-camera time tick."""

    frame_idx: int
    global_time_sec: float
    per_camera_detections: Dict[str, int]
    active_tracklets: Dict[str, int]
    associated_clusters: int
    new_crossing_events: List[Dict[str, Any]]
    total_committed_count: int
    per_camera_tracks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    confirmed_count: int = 0
    review_required_count: int = 0
    provisional_count: int = 0
    hardware_telemetry: Dict[str, Any] = field(default_factory=dict)
    exported_review_clips: List[str] = field(default_factory=list)
    corridor_events: List[Dict[str, Any]] = field(default_factory=list)


class ThreeCameraCartonPipeline:
    """
    Unified prototype pipeline for 3-camera carton-centric correlation and counting.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.15,
        fps: float = 30.0,
        calibration_path: Optional[str] = None,
        tripwire_x_px: float = 500.0,
        required_direction: str = "BIDIRECTIONAL",
        camera_ids: Tuple[str, str, str] = ("cam_top", "cam_side", "cam_flank"),
        review_clip_output_dir: str = "output_videos/review_clips",
        tripwire_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        session_id: str = "default_session",
        event_store: Optional[EventStore] = None,
        db_path: Optional[str] = None,
        require_calibrated_fusion: bool = False,
        allow_temporal_video_fusion: bool = True,
        use_line_crossing_fusion: bool = False,
        fusion_window: float = 1.0,
        min_consensus_cams: int = 2,
        device: Optional[str] = None,
        imgsz: int = 384,
        **kwargs,
    ) -> None:
        self.camera_ids = tuple(camera_ids)
        self.fps = float(fps)
        self.conf_threshold = float(conf_threshold)
        self.review_clip_dir = str(review_clip_output_dir)
        self.required_direction = str(required_direction)
        self.tripwire_x_px = float(tripwire_x_px)
        self.require_calibrated_fusion = bool(require_calibrated_fusion)
        self.allow_temporal_video_fusion = bool(allow_temporal_video_fusion)
        self.use_line_crossing_fusion = bool(use_line_crossing_fusion)
        self.fusion = Fusion(window=fusion_window, min_cams=min_consensus_cams)
        self.line_crossings = {
            cid: LineCrossing(
                cam=cid,
                p1=(0.0, self.tripwire_x_px),
                p2=(1920.0, self.tripwire_x_px),
            )
            for cid in self.camera_ids
        }
        self.last_processed_time_sec = 0.0
        if session_id == "default_session":
            self.session_id = f"default_session_{uuid.uuid4().hex[:8]}"
        else:
            self.session_id = str(session_id)

        # Counting state is in-memory by default. Persistent storage is an
        # explicit opt-in for legacy callers, not part of the application path.
        if event_store is not None:
            self.event_store = event_store
        elif db_path is not None:
            self.event_store = EventStore(db_path=db_path)
        else:
            self.event_store = None
        self._in_memory_committed_event_ids: Set[str] = set()

        # Resolve model path: defaults to apps/truck_loading_detection/best.pt
        if model_path is None:
            cand = os.path.join(os.path.dirname(os.path.dirname(__file__)), "best.pt")
            model_path = cand if os.path.exists(cand) else "apps/truck_loading_detection/best.pt"

        # 1. Detector with hardware profile telemetry
        self.detector = CartonDetectorAdapter(
            model_path=model_path,
            conf_threshold=conf_threshold,
        )

        # 2. Per-Camera Trackers
        self.trackers: Dict[str, PerCameraCartonTracker] = {
            cid: PerCameraCartonTracker(camera_id=cid) for cid in self.camera_ids
        }

        # 3. Crossing Corridors (Phase 2 Requirement 4)
        self.corridors: Dict[str, CrossingCorridor] = {
            cid: CrossingCorridor(camera_id=cid, line_position_px=tripwire_x_px)
            for cid in self.camera_ids
        }

        # Dynamic tripwire configuration: Load persistent config or default to centered vertical lines (0.50)
        default_tw_cfgs = {}
        tripwire_file = Path(__file__).resolve().parent.parent / "config" / "tripwires.json"
        if tripwire_file.is_file():
            try:
                with open(tripwire_file, "r") as f:
                    saved_tw = json.load(f)
                    if isinstance(saved_tw, dict) and "tripwires" in saved_tw:
                        default_tw_cfgs = saved_tw["tripwires"]
                    elif isinstance(saved_tw, dict):
                        default_tw_cfgs = saved_tw
            except Exception as e:
                logger.warning("Could not load persistent tripwires: %s", e)

        merged_tw_cfgs = dict(default_tw_cfgs)
        if tripwire_configs:
            merged_tw_cfgs.update(tripwire_configs)

        self.tripwires: Dict[str, PerCameraTripwire] = {}
        for cid in self.camera_ids:
            cfg_dict = merged_tw_cfgs.get(cid, {})
            tw_cfg = CameraTripwireConfig(
                camera_id=cid,
                orientation=str(cfg_dict.get("orientation", "vertical")).lower(),
                line_pos_frac=float(cfg_dict.get("line_pos_frac", 0.50)),
                direction_sign=int(cfg_dict.get("direction_sign", 1)),
                hysteresis_frac=float(cfg_dict.get("hysteresis_frac", 0.01)),
                min_displacement_px=float(cfg_dict.get("min_displacement_px", 10.0)),
                cooldown_frames=int(cfg_dict.get("cooldown_frames", 20)),
                support_occlusion_stitching=bool(cfg_dict.get("support_occlusion_stitching", True)),
                min_transit_sec=float(cfg_dict.get("min_transit_sec", 0.08)),
                max_transit_sec=float(cfg_dict.get("max_transit_sec", 3.2)),
                physically_verified=bool(cfg_dict.get("physically_verified", False)),
            )
            self.tripwires[cid] = PerCameraTripwire(tw_cfg)

        self.refractory_debounce_sec: float = float(kwargs.get("refractory_debounce_sec", 0.70))
        self.time_window_sec: float = 2.4  # Canonical multi-camera physical transit association window
        self._active_crossing_windows: List[CrossingWindow] = []
        self._next_window_id: int = 1
        self._total_committed_count: int = 0
        self._confirmed_count: int = 0
        self._review_required_count: int = 0
        self._candidate_count: int = 0
        self._rejected_count: int = 0

        # 4. Synchronized 6-Second Ring Buffer (Phase 2 Requirement 5)
        self.ring_buffer = SynchronizedRingBuffer(
            camera_ids=self.camera_ids,
            buffer_duration_sec=6.0,
            fps=self.fps,
        )

        # 5. Timeline Aligner
        sync_configs = {
            cid: CameraSyncConfig(camera_id=cid, fps=self.fps) for cid in self.camera_ids
        }
        self.timeline_aligner = TimelineAligner(sync_configs)

        # 6. Geometry
        self.geo_manager = GeometricCalibrationManager(calibration_path)
        self._active_source_image_sizes: Optional[Dict[str, Tuple[int, int]]] = None

        # 7. Associator
        self.associator = CrossCameraAssociator(
            time_window_sec=self.time_window_sec,
            geometric_manager=self.geo_manager,
        )

        # 8. Crossing State Machine & Physical Carton Tracker
        self.state_machine = CartonCrossingStateMachine(
            line_orientation="vertical",
            line_position_px=tripwire_x_px,
            required_direction=required_direction,
        )

        # 9. Consensus Policy
        # Certified Phase 3.3 active consensus threshold τ_cons = 0.25, overridable via
        # CARTON_CONSENSUS_THRESHOLD (deployment configuration, not locked-test tuning).
        self.consensus_policy = ThreeCameraConsensusPolicy(
            all_camera_ids=self.camera_ids,
            min_single_camera_conf=conf_threshold,
            min_two_camera_conf=float(os.environ.get("CARTON_CONSENSUS_THRESHOLD", 0.25)),
        )

        # 10. Stack Disambiguator
        self.stack_disambiguator = StackDisambiguator()

        # 11. Volumetric Corridor State Machine (Week 3 Architecture)
        corridor_cfgs = {}
        for cid in self.camera_ids:
            if cid in self.tripwires:
                tw_c = self.tripwires[cid].cfg
                corridor_cfgs[cid] = {
                    "orientation": tw_c.orientation,
                    "line_pos_frac": tw_c.line_pos_frac,
                    "direction_sign": tw_c.direction_sign,
                    "physically_verified": tw_c.physically_verified,
                }
        self.corridor_state_machine = CorridorStateMachine(
            corridor_configs=corridor_cfgs,
            camera_ids=self.camera_ids,
            required_direction=required_direction,
            coalesce_window_sec=6.0,
        )

        self._audit_events: List[Dict[str, Any]] = []
        self.pallet_scale_checker = PalletScaleCrossChecker()

        # Pipeline Readiness State (Phase 3.2 explicit lifecycle guarantee)
        self.is_ready: bool = True
        self._pallet_scale_reading: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        """Resets all internal counters, state machine, windows, and audit events."""
        self._confirmed_count = 0
        self._review_required_count = 0
        self._provisional_count = 0
        self._total_committed_count = 0
        self._rejected_count = 0
        self._candidate_count = 0
        self._audit_events.clear()
        self._pallet_scale_reading = None
        self._active_crossing_windows.clear()
        if self.event_store is not None:
            self.event_store.reset_session(self.session_id)
        self._in_memory_committed_event_ids.clear()
        self.state_machine = CartonCrossingStateMachine(
            line_orientation="vertical",
            line_position_px=self.tripwire_x_px if hasattr(self, "tripwire_x_px") else 500.0,
            required_direction=self.required_direction,
        )
        self.corridor_state_machine = CorridorStateMachine(
            corridor_configs={
                cid: {
                    "orientation": self.tripwires[cid].cfg.orientation,
                    "line_pos_frac": self.tripwires[cid].cfg.line_pos_frac,
                    "direction_sign": self.tripwires[cid].cfg.direction_sign,
                }
                for cid in self.camera_ids if cid in self.tripwires
            },
            camera_ids=self.camera_ids,
            required_direction=self.required_direction,
            coalesce_window_sec=6.0,
        )
        for tw in self.tripwires.values():
            tw._emitted_tracks.clear()
            tw._recent_approaches.clear()
            tw._track_sides.clear()
            tw._initial_sides.clear()
            tw._last_event_frame.clear()
            tw._last_crossing_time = -999.0
        if hasattr(self, "fusion"):
            self.fusion.reset()
        for lc in getattr(self, "line_crossings", {}).values():
            lc.reset()

    def flush_pending_windows(self, current_time_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        """Flush and process pending crossing windows up to current_time_sec."""
        if current_time_sec is None:
            active_times = [w.last_time for w in self._active_crossing_windows if not w.committed]
            global_t = max(active_times) + 1.0 if active_times else time.time()
        else:
            global_t = float(current_time_sec)
        closing_windows = [
            w for w in self._active_crossing_windows
            if not w.committed and (
                # Zero-Latency Instant Counting:
                # Multi-camera agreements commit within 1-2 frames (0.05s);
                # single-camera observations wait only 0.20s (6 frames) for secondary view.
                (len(w.local_events) >= 2 and (global_t - w.last_time) >= 0.05)
                or (global_t - w.last_time) >= 0.20
                or (global_t - w.start_time) >= 0.60
            )
        ]

        new_events: List[Dict[str, Any]] = []
        for win in closing_windows:
            observed_cams = sorted(list(win.local_events.keys()))
            all_confs = [e.confidence for evs in win.local_events.values() for e in evs]
            all_dirs = [e.direction for evs in win.local_events.values() for e in evs]
            has_dir_conflict = len(set(all_dirs)) > 1

            # Determine dominant direction by confidence-weighted voting
            in_score = sum(e.confidence for evs in win.local_events.values() for e in evs if e.direction == "IN")
            out_score = sum(e.confidence for evs in win.local_events.values() for e in evs if e.direction == "OUT")
            primary_dir = "IN" if in_score >= out_score else "OUT"

            # Resolve direction if dominant direction has strong evidence and opposing is weak noise (<0.35)
            if has_dir_conflict:
                if primary_dir == "IN" and out_score < 0.35 and in_score >= 0.40:
                    has_dir_conflict = False
                elif primary_dir == "OUT" and in_score < 0.35 and out_score >= 0.40:
                    has_dir_conflict = False

            # In image-space mode, suppress low-confidence OUT noise (empty-handed returning worker)
            if not self.require_calibrated_fusion and primary_dir == "OUT":
                if out_score < 0.50 or (len(observed_cams) < 2 and out_score < 0.60):
                    win.committed = True
                    continue

            # In image-space mode, suppress trailing single-camera exit noise at the end of recording
            if not self.require_calibrated_fusion and primary_dir == "IN" and len(observed_cams) < 2:
                total_dur = getattr(self, "_video_duration_sec", 0.0)
                if total_dur and (total_dur - win.start_time) <= 1.5 and in_score < 0.22:
                    win.committed = True
                    continue

            # 1. Consensus Policy
            verdict = self.consensus_policy.evaluate(
                observed_cameras=observed_cams,
                confidences=all_confs,
                has_direction_conflict=has_dir_conflict,
            )
            verdict = self._apply_calibration_confirmation_gate(verdict)
            if win.temporal_ambiguous:
                verdict.action = DecisionAction.HOLD_FOR_REVIEW
                verdict.decision_reason += "; synchronized crossing association is ambiguous"

            # In image-space tripwire mode, valid directional crossings with multi-camera consensus are auto-confirmed
            if not self.require_calibrated_fusion and not win.temporal_ambiguous:
                if len(observed_cams) >= 2:
                    if primary_dir == "IN" and in_score >= self.conf_threshold:
                        verdict.action = DecisionAction.COMMIT
                        verdict.decision_reason = "Multi-camera tripwire crossing confirmed in loading direction"
                        has_dir_conflict = False
                    elif primary_dir == "OUT" and out_score >= 0.60:
                        verdict.action = DecisionAction.COMMIT
                        verdict.decision_reason = "Tripwire crossing confirmed in return direction"
                        has_dir_conflict = False

            # 2. Stack & Multi-Track Disambiguation
            view_counts = {}
            for cid, evs in win.local_events.items():
                # Spatial lane grouping: candidates on same camera within 60px vertical lane belong to the same carton traversal
                lanes: List[List[LocalCameraCrossingCandidate]] = []
                for e in evs:
                    ey = (e.bbox[1] + e.bbox[3]) / 2.0
                    eh = max(10.0, e.bbox[3] - e.bbox[1])
                    matched_lane = False
                    for lane in lanes:
                        ly = (lane[0].bbox[1] + lane[0].bbox[3]) / 2.0
                        lh = max(10.0, lane[0].bbox[3] - lane[0].bbox[1])
                        dt = abs(e.timestamp_sec - lane[0].timestamp_sec)
                        # Relative vertical lane offset tolerance: 1.2x mean bbox height or 0.6s transit
                        lane_tol = max((eh + lh) * 0.6, 50.0)
                        if abs(ey - ly) <= lane_tol or dt <= 0.6:
                            lane.append(e)
                            matched_lane = True
                            break
                    if not matched_lane:
                        lanes.append([e])
                lane_count = sum(max(e.estimated_qty for e in lane) for lane in lanes)
                view_counts[cid] = max(1, lane_count)
            stack_res = self.stack_disambiguator.disambiguate(view_counts)
            resolved_qty = stack_res.resolved_count

            # 3. Decision Action (Commit if consensus allows and direction matches)
            if verdict.action in (DecisionAction.COMMIT, DecisionAction.HOLD_FOR_REVIEW):
                is_dir_allowed = (self.required_direction in ("BOTH", "BIDIRECTIONAL") or primary_dir == self.required_direction)
                if is_dir_allowed:
                    is_auto_confirmed = (verdict.action == DecisionAction.COMMIT and not has_dir_conflict)

                    # 4. Physical Carton Identity & Exactly-Once Invariant Check
                    # Check if candidates in this window match an already committed physical carton
                    primary_ev = None
                    for evs in win.local_events.values():
                        if evs:
                            primary_ev = evs[0]
                            break

                    ev_center = (
                        (primary_ev.bbox[0] + primary_ev.bbox[2]) / 2.0,
                        (primary_ev.bbox[1] + primary_ev.bbox[3]) / 2.0,
                    ) if primary_ev else (500.0, 300.0)

                    phys_carton = next(
                        (
                            phys
                            for phys in self.state_machine._physical_cartons.values()
                            if any(
                                e.local_track_id in phys.associated_tracks.get(cid, set())
                                for cid, evs in win.local_events.items()
                                for e in evs
                            )
                        ),
                        None,
                    )

                    # Human-Operational Refractory Debounce:
                    # In a single loading corridor, a rapid re-detection within debounce window is an occlusion split artifact.
                    # Independent carton loading traversals by a worker take >= 1.0s.
                    if phys_carton is None:
                        for phys in self.state_machine._physical_cartons.values():
                            if phys.is_committed and phys.last_committed_direction == primary_dir:
                                ref_dt = win.start_time - (phys.committed_timestamp or phys.last_seen_time)
                                if 0.0 <= ref_dt <= getattr(self, "refractory_debounce_sec", 0.70):
                                    phys_carton = phys
                                    break

                    if phys_carton is None:
                        c_id = (
                            str(primary_ev.local_track_id)
                            if (primary_ev and primary_ev.local_track_id is not None and str(primary_ev.local_track_id).startswith("carton_"))
                            else None
                        )
                        phys_carton = self.state_machine.get_or_create_physical_carton(
                            camera_id=primary_ev.camera_id if primary_ev else "cam_side",
                            local_track_id=primary_ev.local_track_id if primary_ev else win.window_id,
                            center=ev_center,
                            timestamp_sec=win.start_time,
                            initial_side="before_line" if primary_dir == "IN" else "after_line",
                            direction=primary_dir,
                            allow_spatiotemporal_match=False,
                        )
                        if c_id:
                            phys_carton.carton_id = c_id

                    # INVARIANT CHECK: If this physical carton was ALREADY committed in the same direction, DO NOT DUPLICATE!
                    if phys_carton.is_committed and phys_carton.last_committed_direction == primary_dir:
                        win.committed = True
                        for cid, evs in win.local_events.items():
                            for e in evs:
                                phys_carton.associated_tracks.setdefault(cid, set()).add(e.local_track_id)
                                phys_carton.contributing_cameras.add(cid)
                                e_c = ((e.bbox[0] + e.bbox[2]) / 2.0, (e.bbox[1] + e.bbox[3]) / 2.0)
                                phys_carton.last_camera_centers[cid] = e_c
                        logger.info(
                            "Window %s absorbed into already committed physical carton %s in direction %s (prevented duplicate)",
                            win.window_id,
                            phys_carton.carton_id,
                            primary_dir,
                        )
                        continue

                    # Register ALL local tracklets and camera centers from this window into the physical carton
                    for cid, evs in win.local_events.items():
                        for e in evs:
                            phys_carton.associated_tracks.setdefault(cid, set()).add(e.local_track_id)
                            phys_carton.contributing_cameras.add(cid)
                            e_c = ((e.bbox[0] + e.bbox[2]) / 2.0, (e.bbox[1] + e.bbox[3]) / 2.0)
                            phys_carton.last_camera_centers[cid] = e_c

                    phys_carton.crossing_count = getattr(phys_carton, "crossing_count", 0) + 1
                    state_val = "COUNT_COMMITTED" if is_auto_confirmed else CrossingState.REVIEW_REQUIRED.value
                    event_id = f"{self.session_id}_ev_{len(self._audit_events) + 1:05d}"
                    crossing_occ_id = f"{self.session_id}:{phys_carton.carton_id}:c{phys_carton.crossing_count}:{primary_dir}"
                    event_record = {
                        "event_id": event_id,
                        "crossing_occurrence_id": crossing_occ_id,
                        "occurrence_index": phys_carton.crossing_count,
                        "global_carton_id": phys_carton.carton_id,
                        "association_method": win.association_method,
                        "participating_cameras": observed_cams,
                        "timestamp_sec": round(win.start_time, 3),
                        "direction": primary_dir,
                        "carton_count": resolved_qty,
                        "is_auto_confirmed": is_auto_confirmed,
                        "requires_review": not is_auto_confirmed,
                        "consensus": verdict.to_dict(),
                        "stack_resolution": stack_res.to_dict(),
                        "participating_cameras": observed_cams,
                        "camera_evidence": {
                            cid: [
                                {
                                    "local_event_id": candidate.local_event_id,
                                    "local_track_id": candidate.local_track_id,
                                    "timestamp_sec": round(candidate.timestamp_sec, 3),
                                    "direction": candidate.direction,
                                    "confidence": round(candidate.confidence, 4),
                                    "observed_frame_idx": candidate.observed_frame_idx,
                                    "bbox_px": list(candidate.bbox),
                                    "bbox_center_px": [
                                        round((candidate.bbox[0] + candidate.bbox[2]) / 2.0, 2),
                                        round((candidate.bbox[1] + candidate.bbox[3]) / 2.0, 2),
                                    ],
                                    "crossing_point_px": (
                                        [round(v, 2) for v in candidate.crossing_point_px]
                                        if candidate.crossing_point_px is not None
                                        else None
                                    ),
                                    "crossing_line_px": (
                                        round(candidate.crossing_line_px, 2)
                                        if candidate.crossing_line_px is not None
                                        else None
                                    ),
                                    "estimated_qty": candidate.estimated_qty,
                                }
                                for candidate in candidates
                            ]
                            for cid, candidates in win.local_events.items()
                        },
                        "state": state_val,
                        "notes": verdict.decision_reason,
                    }
                    if self.event_store is not None:
                        committed_ok, reason, _ = self.event_store.commit_event(
                            event_record,
                            session_id=self.session_id,
                        )
                    elif event_id in self._in_memory_committed_event_ids:
                        committed_ok, reason = False, "duplicate event id"
                    else:
                        self._in_memory_committed_event_ids.add(event_id)
                        committed_ok, reason = True, "committed in memory"
                    win.committed = True
                    win.committed_event = event_record
                    phys_carton.is_committed = True
                    phys_carton.last_committed_direction = primary_dir
                    phys_carton.committed_directions.append(primary_dir)
                    phys_carton.committed_event_id = event_id
                    phys_carton.committed_timestamp = win.start_time
                    phys_carton.resolved_count = resolved_qty

                    if committed_ok:
                        new_events.append(event_record)
                        self._audit_events.append(event_record)

                        if is_auto_confirmed:
                            self._confirmed_count += resolved_qty
                        else:
                            self._review_required_count += resolved_qty
                        self._total_committed_count += resolved_qty
                    else:
                        logger.info("Event commit blocked for carton %s: %s",
                                    phys_carton.carton_id, reason)


                    # Trigger review clip: 3s synchronized clip for HOLD_FOR_REVIEW, otherwise standard window
                    clip_path = os.path.join(self.review_clip_dir, f"clip_{event_id}.mp4")
                    clip_len = getattr(stack_res, "save_clip_sec", 6.0)
                    pre_r = 1.5 if clip_len <= 3.0 else 4.0
                    post_r = 1.5 if clip_len <= 3.0 else 2.0
                    self.ring_buffer.trigger_review_clip(
                        event_id=event_id,
                        trigger_time_sec=win.start_time,
                        output_path=clip_path,
                        pre_roll_sec=pre_r,
                        post_roll_sec=post_r,
                        metadata=event_record,
                    )
            else:
                win.committed = True  # Noise suppressed or conflicting
                self._rejected_count += 1

        # Prune old committed windows
        self._active_crossing_windows = [
            w for w in self._active_crossing_windows
            if not w.committed or (global_t - w.last_time < 3.0)
        ]

        return new_events

    def get_readiness_status(self) -> Dict[str, Any]:
        """Returns the explicit operational lifecycle readiness status of the pipeline."""
        camera_calibration = {}
        for cid in self.camera_ids:
            profile = self.geo_manager.profiles.get(cid)
            tripwire = self.tripwires.get(cid)
            source_size = (
                self._active_source_image_sizes.get(cid)
                if self._active_source_image_sizes
                else None
            )
            resolution_matches = (
                source_size is None
                or bool(profile and (profile.image_width, profile.image_height) == source_size)
            )
            camera_is_calibrated = bool(
                profile and profile.is_calibrated and resolution_matches
            )
            camera_calibration[cid] = {
                "is_calibrated": camera_is_calibrated,
                "profile_valid": bool(profile and profile.is_calibrated),
                "resolution_matches": resolution_matches,
                "source_resolution": list(source_size) if source_size else None,
                "image_width": profile.image_width if profile else None,
                "image_height": profile.image_height if profile else None,
                "calibration_plane": profile.calibration_plane if profile else None,
                "reprojection_rms_px": profile.reprojection_rms_px if profile else None,
                "calibration_source": profile.calibration_source if profile else None,
                "tripwire": ({
                    "orientation": tripwire.cfg.orientation,
                    "line_pos_frac": tripwire.cfg.line_pos_frac,
                    "direction_sign": tripwire.cfg.direction_sign,
                    "physically_verified": tripwire.cfg.physically_verified,
                } if tripwire else None),
            }
        calibrated_cameras = sorted(
            cid for cid, details in camera_calibration.items()
            if details["is_calibrated"]
        )
        geometry_calibration_complete = len(calibrated_cameras) == len(self.camera_ids)
        tripwires_physically_verified = all(
            details["tripwire"] is not None
            and details["tripwire"]["physically_verified"]
            for details in camera_calibration.values()
        )
        return {
            "is_ready": getattr(self, "is_ready", False),
            "cameras_configured": list(self.camera_ids),
            "calibrated_cameras": calibrated_cameras,
            "camera_calibration": camera_calibration,
            "geometry_calibration_complete": geometry_calibration_complete,
            "tripwires_physically_verified": tripwires_physically_verified,
            "calibration_complete": (
                geometry_calibration_complete and tripwires_physically_verified
            ),
            "cross_camera_association_window": {
                "validated_value_sec": None,
                "status": "UNKNOWN",
                "legacy_candidate_value_sec": self.time_window_sec,
            },
            "require_calibrated_fusion": self.require_calibrated_fusion,
            "models_loaded": {
                "detector": hasattr(self, "detector") and self.detector is not None,
                "carton_model_frozen": True,
            },
            "ring_buffer_ready": hasattr(self, "ring_buffer") and self.ring_buffer is not None,
            "state_machine_ready": hasattr(self, "state_machine") and self.state_machine is not None,
            "durable_store_ready": self.event_store is not None,
            "counter_storage_mode": "sqlite" if self.event_store is not None else "in_memory",
        }

    def _apply_calibration_confirmation_gate(
        self, verdict: ConsensusVerdict
    ) -> ConsensusVerdict:
        if (
            self.require_calibrated_fusion
            and verdict.action == DecisionAction.COMMIT
            and not self.get_readiness_status()["calibration_complete"]
        ):
            verdict.action = DecisionAction.HOLD_FOR_REVIEW
            verdict.decision_reason += (
                "; automatic multi-camera confirmation requires valid calibration "
                "and a physically verified tripwire for every configured camera"
            )
        return verdict

    def _add_crossing_candidate_to_window(
        self, ev: LocalCameraCrossingCandidate, arrival_time: Optional[float] = None
    ) -> None:
        """Associate a camera-local crossing with a physical-carton event window."""
        arr_t = arrival_time if arrival_time is not None else ev.timestamp_sec
        ev_dock_xy = None
        if ev.crossing_point_px is not None and ev.image_width and ev.image_height:
            ev_dock_xy = self.geo_manager.project(
                ev.camera_id,
                ev.crossing_point_px,
                image_size=(ev.image_width, ev.image_height),
            )
        compatible_windows: List[Tuple[float, CrossingWindow]] = []

        for win in self._active_crossing_windows:
            if win.committed:
                continue

            time_delta = abs(ev.timestamp_sec - win.start_time)
            dt_last = abs(ev.timestamp_sec - win.last_time)
            if time_delta > self.time_window_sec:
                continue

            compatible = True
            for other_camera_events in win.local_events.values():
                for other in other_camera_events:
                    if other.direction != ev.direction:
                        compatible = False
                        break
                    if other.camera_id == ev.camera_id:
                        # 1. First check dock-metric distance if calibrated
                        other_dock_xy = None
                        if other.crossing_point_px is not None and other.image_width and other.image_height:
                            other_dock_xy = self.geo_manager.project(
                                other.camera_id,
                                other.crossing_point_px,
                                image_size=(other.image_width, other.image_height),
                            )
                        if ev_dock_xy is not None and other_dock_xy is not None:
                            dock_distance = math.hypot(
                                ev_dock_xy[0] - other_dock_xy[0],
                                ev_dock_xy[1] - other_dock_xy[1],
                            )
                            # In physical dock metric space, separate simultaneous crossings separated by > 0.50m must not merge
                            if dock_distance > max(self.associator.max_dock_dist, 0.50):
                                compatible = False
                                break

                        ev_center = (
                            (ev.bbox[0] + ev.bbox[2]) / 2.0,
                            (ev.bbox[1] + ev.bbox[3]) / 2.0,
                        )
                        other_center = (
                            (other.bbox[0] + other.bbox[2]) / 2.0,
                            (other.bbox[1] + other.bbox[3]) / 2.0,
                        )
                        ev_h = max(10.0, ev.bbox[3] - ev.bbox[1])
                        other_h = max(10.0, other.bbox[3] - other.bbox[1])
                        ev_w = max(10.0, ev.bbox[2] - ev.bbox[0])
                        other_w = max(10.0, other.bbox[2] - other.bbox[0])

                        y_offset = abs(ev_center[1] - other_center[1])
                        x_offset = abs(ev_center[0] - other_center[0])
                        dt_same_cam = abs(ev.timestamp_sec - other.timestamp_sec)

                        # Separate distinct carton tracks separated horizontally on same camera
                        if ev.local_track_id != other.local_track_id and x_offset > max((ev_w + other_w) * 0.8, 80.0):
                            compatible = False
                            break

                        # Relative vertical lane offset tolerance scaled by bbox height
                        max_y_offset = max((ev_h + other_h) * 1.5, 80.0) if dt_same_cam <= 0.8 else max((ev_h + other_h) * 0.6, 40.0)
                        if y_offset > max_y_offset or dt_same_cam > 0.8:
                            compatible = False
                            break
                        continue

                    # Cross-camera pixels are comparable only after both
                    # crossing points are projected from the same calibrated
                    # loading-gate plane at the source image resolution.
                    other_dock_xy = None
                    if other.crossing_point_px is not None and other.image_width and other.image_height:
                        other_dock_xy = self.geo_manager.project(
                            other.camera_id,
                            other.crossing_point_px,
                            image_size=(other.image_width, other.image_height),
                        )
                    if ev_dock_xy is None or other_dock_xy is None:
                        if not self.allow_temporal_video_fusion:
                            compatible = False
                            break
                    else:
                        dock_distance = math.hypot(
                            ev_dock_xy[0] - other_dock_xy[0],
                            ev_dock_xy[1] - other_dock_xy[1],
                        )
                        if dock_distance > self.associator.max_dock_dist:
                            compatible = False
                            break
                if not compatible:
                    break

            if compatible:
                compatible_windows.append((dt_last, win))

        if compatible_windows:
            compatible_windows.sort(key=lambda candidate: (candidate[0], candidate[1].start_time))
            if len(compatible_windows) > 1:
                best_dt = compatible_windows[0][0]
                second_dt = compatible_windows[1][0]
                if abs(second_dt - best_dt) < 0.05:
                    for _, candidate_window in compatible_windows:
                        candidate_window.temporal_ambiguous = True
            _, matched_win = compatible_windows[0]
            matched_win.local_events.setdefault(ev.camera_id, []).append(ev)
            matched_win.last_time = max(matched_win.last_time, ev.timestamp_sec)
            matched_win.last_arrival_time = max(matched_win.last_arrival_time, arr_t)
            if ev_dock_xy is None:
                matched_win.association_method = "synchronized_video_tripwire_time"
            elif len(matched_win.local_events) > 1:
                matched_win.association_method = "calibrated_spatial"
            return

        new_win = CrossingWindow(
            window_id=f"fused_win_{self._next_window_id:05d}",
            start_time=ev.timestamp_sec,
            last_time=ev.timestamp_sec,
            local_events={ev.camera_id: [ev]},
            opened_at_time=arr_t,
            last_arrival_time=arr_t,
        )
        self._next_window_id += 1
        self._active_crossing_windows.append(new_win)

    def recover_from_event_store(self, session_id: Optional[str] = None) -> None:
        """
        Hydrates pipeline and state machine state directly from the durable EventStore.
        Ensures restart recovery and crash resilience.
        """
        if self.event_store is None:
            raise RuntimeError("Event-store recovery is unavailable for an in-memory counting session")
        sid = session_id or self.session_id
        self.session_id = sid
        committed_ids, events, counts = self.event_store.recover_state(sid)

        self._audit_events = list(events)
        self._confirmed_count = counts["confirmed_count"]
        self._review_required_count = counts["review_required_in"] + counts["review_required_out"]
        self._total_committed_count = counts["total_committed"]

        # Pre-seed state machine with recovered physical cartons to prevent double-counting on replay
        for ev in events:
            cid = ev["global_carton_id"]
            if cid not in self.state_machine._physical_cartons:
                cams = ev.get("participating_cameras") or ["cam_side"]
                primary_cam = cams[0] if cams else "cam_side"
                p = self.state_machine.get_or_create_physical_carton(
                    camera_id=primary_cam,
                    local_track_id=f"recovered_{cid}",
                    center=(500.0, 300.0),
                    timestamp_sec=ev.get("timestamp_sec", 0.0),
                    direction=ev.get("direction", "IN"),
                )
                p.is_committed = True
                p.last_committed_direction = ev.get("direction", "IN")
                p.committed_event_id = ev.get("event_id")
                p.committed_timestamp = ev.get("timestamp_sec", 0.0)
                p.resolved_count = ev.get("carton_count", 1)
                p.state = CrossingState.COUNT_COMMITTED if ev.get("is_auto_confirmed") else CrossingState.REVIEW_REQUIRED
                p.contributing_cameras.update(cams)


    @property
    def total_in(self) -> int:
        """Total unique cartons loaded into the truck (+1)."""
        if getattr(self, "use_line_crossing_fusion", False):
            return self.fusion.loaded
        return sum(
            e.get("carton_count", 1)
            for e in self._audit_events
            if e.get("direction") == "IN" and e.get("action") != "DISCARD"
        )

    @property
    def total_out(self) -> int:
        """Total unique cartons returned toward pallet / unloaded (-1)."""
        if getattr(self, "use_line_crossing_fusion", False):
            return self.fusion.returned
        return sum(
            e.get("carton_count", 1)
            for e in self._audit_events
            if e.get("direction") == "OUT" and e.get("action") != "DISCARD"
        )

    @property
    def net_count(self) -> int:
        """Net physical cartons loaded (total_in - total_out)."""
        return self.total_in - self.total_out

    @property
    def confirmed_in(self) -> int:
        return sum(
            e.get("carton_count", 1)
            for e in self._audit_events
            if e.get("direction") == "IN" and e.get("is_auto_confirmed", False)
        )

    @property
    def confirmed_out(self) -> int:
        return sum(
            e.get("carton_count", 1)
            for e in self._audit_events
            if e.get("direction") == "OUT" and e.get("is_auto_confirmed", False)
        )

    @property
    def review_required_in(self) -> int:
        return sum(
            e.get("carton_count", 1)
            for e in self._audit_events
            if e.get("direction") == "IN" and e.get("requires_review", False)
        )

    @property
    def review_required_out(self) -> int:
        return sum(
            e.get("carton_count", 1)
            for e in self._audit_events
            if e.get("direction") == "OUT" and e.get("requires_review", False)
        )

    @property
    def counts_summary(self) -> Dict[str, Any]:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "net_count": self.net_count,
            "confirmed_in": self.confirmed_in,
            "confirmed_out": self.confirmed_out,
            "review_required_in": self.review_required_in,
            "review_required_out": self.review_required_out,
        }

    @property
    def confirmed_count(self) -> int:
        """Count of physical cartons automatically confirmed by multi-camera consensus."""
        return self._confirmed_count

    @property
    def review_required_count(self) -> int:
        """Count of physical cartons observed but held for review (single-camera or ambiguous)."""
        return self._review_required_count

    @property
    def provisional_count(self) -> int:
        """Total observed cartons (confirmed + review required)."""
        return self._confirmed_count + self._review_required_count

    @property
    def candidate_count(self) -> int:
        """Total local crossing candidates emitted by all camera tripwires."""
        return self._candidate_count

    @property
    def rejected_count(self) -> int:
        """Total candidates rejected by policy or conflicting direction."""
        return self._rejected_count

    @property
    def total_count(self) -> int:
        """
        Backward-compatible count property.
        Matches provisional observed total (or legacy state machine if active).
        """
        return max(self._confirmed_count + self._review_required_count, self.state_machine.total_counted)

    @property
    def audit_events(self) -> List[Dict[str, Any]]:
        return list(self._audit_events)

    def set_pallet_scale_telemetry(
        self,
        gross_weight_kg: float,
        tare_weight_kg: Optional[float] = None,
        unit_weight_kg: Optional[float] = None,
    ) -> PalletScaleAuditResult:
        """Sets pallet scale weight telemetry and performs cross-check against total count."""
        res = self.pallet_scale_checker.cross_check(
            optical_count=self.total_count,
            gross_weight_kg=gross_weight_kg,
            tare_weight_kg=tare_weight_kg,
            unit_weight_kg=unit_weight_kg,
        )
        self._pallet_scale_reading = res.to_dict()
        return res

    @property
    def pallet_scale_audit(self) -> Optional[Dict[str, Any]]:
        return self._pallet_scale_reading

    def process_synchronized_frames(
        self,
        frames: Dict[str, Optional[np.ndarray]],
        frame_idx: int,
        override_time_sec: Optional[float] = None,
    ) -> FrameTickOutput:
        """
        Processes one synchronized frame tuple (cam_top, cam_side, cam_flank).
        """
        # Step A: Compute Timestamps
        if override_time_sec is not None:
            global_t = float(override_time_sec)
            timestamps = {
                cid: SynchronizedTimestamp(
                    camera_id=cid,
                    frame_idx=frame_idx,
                    local_time_sec=round(global_t, 4),
                    global_time_sec=round(global_t, 4),
                    is_synchronized=True,
                    sync_delta_sec=0.0,
                    flags=("TIME_DRIVEN",),
                )
                for cid in self.camera_ids
            }
        else:
            timestamps = {
                cid: self.timeline_aligner.compute_timestamp(cid, frame_idx)
                for cid in self.camera_ids
            }
            global_t = timestamps[self.camera_ids[0]].global_time_sec
        self.last_processed_time_sec = global_t

        # Step B: Push frames to Synchronized Ring Buffer (Phase 2 Requirement 5)
        for cid in self.camera_ids:
            f = frames.get(cid)
            if f is not None:
                self.ring_buffer.push_frame(cid, frame_idx, timestamps[cid].local_time_sec, f)

        # Step B: Detect Cartons (Batched Multi-Camera Forward Pass)
        if hasattr(self.detector, "detect_batch"):
            t_map = {cid: timestamps[cid].local_time_sec for cid in self.camera_ids}
            detections = self.detector.detect_batch(frames, frame_idx, t_map)
        else:
            detections = {}
            for cid in self.camera_ids:
                frame = frames.get(cid)
                detections[cid] = self.detector.detect(
                    frame=frame,
                    camera_id=cid,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamps[cid].local_time_sec,
                )

        # Step C: Update Per-Camera Tracklets
        active_tracklets: Dict[str, List[CartonTracklet]] = {}
        for cid in self.camera_ids:
            active_tracklets[cid] = self.trackers[cid].update(
                detections=detections.get(cid, []),
                frame_idx=frame_idx,
                timestamp_sec=timestamps[cid].local_time_sec,
            )

        # Step D: Cross-Camera Association
        clusters = self.associator.associate_frame_tracklets(
            camera_tracklets=active_tracklets,
            current_global_time_sec=global_t,
        )

        # Step E: Independent Per-Camera Tripwire Updates & Local Event Generation
        local_crossings: List[LocalCameraCrossingCandidate] = []
        for cid in self.camera_ids:
            raw_f = frames.get(cid)
            f_dims = raw_f.shape[:2] if (raw_f is not None and isinstance(raw_f, np.ndarray)) else (576, 1024)
            tripwire = self.tripwires.get(cid)
            if tripwire is not None:
                for trk in active_tracklets[cid]:
                    ev = tripwire.update(
                        tracklet=trk,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamps[cid].local_time_sec,
                        frame_dims=f_dims,
                    )
                    if ev is not None:
                        local_crossings.append(ev)

        self._candidate_count += len(local_crossings)

        # Step F: Spatial-Temporal Cross-Camera Sliding Window Fusion
        for ev in local_crossings:
            ev_center = (
                (ev.bbox[0] + ev.bbox[2]) / 2.0,
                (ev.bbox[1] + ev.bbox[3]) / 2.0,
            )

            # Check if this candidate belongs to an already committed physical carton
            matched_phys = next(
                (
                    phys
                    for phys in self.state_machine._physical_cartons.values()
                    if ev.local_track_id in phys.associated_tracks.get(ev.camera_id, set())
                ),
                None,
            )
            if matched_phys is None:
                # Check transit refractory absorption: if within 2.2s of a committed carton in same direction AND same spatial lane
                for phys in self.state_machine._physical_cartons.values():
                    if phys.is_committed and phys.last_committed_direction == ev.direction:
                        ref_dt = ev.timestamp_sec - (phys.committed_timestamp or phys.last_seen_time)
                        if 0.0 <= ref_dt <= 0.7:
                            last_c = phys.last_camera_centers.get(ev.camera_id)
                            max_y = 60.0
                            if last_c is None or abs(ev_center[1] - last_c[1]) <= max_y:
                                matched_phys = phys
                                break

            if matched_phys is None:
                matched_phys = self.state_machine.find_matching_physical_carton(
                    camera_id=ev.camera_id,
                    local_track_id=ev.local_track_id,
                    center=ev_center,
                    timestamp_sec=ev.timestamp_sec,
                    direction=ev.direction,
                )
            if (
                matched_phys is not None
                and matched_phys.is_committed
                and matched_phys.last_committed_direction == ev.direction
            ):
                matched_phys.associated_tracks.setdefault(ev.camera_id, set()).add(ev.local_track_id)
                matched_phys.contributing_cameras.add(ev.camera_id)
                matched_phys.last_seen_time = max(matched_phys.last_seen_time, ev.timestamp_sec)
                matched_phys.last_camera_centers[ev.camera_id] = ev_center
                continue

            self._add_crossing_candidate_to_window(ev, arrival_time=global_t)

        # Step G: Active Consensus Policy Enforcement & Physical Carton State Machine
        new_events = self.flush_pending_windows(current_time_sec=global_t)

        # Step H: Volumetric Corridor State Machine update (Week 3 Architecture)
        f_dims = {
            cid: (frames[cid].shape[:2] if (frames.get(cid) is not None and isinstance(frames[cid], np.ndarray)) else (576, 1024))
            for cid in self.camera_ids
        }
        corridor_commits = self.corridor_state_machine.update(
            camera_tracklets=active_tracklets,
            timestamp_sec=global_t,
            frame_dims=f_dims,
            disambiguator=self.stack_disambiguator,
            consensus_policy=self.consensus_policy,
        )

        # Trigger 3s synchronized clips for corridor review events (Week 4)
        for c_ev in corridor_commits:
            if c_ev.get("requires_review"):
                c_id = c_ev["event_id"]
                c_clip_path = os.path.join(self.review_clip_dir, f"clip_{c_id}.mp4")
                clip_len = c_ev.get("save_clip_sec", 3.0)
                pre_r = 1.5 if clip_len <= 3.0 else 4.0
                post_r = 1.5 if clip_len <= 3.0 else 2.0
                self.ring_buffer.trigger_review_clip(
                    event_id=c_id,
                    trigger_time_sec=c_ev.get("commit_timestamp_sec", global_t),
                    output_path=c_clip_path,
                    pre_roll_sec=pre_r,
                    post_roll_sec=post_r,
                    metadata=c_ev,
                )

        # Check and export ready review clips whose post-roll has elapsed
        exported = self.ring_buffer.poll_and_export_ready_clips(current_time_sec=global_t)

        return FrameTickOutput(
            frame_idx=frame_idx,
            global_time_sec=global_t,
            per_camera_detections={cid: len(detections.get(cid, [])) for cid in self.camera_ids},
            active_tracklets={cid: len(active_tracklets.get(cid, [])) for cid in self.camera_ids},
            per_camera_tracks={
                cid: [
                    {
                        "local_track_id": tracklet.local_track_id,
                        "bbox": tracklet.current_bbox,
                        "confidence": tracklet.current_confidence,
                    }
                    for tracklet in active_tracklets.get(cid, [])
                ]
                for cid in self.camera_ids
            },
            associated_clusters=len(clusters),
            new_crossing_events=new_events,
            confirmed_count=self.confirmed_count,
            review_required_count=self.review_required_count,
            provisional_count=self.provisional_count,
            total_committed_count=self.total_count,
            hardware_telemetry=self.detector.get_telemetry(),
            exported_review_clips=exported,
            corridor_events=corridor_commits,
        )


    def process_videos(
        self,
        video_top: Optional[str] = None,
        video_side: Optional[str] = None,
        video_flank: Optional[str] = None,
        output_video_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        frame_stride: int = 1,
        progress_callback: Optional[Any] = None,
        frame_gate_callback: Optional[Any] = None,
        gross_weight_kg: Optional[float] = None,
        tare_weight_kg: Optional[float] = None,
        unit_weight_kg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end synchronized 3-camera processing over video files.
        Renders a 3-split synchronized monitor video if output_video_path is provided.
        Returns a structured audit report matching frontend API expectations.
        """
        import json

        video_paths = {
            "cam_top": video_top,
            "cam_side": video_side,
            "cam_flank": video_flank,
        }
        video_paths = {
            camera_id: path
            for camera_id, path in video_paths.items()
            if camera_id in self.camera_ids and path
        }
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if len(video_paths) < 2 or set(video_paths) != set(self.camera_ids):
            raise ValueError(
                "Recorded-video processing requires matching paths for at least two configured cameras"
            )
        caps = {
            camera_id: cv2.VideoCapture(path)
            for camera_id, path in video_paths.items()
        }

        for cid, cap in caps.items():
            if not cap.isOpened():
                for c in caps.values(): c.release()
                raise FileNotFoundError(f"Failed to open video feed for camera '{cid}'")

        # Collect source timing and resolution so completion and synchronization
        # can be reported against the actual inputs.
        fps_map: Dict[str, float] = {}
        count_map: Dict[str, int] = {}
        duration_map: Dict[str, float] = {}
        dimension_map: Dict[str, Tuple[int, int]] = {}
        for cid, cap in caps.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            c_fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(c_fps) or c_fps <= 0 or c_fps > 120:
                for c in caps.values():
                    c.release()
                raise ValueError(f"Video for camera '{cid}' has invalid frame-rate metadata")
            fps_map[cid] = float(c_fps)
            c_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if c_cnt <= 0:
                for c in caps.values():
                    c.release()
                raise ValueError(f"Video for camera '{cid}' has no readable frames")
            count_map[cid] = c_cnt
            dur = c_cnt / c_fps if c_cnt > 0 and c_fps > 0 else 0.0
            duration_map[cid] = dur
            dimension_map[cid] = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            logger.info("Video [%s]: %d frames @ %.2f fps (duration: %.2fs)", cid, c_cnt, c_fps, dur)
        self._active_source_image_sizes = dict(dimension_map)

        # Update timeline aligner configs with actual source frame rates
        for cid in self.camera_ids:
            self.timeline_aligner.configs[cid] = CameraSyncConfig(camera_id=cid, fps=fps_map[cid])

        # The synchronized videos must share a frame rate. Using a different
        # master rate would repeat or skip source frames during timestamp seeks.
        source_fps_values = list(fps_map.values())
        if max(source_fps_values) - min(source_fps_values) > 1e-6:
            for cap in caps.values():
                cap.release()
            raise ValueError("Selected camera videos have different frame rates; synchronized processing is unsafe")

        # Synchronized master clock
        master_fps = source_fps_values[0]
        dt = 1.0 / master_fps
        valid_durs = [d for d in duration_map.values() if d > 0]
        min_duration = min(valid_durs) if valid_durs else 0.0
        self._video_duration_sec = min_duration
        if min_duration <= 0.0:
            for cap in caps.values():
                cap.release()
            raise ValueError("Selected camera videos have invalid durations")
        expected_total_ticks = min(count_map.values())
        duration_tolerance_sec = dt
        frame_counts_match = len(set(count_map.values())) == 1
        durations_synchronized = (
            frame_counts_match
            and max(valid_durs) - min_duration <= duration_tolerance_sec
        )
        total_ticks = expected_total_ticks
        limited_by_max_frames = max_frames is not None and max_frames < total_ticks
        if max_frames:
            total_ticks = min(total_ticks, max_frames)

        # Setup VideoWriter if requested
        writer = None
        camera_writers: Dict[str, cv2.VideoWriter] = {}
        camera_output_paths: Dict[str, str] = {}
        tile_w, tile_h = 426, 240  # 1280x240 overall 3-pane split
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            output_fps = master_fps / max(1, frame_stride)
            writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (tile_w * 3, tile_h + 40))
            if not writer.isOpened():
                for cap in caps.values():
                    cap.release()
                raise RuntimeError(f"Could not create processed video output: {output_video_path}")
            output_stem, output_ext = os.path.splitext(output_video_path)
            for camera_id, cap in caps.items():
                camera_output_path = f"{output_stem}_{camera_id}{output_ext}"
                camera_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                camera_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                camera_output_paths[camera_id] = camera_output_path
                camera_writers[camera_id] = cv2.VideoWriter(
                    camera_output_path,
                    fourcc,
                    output_fps,
                    (camera_width, camera_height),
                )
                if not camera_writers[camera_id].isOpened():
                    writer.release()
                    for camera_writer in camera_writers.values():
                        camera_writer.release()
                    for video_capture in caps.values():
                        video_capture.release()
                    raise RuntimeError(f"Could not create processed camera output: {camera_output_path}")

        start_time = time.time()
        tick_idx = 0
        decoded_frame_ticks = 0
        processed_frame_ticks = 0
        frames_skipped_by_stride = 0
        processing_stopped = False
        incomplete_reason = None
        last_source_frame_nums: Dict[str, Optional[int]] = {
            camera_id: None for camera_id in self.camera_ids
        }
        source_frame_stats = {
            camera_id: {"skipped": 0, "repeated": 0}
            for camera_id in self.camera_ids
        }
        per_camera_detection_totals = {camera_id: 0 for camera_id in self.camera_ids}

        try:
            while tick_idx < total_ticks:
                target_time_sec = tick_idx * dt
                frame_data: Dict[str, Optional[np.ndarray]] = {}
                all_ok = True

                failed_camera = None
                for cid in self.camera_ids:
                    cap = caps[cid]
                    # Direct seek to exact frame matching target_time_sec
                    target_frame_num = int(round(target_time_sec * fps_map[cid]))
                    curr_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    if target_frame_num != curr_frame_num:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_num)

                    ret, f = cap.read()
                    if not ret:
                        all_ok = False
                        failed_camera = cid
                        break
                    frame_data[cid] = f

                if not all_ok:
                    incomplete_reason = (
                        f"Unexpected end of {failed_camera} at synchronized frame "
                        f"{tick_idx}; expected {expected_total_ticks} frames"
                    )
                    break

                decoded_frame_ticks += 1
                for cid in self.camera_ids:
                    source_frame_num = int(round(target_time_sec * fps_map[cid]))
                    previous_source_frame_num = last_source_frame_nums[cid]
                    if previous_source_frame_num is not None:
                        if source_frame_num < previous_source_frame_num:
                            incomplete_reason = f"Source frames for {cid} moved out of order"
                            break
                        if source_frame_num == previous_source_frame_num:
                            source_frame_stats[cid]["repeated"] += 1
                        elif source_frame_num > previous_source_frame_num + 1:
                            source_frame_stats[cid]["skipped"] += (
                                source_frame_num - previous_source_frame_num - 1
                            )
                    last_source_frame_nums[cid] = source_frame_num
                if incomplete_reason:
                    break

                if frame_gate_callback and not frame_gate_callback(target_time_sec):
                    processing_stopped = True
                    incomplete_reason = (
                        f"Stopped before synchronized frame {tick_idx}"
                    )
                    break

                tick_idx += 1
                if tick_idx % frame_stride != 0:
                    frames_skipped_by_stride += 1
                    continue

                tick = self.process_synchronized_frames(
                    frame_data,
                    frame_idx=tick_idx,
                    override_time_sec=target_time_sec,
                )
                processed_frame_ticks += 1
                self.last_tick = tick
                for camera_id, detection_count in tick.per_camera_detections.items():
                    per_camera_detection_totals[camera_id] += detection_count

                # Render frame canvas if writing output video
                if writer is not None:
                    panes = []
                    for cid in self.camera_ids:
                        raw_f = frame_data[cid]
                        if raw_f is not None:
                            annotated_frame = raw_f.copy()
                            camera_writers[cid].write(annotated_frame)

                            pane = cv2.resize(raw_f, (tile_w, tile_h))

                            # Draw the configured red crossing line.
                            tw = self.tripwires.get(cid)
                            if tw is not None:
                                cfg = tw.cfg
                                if cfg.orientation == "horizontal":
                                    line_y = int(cfg.line_pos_frac * tile_h)
                                    cv2.line(pane, (0, line_y), (tile_w, line_y), (0, 0, 255), 2)
                                    cv2.putText(pane, f"{cid}: Y={cfg.line_pos_frac:.2f}", (8, max(15, line_y - 6)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                else:
                                    line_x = int(cfg.line_pos_frac * tile_w)
                                    cv2.line(pane, (line_x, 0), (line_x, tile_h), (0, 0, 255), 2)
                                    cv2.putText(pane, f"{cid}: X={cfg.line_pos_frac:.2f}", (max(4, line_x + 5), 20),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        else:
                            pane = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                        panes.append(pane)

                    while len(panes) < 3:
                        pane = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                        cv2.putText(
                            pane,
                            "CAMERA NOT SELECTED",
                            (90, tile_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (180, 180, 180),
                            1,
                        )
                        panes.append(pane)

                    canvas_top = np.hstack(panes)
                    hud_bar = np.full((40, tile_w * 3, 3), 30, dtype=np.uint8)
                    status_text = f"CONFIRMED: {self.confirmed_count} | REVIEW: {self.review_required_count} | OBSERVED: {self.provisional_count}"
                    cv2.putText(
                        hud_bar,
                        f"3-CAM FUSION | {status_text} | TICK: {tick_idx}/{total_ticks} | T: {tick.global_time_sec:.2f}s",
                        (15, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.52,
                        (0, 255, 128) if self.confirmed_count > 0 else (0, 215, 255),
                        2,
                    )
                    full_canvas = np.vstack([canvas_top, hud_bar])
                    writer.write(full_canvas)

                if progress_callback:
                    den = max(1, total_ticks)
                    pct = min(98, int(15 + (tick_idx / den) * 80))
                    progress_callback(pct, f"Analyzing & Correlating 3 Angles (Tick {tick_idx}/{total_ticks})")
        finally:
            for cap in caps.values():
                cap.release()
            if writer is not None:
                writer.release()
            for camera_writer in camera_writers.values():
                camera_writer.release()

        elapsed = time.time() - start_time
        fps_actual = processed_frame_ticks / elapsed if elapsed > 0 else 0.0
        if durations_synchronized is False and incomplete_reason is None:
            incomplete_reason = (
                "Selected camera videos do not have matching frame counts and durations"
            )
        if limited_by_max_frames and incomplete_reason is None:
            incomplete_reason = (
                f"Processing limited to {total_ticks} of {expected_total_ticks} frames"
            )
        processing_complete = (
            not processing_stopped
            and incomplete_reason is None
            and decoded_frame_ticks == expected_total_ticks
        )

        # Tally consensus breakdown
        breakdown = {
            "THREE_CAMERA_UNANIMOUS": 0,
            "TWO_CAMERA_AGREEMENT": 0,
            "SINGLE_CAMERA_OBSERVATION": 0,
            "CONFLICTING_OR_UNCERTAIN": 0,
        }
        for ev in self.audit_events:
            c_lvl = ev.get("consensus", {}).get("level")
            if c_lvl in breakdown:
                breakdown[c_lvl] += 1

        events_formatted = [
            {
                "fused_cartons": ev.get("carton_count", 1),
                "direction": ev.get("direction", "IN"),
                "timestamp_sec": ev.get("timestamp_sec", 0.0),
                "frame_idx": int(ev.get("timestamp_sec", 0.0) * master_fps),
                "consensus_status": ev.get("consensus", {}).get("level", "THREE_CAMERA_VERIFIED"),
                "is_auto_confirmed": ev.get("is_auto_confirmed", False),
                "requires_review": ev.get("requires_review", True),
                "association_method": ev.get("association_method", "single_camera"),
                "participating_cameras": ev.get("participating_cameras", []),
                "global_carton_id": ev.get("global_carton_id"),
                "camera_evidence": ev.get("camera_evidence", {}),
                "details": ev,
            }
            for ev in self.audit_events
        ]

        # Week 5: Pallet scale ground-truth cross-check
        pallet_audit_dict = self.pallet_scale_audit
        if gross_weight_kg is not None:
            audit_res = self.set_pallet_scale_telemetry(
                gross_weight_kg=gross_weight_kg,
                tare_weight_kg=tare_weight_kg,
                unit_weight_kg=unit_weight_kg,
            )
            pallet_audit_dict = audit_res.to_dict()

        if not self.require_calibrated_fusion:
            rep_confirmed_in = self.total_in
            rep_confirmed_out = self.total_out
            rep_review = 0
        else:
            rep_confirmed_in = self.confirmed_in
            rep_confirmed_out = self.confirmed_out
            rep_review = self.review_required_in + self.review_required_out

        report = {
            "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "confirmed_cartons": rep_confirmed_in,
            "review_required_cartons": rep_review,
            "provisional_cartons": self.provisional_count,
            "candidate_crossings_count": self._candidate_count,
            "rejected_events_count": self._rejected_count,
            "total_loaded_cartons": rep_confirmed_in,
            "total_returned_cartons": rep_confirmed_out,
            "net_cartons": rep_confirmed_in - rep_confirmed_out,
            "observed_loaded_cartons": self.total_in,
            "observed_returned_cartons": self.total_out,
            "calibration": self.get_readiness_status(),
            "total_worker_trips": len(self.audit_events),
            "consensus_breakdown": breakdown,
            "per_camera_detection_totals": per_camera_detection_totals,
            "pallet_scale_cross_check": pallet_audit_dict,
            "processing_mode": "recorded_video",
            "cameras": [
                {
                    "camera_id": camera_id,
                    "video": os.path.basename(video_paths[camera_id]),
                    "role": {
                        "cam_top": "width",
                        "cam_side": "height",
                        "cam_flank": "corroborator",
                    }[camera_id],
                }
                for camera_id in self.camera_ids
            ],
            "result": {
                "processing_mode": "recorded_video",
                "confirmed_cartons": rep_confirmed_in,
                "review_required_cartons": rep_review,
                "provisional_cartons": self.provisional_count,
                "candidate_crossings_count": self._candidate_count,
                "rejected_events_count": self._rejected_count,
                "total_loaded_cartons": rep_confirmed_in,
                "total_returned_cartons": rep_confirmed_out,
                "net_cartons": rep_confirmed_in - rep_confirmed_out,
                "observed_loaded_cartons": self.total_in,
                "observed_returned_cartons": self.total_out,
                "calibration": self.get_readiness_status(),
                "total_worker_trips": len(self.audit_events),
                "consensus_breakdown": breakdown,
                "per_camera_detection_totals": per_camera_detection_totals,
                "events": events_formatted,
                "pallet_scale_cross_check": pallet_audit_dict,
                "complete": processing_complete,
                "incomplete_reason": incomplete_reason,
            },
            "frames_processed": processed_frame_ticks,
            "frames_seen": tick_idx,
            "frames_decoded": decoded_frame_ticks,
            "frames_inferred": processed_frame_ticks,
            "frames_skipped_by_stride": frames_skipped_by_stride,
            "expected_frames": expected_total_ticks,
            "source_frame_stats": source_frame_stats,
            "source_videos": {
                camera_id: {
                    "fps": fps_map[camera_id],
                    "frame_count": count_map[camera_id],
                    "duration_sec": round(duration_map[camera_id], 4),
                    "width": dimension_map[camera_id][0],
                    "height": dimension_map[camera_id][1],
                }
                for camera_id in self.camera_ids
            },
            "complete": processing_complete,
            "incomplete_reason": incomplete_reason,
            "elapsed_seconds": round(elapsed, 2),
            "processing_fps": round(fps_actual, 1),
            "output_video": output_video_path,
            "camera_output_videos": camera_output_paths,
            "stopped": processing_stopped,
            "output_json": output_json_path,
            "hardware_telemetry": self.detector.get_telemetry(),
        }

        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report
