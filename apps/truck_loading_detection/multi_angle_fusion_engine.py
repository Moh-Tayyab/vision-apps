#!/usr/bin/env python3
"""
Multi-Angle Fusion Engine for Truck Loading Carton & Worker Counting.

Synchronizes and fuses detections across multiple camera angles:
- Camera 1 (Top-Down / Overhead): Horizontal or angled line tracking horizontal width (W cartons).
- Camera 2 (Side Profile): Vertical line tracking vertical stack height (H cartons).
- Camera 3 (Optional Flank): Opposite side profile resolving mutual worker occlusions.

Applies temporal event-window consensus:
  Event Count = Top_Width * Side_Stack_Height
Resolves discrepancies, occlusions, and single-camera dropouts.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence, Tuple, Any
import time
import math
import os
import sys

_sys_path_inserted = False
for _p in (os.path.dirname(os.path.abspath(__file__)),):
    if _p not in sys.path:
        sys.path.insert(0, _p)
        _sys_path_inserted = True

from carton_crossing_fusion import CartonCrossingFusion, CartonCrossing  # noqa: E402


class CameraAngleConfig:
    """Configuration for an individual camera perspective."""
    def __init__(
        self,
        camera_id: str,
        angle_type: str = "side_profile",  # "top_overhead" or "side_profile"
        line_orientation: str = "vertical",  # "vertical" (X line) or "horizontal" (Y line)
        line_pos: int = 400,
        direction: str = "left_to_right",  # "left_to_right", "right_to_left", "up_to_down", "down_to_up"
        hysteresis: int = 15,
        cooldown_frames: int = 15,
        role: Optional[str] = None,  # "width" | "height" | "corroborator"
        line_pos_frac: Optional[float] = None,  # 0..1 of the frame dimension
        start_offset_sec: float = 0.0  # global-timeline time when this camera started recording
    ):
        self.camera_id = camera_id
        self.angle_type = angle_type
        self.line_orientation = line_orientation
        self.line_pos = line_pos
        self.direction = direction
        self.hysteresis = hysteresis
        self.cooldown_frames = cooldown_frames
        self.role = role if role is not None else CameraAngleConfig.infer_role(angle_type)
        self.line_pos_frac = line_pos_frac
        self.start_offset_sec = start_offset_sec

    @staticmethod
    def infer_role(angle_type: str) -> str:
        """Maps a legacy angle_type onto a fusion role."""
        if angle_type == "top_overhead":
            return "width"
        if angle_type == "side_profile":
            return "height"
        return "corroborator"

class AngleTripwireTracker:
    """
    Per-angle directional tripwire tracker supporting both:
    1. Vertical lines (X-axis) for side profile horizontal movement.
    2. Horizontal lines (Y-axis) for top-down vertical movement.
    """
    def __init__(self, config: CameraAngleConfig):
        self.cfg = config
        self.track_positions: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.track_sides: Dict[int, str] = {}
        self.track_last_frame: Dict[int, int] = {}
        self.cooldowns: Dict[int, int] = {}
        self.events: List[Dict[str, Any]] = []

        self.total_in = 0
        self.total_out = 0

    def reset(self):
        self.track_positions.clear()
        self.track_sides.clear()
        self.track_last_frame.clear()
        self.cooldowns.clear()
        self.events.clear()
        self.total_in = 0
        self.total_out = 0

    def update(
        self,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float, int]],
        frame_idx: int,
        timestamp_sec: float
    ) -> List[Dict[str, Any]]:
        """
        tracked_objects: List of (track_id, (x1, y1, x2, y2), class_name, confidence, carton_qty)

        `carton_qty` is the number of cartons the worker is CURRENTLY holding.
        A value of 0 means the worker is empty handed, and such a crossing
        contributes 0 cartons (it must not be counted as a single carton).
        """
        frame_events = []

        for track_id, (x1, y1, x2, y2), cls_name, conf, carton_qty in tracked_objects:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self.track_positions[track_id].append((cx, cy, frame_idx))
            self.track_last_frame[track_id] = frame_idx

            # Determine position along counting axis
            curr_coord = cx if self.cfg.line_orientation == "vertical" else cy
            line = self.cfg.line_pos
            hyst = self.cfg.hysteresis

            # Determine side ("before" vs "after" tripwire)
            if self.cfg.direction in ("left_to_right", "up_to_down"):
                curr_side = "after" if curr_coord >= (line + hyst) else ("before" if curr_coord <= (line - hyst) else None)
            else:  # right_to_left or down_to_up
                curr_side = "after" if curr_coord <= (line - hyst) else ("before" if curr_coord >= (line + hyst) else None)

            prev_side = self.track_sides.get(track_id)
            if prev_side is None:
                if curr_side is not None:
                    self.track_sides[track_id] = curr_side
                continue

            # Check crossing
            if curr_side is not None and curr_side != prev_side:
                # Check cooldown
                last_cool = self.cooldowns.get(track_id, -999)
                if (frame_idx - last_cool) >= self.cfg.cooldown_frames:
                    # Direction determination
                    if prev_side == "before" and curr_side == "after":
                        event_dir = "IN"
                        self.total_in += carton_qty
                    else:
                        event_dir = "OUT"
                        self.total_out += carton_qty

                    self.cooldowns[track_id] = frame_idx
                    self.track_sides[track_id] = curr_side

                    event_record = {
                        "camera_id": self.cfg.camera_id,
                        "angle_type": self.cfg.angle_type,
                        "track_id": track_id,
                        "class_name": cls_name,
                        "confidence": float(conf),
                        "direction": event_dir,
                        "carton_qty": max(0, int(carton_qty)),
                        "holding": int(carton_qty) > 0,
                        "frame_idx": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 3),
                        "bbox": (int(x1), int(y1), int(x2), int(y2))
                    }
                    frame_events.append(event_record)
                    self.events.append(event_record)

        return frame_events


class MultiAngleFusionEngine:
    """
    Fuses passing events across all configured camera angles.
    Maintains a sliding temporal window to match events from Camera 1 and Camera 2.
    """
    def __init__(self, time_window_sec: float = 0.85, count_directions: Optional[Sequence[str]] = None,
                 fusion_mode: str = "wh"):
        self.time_window_sec = time_window_sec
        # Which tripwire directions contribute to the carton totals. Set it to
        # ["IN"] when the cartons only ever travel one way (e.g. a pallet that
        # is only being loaded), so a worker walking back past the staging
        # area cannot add phantom "returned" cartons.
        self.count_directions = tuple(count_directions) if count_directions else ("IN", "OUT")
        # Fusion mode: "wh" = legacy Width x Height (stacked load), "carton" = direct
        # carton-centric counting with cross-camera deduplication.
        self.fusion_mode = fusion_mode
        self.trackers: Dict[str, AngleTripwireTracker] = {}
        self.fused_events: List[Dict[str, Any]] = []
        # Carton-centric fusion engine (used when fusion_mode == "carton")
        self._carton_fusion: Optional[CartonCrossingFusion] = None
        # role == "width_sampler": (timestamp, width) readings taken from a
        # camera that cannot produce tripwire crossings (e.g. a handheld
        # overhead shot) but can still report how WIDE a load is.
        self.sampler_readings: Dict[str, List[Tuple[float, int]]] = defaultdict(list)

        # Overall Master Running Counts
        self.total_loaded_cartons = 0
        self.total_returned_cartons = 0
        self.total_worker_trips = 0

    def add_camera(self, config: CameraAngleConfig) -> None:
        if config.role == "width_sampler":
            self.sampler_readings.setdefault(config.camera_id, [])
            return
        self.trackers[config.camera_id] = AngleTripwireTracker(config)
        if self.fusion_mode == "carton":
            if self._carton_fusion is None:
                self._carton_fusion = CartonCrossingFusion(
                    time_window_sec=self.time_window_sec,
                    count_directions=self.count_directions,
                    min_cameras=1,
                )
            self._carton_fusion.add_camera(config.camera_id)

    def reset(self) -> None:
        for t in self.trackers.values():
            t.reset()
        self.sampler_readings.clear()
        self.fused_events.clear()
        self.total_loaded_cartons = 0
        self.total_returned_cartons = 0
        self.total_worker_trips = 0
        if self._carton_fusion is not None:
            self._carton_fusion.reset()

    def update_camera_frame(
        self,
        camera_id: str,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float, int]],
        frame_idx: int,
        timestamp_sec: float
    ) -> List[Dict[str, Any]]:
        """Updates tracker for a specific camera and checks for new angle events."""
        if camera_id in self.sampler_readings:
            # Width sampler: record how many cartons the widest held load is.
            widths = [int(o[4]) for o in tracked_objects if int(o[4]) > 0]
            if widths:
                self.sampler_readings[camera_id].append((timestamp_sec, max(widths)))
            return []

        if camera_id not in self.trackers:
            raise ValueError(f"Unknown camera_id: {camera_id}")

        tracker = self.trackers[camera_id]
        events = tracker.update(tracked_objects, frame_idx, timestamp_sec)

        # Carton-centric mode: ingest each crossing into the dedup engine.
        if self.fusion_mode == "carton" and self._carton_fusion is not None:
            for ev in events:
                self._carton_fusion.ingest(CartonCrossing(
                    camera_id=ev["camera_id"],
                    track_id=ev["track_id"],
                    timestamp_sec=ev["timestamp_sec"],
                    direction=ev["direction"],
                    bbox=ev["bbox"],
                    confidence=ev.get("confidence", 1.0),
                ))
        return events

    def _sampled_width(self, camera_id: str, timestamp_sec: float) -> Optional[int]:
        """Most recent width reading for a sampler camera, within the window."""
        readings = self.sampler_readings.get(camera_id)
        if not readings:
            return None
        best = None
        best_diff = None
        for ts, w in readings:
            diff = abs(ts - timestamp_sec)
            if diff <= self.time_window_sec and (best_diff is None or diff < best_diff):
                best, best_diff = w, diff
        return best


    def _apply_count(self, direction: str, cartons: int) -> None:
        """Adds cartons to the running totals, honouring count_directions."""
        if direction not in self.count_directions:
            return
        if direction == "IN":
            self.total_loaded_cartons += cartons
        else:
            self.total_returned_cartons += cartons

    def process_time_window_consensus(self, current_time_sec: float) -> List[Dict[str, Any]]:
        """
        Scans recent angle events from all cameras within the time window.
        Fuses complementary information (Width x Height) OR direct carton
        deduplication, depending on fusion_mode.
        """
        # --- Carton-centric mode: direct deduplication via CartonCrossingFusion ---
        if self.fusion_mode == "carton":
            if self._carton_fusion is None:
                return []
            summary = self._carton_fusion.fuse(current_time_sec)
            fused_now = []
            for carton in summary.cartons:
                f_record = {
                    "fused_id": carton.carton_id,
                    "timestamp_sec": carton.timestamp_sec,
                    "direction": carton.direction,
                    "fused_cartons": 1,
                    "top_camera_width": 1,
                    "side_camera_height": 1,
                    "consensus_status": (
                        "THREE_CAMERA_VERIFIED" if len(carton.cameras_seen) >= 3
                        else ("DUAL_CAMERA_VERIFIED" if len(carton.cameras_seen) >= 2
                              else "SINGLE_CAMERA")
                    ),
                    "corroborating_cameras": [c for c in carton.cameras_seen],
                    "contributing_events": [
                        {
                            "camera_id": e.camera_id,
                            "track_id": e.track_id,
                            "direction": e.direction,
                            "timestamp_sec": e.timestamp_sec,
                            "bbox": e.bbox,
                            "confidence": e.confidence,
                        }
                        for e in carton.evidences
                    ],
                }
                # Update running totals
                self._apply_count(carton.direction, 1)
                self.total_worker_trips += 1
                self.fused_events.append(f_record)
                fused_now.append(f_record)
            return fused_now

        # --- Legacy W x H mode ---
        unfused_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Collect events that haven't been incorporated into a fused event
        already_fused_events = set()
        for fe in self.fused_events:
            for ev in fe.get("contributing_events", []):
                already_fused_events.add((ev["camera_id"], ev["frame_idx"], ev["track_id"]))

        for cam_id, tracker in self.trackers.items():
            role = getattr(tracker.cfg, "role", "corroborator")
            for ev in tracker.events:
                key = (ev["camera_id"], ev["frame_idx"], ev["track_id"])
                if key not in already_fused_events:
                    # Check if event is mature enough (older than window or within sync)
                    if (current_time_sec - ev["timestamp_sec"]) <= (self.time_window_sec * 3):
                        unfused_events[role].append(ev)

        width_events = unfused_events.get("width", [])
        height_events = unfused_events.get("height", [])
        corrob_events = unfused_events.get("corroborator", [])
        sampler_cams = list(self.sampler_readings.keys())

        # 1. Match Width events with Height events
        matched_height_indices = set()
        matched_corrob_indices = set()
        fused_now = []

        for w_ev in width_events:
            best_height = None
            best_height_idx = -1
            best_diff = 999.0

            for h_idx, h_ev in enumerate(height_events):
                if h_idx in matched_height_indices:
                    continue
                # Direction must match (both IN or both OUT)
                if w_ev["direction"] != h_ev["direction"]:
                    continue
                diff = abs(w_ev["timestamp_sec"] - h_ev["timestamp_sec"])
                if diff <= self.time_window_sec and diff < best_diff:
                    best_diff = diff
                    best_height = h_ev
                    best_height_idx = h_idx

            if best_height is not None:
                matched_height_indices.add(best_height_idx)

                # Third angle corroboration (optional, same direction, within window)
                corrob = None
                corrob_idx = -1
                for c_idx, c_ev in enumerate(corrob_events):
                    if c_idx in matched_corrob_indices:
                        continue
                    if c_ev["direction"] != w_ev["direction"]:
                        continue
                    diff = abs(w_ev["timestamp_sec"] - c_ev["timestamp_sec"])
                    if diff <= self.time_window_sec:
                        corrob = c_ev
                        corrob_idx = c_idx
                        break
                if corrob is not None:
                    matched_corrob_indices.add(corrob_idx)

                # Compute fused count:
                # Width camera provides horizontal span (W), Height camera provides stack (H)
                # A zero on either side means the worker was empty handed -> 0 cartons.
                w_count = max(0, int(w_ev.get("carton_qty", 0)))
                h_count = max(0, int(best_height.get("carton_qty", 0)))
                fused_cartons = w_count * h_count

                contributing = [w_ev, best_height]
                if corrob is not None:
                    contributing.append(corrob)

                f_record = {
                    "fused_id": len(self.fused_events) + 1,
                    "timestamp_sec": round((w_ev["timestamp_sec"] + best_height["timestamp_sec"]) / 2, 3),
                    "direction": w_ev["direction"],
                    "fused_cartons": fused_cartons,
                    "top_camera_width": w_count,
                    "side_camera_height": h_count,
                    "consensus_status": (
                        "THREE_CAMERA_VERIFIED" if corrob is not None else "DUAL_CAMERA_VERIFIED"
                    ),
                    "corroborating_cameras": [corrob["camera_id"]] if corrob is not None else [],
                    "contributing_events": contributing
                }
                self._apply_count(w_ev["direction"], fused_cartons)
                self.total_worker_trips += 1
                self.fused_events.append(f_record)
                fused_now.append(f_record)
            else:
                # Width-only event (waiting or fallback if window expired)
                if (current_time_sec - w_ev["timestamp_sec"]) > self.time_window_sec:
                    qty = max(0, int(w_ev.get("carton_qty", 0)))
                    f_record = {
                        "fused_id": len(self.fused_events) + 1,
                        "timestamp_sec": w_ev["timestamp_sec"],
                        "direction": w_ev["direction"],
                        "fused_cartons": qty,
                        "top_camera_width": qty,
                        "side_camera_height": 1,
                        "consensus_status": "TOP_ONLY_FALLBACK",
                        "corroborating_cameras": [],
                        "contributing_events": [w_ev]
                    }
                    self._apply_count(w_ev["direction"], qty)
                    self.total_worker_trips += 1
                    self.fused_events.append(f_record)
                    fused_now.append(f_record)

        # 2. Check remaining unmatched Height events (if window expired)
        for h_idx, h_ev in enumerate(height_events):
            if h_idx not in matched_height_indices:
                if (current_time_sec - h_ev["timestamp_sec"]) > self.time_window_sec:
                    qty = max(0, int(h_ev.get("carton_qty", 0)))

                    # A width sampler supplies W for the height event even
                    # though it produced no tripwire event of its own.
                    sampled_w = None
                    sampler_cam = None
                    for cam in sampler_cams:
                        w = self._sampled_width(cam, h_ev["timestamp_sec"])
                        if w is not None:
                            sampled_w = w
                            sampler_cam = cam
                            break

                    w_count = sampled_w if sampled_w is not None else 1
                    h_count = qty
                    fused_cartons = w_count * h_count

                    corrob = None
                    for c_ev in corrob_events:
                        if c_ev["direction"] != h_ev["direction"]:
                            continue
                        if abs(c_ev["timestamp_sec"] - h_ev["timestamp_sec"]) <= self.time_window_sec:
                            corrob = c_ev
                            break

                    status = "HEIGHT_ONLY_FALLBACK"
                    if sampled_w is not None:
                        status = "WIDTH_SAMPLED_VERIFIED" if corrob is None else "WIDTH_SAMPLED_3CAM"
                    elif corrob is not None:
                        status = "SIDE_CORROBORATED"

                    contributing = [h_ev]
                    if corrob is not None:
                        contributing.append(corrob)

                    f_record = {
                        "fused_id": len(self.fused_events) + 1,
                        "timestamp_sec": h_ev["timestamp_sec"],
                        "direction": h_ev["direction"],
                        "fused_cartons": fused_cartons,
                        "top_camera_width": w_count,
                        "side_camera_height": h_count,
                        "consensus_status": status,
                        "corroborating_cameras": ([corrob["camera_id"]] if corrob is not None else []),
                        "width_sampler": sampler_cam,
                        "contributing_events": contributing
                    }
                    self._apply_count(h_ev["direction"], fused_cartons)
                    self.total_worker_trips += 1
                    self.fused_events.append(f_record)
                    fused_now.append(f_record)

        return fused_now

    def get_summary(self) -> Dict[str, Any]:
        """Returns consolidated audit metrics."""
        status_counts: Dict[str, int] = defaultdict(int)
        for fe in self.fused_events:
            status_counts[fe["consensus_status"]] += 1

        return {
            "total_loaded_cartons": self.total_loaded_cartons,
            "total_returned_cartons": self.total_returned_cartons,
            "net_cartons": self.total_loaded_cartons - self.total_returned_cartons,
            "total_worker_trips": self.total_worker_trips,
            "fused_events_count": len(self.fused_events),
            "consensus_breakdown": dict(status_counts),
            "events": self.fused_events
        }
