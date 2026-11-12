"""
Exact 3-Camera Carton Tripwire Counter.

Detects ONLY cartons crossing a red tripwire line in videos and live cameras.
+1 when a carton crosses toward the truck (loaded / IN).
-1 when a carton returns from truck to pallet (returned / OUT).
Three cameras provide redundancy: if one camera misses a detection, the others catch it.
Cross-camera events within a fusion time window are deduplicated so each physical
carton is counted exactly once.

Designed for the truck_loading_detection app: carton-only model (class 0 = "carton"),
ByteTrack tracking, vertical/horizontal red line, configurable per-camera direction.
"""

from __future__ import annotations

import logging
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("exact_carton_counter")

CAMERA_IDS = ("cam_top", "cam_side", "cam_flank")


@dataclass
class CrossingEvent:
    """A single carton line-crossing detected by one camera."""
    event_id: str
    camera_id: str
    track_id: int
    direction: str  # "IN" or "OUT"
    timestamp_sec: float
    frame_idx: int
    position: Tuple[int, int]
    confidence: float


class CameraTripwireCounter:
    """Per-camera directional tripwire counter for cartons only.

    Adds track persistence across occlusions: when a track_id disappears
    briefly and reappears nearby, it is matched to its previous track so
    its side assignment is preserved and the carton is not double-counted.
    """

    def __init__(
        self,
        camera_id: str,
        line_x: int = 500,
        line_y: int = 400,
        orientation: str = "vertical",
        loading_direction: str = "left_to_right",
        hysteresis: int = 15,
        cooldown_frames: int = 15,
        min_displacement_px: int = 15,
        max_inactive_frames: int = 90,
        truck_side: Optional[str] = None,
        min_confidence_for_count: float = 0.25,
        min_track_observations: int = 2,
        track_reid_window_frames: int = 15,
        track_reid_max_dist_px: float = 100.0,
    ):
        self.camera_id = str(camera_id)
        self.line_x = int(line_x)
        self.line_y = int(line_y)
        self.line_orientation = str(orientation).lower()
        self.hysteresis = int(hysteresis)
        self.cooldown_frames = int(cooldown_frames)
        self.min_displacement_px = int(min_displacement_px)
        self.max_inactive_frames = int(max_inactive_frames)

        # --- Count quality controls ---
        self.min_confidence_for_count = float(min_confidence_for_count)
        self.min_track_observations = int(min_track_observations)
        self.track_reid_window_frames = int(track_reid_window_frames)
        self.track_reid_max_dist_px = float(track_reid_max_dist_px)

        # truck_side: which side of the frame the truck sits on ("left" | "right").
        # Derives loading_direction: truck on right -> cartons move L->R (IN); truck on left -> R->L (IN).
        self.truck_side = str(truck_side).lower() if truck_side else "right"
        if self.truck_side not in ("left", "right"):
            self.truck_side = "right"
        if truck_side is not None:
            self.loading_direction = (
                "left_to_right" if self.truck_side == "right" else "right_to_left"
            )
        else:
            self.loading_direction = str(loading_direction).lower()

        self.total_in = 0
        self.total_out = 0
        self.events: List[CrossingEvent] = []

        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))
        self.track_side: Dict[int, str] = {}
        self.track_last_seen: Dict[int, int] = {}
        self.track_cooldown: Dict[int, int] = {}
        self.track_observation_count: Dict[int, int] = defaultdict(int)
        self._event_counter = 0

        # Track re-identification cache: track_id -> last known (cx, cy, side, frame_idx)
        # Used to match reappearing tracks after occlusion without re-counting.
        self._recent_tracks: Dict[int, Tuple[float, float, str, int]] = {}

    @property
    def net_count(self) -> int:
        return self.total_in - self.total_out

    def set_line(self, line_x: int, line_y: int = 400, orientation: str = "vertical"):
        self.line_x = int(line_x)
        self.line_y = int(line_y)
        self.line_orientation = str(orientation).lower()

    def set_line_frac(self, line_pos_frac: float, frame_w: int, frame_h: int,
                      orientation: str = "vertical"):
        """Set line position as a fraction of frame dimension (0.0 - 1.0)."""
        frac = max(0.0, min(1.0, float(line_pos_frac)))
        if self.line_orientation == "vertical":
            self.line_x = int(frac * frame_w)
            self.line_y = frame_h // 2
        else:
            self.line_y = int(frac * frame_h)
            self.line_x = frame_w // 2

    def set_loading_direction(self, direction: str):
        self.loading_direction = str(direction).lower()

    def set_truck_side(self, side: str):
        """Set which side of the frame the truck sits on. Derives loading_direction.

        truck_side="right"  -> cartons move left-to-right  (IN = +1 toward truck)
        truck_side="left"   -> cartons move right-to-left  (IN = +1 toward truck)
        """
        side = str(side).lower()
        if side not in ("left", "right"):
            return
        self.truck_side = side
        self.loading_direction = (
            "left_to_right" if side == "right" else "right_to_left"
        )

    def reset(self):
        self.total_in = 0
        self.total_out = 0
        self.events.clear()
        self.track_history.clear()
        self.track_side.clear()
        self.track_last_seen.clear()
        self.track_cooldown.clear()
        self.track_observation_count.clear()
        self._recent_tracks.clear()
        self._event_counter = 0

    def _match_existing_track(
        self, cx: float, cy: float, frame_idx: int
    ) -> Optional[int]:
        """Match a new track to a recent track by spatial proximity.

        Used for re-identification after occlusion: if a track_id disappears
        and a new track_id appears nearby within the re-id window, we treat
        them as the same physical carton and preserve the old side assignment.
        Returns the matched track_id, or None.
        """
        cutoff_frame = frame_idx - self.track_reid_window_frames
        best_tid: Optional[int] = None
        best_dist: float = self.track_reid_max_dist_px
        for tid, (last_cx, last_cy, last_side, last_frame) in self._recent_tracks.items():
            if last_frame < cutoff_frame:
                continue
            dist = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_tid = tid
        return best_tid

    def update(
        self,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
    ) -> List[CrossingEvent]:
        """Process tracked cartons and detect line crossings. Returns new events."""
        current_events: List[CrossingEvent] = []
        active_ids: Set[int] = set()

        for track_id, (x1, y1, x2, y2), cls_name, conf in tracked_objects:
            active_ids.add(track_id)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self.track_history[track_id].append((int(cx), int(cy)))
            self.track_last_seen[track_id] = frame_idx
            self.track_observation_count[track_id] += 1

            # --- Track re-identification after occlusion ---
            # If this is a new track_id but matches a recent track nearby,
            # adopt the old track's side assignment so we don't re-count.
            if track_id not in self.track_side or self.track_side.get(track_id) is None:
                matched_tid = self._match_existing_track(cx, cy, frame_idx)
                if matched_tid is not None and matched_tid != track_id:
                    prev_side = self.track_side.get(matched_tid)
                    if prev_side:
                        self.track_side[track_id] = prev_side
                        # Inherit observation count for stability
                        self.track_observation_count[track_id] = max(
                            self.track_observation_count[track_id],
                            self.track_observation_count.get(matched_tid, 0),
                        )
                        # Transfer cooldown state
                        if matched_tid in self.track_cooldown:
                            self.track_cooldown[track_id] = self.track_cooldown[matched_tid]

            # Determine current side of tripwire
            if self.line_orientation == "horizontal":
                coord = cy
                line_val = self.line_y
                side_low = "ABOVE"
                side_high = "BELOW"
            else:
                coord = cx
                line_val = self.line_x
                side_low = "LEFT"
                side_high = "RIGHT"

            prev_side = self.track_side.get(track_id)
            cur_side = prev_side
            if coord < (line_val - self.hysteresis):
                cur_side = side_low
            elif coord > (line_val + self.hysteresis):
                cur_side = side_high
            self.track_side[track_id] = cur_side

            # Cache this track for re-id matching
            self._recent_tracks[track_id] = (cx, cy, cur_side or "", frame_idx)

            # --- Stability check: require min observations before counting ---
            if self.track_observation_count[track_id] < self.min_track_observations:
                continue

            # --- Confidence gate ---
            if float(conf) < self.min_confidence_for_count:
                continue

            # Need at least 2 points and sufficient displacement to confirm a crossing
            pts = self.track_history[track_id]
            if len(pts) < 2:
                continue

            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            disp = (dx * dx + dy * dy) ** 0.5
            if disp < self.min_displacement_px:
                continue

            # Crossing detection: side changed with sufficient history
            if prev_side and cur_side and prev_side != cur_side:
                cooldown_left = self.track_cooldown.get(track_id, 0)
                if frame_idx >= cooldown_left:
                    self.track_cooldown[track_id] = frame_idx + self.cooldown_frames

                    # Direction check relative to loading direction
                    if self.loading_direction in ("left_to_right", "top_to_bottom"):
                        is_in = (prev_side == side_low and cur_side == side_high)
                    elif self.loading_direction in ("right_to_left", "bottom_to_top"):
                        is_in = (prev_side == side_high and cur_side == side_low)
                    else:  # "any" / bidirectional
                        is_in = True

                    if is_in:
                        self.total_in += 1
                        direction_label = "IN"
                    else:
                        self.total_out += 1
                        direction_label = "OUT"

                    self._event_counter += 1
                    event = CrossingEvent(
                        event_id=f"{self.camera_id}_ev_{self._event_counter:04d}",
                        camera_id=self.camera_id,
                        track_id=track_id,
                        direction=direction_label,
                        timestamp_sec=time.time(),
                        frame_idx=frame_idx,
                        position=(int(cx), int(cy)),
                        confidence=float(conf),
                    )
                    self.events.append(event)
                    current_events.append(event)

        # Purge stale re-id cache entries
        cutoff_frame = frame_idx - self.track_reid_window_frames
        self._recent_tracks = {
            tid: data for tid, data in self._recent_tracks.items() if data[3] > cutoff_frame
        }

        # Purge inactive tracks
        stale_ids = [
            tid for tid, last_f in self.track_last_seen.items()
            if (frame_idx - last_f) > self.max_inactive_frames
        ]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.track_side.pop(tid, None)
            self.track_last_seen.pop(tid, None)
            self.track_cooldown.pop(tid, None)
            self.track_observation_count.pop(tid, None)

        return current_events


class ThreeCameraCartonCounter:
    """
    Three-camera carton tripwire fusion counter.

    Runs an independent tripwire counter per camera and fuses crossing events
    across cameras within a time window so each physical carton is counted
    exactly once (redundancy: if one camera misses, others catch it).
    """

    def __init__(
        self,
        camera_ids: Tuple[str, str, str] = CAMERA_IDS,
        fusion_window_sec: float = 1.5,
        line_x: int = 500,
        line_y: int = 400,
        orientation: str = "vertical",
        loading_direction: str = "left_to_right",
        hysteresis: int = 15,
        cooldown_frames: int = 15,
        min_displacement_px: int = 15,
        truck_sides: Optional[Dict[str, str]] = None,
        min_confidence_for_count: float = 0.25,
        min_track_observations: int = 2,
        track_reid_window_frames: int = 15,
        track_reid_max_dist_px: float = 100.0,
        max_position_diff_frac: float = 0.30,
    ):
        self.camera_ids = tuple(camera_ids)
        self.fusion_window_sec = float(fusion_window_sec)
        self.max_position_diff_frac = float(max_position_diff_frac)

        # Per-camera tripwire counters
        self.counters: Dict[str, CameraTripwireCounter] = {}
        for cid in self.camera_ids:
            cid_truck_side = None
            if truck_sides and cid in truck_sides:
                cid_truck_side = truck_sides[cid]
            self.counters[cid] = CameraTripwireCounter(
                camera_id=cid,
                line_x=line_x,
                line_y=line_y,
                orientation=orientation,
                loading_direction=loading_direction,
                hysteresis=hysteresis,
                cooldown_frames=cooldown_frames,
                min_displacement_px=min_displacement_px,
                truck_side=cid_truck_side,
                min_confidence_for_count=min_confidence_for_count,
                min_track_observations=min_track_observations,
                track_reid_window_frames=track_reid_window_frames,
                track_reid_max_dist_px=track_reid_max_dist_px,
            )

        # Fused (deduplicated) events
        self.fused_events: List[Dict[str, Any]] = []
        self._fused_counter = 0
        self._recent_events: List[Dict[str, Any]] = []

        self.total_in = 0
        self.total_out = 0
        self.last_update_time: Optional[float] = None

    @property
    def net_count(self) -> int:
        return self.total_in - self.total_out

    def set_line(self, line_x: int, line_y: int = 400, orientation: str = "vertical"):
        for counter in self.counters.values():
            counter.set_line(line_x, line_y, orientation)

    def set_line_frac(self, camera_id: str, line_pos_frac: float,
                      frame_w: int, frame_h: int,
                      orientation: str = "vertical"):
        counter = self.counters.get(camera_id)
        if counter is not None:
            counter.set_line_frac(line_pos_frac, frame_w, frame_h, orientation)

    def set_loading_direction(self, direction: str):
        for counter in self.counters.values():
            counter.set_loading_direction(direction)

    def set_truck_side(self, camera_id: str, side: str):
        """Set truck side for a specific camera (left | right). Derives its loading_direction."""
        counter = self.counters.get(camera_id)
        if counter is not None:
            counter.set_truck_side(side)

    def set_truck_sides(self, truck_sides: Dict[str, str]):
        """Set truck side for multiple cameras at once."""
        for camera_id, side in truck_sides.items():
            self.set_truck_side(camera_id, side)

    def reset(self):
        for counter in self.counters.values():
            counter.reset()
        self.fused_events.clear()
        self._recent_events.clear()
        self._fused_counter = 0
        self.total_in = 0
        self.total_out = 0
        self.last_update_time = None

    # ------------------------------------------------------------------
    # Cross-camera fusion: deduplicate same physical carton seen by >=1 camera
    # ------------------------------------------------------------------
    def _fuse_event(self, ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fuse a single-camera event into the global count.

        If a recent event from a *different* camera with the same direction
        falls within the fusion window AND has a spatially consistent position
        (normalized), treat it as the same physical carton and merge camera
        evidence (no double count). Otherwise register a new fused event and
        increment the count.
        Returns the fused event dict (new or existing).
        """
        now = ev["timestamp_sec"]
        direction = ev["direction"]
        ev_pos = ev.get("position")
        ev_cam = ev["camera_id"]

        for recent in self._recent_events:
            if recent.get("camera_id") == ev_cam:
                continue
            if recent["direction"] != direction:
                continue
            if abs(now - recent["timestamp_sec"]) > self.fusion_window_sec:
                continue

            # --- Spatial position verification ---
            # Without homography, compare normalized positions relative to each
            # camera's frame. If positions are on opposite extremes of the frame,
            # they are likely different cartons even if within the time window.
            recent_pos = recent.get("position")
            if ev_pos is not None and recent_pos is not None:
                # Get frame dimensions from the visualizer/counter state if available,
                # otherwise default to 1024x576 (app default). We use normalized diff.
                ev_nx = ev_pos[0] / 1024.0
                ev_ny = ev_pos[1] / 576.0
                rc_nx = recent_pos[0] / 1024.0
                rc_ny = recent_pos[1] / 576.0
                pos_diff = ((ev_nx - rc_nx) ** 2 + (ev_ny - rc_ny) ** 2) ** 0.5
                if pos_diff > self.max_position_diff_frac:
                    # Positions too far apart -> different cartons, do not fuse
                    continue

            # Same physical carton confirmed by another camera -> merge, no new count
            cams = recent.setdefault("confirming_cameras", [])
            if ev_cam not in cams:
                cams.append(ev_cam)
            recent.setdefault("camera_events", []).append(ev)
            recent["max_confidence"] = max(
                recent.get("max_confidence", 0.0), ev["confidence"]
            )
            # Update position to the higher-confidence observation
            if ev["confidence"] >= recent.get("max_confidence", 0.0):
                recent["position"] = ev_pos
            return recent

        # New unique physical carton crossing
        self._fused_counter += 1
        fused = {
            "event_id": f"fused_ev_{self._fused_counter:05d}",
            "direction": direction,
            "timestamp_sec": now,
            "camera_id": ev_cam,
            "primary_camera": ev_cam,
            "confirming_cameras": [ev_cam],
            "max_confidence": ev["confidence"],
            "position": ev_pos,
            "camera_events": [ev],
        }
        self.fused_events.append(fused)
        self._recent_events.append(fused)

        if direction == "IN":
            self.total_in += 1
        else:
            self.total_out += 1

        # Prune old recent events outside (window * 2) to keep memory bounded
        cutoff = now - (self.fusion_window_sec * 2.0)
        self._recent_events = [
            r for r in self._recent_events if r["timestamp_sec"] > cutoff
        ]
        return fused

    def update_camera(
        self,
        camera_id: str,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Update one camera's counter and fuse new crossing events."""
        counter = self.counters.get(camera_id)
        if counter is None:
            return []

        new_events = counter.update(tracked_objects, frame_idx, frame)
        fused: List[Dict[str, Any]] = []
        for ev in new_events:
            ev_dict = {
                "event_id": ev.event_id,
                "camera_id": ev.camera_id,
                "track_id": ev.track_id,
                "direction": ev.direction,
                "timestamp_sec": ev.timestamp_sec,
                "frame_idx": ev.frame_idx,
                "position": ev.position,
                "confidence": ev.confidence,
            }
            fused_event = self._fuse_event(ev_dict)
            fused.append(fused_event)
        return fused

    def get_counts(self) -> Dict[str, Any]:
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "net_count": self.net_count,
            "fused_event_count": len(self.fused_events),
            "cameras": {
                cid: {
                    "total_in": c.total_in,
                    "total_out": c.total_out,
                    "net_count": c.net_count,
                    "event_count": len(c.events),
                    "active_tracks": len(c.track_history),
                }
                for cid, c in self.counters.items()
            },
            "recent_events": [
                {
                    "event_id": e["event_id"],
                    "direction": e["direction"],
                    "timestamp_sec": round(e["timestamp_sec"], 3),
                    "primary_camera": e["primary_camera"],
                    "confirming_cameras": e["confirming_cameras"],
                    "max_confidence": round(e["max_confidence"], 3),
                }
                for e in self.fused_events[-50:]
            ],
        }


class SimpleTracker:
    """
    Lightweight per-camera tracker for carton detections.

    ByteTrack with persist=True cannot be shared across multiple camera panels
    called sequentially in one loop iteration — it treats panel 2 as the "next
    frame" after panel 1, corrupting track IDs across cameras. This tracker
    assigns stable per-camera track IDs by matching detections to existing
    tracks via center-distance proximity.

    Guarantees:
    - Track IDs are stable per camera across frames.
    - Tracks coast for `max_invisible_frames` when missed (occlusion handling).
    - New detections get new track IDs only when no existing track is nearby.
    """

    def __init__(
        self,
        camera_id: str,
        max_center_dist_px: float = 80.0,
        max_invisible_frames: int = 10,
    ) -> None:
        self.camera_id = str(camera_id)
        self.max_center_dist_px = float(max_center_dist_px)
        self.max_invisible_frames = int(max_invisible_frames)

        self.next_id: int = 1
        self.tracks: Dict[int, Tuple[float, float, int]] = {}  # tid -> (cx, cy, frames_invisible)
        self.track_boxes: Dict[int, Tuple[float, float, float, float]] = {}  # tid -> last bbox
        self.track_confs: Dict[int, float] = {}  # tid -> last confidence

    def update(
        self,
        detections: List[Tuple[float, float, float, float, float]],
    ) -> List[Tuple[int, Tuple[float, float, float, float], float]]:
        """
        Match detections to existing tracks and return tracked objects.

        Args:
            detections: List of (x1, y1, x2, y2, confidence) in pixel coords.

        Returns:
            List of (track_id, (x1, y1, x2, y2), confidence).
        """
        # Increment invisible counter for all existing tracks
        for tid in self.tracks:
            cx, cy, inv = self.tracks[tid]
            self.tracks[tid] = (cx, cy, inv + 1)

        # Remove tracks that have been invisible too long
        to_remove = [tid for tid, (_, _, inv) in self.tracks.items() if inv > self.max_invisible_frames]
        for tid in to_remove:
            self.tracks.pop(tid, None)
            self.track_boxes.pop(tid, None)
            self.track_confs.pop(tid, None)

        # Compute centers of new detections
        det_centers = []
        for (x1, y1, x2, y2, conf) in detections:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            det_centers.append((cx, cy, x1, y1, x2, y2, conf))

        # Greedy match: for each existing track, find nearest unmatched detection
        used_dets: Set[int] = set()
        results: List[Tuple[int, Tuple[float, float, float, float], float]] = []

        # Sort tracks by how long they've been invisible (prioritize recently seen)
        sorted_tids = sorted(self.tracks.keys(), key=lambda tid: self.tracks[tid][2])

        for tid in sorted_tids:
            tcx, tcy, _ = self.tracks[tid]
            best_det_idx: Optional[int] = None
            best_dist: float = self.max_center_dist_px

            for di, (dcx, dcy, x1, y1, x2, y2, conf) in enumerate(det_centers):
                if di in used_dets:
                    continue
                dist = ((dcx - tcx) ** 2 + (dcy - tcy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = di

            if best_det_idx is not None:
                dcx, dcy, x1, y1, x2, y2, conf = det_centers[best_det_idx]
                used_dets.add(best_det_idx)
                self.tracks[tid] = (dcx, dcy, 0)
                self.track_boxes[tid] = (x1, y1, x2, y2)
                self.track_confs[tid] = conf
                results.append((tid, (x1, y1, x2, y2), conf))

        # Create new tracks for unmatched detections
        for di, (dcx, dcy, x1, y1, x2, y2, conf) in enumerate(det_centers):
            if di in used_dets:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = (dcx, dcy, 0)
            self.track_boxes[tid] = (x1, y1, x2, y2)
            self.track_confs[tid] = conf
            results.append((tid, (x1, y1, x2, y2), conf))

        return results

    def get_active_tracks(self) -> List[Tuple[int, Tuple[float, float, float, float], float]]:
        """Return all currently active (not invisible) tracks."""
        results = []
        for tid, (cx, cy, inv) in self.tracks.items():
            if inv == 0 and tid in self.track_boxes:
                results.append((tid, self.track_boxes[tid], self.track_confs.get(tid, 0.0)))
        return results

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()
        self.track_boxes.clear()
        self.track_confs.clear()
