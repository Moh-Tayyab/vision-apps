"""Live carton counting system for top/overhead camera.

Integrates YOLO detection with NMS deduplication, ByteTrack tracking,
worker hand detection, and top-row / top-layer spatial filtering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from detector import CartonDetector, DetectorError
from tracker import ByteTracker, Track
from worker_detector import WorkerDetector
from state_machine import CartonStateMachine, CartonState


@dataclass
class PalletROI:
    """Region of Interest for the pallet area."""
    x1: int
    y1: int
    x2: int
    y2: int
    layer_heights: List[float] = field(default_factory=list)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass
class LiveCountResult:
    """Result from live counting."""
    current_layer: int
    initial_row_cartons: int
    cartons_remaining: int
    cartons_removed_from_row: int
    total_active: int
    total_picked: int
    top_row_count: int
    layer_counts: Dict[int, int]
    cartons_by_state: Dict[str, int]
    hand_detected: bool
    picking_in_progress: bool
    frame_time_ms: float
    tracks: List[dict]
    events: List[dict]
    top_row_only: bool = True
    layer_transition_triggered: bool = False
    total_rows: int = 0  # Total rows/layers visible from angled view


def filter_top_row_cartons(
    detections: List[Tuple[float, float, float, float, float]],
    img_height: int,
    img_width: int,
    top_row_only: bool = True,
) -> Tuple[List[Tuple[float, float, float, float, float]], Dict[int, int], int]:
    """Classifies cartons into vertical stacks and extracts the topmost layer/row.
    
    Returns:
        Tuple of (filtered_detections, layer_map, total_rows)
    """
    if not detections:
        return [], {}, 0

    img_area = max(1, img_height * img_width)

    # 1. Reject background/distant small boxes and extreme vertical side slivers
    valid_dets = []
    for d in detections:
        x1, y1, x2, y2, conf = d
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        rel_area = (w * h) / img_area

        # Distant background rack boxes
        if y2 < 0.16 * img_height and rel_area < 0.025:
            continue

        # Tall vertical slivers spanning across multiple tiers
        if h > 3.0 * w and h > 0.40 * img_height:
            continue

        valid_dets.append(d)

    if not valid_dets:
        valid_dets = detections

    if len(valid_dets) == 1:
        return valid_dets, {0: 0}, 1

    # Group into columns/stacks by X-overlap
    det_items = []
    for idx, d in enumerate(valid_dets):
        x1, y1, x2, y2, conf = d
        det_items.append({
            "idx": idx,
            "det": d,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": x2 - x1, "h": y2 - y1,
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
            "conf": conf,
        })

    # Sort items by X position
    det_items_sorted = sorted(det_items, key=lambda it: it["cx"])

    columns: List[List[dict]] = []
    for item in det_items_sorted:
        matched_col = None
        for col in columns:
            for col_item in col:
                overlap_x = max(0.0, min(item["x2"], col_item["x2"]) - max(item["x1"], col_item["x1"]))
                min_w = min(item["w"], col_item["w"])
                if min_w > 0 and (overlap_x / min_w) > 0.38:
                    matched_col = col
                    break
            if matched_col is not None:
                break

        if matched_col is not None:
            matched_col.append(item)
        else:
            columns.append([item])

    # Count total rows from column structure
    # In angled view, more cartons in a column = more rows
    total_rows = max(len(col) for col in columns) if columns else 0

    top_items = []
    layer_map: Dict[int, int] = {}

    for col in columns:
        # In overhead view, the top carton has lowest Y (uppermost position in stack)
        col_sorted = sorted(col, key=lambda it: (it["y1"] * 0.7 + it["cy"] * 0.3))
        top = col_sorted[0]
        top_items.append(top["det"])
        layer_map[top["idx"]] = 0

        for l_idx, lower in enumerate(col_sorted[1:], start=1):
            layer_map[lower["idx"]] = l_idx

    # Deduplicate top items that overlap horizontally in final list
    top_items_deduped = []
    for td in sorted(top_items, key=lambda d: (d[0] + d[2]) / 2.0):
        overlap = False
        for saved in top_items_deduped:
            ox = max(0.0, min(td[2], saved[2]) - max(td[0], saved[0]))
            min_w = min(td[2] - td[0], saved[2] - saved[0])
            if min_w > 0 and (ox / min_w) > 0.55:
                overlap = True
                break
        if not overlap:
            top_items_deduped.append(td)

    if top_row_only:
        return top_items_deduped, {i: 0 for i in range(len(top_items_deduped))}, total_rows
    return valid_dets, layer_map, total_rows


class LiveCartonCounter:
    """Real-time carton counter with layer lifecycle and pick detection."""

    def __init__(
        self,
        detector: CartonDetector,
        pallet_roi: Optional[PalletROI] = None,
        top_row_only: bool = True,
        tracker_max_age: int = 12,
        tracker_min_hits: int = 2,
        occlusion_timeout: int = 8,
        hand_proximity_threshold: float = 100.0,
    ):
        self.detector = detector
        self.pallet_roi = pallet_roi
        self.top_row_only = top_row_only
        self.tracker = ByteTracker(
            max_age=tracker_max_age,
            min_hits=tracker_min_hits,
        )
        self.worker_detector = WorkerDetector(pallet_roi=pallet_roi)
        self.state_machine = CartonStateMachine(
            occlusion_timeout=occlusion_timeout,
            hand_proximity_threshold=hand_proximity_threshold,
        )
        self.frame_count = 0
        self.initial_cartons: Optional[int] = None

        # Layer tracking states
        self.current_layer: int = 1
        self.initial_row_cartons: int = 0
        self.total_pallet_picked: int = 0
        self.layer_locked: bool = False
        self.confirmed_initial_count: bool = False
        self.stable_frames: int = 0
        self.last_visible_count: int = 0
        self.zero_count_frames: int = 0

        # Auto-count: detect initial cartons from first frames
        self.auto_count_done: bool = False
        self.auto_count_frames: int = 0
        self.auto_count_stable: int = 0
        self.auto_count_last: int = 0

    def set_initial_count(self, count: int):
        """Set initial carton count manually (optional override)."""
        self.initial_cartons = count
        self.initial_row_cartons = count
        self.layer_locked = True
        self.confirmed_initial_count = True
        self.auto_count_done = True

    def process_frame(
        self,
        frame: np.ndarray,
        top_row_only: Optional[bool] = None,
    ) -> LiveCountResult:
        """Process a single frame and return count result."""
        start = time.perf_counter()
        use_top_row = self.top_row_only if top_row_only is None else top_row_only
        self.frame_count += 1

        # Detect cartons with NMS
        try:
            boxes, inference_ms = self.detector.detect(frame, apply_nms=True)
        except DetectorError:
            boxes = []
            inference_ms = 0.0

        # Convert boxes to detection tuples
        detections = [(b["x1"], b["y1"], b["x2"], b["y2"], b["confidence"]) for b in boxes]

        # Filter to ROI if defined
        if self.pallet_roi:
            detections = [
                d for d in detections
                if self.pallet_roi.contains((d[0] + d[2]) / 2, (d[1] + d[3]) / 2)
            ]

        # Apply Top-Row / Top-Layer spatial filter
        h, w = frame.shape[:2]
        filtered_detections, layer_map, total_rows = filter_top_row_cartons(
            detections, img_height=h, img_width=w, top_row_only=use_top_row
        )

        # Update tracker with top row / active detections
        active_tracks = self.tracker.update(filtered_detections)

        # Detect worker hands/pose
        worker_result = self.worker_detector.detect(frame)
        hand_positions = [h.position for h in worker_result.get("hands", [])]
        hand_velocity = self.worker_detector.get_hand_velocity()

        # Update state machine
        detected_ids = [t.track_id for t in active_tracks]
        detected_bboxes = {t.track_id: t.bbox for t in active_tracks}
        detected_confidences = {t.track_id: t.confidence for t in active_tracks}

        events = self.state_machine.update(
            detected_ids=detected_ids,
            detected_bboxes=detected_bboxes,
            detected_confidences=detected_confidences,
            hand_positions=hand_positions,
            hand_velocity=hand_velocity,
        )

        # Active cartons currently resting on top
        active_cartons = self.state_machine.get_active_cartons(include_occluded=False)
        current_visible = len(active_cartons)
        layer_transition = False

        # AUTO-COUNT: Detect initial cartons from first few stable frames
        if not self.auto_count_done and current_visible > 0:
            if current_visible == self.auto_count_last:
                self.auto_count_stable += 1
            else:
                self.auto_count_stable = 1
                self.auto_count_last = current_visible

            # After 5 stable frames, lock the count as baseline
            if self.auto_count_stable >= 5:
                self.initial_row_cartons = current_visible
                self.initial_cartons = current_visible
                self.layer_locked = True
                self.confirmed_initial_count = True
                self.auto_count_done = True

        # Layer transition: when visible drops to 0, move to next layer
        elif self.layer_locked:
            if current_visible == 0 and self.initial_row_cartons > 0:
                self.zero_count_frames += 1
                if self.zero_count_frames >= 6:
                    layer_transition = True
                    self.total_pallet_picked += self.initial_row_cartons
                    events.append({
                        "event": f"layer_{self.current_layer}_cleared",
                        "layer": self.current_layer,
                        "time": time.time(),
                        "track_id": 0,
                    })
                    self.current_layer += 1
                    self.layer_locked = False
                    self.confirmed_initial_count = False
                    self.initial_row_cartons = 0
                    self.zero_count_frames = 0
                    self.auto_count_done = False
                    self.auto_count_stable = 0
                    self.auto_count_last = 0
                    self.tracker.reset()
                    self.state_machine.reset()
            else:
                self.zero_count_frames = 0

        # FORMULA:
        # Cartons Removed = Initial Row Cartons - Current Visible Cartons
        if self.layer_locked and self.initial_row_cartons >= current_visible:
            cartons_removed = self.initial_row_cartons - current_visible
        else:
            cartons_removed = 0

        total_picked = self.total_pallet_picked + cartons_removed
        layer_counts = self.state_machine.get_layer_counts()

        cartons_by_state = {}
        for state in CartonState:
            count = len(self.state_machine.get_cartons_by_state(state))
            if count > 0:
                cartons_by_state[state.value] = count

        frame_time = (time.perf_counter() - start) * 1000.0

        return LiveCountResult(
            current_layer=self.current_layer,
            initial_row_cartons=self.initial_row_cartons,
            cartons_remaining=current_visible,
            cartons_removed_from_row=cartons_removed,
            total_active=current_visible,
            total_picked=total_picked,
            top_row_count=current_visible,
            layer_counts=layer_counts,
            cartons_by_state=cartons_by_state,
            hand_detected=len(hand_positions) > 0,
            picking_in_progress=any(
                t.state == CartonState.BEING_PICKED
                for t in self.state_machine.tracks.values()
            ),
            frame_time_ms=frame_time,
            tracks=[
                {
                    "id": t.track_id,
                    "bbox": t.bbox,
                    "state": t.state.value,
                    "row": t.row,
                    "layer": t.layer,
                }
                for t in active_cartons
            ],
            events=events,
            top_row_only=use_top_row,
            layer_transition_triggered=layer_transition,
            total_rows=total_rows,
        )

    def annotate_frame(
        self, frame: np.ndarray, result: LiveCountResult
    ) -> np.ndarray:
        """Draw clean, high-visibility annotations on frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw pallet ROI if configured
        if self.pallet_roi:
            cv2.rectangle(
                vis,
                (self.pallet_roi.x1, self.pallet_roi.y1),
                (self.pallet_roi.x2, self.pallet_roi.y2),
                (200, 200, 0), 2
            )

        # Draw detected Top Row cartons
        for idx, track in enumerate(result.tracks, start=1):
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
            state = track["state"]

            # Color scheme: Emerald green for top row present, bright orange for picking
            if state == "being_picked":
                box_color = (0, 165, 255) # Orange
                badge_text = f"PICKING ID:{track['id']}"
            else:
                box_color = (50, 220, 100) # Bright Emerald Green
                badge_text = f"L{result.current_layer} #{idx} (ID:{track['id']})"

            # Draw bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 3)

            # Draw label background badge
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = max(text_h + 10, y1)
            cv2.rectangle(vis, (x1, badge_y1), (x1 + text_w + 12, badge_y2), box_color, -1)
            cv2.putText(
                vis, badge_text, (x1 + 6, badge_y2 - 6),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
            )

        # Draw worker hands
        for hand in self.worker_detector.hand_history[-4:]:
            hx, hy = int(hand.position[0]), int(hand.position[1])
            cv2.circle(vis, (hx, hy), 14, (255, 50, 200), -1)
            cv2.circle(vis, (hx, hy), 18, (255, 255, 255), 2)
            cv2.putText(vis, "HAND", (hx + 12, hy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Draw top status banner HUD
        banner_h = 56
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.88, vis, 0.12, 0, vis)

        row_status = f"Active Layer: {result.current_layer}"
        rem_text = f"Remaining on Top: {result.cartons_remaining}"
        removed_text = f"Removed from Row: {result.cartons_removed_from_row}"
        total_text = f"Total Picked: {result.total_picked}"
        rows_text = f"Total Rows: {result.total_rows}"

        hud_line1 = f"{row_status}   |   {rem_text}   |   {removed_text}"
        hud_line2 = f"{total_text}   |   Initial Layer Cartons: {result.initial_row_cartons}"
        hud_line3 = f"{rows_text}"

        cv2.putText(vis, hud_line1, (16, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2, cv2.LINE_AA)
        cv2.putText(vis, hud_line2, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(vis, hud_line3, (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1, cv2.LINE_AA)

        return vis

    def reset(self):
        """Reset all counters and state."""
        self.tracker.reset()
        self.worker_detector.reset()
        self.state_machine.reset()
        self.frame_count = 0
        self.initial_cartons = None
        self.current_layer = 1
        self.initial_row_cartons = 0
        self.total_pallet_picked = 0
        self.layer_locked = False
        self.confirmed_initial_count = False
        self.stable_frames = 0
        self.last_visible_count = 0
        self.zero_count_frames = 0
        # Reset auto-count
        self.auto_count_done = False
        self.auto_count_frames = 0
        self.auto_count_stable = 0
        self.auto_count_last = 0
