from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple, Any
import time
import numpy as np
import cv2

from carton_preprocessor import CartonHoldingDetector


# Standard class taxonomies
PERSON_CLASSES = {"person", "worker", "human", "man", "woman"}
CARGO_CLASSES = {"carton", "box", "package", "suitcase", "backpack", "tote", "crate", "bag", "cardboard box"}


def compute_box_intersection(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    """Compute intersection area between two bounding boxes (x1, y1, x2, y2)."""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    return inter_w * inter_h


def compute_box_area(box: Tuple[float, float, float, float]) -> float:
    """Compute area of a bounding box."""
    w = max(0.0, box[2] - box[0])
    h = max(0.0, box[3] - box[1])
    return w * h


def is_cargo_held_by_person(
    cargo_box: Tuple[float, float, float, float],
    person_box: Tuple[float, float, float, float],
    ioca_threshold: float = 0.25,
) -> Tuple[bool, float]:
    """
    Check if a cargo box is held / carried by a person.
    
    Uses:
      1. IoCA (Intersection over Cargo Area): Area(Cargo ∩ Person) / Area(Cargo)
      2. Carrying Zone check: Cargo centroid lies within worker carrying region (torso / hands).
    
    Returns:
      (is_held, ioca_score)
    """
    cargo_area = compute_box_area(cargo_box)
    if cargo_area <= 0:
        return False, 0.0

    inter_area = compute_box_intersection(cargo_box, person_box)
    ioca = inter_area / cargo_area

    # Cargo centroid
    c_cx = (cargo_box[0] + cargo_box[2]) / 2.0
    c_cy = (cargo_box[1] + cargo_box[3]) / 2.0

    # Person carrying region (expand slightly horizontally, upper 85% of body for hands/chest)
    p_w = person_box[2] - person_box[0]
    p_h = person_box[3] - person_box[1]
    
    in_carrying_x = (person_box[0] - 0.20 * p_w) <= c_cx <= (person_box[2] + 0.20 * p_w)
    in_carrying_y = (person_box[1] + 0.10 * p_h) <= c_cy <= (person_box[3] + 0.05 * p_h)
    in_carrying_zone = in_carrying_x and in_carrying_y

    is_held = (ioca >= ioca_threshold) or (in_carrying_zone and ioca >= 0.12)
    return is_held, ioca


class FlightCartonTracker:
    """
    Zero-training Classical Computer Vision Engine for thrown / in-flight cartons passing
    through the vertical tripwire corridor. Tracks cardboard kraft & tape color trajectories.
    """
    def __init__(self, line_x: int, hysteresis: int = 15, loading_direction: str = "right_to_left", cooldown: int = 15):
        self.line_x = line_x
        self.hysteresis = hysteresis
        self.loading_direction = loading_direction.lower()
        self.cooldown = cooldown
        self.crossed_tracks: Set[int] = set()
        self.tracks: Dict[int, List[Tuple[int, int, int]]] = {}
        self.next_id = 500
        self.active_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        self.recent_worker_boxes: List[Tuple[Tuple[float, float, float, float], int]] = []

        # Cardboard HSV color bounds
        self.lower_brown = np.array([6, 25, 35])
        self.upper_brown = np.array([32, 255, 235])
        self.lower_tape = np.array([0, 0, 140])
        self.upper_tape = np.array([180, 50, 255])

    def set_line_x(self, line_x: int):
        self.line_x = line_x

    def set_loading_direction(self, direction: str):
        self.loading_direction = direction.lower()

    def reset(self):
        self.tracks.clear()
        self.active_boxes.clear()
        self.crossed_tracks.clear()
        self.recent_worker_boxes.clear()
        self.last_throw_f = -99
        self.next_id = 500

    def update(
        self,
        frame: np.ndarray,
        frame_idx: int,
        prev_gray: Optional[np.ndarray],
        worker_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[Tuple[int, Tuple[int, int, int, int], int, int, int]]:
        if prev_gray is None or frame is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Protect against video switches or resolution changes
        if prev_gray.shape != gray.shape:
            return []

        diff = cv2.absdiff(gray, prev_gray)
        _, motion_mask = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cmask = cv2.bitwise_or(cv2.inRange(hsv, self.lower_brown, self.upper_brown),
                               cv2.inRange(hsv, self.lower_tape, self.upper_tape))
        carton_motion = cv2.bitwise_and(motion_mask, cmask)

        # Mask out worker bodies (40px margin covers carried cartons extending beyond person bbox)
        if worker_boxes:
            for wx1, wy1, wx2, wy2 in worker_boxes:
                cv2.rectangle(
                    carton_motion,
                    (max(0, int(wx1) - 40), max(0, int(wy1) - 40)),
                    (min(frame.shape[1], int(wx2) + 40), min(frame.shape[0], int(wy2) + 40)),
                    0,
                    -1
                )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(carton_motion, cv2.MORPH_CLOSE, kernel)

        roi_y1 = max(30, int(frame.shape[0] * 0.12))
        roi_y2 = int(frame.shape[0] * 0.94)
        roi_x1 = max(0, self.line_x - 105)
        roi_x2 = min(frame.shape[1], self.line_x + 105)
        zone = closed[roi_y1:roi_y2, roi_x1:roi_x2]

        cnts, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        min_area = max(550, int(1000 * (frame.shape[1] / 720.0)))
        for c in cnts:
            area = cv2.contourArea(c)
            if area >= min_area:
                bx, by, bw, bh = cv2.boundingRect(c)
                cx = roi_x1 + bx + bw // 2
                cy = roi_y1 + by + bh // 2
                candidates.append((cx, cy, bx + roi_x1, by + roi_y1, bw, bh, area))

        # Distinct candidates filtering (keep at most 2 distinct cartons if center distance >= 60px)
        sorted_cands = sorted(candidates, key=lambda c: c[6], reverse=True)
        distinct_cands = []
        for cand in sorted_cands:
            cx, cy = cand[0], cand[1]
            is_frag = False
            for d in distinct_cands:
                if abs(cx - d[0]) < 60 and abs(cy - d[1]) < 60:
                    is_frag = True
                    break
            if not is_frag:
                distinct_cands.append(cand)
                if len(distinct_cands) >= 2:
                    break
        candidates = distinct_cands

        crossings = []
        self.active_boxes.clear()

        for cx, cy, x, y, bw, bh, a in candidates:
            matched = False
            for tid, pts in list(self.tracks.items()):
                last_x, last_y, last_f = pts[-1]
                if frame_idx - last_f <= 4 and abs(cx - last_x) < 95 and abs(cy - last_y) < 70:
                    pts.append((cx, cy, frame_idx))
                    matched = True
                    self.active_boxes[tid] = (x, y, x + bw, y + bh)
                    first_x = pts[0][0]
                    # Require minimum 5 frames of tracking before counting a flight crossing
                    if tid not in self.crossed_tracks and len(pts) >= 5:
                        if frame_idx - getattr(self, "last_throw_f", -99) >= 10:
                            throw_qty = 2 if (a >= 2500 or bh >= 110) else 1
                            if self.loading_direction == "right_to_left":
                                if first_x >= self.line_x - 5 and cx <= self.line_x - self.hysteresis:
                                    self.crossed_tracks.add(tid)
                                    self.last_throw_f = frame_idx
                                    crossings.append((tid, (x, y, x + bw, y + bh), cx, cy, throw_qty))
                            else:
                                if first_x <= self.line_x + 5 and cx >= self.line_x + self.hysteresis:
                                    self.crossed_tracks.add(tid)
                                    self.last_throw_f = frame_idx
                                    crossings.append((tid, (x, y, x + bw, y + bh), cx, cy, throw_qty))
                    break
            if not matched:
                self.tracks[self.next_id] = [(cx, cy, frame_idx)]
                self.active_boxes[self.next_id] = (x, y, x + bw, y + bh)
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if frame_idx - self.tracks[tid][-1][2] > 7:
                del self.tracks[tid]
                self.crossed_tracks.discard(tid)

        return crossings


class TripwireCounter:
    """
    Industrial-grade directional tripwire counter with Worker-Carton Association:
      - Prevents double-counting when workers carry cartons across the line
      - Spatial-temporal association (Holding Logic) linking workers to carried cargo
      - Carton Priority: only cargo increments the cargo count; associated workers are suppressed
      - Memory-bounded tracking and dead track purging for 24/7 stability
      - Trajectory displacement validation to prevent jitter oscillations
    """

    def __init__(
        self,
        line_x: int,
        hysteresis: int = 15,
        history_len: int = 30,
        cooldown_frames: int = 15,
        max_inactive_frames: int = 90,
        max_events: int = 500,
        countable_classes: Optional[Set[str]] = None,
        min_displacement_px: int = 15,
        holding_memory_frames: int = 25,
        suppression_window_frames: int = 30,
        loading_direction: str = "left_to_right",
    ):
        """
        Args:
            line_x: X-coordinate of the vertical virtual line.
            hysteresis: Pixel buffer around the line to prevent noise oscillation.
            history_len: Max past positions to maintain per track ID.
            cooldown_frames: Number of frames before the same track ID can trigger another crossing.
            max_inactive_frames: Frame inactivity threshold before purging dead track IDs.
            max_events: Maximum number of events to retain in memory history.
            countable_classes: Set of lowercase class names that increment counters. If None, default cargo classes are counted.
            min_displacement_px: Minimum trajectory travel distance required across the line.
            holding_memory_frames: Frames to remember worker-carton association through brief occlusions.
            suppression_window_frames: Frame window to suppress worker double-counting after carton crossing.
            loading_direction: Direction that counts as loading ("left_to_right" or "right_to_left").
        """
        self.line_x = line_x
        self.hysteresis = hysteresis
        self.history_len = history_len
        self.cooldown_frames = cooldown_frames
        self.max_inactive_frames = max_inactive_frames
        self.min_displacement_px = min_displacement_px
        self.holding_memory_frames = holding_memory_frames
        self.suppression_window_frames = suppression_window_frames
        self.loading_direction = loading_direction.lower()

        if countable_classes is not None:
            self.countable_classes = {c.lower() for c in countable_classes}
        else:
            self.countable_classes = CARGO_CLASSES

        # State tracking
        self.track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.history_len))
        self.track_side: Dict[int, str] = {}  # 'left' or 'right'
        self.last_cross_frame: Dict[int, int] = {}
        self.track_classes: Dict[int, str] = {}
        self.track_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        self.last_seen_frame: Dict[int, int] = {}

        # Worker-Carton Association & Holding States
        # cargo_id -> (person_id, last_associated_frame, ioca_score)
        self.cargo_to_person: Dict[int, Tuple[int, int, float]] = {}
        # person_id -> Dict[cargo_id, last_associated_frame]
        self.person_to_cargos: Dict[int, Dict[int, int]] = defaultdict(dict)
        # Suppressed track IDs (e.g. worker suppressed from double counting after carton crosses)
        self.suppressed_tracks: Dict[int, int] = {}  # track_id -> suppressed_until_frame

        # Active associations in the current frame: cargo_id -> person_id
        self.current_associations: Dict[int, int] = {}

        # Zero-training Classical CV Pre-processor (Carried Cartons)
        self.carton_detector = CartonHoldingDetector()
        self.track_holding_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=25))
        self.track_holding_qty: Dict[int, deque] = defaultdict(lambda: deque(maxlen=25))
        self.track_carton_boxes: Dict[int, Tuple[int, int, int, int]] = {}
        self.track_holding_scores: Dict[int, float] = {}
        self.current_worker_qty: Dict[int, int] = {}

        # Zero-training Classical CV Flight Tracker (Thrown / Air-passed Cartons)
        self.flight_tracker = FlightCartonTracker(
            line_x=self.line_x,
            hysteresis=self.hysteresis,
            loading_direction=self.loading_direction,
            cooldown=15,
        )
        self.prev_gray: Optional[np.ndarray] = None

        # Cargo Counters
        self.total_in = 0    # Loading into truck (+1)
        self.total_out = 0   # Returning away from truck (-1)
        self.worker_trips_in = 0   # Worker loading trips (+1)
        self.worker_trips_out = 0  # Worker return trips (-1)
        
        self.events: deque = deque(maxlen=max_events)
        self.recent_worker_crossings: List[Tuple[int, int]] = []

        # Active toast notification for UI
        self.recent_event: Optional[dict] = None
        self.recent_event_expiry: float = 0.0

    @property
    def net_count(self) -> int:
        return self.total_in - self.total_out

    @property
    def net_worker_trips(self) -> int:
        return self.worker_trips_in - self.worker_trips_out

    def set_line_x(self, new_x: int) -> None:
        """Update virtual line X coordinate interactively."""
        self.line_x = new_x
        self.flight_tracker.set_line_x(new_x)

    def set_loading_direction(self, direction: str) -> None:
        """Set primary loading direction: 'left_to_right' or 'right_to_left'."""
        self.loading_direction = direction.lower()
        self.flight_tracker.set_loading_direction(direction)

    def set_countable_classes(self, classes: Optional[Set[str]]) -> None:
        """Dynamically update countable classes filter."""
        if classes is not None:
            self.countable_classes = {c.lower() for c in classes}
        else:
            self.countable_classes = CARGO_CLASSES

    def reset_counts(self) -> None:
        """Reset all counters and history."""
        self.total_in = 0
        self.total_out = 0
        self.worker_trips_in = 0
        self.worker_trips_out = 0
        self.track_history.clear()
        self.track_side.clear()
        self.last_cross_frame.clear()
        self.track_classes.clear()
        self.track_boxes.clear()
        self.last_seen_frame.clear()
        self.cargo_to_person.clear()
        self.person_to_cargos.clear()
        self.suppressed_tracks.clear()
        self.current_associations.clear()
        self.track_holding_history.clear()
        self.track_holding_qty.clear()
        self.track_carton_boxes.clear()
        self.track_holding_scores.clear()
        self.current_worker_qty.clear()
        self.recent_worker_crossings.clear()
        self.flight_tracker.reset()
        self.prev_gray = None
        self.events.clear()
        self.recent_event = None

    def purge_inactive_tracks(self, current_frame_idx: int) -> int:
        """Removes dead tracks to prevent memory leaks in 24/7 continuous operation."""
        stale_ids = [
            tid for tid, last_f in self.last_seen_frame.items()
            if (current_frame_idx - last_f) > self.max_inactive_frames
        ]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.track_side.pop(tid, None)
            self.last_cross_frame.pop(tid, None)
            self.track_classes.pop(tid, None)
            self.track_boxes.pop(tid, None)
            self.last_seen_frame.pop(tid, None)
            self.cargo_to_person.pop(tid, None)
            self.person_to_cargos.pop(tid, None)
            self.suppressed_tracks.pop(tid, None)
            self.current_associations.pop(tid, None)
            self.track_holding_history.pop(tid, None)
            self.track_holding_qty.pop(tid, None)
            self.track_carton_boxes.pop(tid, None)
            self.track_holding_scores.pop(tid, None)
            self.current_worker_qty.pop(tid, None)

        # Cleanup stale associations
        for p_id in list(self.person_to_cargos.keys()):
            self.person_to_cargos[p_id] = {
                c_id: f for c_id, f in self.person_to_cargos[p_id].items()
                if (current_frame_idx - f) <= self.holding_memory_frames
            }
            if not self.person_to_cargos[p_id]:
                self.person_to_cargos.pop(p_id, None)

        for c_id, (p_id, f, _) in list(self.cargo_to_person.items()):
            if (current_frame_idx - f) > self.holding_memory_frames:
                self.cargo_to_person.pop(c_id, None)

        return len(stale_ids)

    def is_cargo_class(self, class_name: str) -> bool:
        """Check if class is a cargo item (carton, box, package, etc.)."""
        c = class_name.lower()
        return any(cargo_key in c for cargo_key in CARGO_CLASSES)

    def is_person_class(self, class_name: str) -> bool:
        """Check if class is a person / worker."""
        c = class_name.lower()
        return any(p_key in c for p_key in PERSON_CLASSES)

    def is_countable(self, class_name: str) -> bool:
        """Check if detected class is allowed to increment cargo counters."""
        if self.countable_classes is None:
            return True
        c = class_name.lower()
        return any(item in c for item in self.countable_classes)

    def _update_associations(
        self,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float]],
        frame_idx: int,
    ) -> None:
        """
        Calculates spatial overlap (IoCA) between detected workers and cartons in the current frame,
        and updates temporal holding memory.
        """
        self.current_associations.clear()

        # Partition active detections into persons and cargo
        persons = []
        cargos = []

        for track_id, box, cls_name, conf in tracked_objects:
            if self.is_person_class(cls_name):
                persons.append((track_id, box, conf))
            elif self.is_cargo_class(cls_name) or self.is_countable(cls_name):
                cargos.append((track_id, box, conf))

        # Perform bipartite matching based on IoCA & proximity
        for c_id, c_box, c_conf in cargos:
            best_person_id = None
            best_ioca = 0.0

            for p_id, p_box, p_conf in persons:
                is_held, ioca = is_cargo_held_by_person(c_box, p_box)
                if is_held and ioca > best_ioca:
                    best_ioca = ioca
                    best_person_id = p_id

            if best_person_id is not None:
                self.current_associations[c_id] = best_person_id
                self.cargo_to_person[c_id] = (best_person_id, frame_idx, best_ioca)
                self.person_to_cargos[best_person_id][c_id] = frame_idx
            else:
                # Check recent memory (within holding_memory_frames)
                if c_id in self.cargo_to_person:
                    prev_p_id, prev_f, prev_ioca = self.cargo_to_person[c_id]
                    if (frame_idx - prev_f) <= self.holding_memory_frames:
                        self.current_associations[c_id] = prev_p_id

    def get_associated_person(self, cargo_id: int, frame_idx: int) -> Optional[int]:
        """Returns track ID of worker holding this cargo, if any."""
        if cargo_id in self.current_associations:
            return self.current_associations[cargo_id]
        if cargo_id in self.cargo_to_person:
            p_id, f, _ = self.cargo_to_person[cargo_id]
            if (frame_idx - f) <= self.holding_memory_frames:
                return p_id
        return None

    def get_held_cargos(self, person_id: int, frame_idx: int) -> List[int]:
        """Returns list of cargo track IDs held by this worker."""
        held = []
        if person_id in self.person_to_cargos:
            for c_id, f in self.person_to_cargos[person_id].items():
                if (frame_idx - f) <= self.holding_memory_frames:
                    held.append(c_id)
        return held

    def update(
        self,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float]],
        frame_idx: int,
        frame: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Update tracker states with detections in the current frame and detect line crossings
        with Worker-Carton Association & Anti-Double-Counting.

        Args:
            tracked_objects: List of tuples (track_id, (x1, y1, x2, y2), class_name, confidence)
            frame_idx: Current video frame index.

        Returns:
            List of crossing event dicts triggered in this frame.
        """
        current_frame_events = []
        active_ids = set()

        # 1. Update basic states & positions
        for track_id, (x1, y1, x2, y2), cls_name, conf in tracked_objects:
            active_ids.add(track_id)
            self.track_classes[track_id] = cls_name
            self.track_boxes[track_id] = (x1, y1, x2, y2)
            self.last_seen_frame[track_id] = frame_idx
            
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            history = self.track_history[track_id]
            history.append((center_x, center_y))

            # Initial side assignment (direction-aware assignment avoiding hysteresis dead-zone traps)
            if track_id not in self.track_side:
                if self.loading_direction == "left_to_right":
                    self.track_side[track_id] = "right" if center_x >= (self.line_x + self.hysteresis) else "left"
                else:
                    self.track_side[track_id] = "left" if center_x <= (self.line_x - self.hysteresis) else "right"

            # Track carried boxes from true object detections (no noisy HSV brown mask)
            if self.is_person_class(cls_name):
                held_cargos = self.get_held_cargos(track_id, frame_idx)
                if held_cargos:
                    # Associate first held cargo's box with worker for visualization
                    c_id = held_cargos[0]
                    if c_id in self.track_boxes:
                        cb = self.track_boxes[c_id]
                        self.track_carton_boxes[track_id] = (int(cb[0]), int(cb[1]), int(cb[2]), int(cb[3]))
                        self.current_worker_qty[track_id] = min(len(held_cargos), 2)
                else:
                    self.track_carton_boxes.pop(track_id, None)
                    self.current_worker_qty[track_id] = 0

        # 2. Update Worker-Carton Holding Associations
        self._update_associations(tracked_objects, frame_idx)

        # 3. Detect Tripwire Crossings with Carton Priority & Worker Suppression
        # Process CARGO first so any associated workers are immediately suppressed
        sorted_tracks = sorted(
            tracked_objects,
            key=lambda item: 0 if (self.is_cargo_class(item[2]) or self.is_countable(item[2])) else 1
        )

        for track_id, (x1, y1, x2, y2), cls_name, conf in sorted_tracks:
            if track_id not in self.track_side:
                continue

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            history = self.track_history[track_id]
            last_side = self.track_side[track_id]
            last_frame = self.last_cross_frame.get(track_id, -self.cooldown_frames)

            # Check cooldown
            if frame_idx - last_frame < self.cooldown_frames:
                continue

            # Trajectory displacement validation
            if len(history) >= 2:
                start_x = history[0][0]
                displacement = center_x - start_x
            else:
                displacement = 0

            is_cargo = self.is_cargo_class(cls_name)
            is_person = self.is_person_class(cls_name)

            # Determine if this movement is in the loading direction
            moved_left_to_right = (last_side == "left" and center_x >= self.line_x + self.hysteresis)
            moved_right_to_left = (last_side == "right" and center_x <= self.line_x - self.hysteresis)

            if not (moved_left_to_right or moved_right_to_left):
                continue

            if moved_left_to_right:
                valid_motion = (displacement >= self.min_displacement_px or len(history) < 2)
                new_side = "right"
                is_loading = (self.loading_direction == "left_to_right")
            else:
                valid_motion = (displacement <= -self.min_displacement_px or len(history) < 2)
                new_side = "left"
                is_loading = (self.loading_direction == "right_to_left")

            if not valid_motion:
                continue

            self.track_side[track_id] = new_side
            self.last_cross_frame[track_id] = frame_idx

            # Case A: CARGO item crossed line (CARTON PRIORITY)
            if is_cargo or (self.is_countable(cls_name) and not is_person):
                associated_worker = self.get_associated_person(track_id, frame_idx)

                # PROXIMITY SUPPRESSION: If cargo is near a person (likely misclassified worker or carried carton)
                # that is NOT associated via IoCA, check if any active person is within 40px of this cargo center.
                # If so, suppress this cargo from independent counting - it will be counted via the person path.
                if associated_worker is None:
                    for p_id, p_box, p_cls, p_conf in tracked_objects:
                        if not self.is_person_class(p_cls):
                            continue
                        px1, py1, px2, py2 = p_box
                        p_cx = (px1 + px2) / 2.0
                        p_cy = (py1 + py2) / 2.0
                        dist = ((center_x - p_cx) ** 2 + (center_y - p_cy) ** 2) ** 0.5
                        # If cargo centroid is within person bounding box or very close
                        in_box = (px1 <= center_x <= px2) and (py1 <= center_y <= py2)
                        if in_box or dist < 40:
                            associated_worker = p_id
                            self.suppressed_tracks[track_id] = frame_idx + self.suppression_window_frames
                            break

                # Suppress associated worker from double counting
                if associated_worker is not None:
                    self.suppressed_tracks[associated_worker] = frame_idx + self.suppression_window_frames

                if is_loading:
                    self.total_in += 1
                    delta = +1
                    direction_label = "IN"
                    note_str = f"Carton loaded (Worker #{associated_worker})" if associated_worker else "Carton loaded"
                else:
                    self.total_out += 1
                    delta = -1
                    direction_label = "OUT"
                    note_str = f"Carton returned (Worker #{associated_worker})" if associated_worker else "Carton returned"

                event = {
                    "frame": frame_idx,
                    "timestamp": time.time(),
                    "track_id": track_id,
                    "class_name": cls_name,
                    "is_cargo": True,
                    "carried_by": associated_worker,
                    "direction": direction_label,
                    "delta": delta,
                    "position": (center_x, center_y),
                    "total_in": self.total_in,
                    "total_out": self.total_out,
                    "net_count": self.net_count,
                    "note": note_str,
                }
                self.events.append(event)
                current_frame_events.append(event)
                self.recent_event = event
                self.recent_event_expiry = time.time() + 2.0

            # Case B: PERSON crossed line (WORKER CHECK)
            elif is_person:
                if is_loading:
                    self.worker_trips_in += 1
                else:
                    self.worker_trips_out += 1

                is_suppressed = frame_idx < self.suppressed_tracks.get(track_id, 0)
                held_cargos = self.get_held_cargos(track_id, frame_idx)
                uncounted_cargos = [cid for cid in held_cargos if cid not in self.suppressed_tracks]
                is_carrying_carton = len(uncounted_cargos) > 0

                # If no YOLO cargo detected, use classical CV CartonHoldingDetector
                cv_carton_qty = 0
                if not is_carrying_carton and frame is not None and is_loading:
                    try:
                        direction_str = self.loading_direction
                        is_holding, confidence, carton_box, carton_qty, debug_info = self.carton_detector.detect_carton(
                            frame, (x1, y1, x2, y2), direction=direction_str
                        )
                        if is_holding and confidence >= 0.28:
                            cv_carton_qty = min(carton_qty, 2)
                            self.track_holding_history[track_id].append((frame_idx, confidence))
                            self.track_holding_qty[track_id].append(carton_qty)
                            if carton_box:
                                self.track_carton_boxes[track_id] = carton_box
                            self.track_holding_scores[track_id] = confidence
                    except Exception:
                        pass

                if is_suppressed:
                    # Associated carton was already counted in this crossing window
                    pass
                elif is_loading:
                    self.recent_worker_crossings.append((center_y, frame_idx))
                    self.recent_worker_crossings = [
                        (cy, f) for (cy, f) in self.recent_worker_crossings
                        if (frame_idx - f) <= 60
                    ]
                    if is_carrying_carton:
                        # Worker crossed with verified held carton(s) via YOLO association
                        cargo_qty = min(len(uncounted_cargos), 2)
                        self.total_in += cargo_qty
                        for cid in uncounted_cargos[:cargo_qty]:
                            self.suppressed_tracks[cid] = frame_idx + self.suppression_window_frames

                        qty_str = f"{cargo_qty} Cartons" if cargo_qty > 1 else "Carton"
                        note_suffix = " [DOUBLE PICKUP]" if cargo_qty > 1 else ""
                        event = {
                            "frame": frame_idx,
                            "timestamp": time.time(),
                            "track_id": track_id,
                            "class_name": "worker",
                            "is_cargo": True,
                            "carried_by": track_id,
                            "direction": "IN",
                            "delta": cargo_qty,
                            "position": (center_x, center_y),
                            "total_in": self.total_in,
                            "total_out": self.total_out,
                            "net_count": self.net_count,
                            "note": f"{qty_str} loaded by Worker #{track_id} (YOLO verified{note_suffix})",
                        }
                        self.events.append(event)
                        current_frame_events.append(event)
                        self.recent_event = event
                        self.recent_event_expiry = time.time() + 2.0
                    elif cv_carton_qty > 0:
                        # Worker crossed with carton detected via classical CV
                        self.total_in += cv_carton_qty
                        qty_str = f"{cv_carton_qty} Cartons" if cv_carton_qty > 1 else "Carton"
                        note_suffix = " [DOUBLE PICKUP]" if cv_carton_qty > 1 else ""
                        event = {
                            "frame": frame_idx,
                            "timestamp": time.time(),
                            "track_id": track_id,
                            "class_name": "worker",
                            "is_cargo": True,
                            "carried_by": track_id,
                            "direction": "IN",
                            "delta": cv_carton_qty,
                            "position": (center_x, center_y),
                            "total_in": self.total_in,
                            "total_out": self.total_out,
                            "net_count": self.net_count,
                            "note": f"{qty_str} loaded by Worker #{track_id} (CV detected{note_suffix})",
                        }
                        self.events.append(event)
                        current_frame_events.append(event)
                        self.recent_event = event
                        self.recent_event_expiry = time.time() + 2.0
                    else:
                        # Empty-handed worker entered truck - log entry without incrementing cargo count
                        event = {
                            "frame": frame_idx,
                            "timestamp": time.time(),
                            "track_id": track_id,
                            "class_name": "worker",
                            "is_cargo": False,
                            "carried_by": None,
                            "direction": "IN",
                            "delta": 0,
                            "position": (center_x, center_y),
                            "total_in": self.total_in,
                            "total_out": self.total_out,
                            "net_count": self.net_count,
                            "note": f"Worker #{track_id} entered truck (empty-handed)",
                        }
                        self.events.append(event)
                        current_frame_events.append(event)
                else:
                    # Worker returning away from truck (empty-handed return does not decrement cargo)
                    if "person" in (self.countable_classes or set()):
                        self.total_out += 1
                        event = {
                            "frame": frame_idx,
                            "timestamp": time.time(),
                            "track_id": track_id,
                            "class_name": "worker",
                            "is_cargo": False,
                            "carried_by": None,
                            "direction": "OUT",
                            "delta": -1,
                            "position": (center_x, center_y),
                            "total_in": self.total_in,
                            "total_out": self.total_out,
                            "net_count": self.net_count,
                            "note": f"Worker #{track_id} returned (empty-handed)",
                        }
                        self.events.append(event)
                        current_frame_events.append(event)

        # 4. Detect Thrown / In-Flight Cartons passing through tripwire corridor
        if frame is not None:
            worker_boxes = [
                box for tid, box, cls_name, _ in tracked_objects
                if self.is_person_class(cls_name)
            ]
            # Update prev_gray for next frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flight_crossings = []
            if self.prev_gray is not None:
                # Re-enable flight tracker for thrown cartons
                flight_crossings = self.flight_tracker.update(
                    frame, frame_idx, self.prev_gray, worker_boxes
                )
                for fc_tid, fc_box, fc_cx, fc_cy, fc_qty in flight_crossings:
                    # PROXIMITY SUPPRESSION: If flight carton is near any detected person OR
                    # near any person seen in the last 30 frames, skip it.
                    near_person = False
                    for p_tid, p_box, p_cls, p_conf in tracked_objects:
                        if not self.is_person_class(p_cls):
                            continue
                        px1, py1, px2, py2 = p_box
                        p_cx = (px1 + px2) / 2.0
                        p_cy = (py1 + py2) / 2.0
                        dist = ((fc_cx - p_cx) ** 2 + (fc_cy - p_cy) ** 2) ** 0.5
                        if dist < 100:
                            near_person = True
                            break
                    # Also check if any person track was recently seen nearby
                    if not near_person:
                        for p_tid, p_cls in self.track_classes.items():
                            if not self.is_person_class(p_cls):
                                continue
                            if p_tid not in self.track_boxes:
                                continue
                            last_f = self.last_seen_frame.get(p_tid, 0)
                            if (frame_idx - last_f) > 30:
                                continue
                            pb = self.track_boxes[p_tid]
                            pcx = (pb[0] + pb[2]) / 2.0
                            pcy = (pb[1] + pb[3]) / 2.0
                            dist = ((fc_cx - pcx) ** 2 + (fc_cy - pcy) ** 2) ** 0.5
                            if dist < 150:
                                near_person = True
                                break
                    # Also suppress if any person was counted in loading direction in last 45 frames
                    if not near_person:
                        for p_tid2, p_f in self.recent_worker_crossings:
                            if (frame_idx - p_f) <= 45:
                                near_person = True
                                break
                    if near_person:
                        continue

                    # Check direction validity for flight crossings
                    is_loading_flight = False
                    if self.loading_direction == "right_to_left":
                        is_loading_flight = (fc_cx <= self.line_x - self.hysteresis)
                    else:
                        is_loading_flight = (fc_cx >= self.line_x + self.hysteresis)

                    if is_loading_flight:
                        self.total_in += fc_qty
                        event = {
                            "frame": frame_idx,
                            "timestamp": time.time(),
                            "track_id": fc_tid,
                            "class_name": "flight_carton",
                            "is_cargo": True,
                            "carried_by": None,
                            "direction": "IN",
                            "delta": fc_qty,
                            "position": (fc_cx, fc_cy),
                            "total_in": self.total_in,
                            "total_out": self.total_out,
                            "net_count": self.net_count,
                            "note": f"Flight carton x{fc_qty} loaded" if fc_qty > 1 else "Flight carton loaded",
                        }
                        self.events.append(event)
                        current_frame_events.append(event)
                        self.recent_event = event
                        self.recent_event_expiry = time.time() + 2.0
            self.prev_gray = gray

        # Periodic cleanup of dead tracks
        if frame_idx % 30 == 0:
            self.purge_inactive_tracks(frame_idx)

        return current_frame_events
