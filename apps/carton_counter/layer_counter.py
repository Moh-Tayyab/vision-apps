"""Per-Layer Carton Counting Engine for Vertical Pan/Tilt Pallet Videos.

ALGORITHM IMPLEMENTATION:
1. Frame Extraction: Sample frames in strict temporal order (frame 0 = top of stack).
2. Detection: Run YOLO / Roboflow carton detector on each frame.
3. Vertical Normalization:
   - Estimate inter-frame vertical camera motion using median vertical displacement
     of high-IoU matched bounding boxes (IoU > 0.4 within search window).
   - Fallback to whole-image vertical phase correlation (cv2.phaseCorrelate with Hanning window)
     if insufficient box matches.
   - Accumulate camera offsets:
       camera_offset[0] = 0
       camera_offset[t+1] = camera_offset[t] + median_shift(t -> t+1)
   - Compute normalized vertical position for every detection:
       normalized_y = camera_offset[frame_idx] + (pixel_y_center - frame_height / 2)
4. Layer Clustering:
   - Sort normalized_y across all frames.
   - Compute consecutive gaps: diff(sorted_normalized_y).
   - Compute hybrid threshold:
       threshold = max(gap_multiplier * median_gap, 0.6 * median_box_height)
     where gap_multiplier defaults to 1.7 (configurable).
   - Split into clusters wherever gap > threshold (ordered from top to bottom).
5. Intra-Layer De-duplication:
   - Inside each layer cluster only, de-duplicate bounding boxes from adjacent/overlapping
     frames using shared coordinate IoU (threshold ~0.4-0.5) + temporal proximity.
   - Never de-duplicate across layers.
6. Final Count:
   - Sum of de-duplicated counts of every layer (never multiply rows x layers).
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from detector import CartonDetector, Detection, DetectionResult


@dataclass
class LayerDetection:
    """Detection with normalized global coordinates and frame origin."""
    bbox: List[float]  # [x1, y1, x2, y2] in original frame pixel coordinates
    confidence: float
    class_id: int
    class_name: str
    frame_idx: int
    normalized_y: float
    box_height: float
    box_width: float
    global_bbox: List[float] = field(default_factory=list)  # [x1, norm_y - h/2, x2, norm_y + h/2]
    layer_index: int = -1

    def to_dict(self) -> dict:
        return {
            "bbox": [round(c, 2) for c in self.bbox],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "frame_idx": self.frame_idx,
            "normalized_y": round(self.normalized_y, 2),
            "box_height": round(self.box_height, 2),
            "box_width": round(self.box_width, 2),
            "layer_index": self.layer_index,
        }


@dataclass
class LayerBreakdown:
    layer_index: int
    count: int
    normalized_y_range: List[float]
    detections: List[LayerDetection] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "count": self.count,
            "normalized_y_range": [
                round(self.normalized_y_range[0], 2),
                round(self.normalized_y_range[1], 2),
            ],
        }


@dataclass
class PanCountResult:
    total_count: int
    per_layer_breakdown: List[LayerBreakdown]
    gap_threshold_used: float
    gap_multiplier: float
    method: str = "per_layer_pan"
    annotated_frames: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    camera_offsets: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_count": self.total_count,
            "per_layer_breakdown": [layer.to_dict() for layer in self.per_layer_breakdown],
            "gap_threshold_used": round(self.gap_threshold_used, 2),
            "gap_multiplier": round(self.gap_multiplier, 2),
            "method": self.method,
            "annotated_frames": self.annotated_frames,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


def compute_box_iou(box_a: List[float], box_b: List[float]) -> float:
    """Standard 2D Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def estimate_inter_frame_vertical_shift(
    frame_prev: np.ndarray,
    frame_curr: np.ndarray,
    dets_prev: List[Detection],
    dets_curr: List[Detection],
    iou_match_threshold: float = 0.4,
    min_box_matches: int = 1,
) -> Tuple[float, str]:
    """Estimate the vertical downward camera displacement between frame t and frame t+1.

    When the camera moves downward (panning top to bottom):
    - Physical cartons in frame t+1 appear higher up in pixel coordinates:
      dy_pixel = y_curr - y_prev < 0.
    - Camera downward motion in the shared world coordinate system is:
      camera_displacement = -dy_pixel > 0.

    Strategy:
    1. Match boxes between frame t and t+1 having overlapping horizontal spans and similar aspect/size.
    2. Compute vertical alignment shift that maximizes IoU. If IoU > iou_match_threshold, record displacement.
    3. Return median displacement across matched boxes.
    4. If insufficient box matches, fallback to whole-image vertical phase correlation via cv2.phaseCorrelate.
    """
    matched_camera_displacements: List[float] = []

    if dets_prev and dets_curr:
        for p in dets_prev:
            p_w = p.bbox[2] - p.bbox[0]
            p_h = p.bbox[3] - p.bbox[1]
            p_yc = (p.bbox[1] + p.bbox[3]) / 2.0
            p_xc = (p.bbox[0] + p.bbox[2]) / 2.0

            best_iou = 0.0
            best_disp = 0.0

            for c in dets_curr:
                c_w = c.bbox[2] - c.bbox[0]
                c_h = c.bbox[3] - c.bbox[1]
                c_yc = (c.bbox[1] + c.bbox[3]) / 2.0
                c_xc = (c.bbox[0] + c.bbox[2]) / 2.0

                # Check horizontal alignment compatibility
                x_overlap = max(0.0, min(p.bbox[2], c.bbox[2]) - max(p.bbox[0], c.bbox[0]))
                min_w = min(p_w, c_w)
                if min_w <= 0 or (x_overlap / min_w) < 0.4:
                    continue

                # Check size similarity (within 2x height ratio)
                if p_h <= 0 or c_h <= 0 or not (0.5 <= (p_h / c_h) <= 2.0):
                    continue

                # Pixel vertical movement of carton: dy_pixel = c_yc - p_yc
                dy_pixel = c_yc - p_yc
                # Candidate shifted box for p to test IoU against c
                p_shifted = [p.bbox[0], p.bbox[1] + dy_pixel, p.bbox[2], p.bbox[3] + dy_pixel]
                iou = compute_box_iou(p_shifted, c.bbox)

                if iou > best_iou and iou >= iou_match_threshold:
                    best_iou = iou
                    best_disp = -dy_pixel  # camera downward displacement

            if best_iou >= iou_match_threshold:
                matched_camera_displacements.append(best_disp)

    if len(matched_camera_displacements) >= min_box_matches:
        median_disp = float(np.median(matched_camera_displacements))
        return median_disp, "box_matching"

    # Fallback: Whole-image vertical phase correlation with Hanning window
    h, w = frame_prev.shape[:2]
    gray_prev = cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_curr = cv2.cvtColor(frame_curr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    try:
        (shift_x, shift_y), response = cv2.phaseCorrelate(gray_prev, gray_curr, window)
        # shift_y is the pixel translation from gray_prev to gray_curr.
        # When camera moves down, image content moves up (shift_y < 0).
        # camera downward displacement is -shift_y.
        camera_downward_disp = -float(shift_y)
        return camera_downward_disp, f"phase_correlation (resp={response:.2f})"
    except Exception:
        return 0.0, "fallback_zero"


class PerLayerCartonCounter:
    """Per-layer carton counting engine for vertical pan pallet video inspection."""

    def __init__(
        self,
        detector: CartonDetector,
        default_gap_multiplier: float = 1.7,
        intra_layer_iou_threshold: float = 0.45,
    ):
        self.detector = detector
        self.default_gap_multiplier = default_gap_multiplier
        self.intra_layer_iou_threshold = intra_layer_iou_threshold

    def sample_frames_from_video(
        self,
        video_path: str,
        sample_interval_sec: float = 0.6,
        min_frames: int = 4,
        max_frames: int = 30,
    ) -> List[np.ndarray]:
        """Extract frames at regular time intervals in strict temporal order (top -> bottom)."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains no readable frames")

        frame_step = max(1, int(fps * sample_interval_sec))
        sampled_indices = list(range(0, total_frames, frame_step))

        # Clamp to bounds
        if len(sampled_indices) > max_frames:
            sampled_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
        elif len(sampled_indices) < min_frames and total_frames >= min_frames:
            sampled_indices = np.linspace(0, total_frames - 1, min_frames, dtype=int).tolist()

        # Remove potential duplicates
        sampled_indices = sorted(list(dict.fromkeys(sampled_indices)))

        frames = []
        for idx in sampled_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        return frames

    def count_pan(
        self,
        frames: List[np.ndarray],
        gap_multiplier: Optional[float] = None,
        confidence: Optional[float] = None,
        annotate: bool = True,
    ) -> PanCountResult:
        """Run the full 6-step Per-Layer Pan counting algorithm on a sequence of frames."""
        start_time = time.perf_counter()

        if not frames:
            return PanCountResult(
                total_count=0,
                per_layer_breakdown=[],
                gap_threshold_used=0.0,
                gap_multiplier=gap_multiplier or self.default_gap_multiplier,
                annotated_frames=[],
                processing_time_ms=0.0,
            )

        gap_mult = gap_multiplier if gap_multiplier is not None else self.default_gap_multiplier
        num_frames = len(frames)
        h_frame, w_frame = frames[0].shape[:2]

        # ── Step 2: Detection on each frame ──
        detections_per_frame: List[List[Detection]] = []
        for frame in frames:
            res = self.detector.detect(frame, confidence=confidence)
            detections_per_frame.append(res.detections)

        # ── Step 3: Normalization to Shared Vertical Coordinate ──
        camera_offsets: List[float] = [0.0] * num_frames
        for t in range(num_frames - 1):
            disp, method = estimate_inter_frame_vertical_shift(
                frame_prev=frames[t],
                frame_curr=frames[t + 1],
                dets_prev=detections_per_frame[t],
                dets_curr=detections_per_frame[t + 1],
            )
            camera_offsets[t + 1] = camera_offsets[t] + disp

        # Convert each raw detection to LayerDetection with shared normalized_y
        all_detections: List[LayerDetection] = []
        for t, frame_dets in enumerate(detections_per_frame):
            offset_t = camera_offsets[t]
            for d in frame_dets:
                x1, y1, x2, y2 = d.bbox
                box_h = y2 - y1
                box_w = x2 - x1
                pixel_yc = (y1 + y2) / 2.0
                norm_y = offset_t + (pixel_yc - h_frame / 2.0)
                global_box = [x1, norm_y - box_h / 2.0, x2, norm_y + box_h / 2.0]

                ld = LayerDetection(
                    bbox=[x1, y1, x2, y2],
                    confidence=d.confidence,
                    class_id=d.class_id,
                    class_name=d.class_name,
                    frame_idx=t,
                    normalized_y=norm_y,
                    box_height=box_h,
                    box_width=box_w,
                    global_bbox=global_box,
                )
                all_detections.append(ld)

        # ── Step 4: Layer Clustering via Hybrid Gap Threshold ──
        if not all_detections:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            annotated_b64: List[str] = []
            if annotate:
                annotated_b64 = self._generate_annotated_frames(
                    frames=frames,
                    all_detections=[],
                    per_layer_breakdown=[],
                    camera_offsets=camera_offsets,
                )
            return PanCountResult(
                total_count=0,
                per_layer_breakdown=[],
                gap_threshold_used=0.0,
                gap_multiplier=gap_mult,
                annotated_frames=annotated_b64,
                processing_time_ms=elapsed_ms,
                camera_offsets=camera_offsets,
            )

        # Sort all detections by normalized_y (top of pallet to bottom)
        all_detections.sort(key=lambda d: d.normalized_y)
        norm_y_vals = np.array([d.normalized_y for d in all_detections])
        box_heights = np.array([d.box_height for d in all_detections])

        median_box_height = float(np.median(box_heights)) if len(box_heights) > 0 else float(h_frame * 0.25)
        
        if len(norm_y_vals) > 1:
            gaps = np.diff(norm_y_vals)
            median_gap = float(np.median(gaps))
        else:
            gaps = np.array([], dtype=np.float32)
            median_gap = 0.0

        # Hybrid gap threshold formula: threshold = max(gap_multiplier * median_gap, 0.6 * median_box_height)
        threshold = max(gap_mult * median_gap, 0.6 * median_box_height)

        # Split into clusters wherever gap > threshold
        clusters: List[List[LayerDetection]] = []
        current_cluster: List[LayerDetection] = [all_detections[0]]

        for i in range(len(gaps)):
            gap = gaps[i]
            next_det = all_detections[i + 1]
            if gap > threshold:
                clusters.append(current_cluster)
                current_cluster = [next_det]
            else:
                current_cluster.append(next_det)
        if current_cluster:
            clusters.append(current_cluster)

        # Assign layer_index to each detection
        for layer_idx, cluster in enumerate(clusters):
            for d in cluster:
                d.layer_index = layer_idx

        # ── Step 5: Intra-Layer De-duplication ──
        per_layer_breakdown: List[LayerBreakdown] = []
        total_count = 0

        for layer_idx, cluster in enumerate(clusters):
            deduped_boxes = self._deduplicate_cluster(cluster)
            layer_count = len(deduped_boxes)
            total_count += layer_count

            y_min = float(min(d.normalized_y for d in cluster))
            y_max = float(max(d.normalized_y for d in cluster))

            per_layer_breakdown.append(
                LayerBreakdown(
                    layer_index=layer_idx,
                    count=layer_count,
                    normalized_y_range=[y_min, y_max],
                    detections=deduped_boxes,
                )
            )

        # ── Step 7: Annotated Frames Generation ──
        annotated_b64: List[str] = []
        if annotate:
            annotated_b64 = self._generate_annotated_frames(
                frames=frames,
                all_detections=all_detections,
                per_layer_breakdown=per_layer_breakdown,
                camera_offsets=camera_offsets,
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return PanCountResult(
            total_count=total_count,
            per_layer_breakdown=per_layer_breakdown,
            gap_threshold_used=threshold,
            gap_multiplier=gap_mult,
            method="per_layer_pan",
            annotated_frames=annotated_b64,
            processing_time_ms=elapsed_ms,
            camera_offsets=camera_offsets,
        )

    def _deduplicate_cluster(self, cluster: List[LayerDetection]) -> List[LayerDetection]:
        """De-duplicate detections within a single layer cluster across overlapping frames."""
        if not cluster:
            return []

        n = len(cluster)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for j in range(i + 1, n):
                det_a = cluster[i]
                det_b = cluster[j]

                # If from same frame with distinct boxes, compute standard IoU
                # If from different frames, compute IoU in global pallet coordinates
                iou = compute_box_iou(det_a.global_bbox, det_b.global_bbox)
                
                # Check horizontal coordinate alignment: same physical box has matching x1, x2
                x_inter = max(0.0, min(det_a.bbox[2], det_b.bbox[2]) - max(det_a.bbox[0], det_b.bbox[0]))
                x_union = max(det_a.bbox[2], det_b.bbox[2]) - min(det_a.bbox[0], det_b.bbox[0])
                x_iou = x_inter / x_union if x_union > 0 else 0.0

                if iou >= self.intra_layer_iou_threshold or (x_iou >= 0.65 and iou >= 0.30):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[rj] = ri

        grouped: dict[int, List[LayerDetection]] = {}
        for i in range(n):
            grouped.setdefault(find(i), []).append(cluster[i])

        deduped: List[LayerDetection] = []
        for group in grouped.values():
            # Pick the detection with the highest confidence
            best = max(group, key=lambda d: d.confidence)
            deduped.append(best)

        return deduped

    def _generate_annotated_frames(
        self,
        frames: List[np.ndarray],
        all_detections: List[LayerDetection],
        per_layer_breakdown: List[LayerBreakdown],
        camera_offsets: List[float],
    ) -> List[str]:
        """Draw horizontal layer boundary lines and bounding boxes onto each sampled frame."""
        annotated_b64: List[str] = []
        h_frame, w_frame = frames[0].shape[:2]

        # Colors for layer boundaries and bounding boxes
        layer_colors = [
            (34, 197, 94),   # Green
            (59, 130, 246),  # Blue
            (249, 115, 22),  # Orange
            (168, 85, 247),  # Purple
            (236, 72, 153),  # Pink
            (20, 184, 166),  # Teal
            (234, 179, 8),   # Yellow
        ]

        # Group detections by frame index
        dets_by_frame: dict[int, List[LayerDetection]] = {}
        for d in all_detections:
            dets_by_frame.setdefault(d.frame_idx, []).append(d)

        for t, raw_frame in enumerate(frames):
            vis = raw_frame.copy()
            offset_t = camera_offsets[t]

            # 1. Draw horizontal layer boundary bands and separator lines
            for layer in per_layer_breakdown:
                color = layer_colors[layer.layer_index % len(layer_colors)]
                y_min_norm, y_max_norm = layer.normalized_y_range

                # Convert normalized_y back to current frame's pixel coordinate:
                # normalized_y = camera_offset[t] + (pixel_y - h_frame / 2)
                # pixel_y = normalized_y - camera_offset[t] + h_frame / 2
                pixel_y_top = int(y_min_norm - offset_t + h_frame / 2.0)
                pixel_y_bot = int(y_max_norm - offset_t + h_frame / 2.0)

                # Draw horizontal guideline if visible in this frame
                if -50 <= pixel_y_top <= h_frame + 50 or -50 <= pixel_y_bot <= h_frame + 50:
                    mid_y = int((pixel_y_top + pixel_y_bot) / 2.0)
                    if 0 <= mid_y <= h_frame - 1:
                        # Draw horizontal layer center line
                        cv2.line(vis, (0, mid_y), (w_frame, mid_y), color, 1, cv2.LINE_AA)
                        tag = f"Layer {layer.layer_index} ({layer.count} cartons)"
                        cv2.putText(
                            vis,
                            tag,
                            (10, max(20, min(h_frame - 10, mid_y - 8))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

            # 2. Draw detections for this frame
            frame_dets = dets_by_frame.get(t, [])
            for d in frame_dets:
                color = layer_colors[d.layer_index % len(layer_colors)]
                x1, y1, x2, y2 = [int(v) for v in d.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_frame - 1, x2), min(h_frame - 1, y2)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                lbl = f"L{d.layer_index}: {d.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    vis,
                    lbl,
                    (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 3. Draw top summary banner
            cv2.rectangle(vis, (0, 0), (w_frame, 38), (15, 23, 42), -1)
            summary_txt = f"Frame {t+1}/{len(frames)} | Pallet Face: {len(all_detections)} raw dets | {len(per_layer_breakdown)} Layers"
            cv2.putText(
                vis,
                summary_txt,
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (74, 222, 128),
                2,
                cv2.LINE_AA,
            )

            # Encode to JPEG base64
            _, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            annotated_b64.append(f"data:image/jpeg;base64,{b64}")

        return annotated_b64
