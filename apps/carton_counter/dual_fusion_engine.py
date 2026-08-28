"""Dual-Camera Layer-Wise Carton Counting Fusion Engine.

ALGORITHM:
1. Receives Front (Camera 1) and Side (Camera 2) images / detections.
2. Identifies horizontal pallet layers (top-to-bottom) on each camera view by clustering
   bounding boxes along the vertical (Y) axis.
3. Aligns layers between Front and Side views.
4. For each layer k:
     - N1_k = number of cartons visible on Front face at layer k
     - N2_k = number of cartons visible on Side face at layer k
     - Layer Total = N1_k * N2_k
5. Pallet Total Count = Sum_k (N1_k * N2_k)
6. Generates synchronized annotated visual frames with layer boundary bands and labels.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from detector import CartonDetector, Detection, DetectionResult


@dataclass
class LayerInfo:
    """Represents a single horizontal pallet layer with Front & Side carton counts."""
    layer_index: int
    front_count: int
    side_count: int
    layer_total: int  # N1_k * N2_k
    front_bboxes: List[List[float]] = field(default_factory=list)
    side_bboxes: List[List[float]] = field(default_factory=list)
    y_range_front: List[float] = field(default_factory=list)
    y_range_side: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "layer_index": self.layer_index,
            "front_count": self.front_count,
            "side_count": self.side_count,
            "layer_total": self.layer_total,
            "y_range_front": [round(y, 1) for y in self.y_range_front] if self.y_range_front else [],
            "y_range_side": [round(y, 1) for y in self.y_range_side] if self.y_range_side else [],
        }


@dataclass
class DualFusionResult:
    """Result of dual-camera layer-wise fusion."""
    total_count: int
    layers_count: int
    layers: List[LayerInfo]
    front_count_raw: int
    side_count_raw: int
    front_annotated_b64: Optional[str] = None
    side_annotated_b64: Optional[str] = None
    processing_time_ms: float = 0.0
    method: str = "dual_layer_multiplication"

    def to_dict(self) -> dict:
        res = {
            "total_count": self.total_count,
            "layers_count": self.layers_count,
            "layers": [layer.to_dict() for layer in self.layers],
            "front_count_raw": self.front_count_raw,
            "side_count_raw": self.side_count_raw,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "method": self.method,
        }
        if self.front_annotated_b64:
            res["front_annotated_base64"] = self.front_annotated_b64
        if self.side_annotated_b64:
            res["side_annotated_base64"] = self.side_annotated_b64
        return res


# High-contrast colors for layer visualization
LAYER_COLORS = [
    (34, 197, 94),   # Green
    (59, 130, 246),  # Blue
    (249, 115, 22),  # Orange
    (168, 85, 247),  # Purple
    (236, 72, 153),  # Pink
    (20, 184, 166),  # Teal
    (234, 179, 8),   # Yellow
    (14, 165, 233),  # Sky Blue
]


class DualFusionEngine:
    """Performs layer-wise 3D pallet carton counting from Front and Side views."""

    def __init__(self, detector: Optional[CartonDetector] = None):
        self.detector = detector

    @staticmethod
    def cluster_layers_from_detections(
        detections: List[Detection],
        frame_height: int,
        vertical_overlap_threshold: float = 0.30,
    ) -> List[List[Detection]]:
        """Cluster 2D bounding boxes into distinct horizontal layers from top to bottom.

        Args:
            detections: List of carton detections on a single view.
            frame_height: Height of the image frame in pixels.
            vertical_overlap_threshold: Minimum overlap ratio on Y-axis to consider same layer.

        Returns:
            List of clusters (each cluster is a List[Detection]), ordered top to bottom.
        """
        if not detections:
            return []

        # Calculate bounding box metrics
        boxes_with_metrics = []
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            yc = (y1 + y2) / 2.0
            h = max(1.0, y2 - y1)
            boxes_with_metrics.append((yc, h, y1, y2, d))

        # Sort detections primarily by vertical centroid (top of pallet to bottom)
        boxes_with_metrics.sort(key=lambda item: item[0])

        heights = [item[1] for item in boxes_with_metrics]
        median_h = float(np.median(heights)) if heights else frame_height * 0.25

        clusters: List[List[Detection]] = []
        current_cluster: List[Detection] = [boxes_with_metrics[0][4]]
        current_cluster_y_span = [boxes_with_metrics[0][2], boxes_with_metrics[0][3]]  # [min_y1, max_y2]

        for i in range(1, len(boxes_with_metrics)):
            yc, h, y1, y2, d = boxes_with_metrics[i]
            prev_min_y, prev_max_y = current_cluster_y_span
            prev_yc = (prev_min_y + prev_max_y) / 2.0

            # Vertical overlap between this box and the current layer cluster
            overlap = max(0.0, min(prev_max_y, y2) - max(prev_min_y, y1))
            min_h = min(prev_max_y - prev_min_y, h)
            overlap_ratio = overlap / min_h if min_h > 0 else 0.0

            # Check if this box belongs to the current horizontal layer
            # Condition 1: Substantial vertical overlap
            # Condition 2: Centroids are very close vertically (< 0.55 * median box height)
            if overlap_ratio >= vertical_overlap_threshold or abs(yc - prev_yc) < 0.55 * median_h:
                current_cluster.append(d)
                current_cluster_y_span[0] = min(current_cluster_y_span[0], y1)
                current_cluster_y_span[1] = max(current_cluster_y_span[1], y2)
            else:
                # Start new layer cluster
                clusters.append(current_cluster)
                current_cluster = [d]
                current_cluster_y_span = [y1, y2]

        if current_cluster:
            clusters.append(current_cluster)

        # Sort boxes within each layer from left to right (X-axis)
        for cluster in clusters:
            cluster.sort(key=lambda d: d.bbox[0])

        return clusters

    @classmethod
    def align_and_multiply_layers(
        cls,
        front_clusters: List[List[Detection]],
        side_clusters: List[List[Detection]],
        h_front: int,
        h_side: int,
    ) -> Tuple[List[LayerInfo], int]:
        """Align horizontal layers between Front and Side views and compute N1_k * N2_k.

        Returns:
            Tuple of (List[LayerInfo], total_count)
        """
        l1 = len(front_clusters)
        l2 = len(side_clusters)

        # Edge case: No detections in both
        if l1 == 0 and l2 == 0:
            return [], 0

        # Edge case: Only Front has detections
        if l1 > 0 and l2 == 0:
            layers = []
            total = 0
            for idx, fc in enumerate(front_clusters):
                cnt = len(fc)
                y1 = min(d.bbox[1] for d in fc)
                y2 = max(d.bbox[3] for d in fc)
                layers.append(
                    LayerInfo(
                        layer_index=idx + 1,
                        front_count=cnt,
                        side_count=1,
                        layer_total=cnt,
                        front_bboxes=[d.bbox for d in fc],
                        side_bboxes=[],
                        y_range_front=[y1, y2],
                        y_range_side=[],
                    )
                )
                total += cnt
            return layers, total

        # Edge case: Only Side has detections
        if l1 == 0 and l2 > 0:
            layers = []
            total = 0
            for idx, sc in enumerate(side_clusters):
                cnt = len(sc)
                y1 = min(d.bbox[1] for d in sc)
                y2 = max(d.bbox[3] for d in sc)
                layers.append(
                    LayerInfo(
                        layer_index=idx + 1,
                        front_count=1,
                        side_count=cnt,
                        layer_total=cnt,
                        front_bboxes=[],
                        side_bboxes=[d.bbox for d in sc],
                        y_range_front=[],
                        y_range_side=[y1, y2],
                    )
                )
                total += cnt
            return layers, total

        # Standard Case: Equal number of layers detected on both views
        if l1 == l2:
            layers = []
            total = 0
            for idx in range(l1):
                fc = front_clusters[idx]
                sc = side_clusters[idx]
                n1 = len(fc)
                n2 = len(sc)
                layer_total = n1 * n2
                y_front = [min(d.bbox[1] for d in fc), max(d.bbox[3] for d in fc)]
                y_side = [min(d.bbox[1] for d in sc), max(d.bbox[3] for d in sc)]

                layers.append(
                    LayerInfo(
                        layer_index=idx + 1,
                        front_count=n1,
                        side_count=n2,
                        layer_total=layer_total,
                        front_bboxes=[d.bbox for d in fc],
                        side_bboxes=[d.bbox for d in sc],
                        y_range_front=y_front,
                        y_range_side=y_side,
                    )
                )
                total += layer_total
            return layers, total

        # Unequal Layer Counts: Robust Vertical Alignment
        # e.g., Front sees 4 layers, Side sees 3 layers due to camera pitch or slight occlusion
        # Match each layer by relative normalized vertical height from top of pallet
        num_layers = max(l1, l2)
        layers = []
        total = 0

        # Compute median column counts to use as fallback for missing layer in a view
        front_counts = [len(c) for c in front_clusters]
        side_counts = [len(c) for c in side_clusters]
        median_n1 = int(round(np.median(front_counts))) if front_counts else 1
        median_n2 = int(round(np.median(side_counts))) if side_counts else 1

        for idx in range(num_layers):
            fc = front_clusters[idx] if idx < l1 else []
            sc = side_clusters[idx] if idx < l2 else []

            n1 = len(fc) if fc else median_n1
            n2 = len(sc) if sc else median_n2
            layer_total = n1 * n2

            y_front = [min(d.bbox[1] for d in fc), max(d.bbox[3] for d in fc)] if fc else []
            y_side = [min(d.bbox[1] for d in sc), max(d.bbox[3] for d in sc)] if sc else []

            layers.append(
                LayerInfo(
                    layer_index=idx + 1,
                    front_count=n1,
                    side_count=n2,
                    layer_total=layer_total,
                    front_bboxes=[d.bbox for d in fc] if fc else [],
                    side_bboxes=[d.bbox for d in sc] if sc else [],
                    y_range_front=y_front,
                    y_range_side=y_side,
                )
            )
            total += layer_total

        return layers, total

    @classmethod
    def draw_annotated_frame(
        cls,
        image: np.ndarray,
        layer_clusters: List[List[Detection]],
        layers_info: List[LayerInfo],
        view_name: str = "FRONT (CAM 1)",
        total_count: int = 0,
    ) -> np.ndarray:
        """Draw layer guidelines, bounding boxes, and top summary banner."""
        vis = image.copy()
        h, w = vis.shape[:2]

        # Draw detections and layer separator lines
        for idx, cluster in enumerate(layer_clusters):
            color = LAYER_COLORS[idx % len(LAYER_COLORS)]
            layer_num = idx + 1

            # Get layer count info
            l_info = layers_info[idx] if idx < len(layers_info) else None
            n1 = l_info.front_count if l_info else len(cluster)
            n2 = l_info.side_count if l_info else 1
            l_total = l_info.layer_total if l_info else len(cluster)

            # Draw horizontal guideline at layer center
            cluster_y1 = min(d.bbox[1] for d in cluster)
            cluster_y2 = max(d.bbox[3] for d in cluster)
            mid_y = int((cluster_y1 + cluster_y2) / 2.0)

            if 0 <= mid_y < h:
                cv2.line(vis, (0, mid_y), (w, mid_y), color, 1, cv2.LINE_AA)
                layer_label = f"Layer {layer_num}: {len(cluster)} visible ({n1}x{n2} = {l_total})"
                cv2.putText(
                    vis,
                    layer_label,
                    (10, max(22, min(h - 10, mid_y - 8))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Draw carton bounding boxes
            for col_idx, d in enumerate(cluster):
                x1, y1, x2, y2 = [int(v) for v in d.bbox]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                box_lbl = f"L{layer_num}-{col_idx+1}: {d.confidence:.2f}"
                (tw, th), _ = cv2.getTextSize(box_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    vis,
                    box_lbl,
                    (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Draw Top Summary Header
        cv2.rectangle(vis, (0, 0), (w, 42), (15, 23, 42), -1)
        summary_txt = f"{view_name} | {len(layer_clusters)} Layers | TOTAL PALLET: {total_count}"
        cv2.putText(
            vis,
            summary_txt,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (74, 222, 128),
            2,
            cv2.LINE_AA,
        )

        return vis

    def fuse(
        self,
        front_image: np.ndarray,
        side_image: np.ndarray,
        confidence: Optional[float] = None,
        annotate: bool = True,
    ) -> DualFusionResult:
        """Run full dual-camera layer-wise counting pipeline on two views."""
        start_time = time.perf_counter()

        if self.detector is None:
            raise ValueError("CartonDetector instance is required for inference in DualFusionEngine")

        # 1. Run YOLO inference on both images
        front_result = self.detector.detect(front_image, confidence=confidence)
        side_result = self.detector.detect(side_image, confidence=confidence)

        # 2. Cluster layers on each view
        h_front = front_image.shape[0]
        h_side = side_image.shape[0]
        front_clusters = self.cluster_layers_from_detections(front_result.detections, h_front)
        side_clusters = self.cluster_layers_from_detections(side_result.detections, h_side)

        # 3. Align layers and compute N1_k * N2_k
        layers_info, total_count = self.align_and_multiply_layers(
            front_clusters=front_clusters,
            side_clusters=side_clusters,
            h_front=h_front,
            h_side=h_side,
        )

        # 4. Generate visual annotations if requested
        front_b64 = None
        side_b64 = None
        if annotate:
            front_vis = self.draw_annotated_frame(
                image=front_image,
                layer_clusters=front_clusters,
                layers_info=layers_info,
                view_name="FRONT (CAM 1)",
                total_count=total_count,
            )
            side_vis = self.draw_annotated_frame(
                image=side_image,
                layer_clusters=side_clusters,
                layers_info=layers_info,
                view_name="SIDE (CAM 2)",
                total_count=total_count,
            )

            _, buf_front = cv2.imencode(".jpg", front_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, buf_side = cv2.imencode(".jpg", side_vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            front_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_front.tobytes()).decode('ascii')}"
            side_b64 = f"data:image/jpeg;base64,{base64.b64encode(buf_side.tobytes()).decode('ascii')}"

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return DualFusionResult(
            total_count=total_count,
            layers_count=len(layers_info),
            layers=layers_info,
            front_count_raw=len(front_result.detections),
            side_count_raw=len(side_result.detections),
            front_annotated_b64=front_b64,
            side_annotated_b64=side_b64,
            processing_time_ms=elapsed_ms,
            method="dual_layer_multiplication",
        )
