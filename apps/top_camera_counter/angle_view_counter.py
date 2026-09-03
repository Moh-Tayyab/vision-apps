"""Angled View Carton Counter.

Processes trained YOLO model with classes:
- 'row' (side-view horizontal layers/rows)
- 'top_count' (top-surface cartons)
Or single-class fallback.

Calculates:
- Total Rows (Side View)
- Top Row Cartons (Top View)
- Estimated Total Pallet Cartons = Top Cartons × Total Rows
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AngledViewResult:
    """Result from angled view counting."""
    total_rows: int
    top_row_cartons: int
    estimated_total_cartons: int
    total_cartons_detected: int
    columns: List[Dict]
    top_cartons: List[Dict]
    all_cartons: List[Dict]
    row_boxes: List[Dict]
    inference_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_rows": self.total_rows,
            "top_row_cartons": self.top_row_cartons,
            "estimated_total_cartons": self.estimated_total_cartons,
            "total_cartons_detected": self.total_cartons_detected,
            "columns_count": len(self.columns),
            "columns": self.columns,
            "top_cartons": self.top_cartons,
            "row_boxes": self.row_boxes,
            "all_cartons": self.all_cartons,
            "inference_time_ms": round(self.inference_time_ms, 1),
            "formula": f"Total Estimated = {self.top_row_cartons} (Top) × {self.total_rows} (Rows) = {self.estimated_total_cartons} Cartons",
        }


def compute_iou(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Compute standard Intersection over Union (IoU)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ios(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Smaller Area (containment)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    min_area = min(area1, area2)
    return inter / min_area if min_area > 0 else 0.0


def non_max_suppression(
    boxes: List[Dict],
    iou_thresh: float = 0.35,
    ios_thresh: float = 0.50,
) -> List[Dict]:
    """Applies IoU NMS and containment suppression."""
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b.get("confidence", 0.0), reverse=True)
    kept: List[Dict] = []

    for cur in sorted_boxes:
        cur_b = (cur["x1"], cur["y1"], cur["x2"], cur["y2"])
        suppressed = False
        for saved in kept:
            saved_b = (saved["x1"], saved["y1"], saved["x2"], saved["y2"])
            if compute_iou(cur_b, saved_b) > iou_thresh or compute_ios(cur_b, saved_b) > ios_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(cur)

    return kept


def analyze_angled_view(
    image: np.ndarray,
    model,
    confidence: float = 0.25,
    overlap_threshold: float = 0.30,
    nms_iou: float = 0.35,
) -> AngledViewResult:
    """Full analysis of angled camera view using trained YOLO model."""
    import time
    start = time.perf_counter()

    results = model(image, conf=confidence, verbose=False)
    img_h, img_w = image.shape[:2]

    # Inspect class names
    class_names = getattr(model, "names", {0: "carton"})
    has_two_classes = any("row" in str(v).lower() for v in class_names.values())

    raw_top_boxes = []
    raw_row_boxes = []
    raw_all_boxes = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            c_name = str(class_names.get(cls_id, "carton")).lower()

            w = float(x2 - x1)
            h = float(y2 - y1)
            if w <= 10 or h <= 10:
                continue

            item = {
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "cx": float(x1 + x2) / 2.0,
                "cy": float(y1 + y2) / 2.0,
                "w": w, "h": h,
                "confidence": round(conf, 3),
                "class_id": cls_id,
                "class_name": c_name,
            }

            raw_all_boxes.append(item)

            if "row" in c_name:
                raw_row_boxes.append(item)
            elif "top" in c_name or not has_two_classes:
                raw_top_boxes.append(item)

    # Apply NMS deduplication
    clean_top_boxes = non_max_suppression(raw_top_boxes, iou_thresh=nms_iou, ios_thresh=0.50)
    clean_row_boxes = non_max_suppression(raw_row_boxes, iou_thresh=nms_iou, ios_thresh=0.50)

    # Sort Top Cartons from left to right
    top_cartons = sorted(clean_top_boxes, key=lambda b: b["cx"])
    top_row_count = len(top_cartons)

    # Process Rows: Sort rows from bottom (Row 1) to top (Row N) by Y coordinate (highest Y = lowest physical row)
    if clean_row_boxes:
        sorted_rows_by_y = sorted(clean_row_boxes, key=lambda b: b["cy"], reverse=True)
        row_boxes = []
        for idx, rb in enumerate(sorted_rows_by_y, start=1):
            rb_copy = dict(rb)
            rb_copy["row_number"] = idx
            rb_copy["name"] = f"Row {idx}" + (" (Bottom)" if idx == 1 else " (Top)" if idx == len(sorted_rows_by_y) else "")
            row_boxes.append(rb_copy)
        total_rows = len(row_boxes)
    else:
        # Fallback: estimate rows from top boxes geometry if no direct row boxes
        if top_cartons:
            med_h = float(np.median([b["h"] for b in top_cartons]))
            min_y = float(min(b["y1"] for b in top_cartons))
            max_y = float(max(b["y2"] for b in top_cartons))
            total_rows = max(1, int(round((max_y - min_y) / max(1.0, med_h))))
        else:
            total_rows = 1
        row_boxes = []

    # Calculate estimated pallet total = Top Cartons * Total Rows
    estimated_total = top_row_count * total_rows if total_rows > 0 else len(raw_all_boxes)

    # Group top cartons into columns for UI breakdown
    structured_columns = []
    for idx, tc in enumerate(top_cartons, start=1):
        structured_columns.append({
            "column_index": idx,
            "cartons_count": 1,
            "top_carton": tc,
        })

    inference_ms = (time.perf_counter() - start) * 1000.0

    return AngledViewResult(
        total_rows=total_rows,
        top_row_cartons=top_row_count,
        estimated_total_cartons=estimated_total,
        total_cartons_detected=len(raw_all_boxes),
        columns=structured_columns,
        top_cartons=top_cartons,
        all_cartons=raw_all_boxes,
        row_boxes=row_boxes,
        inference_time_ms=inference_ms,
    )


def annotate_angled_view(
    image: np.ndarray,
    result: AngledViewResult,
    show_top_highlight: bool = True,
    show_side_rows: bool = True,
    show_column_colors: bool = True,
    show_formula_banner: bool = True,
    **kwargs,
) -> np.ndarray:
    """Draw annotations matching user labeling:
    - Pink/Red boxes for Top Cartons.
    - Purple boxes for Side Rows (Row 1, Row 2, Row 3).
    - Header HUD banner with formula.
    """
    vis = image.copy()
    img_h, img_w = vis.shape[:2]

    # 1. Draw Purple Boxes on Side Rows
    if show_side_rows and result.row_boxes:
        # Purple color: BGR (220, 50, 180)
        purple_color = (220, 50, 180)

        for row in result.row_boxes:
            rx1, ry1 = int(round(row["x1"])), int(round(row["y1"]))
            rx2, ry2 = int(round(row["x2"])), int(round(row["y2"]))
            r_name = row.get("name", f"Row {row.get('row_number', 1)}")

            # Draw purple rectangle
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), purple_color, 3)

            # Draw row label badge
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            thick = 2
            (tw, th), _ = cv2.getTextSize(r_name, font, font_scale, thick)

            badge_y = max(th + 6, ry1 - 4)
            cv2.rectangle(vis, (rx1, badge_y - th - 6), (rx1 + tw + 10, badge_y + 4), purple_color, -1)
            cv2.putText(vis, r_name, (rx1 + 5, badge_y - 1), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    # 2. Draw Pink/Red Boxes for TOP Cartons
    if show_top_highlight:
        # Pink/Red color: BGR (80, 50, 245)
        top_color = (80, 50, 245)

        for idx, top in enumerate(result.top_cartons, start=1):
            x1, y1 = int(round(top["x1"])), int(round(top["y1"]))
            x2, y2 = int(round(top["x2"])), int(round(top["y2"]))

            # Draw thick bounding box
            cv2.rectangle(vis, (x1, y1), (x2, y2), top_color, 3)

            # TOP #1, TOP #2 badge
            label = f"TOP #{idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.52
            thick = 2
            (tw, th), _ = cv2.getTextSize(label, font, font_scale, thick)

            badge_y = max(th + 6, y1 - 4)
            cv2.rectangle(vis, (x1, badge_y - th - 6), (x1 + tw + 10, badge_y + 4), top_color, -1)
            cv2.putText(vis, label, (x1 + 5, badge_y - 1), font, font_scale, (255, 255, 255), thick, cv2.LINE_AA)

    # 3. Header HUD Banner
    if show_formula_banner:
        banner_h = 88 if img_w >= 640 else 105
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (img_w, banner_h), (10, 15, 26), -1)
        cv2.addWeighted(overlay, 0.90, vis, 0.10, 0, vis)
        cv2.line(vis, (0, banner_h), (img_w, banner_h), (56, 189, 248), 2)

        f_scale_h1 = 0.62 if img_w >= 700 else 0.50
        f_scale_h2 = 0.50 if img_w >= 700 else 0.40

        line1 = f"Top Cartons: {result.top_row_cartons}  |  Side Rows: {result.total_rows}  |  Est Total: {result.estimated_total_cartons} Cartons"
        cv2.putText(vis, line1, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, f_scale_h1, (56, 189, 248), 2, cv2.LINE_AA)

        line2 = f"Formula: {result.top_row_cartons} (Top) x {result.total_rows} (Rows) = {result.estimated_total_cartons} Cartons  |  Detected: {result.total_cartons_detected}"
        cv2.putText(vis, line2, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, f_scale_h2, (203, 213, 225), 1, cv2.LINE_AA)

    return vis
