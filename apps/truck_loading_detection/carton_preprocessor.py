import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class CartonHoldingDetector:
    """
    Zero-Training Classical Computer Vision Pre-Processing Engine.
    Detects whether a tracked worker is carrying a cardboard carton box using:
      1. Dynamic Carrying Region of Interest (ROI) extraction (chest/torso/hands zone)
      2. Color space analysis (corrugated kraft cardboard & packaging tape HSV thresholds)
      3. High-gradient edge filtering (Canny + Sobel)
      4. Morphological closure & Quadrilateral Contour Approximation (approxPolyDP)
      5. Aspect ratio & geometric rectangularity verification
    """

    def __init__(
        self,
        min_roi_area_ratio: float = 0.03,
        min_rect_aspect: float = 0.35,
        max_rect_aspect: float = 2.8,
        min_carton_score: float = 0.28,
    ):
        self.min_roi_area_ratio = min_roi_area_ratio
        self.min_rect_aspect = min_rect_aspect
        self.max_rect_aspect = max_rect_aspect
        self.min_carton_score = min_carton_score

        # Cardboard / Corrugated Kraft Brown HSV bounds
        self.lower_brown_1 = np.array([6, 25, 35])
        self.upper_brown_1 = np.array([32, 255, 235])

        # Light cardboard / pale kraft / tape HSV bounds
        self.lower_tape = np.array([0, 0, 140])
        self.upper_tape = np.array([180, 50, 255])

    def extract_carrying_roi(
        self,
        frame: np.ndarray,
        person_box: Tuple[float, float, float, float],
        direction: str = "left_to_right",
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crops the anatomical carrying region (chest, stomach, hands, front waist).
        Dynamically biases the ROI forward based on travel direction.
        """
        img_h, img_w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in person_box]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Upper chest to waist/hands (25% to 80% of person height)
        roi_y1 = max(0, int(y1 + 0.25 * h))
        roi_y2 = min(img_h, int(y1 + 0.82 * h))

        # Forward horizontal expansion where the box is held
        if direction == "left_to_right":
            # Worker moving right: box protrudes towards the right front
            roi_x1 = max(0, int(x1 - 0.05 * w))
            roi_x2 = min(img_w, int(x2 + 0.25 * w))
        else:
            # Worker moving left: box protrudes towards the left front
            roi_x1 = max(0, int(x1 - 0.25 * w))
            roi_x2 = min(img_w, int(x2 + 0.05 * w))

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        return roi, (roi_x1, roi_y1, roi_x2, roi_y2)

    def detect_carton(
        self,
        frame: np.ndarray,
        person_box: Tuple[float, float, float, float],
        direction: str = "left_to_right",
    ) -> Tuple[bool, float, Optional[Tuple[int, int, int, int]], int, Dict[str, Any]]:
        """
        Runs the multi-stage OpenCV pre-processing pipeline on the worker carrying ROI.
        
        Returns:
            is_holding (bool): True if carton presence is verified.
            confidence (float): Normalized score [0.0 - 1.0].
            carton_box (tuple | None): Absolute bounding box of the carton (x1, y1, x2, y2).
            carton_qty (int): Estimated number of cartons (1 or 2).
            debug_info (dict): Detailed intermediate telemetry metrics.
        """
        roi, roi_coords = self.extract_carrying_roi(frame, person_box, direction)
        roi_h, roi_w = roi.shape[:2]

        if roi_h < 15 or roi_w < 15:
            return False, 0.0, None, {}

        roi_area = float(roi_h * roi_w)
        roi_x1, roi_y1, _, _ = roi_coords

        # -------------------------------------------------------------
        # STAGE 1: Color Space Pre-processing (Cardboard Kraft + Tape)
        # -------------------------------------------------------------
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_brown = cv2.inRange(hsv, self.lower_brown_1, self.upper_brown_1)
        mask_tape = cv2.inRange(hsv, self.lower_tape, self.upper_tape)
        color_mask = cv2.bitwise_or(mask_brown, mask_tape)

        color_pixels = np.count_nonzero(color_mask)
        color_ratio = color_pixels / roi_area

        # Morphological closure on cardboard color to bridge worker hands/arms
        kernel_color = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed_color = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel_color)
        color_cnts, _ = cv2.findContours(closed_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_color_cnt = None
        max_color_area = 0.0
        for c in color_cnts:
            a = cv2.contourArea(c)
            if a > max_color_area:
                max_color_area = a
                best_color_cnt = c

        largest_blob_ratio = max_color_area / roi_area

        # -------------------------------------------------------------
        # STAGE 2: Gradient & Edge Pre-processing (Canny + Morphology)
        # -------------------------------------------------------------
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 40, 140)

        edge_pixels = np.count_nonzero(edges)
        edge_density = edge_pixels / roi_area
        edge_score = min(1.0, edge_density / 0.10)

        # Bridge broken edges along box corners
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge)

        # -------------------------------------------------------------
        # STAGE 3: Rectangular Polygon & Contour Geometric Analysis
        # -------------------------------------------------------------
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_rect = None
        max_rect_area = 0.0
        rect_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (roi_area * self.min_roi_area_ratio):
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            num_vertices = len(approx)

            bx, by, bw, bh = cv2.boundingRect(approx)
            if bh <= 0:
                continue

            aspect = float(bw) / float(bh)
            if not (self.min_rect_aspect <= aspect <= self.max_rect_aspect):
                continue

            bounding_box_area = float(bw * bh)
            fill_factor = area / bounding_box_area if bounding_box_area > 0 else 0.0

            is_good_shape = (4 <= num_vertices <= 8) and (fill_factor >= 0.35)
            if is_good_shape and area > max_rect_area:
                max_rect_area = area
                best_rect = (bx, by, bw, bh)
                rect_score = min(1.0, (area / (roi_area * 0.30)) * fill_factor)

        # -------------------------------------------------------------
        # STAGE 4: Multi-Signal Composite Geometric Verification
        # -------------------------------------------------------------
        has_edge_rect = (best_rect is not None) and (rect_score >= 0.12)
        has_color_blob = False
        color_box = None
        color_box_score = 0.0

        # Geometric validator: distinguish rigid cardboard cartons from soft clothing folds & limbs
        def evaluate_carton_geometry(cnt):
            ca = cv2.contourArea(cnt)
            if ca < (roi_area * 0.06):
                return False, 0.0, None
            cbx, cby, cbw, cbh = cv2.boundingRect(cnt)
            if cbw <= 0 or cbh <= 0:
                return False, 0.0, None
            c_fill = ca / float(cbw * cbh)
            chull = cv2.convexHull(cnt)
            cha = cv2.contourArea(chull)
            c_solidity = ca / cha if cha > 0 else 0.0
            c_aspect = float(cbw) / float(cbh)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            
            # Rigid cardboard boxes have high convexity (solidity >= 0.60), rectangular fill (>= 0.40),
            # and polygonal contours (>= 4 vertices). Soft clothing folds have low solidity/fill (< 0.55).
            is_box = (c_solidity >= 0.60 and c_fill >= 0.40 and len(approx) >= 4 and 0.30 <= c_aspect <= 3.5)
            score = min(1.0, (ca / (roi_area * 0.25)) * c_fill)
            return is_box, score, (cbx, cby, cbw, cbh)

        valid_box_cnts = []
        for c in color_cnts:
            is_b, b_sc, b_box = evaluate_carton_geometry(c)
            if is_b:
                valid_box_cnts.append((c, b_sc, b_box))

        if valid_box_cnts:
            best_vb = max(valid_box_cnts, key=lambda item: item[1])
            has_color_blob = True
            color_box_score = best_vb[1]
            color_box = best_vb[2]
        elif best_color_cnt is not None and has_edge_rect:
            cbx, cby, cbw, cbh = cv2.boundingRect(best_color_cnt)
            has_color_blob = True
            color_box = (cbx, cby, cbw, cbh)

        is_holding = has_edge_rect or (has_color_blob and (rect_score >= 0.04 or edge_density >= 0.020)) or (color_ratio >= 0.30 and edge_density >= 0.038)

        composite_score = min(1.0, (largest_blob_ratio * 1.2) + (edge_score * 0.3) + (max(rect_score, color_box_score) * 0.5))

        carton_box_abs = None
        if is_holding:
            chosen_box = best_rect if best_rect is not None else color_box
            if chosen_box is not None:
                bx, by, bw, bh = chosen_box
                carton_box_abs = (
                    roi_x1 + bx,
                    roi_y1 + by,
                    roi_x1 + bx + bw,
                    roi_y1 + by + bh,
                )

        # Multi-carton quantity estimation (detect single vs double pickup)
        carton_qty = 1
        if is_holding:
            # Condition 1: Multiple distinct rigid cardboard boxes in the carrying zone
            if len(valid_box_cnts) >= 2:
                carton_qty = 2
            # Condition 2: Massive double-volume / tall stacked carton filling >= 52% of ROI with verified box shape
            elif (largest_blob_ratio >= 0.50 or (color_ratio >= 0.52 and largest_blob_ratio >= 0.40)) and (has_edge_rect or has_color_blob):
                carton_qty = 2
            elif best_rect is not None and (best_rect[3] / float(roi_h)) >= 0.56 and rect_score >= 0.18:
                carton_qty = 2

        debug_info = {
            "composite_score": round(float(composite_score), 3),
            "rect_score": round(float(rect_score), 3),
            "color_ratio": round(float(color_ratio), 3),
            "blob_ratio": round(float(largest_blob_ratio), 3),
            "edge_density": round(float(edge_density), 3),
            "has_rect_contour": has_edge_rect or has_color_blob,
            "carton_qty": carton_qty,
        }

        return is_holding, float(composite_score), carton_box_abs, carton_qty, debug_info
