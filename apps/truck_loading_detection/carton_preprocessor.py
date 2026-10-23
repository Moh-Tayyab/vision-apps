import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = float(iw * ih)
    if inter <= 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = float(area_a + area_b) - inter
    return inter / union if union > 0 else 0.0


class CartonHoldingDetector:
    """
    Zero-Training Classical Computer Vision Pre-Processing Engine.

    NOTE: COCO YOLO has no ``carton`` class. Cartons are NOT detected by YOLO.
    This detector uses classical CV on the worker's carrying ROI:

      1. Dynamic Carrying Region of Interest (ROI) extraction (chest/torso/hands zone)
      2. Color space analysis (corrugated kraft cardboard & packaging tape HSV thresholds)
      3. High-gradient edge filtering (Canny + Sobel)
      4. Morphological closure & Quadrilateral Contour Approximation (approxPolyDP)
      5. Aspect ratio & geometric rectangularity verification

    ROI modes (this is what makes the W x H fusion dimensionally correct):
      "wide"   - the whole carrying zone. Counts every carton in the load.
                 Correct for a camera that must measure the load FOOTPRINT
                 (e.g. the overhead camera looking down at a pallet).
      "column" - a narrow VERTICAL slice through the carrying zone. Counts only
                 the cartons that are STACKED in that slice, i.e. the load
                 HEIGHT. Required for a side-profile camera, otherwise two
                 cartons held side by side are misread as a stack of two and
                 the W x H product double counts the load.
      "row"    - a narrow HORIZONTAL slice. Counts only the cartons that sit
                 SIDE BY SIDE along one row, i.e. the load WIDTH.
      "tight"  - the worker's own body box with NO forward expansion. Used by
                 the overhead camera: seen from directly above, a carried
                 carton lies ON the worker, whereas cartons lying on the floor
                 are beside them, so a tight box keeps the floor cartons out.
    """

    def __init__(
        self,
        min_roi_area_ratio: float = 0.03,
        min_rect_aspect: float = 0.35,
        max_rect_aspect: float = 2.8,
        min_carton_score: float = 0.28,
        roi_mode: str = "wide",
        slice_frac: float = 0.30,
        reference_carton_area: Optional[int] = None,
        max_carton_area_ratio: float = 2.6,
        roi_y_frac: Optional[Tuple[float, float]] = None,
    ):
        self.min_roi_area_ratio = min_roi_area_ratio
        self.min_rect_aspect = min_rect_aspect
        self.max_rect_aspect = max_rect_aspect
        self.min_carton_score = min_carton_score
        self.roi_mode = roi_mode
        self.slice_frac = slice_frac
        # Area (px) of ONE carton in this camera's resolution. When it is
        # known, a blob that is much larger is a PILE of cartons (the worker is
        # standing next to the stack) rather than a carton in their hands, and
        # the detection is rejected instead of being counted.
        self.reference_carton_area = reference_carton_area
        self.max_carton_area_ratio = max_carton_area_ratio
        self.roi_y_frac = roi_y_frac

        # Cardboard / Corrugated Kraft Brown HSV bounds
        self.lower_brown_1 = np.array([6, 25, 35])
        self.upper_brown_1 = np.array([32, 255, 235])

        # Light cardboard / pale kraft / tape HSV bounds
        self.lower_tape = np.array([0, 0, 140])
        self.upper_tape = np.array([180, 50, 255])

    def _is_pile(self, area: float) -> bool:
        if not self.reference_carton_area:
            return False
        return area > self.reference_carton_area * self.max_carton_area_ratio

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

        # Upper chest to waist/hands (configurable fraction of person height)
        y_lo, y_hi = (self.roi_y_frac if self.roi_y_frac is not None
                      else (0.25, 0.82))
        roi_y1 = max(0, int(y1 + y_lo * h))
        roi_y2 = min(img_h, int(y1 + y_hi * h))

        # Forward expansion based on travel direction
        # For horizontal movement (left_to_right / right_to_left): box is in front HORIZONTALLY
        # For vertical movement (up_to_down / down_to_up): box is in front VERTICALLY
        if self.roi_mode == "tight":
            # No expansion: keep the worker's own body box, so a carton lying
            # on the floor beside them is not dragged into the ROI.
            roi_x1 = max(0, x1)
            roi_x2 = min(img_w, x2)
            roi_y1 = max(0, y1)
            roi_y2 = min(img_h, y2)
        elif direction == "left_to_right":
            # Worker moving right: box protrudes towards the right front
            roi_x1 = max(0, int(x1 - 0.05 * w))
            roi_x2 = min(img_w, int(x2 + 0.25 * w))
        elif direction == "right_to_left":
            # Worker moving left: box protrudes towards the left front
            roi_x1 = max(0, int(x1 - 0.25 * w))
            roi_x2 = min(img_w, int(x2 + 0.05 * w))
        elif direction == "up_to_down":
            # Worker moving down: box is held at chest/stomach level
            # Extend only slightly down from person's bottom
            roi_x1 = max(0, int(x1 - 0.15 * w))
            roi_x2 = min(img_w, int(x2 + 0.15 * w))
            roi_y1 = max(0, int(y1 + 0.05 * h))
            roi_y2 = min(img_h, int(y2 + 0.55 * h))
        elif direction == "down_to_up":
            # Worker moving up: box protrudes towards the top front (vertical)
            roi_x1 = max(0, int(x1 - 0.10 * w))
            roi_x2 = min(img_w, int(x2 + 0.10 * w))
            roi_y1 = max(0, int(y1 - 0.35 * h))
            roi_y2 = min(img_h, int(y1 + 0.10 * h))
        else:
            # Default: slight horizontal expansion
            roi_x1 = max(0, int(x1 - 0.10 * w))
            roi_x2 = min(img_w, int(x2 + 0.10 * w))

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        # Narrow the slice so that the count means a DIMENSION, not a blob count.
        if self.roi_mode == "column":
            # keep a narrow vertical band: only stacked cartons are counted
            band = max(12, int((roi_x2 - roi_x1) * self.slice_frac))
            centre = (roi_x1 + roi_x2) // 2
            nx1 = max(roi_x1, centre - band // 2)
            nx2 = min(roi_x2, nx1 + band)
            nx1 = max(roi_x1, nx2 - band)
            roi = frame[roi_y1:roi_y2, nx1:max(nx1 + 1, nx2)]
            roi_x1, roi_x2 = nx1, max(nx1 + 1, nx2)
        elif self.roi_mode == "row":
            # keep a narrow horizontal band: only side-by-side cartons count
            band = max(12, int((roi_y2 - roi_y1) * self.slice_frac))
            centre = (roi_y1 + roi_y2) // 2
            ny1 = max(roi_y1, centre - band // 2)
            ny2 = min(roi_y2, ny1 + band)
            ny1 = max(roi_y1, ny2 - band)
            roi = frame[ny1:max(ny1 + 1, ny2), roi_x1:roi_x2]
            roi_y1, roi_y2 = ny1, max(ny1 + 1, ny2)

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
            return False, 0.0, None, 0, {}

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

            # For column mode: reject contours that fill too much of ROI vertically
            # (worker body vs. cartons at chest level)
            if self.roi_mode == "column" and bh > roi_h * 0.55:
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
                return False, 0.0, None, 0
            cbx, cby, cbw, cbh = cv2.boundingRect(cnt)
            if cbw <= 0 or cbh <= 0:
                return False, 0.0, None, 0
            
            # For column mode (side camera measuring stack height): reject contours that
            # fill too much of the ROI vertically -- these are worker bodies, not cartons.
            # Cartons at chest level occupy ~30-50% of ROI height; worker torso fills 70%+.
            if self.roi_mode == "column" and cbh > roi_h * 0.55:
                return False, 0.0, None, 0
            
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
            return is_box, score, (cbx, cby, cbw, cbh), ca

        valid_box_cnts = []
        for c in color_cnts:
            is_b, b_sc, b_box, b_area = evaluate_carton_geometry(c)
            if is_b:
                valid_box_cnts.append((c, b_sc, b_box, b_area))

        # A blob far bigger than ONE carton is the STACK the worker is standing
        # next to, not a carton in their hands. Such blobs are dropped so the
        # pile cannot be counted as a load.
        pile_blobs = 0
        kept_box_cnts = []
        for item in valid_box_cnts:
            if self._is_pile(item[3]):
                pile_blobs += 1
            else:
                kept_box_cnts.append(item)
        valid_box_cnts = kept_box_cnts

        if valid_box_cnts:
            best_vb = max(valid_box_cnts, key=lambda item: item[1])
            has_color_blob = True
            color_box_score = best_vb[1]
            color_box = best_vb[2]
        elif best_color_cnt is not None and has_edge_rect:
            cbx, cby, cbw, cbh = cv2.boundingRect(best_color_cnt)
            if not self._is_pile(cv2.contourArea(best_color_cnt)):
                has_color_blob = True
                color_box = (cbx, cby, cbw, cbh)
            else:
                pile_blobs += 1

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
                carton_qty = min(3, len(valid_box_cnts))
            # Condition 2: Massive double-volume / tall stacked carton filling >= 52% of ROI with verified box shape
            elif (largest_blob_ratio >= 0.50 or (color_ratio >= 0.52 and largest_blob_ratio >= 0.40)) and (has_edge_rect or has_color_blob):
                carton_qty = 2
            elif best_rect is not None and (best_rect[3] / float(roi_h)) >= 0.56 and rect_score >= 0.18:
                carton_qty = 2

        # If the only thing in the ROI was a pile-sized blob, the worker is
        # standing next to the stack, not carrying anything.
        if pile_blobs and not valid_box_cnts and largest_blob_ratio >= 0.30:
            is_holding = False
            carton_qty = 0

        debug_info = {
            "composite_score": round(float(composite_score), 3),
            "rect_score": round(float(rect_score), 3),
            "color_ratio": round(float(color_ratio), 3),
            "blob_ratio": round(float(largest_blob_ratio), 3),
            "edge_density": round(float(edge_density), 3),
            "has_rect_contour": has_edge_rect or has_color_blob,
            "carton_qty": carton_qty,
            "pile_blobs": pile_blobs,
            "boxes": [
                (
                    int(roi_x1 + item[2][0]),
                    int(roi_y1 + item[2][1]),
                    int(roi_x1 + item[2][0] + item[2][2]),
                    int(roi_y1 + item[2][1] + item[2][3]),
                )
                for item in valid_box_cnts
            ],
        }

        return is_holding, float(composite_score), carton_box_abs, carton_qty, debug_info

    def detect_carton_boxes(
        self,
        frame: np.ndarray,
        person_box: Tuple[float, float, float, float],
        direction: str = "left_to_right",
        iou_thresh: float = 0.40,
    ) -> Tuple[List[Tuple[int, int, int, int]], int, Dict[str, Any]]:
        """Return EVERY individual carton box held by this worker.

        ``detect_carton`` only returns the single best box plus a quantity
        estimate. The red-line counter needs one box per carton so that each
        carton can be tracked and counted separately, so this method exposes
        the full set (de-duplicated with greedy IoU suppression).

        Returns:
            (boxes, count, debug_info)
        """
        is_holding, _conf, best_box, _qty, dbg = self.detect_carton(
            frame, person_box, direction
        )
        if not is_holding:
            return [], 0, dbg

        boxes = [tuple(int(v) for v in b) for b in dbg.get("boxes", [])]
        if not boxes and best_box is not None:
            boxes = [tuple(int(v) for v in best_box)]

        # Largest first, drop heavily overlapping duplicates.
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        kept: List[Tuple[int, int, int, int]] = []
        for box in boxes:
            if all(_box_iou(box, other) < iou_thresh for other in kept):
                kept.append(box)
        return kept, len(kept), dbg
