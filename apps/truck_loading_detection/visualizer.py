import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import deque


class LoadingVisualizer:
    """
    Renders an overlay with a virtual tripwire, tracking trails, bounding boxes,
    Worker-Carton holding associations, and a dashboard showing real-time loading metrics.
    """

    def __init__(self):
        # Color palette (BGR)
        self.COLOR_BG_CARD = (25, 28, 36)
        self.COLOR_BORDER = (60, 70, 90)
        self.COLOR_LINE = (0, 70, 255)         # Bright Crimson / Red
        self.COLOR_LINE_GLOW = (0, 35, 140)    # Deeper Glow
        self.COLOR_IN = (72, 219, 90)          # Emerald Green
        self.COLOR_OUT = (50, 50, 240)         # Bright Red
        self.COLOR_NET = (245, 175, 35)        # Cyan / Sky Blue
        self.COLOR_TEXT_MAIN = (245, 245, 245)
        self.COLOR_TEXT_MUTED = (160, 165, 180)
        self.COLOR_TRAIL = (255, 180, 0)       # Amber/Orange for trajectory
        self.COLOR_TETHER = (255, 235, 50)     # Neon Cyan / Yellow for Worker-Carton tether
        self.COLOR_WORKER = (235, 160, 50)     # Worker border color
        self.COLOR_CARTON = (50, 220, 130)     # Carton border color

    def draw_rounded_rect(self, img, pt1, pt2, color, radius=8, thickness=-1):
        """Draws a rounded rectangle on an image."""
        x1, y1 = pt1
        x2, y2 = pt2
        w = x2 - x1
        h = y2 - y1
        r = min(radius, w // 2, h // 2)

        if thickness == -1:
            # Filled rounded rectangle
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        else:
            # Outlined rounded rectangle
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    def draw_tripwire(self, frame: np.ndarray, line_x: int, is_dragging: bool = False, loading_direction: str = "left_to_right"):
        """Draws the virtual tripwire vertical line with glow and directional guides."""
        h, w = frame.shape[:2]
        line_color = (0, 220, 255) if is_dragging else self.COLOR_LINE

        # Draw outer glow line
        overlay = frame.copy()
        cv2.line(overlay, (line_x, 0), (line_x, h), self.COLOR_LINE_GLOW, 8)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Draw crisp center line
        cv2.line(frame, (line_x, 0), (line_x, h), line_color, 2, cv2.LINE_AA)

        # Tripwire Badge at the Top
        badge_w, badge_h = 160, 26
        bx1 = max(10, min(w - badge_w - 10, line_x - badge_w // 2))
        by1 = 15
        
        # Badge background
        sub = frame[by1:by1+badge_h, bx1:bx1+badge_w]
        if sub.shape[0] == badge_h and sub.shape[1] == badge_w:
            card = np.full_like(sub, self.COLOR_BG_CARD)
            cv2.addWeighted(card, 0.85, sub, 0.15, 0, sub)
            frame[by1:by1+badge_h, bx1:bx1+badge_w] = sub
        
        cv2.rectangle(frame, (bx1, by1), (bx1 + badge_w, by1 + badge_h), line_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"TRIPWIRE (X={line_x})",
            (bx1 + 10, by1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Directional arrows along the line
        mid_y = h // 2
        is_r2l = (loading_direction.lower() == "right_to_left")

        # Left label
        left_text = "<< LOAD (+1)" if is_r2l else "RETURN (-1) >>"
        left_color = self.COLOR_IN if is_r2l else self.COLOR_OUT
        cv2.putText(
            frame,
            left_text,
            (max(10, line_x - 140), mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            left_color,
            2,
            cv2.LINE_AA,
        )

        # Right label
        right_text = "<< RETURN (-1)" if not is_r2l else "LOAD (+1) >>"
        right_color = self.COLOR_OUT if not is_r2l else self.COLOR_IN
        cv2.putText(
            frame,
            right_text,
            (min(w - 160, line_x + 15), mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            right_color,
            2,
            cv2.LINE_AA,
        )

    def draw_trajectories(self, frame: np.ndarray, track_history: Dict[int, deque]):
        """Draws centroid motion trails for active tracks."""
        for track_id, points in track_history.items():
            if len(points) < 2:
                continue
            
            pts_list = list(points)
            for i in range(1, len(pts_list)):
                alpha = float(i) / len(pts_list)
                thickness = max(1, int(3 * alpha))
                color = (
                    int(self.COLOR_TRAIL[0] * alpha),
                    int(self.COLOR_TRAIL[1] * alpha),
                    int(self.COLOR_TRAIL[2] * alpha),
                )
                cv2.line(frame, pts_list[i - 1], pts_list[i], color, thickness, cv2.LINE_AA)

    def draw_detections(
        self,
        frame: np.ndarray,
        tracked_objects: List[Tuple[int, Tuple[float, float, float, float], str, float]],
        track_side: Dict[int, str],
        line_x: int,
        associations: Optional[Dict[int, int]] = None,
        carton_boxes: Optional[Dict[int, Tuple[int, int, int, int]]] = None,
        holding_scores: Optional[Dict[int, float]] = None,
        worker_qty: Optional[Dict[int, int]] = None,
    ):
        """
        Draws bounding boxes, centroid dots, track IDs, side indicators, and Worker-Carton associations.
        Also renders classical CV pre-processed carton bounding boxes on workers.
        """
        # Map centroids for association drawing
        centroids = {}
        for track_id, (x1, y1, x2, y2), cls_name, conf in tracked_objects:
            centroids[track_id] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        # 1. Draw Association Tethers first (connecting worker and held carton)
        if associations:
            for cargo_id, worker_id in associations.items():
                if cargo_id in centroids and worker_id in centroids:
                    c_pt = centroids[cargo_id]
                    w_pt = centroids[worker_id]
                    # Glowing tether line
                    cv2.line(frame, w_pt, c_pt, (0, 180, 255), 3, cv2.LINE_AA)
                    cv2.line(frame, w_pt, c_pt, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Midpoint holding badge
                    mid_x = (w_pt[0] + c_pt[0]) // 2
                    mid_y = (w_pt[1] + c_pt[1]) // 2
                    cv2.circle(frame, (mid_x, mid_y), 4, (0, 200, 255), -1, cv2.LINE_AA)

        # 2. Draw Pre-Processed CV Carton Boxes
        if carton_boxes:
            for w_id, c_box in carton_boxes.items():
                if c_box:
                    cx1, cy1, cx2, cy2 = [int(v) for v in c_box]
                    q = (worker_qty.get(w_id, 1) if worker_qty else 1)
                    if w_id >= 500:
                        c_tag = "CARTON (IN FLIGHT)"
                        box_col = (0, 165, 255)  # Vivid Amber/Orange for thrown cartons
                    elif q >= 2:
                        score = holding_scores.get(w_id, 0.0) if holding_scores else 0.0
                        c_tag = f"2 CARTONS (DOUBLE: {score:.2f})"
                        box_col = (0, 255, 180)  # Bright Greenish-Cyan for double pickup
                    else:
                        score = holding_scores.get(w_id, 0.0) if holding_scores else 0.0
                        c_tag = f"CARTON (CV: {score:.2f})"
                        box_col = (0, 220, 255)  # Cyan for carried cartons
                    
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), box_col, 2, cv2.LINE_AA)
                    (cw, ch), _ = cv2.getTextSize(c_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                    cv2.rectangle(frame, (cx1, max(0, cy1 - ch - 4)), (cx1 + cw + 6, cy1), (20, 20, 30), -1)
                    cv2.putText(frame, c_tag, (cx1 + 3, cy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_col, 1, cv2.LINE_AA)

        # 3. Draw Bounding Boxes and Labels
        for track_id, (x1, y1, x2, y2), cls_name, conf in tracked_objects:
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            cx, cy = centroids.get(track_id, (int((x1 + x2) / 2), int((y1 + y2) / 2)))

            # Determine side color
            side = track_side.get(track_id, "left" if cx < line_x else "right")
            box_color = self.COLOR_IN if side == "right" else (220, 180, 50)

            # Draw sleek corner bounding box
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), box_color, 2, cv2.LINE_AA)

            # Corner accents
            line_len = min(15, (ix2 - ix1) // 4, (iy2 - iy1) // 4)
            accent_col = (255, 255, 255)
            # Top-left
            cv2.line(frame, (ix1, iy1), (ix1 + line_len, iy1), accent_col, 3, cv2.LINE_AA)
            cv2.line(frame, (ix1, iy1), (ix1, iy1 + line_len), accent_col, 3, cv2.LINE_AA)
            # Top-right
            cv2.line(frame, (ix2, iy1), (ix2 - line_len, iy1), accent_col, 3, cv2.LINE_AA)
            cv2.line(frame, (ix2, iy1), (ix2, iy1 + line_len), accent_col, 3, cv2.LINE_AA)
            # Bottom-left
            cv2.line(frame, (ix1, iy2), (ix1 + line_len, iy2), accent_col, 3, cv2.LINE_AA)
            cv2.line(frame, (ix1, iy2), (ix1, iy2 - line_len), accent_col, 3, cv2.LINE_AA)
            # Bottom-right
            cv2.line(frame, (ix2, iy2), (ix2 - line_len, iy2), accent_col, 3, cv2.LINE_AA)
            cv2.line(frame, (ix2, iy2), (ix2, iy2 - line_len), accent_col, 3, cv2.LINE_AA)

            # Centroid point
            cv2.circle(frame, (cx, cy), 5, box_color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)

            # Build Contextual Label (with Worker-Carton Holding info)
            is_worker = cls_name.lower() in {"person", "worker", "human"}
            has_cv_carton = bool(carton_boxes and track_id in carton_boxes)
            w_qty = (worker_qty.get(track_id, 1) if worker_qty else 1)

            if associations and track_id in associations:
                p_id = associations[track_id]
                label = f"#{track_id} {cls_name.upper()} [Held by #{p_id}]"
            elif associations and is_worker and (track_id in associations.values()):
                carried_cargos = [c_id for c_id, w_id in associations.items() if w_id == track_id]
                c_str = f"#{carried_cargos[0]}" if len(carried_cargos) == 1 else f"{len(carried_cargos)} pkgs"
                label = f"#{track_id} WORKER [Holding {c_str}]"
            elif is_worker and has_cv_carton:
                if w_qty >= 2:
                    label = f"#{track_id} WORKER [HOLDING 2 CARTONS]"
                else:
                    label = f"#{track_id} WORKER [HOLDING CARTON]"
            elif is_worker:
                label = f"#{track_id} WORKER [EMPTY-HANDED]"
            else:
                label = f"#{track_id} {cls_name.upper()} {conf:.2f}"

            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            
            tag_y1 = max(0, iy1 - lh - 8)
            tag_y2 = max(lh + 8, iy1)
            cv2.rectangle(frame, (ix1, tag_y1), (ix1 + lw + 12, tag_y2), (25, 28, 36), -1)
            cv2.rectangle(frame, (ix1, tag_y1), (ix1 + lw + 12, tag_y2), box_color, 1)
            cv2.putText(
                frame,
                label,
                (ix1 + 6, tag_y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def draw_hud(
        self,
        frame: np.ndarray,
        total_in: int,
        total_out: int,
        net_count: int,
        fps: float,
        active_count: int,
        recent_event: dict = None,
        recent_event_expiry: float = 0.0,
        worker_trips: Optional[int] = None,
    ):
        """Draws the HUD dashboard card in the top-left corner."""
        h, w = frame.shape[:2]
        panel_w = 340
        panel_h = 135
        px1, py1 = 15, 15
        px2, py2 = px1 + panel_w, py1 + panel_h

        # Translucent glassmorphism background
        sub = frame[py1:py2, px1:px2]
        if sub.shape[0] == panel_h and sub.shape[1] == panel_w:
            card = np.full_like(sub, self.COLOR_BG_CARD)
            cv2.addWeighted(card, 0.85, sub, 0.15, 0, sub)
            frame[py1:py2, px1:px2] = sub

        # Card Border
        cv2.rectangle(frame, (px1, py1), (px2, py2), self.COLOR_BORDER, 1, cv2.LINE_AA)

        # Header Title
        cv2.putText(
            frame,
            "TRUCK LOADING MONITOR",
            (px1 + 14, py1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.COLOR_TEXT_MAIN,
            2,
            cv2.LINE_AA,
        )

        # Separator line
        cv2.line(frame, (px1 + 14, py1 + 32), (px2 - 14, py1 + 32), self.COLOR_BORDER, 1)

        # 3-Column Metrics: [LOADED +] [RETURNED -] [NET TOTAL]
        col_w = (panel_w - 28) // 3
        
        # 1. Total Loaded (+1)
        c1_x = px1 + 14
        cv2.putText(frame, "LOADED (+)", (c1_x, py1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"+{total_in}", (c1_x, py1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.COLOR_IN, 2, cv2.LINE_AA)

        # 2. Total Returned (-1)
        c2_x = c1_x + col_w
        cv2.putText(frame, "RETURN (-)", (c2_x, py1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"-{total_out}", (c2_x, py1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.COLOR_OUT, 2, cv2.LINE_AA)

        # 3. Net Count
        c3_x = c2_x + col_w
        cv2.putText(frame, "NET CARGO", (c3_x, py1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{net_count}", (c3_x, py1 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.85, self.COLOR_NET, 2, cv2.LINE_AA)

        # Footer info: FPS and Active Objects
        cv2.line(frame, (px1 + 14, py1 + 92), (px2 - 14, py1 + 92), self.COLOR_BORDER, 1)
        if worker_trips is not None:
            status_text = f"FPS: {fps:4.1f} | Active: {active_count} | Trips: {worker_trips}"
        else:
            status_text = f"FPS: {fps:4.1f}   |   Tracking: {active_count} objs"
        cv2.putText(frame, status_text, (px1 + 14, py1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Toast / Flash notification banner if crossing just occurred
        now = time.time()
        if recent_event and now < recent_event_expiry:
            is_in = recent_event["direction"] == "IN"
            carried_by = recent_event.get("carried_by")
            
            if carried_by is not None:
                toast_text = f"LOADED (+1) | #{recent_event['track_id']} {recent_event['class_name'].upper()} (Worker #{carried_by})" if is_in else f"RETURNED (-1) | #{recent_event['track_id']} {recent_event['class_name'].upper()} (Worker #{carried_by})"
            else:
                toast_text = f"LOADED (+1) | #{recent_event['track_id']} {recent_event['class_name'].upper()}" if is_in else f"RETURNED (-1) | #{recent_event['track_id']} {recent_event['class_name'].upper()}"
            
            toast_color = self.COLOR_IN if is_in else self.COLOR_OUT

            tw_w, tw_h = 380, 36
            tx1 = max(10, w // 2 - tw_w // 2)
            ty1 = h - 60
            tx2, ty2 = tx1 + tw_w, ty1 + tw_h

            sub_toast = frame[ty1:ty2, tx1:tx2]
            if sub_toast.shape[0] == tw_h and sub_toast.shape[1] == tw_w:
                card_t = np.full_like(sub_toast, (20, 20, 25))
                cv2.addWeighted(card_t, 0.90, sub_toast, 0.10, 0, sub_toast)
                frame[ty1:ty2, tx1:tx2] = sub_toast

            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), toast_color, 2, cv2.LINE_AA)
            cv2.putText(
                frame,
                toast_text,
                (tx1 + 12, ty1 + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                toast_color,
                2,
                cv2.LINE_AA,
            )

    def draw_instructions(self, frame: np.ndarray):
        """Draws keybindings hint at the bottom right corner."""
        h, w = frame.shape[:2]
        hints = "[Drag Line / Left-Right Keys] Adjust | [SPACE] Pause | [R] Reset | [Q] Quit"
        (tw, th), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(
            frame,
            hints,
            (w - tw - 15, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 185, 195),
            1,
            cv2.LINE_AA,
        )
