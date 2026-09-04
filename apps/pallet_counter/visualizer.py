"""Annotated output for row-wise pallet counting.

Each view is drawn with its detections grouped by row: one colour per row, a
per-carton index within the row, and a banner summarising the row breakdown.
Rows are labelled with the same numbers used in the JSON response so the two
can be read side by side.
"""

from __future__ import annotations

import base64
from typing import List, Optional, Sequence

import cv2
import numpy as np

from row_engine import PalletResult, Row, RowPairing, ViewResult

# Distinct, high-contrast BGR colours; reused cyclically for tall stacks.
ROW_COLORS = [
    (80, 200, 60),    # green
    (60, 160, 255),   # orange
    (230, 130, 60),   # blue
    (80, 80, 240),    # red
    (220, 120, 220),  # violet
    (60, 220, 220),   # yellow
]

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _row_color(index: int) -> tuple:
    return ROW_COLORS[(index - 1) % len(ROW_COLORS)]


def _scale(image: np.ndarray) -> float:
    """Line and text scale factor so annotations read the same at any size."""
    return max(1.0, min(image.shape[:2]) / 640.0)


def _draw_banner(image: np.ndarray, lines: Sequence[str], color: tuple) -> None:
    """Translucent header carrying the view's summary text."""
    s = _scale(image)
    pad = int(10 * s)
    line_h = int(30 * s)
    height = pad * 2 + line_h * len(lines)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    cv2.line(image, (0, height), (image.shape[1], height), color, max(2, int(2 * s)))

    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - int(8 * s)
        cv2.putText(image, text, (pad, y), FONT, 0.7 * s, (255, 255, 255),
                    max(1, int(1.6 * s)), cv2.LINE_AA)


def annotate_view(
    image: np.ndarray,
    view: ViewResult,
    pairings: Optional[Sequence[RowPairing]] = None,
    total_count: Optional[int] = None,
) -> np.ndarray:
    """Draw row-grouped detections over one view."""
    vis = image.copy()
    s = _scale(vis)
    thickness = max(2, int(2.5 * s))

    for row in view.rows:
        color = _row_color(row.index)

        # Row band: the vertical extent this row occupies.
        y1, y2 = row.y_range
        band = vis.copy()
        cv2.rectangle(band, (0, int(y1)), (vis.shape[1], int(y2)), color, -1)
        cv2.addWeighted(band, 0.10, vis, 0.90, 0, vis)

        for j, b in enumerate(row.boxes, start=1):
            p1 = (int(b.x1), int(b.y1))
            p2 = (int(b.x2), int(b.y2))
            cv2.rectangle(vis, p1, p2, color, thickness)

            # "row.index" label anchored inside the box's top-left corner.
            label = f"R{row.index}-{j}"
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.62 * s, max(1, int(1.6 * s)))
            lx, ly = p1[0] + int(6 * s), p1[1] + int(8 * s)
            cv2.rectangle(vis, (lx, ly), (lx + tw + int(8 * s), ly + th + int(10 * s)),
                          color, -1)
            cv2.putText(vis, label, (lx + int(4 * s), ly + th + int(4 * s)),
                        FONT, 0.62 * s, (255, 255, 255), max(1, int(1.6 * s)), cv2.LINE_AA)

        # Row summary tag on the left edge.
        tag = f"ROW {row.index}: {row.count}"
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.7 * s, max(1, int(2 * s)))
        ty = int((y1 + y2) / 2)
        cv2.rectangle(vis, (0, ty - th - int(10 * s)),
                      (tw + int(16 * s), ty + int(10 * s)), color, -1)
        cv2.putText(vis, tag, (int(8 * s), ty), FONT, 0.7 * s, (255, 255, 255),
                    max(1, int(2 * s)), cv2.LINE_AA)

    lines = [
        f"{view.name.upper()} VIEW - {len(view.rows)} rows, {view.total_faces} faces",
        "rows: " + " | ".join(f"R{r.index}={r.count}" for r in view.rows),
    ]
    if abs(view.tilt_deg) >= 3.0:
        lines[0] += f"  (tilt {view.tilt_deg:+.0f} deg corrected)"
    if total_count is not None:
        lines.append(f"PALLET TOTAL: {total_count}")
    _draw_banner(vis, lines, _row_color(1))
    return vis


def annotate_pair(
    front_image: np.ndarray,
    side_image: np.ndarray,
    result: PalletResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Annotate both views of a counted pallet."""
    front = annotate_view(front_image, result.front, result.rows, result.total_count)
    side = annotate_view(side_image, result.side, result.rows, result.total_count)
    return front, side


def build_summary_panel(result: PalletResult, width: int = 900) -> np.ndarray:
    """Standalone table image: per-row N1 x N2 = total, and the pallet sum."""
    s = max(1.0, width / 900.0)
    header_h = int(90 * s)
    row_h = int(52 * s)
    footer_h = int(80 * s)
    warn_h = int(34 * s) * len(result.warnings)
    height = header_h + row_h * (len(result.rows) + 1) + footer_h + warn_h

    panel = np.full((height, width, 3), 24, dtype=np.uint8)

    cv2.putText(panel, "ROW-WISE PALLET COUNT", (int(24 * s), int(46 * s)),
                FONT, 1.05 * s, (255, 255, 255), max(2, int(2 * s)), cv2.LINE_AA)
    cv2.putText(panel, "front x side = cartons per row", (int(24 * s), int(74 * s)),
                FONT, 0.6 * s, (170, 170, 170), max(1, int(1.4 * s)), cv2.LINE_AA)

    cols = [int(24 * s), int(200 * s), int(400 * s), int(600 * s)]
    y = header_h + int(34 * s)
    for text, x in zip(("ROW", "FRONT", "SIDE", "CARTONS"), cols):
        cv2.putText(panel, text, (x, y), FONT, 0.62 * s, (150, 150, 150),
                    max(1, int(1.5 * s)), cv2.LINE_AA)
    cv2.line(panel, (int(24 * s), y + int(12 * s)),
             (width - int(24 * s), y + int(12 * s)), (70, 70, 70), max(1, int(s)))

    for i, r in enumerate(result.rows):
        ry = y + row_h * (i + 1)
        color = _row_color(r.index)
        cv2.rectangle(panel, (int(8 * s), ry - int(26 * s)),
                      (int(16 * s), ry + int(8 * s)), color, -1)
        values = (f"{r.index}", f"{r.front_count}", f"{r.side_count}", f"{r.total}")
        for text, x in zip(values, cols):
            cv2.putText(panel, text, (x, ry), FONT, 0.78 * s, (255, 255, 255),
                        max(1, int(1.8 * s)), cv2.LINE_AA)

    fy = y + row_h * (len(result.rows) + 1) + int(26 * s)
    cv2.line(panel, (int(24 * s), fy - int(34 * s)),
             (width - int(24 * s), fy - int(34 * s)), (70, 70, 70), max(1, int(s)))
    cv2.putText(panel, "TOTAL", (cols[0], fy), FONT, 0.95 * s, (255, 255, 255),
                max(2, int(2 * s)), cv2.LINE_AA)
    cv2.putText(panel, str(result.total_count), (cols[3], fy), FONT, 1.15 * s,
                (80, 220, 90), max(2, int(2.4 * s)), cv2.LINE_AA)

    for i, w in enumerate(result.warnings):
        wy = fy + int(40 * s) + int(34 * s) * i
        text = w if len(w) < 88 else w[:85] + "..."
        cv2.putText(panel, f"! {text}", (int(24 * s), wy), FONT, 0.52 * s,
                    (80, 190, 240), max(1, int(1.3 * s)), cv2.LINE_AA)

    return panel


def to_data_uri(image: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("failed to encode annotated image")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
