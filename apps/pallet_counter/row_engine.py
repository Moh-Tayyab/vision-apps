"""Row-wise pallet carton counting from two adjacent 90-degree views.

The pallet is a stack of horizontal rows; each row is a rectangular grid of
cartons. A single view only ever shows one side of that grid, so counting
faces in one image cannot give the pallet total. Two views taken 90 degrees
apart give both grid dimensions:

    row total   = N_front x N_side
    pallet total = sum over rows

Rows are recovered per view by clustering detections vertically. A photo taken
square-on has every carton in a row at the same image height, but an oblique
photo makes the row recede: the far end of the row sits higher in the frame
than the near end. Clustering on raw y therefore splits or merges rows on any
shot that is not perfectly square-on. Instead we fit the stack's row direction
and cluster along the axis perpendicular to it, which is stable under tilt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Box:
    """One detected carton face."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def to_dict(self) -> dict:
        return {
            "bbox": [round(v, 1) for v in (self.x1, self.y1, self.x2, self.y2)],
            "confidence": round(self.confidence, 4),
        }


@dataclass
class Row:
    """One horizontal row of cartons within a single view."""
    index: int                      # 1 = top row
    boxes: List[Box]
    tilt_deg: float                 # row direction relative to horizontal

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def y_range(self) -> Tuple[float, float]:
        return (min(b.y1 for b in self.boxes), max(b.y2 for b in self.boxes))

    def to_dict(self) -> dict:
        y1, y2 = self.y_range
        return {
            "row": self.index,
            "count": self.count,
            "tilt_deg": round(self.tilt_deg, 2),
            "y_range": [round(y1, 1), round(y2, 1)],
            "boxes": [b.to_dict() for b in self.boxes],
        }


@dataclass
class ViewResult:
    """Row breakdown for one camera view."""
    name: str
    rows: List[Row]
    tilt_deg: float
    image_size: Tuple[int, int]     # (width, height)
    total_faces: int

    def counts(self) -> List[int]:
        return [r.count for r in self.rows]

    def to_dict(self) -> dict:
        return {
            "view": self.name,
            "rows_detected": len(self.rows),
            "total_faces": self.total_faces,
            "tilt_deg": round(self.tilt_deg, 2),
            "image_size": list(self.image_size),
            "rows": [r.to_dict() for r in self.rows],
        }


@dataclass
class RowPairing:
    """One row of the pallet, counted from both views."""
    index: int
    front_count: int
    side_count: int
    total: int
    front_row: Optional[int] = None   # source row index in each view
    side_row: Optional[int] = None
    note: str = ""

    def to_dict(self) -> dict:
        d = {
            "row": self.index,
            "front_count": self.front_count,
            "side_count": self.side_count,
            "row_total": self.total,
            "formula": f"{self.front_count} x {self.side_count} = {self.total}",
        }
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class PalletResult:
    """Full two-view pallet count."""
    total_count: int
    rows: List[RowPairing]
    front: ViewResult
    side: ViewResult
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_count": self.total_count,
            "rows_counted": len(self.rows),
            "rows": [r.to_dict() for r in self.rows],
            "front_view": self.front.to_dict(),
            "side_view": self.side.to_dict(),
            "warnings": self.warnings,
            "processing_time_ms": round(self.processing_time_ms, 1),
            "method": "row_wise_dual_view_multiplication",
        }


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

def _iou(a: Box, b: Box) -> float:
    xa, ya = max(a.x1, b.x1), max(a.y1, b.y1)
    xb, yb = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def suppress_duplicates(boxes: Sequence[Box], iou_threshold: float = 0.45) -> List[Box]:
    """Drop lower-confidence boxes that substantially overlap a stronger one.

    The detector is reliable on square-on shots but can emit a second box for
    one carton where a seam is ambiguous. Oblique views make axis-aligned boxes
    clip each other's corners harmlessly, so the threshold is deliberately high:
    only genuine duplicates are removed, not neighbouring cartons.
    """
    ordered = sorted(boxes, key=lambda b: b.confidence, reverse=True)
    kept: List[Box] = []
    for cand in ordered:
        if all(_iou(cand, k) < iou_threshold for k in kept):
            kept.append(cand)
    return kept


# ---------------------------------------------------------------------------
# Row clustering
# ---------------------------------------------------------------------------

def _projections(boxes: Sequence[Box], tilt_deg: float) -> np.ndarray:
    """Distance of each box centre along the axis normal to the row direction.

    At tilt 0 this is just the centre y; at other tilts it is the coordinate in
    which a receding row collapses to a single value.
    """
    theta = np.radians(tilt_deg)
    return np.array([-b.cx * np.sin(theta) + b.cy * np.cos(theta) for b in boxes])


def _separation_score(boxes: Sequence[Box], tilt_deg: float) -> float:
    """How cleanly a given tilt separates the stack into rows.

    A correct tilt makes every row project to a tight band with clear empty
    space between bands. Scoring that directly -- widest between-row gap minus
    the worst within-row spread -- is far more reliable than measuring the
    angle geometrically, because neighbouring rows sit closer together than a
    carton is tall, so any local angle estimate is easily contaminated by boxes
    from the row above or below.
    """
    proj = np.sort(_projections(boxes, tilt_deg))
    if len(proj) < 2:
        return 0.0

    median_h = float(np.median([b.h for b in boxes]))
    gaps = np.diff(proj)
    threshold = 0.55 * median_h

    # Split into candidate rows at every large gap.
    split_at = np.flatnonzero(gaps > threshold)
    if split_at.size == 0:
        return -float(proj[-1] - proj[0])  # everything merged: worst case

    groups: List[np.ndarray] = []
    start = 0
    for idx in split_at:
        groups.append(proj[start:idx + 1])
        start = idx + 1
    groups.append(proj[start:])

    worst_spread = max(float(g[-1] - g[0]) for g in groups)
    smallest_gap = float(np.min(gaps[split_at]))
    return smallest_gap - worst_spread


def estimate_tilt(
    boxes: Sequence[Box],
    max_tilt_deg: float = 35.0,
    step_deg: float = 1.0,
) -> float:
    """Find the row direction by searching for the tilt that separates rows best.

    Rows recede in an oblique photo, so boxes in one row sit at visibly
    different image heights. Sweeping the candidate tilt and keeping the angle
    with the cleanest row separation recovers the true row direction even when
    rows are spaced closer than a carton's height.
    """
    if len(boxes) < 3:
        return 0.0

    candidates = np.arange(-max_tilt_deg, max_tilt_deg + step_deg, step_deg)
    scores = [(_separation_score(boxes, t), abs(t), t) for t in candidates]
    # Best separation wins; ties break toward the smaller tilt so a square-on
    # photo is not given a spurious angle.
    scores.sort(key=lambda s: (-s[0], s[1]))
    return float(scores[0][2])


def cluster_rows(
    boxes: Sequence[Box],
    tilt_deg: Optional[float] = None,
    gap_ratio: float = 0.55,
) -> Tuple[List[Row], float]:
    """Group detections into rows, ordered top to bottom.

    Boxes are projected onto the axis perpendicular to the row direction, so a
    receding row collapses to a single tight cluster regardless of tilt. Rows
    are split wherever the gap between consecutive projections exceeds
    ``gap_ratio`` of the median box height.
    """
    if not boxes:
        return [], 0.0

    tilt = estimate_tilt(boxes) if tilt_deg is None else tilt_deg
    proj = list(zip(_projections(boxes, tilt), boxes))
    proj.sort(key=lambda t: t[0])

    median_h = float(np.median([b.h for b in boxes]))
    threshold = gap_ratio * median_h

    clusters: List[List[Box]] = [[proj[0][1]]]
    for prev, cur in zip(proj, proj[1:]):
        if cur[0] - prev[0] > threshold:
            clusters.append([cur[1]])
        else:
            clusters[-1].append(cur[1])

    rows: List[Row] = []
    for i, cluster in enumerate(clusters, start=1):
        cluster.sort(key=lambda b: b.cx)
        rows.append(Row(index=i, boxes=cluster, tilt_deg=tilt))
    return rows, tilt


def build_view(name: str, boxes: Sequence[Box], image_size: Tuple[int, int]) -> ViewResult:
    """Run suppression and row clustering for one view."""
    kept = suppress_duplicates(boxes)
    rows, tilt = cluster_rows(kept)
    return ViewResult(
        name=name,
        rows=rows,
        tilt_deg=tilt,
        image_size=image_size,
        total_faces=len(kept),
    )


# ---------------------------------------------------------------------------
# Cross-view row pairing
# ---------------------------------------------------------------------------

def _normalised_bands(view: ViewResult) -> List[Tuple[float, float]]:
    """Each row's vertical band, rescaled so the stack spans 0..1.

    Normalising against the stack rather than the image lets rows be matched
    between two photos taken at different distances or framings.
    """
    if not view.rows:
        return []
    top = min(r.y_range[0] for r in view.rows)
    bottom = max(r.y_range[1] for r in view.rows)
    span = max(1.0, bottom - top)
    return [((r.y_range[0] - top) / span, (r.y_range[1] - top) / span) for r in view.rows]


def pair_rows(front: ViewResult, side: ViewResult) -> Tuple[List[RowPairing], List[str]]:
    """Match rows between the two views and compute each row's carton count.

    Rows are matched by normalised vertical position rather than by list index,
    so a view that misses a row does not shift every later pairing.
    """
    warnings: List[str] = []
    f_bands = _normalised_bands(front)
    s_bands = _normalised_bands(side)

    if not front.rows and not side.rows:
        return [], ["No cartons detected in either view."]

    # One view empty: fall back to the other view's face count. This is a 2D
    # face count, not a pallet total, so it is flagged rather than reported
    # silently as if both dimensions had been measured.
    if not front.rows or not side.rows:
        present = front if front.rows else side
        missing = "side" if front.rows else "front"
        warnings.append(
            f"No cartons detected in the {missing} view; reporting visible faces "
            f"from the {present.name} view only. This is NOT a pallet total."
        )
        return (
            [RowPairing(index=r.index, front_count=r.count, side_count=1,
                        total=r.count, note="single view only")
             for r in present.rows],
            warnings,
        )

    if len(front.rows) != len(side.rows):
        warnings.append(
            f"Views disagree on row count (front {len(front.rows)}, "
            f"side {len(side.rows)}); rows matched by vertical position."
        )

    def overlap(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))

    pairings: List[RowPairing] = []
    used_side: set[int] = set()
    n_rows = max(len(front.rows), len(side.rows))

    median_front = int(round(float(np.median([r.count for r in front.rows]))))
    median_side = int(round(float(np.median([r.count for r in side.rows]))))

    for i in range(n_rows):
        f_row = front.rows[i] if i < len(front.rows) else None

        s_row = None
        if f_row is not None:
            best, best_ov = None, 0.0
            for j, band in enumerate(s_bands):
                if j in used_side:
                    continue
                ov = overlap(f_bands[i], band)
                if ov > best_ov:
                    best, best_ov = j, ov
            if best is not None:
                s_row = side.rows[best]
                used_side.add(best)
        else:
            for j in range(len(side.rows)):
                if j not in used_side:
                    s_row = side.rows[j]
                    used_side.add(j)
                    break

        note = ""
        if f_row is None:
            n1, n2 = median_front, s_row.count
            note = "front row missing; substituted median front count"
        elif s_row is None:
            n1, n2 = f_row.count, median_side
            note = "side row missing; substituted median side count"
        else:
            n1, n2 = f_row.count, s_row.count

        pairings.append(RowPairing(
            index=i + 1,
            front_count=n1,
            side_count=n2,
            total=n1 * n2,
            front_row=f_row.index if f_row else None,
            side_row=s_row.index if s_row else None,
            note=note,
        ))

    return pairings, warnings


def count_pallet(
    front_boxes: Sequence[Box],
    front_size: Tuple[int, int],
    side_boxes: Sequence[Box],
    side_size: Tuple[int, int],
    elapsed_ms: float = 0.0,
) -> PalletResult:
    """Count a pallet from detections in two adjacent 90-degree views."""
    front = build_view("front", front_boxes, front_size)
    side = build_view("side", side_boxes, side_size)
    rows, warnings = pair_rows(front, side)

    for view in (front, side):
        if abs(view.tilt_deg) > 12.0:
            warnings.append(
                f"{view.name} view is tilted {view.tilt_deg:.1f} degrees; "
                f"a square-on photo gives more reliable rows."
            )

    return PalletResult(
        total_count=sum(r.total for r in rows),
        rows=rows,
        front=front,
        side=side,
        warnings=warnings,
        processing_time_ms=elapsed_ms,
    )
