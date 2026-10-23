#!/usr/bin/env python3
"""
Carton-level red-line crossing counter with cross-camera de-duplication.

This module implements the counting rule requested for the warehouse scenario:

    A carton is counted as ONE only when it crosses the red line in the
    direction of the truck. The same carton seen crossing the line by two or
    three cameras is ONE carton, never two or three.

Two layers
----------
1. ``CartonLineTracker`` - one per camera. It consumes per-carton detections
   (one box per carton, NOT one box per worker) and emits a ``CartonCrossing``
   when the centre of a tracked carton crosses the line. It uses hysteresis,
   a per-track cooldown and a minimum displacement so that a carton vibrating
   on the line is not counted repeatedly.

2. ``CartonCrossingFusion`` - one for the whole camera set. It consolidates
   the crossing streams. One camera is the ANCHOR: every anchor crossing is
   one physical carton. Every crossing from the other cameras is attached to
   at most one anchor carton (nearest in time inside the window). A crossing
   that cannot be attached to any anchor becomes its own carton, so a carton
   that the anchor camera missed is still counted (camera dropout safety).

Why anchor + one-to-one instead of "cluster everything in the window"
----------------------------------------------------------------------
Naive temporal clustering merges N cartons that cross together into one blob
(undercount). Naive summing counts one carton N times (overcount). Anchoring
and one-to-one assignment gives both guarantees:

    one carton seen by 3 cameras  -> 1
    N cartons seen by 3 cameras   -> N

Only one crossing per camera can belong to a given carton, which is exactly
the redundancy that has to be removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

BBox = Tuple[float, float, float, float]
# (track_id, (x1, y1, x2, y2), confidence)
Detection = Tuple[int, Sequence[float], float]

IN = "IN"
OUT = "OUT"

_HORIZONTAL_DIRECTIONS = {"up_to_down", "down_to_up"}
_VALID_DIRECTIONS = {"left_to_right", "right_to_left", "up_to_down", "down_to_up"}


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------

@dataclass
class CartonCrossing:
    """One carton crossing the red line, as seen by ONE camera."""

    camera_id: str
    track_id: int
    timestamp_sec: float
    direction: str          # "IN" (towards the truck) or "OUT"
    bbox: BBox
    confidence: float = 1.0

    def as_dict(self) -> Dict[str, object]:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "direction": self.direction,
            "bbox": [round(v, 1) for v in self.bbox],
            "confidence": round(self.confidence, 4),
        }


@dataclass
class FusedCarton:
    """One physical carton, assembled from 1..N camera crossings."""

    carton_id: int
    timestamp_sec: float
    direction: str
    evidences: List[CartonCrossing] = field(default_factory=list)

    @property
    def cameras_seen(self) -> List[str]:
        seen: List[str] = []
        for ev in self.evidences:
            if ev.camera_id not in seen:
                seen.append(ev.camera_id)
        return seen

    @property
    def evidence_count(self) -> int:
        return len(self.evidences)

    @property
    def corroborated(self) -> bool:
        return len(self.cameras_seen) >= 2

    def as_dict(self) -> Dict[str, object]:
        return {
            "carton_id": self.carton_id,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "direction": self.direction,
            "cameras_seen": self.cameras_seen,
            "evidence_count": self.evidence_count,
            "corroborated": self.corroborated,
            "evidences": [e.as_dict() for e in self.evidences],
        }


@dataclass
class FusionSummary:
    """Result of consolidating the per-camera crossing streams."""

    total_unique_cartons: int
    cartons: List[FusedCarton]
    raw_crossings_by_camera: Dict[str, int]
    raw_crossings_total: int
    duplicates_removed: int
    uncorroborated_cartons: int
    # Directional balance. ``net_cartons = loaded_cartons - returned_cartons``
    # is the number that should be shown to the operator (gross IN and OUT are
    # kept separately so a returned carton is visible, not silently hidden).
    loaded_cartons: int = 0
    returned_cartons: int = 0
    net_cartons: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_unique_cartons": self.total_unique_cartons,
            "loaded_cartons": self.loaded_cartons,
            "returned_cartons": self.returned_cartons,
            "net_cartons": self.net_cartons,
            "raw_crossings_by_camera": dict(self.raw_crossings_by_camera),
            "raw_crossings_total": self.raw_crossings_total,
            "duplicates_removed": self.duplicates_removed,
            "uncorroborated_cartons": self.uncorroborated_cartons,
            "cartons": [c.as_dict() for c in self.cartons],
        }


@dataclass
class _TrackState:
    side: Optional[str] = None
    first_axis: float = 0.0
    last_axis: float = 0.0
    last_frame: int = -1
    frames_seen: int = 0
    last_cross_frame: int = -10 ** 9


# ---------------------------------------------------------------------------
# Layer 1 - one camera
# ---------------------------------------------------------------------------

class CartonLineTracker:
    """Tracks individual cartons and emits a crossing when one passes the line.

    Two distinct mechanisms keep the count stable:

    1. ByteTrack persistence (``tracker="bytetrack.yaml", persist=True`` in the
       YOLO pipeline) keeps a stable track ID across frames so a carton is not
       lost between detections. This is handled in ``multi_angle_pipeline.py``,
       not here.

    2. Tripwire hysteresis (this class): a dead zone around the red line. A
       carton must be clearly on one side before a crossing is accepted, so
       jitter on the line does not emit repeated crossings.

    Parameters
    ----------
    camera_id:
        Stable name used by the fusion layer.
    line_pos:
        Position of the red line in the camera's own pixels (x for a vertical
        line, y for a horizontal line).
    orientation:
        ``"vertical"`` (cartons move left/right) or ``"horizontal"``
        (cartons move up/down, e.g. the top-down camera).
    direction:
        Direction that means "into the truck": ``left_to_right``,
        ``right_to_left``, ``up_to_down`` or ``down_to_up``.
    hysteresis:
        Tripwire dead-zone half-width in pixels. A carton must be clearly on
        one side before a crossing is accepted. Distinct from tracker
        persistence (ByteTrack), which keeps IDs across frames.
    cooldown_frames:
        Minimum frames between two crossings of the same track.
    min_track_frames:
        The track must have been observed this many times before it may count,
        so a single noisy detection cannot create a carton.
    min_displacement_px:
        Minimum travel along the counting axis between first sight and the
        crossing, so jitter on the line is not a crossing.
    max_inactive_frames:
        Tracks unseen for longer than this are forgotten (memory bound).
    """

    def __init__(
        self,
        camera_id: str,
        line_pos: float,
        orientation: str = "vertical",
        direction: str = "left_to_right",
        hysteresis: float = 12.0,
        cooldown_frames: int = 15,
        min_track_frames: int = 3,
        min_displacement_px: float = 10.0,
        max_inactive_frames: int = 90,
    ) -> None:
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(f"orientation must be vertical|horizontal, got {orientation!r}")
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"unknown direction {direction!r}")
        if hysteresis < 0:
            raise ValueError("hysteresis must be >= 0")

        self.camera_id = camera_id
        self.line_pos = float(line_pos)
        self.orientation = orientation
        self.direction = direction
        self.hysteresis = float(hysteresis)
        self.cooldown_frames = int(cooldown_frames)
        self.min_track_frames = int(min_track_frames)
        self.min_displacement_px = float(min_displacement_px)
        self.max_inactive_frames = int(max_inactive_frames)

        self._tracks: Dict[int, _TrackState] = {}
        self.crossings: List[CartonCrossing] = []

    # -- geometry helpers ---------------------------------------------------

    def _axis(self, bbox: Sequence[float]) -> float:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        return (y1 + y2) / 2.0 if self.orientation == "horizontal" else (x1 + x2) / 2.0

    def _side(self, value: float) -> Optional[str]:
        """Return 'before', 'after' or None (inside the hysteresis dead zone)."""
        line, hyst = self.line_pos, self.hysteresis
        if self.direction in ("left_to_right", "up_to_down"):
            if value >= line + hyst:
                return "after"
            if value <= line - hyst:
                return "before"
            return None
        # right_to_left / down_to_up
        if value <= line - hyst:
            return "after"
        if value >= line + hyst:
            return "before"
        return None

    # -- main update --------------------------------------------------------

    def update(
        self,
        detections: Iterable[Detection],
        frame_idx: int,
        timestamp_sec: float,
    ) -> List[CartonCrossing]:
        """Feed one frame of per-carton detections. Returns new crossings.

        ``detections`` must contain ONE box per carton. A box that contains
        two stacked cartons is one detection and will be counted once; the
        detector is responsible for splitting stacks.
        """
        emitted: List[CartonCrossing] = []

        for det in detections:
            if len(det) < 2:
                continue
            track_id = int(det[0])
            bbox = tuple(float(v) for v in det[1][:4])
            if len(bbox) != 4:
                continue
            conf = float(det[2]) if len(det) > 2 else 1.0

            axis = self._axis(bbox)
            state = self._tracks.get(track_id)
            if state is None:
                state = _TrackState(first_axis=axis)
                self._tracks[track_id] = state
            state.frames_seen += 1
            state.last_axis = axis
            state.last_frame = frame_idx

            curr_side = self._side(axis)
            if state.side is None:
                # First usable observation: remember the side, do not count.
                if curr_side is not None:
                    state.side = curr_side
                continue

            if curr_side is None or curr_side == state.side:
                # Inside the dead zone, or still on the same side: not a cross.
                continue

            # Hysteresis has been cleared on the opposite side.
            if state.frames_seen < self.min_track_frames:
                state.side = curr_side
                continue
            if abs(axis - state.first_axis) < self.min_displacement_px:
                continue
            if (frame_idx - state.last_cross_frame) < self.cooldown_frames:
                state.side = curr_side
                continue

            direction = IN if (state.side == "before" and curr_side == "after") else OUT
            crossing = CartonCrossing(
                camera_id=self.camera_id,
                track_id=track_id,
                timestamp_sec=float(timestamp_sec),
                direction=direction,
                bbox=bbox,  # type: ignore[arg-type]
                confidence=conf,
            )
            emitted.append(crossing)
            self.crossings.append(crossing)

            state.last_cross_frame = frame_idx
            state.side = curr_side

        self._purge(frame_idx)
        return emitted

    def _purge(self, frame_idx: int) -> None:
        stale = [
            tid for tid, st in self._tracks.items()
            if (frame_idx - st.last_frame) > self.max_inactive_frames
        ]
        for tid in stale:
            del self._tracks[tid]

    def reset(self) -> None:
        self._tracks.clear()
        self.crossings.clear()


# ---------------------------------------------------------------------------
# Layer 2 - all cameras
# ---------------------------------------------------------------------------

class CartonCrossingFusion:
    """Consolidates per-camera carton crossings into one physical count.

    Parameters
    ----------
    time_window_sec:
        Maximum time between two cameras seeing the same carton. Must be
        smaller than the gap between two consecutive cartons.
    count_directions:
        Which crossing directions count. Use ``("IN",)`` for a one-way flow.
    min_cameras:
        Minimum number of cameras that must agree for a carton to be counted.
        ``1`` never loses a carton; ``2`` rejects single-camera false
        positives but also loses cartons when two cameras drop out.
    anchor_camera:
        Camera whose crossings define the carton count. By default the camera
        with the most crossings is used (tie-break: registration order).
    same_camera_dedup_sec:
        Optional. Drop a later crossing from the SAME camera if it is within
        this many seconds and geometrically close to an already kept crossing.
        Useful when a tracker loses a carton and re-identifies it. ``0``
        disables it (two real cartons side by side stay separate).
    same_camera_center_frac:
        Geometry threshold for same-camera de-duplication, as a fraction of
        the bbox diagonal.
    """

    def __init__(
        self,
        time_window_sec: float = 0.8,
        count_directions: Sequence[str] = (IN,),
        min_cameras: int = 1,
        anchor_camera: Optional[str] = None,
        same_camera_dedup_sec: float = 0.0,
        same_camera_center_frac: float = 0.75,
    ) -> None:
        if time_window_sec <= 0:
            raise ValueError("time_window_sec must be > 0")
        if min_cameras < 1:
            raise ValueError("min_cameras must be >= 1")
        self.time_window_sec = float(time_window_sec)
        self.count_directions = tuple(d.upper() for d in count_directions)
        self.min_cameras = int(min_cameras)
        self.anchor_camera = anchor_camera
        self.same_camera_dedup_sec = float(same_camera_dedup_sec)
        self.same_camera_center_frac = float(same_camera_center_frac)

        self._camera_order: List[str] = []
        self._events: List[CartonCrossing] = []
        self._last_fused: Optional[FusionSummary] = None

    # -- ingestion ----------------------------------------------------------

    def add_camera(self, camera_id: str) -> None:
        if camera_id not in self._camera_order:
            self._camera_order.append(camera_id)

    def ingest(self, crossing: CartonCrossing) -> bool:
        """Add one crossing. Returns False if the direction is invalid.

        OUT crossings are always retained so the net balance
        (``loaded - returned``) can be reported, even when only IN is counted
        in ``total_unique_cartons``.
        """
        if crossing.direction not in (IN, OUT):
            return False
        self.add_camera(crossing.camera_id)
        self._events.append(crossing)
        return True

    def ingest_many(self, crossings: Iterable[CartonCrossing]) -> int:
        added = 0
        for c in crossings:
            if self.ingest(c):
                added += 1
        return added

    def reset(self) -> None:
        self._camera_order.clear()
        self._events.clear()
        self._last_fused = None

    def prune_before(self, timestamp_sec: float) -> int:
        """Forget crossings older than ``timestamp_sec`` (bounded memory).

        Only safe once those crossings have already been reported and can no
        longer pair with a future crossing, i.e. once they are older than the
        fusion window. The next ``fuse()`` call will no longer include them, so
        a live caller should keep its own cumulative total and add only the
        newly matured cartons.

        Returns the number of crossings dropped.
        """
        cutoff = float(timestamp_sec)
        before = len(self._events)
        self._events = [e for e in self._events if e.timestamp_sec >= cutoff]
        return before - len(self._events)

    # -- de-duplication -----------------------------------------------------

    @staticmethod
    def _bbox_center(bbox: BBox) -> Tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    @staticmethod
    def _bbox_diag(bbox: BBox) -> float:
        w = max(1e-6, bbox[2] - bbox[0])
        h = max(1e-6, bbox[3] - bbox[1])
        return (w * w + h * h) ** 0.5

    def _dedupe_same_camera(self, events: List[CartonCrossing]) -> List[CartonCrossing]:
        """Remove repeated crossings of the same carton within one camera.

        The tracker already emits at most one crossing per track, so this only
        matters when a tracker re-identifies the same carton under a new ID.
        """
        if self.same_camera_dedup_sec <= 0:
            return events

        kept: List[CartonCrossing] = []
        for ev in sorted(events, key=lambda e: e.timestamp_sec):
            duplicate = False
            for prev in kept:
                if prev.camera_id != ev.camera_id:
                    continue
                if abs(prev.timestamp_sec - ev.timestamp_sec) > self.same_camera_dedup_sec:
                    continue
                pc = self._bbox_center(prev.bbox)
                ec = self._bbox_center(ev.bbox)
                dist = ((pc[0] - ec[0]) ** 2 + (pc[1] - ec[1]) ** 2) ** 0.5
                scale = max(self._bbox_diag(prev.bbox), self._bbox_diag(ev.bbox))
                if dist <= self.same_camera_center_frac * scale:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(ev)
        return kept

    # -- fusion -------------------------------------------------------------

    def _choose_anchor(self, events: List[CartonCrossing]) -> Optional[str]:
        if self.anchor_camera is not None and any(
            e.camera_id == self.anchor_camera for e in events
        ):
            return self.anchor_camera
        counts: Dict[str, int] = {}
        for ev in events:
            counts[ev.camera_id] = counts.get(ev.camera_id, 0) + 1
        if not counts:
            return None

        order = {cam: i for i, cam in enumerate(self._camera_order)}
        return max(counts, key=lambda cam: (counts[cam], -order.get(cam, 10 ** 6)))

    def _match_ordered(
        self,
        cartons: List[FusedCarton],
        events: List[CartonCrossing],
    ) -> List[Tuple[int, int]]:
        """Order-preserving matching between ONE camera's events and cartons.

        Cartons cross the line in the same physical order in every camera, so
        an order-preserving alignment is the correct model. A greedy
        nearest-in-time match can steal a carton's partner when the per-camera
        delay is comparable to the gap between cartons, leaving a later event
        with nowhere to attach and creating a phantom carton.

        Returns ``[(event_index, carton_index), ...]``.
        """
        n, m = len(events), len(cartons)
        if n == 0 or m == 0:
            return []

        # DP cell = (matches, -total_time_error). More matches wins; on equal
        # matches the smaller total error wins.
        neg = (-1, -1e18)
        dp: List[List[Tuple[int, float]]] = [[neg] * (m + 1) for _ in range(n + 1)]
        choice: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = (0, 0.0)
        for j in range(m + 1):
            dp[0][j] = (0, 0.0)

        for i in range(1, n + 1):
            ev = events[i - 1]
            for j in range(1, m + 1):
                carton = cartons[j - 1]
                best = dp[i - 1][j]
                ch = 0  # skip this event
                if dp[i][j - 1] > best:
                    best = dp[i][j - 1]
                    ch = 1  # skip this carton
                if (
                    ev.direction == carton.direction
                    and ev.camera_id not in carton.cameras_seen
                ):
                    diff = abs(carton.timestamp_sec - ev.timestamp_sec)
                    if diff <= self.time_window_sec:
                        prev = dp[i - 1][j - 1]
                        cand = (prev[0] + 1, prev[1] - diff)
                        if cand > best:
                            best = cand
                            ch = 2  # match
                dp[i][j] = best
                choice[i][j] = ch

        matches: List[Tuple[int, int]] = []
        i, j = n, m
        while i > 0 and j > 0:
            ch = choice[i][j]
            if ch == 2:
                matches.append((i - 1, j - 1))
                i -= 1
                j -= 1
            elif ch == 0:
                i -= 1
            else:
                j -= 1
        matches.reverse()
        return matches

    def _fuse_events(self, events: List[CartonCrossing]) -> List[FusedCarton]:
        """Fuse a single-direction event stream into physical cartons."""
        events = self._dedupe_same_camera(events)
        anchor = self._choose_anchor(events)
        cartons: List[FusedCarton] = []
        if anchor is None:
            return cartons

        anchor_events = sorted(
            (e for e in events if e.camera_id == anchor),
            key=lambda e: e.timestamp_sec,
        )
        for ev in anchor_events:
            cartons.append(FusedCarton(
                carton_id=0,
                timestamp_sec=ev.timestamp_sec,
                direction=ev.direction,
                evidences=[ev],
            ))

        # Attach every other camera's crossings in registration order. Each
        # camera is aligned against the cartons built so far with an
        # order-preserving match, so a large per-camera delay cannot make one
        # carton steal another carton's evidence.
        others_by_cam: Dict[str, List[CartonCrossing]] = {}
        for ev in events:
            if ev.camera_id != anchor:
                others_by_cam.setdefault(ev.camera_id, []).append(ev)

        cam_order = [c for c in self._camera_order if c in others_by_cam]
        cam_order += [c for c in others_by_cam if c not in cam_order]

        for cam in cam_order:
            cam_events = sorted(others_by_cam[cam], key=lambda e: e.timestamp_sec)
            matches = self._match_ordered(cartons, cam_events)
            matched: Set[int] = set()
            for ei, cj in matches:
                cartons[cj].evidences.append(cam_events[ei])
                matched.add(ei)
            # An unmatched crossing means the anchor camera missed this carton
            # (or it is a false positive). Keep it as its own carton;
            # min_cameras decides whether it survives.
            for ei, ev in enumerate(cam_events):
                if ei not in matched:
                    cartons.append(FusedCarton(
                        carton_id=0,
                        timestamp_sec=ev.timestamp_sec,
                        direction=ev.direction,
                        evidences=[ev],
                    ))
            cartons.sort(key=lambda c: c.timestamp_sec)

        cartons.sort(key=lambda c: c.timestamp_sec)
        return cartons

    def fuse(self, now_sec: Optional[float] = None) -> FusionSummary:
        """Consolidate everything ingested so far.

        If ``now_sec`` is given, only crossings older than
        ``now_sec - time_window_sec`` are used, so a live HUD never commits a
        carton before its duplicate-elimination window has elapsed.

        IN and OUT are fused independently, so a carton carried into the truck
        and later brought back is one loaded carton plus one returned carton,
        not one ambiguous crossing. ``net_cartons = loaded - returned`` is the
        operator-facing balance.
        """
        all_events = list(self._events)
        if now_sec is not None:
            cutoff = float(now_sec) - self.time_window_sec
            all_events = [e for e in all_events if e.timestamp_sec <= cutoff]

        counted_events = [e for e in all_events if e.direction in self.count_directions]

        # Raw counts are measured BEFORE any de-duplication so the report can
        # show exactly how much redundancy the fusion removed.
        raw_by_cam: Dict[str, int] = {}
        for ev in counted_events:
            raw_by_cam[ev.camera_id] = raw_by_cam.get(ev.camera_id, 0) + 1

        in_cartons = self._fuse_events([e for e in all_events if e.direction == IN])
        out_cartons = self._fuse_events([e for e in all_events if e.direction == OUT])

        cartons: List[FusedCarton] = []
        if IN in self.count_directions:
            cartons.extend(in_cartons)
        if OUT in self.count_directions:
            cartons.extend(out_cartons)
        cartons.sort(key=lambda c: c.timestamp_sec)
        for i, carton in enumerate(cartons, start=1):
            carton.carton_id = i

        counted = [c for c in cartons if len(c.cameras_seen) >= self.min_cameras]
        uncorroborated = sum(1 for c in cartons if len(c.cameras_seen) < self.min_cameras)

        loaded = sum(1 for c in in_cartons if len(c.cameras_seen) >= self.min_cameras)
        returned = sum(1 for c in out_cartons if len(c.cameras_seen) >= self.min_cameras)

        summary = FusionSummary(
            total_unique_cartons=len(counted),
            cartons=cartons,
            raw_crossings_by_camera=raw_by_cam,
            raw_crossings_total=sum(raw_by_cam.values()),
            duplicates_removed=max(0, sum(raw_by_cam.values()) - len(cartons)),
            uncorroborated_cartons=uncorroborated,
            loaded_cartons=loaded,
            returned_cartons=returned,
            net_cartons=loaded - returned,
        )
        self._last_fused = summary
        return summary

    @property
    def last_summary(self) -> Optional[FusionSummary]:
        return self._last_fused


# ---------------------------------------------------------------------------
# Convenience: run a whole scenario offline
# ---------------------------------------------------------------------------

def run_scenario(
    per_camera_detections: Dict[str, List[Tuple[int, List[Detection]]]],
    tracker_configs: Dict[str, Dict[str, object]],
    *,
    fps: float = 30.0,
    time_window_sec: float = 0.8,
    min_cameras: int = 1,
    anchor_camera: Optional[str] = None,
) -> FusionSummary:
    """Run tracker + fusion over pre-recorded per-frame detections.

    ``per_camera_detections[cam][frame_index]`` is the detection list for that
    frame. Used by the verification tests and the demo renderer.
    """
    fusion = CartonCrossingFusion(
        time_window_sec=time_window_sec,
        min_cameras=min_cameras,
        anchor_camera=anchor_camera,
    )
    trackers: Dict[str, CartonLineTracker] = {}
    for cam, cfg in tracker_configs.items():
        trackers[cam] = CartonLineTracker(camera_id=cam, **cfg)  # type: ignore[arg-type]
        fusion.add_camera(cam)

    n_frames = max((len(v) for v in per_camera_detections.values()), default=0)
    for frame_idx in range(n_frames):
        ts = frame_idx / fps
        for cam, frames in per_camera_detections.items():
            if frame_idx >= len(frames):
                continue
            fusion.ingest_many(trackers[cam].update(frames[frame_idx], frame_idx, ts))

    return fusion.fuse()
