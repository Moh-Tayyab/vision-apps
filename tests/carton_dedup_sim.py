#!/usr/bin/env python3
"""
Synthetic multi-camera carton scenarios.

Produces per-frame, per-camera per-carton detections for a known ground truth.
Used by:
  - ``tests/test_carton_crossing_fusion.py`` to verify the counting core
  - ``scripts/make_carton_dedup_demo.py`` to render the visual proof

The generator models ONE physical set of cartons crossing ONE red line and
projects it into several camera views. That is exactly the redundancy the
fusion layer must remove: the same carton appears once per camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from carton_crossing_fusion import Detection


@dataclass
class CameraGeometry:
    """How one camera sees the shared counting line."""

    camera_id: str
    orientation: str = "vertical"          # "vertical" (x line) | "horizontal" (y line)
    line_pos: float = 640.0
    direction: str = "left_to_right"        # direction into the truck
    lane_axis: float = 360.0                # fixed coordinate perpendicular to travel
    per_camera_delay: float = 0.0           # seconds after the true crossing
    box_w: float = 60.0
    box_h: float = 60.0
    frame_w: int = 1280
    frame_h: int = 720

    def tracker_config(self, hysteresis: float = 10.0, cooldown_frames: int = 15) -> Dict[str, object]:
        return {
            "line_pos": self.line_pos,
            "orientation": self.orientation,
            "direction": self.direction,
            "hysteresis": hysteresis,
            "cooldown_frames": cooldown_frames,
            "min_track_frames": 3,
            "min_displacement_px": 10.0,
        }


DEFAULT_CAMERAS: List[CameraGeometry] = [
    CameraGeometry("cam_top", "horizontal", line_pos=300.0,
                   direction="up_to_down", lane_axis=640.0, per_camera_delay=0.00,
                   frame_w=1280, frame_h=720),
    CameraGeometry("cam_right", "vertical", line_pos=640.0,
                   direction="left_to_right", lane_axis=360.0, per_camera_delay=0.06,
                   frame_w=1280, frame_h=720),
    CameraGeometry("cam_left", "vertical", line_pos=640.0,
                   direction="right_to_left", lane_axis=360.0, per_camera_delay=0.12,
                   frame_w=1280, frame_h=720),
]


def _axis_at(
    geom: CameraGeometry,
    cross_time: float,
    t: float,
    speed: float,
) -> float:
    """Signed distance from the line at time ``t`` (negative = before)."""
    into_positive = geom.direction in ("left_to_right", "up_to_down")
    delta = speed * (t - cross_time)
    return geom.line_pos + (delta if into_positive else -delta)


def _before_offset(geom: CameraGeometry, distance: float) -> float:
    into_positive = geom.direction in ("left_to_right", "up_to_down")
    return geom.line_pos - (distance if into_positive else -distance)


def simulate_camera(
    cartons: Sequence[Dict[str, object]],
    geom: CameraGeometry,
    *,
    fps: float = 30.0,
    duration_sec: float = 10.0,
    travel_before: float = 1.0,
    travel_after: float = 1.0,
    speed: float = 200.0,
    skip_carton_ids: Sequence[int] = (),
) -> List[List[Detection]]:
    """Return ``frames[frame_idx] -> [(track_id, bbox, conf), ...]``.

    ``cartons`` entries:
        ``{"carton_id": int, "cross_time": float | None, "lane": float}``

    ``cross_time`` is the instant the carton centre is exactly on the line.
    ``None`` means the carton approaches but never crosses.
    """
    n_frames = int(round(duration_sec * fps))
    skip = set(int(c) for c in skip_carton_ids)
    frames: List[List[Detection]] = []

    for f in range(n_frames):
        t = f / fps
        dets: List[Detection] = []
        for carton in cartons:
            cid = int(carton["carton_id"])  # type: ignore[arg-type]
            if cid in skip:
                continue
            lane = float(carton.get("lane", 0.0))  # type: ignore[arg-type]
            cross_time = carton.get("cross_time")

            if cross_time is None:
                # Visible only in the final approach, never reaches the line.
                if t < duration_sec - travel_before:
                    continue
                axis = _before_offset(geom, 80.0)
            else:
                ct = float(cross_time)  # type: ignore[arg-type]
                local_t = t - geom.per_camera_delay
                if local_t < ct - travel_before or local_t > ct + travel_after:
                    continue
                axis = _axis_at(geom, ct, local_t, speed)

            if geom.orientation == "vertical":
                cx, cy = axis, geom.lane_axis + lane
            else:
                cx, cy = geom.lane_axis + lane, axis

            bbox = (
                cx - geom.box_w / 2.0,
                cy - geom.box_h / 2.0,
                cx + geom.box_w / 2.0,
                cy + geom.box_h / 2.0,
            )
            dets.append((cid, bbox, 0.92))

        frames.append(dets)

    return frames


def simulate_scenario(
    cartons: Sequence[Dict[str, object]],
    cameras: Sequence[CameraGeometry] = DEFAULT_CAMERAS,
    *,
    fps: float = 30.0,
    duration_sec: float = 10.0,
    skip_by_camera: Optional[Dict[str, Sequence[int]]] = None,
    **kwargs: object,
) -> Dict[str, List[List[Detection]]]:
    """Project one carton set into every camera.

    ``skip_by_camera`` removes specific cartons from specific cameras, which
    simulates blind spots / dropouts.
    """
    skip_by_camera = skip_by_camera or {}
    return {
        cam.camera_id: simulate_camera(
            cartons,
            cam,
            fps=fps,
            duration_sec=duration_sec,
            skip_carton_ids=skip_by_camera.get(cam.camera_id, ()),
            **kwargs,  # type: ignore[arg-type]
        )
        for cam in cameras
    }


def evenly_spaced_cartons(
    count: int,
    first_cross_time: float = 1.0,
    interval: float = 1.0,
) -> List[Dict[str, object]]:
    """``count`` cartons crossing one after another, all in the same lane."""
    return [
        {"carton_id": i + 1, "cross_time": first_cross_time + i * interval, "lane": 0.0}
        for i in range(count)
    ]
