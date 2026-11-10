"""
Synchronized Multi-Camera Ring Buffer and Review Clip Generator (Phase 2 Requirement 5).

Maintains a fixed-capacity rolling circular queue of timestamped frames for each camera.
On trigger (e.g. crossing event commit or operator review flag):
- Continues recording for a configurable post-roll duration.
- Extracts synchronized frame sequences across all camera perspectives.
- Generates an auditable synchronized review clip (tiled/split-view MP4) with HUD timestamp metadata.
"""

from __future__ import annotations

import collections
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedFrame:
    """A timestamped video frame stored in the circular ring buffer."""

    camera_id: str
    frame_idx: int
    timestamp_sec: float
    frame: np.ndarray  # BGR image


@dataclass
class PendingClipTrigger:
    """Tracks an active clip generation request awaiting post-roll completion."""

    event_id: str
    trigger_time_sec: float
    post_roll_sec: float
    pre_roll_sec: float
    output_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_ready_to_export: bool = False


class SynchronizedRingBuffer:
    """
    Rolling circular frame buffer supporting N synchronized camera streams.
    
    Guarantees:
    - Fixed memory footprint: automatically discards oldest frames beyond buffer duration.
    - Synchronized temporal slicing: extracts exact [t_event - pre_roll, t_event + post_roll] clips.
    - Post-roll timing: delays export until post-roll frames have arrived.
    """

    def __init__(
        self,
        camera_ids: Sequence[str] = ("cam_top", "cam_side", "cam_flank"),
        buffer_duration_sec: float = 6.0,
        fps: float = 30.0,
    ) -> None:
        self.camera_ids = tuple(camera_ids)
        self.buffer_duration_sec = float(buffer_duration_sec)
        self.fps = float(fps)
        self.capacity_frames = int(round(self.buffer_duration_sec * self.fps))

        # Circular buffers per camera
        self._buffers: Dict[str, Deque[BufferedFrame]] = {
            cid: collections.deque(maxlen=self.capacity_frames) for cid in self.camera_ids
        }
        self._pending_triggers: List[PendingClipTrigger] = []

    @property
    def buffer_sizes(self) -> Dict[str, int]:
        return {cid: len(buf) for cid, buf in self._buffers.items()}

    def push_frame(
        self,
        camera_id: str,
        frame_idx: int,
        timestamp_sec: float,
        frame: Optional[np.ndarray],
    ) -> None:
        """Stores a new frame in the camera's circular queue."""
        if frame is None or camera_id not in self._buffers:
            return

        buffered = BufferedFrame(
            camera_id=camera_id,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            frame=frame if getattr(self, "disable_deep_copy", True) else frame.copy(),
        )
        self._buffers[camera_id].append(buffered)

    def trigger_review_clip(
        self,
        event_id: str,
        trigger_time_sec: float,
        output_path: str,
        pre_roll_sec: float = 4.0,
        post_roll_sec: float = 2.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PendingClipTrigger:
        """
        Registers an event for review-clip generation with pre-roll and post-roll timing.
        Total clip duration = pre_roll + post_roll (default 4.0s + 2.0s = 6.0s).
        """
        trigger = PendingClipTrigger(
            event_id=str(event_id),
            trigger_time_sec=float(trigger_time_sec),
            pre_roll_sec=float(pre_roll_sec),
            post_roll_sec=float(post_roll_sec),
            output_path=str(output_path),
            metadata=dict(metadata or {}),
        )
        self._pending_triggers.append(trigger)
        return trigger

    def poll_and_export_ready_clips(
        self,
        current_time_sec: float,
        force_export: bool = False,
    ) -> List[str]:
        """
        Checks pending triggers. If post-roll has elapsed, renders synchronized MP4 clip.
        Returns list of successfully exported video filepaths.
        """
        exported: List[str] = []
        remaining: List[PendingClipTrigger] = []

        for trig in self._pending_triggers:
            # Check if post-roll has elapsed
            if force_export or (current_time_sec >= (trig.trigger_time_sec + trig.post_roll_sec)):
                success = self._export_clip(trig)
                if success:
                    exported.append(trig.output_path)
            else:
                remaining.append(trig)

        self._pending_triggers = remaining
        return exported

    def _export_clip(self, trigger: PendingClipTrigger) -> bool:
        """Renders synchronized side-by-side / tiled MP4 review clip."""
        t_start = trigger.trigger_time_sec - trigger.pre_roll_sec
        t_end = trigger.trigger_time_sec + trigger.post_roll_sec

        # Extract frames in temporal slice for each camera
        slices: Dict[str, List[BufferedFrame]] = {}
        for cid in self.camera_ids:
            buf = self._buffers.get(cid, collections.deque())
            slices[cid] = [bf for bf in buf if t_start <= bf.timestamp_sec <= t_end]

        # Determine common frame count
        min_frames = min((len(s) for s in slices.values()), default=0)
        if min_frames == 0:
            logger.warning("No frames in ring buffer for clip trigger '%s' in range [%.2f, %.2f]", trigger.event_id, t_start, t_end)
            return False

        # Target pane size: 480x270 for each camera tile
        tile_w, tile_h = 480, 270
        n_cams = len(self.camera_ids)
        # Layout: horizontally stacked panes
        out_w = tile_w * n_cams
        out_h = tile_h

        out_path = Path(trigger.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (out_w, out_h))
        if not writer.isOpened():
            logger.error("Failed to open VideoWriter for '%s'", str(out_path))
            return False

        try:
            for i in range(min_frames):
                panes = []
                common_ts = 0.0
                for cid in self.camera_ids:
                    bf = slices[cid][i]
                    common_ts = bf.timestamp_sec
                    resized = cv2.resize(bf.frame, (tile_w, tile_h))
                    # Add HUD badge to pane
                    cv2.putText(
                        resized,
                        f"{cid.upper()} | t={bf.timestamp_sec:.2f}s",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                    )
                    panes.append(resized)

                # Combine horizontally
                canvas = np.hstack(panes)

                # Overlay event ID and review status banner at bottom
                is_near_trigger = abs(common_ts - trigger.trigger_time_sec) < 0.25
                banner_color = (0, 0, 255) if is_near_trigger else (50, 50, 50)
                cv2.rectangle(canvas, (0, out_h - 30), (out_w, out_h), banner_color, -1)
                cv2.putText(
                    canvas,
                    f"EVENT: {trigger.event_id} | TRIGGER: {trigger.trigger_time_sec:.2f}s | REVIEW CLIP (6s RING BUFFER)",
                    (20, out_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

                writer.write(canvas)
            return True
        except Exception as e:
            logger.error("Failed writing review clip '%s': %s", str(out_path), e)
            return False
        finally:
            writer.release()
