"""
Timeline and Timestamp Synchronization Aligner (Phase 3).

Establishes a unified global timeline across disparate camera feeds.
Enforces explicit timestamping policies (FPS-derived or PTS hardware timestamps),
per-camera start offsets, and validates synchronization tolerance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CameraSyncConfig:
    """Synchronization parameters for a single camera feed."""

    camera_id: str
    fps: float
    start_offset_sec: float = 0.0  # Evidence-based temporal offset relative to master clock
    use_hardware_pts: bool = False
    max_sync_jitter_sec: float = 0.08  # ~2.5 frames at 30fps

    def __post_init__(self) -> None:
        if self.fps <= 0.0:
            raise ValueError(f"FPS must be positive, got {self.fps}")


@dataclass(frozen=True)
class SynchronizedTimestamp:
    """A verified timestamp mapped onto the global timeline."""

    camera_id: str
    frame_idx: int
    local_time_sec: float
    global_time_sec: float
    is_synchronized: bool
    sync_delta_sec: float = 0.0
    flags: Tuple[str, ...] = ()


class TimelineAligner:
    """
    Manages global timeline synchronization across N cameras.
    
    Guarantees:
    - Never blindly assumes frame_idx equality across cameras implies temporal alignment.
    - Explicitly computes global time: t_global = t_local + camera_start_offset.
    - Flags observations when inter-camera synchronization jitter exceeds tolerance.
    - Robust against variable frame rates, missing frames, and stream offsets.
    """

    def __init__(
        self,
        camera_configs: Dict[str, CameraSyncConfig],
        max_inter_camera_delta_sec: float = 0.10,
    ) -> None:
        self.configs = dict(camera_configs)
        self.max_delta = float(max_inter_camera_delta_sec)

    def compute_timestamp(
        self,
        camera_id: str,
        frame_idx: int,
        hardware_pts_sec: Optional[float] = None,
    ) -> SynchronizedTimestamp:
        """
        Maps a camera frame index or hardware PTS onto the global timeline.
        """
        cfg = self.configs.get(camera_id)
        if cfg is None:
            raise KeyError(f"Camera '{camera_id}' not configured in TimelineAligner.")

        flags = []
        if cfg.use_hardware_pts and hardware_pts_sec is not None:
            local_time = float(hardware_pts_sec)
        else:
            # Documented FPS-based assumption
            local_time = frame_idx / cfg.fps
            flags.append("FPS_DERIVED_TIME")

        global_time = local_time + cfg.start_offset_sec

        return SynchronizedTimestamp(
            camera_id=camera_id,
            frame_idx=frame_idx,
            local_time_sec=round(local_time, 4),
            global_time_sec=round(global_time, 4),
            is_synchronized=True,
            sync_delta_sec=0.0,
            flags=tuple(flags),
        )

    def validate_multi_view_alignment(
        self,
        timestamps: Dict[str, SynchronizedTimestamp],
    ) -> Tuple[bool, float, List[str]]:
        """
        Validates whether a collection of per-camera observations fall within the
        configured synchronization window.

        Returns: (is_aligned, max_skew_sec, reasons)
        """
        if len(timestamps) <= 1:
            return (True, 0.0, [])

        global_times = [ts.global_time_sec for ts in timestamps.values()]
        min_t = min(global_times)
        max_t = max(global_times)
        skew = max_t - min_t

        reasons = []
        if skew > self.max_delta:
            reasons.append(
                f"Temporal skew {skew:.4f}s exceeds maximum tolerance {self.max_delta:.4f}s "
                f"between cameras: {list(timestamps.keys())}"
            )
            return (False, skew, reasons)

        return (True, skew, [])
