"""Build per-camera observations without assuming uncalibrated global identity."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .geometric_adapter import (
    GeometricCalibrationManager,
)
from .per_camera_tracker import (
    CartonTracklet,
    TrackStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class AssociatedCartonCluster:
    """A camera-local observation; not a global carton identity."""

    cluster_id: str
    global_timestamp_sec: float
    observations: Dict[str, CartonTracklet] = field(default_factory=dict)  # camera_id -> tracklet
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    geometry_verified: bool = False
    world_dock_xy: Optional[Tuple[float, float]] = None

    @property
    def camera_ids(self) -> List[str]:
        return sorted(self.observations.keys())

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def mean_confidence(self) -> float:
        if not self.observations:
            return 0.0
        return sum(t.current_confidence for t in self.observations.values()) / len(self.observations)


class CrossCameraAssociator:
    """
    Associates per-camera tracklets across multiple perspectives.
    """

    def __init__(
        self,
        time_window_sec: float = 0.35,
        max_dock_dist_meters: float = 0.60,
        geometric_manager: Optional[GeometricCalibrationManager] = None,
    ) -> None:
        self.time_window = float(time_window_sec)
        self.max_dock_dist = float(max_dock_dist_meters)
        self.geo_manager = geometric_manager or GeometricCalibrationManager()
        self._next_cluster_id: int = 1

    def associate_frame_tracklets(
        self,
        camera_tracklets: Dict[str, List[CartonTracklet]],
        current_global_time_sec: float,
    ) -> List[AssociatedCartonCluster]:
        """
        Returns camera-local observations without inferring global identities from
        arbitrary carton centroids. Cross-camera identity is assigned only for
        calibrated loading-line crossing points in the central counting pipeline.
        """
        clusters: List[AssociatedCartonCluster] = []
        for camera_id in sorted(camera_tracklets):
            for tracklet in camera_tracklets[camera_id]:
                if tracklet.status not in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE):
                    continue
                cluster_id = f"cluster_{self._next_cluster_id:05d}"
                self._next_cluster_id += 1
                clusters.append(AssociatedCartonCluster(
                    cluster_id=cluster_id,
                    global_timestamp_sec=current_global_time_sec,
                    observations={camera_id: tracklet},
                ))

        return clusters
