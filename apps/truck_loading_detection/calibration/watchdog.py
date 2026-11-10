"""
Adaptive Vibration Baseline & NCC Watchdog Engine (Phase 0.8, 0.9, 0.10).

Features:
1. 300-second vibration baseline recorder calculating mu_NCC and sigma_NCC.
2. Dynamic alarm threshold tau = min(0.80, mu_NCC - 4 * sigma_NCC).
3. Rolling 60-second edge similarity watchdog.
4. Camera alignment state machine with CAMERA_MISALIGNED alert and
   strict spatial-penalty suppression until validated recalibration.
"""

from __future__ import annotations

import collections
import enum
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("truck_loading.watchdog")


class CameraAlignmentState(str, enum.Enum):
    """Explicit camera alignment and calibration state machine."""

    CALIBRATED_OPERATIONAL = "CALIBRATED_OPERATIONAL"
    CALIBRATION_UNAVAILABLE = "CALIBRATION_UNAVAILABLE"
    CALIBRATION_INVALID = "CALIBRATION_INVALID"
    CAMERA_MISALIGNED = "CAMERA_MISALIGNED"
    RECALIBRATING = "RECALIBRATING"


@dataclass
class WatchdogAlert:
    """Alert event emitted on camera shift or misalignment."""

    alert_type: str
    camera_id: str
    site_id: str
    dock_id: str
    timestamp_mono: float
    current_similarity: float
    threshold_tau: float
    reason: str
    spatial_penalties_disabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "camera_id": self.camera_id,
            "site_id": self.site_id,
            "dock_id": self.dock_id,
            "timestamp_mono": round(self.timestamp_mono, 3),
            "current_similarity": round(self.current_similarity, 4),
            "threshold_tau": round(self.threshold_tau, 4),
            "reason": self.reason,
            "spatial_penalties_disabled": self.spatial_penalties_disabled,
        }


@dataclass
class BaselineStatistics:
    """Persisted vibration baseline metrics."""

    mu_NCC: float
    sigma_NCC: float
    threshold_tau: float
    sample_count: int
    duration_sec: float
    created_at_mono: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.mu_NCC) and math.isfinite(self.sigma_NCC)):
            raise ValueError("mu_NCC and sigma_NCC must be finite floats")
        if self.sigma_NCC < 0:
            raise ValueError(f"sigma_NCC must be non-negative, got {self.sigma_NCC}")
        if not math.isfinite(self.threshold_tau):
            raise ValueError("threshold_tau must be finite")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mu_NCC": float(self.mu_NCC),
            "sigma_NCC": float(self.sigma_NCC),
            "threshold_tau": float(self.threshold_tau),
            "sample_count": int(self.sample_count),
            "duration_sec": float(self.duration_sec),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaselineStatistics":
        return cls(
            mu_NCC=float(data["mu_NCC"]),
            sigma_NCC=float(data["sigma_NCC"]),
            threshold_tau=float(data["threshold_tau"]),
            sample_count=int(data["sample_count"]),
            duration_sec=float(data["duration_sec"]),
        )


def compute_edge_ncc(template_edges: np.ndarray, current_edges: np.ndarray) -> float:
    """
    Compute Normalized Cross-Correlation (NCC) between template and current edge maps.
    Returns float in [-1.0, 1.0].
    """
    t = template_edges.astype(np.float32)
    c = current_edges.astype(np.float32)

    t_mean = np.mean(t)
    c_mean = np.mean(c)

    t_diff = t - t_mean
    c_diff = c - c_mean

    t_var = np.sum(t_diff ** 2)
    c_var = np.sum(c_diff ** 2)

    denom = math.sqrt(t_var * c_var)
    if denom < 1e-7:
        # Uniform or zero edge maps
        return 1.0 if np.array_equal(template_edges, current_edges) else 0.0

    ncc = float(np.sum(t_diff * c_diff) / denom)
    return float(np.clip(ncc, -1.0, 1.0))


def extract_structural_edges(gray: np.ndarray) -> np.ndarray:
    """Extract Canny edges with slight morphological dilation for continuous vibration overlap."""
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edges, kernel)


class VibrationBaselineRecorder:
    """
    Records 300 seconds of dock vibration and computes baseline NCC statistics.
    Supports controllable clock injection for deterministic testing.
    """

    def __init__(
        self,
        camera_id: str,
        site_id: str,
        dock_id: str,
        target_duration_sec: float = 300.0,
        min_samples: int = 30,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.camera_id = camera_id
        self.site_id = site_id
        self.dock_id = dock_id
        self.target_duration = target_duration_sec
        self.min_samples = min_samples
        self.clock = clock_fn or time.monotonic

        self.start_time: Optional[float] = None
        self.reference_template: Optional[np.ndarray] = None
        self.samples: List[float] = []
        self.is_finished: bool = False

    def start(self, initial_frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> None:
        """Initialize reference edge template from first undistorted frame."""
        self.start_time = self.clock()
        self.samples.clear()
        self.is_finished = False

        gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY) if initial_frame.ndim == 3 else initial_frame
        if roi:
            x, y, w, h = roi
            gray = gray[y : y + h, x : x + w]

        self.reference_template = extract_structural_edges(gray)
        # First sample against itself is 1.0
        self.samples.append(1.0)

    def record_frame(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Record a frame observation.
        Returns True if the 300-second recording is complete, False otherwise.
        """
        if self.reference_template is None or self.start_time is None:
            raise RuntimeError("Baseline recording has not been started")
        if self.is_finished:
            return True

        now = self.clock()
        elapsed = now - self.start_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if roi:
            x, y, w, h = roi
            gray = gray[y : y + h, x : x + w]

        current_edges = extract_structural_edges(gray)
        ncc = compute_edge_ncc(self.reference_template, current_edges)
        if math.isfinite(ncc):
            self.samples.append(ncc)

        if elapsed >= self.target_duration:
            self.is_finished = True
            return True
        return False

    def finalize(self) -> BaselineStatistics:
        """Compute final mu_NCC, sigma_NCC, and dynamic threshold tau."""
        if len(self.samples) < self.min_samples:
            raise ValueError(
                f"Insufficient samples for valid vibration baseline: {len(self.samples)} < {self.min_samples}"
            )

        samples_arr = np.array(self.samples, dtype=np.float64)
        mu = float(np.mean(samples_arr))
        sigma = float(np.std(samples_arr))

        # Dynamic alarm threshold: tau = min(0.80, mu_NCC - 4 * sigma_NCC)
        tau = min(0.80, mu - 4.0 * sigma)

        now = self.clock()
        duration = now - self.start_time if self.start_time else self.target_duration

        return BaselineStatistics(
            mu_NCC=mu,
            sigma_NCC=sigma,
            threshold_tau=tau,
            sample_count=len(self.samples),
            duration_sec=duration,
        )


class NCCWatchdog:
    """
    Monitors live camera stability using rolling 60-second similarity window.
    Triggers CAMERA_MISALIGNED and controls spatial penalty suppression.
    """

    def __init__(
        self,
        camera_id: str,
        site_id: str,
        dock_id: str,
        baseline: Optional[BaselineStatistics] = None,
        reference_template: Optional[np.ndarray] = None,
        rolling_window_sec: float = 60.0,
        clock_fn: Optional[Callable[[], float]] = None,
    ) -> None:
        self.camera_id = camera_id
        self.site_id = site_id
        self.dock_id = dock_id
        self.baseline = baseline
        self.reference_template = reference_template
        self.rolling_window_sec = rolling_window_sec
        self.clock = clock_fn or time.monotonic

        # Rolling history of (timestamp, ncc_value)
        self._history: Deque[Tuple[float, float]] = collections.deque()
        self.state: CameraAlignmentState = (
            CameraAlignmentState.CALIBRATED_OPERATIONAL
            if baseline is not None
            else CameraAlignmentState.CALIBRATION_UNAVAILABLE
        )
        self.last_alert: Optional[WatchdogAlert] = None
        self.alerts_history: List[WatchdogAlert] = []

    @property
    def spatial_penalties_enabled(self) -> bool:
        """
        Spatial coordinate penalties MUST BE DISABLED whenever camera is misaligned
        or uncalibrated. Enabled strictly when state is CALIBRATED_OPERATIONAL.
        """
        return self.state == CameraAlignmentState.CALIBRATED_OPERATIONAL

    def update_baseline(
        self, baseline: BaselineStatistics, reference_template: np.ndarray
    ) -> None:
        """Set or update validated baseline, transitioning state back to operational."""
        self.baseline = baseline
        self.reference_template = reference_template
        self._history.clear()
        self.state = CameraAlignmentState.CALIBRATED_OPERATIONAL
        self.last_alert = None
        logger.info(
            "Camera %s baseline updated: mu=%.3f, sigma=%.3f, tau=%.3f. Spatial penalties restored.",
            self.camera_id,
            baseline.mu_NCC,
            baseline.sigma_NCC,
            baseline.threshold_tau,
        )

    def set_recalibrating(self) -> None:
        """Explicitly transition to recalibrating state."""
        self.state = CameraAlignmentState.RECALIBRATING

    def check_frame(
        self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Tuple[CameraAlignmentState, Optional[WatchdogAlert]]:
        """
        Process a live undistorted frame, compute NCC, and update alignment state.
        Returns (state, alert_if_any).
        """
        now = self.clock()

        if self.reference_template is None or self.baseline is None:
            self.state = CameraAlignmentState.CALIBRATION_UNAVAILABLE
            return self.state, None

        # Extract edges from current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if roi:
            x, y, w, h = roi
            gray = gray[y : y + h, x : x + w]

        current_edges = extract_structural_edges(gray)
        ncc = compute_edge_ncc(self.reference_template, current_edges)

        # Record in rolling history
        self._history.append((now, ncc))

        # Evict samples older than rolling_window_sec
        cutoff = now - self.rolling_window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Compute rolling mean similarity
        rolling_values = [val for _, val in self._history]
        rolling_similarity = float(np.mean(rolling_values)) if rolling_values else ncc

        # Evaluate against alarm threshold
        tau = self.baseline.threshold_tau
        if rolling_similarity < tau:
            # Camera shift detected!
            self.state = CameraAlignmentState.CAMERA_MISALIGNED
            alert = WatchdogAlert(
                alert_type="CAMERA_MISALIGNED",
                camera_id=self.camera_id,
                site_id=self.site_id,
                dock_id=self.dock_id,
                timestamp_mono=now,
                current_similarity=rolling_similarity,
                threshold_tau=tau,
                reason=(
                    f"Rolling {self.rolling_window_sec:.0f}s edge similarity "
                    f"({rolling_similarity:.3f}) dropped below threshold tau ({tau:.3f})"
                ),
                spatial_penalties_disabled=True,
            )
            self.last_alert = alert
            self.alerts_history.append(alert)
            logger.warning("ALARM TRIGGERED: %s for camera %s", alert.reason, self.camera_id)
            return self.state, alert

        return self.state, None
