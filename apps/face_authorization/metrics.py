"""Prometheus Metrics Exporter & Real-Time Performance Monitor.

Exposes standard Prometheus metrics format at /metrics for Grafana / Prometheus scraping.
"""

from __future__ import annotations

import threading
import time
from typing import Dict


class MetricsCollector:
    """Thread-safe collector for real-time Prometheus monitoring metrics."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_metrics()
            return cls._instance

    def _init_metrics(self):
        self._lock = threading.Lock()
        self.frames_processed = 0
        self.inference_count = 0
        self.total_inference_time_sec = 0.0
        self.last_inference_latency_ms = 0.0
        self.counts: Dict[str, int] = {
            "authorized": 0,
            "unauthorized": 0,
            "spoof": 0,
            "unknown": 0,
        }
        self.camera_fps = 0.0
        self.start_time = time.time()

    def record_frame(self):
        with self._lock:
            self.frames_processed += 1

    def record_inference(self, duration_sec: float, detections: list, camera_fps: float = 0.0):
        with self._lock:
            self.inference_count += 1
            self.total_inference_time_sec += duration_sec
            self.last_inference_latency_ms = duration_sec * 1000.0
            self.camera_fps = camera_fps
            for d in detections:
                status = d.get("status", "unknown").lower()
                if status in self.counts:
                    self.counts[status] += 1
                else:
                    self.counts["unknown"] += 1

    def generate_prometheus_text(self, enrolled_persons_count: int = 0) -> str:
        """Generate Prometheus exposition text format (version 0.0.4)."""
        with self._lock:
            uptime = time.time() - self.start_time
            avg_latency = (
                (self.total_inference_time_sec / self.inference_count)
                if self.inference_count > 0
                else 0.0
            )

            lines = [
                "# HELP face_auth_up Whether the Face Authorization service is up (1 = up)",
                "# TYPE face_auth_up gauge",
                "face_auth_up 1",
                "",
                "# HELP face_auth_uptime_seconds Service uptime in seconds",
                "# TYPE face_auth_uptime_seconds counter",
                f"face_auth_uptime_seconds {uptime:.1f}",
                "",
                "# HELP face_auth_frames_total Total camera frames processed",
                "# TYPE face_auth_frames_total counter",
                f"face_auth_frames_total {self.frames_processed}",
                "",
                "# HELP face_auth_inference_total Total inference executions",
                "# TYPE face_auth_inference_total counter",
                f"face_auth_inference_total {self.inference_count}",
                "",
                "# HELP face_auth_inference_latency_seconds Average AI inference latency",
                "# TYPE face_auth_inference_latency_seconds gauge",
                f"face_auth_inference_latency_seconds {avg_latency:.4f}",
                "",
                "# HELP face_auth_camera_fps Current stream FPS",
                "# TYPE face_auth_camera_fps gauge",
                f"face_auth_camera_fps {self.camera_fps:.1f}",
                "",
                "# HELP face_auth_enrolled_persons Total authorized persons in database",
                "# TYPE face_auth_enrolled_persons gauge",
                f"face_auth_enrolled_persons {enrolled_persons_count}",
                "",
                "# HELP face_auth_detections_total Total detections by classification status",
                "# TYPE face_auth_detections_total counter",
                f'face_auth_detections_total{{status="authorized"}} {self.counts["authorized"]}',
                f'face_auth_detections_total{{status="unauthorized"}} {self.counts["unauthorized"]}',
                f'face_auth_detections_total{{status="spoof"}} {self.counts["spoof"]}',
                f'face_auth_detections_total{{status="unknown"}} {self.counts["unknown"]}',
            ]
            return "\n".join(lines) + "\n"


metrics = MetricsCollector()
