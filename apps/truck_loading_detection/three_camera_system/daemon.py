"""
Standalone High-Availability Counting Daemon (Piece 5 Architecture).

Runs unattended carton counting for industrial loading docks:
- Ingests synchronized 3-camera RTSP/video feeds via LiveCameraManager.
- Executes real-time batch-3 canonical inference and metric 3D dock association.
- Persists all crossing events to durable SQLite WAL EventStore with exactly-once idempotency.
- Handles crash/restart recovery on boot (zero count loss, zero duplicates).
- Emits atomic heartbeat files for Kubernetes, systemd, or Docker healthcheck probes.
- Enforces Pallet Scale Shadow-Mode QA Gate with automated 3-camera evidence capture.
- Provides clean SIGINT/SIGTERM graceful shutdown with window flushing.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .event_store import EventStore
from .geometric_adapter import GeometricCalibrationManager
from .live_cameras import CAMERA_IDS, LiveCameraManager
from .pipeline import ThreeCameraCartonPipeline
from .shadow_mode import ShadowModeManager

logger = logging.getLogger("three_camera_system.daemon")


@dataclass
class DaemonConfig:
    session_id: str = "dock01_session"
    cam_sources: Dict[str, Union[str, int]] = field(default_factory=dict)
    calibration_file: Optional[str] = None
    model_path: str = "best.pt"
    device: str = "auto"
    target_fps: float = 20.0
    db_path: Optional[str] = None
    heartbeat_path: str = "config/daemon_heartbeat.json"
    heartbeat_interval_sec: float = 2.0
    pallet_tare_weight_kg: float = 25.0
    pallet_nominal_carton_weight_kg: float = 12.5
    conf_threshold: float = 0.15
    require_calibrated_fusion: bool = False


class TruckLoadingProductionDaemon:
    """
    Industrial high-availability daemon running the 3-camera carton counting pipeline.
    """

    def __init__(self, config: Optional[DaemonConfig] = None) -> None:
        self.config = config or DaemonConfig()
        self._stop_event = threading.Event()
        self._is_running = False
        self._start_time: float = 0.0

        # Durable state store
        db_path = self.config.db_path or os.environ.get("CARTON_EVENTS_DB_PATH")
        self.event_store = EventStore(db_path=db_path)

        # Operational shadow mode manager
        self.shadow_manager = ShadowModeManager(
            event_store=self.event_store,
            session_id=self.config.session_id,
            authoritative_inventory_locked=True,
        )

        # Multi-camera pipeline
        calib_file = self.config.calibration_file
        if not calib_file:
            default_calib = Path(__file__).resolve().parent.parent / "config" / "calibrations" / "warehouse_dock_01_verified.json"
            if default_calib.exists():
                calib_file = str(default_calib)

        self.pipeline = ThreeCameraCartonPipeline(
            model_path=self.config.model_path,
            calibration_path=calib_file,
            event_store=self.event_store,
            session_id=self.config.session_id,
            device=self.config.device,
            fps=self.config.target_fps,
            conf_threshold=self.config.conf_threshold,
            require_calibrated_fusion=self.config.require_calibrated_fusion,
            allow_temporal_video_fusion=True,
        )

        # Live camera ingestion manager
        self.live_camera_manager = LiveCameraManager()

        # Heartbeat background thread
        self._heartbeat_thread: Optional[threading.Thread] = None

    def start(self, install_signal_handlers: bool = True) -> None:
        """Starts live ingestion, live counting worker, and supervisory heartbeat."""
        if self._is_running:
            logger.warning("Daemon is already running")
            return

        self._start_time = time.time()
        self._stop_event.clear()

        # 1. Automatic boot recovery from durable EventStore
        try:
            self.pipeline.recover_from_event_store(self.config.session_id)
            recovered_summary = self.shadow_manager.recover_session_state()
            logger.info(
                "Durable boot recovery successful for session %s: %d confirmed cartons recovered",
                self.config.session_id,
                recovered_summary.get("system_total_confirmed", 0),
            )
        except Exception:
            logger.exception("Error during session recovery from EventStore")

        # 2. Register OS signal handlers for graceful shutdown if running in main thread
        if install_signal_handlers:
            try:
                signal.signal(signal.SIGINT, self._handle_os_signal)
                signal.signal(signal.SIGTERM, self._handle_os_signal)
            except (ValueError, AttributeError):
                pass  # Not in main thread

        # 3. Start live camera streams if sources configured
        if self.config.cam_sources and set(self.config.cam_sources.keys()) == set(CAMERA_IDS):
            logger.info("Connecting live cameras: %s", self.config.cam_sources)
            self.live_camera_manager.start(self.config.cam_sources)

        # 4. Start live counting worker
        self.live_camera_manager.start_counting(self.pipeline, target_fps=self.config.target_fps)

        # 5. Start heartbeat writer thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="daemon-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

        self._is_running = True
        self.write_heartbeat(status="running")
        logger.info("Production counting daemon successfully started (session=%s)", self.config.session_id)

    def stop(self) -> None:
        """Gracefully halts live streams, flushes pending windows, and writes final status."""
        if not self._is_running:
            return

        logger.info("Stopping production counting daemon...")
        self._stop_event.set()

        # 1. Stop live cameras and counting worker
        self.live_camera_manager.stop()

        # 2. Flush uncommitted crossing windows to prevent dropped counts
        flushed = self.pipeline.flush_pending_windows()
        if flushed:
            logger.info("Flushed %d uncommitted crossing windows on shutdown", len(flushed))

        # 3. Write final stopped heartbeat
        self._is_running = False
        self.write_heartbeat(status="stopped")

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=3.0)

        logger.info("Production counting daemon stopped cleanly")

    def _handle_os_signal(self, signum: int, frame: Any) -> None:
        signame = signal.Signals(signum).name
        logger.info("Received termination signal %s. Initiating graceful shutdown...", signame)
        self.stop()
        sys.exit(0)

    def _heartbeat_loop(self) -> None:
        interval = max(0.5, self.config.heartbeat_interval_sec)
        while not self._stop_event.is_set():
            try:
                self.write_heartbeat(status="running")
            except Exception:
                logger.exception("Error writing daemon heartbeat")
            self._stop_event.wait(interval)

    def write_heartbeat(self, status: str = "running") -> Dict[str, Any]:
        """Atomically writes supervisory status JSON to disk for external health checks."""
        now = time.time()
        uptime = round(now - self._start_time, 2) if self._start_time > 0 else 0.0

        counts = self.pipeline.counts_summary
        cam_status = self.live_camera_manager.status()
        shadow_summary = self.shadow_manager.get_shadow_counts()

        heartbeat_data = {
            "daemon": "truck_loading_counter",
            "status": status,
            "session_id": self.config.session_id,
            "timestamp": now,
            "uptime_sec": uptime,
            "pid": os.getpid(),
            "target_fps": self.config.target_fps,
            "model_path": self.config.model_path,
            "device": self.config.device,
            "counts": counts,
            "shadow_summary": shadow_summary,
            "camera_health": cam_status,
            "durable_storage": "sqlite_wal",
        }

        hb_path = Path(self.config.heartbeat_path)
        hb_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = hb_path.with_suffix(".tmp")

        try:
            with open(tmp_path, "w") as f:
                json.dump(heartbeat_data, f, indent=2)
            os.replace(tmp_path, hb_path)
        except Exception as e:
            logger.warning("Could not atomically write heartbeat file: %s", e)

        return heartbeat_data

    def verify_pallet_scale(
        self,
        gross_weight_kg: float,
        tare_weight_kg: Optional[float] = None,
        nominal_carton_weight_kg: Optional[float] = None,
        optical_count: Optional[int] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Executes Pallet Scale Shadow QA Gate.
        Compares scale weight against optical count and triggers audit collage on discrepancy.
        """
        tare = self.config.pallet_tare_weight_kg if tare_weight_kg is None else float(tare_weight_kg)
        unit = self.config.pallet_nominal_carton_weight_kg if nominal_carton_weight_kg is None else float(nominal_carton_weight_kg)

        opt_cnt = self.pipeline.confirmed_in if optical_count is None else int(optical_count)

        return self.shadow_manager.record_pallet_scale_audit(
            gross_weight_kg=gross_weight_kg,
            tare_weight_kg=tare,
            nominal_carton_weight_kg=unit,
            optical_count=opt_cnt,
            live_camera_manager=self.live_camera_manager,
            notes=notes,
        )

    def get_status(self) -> Dict[str, Any]:
        """Returns instantaneous daemon operational status."""
        return self.write_heartbeat(status="running" if self._is_running else "stopped")

    def run_forever(self) -> None:
        """Blocks and runs daemon until interrupted."""
        self.start(install_signal_handlers=True)
        try:
            while not self._stop_event.is_set():
                time.sleep(0.5)
        except (KeyboardInterrupt, SystemExit):
            self.stop()
