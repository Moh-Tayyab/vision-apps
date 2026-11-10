"""
apps/truck_loading_detection/three_camera_system/shadow_mode.py

Phase 3.3 Workstream E — Operational Shadow Mode Pipeline.
Provides an isolated, non-intrusive shadow execution environment:
- Ingests live or recorded multi-camera video streams.
- Runs full canonical detection, cross-camera association, and durable event persistence.
- Strictly ISOLATES authoritative warehouse ERP/WMS inventory from automated mutations.
- Maintains independent shadow counters:
    * human_physical_count
    * automatic_confirmed_count
    * human_approved_count
    * unresolved_count (review required)
    * discrepancy_count
- Monitors stream health (dropped frames, synchronization drift, queue latency).
- Logs structured incident alerts.
- Supports crash/restart recovery directly from SQLite WAL EventStore.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .event_store import EventStore
from .evaluator import PalletScaleCrossChecker, PalletScaleAuditResult
from .pipeline import ThreeCameraCartonPipeline

logger = logging.getLogger(__name__)


@dataclass
class ShadowStreamHealth:
    total_frames_ingested: int = 0
    dropped_frames: int = 0
    max_sync_jitter_ms: float = 0.0
    current_sync_jitter_ms: float = 0.0
    processing_fps: float = 0.0
    average_latency_ms: float = 0.0
    incidents_logged: int = 0


@dataclass
class ShadowIncident:
    incident_id: str
    incident_type: str  # e.g., SYNC_JITTER_EXCEEDED, CAMERA_FRAME_DROP, COUNT_DISCREPANCY, RESTART_RECOVERY
    severity: str  # INFO, WARNING, ERROR, CRITICAL
    timestamp_sec: float
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ShadowModeManager:
    """
    Manages non-intrusive warehouse dock monitoring without modifying production inventory.
    """

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        db_path: Optional[str] = None,
        session_id: str = "shadow_session",
        authoritative_inventory_locked: bool = True,
    ) -> None:
        self.session_id = session_id
        self.authoritative_inventory_locked = authoritative_inventory_locked
        if event_store is not None:
            self.event_store = event_store
        elif db_path is not None:
            self.event_store = EventStore(db_path=db_path)
        else:
            self.event_store = EventStore()

        self.health = ShadowStreamHealth()
        self.incidents: List[ShadowIncident] = []
        self.human_physical_count: int = 0

        self._init_shadow_db()

    def _init_shadow_db(self) -> None:
        """Initializes shadow tables in SQLite EventStore."""
        with self.event_store._lock:
            with self.event_store._get_connection() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_incidents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    incident_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp_sec REAL NOT NULL,
                    description TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_session_manifest (
                    session_id TEXT PRIMARY KEY,
                    human_physical_count INTEGER NOT NULL DEFAULT 0,
                    is_inventory_locked INTEGER NOT NULL DEFAULT 1,
                    started_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS pallet_scale_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT UNIQUE NOT NULL,
                    session_id TEXT NOT NULL,
                    gross_weight_kg REAL NOT NULL,
                    tare_weight_kg REAL NOT NULL,
                    net_weight_kg REAL NOT NULL,
                    nominal_carton_weight_kg REAL NOT NULL,
                    scale_carton_count INTEGER NOT NULL,
                    optical_carton_count INTEGER NOT NULL,
                    discrepancy_delta INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    is_discrepancy INTEGER NOT NULL,
                    evidence_path TEXT,
                    timestamp_sec REAL NOT NULL,
                    notes TEXT
                );
                """)
                conn.commit()

    def set_human_physical_count(self, count: int) -> None:
        """Registers verified human physical count for discrepancy monitoring."""
        self.human_physical_count = max(0, int(count))
        now = time.time()
        with self.event_store._lock:
            with self.event_store._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO shadow_session_manifest (
                        session_id, human_physical_count, is_inventory_locked, started_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        human_physical_count = excluded.human_physical_count,
                        updated_at = excluded.updated_at;
                    """,
                    (self.session_id, self.human_physical_count, 1 if self.authoritative_inventory_locked else 0, now, now),
                )
                conn.commit()
        logger.info("Shadow session %s registered physical GT count: %d", self.session_id, self.human_physical_count)

    def log_incident(
        self,
        incident_type: str,
        severity: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ShadowIncident:
        """Transactionally logs a structured operational incident."""
        now = time.time()
        inc_id = f"inc_{int(now * 1000)}_{len(self.incidents)}"
        inc = ShadowIncident(
            incident_id=inc_id,
            incident_type=incident_type,
            severity=severity,
            timestamp_sec=now,
            description=description,
            metadata=metadata or {},
        )
        self.incidents.append(inc)
        self.health.incidents_logged += 1

        with self.event_store._lock:
            with self.event_store._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO shadow_incidents (
                        incident_id, session_id, incident_type, severity, timestamp_sec, description, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (inc_id, self.session_id, incident_type, severity, now, description, json.dumps(metadata or {})),
                )
                conn.commit()

        logger.warning("Shadow Incident [%s] (%s): %s", severity, incident_type, description)
        return inc

    def record_frame_health(
        self,
        jitter_ms: float,
        is_dropped: bool = False,
        elapsed_sec: float = 0.0,
    ) -> None:
        """Updates frame ingestion and synchronization metrics."""
        self.health.total_frames_ingested += 1
        if is_dropped:
            self.health.dropped_frames += 1
            if self.health.dropped_frames % 50 == 0:
                self.log_incident(
                    incident_type="CAMERA_FRAME_DROP",
                    severity="WARNING",
                    description=f"Accumulated {self.health.dropped_frames} dropped frames",
                    metadata={"total_dropped": self.health.dropped_frames},
                )

        self.health.current_sync_jitter_ms = float(jitter_ms)
        if jitter_ms > self.health.max_sync_jitter_ms:
            self.health.max_sync_jitter_ms = float(jitter_ms)

        if jitter_ms > 33.3:  # Greater than 1 frame at 30fps
            self.log_incident(
                incident_type="SYNC_JITTER_EXCEEDED",
                severity="WARNING",
                description=f"Multi-camera sync jitter reached {jitter_ms:.2f}ms (>33.3ms threshold)",
                metadata={"jitter_ms": jitter_ms},
            )

        if elapsed_sec > 0:
            self.health.processing_fps = round(self.health.total_frames_ingested / elapsed_sec, 2)

    def get_shadow_counts(self) -> Dict[str, Any]:
        """
        Computes side-by-side shadow counts without modifying external ERP/WMS inventory.
        """
        db_counts = self.event_store.get_counts(session_id=self.session_id)

        auto_confirmed = db_counts.get("auto_confirmed_count", 0)
        human_approved = db_counts.get("human_approved_count", 0)
        system_total_confirmed = auto_confirmed + human_approved
        unresolved = db_counts.get("review_required_count", 0)

        discrepancy = abs(self.human_physical_count - system_total_confirmed) if self.human_physical_count > 0 else 0

        return {
            "session_id": self.session_id,
            "authoritative_inventory_locked": self.authoritative_inventory_locked,
            "human_physical_count": self.human_physical_count,
            "automatic_confirmed_count": auto_confirmed,
            "human_approved_count": human_approved,
            "system_total_confirmed": system_total_confirmed,
            "unresolved_review_count": unresolved,
            "discrepancy_count": discrepancy,
            "discrepancy_rate_pct": round((discrepancy / max(1, self.human_physical_count)) * 100, 2) if self.human_physical_count > 0 else 0.0,
            "counts_breakdown": db_counts,
            "health": asdict(self.health),
        }

    def recover_session_state(self) -> Dict[str, Any]:
        """Recovers session state after system restart."""
        with self.event_store._lock:
            with self.event_store._get_connection() as conn:
                cur = conn.execute(
                    "SELECT human_physical_count, is_inventory_locked FROM shadow_session_manifest WHERE session_id = ?;",
                    (self.session_id,),
                )
                row = cur.fetchone()
                if row:
                    self.human_physical_count = int(row["human_physical_count"])
                    self.authoritative_inventory_locked = bool(row["is_inventory_locked"])

                cur_inc = conn.execute(
                    "SELECT * FROM shadow_incidents WHERE session_id = ? ORDER BY timestamp_sec ASC;",
                    (self.session_id,),
                )
                self.incidents = [
                    ShadowIncident(
                        incident_id=r["incident_id"],
                        incident_type=r["incident_type"],
                        severity=r["severity"],
                        timestamp_sec=r["timestamp_sec"],
                        description=r["description"],
                        metadata=json.loads(r["metadata_json"] or "{}"),
                    )
                    for r in cur_inc.fetchall()
                ]
                self.health.incidents_logged = len(self.incidents)

        self.log_incident(
            incident_type="RESTART_RECOVERY",
            severity="INFO",
            description=f"Session {self.session_id} state successfully recovered from durable SQLite WAL",
            metadata={"recovered_incidents": len(self.incidents), "gt_count": self.human_physical_count},
        )
        return self.get_shadow_counts()

    def record_pallet_scale_audit(
        self,
        gross_weight_kg: float,
        tare_weight_kg: float = 25.0,
        nominal_carton_weight_kg: float = 12.5,
        optical_count: Optional[int] = None,
        live_camera_manager: Optional[Any] = None,
        custom_frames: Optional[Dict[str, np.ndarray]] = None,
        notes: str = "",
    ) -> Dict[str, Any]:
        """
        Runs Pallet Scale Cross-Check QA Gate.
        Formula: N_pallet = round((gross_weight - tare_weight) / nominal_carton_weight)
        If delta != 0:
            Triggers COUNT_DISCREPANCY_SCALE_MISMATCH incident and synchronized collage evidence.
        """
        now = time.time()
        checker = PalletScaleCrossChecker(
            nominal_carton_weight_kg=nominal_carton_weight_kg,
            default_tare_weight_kg=tare_weight_kg,
        )

        if optical_count is None:
            counts = self.event_store.get_counts(session_id=self.session_id)
            opt_count = counts.get("confirmed_count", 0)
        else:
            opt_count = int(optical_count)

        res: PalletScaleAuditResult = checker.cross_check(
            optical_count=opt_count,
            gross_weight_kg=gross_weight_kg,
            tare_weight_kg=tare_weight_kg,
            unit_weight_kg=nominal_carton_weight_kg,
        )

        audit_id = f"scale_audit_{int(now * 1000)}"
        delta = res.discrepancy_delta
        is_discrepancy = delta != 0
        evidence_path: Optional[str] = None

        if live_camera_manager is not None:
            title = "PALLET SCALE DISCREPANCY" if is_discrepancy else "PALLET SCALE VERIFIED"
            evidence_path = live_camera_manager.capture_evidence_collage(
                title=title,
                subtitle=res.diagnostic_message,
                is_discrepancy=is_discrepancy,
                custom_frames=custom_frames,
            )

        if is_discrepancy:
            self.log_incident(
                incident_type="COUNT_DISCREPANCY_SCALE_MISMATCH",
                severity="CRITICAL" if abs(delta) >= 3 else "WARNING",
                description=res.diagnostic_message,
                metadata={
                    "audit_id": audit_id,
                    "optical_count": opt_count,
                    "scale_carton_count": res.scale_carton_count,
                    "discrepancy_delta": delta,
                    "gross_weight_kg": gross_weight_kg,
                    "tare_weight_kg": tare_weight_kg,
                    "net_weight_kg": res.net_weight_kg,
                    "evidence_path": evidence_path,
                },
            )
        else:
            self.log_incident(
                incident_type="PALLET_SCALE_VERIFIED",
                severity="INFO",
                description=res.diagnostic_message,
                metadata={
                    "audit_id": audit_id,
                    "optical_count": opt_count,
                    "scale_carton_count": res.scale_carton_count,
                    "gross_weight_kg": gross_weight_kg,
                    "evidence_path": evidence_path,
                },
            )

        with self.event_store._lock:
            with self.event_store._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO pallet_scale_audits (
                        audit_id, session_id, gross_weight_kg, tare_weight_kg, net_weight_kg,
                        nominal_carton_weight_kg, scale_carton_count, optical_carton_count,
                        discrepancy_delta, status, is_discrepancy, evidence_path, timestamp_sec, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        audit_id,
                        self.session_id,
                        gross_weight_kg,
                        tare_weight_kg,
                        res.net_weight_kg,
                        nominal_carton_weight_kg,
                        res.scale_carton_count,
                        opt_count,
                        delta,
                        res.status,
                        1 if is_discrepancy else 0,
                        evidence_path,
                        now,
                        notes,
                    ),
                )
                conn.commit()

        audit_dict = res.to_dict()
        audit_dict["audit_id"] = audit_id
        audit_dict["session_id"] = self.session_id
        audit_dict["is_discrepancy"] = is_discrepancy
        audit_dict["evidence_path"] = evidence_path
        audit_dict["timestamp_sec"] = now
        return audit_dict

    def get_pallet_scale_audits(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves all pallet scale audit records for the session."""
        sid = session_id or self.session_id
        with self.event_store._lock:
            with self.event_store._get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM pallet_scale_audits WHERE session_id = ? ORDER BY timestamp_sec DESC;",
                    (sid,),
                )
                return [dict(r) for r in cur.fetchall()]

