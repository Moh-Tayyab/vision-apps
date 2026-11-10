"""
Live Shadow-Mode Execution & Audit Engine (Phase 1.10).

Runs parallel shadow-mode observation:
  - Records crossing predictions alongside ground truth / verification audit data.
  - Guarantees zero interference with production counting or authoritative decisions.
  - Tracks progress toward the >= 30 sessions and >= 3,000 crossings certification milestone.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .event_ground_truth import GroundTruthEvent, PredictedEvent

logger = logging.getLogger("truck_loading.shadow_mode")


@dataclass
class ShadowEventRecord:
    """Audit record for a single shadow-mode crossing event."""

    event_uuid: str
    session_id: str
    camera_id: str
    timestamp_sec: float
    direction: str
    predicted_count: int
    ground_truth_count: Optional[int]
    is_verified: bool
    model_version: str
    engine_profile: str  # "gpu_tensorrt", "openvino_int8", "pytorch_cpu"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ShadowModeAuditor:
    """Manages shadow-mode event logging and certification aggregation."""

    def __init__(
        self,
        log_dir: Path | str = "data/shadow_mode",
        model_version: str = "yolo11s_v2.0",
        engine_profile: str = "pytorch_cpu",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version
        self.engine_profile = engine_profile

        self._active_sessions: Dict[str, List[ShadowEventRecord]] = {}

    def log_event(
        self,
        session_id: str,
        camera_id: str,
        timestamp_sec: float,
        direction: str,
        predicted_count: int,
        ground_truth_count: Optional[int] = None,
        notes: str = "",
    ) -> ShadowEventRecord:
        """Record an event in shadow mode without affecting production counters."""
        record = ShadowEventRecord(
            event_uuid=str(uuid.uuid4()),
            session_id=session_id,
            camera_id=camera_id,
            timestamp_sec=timestamp_sec,
            direction=direction,
            predicted_count=predicted_count,
            ground_truth_count=ground_truth_count,
            is_verified=ground_truth_count is not None,
            model_version=self.model_version,
            engine_profile=self.engine_profile,
            notes=notes,
        )

        session_list = self._active_sessions.setdefault(session_id, [])
        session_list.append(record)

        # Append to JSONL audit log
        log_file = self.log_dir / f"shadow_{session_id}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        return record

    def get_progress_summary(self) -> Dict[str, Any]:
        """Summarize shadow mode progress toward 30 sessions and 3,000 events."""
        total_sessions = len(self._active_sessions)
        total_events = sum(len(evs) for evs in self._active_sessions.values())
        verified_events = sum(
            sum(1 for e in evs if e.is_verified) for evs in self._active_sessions.values()
        )

        return {
            "total_sessions": total_sessions,
            "target_sessions": 30,
            "sessions_progress_pct": round(min(100.0, (total_sessions / 30.0) * 100), 1),
            "total_events": total_events,
            "target_events": 3000,
            "events_progress_pct": round(min(100.0, (total_events / 3000.0) * 100), 1),
            "verified_events": verified_events,
            "is_ready_for_certification": total_sessions >= 30 and verified_events >= 3000,
        }
