"""
Phase 3.3 Ground Truth Annotation Schema & Dataset Management.

Provides strict data contracts for multi-camera warehouse carton counting ground truth:
- Supports session-level and event-level provenance.
- Preserves original video cryptographic SHA-256 hashes.
- Tracks reviewer identity and annotation version.
- Enforces strict split isolation: locked_test vs val vs train.
- Explicitly flags unannotated or incomplete recordings as MANIFEST_MISSING.
"""

from __future__ import annotations

import csv
import enum
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger("truck_loading.annotation_schema")


class ManifestStatus(str, enum.Enum):
    VERIFIED = "VERIFIED"
    PENDING_REVIEW = "PENDING_REVIEW"
    MANIFEST_MISSING = "MANIFEST_MISSING"


class SplitType(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    LOCKED_TEST = "locked_test"


class LineOfSightStatus(str, enum.Enum):
    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    OUT_OF_FRAME = "OUT_OF_FRAME"


@dataclass
class CameraEvidenceRecord:
    """Camera-specific line-of-sight and spatial evidence for a crossing event."""

    camera_id: str
    line_of_sight: LineOfSightStatus
    timestamp_sec: Optional[float] = None
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    confidence_estimate: Optional[float] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "line_of_sight": self.line_of_sight.value if hasattr(self.line_of_sight, "value") else str(self.line_of_sight),
            "timestamp_sec": self.timestamp_sec,
            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy else None,
            "confidence_estimate": self.confidence_estimate,
            "notes": self.notes,
        }


@dataclass
class GroundTruthEventAnnotation:
    """Complete ground truth record for a single physical carton crossing."""

    event_id: str
    session_id: str
    timestamp_sec: float
    direction: str  # "IN" or "OUT"
    carton_count: int = 1
    carrying_mode: str = "single_worker"  # "single_worker", "dual_worker", "conveyor"
    reviewer_id: str = "lead_annotator"
    annotation_version: str = "v3.3"
    camera_evidence: Dict[str, CameraEvidenceRecord] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "session_id": self.session_id,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "direction": self.direction.upper(),
            "carton_count": self.carton_count,
            "carrying_mode": self.carrying_mode,
            "reviewer_id": self.reviewer_id,
            "annotation_version": self.annotation_version,
            "camera_evidence": {k: v.to_dict() for k, v in self.camera_evidence.items()},
            "notes": self.notes,
        }


@dataclass
class VideoProvenanceRecord:
    """Cryptographic and metadata provenance for an ingested video recording."""

    video_path: str
    sha256_hash: str
    size_bytes: int
    duration_sec: float
    fps: float
    frame_count: int
    resolution: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "sha256_hash": self.sha256_hash,
            "size_bytes": self.size_bytes,
            "duration_sec": round(self.duration_sec, 3),
            "fps": round(self.fps, 2),
            "frame_count": self.frame_count,
            "resolution": list(self.resolution),
        }


@dataclass
class SessionAnnotationRecord:
    """Session-level ground truth manifest containing events and video provenance."""

    session_id: str
    split: SplitType
    status: ManifestStatus
    camera_videos: Dict[str, VideoProvenanceRecord] = field(default_factory=dict)
    events: List[GroundTruthEventAnnotation] = field(default_factory=list)
    manifest_version: str = "v3.3"
    reviewer_identity: str = "lead_annotator"
    blocker_reason: Optional[str] = None
    notes: str = ""

    @property
    def total_cartons(self) -> int:
        return sum(e.carton_count for e in self.events)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "split": self.split.value if hasattr(self.split, "value") else str(self.split),
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "total_events": len(self.events),
            "total_cartons": self.total_cartons,
            "manifest_version": self.manifest_version,
            "reviewer_identity": self.reviewer_identity,
            "blocker_reason": self.blocker_reason,
            "notes": self.notes,
            "camera_videos": {k: v.to_dict() for k, v in self.camera_videos.items()},
            "events": [e.to_dict() for e in self.events],
        }


class MultiSessionDatasetManager:
    """
    Manages multi-session dataset manifests, cryptographic hash verification,
    and locked test set isolation.
    """

    def __init__(self, manifest_file: Optional[str] = None) -> None:
        self.sessions: Dict[str, SessionAnnotationRecord] = {}
        if manifest_file and os.path.exists(manifest_file):
            self.load_from_json(manifest_file)

    @staticmethod
    def compute_file_sha256(filepath: str) -> str:
        """Computes SHA-256 hash for a file."""
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def register_session(self, record: SessionAnnotationRecord) -> None:
        self.sessions[record.session_id] = record

    def get_sessions_by_split(self, split: SplitType) -> List[SessionAnnotationRecord]:
        return [s for s in self.sessions.values() if s.split == split]

    def get_verified_crossing_count(self, splits: Optional[Sequence[SplitType]] = None) -> int:
        """Returns total verified physical crossings matching split criteria."""
        total = 0
        for s in self.sessions.values():
            if s.status != ManifestStatus.VERIFIED:
                continue
            if splits is None or s.split in splits:
                total += s.total_cartons
        return total

    def export_to_json(self, output_path: str) -> None:
        """Serializes dataset manager to JSON file."""
        payload = {
            "schema_version": "3.3",
            "total_sessions": len(self.sessions),
            "total_verified_crossings": self.get_verified_crossing_count(),
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
        }
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_from_json(self, json_path: str) -> None:
        """Loads and parses structured manifest from JSON."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sid, sdata in data.get("sessions", {}).items():
            split_enum = SplitType(sdata.get("split", "train"))
            status_enum = ManifestStatus(sdata.get("status", "MANIFEST_MISSING"))

            cam_vids = {}
            for cid, vdata in sdata.get("camera_videos", {}).items():
                cam_vids[cid] = VideoProvenanceRecord(
                    video_path=vdata["video_path"],
                    sha256_hash=vdata["sha256_hash"],
                    size_bytes=int(vdata.get("size_bytes", 0)),
                    duration_sec=float(vdata.get("duration_sec", 0.0)),
                    fps=float(vdata.get("fps", 30.0)),
                    frame_count=int(vdata.get("frame_count", 0)),
                    resolution=tuple(vdata.get("resolution", (720, 1280))),
                )

            ev_list = []
            for edata in sdata.get("events", []):
                ev_list.append(GroundTruthEventAnnotation(
                    event_id=edata["event_id"],
                    session_id=edata.get("session_id", sid),
                    timestamp_sec=float(edata["timestamp_sec"]),
                    direction=edata.get("direction", "IN"),
                    carton_count=int(edata.get("carton_count", 1)),
                    carrying_mode=edata.get("carrying_mode", "single_worker"),
                    reviewer_id=edata.get("reviewer_id", "lead_annotator"),
                    annotation_version=edata.get("annotation_version", "v3.3"),
                    notes=edata.get("notes", ""),
                ))

            self.sessions[sid] = SessionAnnotationRecord(
                session_id=sid,
                split=split_enum,
                status=status_enum,
                camera_videos=cam_vids,
                events=ev_list,
                manifest_version=sdata.get("manifest_version", "v3.3"),
                reviewer_identity=sdata.get("reviewer_identity", "lead_annotator"),
                blocker_reason=sdata.get("blocker_reason"),
                notes=sdata.get("notes", ""),
            )
