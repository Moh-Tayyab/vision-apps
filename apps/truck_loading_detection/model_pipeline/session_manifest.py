"""
Session Manifest & Dataset Split Architecture (Phase 1.2).

Manages the 25-session bootstrap dataset:
  - 15 Training sessions
  - 5 Validation sessions
  - 5 Locked-Test sessions (frozen, hashed, protected from iterative tuning)

Enforces zero session leakage and generates cryptographic SHA-256 manifest audit trail.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ManifestValidationError(Exception):
    """Raised when session manifest violates split or integrity rules."""
    pass


VALID_SPLITS = ("train", "val", "locked_test")


@dataclass
class SessionRecord:
    """Individual recorded warehouse dock session."""

    session_id: str
    video_path: str
    camera_id: str  # "cam_top", "cam_left", "cam_right"
    split: str  # "train", "val", "locked_test"
    ground_truth_events_csv: str
    site_id: str = "wh_lahore_01"
    dock_id: str = "dock_01"
    resolution: Tuple[int, int] = (1280, 720)
    fps: float = 25.0
    video_sha256: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ManifestValidationError(
                f"Invalid split '{self.split}'. Must be one of {VALID_SPLITS}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "site_id": self.site_id,
            "dock_id": self.dock_id,
            "camera_id": self.camera_id,
            "split": self.split,
            "video_path": self.video_path,
            "video_sha256": self.video_sha256,
            "ground_truth_events_csv": self.ground_truth_events_csv,
            "resolution": list(self.resolution),
            "fps": float(self.fps),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        return cls(
            session_id=str(data["session_id"]),
            site_id=str(data.get("site_id", "wh_lahore_01")),
            dock_id=str(data.get("dock_id", "dock_01")),
            camera_id=str(data.get("camera_id", "cam_top")),
            split=str(data["split"]),
            video_path=str(data["video_path"]),
            video_sha256=str(data.get("video_sha256", "")),
            ground_truth_events_csv=str(data["ground_truth_events_csv"]),
            resolution=tuple(data.get("resolution", (1280, 720))),
            fps=float(data.get("fps", 25.0)),
            notes=str(data.get("notes", "")),
        )


@dataclass
class DatasetManifest:
    """Master manifest managing all 25 sessions and split guarantees."""

    manifest_version: str = "2.0"
    sessions: List[SessionRecord] = field(default_factory=list)
    locked_test_hash: str = ""

    def add_session(self, session: SessionRecord) -> None:
        self.sessions.append(session)

    def validate_splits(self) -> None:
        """Verify counts: exactly 15 train, 5 val, 5 locked_test, and no duplicates."""
        seen_ids: Set[str] = set()
        seen_videos: Set[str] = set()
        train_count = 0
        val_count = 0
        locked_test_count = 0

        for s in self.sessions:
            if s.session_id in seen_ids:
                raise ManifestValidationError(f"Duplicate session_id: {s.session_id}")
            seen_ids.add(s.session_id)

            if s.video_path in seen_videos:
                raise ManifestValidationError(
                    f"Data Leakage Risk: video '{s.video_path}' reused across multiple sessions!"
                )
            seen_videos.add(s.video_path)

            if s.split == "train":
                train_count += 1
            elif s.split == "val":
                val_count += 1
            elif s.split == "locked_test":
                locked_test_count += 1

        total = len(self.sessions)
        if total != 25:
            raise ManifestValidationError(
                f"Expected exactly 25 sessions for Stage 1 Bootstrap, got {total}"
            )
        if train_count != 15:
            raise ManifestValidationError(f"Expected 15 train sessions, got {train_count}")
        if val_count != 5:
            raise ManifestValidationError(f"Expected 5 val sessions, got {val_count}")
        if locked_test_count != 5:
            raise ManifestValidationError(f"Expected 5 locked-test sessions, got {locked_test_count}")

    def find_content_leakage(self) -> List[Dict[str, Any]]:
        """Identify duplicate video recordings (by SHA-256) within or across splits."""
        seen: Dict[str, SessionRecord] = {}
        leaks: List[Dict[str, Any]] = []
        for s in self.sessions:
            if not s.video_sha256:
                continue
            if s.video_sha256 in seen:
                prev = seen[s.video_sha256]
                leaks.append({
                    "sha256": s.video_sha256,
                    "session_a": prev.session_id,
                    "split_a": prev.split,
                    "path_a": prev.video_path,
                    "session_b": s.session_id,
                    "split_b": s.split,
                    "path_b": s.video_path,
                    "cross_split_leakage": prev.split != s.split,
                })
            else:
                seen[s.video_sha256] = s
        return leaks

    def compute_locked_test_hash(self) -> str:
        """Compute SHA-256 fingerprint over frozen locked-test sessions."""
        h = hashlib.sha256()
        locked_sessions = sorted(
            [s for s in self.sessions if s.split == "locked_test"],
            key=lambda x: x.session_id,
        )
        for s in locked_sessions:
            h.update(f"{s.session_id}:{s.video_path}:{s.video_sha256}".encode("utf-8"))
        self.locked_test_hash = h.hexdigest()
        return self.locked_test_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "locked_test_hash": self.locked_test_hash or self.compute_locked_test_hash(),
            "sessions": [s.to_dict() for s in self.sessions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetManifest":
        manifest = cls(
            manifest_version=str(data.get("manifest_version", "2.0")),
            locked_test_hash=str(data.get("locked_test_hash", "")),
        )
        for s in data.get("sessions", []):
            manifest.add_session(SessionRecord.from_dict(s))
        return manifest

    def save_json(self, path: Path | str) -> None:
        """Save manifest with validated splits and locked test hash."""
        self.validate_splits()
        self.compute_locked_test_hash()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def count_by_split(self, split: str) -> int:
        return sum(1 for s in self.sessions if s.split == split)

    def get_locked_test_sessions(self) -> List[SessionRecord]:
        return [s for s in self.sessions if s.split == "locked_test"]

    @classmethod
    def load_json(cls, path: Path | str) -> "DatasetManifest":
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        manifest = cls.from_dict(data)
        manifest.validate_splits()
        return manifest

    from_json = load_json
