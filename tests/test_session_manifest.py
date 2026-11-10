"""
Unit tests for 25-Session Manifest & Split Integrity (Phase 1.2).
"""

from pathlib import Path
import pytest

from apps.truck_loading_detection.model_pipeline.session_manifest import (
    DatasetManifest,
    ManifestValidationError,
    SessionRecord,
)


def test_load_official_manifest():
    """Verify official 25-session manifest loads and strictly adheres to splits."""
    manifest_path = Path("config/dataset_manifest_v2.json")
    assert manifest_path.exists(), "config/dataset_manifest_v2.json must exist"

    manifest = DatasetManifest.load_json(manifest_path)
    assert len(manifest.sessions) == 25
    assert len(manifest.locked_test_hash) == 64  # SHA-256

    train_sessions = [s for s in manifest.sessions if s.split == "train"]
    val_sessions = [s for s in manifest.sessions if s.split == "val"]
    locked_sessions = [s for s in manifest.sessions if s.split == "locked_test"]

    assert len(train_sessions) == 15
    assert len(val_sessions) == 5
    assert len(locked_sessions) == 5


def test_reject_duplicate_session_id():
    """Verify rejection of duplicate session IDs."""
    manifest = DatasetManifest()
    s1 = SessionRecord("sess_01", "v1.mp4", "cam_top", "train", "csv1.csv")
    s2 = SessionRecord("sess_01", "v2.mp4", "cam_top", "train", "csv2.csv")
    manifest.add_session(s1)
    manifest.add_session(s2)

    with pytest.raises(ManifestValidationError, match="Duplicate session_id"):
        manifest.validate_splits()


def test_reject_video_data_leakage():
    """Verify rejection of duplicate video reused across train and locked_test."""
    manifest = DatasetManifest()
    s1 = SessionRecord("sess_01", "shared_video.mp4", "cam_top", "train", "csv1.csv")
    s2 = SessionRecord("sess_02", "shared_video.mp4", "cam_top", "locked_test", "csv2.csv")
    manifest.add_session(s1)
    manifest.add_session(s2)

    with pytest.raises(ManifestValidationError, match="Data Leakage Risk"):
        manifest.validate_splits()


def test_locked_test_hash_reproducibility():
    """Verify locked test hash is deterministic and frozen."""
    manifest = DatasetManifest.load_json("config/dataset_manifest_v2.json")
    hash1 = manifest.compute_locked_test_hash()
    hash2 = manifest.compute_locked_test_hash()
    assert hash1 == hash2
    assert hash1 == manifest.locked_test_hash


def test_detect_content_hash_leakage():
    """Verify that find_content_leakage flags duplicate video hashes."""
    manifest = DatasetManifest()
    s1 = SessionRecord("sess_01", "v1.mp4", "cam_top", "train", "csv1.csv", video_sha256="abc123hash")
    s2 = SessionRecord("sess_02", "v2.mp4", "cam_left", "locked_test", "csv2.csv", video_sha256="abc123hash")
    manifest.add_session(s1)
    manifest.add_session(s2)

    leaks = manifest.find_content_leakage()
    assert len(leaks) == 1
    assert leaks[0]["cross_split_leakage"] is True
    assert leaks[0]["session_a"] == "sess_01"
    assert leaks[0]["session_b"] == "sess_02"


def test_proposed_manifest_structure_and_leakage_free():
    """Verify that proposed manifest v2.1 has zero cross-split data leakage."""
    import json
    p = Path("config/proposed_dataset_manifest_v2_1.json")
    assert p.exists(), "Proposed manifest must exist"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    manifest = DatasetManifest.from_dict(data)
    assert len(manifest.sessions) == 23
    leaks = manifest.find_content_leakage()
    assert len(leaks) == 0, f"Proposed manifest must be 100% leakage-free, found: {leaks}"


