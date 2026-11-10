"""
Unit tests for Automated Dataset Integrity & Annotation QA Validator (Phase 1.3).
"""

import pytest
from pathlib import Path
from apps.truck_loading_detection.model_pipeline.dataset_validator import (
    DatasetValidator,
    DatasetQAReport,
)


def test_dataset_validator_synthetic_clean(tmp_path: Path):
    """Verify validator on a compliant dataset structure."""
    ds_dir = tmp_path / "ds"
    for split in ("train", "val", "test"):
        (ds_dir / "images" / split).mkdir(parents=True)
        (ds_dir / "labels" / split).mkdir(parents=True)

    # Add 1 valid image & label per split
    for split in ("train", "val", "test"):
        (ds_dir / "images" / split / f"{split}_01.jpg").touch()
        with open(ds_dir / "labels" / split / f"{split}_01.txt", "w") as f:
            # class 0: carton, class 1: person
            f.write("0 0.5 0.5 0.2 0.2\n")
            f.write("1 0.3 0.3 0.1 0.4\n")

    validator = DatasetValidator(ds_dir)
    report = validator.validate_dataset()

    assert report.is_valid is True
    assert report.num_train_images == 1
    assert report.num_val_images == 1
    assert report.num_test_images == 1
    assert report.num_carton_instances == 3
    assert report.num_person_instances == 3
    assert report.split_leakage_detected is False


def test_dataset_validator_catches_invalid_class(tmp_path: Path):
    """Verify detection of invalid class ID violating 2-class contract."""
    ds_dir = tmp_path / "ds"
    for split in ("train", "val", "test"):
        (ds_dir / "images" / split).mkdir(parents=True)
        (ds_dir / "labels" / split).mkdir(parents=True)

    (ds_dir / "images" / "train" / "img1.jpg").touch()
    with open(ds_dir / "labels" / "train" / "img1.txt", "w") as f:
        f.write("2 0.5 0.5 0.2 0.2\n")  # Class 2 is illegal

    validator = DatasetValidator(ds_dir)
    report = validator.validate_dataset()

    assert report.is_valid is False
    assert any("Invalid class ID 2" in issue for issue in report.issues_found)


def test_dataset_validator_catches_leakage(tmp_path: Path):
    """Verify detection of frame filename duplication between splits."""
    ds_dir = tmp_path / "ds"
    for split in ("train", "val", "test"):
        (ds_dir / "images" / split).mkdir(parents=True)
        (ds_dir / "labels" / split).mkdir(parents=True)

    # Same frame in train and test
    (ds_dir / "images" / "train" / "leak_frame.jpg").touch()
    (ds_dir / "images" / "test" / "leak_frame.jpg").touch()

    validator = DatasetValidator(ds_dir)
    report = validator.validate_dataset()

    assert report.split_leakage_detected is True
    assert report.is_valid is False


def test_dataset_validator_catches_missing_canonical_class(tmp_path: Path):
    """Verify detection when one of the canonical classes (e.g. person) has zero instances."""
    ds_dir = tmp_path / "ds"
    for split in ("train", "val", "test"):
        (ds_dir / "images" / split).mkdir(parents=True)
        (ds_dir / "labels" / split).mkdir(parents=True)

    (ds_dir / "images" / "train" / "img1.jpg").touch()
    with open(ds_dir / "labels" / "train" / "img1.txt", "w") as f:
        f.write("0 0.5 0.5 0.2 0.2\n")  # Only carton (class 0), 0 person instances

    validator = DatasetValidator(ds_dir)
    report = validator.validate_dataset(require_both_classes=True)

    assert report.is_valid is False
    assert any("Missing canonical class: 0 instances of class 1 (person)" in issue for issue in report.issues_found)

