"""
Unit tests for Two-Class Contract and Person Occluder Isolation (Phase 1.1).
"""

import pytest

from apps.truck_loading_detection.model_pipeline.class_contract import (
    CLASS_CARTON,
    CLASS_PERSON,
    DetectionBox,
    InvalidClassContractError,
    is_carton,
    is_person,
    partition_detections,
    validate_carton_event_candidate,
)


def test_canonical_class_constants():
    """Verify strictly two classes: 0 is carton, 1 is person."""
    assert CLASS_CARTON == 0
    assert CLASS_PERSON == 1
    assert is_carton(0) is True
    assert is_carton(1) is False
    assert is_person(1) is True
    assert is_person(0) is False


def test_detection_box_validation():
    """Verify DetectionBox creation and rejection of illegal class IDs."""
    carton_box = DetectionBox(class_id=0, confidence=0.88, bbox=(10.0, 20.0, 50.0, 60.0), track_id=1)
    assert carton_box.is_carton is True
    assert carton_box.is_person is False

    person_box = DetectionBox(class_id=1, confidence=0.92, bbox=(100.0, 50.0, 200.0, 300.0), track_id=2)
    assert person_box.is_carton is False
    assert person_box.is_person is True

    # Class ID 2 is invalid under the 2-class contract
    with pytest.raises(InvalidClassContractError, match="Invalid class_id 2"):
        DetectionBox(class_id=2, confidence=0.80, bbox=(10.0, 20.0, 50.0, 60.0))


def test_person_cannot_enter_carton_event_stream():
    """
    CRITICAL CONTRACT RULE:
    Person detections must NEVER trigger carton crossing events.
    """
    carton_box = DetectionBox(class_id=0, confidence=0.85, bbox=(10.0, 20.0, 50.0, 60.0), track_id=10)
    person_box = DetectionBox(class_id=1, confidence=0.95, bbox=(100.0, 50.0, 200.0, 300.0), track_id=20)

    # Carton is accepted
    assert validate_carton_event_candidate(carton_box) is True

    # Person raises an explicit violation error
    with pytest.raises(InvalidClassContractError, match="Person detection.*attempted to enter carton event"):
        validate_carton_event_candidate(person_box)


def test_partition_detections():
    """Verify clean partitioning into cartons and persons."""
    boxes = [
        DetectionBox(class_id=0, confidence=0.8, bbox=(0, 0, 10, 10)),
        DetectionBox(class_id=1, confidence=0.9, bbox=(20, 20, 50, 50)),
        DetectionBox(class_id=0, confidence=0.7, bbox=(60, 60, 70, 70)),
    ]
    cartons, persons = partition_detections(boxes)
    assert len(cartons) == 2
    assert len(persons) == 1
    assert all(c.is_carton for c in cartons)
    assert all(p.is_person for p in persons)
