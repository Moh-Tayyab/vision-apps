"""
Unit tests for Person Occlusion Mask & Camera Visibility Scoring (Phase 1.5).
"""

import pytest

from apps.truck_loading_detection.model_pipeline.class_contract import (
    CLASS_PERSON,
    DetectionBox,
    InvalidClassContractError,
)
from apps.truck_loading_detection.model_pipeline.person_occlusion import (
    CorridorZone,
    PersonOccluderMask,
)


def test_zero_persons_full_visibility():
    """When no persons are present, Zone B visibility must be 1.0."""
    zone_b = CorridorZone(name="Zone B", x1=400, y1=100, x2=880, y2=620)
    mask_calc = PersonOccluderMask(camera_id="cam_top", zone_b=zone_b)

    res = mask_calc.compute_visibility([])
    assert res.visibility_score == 1.0
    assert res.blocked_ratio == 0.0
    assert res.is_blocked is False


def test_person_outside_zone_full_visibility():
    """A person outside Zone B does not block the corridor."""
    zone_b = CorridorZone(name="Zone B", x1=400, y1=100, x2=880, y2=620)
    mask_calc = PersonOccluderMask(camera_id="cam_top", zone_b=zone_b)

    person_outside = DetectionBox(class_id=1, confidence=0.9, bbox=(50, 50, 200, 300))
    res = mask_calc.compute_visibility([person_outside])

    assert res.visibility_score == 1.0
    assert res.blocked_ratio == 0.0
    assert res.num_persons_in_zone == 0
    assert res.is_blocked is False


def test_moderate_person_blockage():
    """
    Test moderate blockage of Zone B:
    Zone B size = 400x400 = 160,000 px^2
    Person covering 400x80 = 32,000 px^2 -> BlockedRatio = 32000/160000 = 0.20
    V_c = 1.0 - 1.5 * 0.20 = 0.70
    """
    zone_b = CorridorZone(name="Zone B", x1=200, y1=100, x2=600, y2=500)
    mask_calc = PersonOccluderMask(camera_id="cam_left", zone_b=zone_b)

    # Person overlapping exactly bottom 20% of Zone B
    person = DetectionBox(class_id=1, confidence=0.85, bbox=(200, 420, 600, 500))
    res = mask_calc.compute_visibility([person])

    assert pytest.approx(res.blocked_ratio, abs=0.01) == 0.20
    assert pytest.approx(res.visibility_score, abs=0.02) == 0.70
    assert res.is_blocked is False  # 0.70 >= 0.35


def test_severe_person_blockage_triggers_abstention():
    """
    Test heavy blockage of Zone B:
    If BlockedRatio = 0.50 (50% blocked):
    V_c = max(0, 1.0 - 1.5 * 0.50) = 1.0 - 0.75 = 0.25 < 0.35
    is_blocked MUST BE TRUE (triggering camera weight W_c = 0 abstention).
    """
    zone_b = CorridorZone(name="Zone B", x1=200, y1=100, x2=600, y2=500)
    mask_calc = PersonOccluderMask(camera_id="cam_left", zone_b=zone_b)

    # Person overlapping half of Zone B (height 200/400)
    person = DetectionBox(class_id=1, confidence=0.95, bbox=(200, 300, 600, 500))
    res = mask_calc.compute_visibility([person])

    assert pytest.approx(res.blocked_ratio, abs=0.01) == 0.50
    assert pytest.approx(res.visibility_score, abs=0.02) == 0.25
    assert res.is_blocked is True


def test_multi_person_union_without_double_counting():
    """Two overlapping persons must be unioned so area is not counted twice."""
    zone_b = CorridorZone(name="Zone B", x1=100, y1=100, x2=500, y2=500)
    mask_calc = PersonOccluderMask(camera_id="cam_right", zone_b=zone_b)

    # Two identical overlapping persons covering same 200x200 box
    p1 = DetectionBox(class_id=1, confidence=0.9, bbox=(100, 100, 300, 300))
    p2 = DetectionBox(class_id=1, confidence=0.8, bbox=(100, 100, 300, 300))

    res = mask_calc.compute_visibility([p1, p2])
    # Box is (200x200) / (400x400) = 40000 / 160000 = 0.25
    assert pytest.approx(res.blocked_ratio, abs=0.01) == 0.25
    assert res.num_persons_detected == 2
