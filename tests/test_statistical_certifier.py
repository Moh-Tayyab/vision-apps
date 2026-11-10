"""
Unit tests for Exact Clopper-Pearson Binomial Confidence & Statistical Certification (Phase 1.11).
"""

import pytest
import math
from apps.truck_loading_detection.model_pipeline.statistical_certifier import (
    compute_clopper_pearson_lower_bound,
    StatisticalCertifier,
    ClopperPearsonResult,
)


def test_clopper_pearson_boundary_conditions():
    """Verify exact edge cases: 0 successes, all successes, and invalid inputs."""
    # k = 0 -> lower bound 0.0
    assert compute_clopper_pearson_lower_bound(0, 100, confidence=0.95) == 0.0

    # k = n -> exact closed form: 0.05^(1/n)
    n = 300
    expected = 0.05 ** (1.0 / n)
    actual = compute_clopper_pearson_lower_bound(n, n, confidence=0.95)
    assert pytest.approx(actual, rel=1e-6) == expected

    # Invalid cases
    with pytest.raises(ValueError):
        compute_clopper_pearson_lower_bound(10, 0)
    with pytest.raises(ValueError):
        compute_clopper_pearson_lower_bound(11, 10)
    with pytest.raises(ValueError):
        compute_clopper_pearson_lower_bound(5, 10, confidence=1.5)


def test_clopper_pearson_known_benchmarks():
    """Test against statistical reference values for large samples."""
    # 3,000 events, 0 failures: L >= 0.99
    l_bound_perfect = compute_clopper_pearson_lower_bound(3000, 3000, confidence=0.95)
    assert l_bound_perfect >= 0.999
    assert pytest.approx(l_bound_perfect, rel=1e-5) == 0.05 ** (1.0 / 3000)

    # 3,000 events, 10 failures (k = 2990): observed = 99.67%
    l_bound_2990 = compute_clopper_pearson_lower_bound(2990, 3000, confidence=0.95)
    assert l_bound_2990 > 0.99  # Passes 99% lower bound
    assert pytest.approx(l_bound_2990, abs=1e-3) == 0.994

    # 3,000 events, 50 failures (k = 2950): observed = 98.33%
    l_bound_2950 = compute_clopper_pearson_lower_bound(2950, 3000, confidence=0.95)
    assert l_bound_2950 < 0.99  # Fails 99% lower bound


def test_statistical_certifier_shortfalls():
    """Verify that sample size shortfalls prevent certification regardless of rate."""
    certifier = StatisticalCertifier(target_accuracy=0.99, confidence_level=0.95, min_events=3000, min_sessions=30)

    # Case 1: 100% success rate but only 20 sessions (shortfall)
    sessions_short = [(f"sess_{i:02d}", 150, 150) for i in range(20)]  # 3000 events across 20 sessions
    res_short_sessions = certifier.certify_dataset(sessions_short)
    assert not res_short_sessions.is_certified
    assert "SHORTFALL" in res_short_sessions.notes

    # Case 2: 35 sessions, but only 2,500 events (shortfall)
    sessions_few_events = [(f"sess_{i:02d}", 70, 70) for i in range(35)]  # 2450 events
    res_few_events = certifier.certify_dataset(sessions_few_events)
    assert not res_few_events.is_certified
    assert "SHORTFALL" in res_few_events.notes


def test_statistical_certifier_passing_gate():
    """Verify compliant certification with >= 30 sessions and >= 3,000 events satisfying L >= 0.99."""
    certifier = StatisticalCertifier(target_accuracy=0.99, confidence_level=0.95, min_events=3000, min_sessions=30)

    # 30 sessions, 100 events each, 1 failure per session (k = 2970, n = 3000)
    # Observed: 2970/3000 = 99.0%
    # With 30 failures, L = BetaQuantile(0.05, 2970, 31) ≈ 0.985 < 0.99 (should not certify)
    borderline_sessions = [(f"sess_{i:02d}", 99, 100) for i in range(30)]
    res_borderline = certifier.certify_dataset(borderline_sessions)
    assert not res_borderline.is_certified

    # 30 sessions, 100 events each, only 5 failures total (k = 2995, n = 3000)
    # Observed: 99.83%
    high_perf_sessions = [(f"sess_{i:02d}", 100 if i >= 5 else 99, 100) for i in range(30)]
    res_certified = certifier.certify_dataset(high_perf_sessions)
    assert res_certified.is_certified
    assert res_certified.lower_bound >= 0.99
    assert res_certified.session_count == 30
    assert res_certified.total_trials_n == 3000
