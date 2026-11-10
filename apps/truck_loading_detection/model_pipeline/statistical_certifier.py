"""
Statistical Certification & Exact Clopper-Pearson Confidence Engine (Phase 1.11).

Implements the Stage 2 Certification Gate:
  Demonstrates >= 99.0% statistical accuracy at 95.0% Clopper-Pearson confidence
  across >= 30 sessions and >= 3,000 crossing events.

Formula:
  For n trials and k successes, one-sided 95% Clopper-Pearson lower bound:
    L = BetaQuantile(0.05, k, n - k + 1)  for k > 0
    L = 0.05^(1/n)                         for k == n
    L = 0.0                                for k == 0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta


class StatisticalValidationError(Exception):
    """Raised when certification sample size or metric preconditions are violated."""
    pass


@dataclass
class ClopperPearsonResult:
    """Rigorous statistical evaluation result."""

    metric_name: str
    total_trials_n: int
    successes_k: int
    failures: int
    observed_rate: float
    confidence_level: float  # e.g. 0.95
    lower_bound: float
    target_threshold: float  # e.g. 0.99
    is_certified: bool
    session_count: int = 0
    session_rates: List[float] = field(default_factory=list)
    session_rate_std: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "total_trials_n": self.total_trials_n,
            "successes_k": self.successes_k,
            "failures": self.failures,
            "observed_rate": round(self.observed_rate, 5),
            "confidence_level": self.confidence_level,
            "lower_bound": round(self.lower_bound, 5),
            "target_threshold": self.target_threshold,
            "is_certified": self.is_certified,
            "session_count": self.session_count,
            "session_rate_mean": round(float(np.mean(self.session_rates)), 5) if self.session_rates else 0.0,
            "session_rate_std": round(self.session_rate_std, 5),
            "notes": self.notes,
        }


def compute_clopper_pearson_lower_bound(
    successes: int, total: int, confidence: float = 0.95
) -> float:
    """
    Calculate the one-sided exact Clopper-Pearson binomial lower confidence bound.
    confidence: e.g. 0.95 for 95% confidence level (alpha = 1 - 0.95 = 0.05).
    """
    if total <= 0:
        raise ValueError(f"Total trials must be > 0, got {total}")
    if not (0 <= successes <= total):
        raise ValueError(f"Successes must be in [0, {total}], got {successes}")
    if not (0.5 < confidence < 1.0):
        raise ValueError(f"Confidence must be between 0.5 and 1.0, got {confidence}")

    alpha = 1.0 - confidence

    if successes == 0:
        return 0.0
    if successes == total:
        # Exact closed form: alpha^(1/n)
        return float(alpha ** (1.0 / total))

    # Exact Beta quantile
    a = float(successes)
    b = float(total - successes + 1)
    lower = float(beta.ppf(alpha, a, b))
    return float(np.clip(lower, 0.0, 1.0))


class StatisticalCertifier:
    """
    Evaluates Stage 2 Shadow Mode accuracy against the 99% @ 95% confidence requirement.
    """

    def __init__(
        self,
        target_accuracy: float = 0.99,
        confidence_level: float = 0.95,
        min_events: int = 3000,
        min_sessions: int = 30,
    ) -> None:
        self.target = target_accuracy
        self.confidence = confidence_level
        self.min_events = min_events
        self.min_sessions = min_sessions

    def certify_dataset(
        self,
        session_results: List[Tuple[str, int, int]],  # [(session_id, successes, total_events)]
        metric_name: str = "event_recall",
    ) -> ClopperPearsonResult:
        """
        Evaluate full shadow-mode event pool and per-session distribution.
        """
        num_sessions = len(session_results)
        total_n = sum(n for _, _, n in session_results)
        total_k = sum(k for _, k, _ in session_results)
        failures = total_n - total_k

        if total_n < self.min_events:
            notes = (
                f"SHORTFALL: Evaluated {total_n} events < required {self.min_events} events. "
                "Certification gate requires at least 3,000 crossing events."
            )
        elif num_sessions < self.min_sessions:
            notes = (
                f"SHORTFALL: Evaluated {num_sessions} sessions < required {self.min_sessions} sessions."
            )
        else:
            notes = "Dataset satisfies minimum sample size requirements (>= 30 sessions, >= 3,000 events)."

        observed = float(total_k / total_n) if total_n > 0 else 0.0
        lower_bound = compute_clopper_pearson_lower_bound(total_k, total_n, self.confidence) if total_n > 0 else 0.0

        session_rates = [float(k / n) if n > 0 else 0.0 for _, k, n in session_results]
        session_std = float(np.std(session_rates)) if session_rates else 0.0

        # Certified if sample size is met AND lower bound >= 0.99
        is_certified = (
            total_n >= self.min_events
            and num_sessions >= self.min_sessions
            and lower_bound >= self.target
        )

        return ClopperPearsonResult(
            metric_name=metric_name,
            total_trials_n=total_n,
            successes_k=total_k,
            failures=failures,
            observed_rate=observed,
            confidence_level=self.confidence,
            lower_bound=lower_bound,
            target_threshold=self.target,
            is_certified=is_certified,
            session_count=num_sessions,
            session_rates=session_rates,
            session_rate_std=session_std,
            notes=notes,
        )
