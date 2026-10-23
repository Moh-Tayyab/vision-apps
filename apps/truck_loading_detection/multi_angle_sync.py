#!/usr/bin/env python3
"""
Multi-angle temporal synchronisation.

Recovers the recording start offset between camera feeds by aligning the
SEQUENCE OF LOADING EVENTS (tripwire crossings) captured by each camera.

Why event-sequence alignment and not signal cross-correlation:
    Frame-difference / person-count signals from different viewpoints of the
    same scene are weakly correlated (measured NCC 0.12-0.32 on real phone
    footage), so they cannot resolve a reliable lag. The loading events
    themselves are viewpoint independent: the same physical worker crossing a
    tripwire produces one event per camera, so aligning those event
    timestamps yields a sharp, auditable offset.

Offset convention:
    A camera's `start_offset_sec` is the global-timeline time at which its
    recording began. An event captured at that camera's local time `t`
    occurred at global time `t + start_offset_sec`.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple


def _event_key(ev: Dict[str, Any]) -> Tuple[str, float]:
    return (ev["direction"], float(ev["timestamp_sec"]))


def offset_curve(
    reference_events: Sequence[Dict[str, Any]],
    other_events: Sequence[Dict[str, Any]],
    max_offset_sec: float = 30.0,
    tol_sec: float = 1.5,
    fps: float = 30.0,
) -> List[Tuple[float, int, float]]:
    """
    For every candidate offset, count how many reference events can be paired
    with an other-camera event (same direction, within tolerance) and record
    the total residual error of those pairings.
    """
    step = 1.0 / max(fps, 1e-6)
    steps = int(round(max_offset_sec / step))
    curve: List[Tuple[float, int, float]] = []

    for i in range(-steps, steps + 1):
        off = i * step
        matched = 0
        residual = 0.0
        used = set()
        for ref in reference_events:
            best_j, best_d = -1, None
            for j, oth in enumerate(other_events):
                if j in used:
                    continue
                if oth["direction"] != ref["direction"]:
                    continue
                d = abs((float(oth["timestamp_sec"]) + off) - float(ref["timestamp_sec"]))
                if d <= tol_sec and (best_d is None or d < best_d):
                    best_j, best_d = j, d
            if best_j >= 0:
                used.add(best_j)
                matched += 1
                residual += float(best_d)
        curve.append((round(off, 4), matched, round(residual, 4)))
    return curve


def estimate_offset(
    reference_events: Sequence[Dict[str, Any]],
    other_events: Sequence[Dict[str, Any]],
    max_offset_sec: float = 30.0,
    tol_sec: float = 1.5,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """
    Estimates the start offset of `other_events` relative to `reference_events`.

    Returns a dict with the offset, the number of matched events, a confidence
    score in [0, 1] and the full correlation curve for auditing.
    """
    curve = offset_curve(reference_events, other_events, max_offset_sec, tol_sec, fps)
    if not curve:
        return {
            "offset_sec": 0.0,
            "matched_events": 0,
            "confidence": 0.0,
            "reference_event_count": len(reference_events),
            "other_event_count": len(other_events),
            "curve": [],
        }

    # Primary criterion: most matched events. Tie-break: tightest residual
    # (the offset that lines the events up most precisely), then smallest
    # magnitude so a genuine zero-offset alignment is preferred.
    best_offset, best_matches, best_residual = min(
        curve, key=lambda t: (-t[1], t[2], abs(t[0]))
    )
    n_ref = len(reference_events)
    n_other = len(other_events)
    denom = max(1, min(n_ref, n_other))

    # Separation: how much better the peak is than the runner-up outside a
    # +/-1.5s neighbourhood. 1.0 means an unambiguous peak.
    peak_score = best_matches / denom
    rivals = [s for o, s, _r in curve if abs(o - best_offset) > 1.5]
    runner_up = max(rivals) if rivals else 0
    separation = (best_matches - runner_up) / denom

    confidence = max(0.0, min(1.0, 0.6 * peak_score + 0.4 * max(0.0, separation)))

    return {
        "offset_sec": round(best_offset, 3),
        "matched_events": best_matches,
        "residual_sec": round(best_residual, 3),
        "runner_up_matches": runner_up,
        "reference_event_count": n_ref,
        "other_event_count": n_other,
        "confidence": round(confidence, 3),
        "curve": curve,
    }


def resolve_offsets(
    per_camera_events: Dict[str, List[Dict[str, Any]]],
    reference_camera_id: Optional[str] = None,
    max_offset_sec: float = 30.0,
    tol_sec: float = 1.5,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """
    Resolves a start offset for every camera relative to a reference camera.

    Each camera is aligned against the reference directly (not transitively) so
    that estimation error never accumulates across a chain of cameras.
    """
    cams = list(per_camera_events.keys())
    if not cams:
        return {"reference": None, "offsets": {}, "diagnostics": {}}

    if reference_camera_id is None or reference_camera_id not in per_camera_events:
        # Reference = the camera with the most events (most reliable signal).
        reference_camera_id = max(
            cams, key=lambda c: len(per_camera_events.get(c, []))
        )

    ref_events = per_camera_events[reference_camera_id]
    offsets: Dict[str, float] = {reference_camera_id: 0.0}
    diagnostics: Dict[str, Any] = {}

    for cam in cams:
        if cam == reference_camera_id:
            continue
        est = estimate_offset(
            ref_events,
            per_camera_events[cam],
            max_offset_sec=max_offset_sec,
            tol_sec=tol_sec,
            fps=fps,
        )
        offsets[cam] = est["offset_sec"]
        diagnostics[cam] = {k: v for k, v in est.items() if k != "curve"}

    return {
        "reference": reference_camera_id,
        "offsets": offsets,
        "diagnostics": diagnostics,
    }
