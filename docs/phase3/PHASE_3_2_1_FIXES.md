# PHASE 3.2.1 — ARCHITECTURAL FIXES & CODE IMPLEMENTATIONS

**Date:** November 2026  
**System:** Three-Camera Carton Counting & Physical Continuity Engine  
**Author:** Senior Computer Vision Engineer, Tracking Systems Architect, and Database Reliability Engineer  
**Status:** IMPLEMENTED & TESTED  

---

## 1. Overview of Code Modifications

All changes in Phase 3.2.1 were targeted to resolve demonstrated root causes without architectural rewrites, model retraining, or loosening multi-camera consensus rules.

The modified files are:
1. `apps/truck_loading_detection/three_camera_system/consensus_policy.py`
2. `apps/truck_loading_detection/three_camera_system/event_store.py`
3. `apps/truck_loading_detection/three_camera_system/pipeline.py`
4. `tests/test_phase3_2_1_integrity.py` (New comprehensive test suite)
5. `tools/run_phase3_2_1_pipeline_evaluation.py` (New evaluation and metrics harness)

---

## 2. Fix 1: Two-Camera Agreement Confidence Floor (Restoring 100% Precision)

### Problem
In Phase 3.2, `ThreeCameraConsensusPolicy.evaluate()` automatically committed any two-camera agreement regardless of confidence score. Low-confidence edge noise (`session_21_p32_ev_00010`, mean confidence $0.192$) was committed as `is_auto_confirmed=1`, dropping precision to 75%.

### File Modified
`apps/truck_loading_detection/three_camera_system/consensus_policy.py`

### Changes Made
```python
        elif n_supporting == 2:
            # 2 cameras observed, 1 missing (e.g. occlusion)
            if mean_conf >= self.min_two_camera_conf:
                score = min(0.92, mean_conf * 0.95)
                return ConsensusVerdict(
                    level=ConsensusLevel.TWO_CAMERA_AGREEMENT,
                    action=DecisionAction.COMMIT,
                    supporting_cameras=supporting,
                    missing_cameras=missing,
                    mean_confidence=mean_conf,
                    evidence_score=score,
                    decision_reason=f"Two cameras confirm crossing; '{missing[0]}' missed observation (probable occlusion)",
                )
            elif mean_conf >= 0.15:
                # Low confidence two-camera observation: hold for human review
                score = min(0.60, mean_conf * 0.75)
                return ConsensusVerdict(
                    level=ConsensusLevel.CONFLICTING_OR_UNCERTAIN,
                    action=DecisionAction.HOLD_FOR_REVIEW,
                    supporting_cameras=supporting,
                    missing_cameras=missing,
                    mean_confidence=mean_conf,
                    evidence_score=score,
                    decision_reason=f"Two cameras observed crossing but mean confidence {mean_conf:.3f} is below auto-confirm threshold {self.min_two_camera_conf:.2f}; routed to review",
                )
            else:
                # Sub-0.15 noise: suppress
                return ConsensusVerdict(
                    level=ConsensusLevel.CONFLICTING_OR_UNCERTAIN,
                    action=DecisionAction.SUPPRESS_AS_NOISE,
                    supporting_cameras=supporting,
                    missing_cameras=missing,
                    mean_confidence=mean_conf,
                    evidence_score=mean_conf * 0.30,
                    decision_reason=f"Two cameras detected noise below minimum confidence threshold 0.15 (mean: {mean_conf:.3f})",
                )
```

---

## 3. Fix 2: Recovery of Missed Crossing `session_21_ev_02`

### Problem
Single-camera lateral detection on `cam_side` with confidence $0.2239$ was discarded by a hardcoded $0.25$ floor in `pipeline.py`, dropping `session_21_ev_02`.

### File Modified
`apps/truck_loading_detection/three_camera_system/pipeline.py`

### Changes Made
```python
        # 9. Consensus Policy
        self.consensus_policy = ThreeCameraConsensusPolicy(
            all_camera_ids=self.camera_ids,
            min_single_camera_conf=conf_threshold,  # Restored to 0.15 (raw detection floor)
            min_two_camera_conf=0.25,
        )
```
Detections between $0.15$ and $0.25$ are now admitted into the pipeline and routed safely to `HOLD_FOR_REVIEW`.

---

## 4. Fix 3: Direction Isolation in Coalescence Windows (`session_21_ev_04`)

### Problem
`CrossingWindow` coalesced candidate events within the $1.5\text{s}$ transit corridor without checking trajectory directions. A worker exiting (`OUT` candidate at $11.4\text{s}$) was merged with the incoming carton crossing (`IN` candidate at $11.633\text{s}$ and $12.467\text{s}$), triggering `has_dir_conflict = True` and causing the entire window to be silently discarded.

### File Modified
`apps/truck_loading_detection/three_camera_system/pipeline.py`

### Changes Made
```python
            matched_win = None
            for win in self._active_crossing_windows:
                if win.committed:
                    continue
                # Temporal proximity check (canonical transit window)
                if abs(ev.timestamp_sec - win.start_time) <= self.time_window_sec:
                    # Direction compatibility: opposite directions (IN vs OUT) must never share a crossing window
                    if any(o_ev.direction != ev.direction for other_cam_evs in win.local_events.values() for o_ev in other_cam_evs):
                        continue
                    # Spatial separation check ...
```

---

## 5. Fix 4: Durable Crossing Occurrence Idempotency

### Problem
Time-windowed deduplication ($\Delta t \le 3.5\text{s}$) without persistent occurrence tokens allowed replayed events to double-commit if timestamps jittered, and falsely suppressed legitimate repeated crossings.

### File Modified
`apps/truck_loading_detection/three_camera_system/event_store.py`

### Changes Made
1. **Schema Migration:** Added `crossing_occurrence_id` column and unique index `idx_events_occurrence`:
```sql
ALTER TABLE carton_crossing_events ADD COLUMN crossing_occurrence_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_events_occurrence 
ON carton_crossing_events(session_id, crossing_occurrence_id);
```

2. **Durable Token Resolution:**
```python
        with self._lock:
            with self._get_connection() as conn:
                # Resolve occurrence index and durable crossing_occurrence_id
                if event_data.get("crossing_occurrence_id"):
                    crossing_occ_id = str(event_data["crossing_occurrence_id"])
                    occurrence_idx = int(event_data.get("occurrence_index", 1))
                elif "occurrence_index" in event_data:
                    occurrence_idx = int(event_data["occurrence_index"])
                    crossing_occ_id = f"{session_id}:{global_cid}:c{occurrence_idx}:{direction}"
                else:
                    # Query prior occurrences for this carton in this session
                    cur_occ = conn.execute(
                        """
                        SELECT event_id, timestamp_sec, direction, crossing_occurrence_id
                        FROM carton_crossing_events
                        WHERE session_id = ? AND global_carton_id = ?
                        ORDER BY timestamp_sec ASC;
                        """,
                        (session_id, global_cid),
                    )
                    prior_rows = cur_occ.fetchall()
                    if not prior_rows:
                        occurrence_idx = 1
                        crossing_occ_id = f"{session_id}:{global_cid}:c1:{direction}"
                    else:
                        # Check if any prior event is within duplicate window (3.5s)
                        matched_duplicate = False
                        for pr in prior_rows:
                            dt = abs(timestamp_sec - float(pr["timestamp_sec"]))
                            if dt <= 3.5:
                                crossing_occ_id = pr["crossing_occurrence_id"] or f"{session_id}:{global_cid}:c1:{pr['direction']}"
                                matched_duplicate = True
                                break
                        if not matched_duplicate:
                            occurrence_idx = len(prior_rows) + 1
                            crossing_occ_id = f"{session_id}:{global_cid}:c{occurrence_idx}:{direction}"

                # Atomically check existing record by event_id or stable persistent crossing_occurrence_id
                cur = conn.execute(
                    """
                    SELECT event_id, state, timestamp_sec, carton_count, participating_cameras_json, notes, crossing_occurrence_id
                    FROM carton_crossing_events
                    WHERE event_id = ? OR (session_id = ? AND crossing_occurrence_id = ?);
                    """,
                    (event_id, session_id, crossing_occ_id),
                )
```

---

## 6. Fix 5: Multi-Resolution Spatial Normalization

### Problem
Spatial thresholds ($60\text{px}$ lane separation, $140\text{px}$ corridor drift) assumed $720\text{p}$ footage. At higher resolutions, these distances become too tight.

### Implementation
Normalized scaling based on input video frame resolution:
```python
scale_y = frame_height / 720.0
scale_x = frame_width / 1280.0
lane_tol_scaled = base_lane_tol * scale_y
drift_scaled = base_drift * max(scale_x, scale_y)
```
Implemented and verified in `tests/test_phase3_2_1_integrity.py::test_multi_resolution_lane_scaling`.
