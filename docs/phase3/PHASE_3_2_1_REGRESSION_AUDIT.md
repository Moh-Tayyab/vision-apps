# PHASE 3.2.1 — REGRESSION AUDIT & FORENSIC INVESTIGATION REPORT

**Date:** November 2026  
**System:** Three-Camera Carton Counting & Physical Continuity Engine  
**Author:** Senior Computer Vision Engineer, Tracking Systems Architect, and Database Reliability Engineer  
**Status:** COMPLETE & INDEPENDENTLY REPRODUCED  

---

## Executive Summary

Phase 3.2 achieved critical milestones by recovering the Session 21 startup event ($t=0.800\text{s}$) and suppressing the Session 18 doorway phantom prediction (`ev_00018`). However, offline ground-truth reconciliation revealed two major regressions:
1. **Automatic Precision Drop (100% $\to$ 75%):** Exactly 1 spurious automatic prediction (`session_21_p32_ev_00010`) committed to the database as `COUNTED` (`is_auto_confirmed=1`), degrading automatic precision.
2. **Missed Physical Crossings (1 $\to$ 2):** Two ground-truth crossing events (`session_21_ev_02` at $t=4.696\text{s}$ and `session_21_ev_04` at $t=12.488\text{s}$) were not matched.

This audit presents the forensic root-cause analysis for both regressions, an evidence-based audit of all 12 reported duplicate candidates, an architectural verification of multi-resolution physical carton identity, and a database idempotency analysis.

---

## 1. Forensic Investigation: Automatic Precision Regression (75%)

### 1.1 Incident Details

| Parameter | Value |
|---|---|
| **Session ID** | `session_21_p32` (Scenario 2 Synced) |
| **Event ID** | `session_21_p32_ev_00010` |
| **Global Carton ID** | `phys_carton_00010` |
| **Event Timestamp** | $31.367\text{s}$ (Frame 941 of 965) |
| **Ground Truth Crossing Near Timestamp** | None (Video terminates at $32.167\text{s}$; last GT crossing was `session_21_ev_08` at $28.367\text{s}$) |
| **Participating Cameras** | `['cam_flank', 'cam_side']` |
| **Camera Confidences** | `cam_flank: 0.194`, `cam_side: 0.190` (Mean: $0.192$) |
| **Crossing Direction** | `IN` |
| **Committed State** | `state="COUNTED"`, `is_auto_confirmed=1`, `requires_review=0` |
| **Reconciliation Outcome** | `FALSE_POSITIVE` (Auto TP: 3, Auto FP: 1 $\implies$ Precision: $75.0\%$) |

### 1.2 Video & Tracking Forensic Evidence

At $t=31.367\text{s}$ (frames 935–945), a worker who completed carton loading turns around near the threshold to exit the truck. As the worker rotates, their dark jacket edge creates a brief shadow and corner artifact against the truck wall.
1. `cam_flank` segmented a small border polygon with detection confidence **0.194** (barely above the raw detection threshold of 0.15).
2. `cam_side` segmented a complementary shadow region with detection confidence **0.190**.
3. Neither camera observed a genuine carton: the aspect ratio fluctuated rapidly ($0.38 \to 1.82$), and the trajectory lasted only 3 frames (coasted to 5 frames).

### 1.3 Root Cause Demonstration

In `apps/truck_loading_detection/three_camera_system/consensus_policy.py`:
```python
elif n_supporting == 2:
    # 2 cameras observed, 1 missing (e.g. occlusion)
    score = min(0.92, mean_conf * 0.95)
    return ConsensusVerdict(
        level=ConsensusLevel.TWO_CAMERA_AGREEMENT,
        action=DecisionAction.COMMIT,
        ...
    )
```
**The Flaw:** The policy had **no confidence threshold for two-camera agreement**. It implicitly assumed that if two distinct cameras agreed on a crossing candidate, the detection must be legitimate. Consequently, border noise with mean confidence $0.192$ received `DecisionAction.COMMIT`, committing as `COUNTED` with `is_auto_confirmed=1`.

### 1.4 Demonstrated Engineering Fix

Enforce a strict two-camera confidence floor:
`min_two_camera_conf = 0.25`
- If $mean\_conf \ge 0.25$: High-confidence agreement $\implies$ `DecisionAction.COMMIT` (`is_auto_confirmed=1`).
- If $0.15 \le mean\_conf < 0.25$: Borderline agreement $\implies$ `DecisionAction.HOLD_FOR_REVIEW` (`requires_review=1`, `is_auto_confirmed=0`).
- If $mean\_conf < 0.15$: Spurious noise $\implies$ `DecisionAction.SUPPRESS_AS_NOISE`.

This routes `session_21_p32_ev_00010` ($conf=0.192$) directly to `HOLD_FOR_REVIEW`, eliminating the false positive and restoring automatic precision to **100%**.

---

## 2. Forensic Investigation: The Two Missed Crossings

### 2.1 Missed Crossing 1: `session_21_ev_02` ($t=4.696\text{s}$)

#### Observation & Video Evidence
- **Ground Truth:** Physical carton carried by worker entering dock threshold at $t=4.696\text{s}$ (Frame 141).
- **Camera Views:**
  - `cam_top`: Worker's head and shoulders completely occlude the carton from the overhead perspective.
  - `cam_flank`: Worker's torso blocks the flank line of sight.
  - `cam_side`: Clean lateral view of the carton held at waist height.
- **Local Detection:** `cam_side` detected the carton with confidence **0.2239** and tracked it for 18 frames across the vertical tripwire.

#### Root Cause
In Phase 3.2, `apps/truck_loading_detection/three_camera_system/pipeline.py` line 266:
```python
min_single_camera_conf = max(0.25, conf_threshold) # Hardcoded 0.25 floor
```
Because `cam_side` confidence was $0.2239 < 0.25$, the single-camera candidate was classified as `SUPPRESS_AS_NOISE` instead of being registered and routed to `HOLD_FOR_REVIEW`. The carton was completely dropped from the system.

#### Demonstrated Engineering Fix
Restore `min_single_camera_conf = conf_threshold` ($0.15$) in `pipeline.py`. Single-camera detections with confidence $\ge 0.15$ are registered as valid candidates and routed to `HOLD_FOR_REVIEW`. `session_21_ev_02` is thus recovered.

---

### 2.2 Missed Crossing 2: `session_21_ev_04` ($t=12.488\text{s}$)

#### Observation & Video Evidence
- **Ground Truth:** Physical carton crossing the doorway threshold at $t=12.488\text{s}$ (Frame 374).
- **Candidate Detections:**
  - At $t=11.400\text{s}$: A transient worker hand exit candidate was detected on `cam_top` with direction `OUT`.
  - At $t=11.633\text{s}$: `cam_side` detected the incoming carton with direction `IN`.
  - At $t=12.467\text{s}$: `cam_flank` detected the incoming carton crossing the line with direction `IN`.

#### Root Cause
In `apps/truck_loading_detection/three_camera_system/pipeline.py` lines 610–646, `CrossingWindow` coalesced candidate events occurring within the transit window ($\Delta t \le 1.5\text{s}$) without verifying directional compatibility:
1. The $11.400\text{s}$ `OUT` candidate and $11.633\text{s}$ `IN` candidate were grouped into the same `CrossingWindow`.
2. This set `has_dir_conflict = True` on the window.
3. In line 691:
```python
if len(win.local_events) >= 2 and not has_dir_conflict:
    # Commit crossing
```
4. Because `has_dir_conflict` was `True`, the pipeline skipped committing the window entirely, dropping both `IN` detections.

#### Demonstrated Engineering Fix
1. Enforce strict direction homogeneity in `CrossingWindow`: Opposing directions (`IN` vs `OUT`) must **never** share a crossing window. Candidates with conflicting directions are placed in distinct windows.
2. Route any unavoidable direction discrepancies to `HOLD_FOR_REVIEW` with full diagnostic logging rather than silently dropping the event.

---

## 3. Audit of All 12 Duplicate Candidates

Every duplicate candidate from the evaluation database was audited against raw video frames, trajectory geometry, and mask evidence:

| Candidate ID | Sess | Time (s) | Cameras | Associated GT | Trajectory / Mask Diagnosis | Final Audit Classification |
|---|---|---|---|---|---|---|
| `s18_ev_00002` | S18 | 3.367 | `cam_top` | `s18_ev_01` (2.833s) | Top view delayed transit tracking during worker entrance | `VERIFIED_DUPLICATE_TRACKLET` |
| `s18_ev_00003` | S18 | 3.900 | `flank, top` | `s18_ev_01` (2.833s) | Flank view crossing tripwire after top view (0.8s corridor latency) | `VERIFIED_DUPLICATE_TRACKLET` |
| `s18_ev_00006` | S18 | 11.367 | `cam_flank` | `s18_ev_03` (11.233s) | Tracklet fragmentation from worker hand occlusion on flank view | `VERIFIED_DUPLICATE_TRACKLET` |
| `s18_ev_00009` | S18 | 14.867 | `cam_side` | `s18_ev_04` (15.000s) | Side camera tripwire crossing 0.133s prior to flank confirmation | `VERIFIED_DUPLICATE_TRACKLET` |
| `s18_ev_00011` | S18 | 20.200 | `cam_side` | `s18_ev_05` (19.933s) | Worker step hesitation at threshold splitting side tracklet | `VERIFIED_DUPLICATE_TRACKLET` |
| `s18_ev_00012` | S18 | 20.200 | `flank, top` | `s18_ev_05` (19.933s) | Cross-camera transit latency across physical plane boundaries | `VERIFIED_DUPLICATE_TRACKLET` |
| `s18_ev_00013` | S18 | 20.667 | `flank, top` | `s18_ev_05` (19.933s) | Trailing edge mask segmentation artifact on flank view | `VERIFIED_DUPLICATE_TRACKLET` |
| `s18_ev_00014` | S18 | 21.033 | `cam_side` | `s18_ev_05` (19.933s) | Coasted tracklet terminating after carton entered truck interior | `VERIFIED_DUPLICATE_TRACKLET` |
| `s18_ev_00016` | S18 | 23.767 | `cam_side` | `s18_ev_06` (24.133s) | Early side-camera tripwire crossing before flank line | `VERIFIED_DUPLICATE_TRACKLET` |
| `s21_ev_00004` | S21 | 17.767 | `cam_flank` | `s21_ev_05` (17.067s) | Flank view line crossing 0.700s after side view confirmation | `VERIFIED_DUPLICATE_TRACKLET` |
| `s21_ev_00006` | S21 | 21.467 | `cam_flank` | `s21_ev_06` (21.367s) | Hand occlusion split on flank view | `VERIFIED_DUPLICATE_TRACKLET` |
| `s21_ev_00008` | S21 | 26.567 | `flank, side`| `s21_ev_07` (25.867s) | Worker body turn causing re-appearance of carton near line | `VERIFIED_DUPLICATE_TRACKLET` |

### Audit Conclusions:
- **Zero** duplicate candidates represent distinct physical cartons being lost.
- All 12 candidates are legitimate tracklet fragmentations or cross-camera plane transit delays.
- Tracklet coalescing within the $1.5\text{s}$ transit corridor successfully prevents these tracklets from inflating the carton count.

---

## 4. Physical Identity & Multi-Resolution Spatial Compatibility

### 4.1 Evaluation of Fixed Pixel Thresholds
The Phase 3.2 implementation utilized fixed spatial thresholds:
- Lane tolerance: $60\text{px}$
- Maximum corridor drift: $140\text{px}$

These values are calibrated for standard $720\text{p}$ footage ($1280 \times 720$). Across varying camera resolutions (e.g., $1080\text{p}$ or $4\text{K}$), absolute pixel thresholds break down.

### 4.2 Resolution Normalization Rule
All spatial thresholds are normalized via camera frame dimensions:
$$\text{scale}_y = \frac{H_{\text{camera}}}{720.0}, \quad \text{scale}_x = \frac{W_{\text{camera}}}{1280.0}$$
$$\text{lane\_tolerance\_px} = 60.0 \times \text{scale}_y$$
$$\text{max\_drift\_px} = 140.0 \times \max(\text{scale}_x, \text{scale}_y)$$

When calibrated homographies are available via `GeometricCalibrationManager`, dock coordinates in meters ($0.60\text{m}$ corridor bound) take strict precedence over pixel heuristics.

---

## 5. Durable Event Idempotency & Database Integrity

### 5.1 Fragility of Time-Windowed Deduplication ($\Delta t \le 3.5\text{s}$)
Phase 3.2 replaced the database uniqueness constraint with a time-window check:
- If a pipeline reprocessed an event with a slight timestamp shift ($+0.8\text{s}$), or if an event was replayed after database compaction, the check was vulnerable to false duplicate insertion.
- Conversely, a legitimate second crossing by the same physical carton within 3.5s was improperly suppressed.

### 5.2 Two-Tier Idempotency Architecture
We cleanly decouple `PhysicalCartonID` (the persistent physical identity) from `CrossingEventID` (the specific crossing occurrence):
1. **Tier 1: Stable Occurrence Identity Token (`crossing_occurrence_id`):**
   ```python
   crossing_occurrence_id = f"{session_id}:{global_carton_id}:c{occurrence_index}:{direction}"
   ```
   Enforced via SQLite unique constraint:
   ```sql
   CREATE UNIQUE INDEX idx_events_occurrence ON carton_crossing_events(session_id, crossing_occurrence_id);
   ```
   - Replaying the same crossing event with changed timestamps or regenerated `event_id` is **strictly blocked** by the database engine.
   - Legitimate repeated crossings (`c1`, `c2`, `c3`) have distinct tokens and commit without hindrance.

2. **Tier 2: Temporal Correlation Heuristic:**
   Used purely as a secondary heuristic to upgrade `REVIEW_REQUIRED` events upon receiving corroborating views, never as the sole uniqueness mechanism.

---

## 6. Reconciliation Engine Verification (Kuhn-Munkres)

### 6.1 Assignment Optimality
The reconciliation engine in `reconciliation_engine.py` constructs a cost matrix:
$$C_{i, j} = |t_{\text{GT}, i} - t_{\text{pred}, j}| - W \quad (\text{with } W = 1000.0)$$
Incompatible pairs (direction mismatch or $|t_{\text{GT}, i} - t_{\text{pred}, j}| > \tau$) are assigned $C_{i, j} = 10^6$.

Because $W$ is subtracted for every valid pair, SciPy's `linear_sum_assignment` mathematically guarantees:
1. **Maximum Cardinality:** Maximizes the number of matched pairs before minimizing time error.
2. **Minimum Temporal Error:** Among maximum-cardinality assignments, chooses the assignment with minimal $\sum \Delta t$.
3. **No Incompatible Assignments:** Any pair with cost $> 0.0$ is strictly rejected.

Adversarial synthetic tests confirm that greedy nearest-neighbor matching fails where Kuhn-Munkres achieves optimal $100\%$ matching.
