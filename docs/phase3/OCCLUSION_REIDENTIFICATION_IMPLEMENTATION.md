# Phase 3.2 — Occlusion-Aware Tracklet Re-Identification & Physical Carton Continuity Implementation Report

## Executive Summary

During Phase 3.1 evaluation, the three-camera carton counting pipeline exhibited severe physical carton identity fragmentation: across 14 ground-truth crossings, the system generated **16 duplicate candidate events** and **1 unresolved prediction**, yielding an automatic recall of only 28.6% and routing 9 physical crossings to human review.

Phase 3.2 investigates, engineers, and validates an occlusion-aware physical carton re-identification layer that maintains logical identity across temporary detection dropouts, ByteTrack ID changes, and multi-camera transits. This implementation preserves multi-camera consensus rules, maintains frozen YOLO weights (`best.pt`, SHA-256: `98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33`), and prevents the incorrect merging of distinct physical cartons.

---

## 1. Architectural Distinction: 4 Distinct Levels of Identity

A central root cause of tracklet fragmentation in earlier iterations was the conflation of camera-local tracklet IDs with physical carton identity. Phase 3.2 strictly enforces a 4-tier separation:

```
+-----------------------------------------------------------------------------+
| Tier 1: Camera Detection (CanonicalCartonDetection)                         |
| Raw bounding box, confidence, segmentation polygon per camera per frame.    |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Tier 2: Camera-Local Tracklet (CartonTracklet)                              |
| Scoped strictly to one camera perspective (e.g., 'cam_side_t0002').         |
| Managed by PerCameraCartonTracker with constant-velocity coasting.          |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Tier 3: Physical Carton Entity (PhysicalCarton)                             |
| Globally unique real-world carton (e.g., 'phys_carton_00004').              |
| Correlates across multiple camera tracklets, worker occlusions, and time.  |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Tier 4: Crossing Occurrence (CommittedCrossingEvent)                        |
| Specific tripwire threshold traversal instance (e.g., 'session_18_ev_00001').|
| Exactly-once transactional durability with support for repeat crossings.   |
+-----------------------------------------------------------------------------+
```

---

## 2. Root Cause Analysis & Justified Implementations

### A. CrossingWindow Spatial Incompatibility Flaw (`pipeline.py:608`)
- **Flaw:** The previous spatial compatibility logic contained the condition:
  ```python
  if o_ev.local_track_id != ev.local_track_id or dist > 65.0:
      is_spatially_compatible = False
  ```
  Whenever a worker's arm or body occluded a carton for $> 15$ frames, ByteTrack spawned a new local track ID (`cam_side_t0002` replacing `cam_side_t0001`). Because `local_track_id` differed, the pipeline rejected spatial compatibility, forcing the creation of an independent `CrossingWindow`, which subsequently closed and committed as a duplicate candidate.
- **Production Fix:** Modified spatial compatibility check to distinguish between normal drift of the same tracklet versus re-emerging fragmented tracklets in the same corridor lane:
  ```python
  if o_ev.camera_id == ev.camera_id:
      dist = math.hypot(ev_center[0] - o_center[0], ev_center[1] - o_center[1])
      y_diff = abs(ev_center[1] - o_center[1])
      if o_ev.local_track_id == ev.local_track_id:
          if dist > 65.0:
              is_spatially_compatible = False
              break
      else:
          # Different tracklet on same camera:
          # Coalesce fragmented tracklets across occlusion if in same lane (y_diff <= 60px)
          # and within corridor drift (dist <= 140px).
          # Distinct cartons if vertical lane offset > 60px or dist > 140px.
          if y_diff > 60.0 or dist > 140.0:
              is_spatially_compatible = False
              break
  ```

### B. Spatio-Temporal Physical Carton Re-Identification (`crossing_state_machine.py`)
- **Implementation:** Enhanced `find_matching_physical_carton` with formal `PhysicalAssociationOutcome` enum:
  - `MATCHED_EXISTING_PHYSICAL_CARTON`: Exact local track ID match or valid corridor transit continuation.
  - `TEMPORARY_OCCLUSION`: Spatio-temporal re-id across detection loss ($\Delta t \le 2.0\text{s}$, $|y_1 - y_2| \le 60\text{px}$, $dist \le 140\text{px}$).
  - `AMBIGUOUS_ASSOCIATION`: Multiple equidistant physical candidates ($|\Delta dist| < 15\text{px}$), flagged for review.
  - `NEW_PHYSICAL_CARTON`: Genuinely distinct physical carton.
- **Parallel Worker Safety:** Enforces strict lateral lane separation ($\Delta y > 60\text{px}$ in camera frame, $\Delta y > 120\text{px}$ across views) and temporal bounds ($\Delta t > 2.0\text{s}$) to guarantee that cartons carried simultaneously by different workers are never merged.

### C. Tracker Occlusion Search Expansion (`per_camera_tracker.py`)
- **Implementation:** During temporary occlusion (`trk.status == TrackStatus.OCCLUDED`), the maximum association radius is expanded from 120px to 160px:
  ```python
  effective_dist = max(self.max_dist, 160.0) if trk.status == TrackStatus.OCCLUDED else self.max_dist
  if iou >= self.iou_threshold or dist <= effective_dist:
      score = iou + max(0.0, 1.0 - (dist / effective_dist))
      candidates.append((score, tid, did))
  ```
  This allows ByteTrack to recover cartons that undergo substantial conveyor/worker transit during temporary hand/body occlusion without dropping track continuity.

### D. Worker Mask Boundary Leakage Elimination (`mask_refinement.py`)
- **Diagnosis of `ev_00018`:** In Session 18, an empty-handed worker walked backward out of the doorway. Folds on the worker's shirt triggered a low-confidence false detection (confidence 0.1697) on `cam_flank`. The previous mask refinement logic preserved heavily occluded cartons ($\ge 70\%$ worker overlap) without checking whether any independent carton body remained outside the worker.
- **Production Fix:** When worker overlap exceeds 70%, the refiner checks the independent carton area outside the worker's segmented body. If confidence is low ($< 0.25$) and the protruding area is negligible ($< 200\text{px}$ or $< 10\%$ of mask), it is rejected as a worker clothing phantom:
  ```python
  elif worker_overlap_ratio >= 0.70:
      is_occluded = True
      if confidence < 0.25:
          independent_carton = cv2.subtract(carton_bin, worker_bin)
          independent_area = float(cv2.countNonZero(independent_carton))
          if independent_area < max(200.0, 0.10 * initial_area):
              return None
  ```
- **Consensus Guard:** `ThreeCameraConsensusPolicy` enforces `min_single_camera_conf = max(0.25, conf_threshold)` to suppress isolated single-camera noise as `DecisionAction.SUPPRESS_AS_NOISE`.

### E. Tripwire Directional Retreat Guard (`corridor.py`)
- **Implementation:** Added trajectory coordinate delta check to `PerCameraTripwire.update`. If an object's trajectory moves against the required loading direction ($\Delta x < -10\text{px}$ for positive direction), occlusion stitching is aborted, ensuring backward-stepping workers cannot trigger an `IN` crossing.

---

## 3. Verified Regression Test Coverage

The implementation was validated against all 13 required regression scenarios in `tests/test_phase3_2_reid_and_continuity.py`:

| Test Name | Scenario Verified | Result |
|---|---|:---:|
| `test_scenario_01_occlusion_and_tracklet_fragmentation` | Preserves physical carton across ByteTrack ID changes | **PASSED** |
| `test_scenario_02_same_physical_carton_across_camera_tracklets` | Unifies 3 camera tracklets to 1 physical carton | **PASSED** |
| `test_scenario_03_two_simultaneous_cartons_parallel_lanes` | Preserves separation of cartons in parallel lanes ($\Delta y > 60\text{px}$) | **PASSED** |
| `test_scenario_04_parallel_workers_carrying_cartons` | Strict isolation of parallel workers ($\Delta y = 160\text{px}$) | **PASSED** |
| `test_scenario_05_same_lane_with_temporal_overlap` | Consecutive cartons outside transit window ($\Delta t = 2.5\text{s}$) remain distinct | **PASSED** |
| `test_scenario_06_similar_looking_separate_cartons` | Spatially separated cartons ($dist = 250\text{px}$) do not merge | **PASSED** |
| `test_scenario_07_worker_mask_overlap_preservation` | Disentangles worker hands (35% overlap) while preserving carton body | **PASSED** |
| `test_scenario_08_empty_hand_phantom_detection_suppressed` | Suppresses `ev_00018` empty-hand shirt fold phantom | **PASSED** |
| `test_scenario_09_startup_during_carton_crossing` | Recovers `session_21_ev_01` startup in-flight crossing | **PASSED** |
| `test_scenario_10_track_recovery_after_camera_frame_loss` | Coasts and recovers track ID across 8 frames of occlusion | **PASSED** |
| `test_scenario_11_direction_reversal_cancellation` | Cancels count when worker retreats backward | **PASSED** |
| `test_scenario_12_legitimate_repeated_crossing` | Preserves legitimate repeat crossings (t=2s IN, t=30s OUT, t=60s IN) | **PASSED** |
| `test_scenario_13_exactly_once_persistence_after_recovery` | Reconstitutes physical carton state from SQLite EventStore | **PASSED** |

Full regression suite: **72 passed in 13.27s** (0 failures, 0 regressions across all baseline tests).
