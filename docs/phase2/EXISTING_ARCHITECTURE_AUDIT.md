# Phase 2 — Existing Architecture Audit & End-to-End Pipeline Trace

## 1. Executive Summary & Objective

This audit presents an exhaustive, code-level analysis of the `carton-counter` multi-camera truck-loading detection system. The primary invariant of the system is:

> **One physical carton crossing the doorway threshold must produce exactly one confirmed logical carton count (+1 IN / -1 OUT), regardless of whether it is visible in one, two, or three synchronized camera views.**

This document traces the complete execution path from raw frame acquisition through neural inference, mask refinement, worker subtraction, multi-camera association, consensus arbitration, state machine progression, persistence, and REST APIs.

---

## 2. End-to-End Component Map

| Stage | Responsible Python File | Key Classes & Functions | Configuration / Models |
|---|---|---|---|
| **1. Model & Inference** | `apps/truck_loading_detection/three_camera_system/detector_adapter.py` | `ThreeCameraCartonDetectorAdapter.detect_camera_frame()` | `apps/truck_loading_detection/best.pt` (YOLO11n-seg), `yolo11n-seg.pt` (Worker) |
| **2. Mask Refinement & Worker Subtraction** | `apps/truck_loading_detection/model_pipeline/mask_refinement.py` | `CartonMaskRefiner.refine_mask()`, `subtract_worker_mask()` | Morphological kernels, edge snappers, Otsu boundary thresholding |
| **3. Per-Camera Tracking** | `apps/truck_loading_detection/three_camera_system/per_camera_tracker.py` | `PerCameraTracker.update()`, ByteTrack/BoT-SORT integration | `tracker_type="bytetrack"`, `track_thresh=0.45`, `match_thresh=0.70` |
| **4. Multi-Camera Synchronization** | `apps/truck_loading_detection/three_camera_system/timeline_aligner.py` | `TimelineAligner.compute_timestamp()`, `CameraTimestamp` | Frame PTS, per-camera FPS config, drift window $\pm 33\text{ ms}$ |
| **5. Individual Tripwire Detection** | `apps/truck_loading_detection/three_camera_system/corridor.py` | `PerCameraTripwire.update()`, `LocalCameraCrossingCandidate` | `CameraTripwireConfig` (`orientation`, `line_pos_frac`, `direction_sign`) |
| **6. Cross-Camera Association** | `apps/truck_loading_detection/three_camera_system/cross_camera_associator.py` & `pipeline.py` | `CrossCameraAssociator.cluster_detections()`, `CrossingWindow` | `coalesce_window_sec=6.0s`, `time_window_sec=0.75s`, spatial projection |
| **7. Stack Disambiguation** | `apps/truck_loading_detection/three_camera_system/stack_disambiguator.py` | `StackDisambiguator.resolve_stack()`, `StackDisambiguationResult` | Aspect ratio, contour concavity, height-to-width heuristics |
| **8. Event Consensus** | `apps/truck_loading_detection/three_camera_system/consensus_policy.py` | `ThreeCameraConsensusPolicy.evaluate_crossing()`, `ConsensusVerdict` | Unanimous (3), Majority (2), Single-Camera (Hold for review) |
| **9. Crossing State Machine** | `apps/truck_loading_detection/three_camera_system/crossing_state_machine.py` | `CartonCrossingStateMachine`, `PhysicalCarton`, `CommittedCrossingEvent` | States: `NOT_SEEN`, `APPROACHING`, `CROSSING`, `CROSSED`, `COUNT_COMMITTED` |
| **10. Volumetric Corridor Engine** | `apps/truck_loading_detection/three_camera_system/corridor.py` | `CorridorStateMachine.update_carton()` | Multi-camera boundary passage, hysteresis buffer zones |
| **11. Event Persistence** | `apps/truck_loading_detection/three_camera_system/pipeline.py` | `ThreeCameraCartonPipeline._audit_events` (In-memory, upgrading to SQLite WAL) | JSON audit export, clip triggering via `RingBuffer` |
| **12. API & Visual Interface** | `apps/truck_loading_detection/app.py` | Flask endpoints: `/api/3angle/process`, `/api/3angle/job/<job_id>`, `/health` | Visualizer overlay, WebSocket/SSE progress updates |

---

## 3. Detailed Dataflow Walkthrough

### 3.1 Frame Acquisition & Canonical Detection
- Frames arrive from three synchronized camera sources: `cam_top` (overhead view looking down at doorway), `cam_side` (lateral view showing worker carrying height), and `cam_flank` (oblique view showing truck entrance interior).
- Inference is orchestrated by `ThreeCameraCartonDetectorAdapter`:
  - **Carton Segmentation Model:** `apps/truck_loading_detection/best.pt` (SHA-256: `98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33`). Class contract is strictly `{0: 'carton'}`.
  - **Worker Segmentation Model:** `yolo11n-seg.pt` (SHA-256: `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`), running class 0 (`person`).

### 3.2 Mask Refinement & Worker Mask Subtraction
- `CartonMaskRefiner.refine_mask()` receives raw polygon masks from YOLO.
- When workers carry cartons, worker limbs/clothing frequently overlap carton bounding boxes.
- `subtract_worker_mask()` computes morphological difference: $M_{\text{carton\_clean}} = M_{\text{carton}} \setminus \text{Dilate}(M_{\text{worker}}, k=5)$.
- Otsu thresholding along edges removes phantom segmentation bleeding.

### 3.3 Per-Camera Tracking
- Each camera view runs an isolated `PerCameraTracker` instance.
- **Critical Architectural Invariant:** Tracker IDs (`track_id`) are strictly camera-local (`cam_top:12`, `cam_side:4`, etc.). Local tracker IDs are never used directly as physical carton IDs.
- Tracks maintain temporal histories (bounding boxes, centroids, velocities, Kalman states) over up to 60 frames.

### 3.4 Camera Synchronization
- `TimelineAligner` maps frame indices to presentation timestamps ($t_{\text{sec}}$) using camera frame rates (e.g. 30.0 FPS, 25.0 FPS, or 29.97 FPS).
- Camera frames are matched within a maximum synchronization tolerance window ($\pm 33\text{ ms}$).
- When a camera drops frames or lags, `TimelineAligner` flags frame gap warnings without halting adjacent camera pipelines.

### 3.5 Individual Tripwire Crossings
- Virtual tripwires are configured in normalized image coordinates via `CameraTripwireConfig`:
  - `cam_top`: vertical line (`line_pos_frac=0.50`), inward progression along $-X$ (`direction_sign=-1`).
  - `cam_side`: vertical line (`line_pos_frac=0.50`), inward progression along $+X$ (`direction_sign=+1`).
  - `cam_flank`: vertical line (`line_pos_frac=0.55`), inward progression along $+X$ (`direction_sign=+1`).
- Hysteresis margins ($\pm 15$ to $\pm 25\text{ px}$) prevent jitter oscillations when cartons hover near the threshold.
- When a track crosses both boundaries, a `LocalCameraCrossingCandidate` is emitted with timestamp, direction, bounding box, and displacement.

### 3.6 Cross-Camera Association & Physical Carton Identity
- `CrossCameraAssociator` aggregates local candidates across cameras into temporal-spatial clusters.
- Because a human worker takes $400\text{--}700\text{ ms}$ to traverse the physical doorway, camera crossing lines are crossed with non-zero transit delay.
- The system employs a coalescing window (`CrossingWindow`, duration up to $6.0\text{ s}$ with active transit tolerance of $0.75\text{--}1.5\text{ s}$).
- Tracklets across cameras are associated into a single `PhysicalCarton` entity (`phys_carton_00001`, etc.) based on:
  1. Direct tracklet association (matching previous sightings).
  2. Spatio-temporal continuity (Euclidean distance $\le 140\text{ px}$ or calibrated dock space coordinates).
  3. Directional consistency.

### 3.7 Consensus Arbitration
- `ThreeCameraConsensusPolicy` classifies the multi-camera evidence:
  - **THREE_CAMERA_UNANIMOUS:** All three cameras detected crossing with consistent direction. Action: `COMMIT` (Confirmed count $+1$).
  - **TWO_CAMERA_AGREEMENT:** Two cameras agree on direction. Action: `COMMIT` (Confirmed count $+1$).
  - **SINGLE_CAMERA_OBSERVATION:** Only one camera detected crossing. Action: `HOLD_FOR_REVIEW` (State: `REVIEW_REQUIRED`).
  - **CONFLICTING_OR_UNCERTAIN:** Cameras disagree on direction or movement is stationary. Action: `REJECT` or `HOLD_FOR_REVIEW`.

### 3.8 Stack Disambiguation
- When a worker carries two or three stacked cartons simultaneously, segmentation may output a single merged bounding box.
- `StackDisambiguator.resolve_stack()` inspects contour concavity and vertical aspect ratios:
  - If height-to-width ratio exceeds threshold and contour indentations indicate a double carton, resolved quantity is set to $2$.
  - Disambiguation metadata is stored with the event record.

### 3.9 State Machine Lifecycle & Confirmation
- `CartonCrossingStateMachine` enforces strict transitions:
  $$\text{NOT\_SEEN} \longrightarrow \text{APPROACHING} \longrightarrow \text{CROSSING} \longrightarrow \text{CROSSED} \longrightarrow \text{COUNT\_COMMITTED}$$
- **Duplicate Prevention Invariant:**
  - If a `PhysicalCarton` already has `is_committed = True` in direction `"IN"`, subsequent observations within the corridor transit window are absorbed as corroboration, updating `participating_cameras` without generating a second count event.
  - Directional reversal cancels crossing candidates (`REVERSED_CANCELLED`).

---

## 4. Current Persistence Architecture & Gaps Identified

### Current State
- `ThreeCameraCartonPipeline` currently stores events in `self._audit_events` (a standard in-memory Python list) and updates scalar integer counters (`self._confirmed_count`, `self._review_required_count`, `self._total_committed_count`).
- `CartonCrossingStateMachine` provides helper methods `export_state(file_path)` and `load_state(file_path)` for JSON serialization.

### Production Gaps Identified (To be resolved in Step 6)
1. **Lack of Durable ACID Storage:** In-memory lists are vulnerable to server crash, worker restarts, or uncaught exceptions.
2. **Missing Database-Level Unique Constraint:** In concurrent or multi-threaded pipeline executions, two concurrent worker threads could attempt to commit crossings for the same physical carton simultaneously, resulting in double counts.
3. **No Durable Audit Ledger for Review Actions:** When an operator reviews a `REVIEW_REQUIRED` event in the UI and manually confirms or rejects it, the change must be persisted transactionally to a durable database with operator ID and timestamp.
4. **Recovery on Startup:** On application startup, committed counts should be automatically re-hydrated from the durable database to prevent count resets or duplicate reprocessing.

---

## 5. Architectural Invariant Summary

1. **Global Carton Identity:** Generated deterministically via `phys_carton_{idx:05d}` and maintained through camera transitions.
2. **Exactly-Once Delivery:** Enforced at both the state machine level and database uniqueness constraint: `UNIQUE(session_id, global_carton_id, direction)`.
3. **Review Isolation:** Single-camera crossings or occluded events are quarantined as `REVIEW_REQUIRED` and excluded from `confirmed_count` until corroborated or manually approved.
4. **Zero-Latency In-Memory Routing + WAL Durability:** In-memory tracking maintains real-time pipeline throughput while SQLite WAL ensures atomic transaction commits without blocking video decoding.
