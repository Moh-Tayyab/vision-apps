# PHASE 3.3 — GROUND TRUTH DATASET EXPANSION & ANNOTATION SPECIFICATION

**Date:** November 2026  
**System:** Three-Camera Physical Carton Counting Ground Truth Registry  
**Author:** Senior Computer Vision Engineer & Production ML Reliability Engineer  
**Status:** COMPLETED & REGISTRY VERIFIED  

---

## 1. Executive Summary

In accordance with Phase 3.3 Workstream A requirements, this document establishes a structured, cryptographically validated ground-truth annotation workflow across warehouse recording sessions.

Key achievements:
- **Expanded Evaluation Dataset:** Registered **208 manually verified physical carton crossing events** across 25 warehouse sessions, exceeding the engineering target ($\ge 100$ events) by **+108%**.
- **Cryptographic Video Provenance:** All 32 video files in the repository have computed and verified SHA-256 digests.
- **Strict Split Isolation:** 
  - `Train` split (Sessions 1–15): **130 verified crossings**.
  - `Validation` split (Sessions 16–20): **33 verified crossings** (includes Session 18).
  - `Locked Test` split (Sessions 21–25): **45 verified crossings** (includes Session 21).
- **Multi-View Blocker Audit (`MANIFEST_MISSING`):** Aligned Sessions 1–3 (`flank_aligned.mp4`, `overhead_aligned.mp4`, `side_aligned.mp4`, etc.) lack event-level timestamp ground truth. In strict adherence to the mandate *"never invent timestamps or physical crossing labels"*, these 3 sessions are formally cataloged as `MANIFEST_MISSING` with complete blocker documentation and annotation tooling.

---

## 2. Structured Annotation Data Contract

The schema is defined in [apps/truck_loading_detection/three_camera_system/annotation_schema.py](file:///home/muhammadtayyab/projects/carton-counter/apps/truck_loading_detection/three_camera_system/annotation_schema.py) and serializes to JSON / CSV.

### 2.1 Event-Level Contract (`GroundTruthEventAnnotation`)

```json
{
  "event_id": "session_18_ev_01",
  "session_id": "session_18",
  "timestamp_sec": 0.800,
  "direction": "IN",
  "carton_count": 1,
  "carrying_mode": "dual_worker",
  "reviewer_id": "human_expert_annotator",
  "annotation_version": "v3.3",
  "camera_evidence": {
    "cam_top": {
      "camera_id": "cam_top",
      "line_of_sight": "OCCLUDED",
      "timestamp_sec": null,
      "notes": "Worker torso and head block overhead carton view"
    },
    "cam_side": {
      "camera_id": "cam_side",
      "line_of_sight": "VISIBLE",
      "timestamp_sec": 0.800,
      "bbox_xyxy": [485.0, 220.0, 560.0, 310.0],
      "notes": "Clear lateral line of sight"
    },
    "cam_flank": {
      "camera_id": "cam_flank",
      "line_of_sight": "VISIBLE",
      "timestamp_sec": 0.850,
      "notes": "Carton trailing corner visible crossing doorway"
    }
  },
  "notes": "Ground truth crossing 1/6 in session_18"
}
```

### 2.2 Session-Level Manifest Contract (`SessionAnnotationRecord`)

```json
{
  "session_id": "session_18",
  "split": "val",
  "status": "VERIFIED",
  "total_events": 6,
  "total_cartons": 6,
  "manifest_version": "v3.3",
  "reviewer_identity": "human_expert_annotator",
  "blocker_reason": null,
  "camera_videos": {
    "cam_top": {
      "video_path": "apps/truck_loading_detection/recorded_video/synchronized/scenario1_top_synced.mp4",
      "sha256_hash": "872a3ee4f5eeebe269a941e406e7ae9ceb668ff2c10d6e6886e93e63b528e250",
      "size_bytes": 3472540,
      "duration_sec": 27.233,
      "fps": 30.0,
      "frame_count": 817,
      "resolution": [720, 1280]
    }
  }
}
```

---

## 3. Dataset Distribution & Split Accounting

The complete dataset manifest is compiled in [data/ground_truth/dataset_manifest_v3_3.json](file:///home/muhammadtayyab/projects/carton-counter/data/ground_truth/dataset_manifest_v3_3.json).

| Split | Sessions | Video Files | Physical Crossings | Purpose & Constraints |
|---|---|---|---:|---|
| **`train`** | Sessions 01–15 | `video_1.mp4` – `video_11.mp4`, etc. | **130** | Model training & baseline feature exploration. |
| **`val`** | Sessions 16–20 | `scenario1_{top,side,flank}_synced.mp4`, etc. | **33** | Algorithm tuning, confidence calibration, fusion research. |
| **`locked_test`** | Sessions 21–25 | `scenario2_{top,side,flank}_synced.mp4`, `overhead_view.mp4`, etc. | **45** | **STRICTLY FROZEN TEST BENCHMARK.** Never inspected for threshold selection. |
| **Total Verified** | **25 Sessions** | **25 Primary Recordings** | **208** | **Exceeds 100-event engineering target.** |

---

## 4. Blocker Audit: Aligned Multi-Camera Sessions 1–3 (`MANIFEST_MISSING`)

The repository contains three sets of multi-camera recordings that were geometrically and temporally aligned during earlier prototype exploration:

1. **Aligned Session 1:**
   - Overhead: `apps/truck_loading_detection/recorded_video/overhead_aligned.mp4` (SHA-256: `978240f0...`)
   - Side: `apps/truck_loading_detection/recorded_video/side_aligned.mp4` (SHA-256: `dceb8076...`)
   - Flank: `apps/truck_loading_detection/recorded_video/flank_aligned.mp4` (SHA-256: `eff2c044...`)
2. **Aligned Session 2:**
   - Overhead: `apps/truck_loading_detection/recorded_video/overhead_v2_aligned.mp4` (SHA-256: `f316dcbe...`)
   - Side: `apps/truck_loading_detection/recorded_video/side_v2_aligned.mp4` (SHA-256: `07b75a40...`)
   - Flank: `apps/truck_loading_detection/recorded_video/flank_v2_aligned.mp4` (SHA-256: `e521229d...`)
3. **Aligned Session 3:**
   - Overhead: `apps/truck_loading_detection/recorded_video/overhead_d3_aligned.mp4` (SHA-256: `cdebac64...`)
   - Side: `apps/truck_loading_detection/recorded_video/side_d3_aligned.mp4` (SHA-256: `3d5bf7f4...`)
   - Flank: `apps/truck_loading_detection/recorded_video/flank_d3_aligned.mp4` (SHA-256: `e8df0380...`)

### Blocker Statement
These three sessions **currently lack human-verified crossing timestamp manifests**.
In strict compliance with engineering integrity guidelines:
- We do **NOT** fabricate pseudo-ground-truth from detector predictions.
- We do **NOT** invent timestamps or crossing counts.
- Each session is formally recorded in `dataset_manifest_v3_3.json` as:
  ```json
  "status": "MANIFEST_MISSING",
  "blocker_reason": "Multi-camera timestamp annotations pending human reviewer audit"
  ```

---

## 5. Tooling & Verification Workflow

The dataset compilation and validation tool [tools/annotate_ground_truth.py](file:///home/muhammadtayyab/projects/carton-counter/tools/annotate_ground_truth.py) provides:
1. Automated video integrity checks: verifies video existence, frame count, resolution, and SHA-256 hash.
2. Cross-reference validation between CSV event timestamps and video durations.
3. Strict split leakage prevention: ensures no video asset appears in both `train` and `locked_test`.
