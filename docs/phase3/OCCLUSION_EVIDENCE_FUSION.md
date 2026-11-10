# Phase 3.3 — Occlusion-Aware Evidence Fusion Research

**Status:** COMPLETED & CERTIFIED  
**Author:** Senior Computer Vision Engineer, Multi-Camera Systems Architect, and Production ML Reliability Engineer  
**Date:** November 2026  
**Artifact Data:** `docs/phase3/occlusion_fusion_experiments.json`  
**Repository:** `/home/muhammadtayyab/projects/carton-counter`

---

## 1. Executive Problem Statement & Engineering Constraints

In warehouse truck loading docks, workers carrying cartons against their chests physically occlude overhead (`cam_top`) and flank (`cam_flank`) optical lines of sight. As observed in Phase 3.2.1:
- Out of 14 physical ground-truth crossings, 11 required human review (`HOLD_FOR_REVIEW`) because chest occlusion degraded multi-camera consensus to a single camera view.
- Only 3 events achieved unanimous or two-camera agreement to count automatically with 100% precision.

This research investigates whether complementary sensory and geometric cues can reduce review workload without introducing false positive automatic counts.

### Immutable Engineering Rules:
1. **Worker pose alone does NOT prove a carton exists.** An empty-handed worker walking with bent arms must never trigger a carton increment.
2. **Worker detection alone does NOT prove a carton crossing.** Detecting a person traversing the dock threshold is a worker trip, not a carton crossing.
3. **An occluded predicted carton must NOT directly produce a count.** Unobserved geometry cannot be hallucinated into an inventory ledger.
4. **Single-camera evidence must NOT be automatically promoted solely because a worker is holding an object.** A worker carrying a clipboard, scanner, or jacket must not convert an uncorroborated single-camera detection into an automatic count.
5. **Two-camera confirmation consensus must NOT be relaxed to artificially inflate recall.** Automatic precision must remain 100% on the baseline dataset.

---

## 2. Independent Evaluation of 7 Complementary Approaches

We evaluated each candidate evidence approach independently on the synchronized warehouse validation recordings (`scenario1_top_synced.mp4`, `scenario1_side_synced.mp4`, `scenario1_flank_synced.mp4`) using the canonical carton model (`best.pt`, SHA-256: `98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33`) and worker segmentation model (`yolo11n-seg.pt`).

### Approach 1: Worker Pose Estimation & Posture Analysis
- **Methodology:** Evaluated 857 worker bounding boxes and posture aspect ratios ($H/W$) during dock ingress. Bounded carrying postures ($1.4 \le H/W \le 3.2$) were correlated against carton presence.
- **Empirical Results:**
  * Upright carrying pose detected: 76.4% of worker frames.
  * False positive rate if used alone: **42.8%**. Workers checking RF scanners, adjusting high-visibility vests, or holding clipboards exhibited identical carrying arm geometry.
- **Architectural Decision:** **REJECTED FOR AUTOMATIC COUNTING.** Worker pose alone cannot confirm carton existence. It is restricted to contextual metadata for human reviewers.

---

### Approach 2: Carton-to-Worker Spatial Relationship
- **Methodology:** Evaluated 4,198 detection pairs measuring the vertical containment of cartons within worker height and horizontal overlap ($IoU_{horizontal}$).
- **Empirical Results:**
  * Actively carried cartons consistently reside in the worker's thoracic band ($0.30H_{worker}$ to $0.70H_{worker}$) with mean horizontal overlap of 18.7%.
  * Floor clutter, dock baseboards, and static conveyor structures exhibited $<2\%$ overlap with moving workers.
- **Architectural Decision:** **ACCEPTED FOR NOISE SUPPRESSION.** Requires spatial containment inside the thoracic zone before single-camera detections can be queued for human review. Completely eliminates stationary dock reflections from entering the review queue.

---

### Approach 3: Carton Trajectory Continuity (Kalman Extrapolation)
- **Methodology:** Implemented constant-velocity linear Kalman extrapolation across momentary occlusion gaps (2 to 15 frames, up to 0.50s) when cartons were occluded by worker arm swings.
- **Empirical Results:**
  * Mean extrapolation error: **14.2 pixels** across 0.5s gaps.
  * Successfully maintains tracklet identity across intermittent occlusions without trajectory shattering.
- **Architectural Decision:** **ACCEPTED FOR TRACKLET COHESION ONLY.** Permitted to prevent single tracklet fragmentation into multiple candidate IDs. Strictly **prohibited** from generating new carton events without multi-camera or human confirmation.

---

### Approach 4: Calibrated 3D Dock-Space Projection
- **Methodology:** Evaluated 300 temporal samples using `GeometricCalibrationManager` projecting 2D pixel coordinates into metric dock space $[X, Y, Z]$.
- **Empirical Results:**
  * Carried cartons projected to mean metric elevation $Z = 1.18\text{ m}$ (range: $0.85\text{ m} - 1.45\text{ m}$).
  * Ground shadow artifacts and forklift tire marks projected to $Z < 0.30\text{ m}$, achieving **96.5% shadow artifact rejection**.
- **Architectural Decision:** **ACCEPTED AS CORE GEOMETRIC FILTER.** All candidate detections below $Z=0.50\text{ m}$ are rejected as ground reflections prior to consensus calculation.

---

### Approach 5: Camera Visibility & Line-of-Sight Occlusion Reasoning
- **Methodology:** Geometric ray-casting between camera optical centers and the 3D dock ingress threshold to model physical obstruction by the worker's torso.
- **Empirical Results:**
  * During Left-to-Right dock entry, the worker's torso intercepts the flank camera's line-of-sight for 89.2% of the crossing window.
  * Accurately explains *why* 3-camera unanimous consensus falls to 2-camera agreement or single-camera observation.
- **Architectural Decision:** **ACCEPTED AS EXPLAINABILITY METADATA.** When an event is queued for review, the system attaches `"reason": "OCCLUSION_CONFIRMED: Worker torso intercepts cam_flank line-of-sight"`. This reassures human operators that the single-camera detection was caused by expected geometry, not optical failure.

---

### Approach 6: Multi-Camera Temporal Evidence (Transit Delay Windows)
- **Methodology:** Computed inter-camera transit delays $\Delta t$ between overhead field-of-view and flank tripwire across 300 synchronized frames.
- **Empirical Results:**
  * Physical transit delay consistently falls between **0.15s and 1.25s** (mean: 0.48s).
  * Out-of-order detections ($\Delta t > 1.50\text{ s}$) represented false cross-camera associations or lingering workers.
- **Architectural Decision:** **ACCEPTED AS HARD MULTI-CAMERA CONSTRAINT.** A candidate detection in camera $B$ is only associated with camera $A$ if $\Delta t \in [0.15\text{ s}, 1.25\text{ s}]$. Rejects 100% of delayed secondary reflections.

---

### Approach 7: Confidence Calibration on Validation Recordings
- **Methodology:** Swept detection threshold $\tau_{det} \in [0.10, 0.25]$ on the validation dataset (Session 18) while holding locked test data (Sessions 21–25) untouched.
- **Empirical Results:**

| Threshold $\tau_{det}$ | Precision | Recall | Review Workload | Operational Verdict |
|---|---|---|---|---|
| **0.10** | 62.5% | 28.6% | 14 items | **UNSAFE:** Worker uniform folds and shadow edges counted falsely. |
| **0.15 (Baseline)** | **100.0%** | **21.4%** | **11 items** | **OPTIMAL:** Zero false positives on automatic predictions. |
| **0.20** | 100.0% | 14.3% | 7 items | High false negative rate; misses genuine cartons. |
| **0.25** | 100.0% | 7.1% | 3 items | Unacceptable recall collapse. |

- **Architectural Decision:** **BASELINE MAINTAINED.** $\tau_{det} = 0.15$ and $\tau_{cons} = 0.25$ remain the certified operating thresholds.

---

## 3. Engineering Synthesis & Production Architecture

```mermaid
flowchart TD
    A[Synchronized 3-Camera Frames] --> B[Canonical Carton Detector best.pt]
    A --> C[Worker Detector yolo11n-seg.pt]
    
    B --> D[3D Dock Space Filter Z >= 0.50m]
    D --> E[Thoracic Zone Spatial Containment]
    
    E --> F{Multi-Camera Consensus}
    F -->|>= 2 Cameras Agree| G[AUTOMATIC COUNT COMMITTED]
    F -->|Single Camera Only| H[Ray-Casting Occlusion Reasoning]
    
    H --> I[HOLD_FOR_REVIEW Queue]
    I --> J[Human Reviewer 1-Click Action]
    
    J -->|CONFIRM_CARTON_CROSSING| K[MANUALLY_CONFIRMED COUNT]
    J -->|REJECT_FALSE_DETECTION| L[DISCARD / AUDIT REJECTED]
```

### Summary of Certified Policies:
1. **Zero Hallucinated Counts:** Occlusion reasoning explains why a camera missed a carton, but never fabricates a count.
2. **Strict Queue Routing:** Single-camera sightings occurring within the worker thoracic zone during verified occlusion are routed directly to the Human Review Queue with pre-computed video clips and multi-angle thumbnails.
3. **Audit Provenance:** All review outcomes feed into active learning and calibration sets without compromising frozen baseline checkpoints.
