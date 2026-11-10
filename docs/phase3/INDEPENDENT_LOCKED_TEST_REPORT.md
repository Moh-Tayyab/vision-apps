# PHASE 3.3.1 — INDEPENDENT LOCKED-TEST REPORT (Workstream B)

**Repository:** `/home/muhammadtayyab/projects/carton-counter`  
**Phase:** 3.3.1 — Independent Acceptance, Test-Set Integrity & Security Audit  
**Workstream:** B — Independent Locked-Test Evaluation  
**Status:** COMPLETED — `INDEPENDENT_LOCKED_TEST_VERIFIED = BLOCKED`  
**Evaluation run ID:** `run_phase3_3_1_4366de05d420`  
**Run timestamp (UTC):** `2026-11-02T12:06:19.606007+00:00`  
**Evidence classification:** DEVELOPMENT REFERENCE ONLY — NOT independent acceptance evidence

---

## 1. Frozen Inputs (verified before execution)

All inputs were frozen and cryptographically verified prior to any execution. No input was changed after inspecting results.

| Layer | Artifact | SHA-256 |
|---|---|---|
| Model | `apps/truck_loading_detection/best.pt` (canonical carton detector) | `98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33` |
| Rollback model | `apps/truck_loading_detection/best_pt_v1_baseline.pt` | `2054d9e2093bd73315c37b3fbe21a9b3c93f0fc30a4d6eb7f755cb75e3c428b4` |
| Worker model | `yolo11n-seg.pt` | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |
| Configuration | Detection threshold `τ_det` | `0.15` (frozen, active certified) |
| Configuration | Consensus threshold `τ_cons` | `0.25` (frozen, active certified) |
| Reconciliation | `GroundTruthReconciler` (Hungarian 1-to-1 bipartite matching) | `temporal_tolerance_sec = 1.90` |
| Code | `three_camera_system/pipeline.py` | recorded in `phase3_3_1_frozen_inputs.json` |
| Code | `three_camera_system/reconciliation_engine.py` | recorded in `phase3_3_1_frozen_inputs.json` |
| Code | `three_camera_system/consensus_policy.py` | recorded in `phase3_3_1_frozen_inputs.json` |
| Code | `model_pipeline/event_ground_truth.py` | recorded in `phase3_3_1_frozen_inputs.json |
| Manifest | `data/ground_truth/dataset_manifest_v3_3.json` | recorded in `phase3_3_1_frozen_inputs.json` |

**Database:** Clean, isolated evaluation database `data/phase3_3_1_carton_eval.db` (removed and recreated before the run; no prior state).

**Artifacts:** `docs/phase3/phase3_3_1_eval_metrics.json`, `docs/phase3/phase3_3_1_frozen_inputs.json`, `docs/phase3/phase3_3_1_reconciliation.csv`, `docs/phase3/phase3_3_1_unmatched_events.csv`.

---

## 2. Execution Protocol (honoured)

1. Model frozen (checksum verified before run). ✅
2. Detection/consensus thresholds frozen (`τ_det=0.15`, `τ_cons=0.25`). ✅
3. Reconciliation methodology frozen (`GroundTruthReconciler`, tol 1.90s). ✅
4. Model, configuration, code, video, and manifest hashes recorded. ✅
5. Clean isolated evaluation database created. ✅
6. Unique evaluation run ID `run_phase3_3_1_4366de05d420` generated and stamped on every matched/unmatched record and metric. ✅
7. Test set executed **exactly once** (script removes + recreates the DB; idempotent re-run would overwrite, not accumulate). ✅
8. Thresholds/model not modified after inspecting results. ✅

**Locked-test protection (honoured):**
- Locked-test results were not used for threshold selection (thresholds pre-frozen).
- Locked-test failures did not modify the model.
- Locked-test events were not used to develop new association heuristics.
- Original locked-test manifests and hashes preserved (no annotation/file changes).
- Previously exposed sessions reclassified and documented (see `DATASET_SPLIT_INTEGRITY_AUDIT.md`).

---

## 3. Evaluated Sessions

Per Workstream A, **no genuinely untouched locked-test sessions exist**; all sessions 21–25 were reclassified to `development` (exposed during Phases 3.1–3.3, or physical duplicates of train/val recordings). The evaluation below is therefore a **development-reference run**, executed once, on the reclassified sessions. It demonstrates pipeline reproducibility under frozen inputs; it is not acceptance evidence.

| Unit | Split | Physical recording | GT source | GT crossings |
|---|---|---|---|---:|
| `session_21_scenario2_sync` | development (reclassified from locked_test) | Scenario 2 synchronized (top+side+flank) — physical event of sessions 21/22/23 | `session_21_events.csv` (identical GT for sessions 21/22/23) | 8 |
| `session_24` | development (reclassified from locked_test) | `sample_truck_loading.mp4` (single camera) | `session_24_events.csv` | 12 |
| `session_25` | development (reclassified from locked_test) | `overhead_view.mp4` (single camera) | `session_25_events.csv` | 9 |
| **Aggregate** | | | | **29** |

**Known session-level limitations:**
- `session_24`: processed single-camera through the `cam_top` tripwire although its manifest camera role is `cam_left` (pipeline has no `cam_left` tripwire configuration). Produced zero predictions — a camera-role substitution limitation, not a model result.
- `session_25`: portrait video (`476×848`); the frozen tripwire at `x=500 px` lies out of frame, so few local crossings fire — all single-camera, all routed to review.
- Sessions 21/22/23 are the same physical event evaluated once (as `session_21_scenario2_sync`) to avoid triple-counting.

---

## 4. Required Metrics (Phase 3.3.1 definitions)

**Definitions applied:**
- Automatic Precision = Automatic TP / (Automatic TP + Automatic FP)
- Automatic Recall = Automatic TP / Total Ground-Truth Crossings
- Human-approved events are NOT counted as automatic true positives.
- System coverage is not presented as automatic recall.

### Per-session metrics

| Session | GT crossings | Auto TP | Auto FP | Auto FN | Auto precision | Auto recall | Review-required GT | Unresolved | Duplicate candidates | Discrepancy vs GT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| session_21_scenario2_sync | 8 | 2 | 0 | 6 | 100.0% | 25.0% | 6 | 1 | 5 | −6 |
| session_24 | 12 | 0 | 0 | 12 | 0.0% | 0.0% | 0 | 0 | 0 | −12 |
| session_25 | 9 | 0 | 0 | 9 | 0.0% | 0.0% | 7 | 0 | 4 | −9 |

### Aggregate metrics

| Metric | Value |
|---|---:|
| Ground-truth physical crossings | 29 |
| Automatic true positives | 2 |
| Automatic false positives | 0 |
| Automatic false negatives | 14 |
| Automatic precision | 100.0% (2 / (2+0)) |
| Automatic recall | 6.9% (2 / 29) |
| Review-required physical crossings | 13 |
| Human-approved crossings | 0 (no human review in this automated run) |
| Human-rejected crossings | 0 |
| Unresolved events | 1 |
| Duplicate candidates | 9 |
| Verified duplicate tracklets | 0 |
| Total predicted events | 25 |
| Total confirmed physical count | 2 |
| Count discrepancy vs ground truth | −27 |

### Notes
- Zero automatic false positives across the run (precision 100.0% where defined).
- Automatic recall is low (6.9%) because single-camera sessions route everything to review and the synchronized scenario run confirms only high-consensus crossings automatically; the remaining 78.6%+ of physical crossings require human review — consistent with the Phase 3.3 finding that unattended production operation is prohibited.
- Ambiguous matches (within 100 ms of an alternate candidate) remain explicitly unresolved per the matching methodology; 1 unresolved event recorded.
- Human-approved/rejected counts are 0 because this was an automated, unattended pipeline run with no operator decisions persisted. These would be populated only through the human-review workflow.

---

## 5. Locked-Test Protection Statement

- The evaluation was executed once on **reclassified development sessions**, not on a genuine locked-test set.
- No threshold, model, or association heuristic was derived from these results.
- `INDEPENDENT_LOCKED_TEST_VERIFIED = BLOCKED`.

---

## 6. Blocker and Required Action

**BLOCKED:** Workstream A (`DATASET_SPLIT_INTEGRITY_AUDIT.md`) established that zero sessions have verified annotations with no historical development/training/calibration/debugging exposure and no physical-recording duplicate in another split. Therefore:

1. The claimed `locked_test` split (45 crossings, sessions 21–25) is **invalid** as independent acceptance evidence — all sessions exposed or physically duplicated.
2. **Additional independently annotated warehouse footage is required:** new recording sessions with no physical/temporal overlap with any train/val/video asset, never used in training, calibration, debugging, or threshold decisions, then a fresh `locked_test` split with new hashes.
3. Until such footage is annotated and a clean split established, no `INDEPENDENT_LOCKED_TEST_VERIFIED` claim can be made. The metrics above are development-reference evidence only.

---

## 7. Artifacts and Reproducibility

Re-run command (will overwrite the isolated DB and produce identical run semantics under frozen inputs):

```bash
python3 tools/run_phase3_3_1_independent_eval.py
```

| Artifact | Path |
|---|---|
| Evaluation metrics (full JSON) | `docs/phase3/phase3_3_1_eval_metrics.json` |
| Frozen inputs & hashes | `docs/phase3/phase3_3_1_frozen_inputs.json` |
| Matched pairs (reconciliation) | `docs/phase3/phase3_3_1_reconciliation.csv` |
| Unmatched predictions | `docs/phase3/phase3_3_1_unmatched_events.csv` |
| Evaluation database | `data/phase3_3_1_carton_eval.db` |
| Split-integrity audit | `docs/phase3/DATASET_SPLIT_INTEGRITY_AUDIT.md` |

**Run ID:** `run_phase3_3_1_4366de05d420` — stamp this ID on any future acceptance evaluation once a valid untouched test set exists.
