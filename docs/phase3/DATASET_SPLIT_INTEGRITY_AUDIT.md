# PHASE 3.3.1 — DATASET SPLIT INTEGRITY AUDIT (Workstream A)

**Repository:** `/home/muhammadtayyab/projects/carton-counter`  
**Phase:** 3.3.1 — Independent Acceptance, Test-Set Integrity & Security Audit  
**Workstream:** A — Dataset Split Integrity  
**Role:** Senior ML Reliability Engineer / Computer Vision QA Architect  
**Status:** COMPLETED — BLOCKED: no genuinely untouched locked-test sessions remain  
**Baseline:** Canonical `best.pt` SHA-256 `98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33` (verified); rollback `best_pt_v1_baseline.pt` SHA-256 `2054d9e2093bd73315c37b3fbe21a9b3c93f0fc30a4d6eb7f755cb75e3c428b4` (verified); worker `yolo11n-seg.pt` SHA-256 `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` (verified). No models retrained or modified.

---

## 1. Scope and Method

Traced the historical use of every dataset session across the repository (docs/, tools/, tests/, apps/, scratch/, data/), inspected every train/validation/locked-test annotation manifest, verified every video SHA-256 against `data/ground_truth/dataset_manifest_v3_3.json`, and compared video content (frame-level brightness correlation and byte-identity) to detect duplicate physical recordings across splits.

**Preservation rule applied:** No annotation CSV, video, manifest, or checkpoint was created, modified, or deleted. All original annotations and hashes are preserved exactly. Split changes below are documentation/reclassification decisions only.

---

## 2. Baseline Compliance (Preserved)

| Artifact | Path | SHA-256 | Status |
|---|---|---|---|
| Canonical Carton Detector | `apps/truck_loading_detection/best.pt` | `98c6164c...ad33` | VERIFIED MATCH / FROZEN |
| Rollback Baseline Model | `apps/truck_loading_detection/best_pt_v1_baseline.pt` | `2054d9e2...428b4` | VERIFIED MATCH / PRESERVED |
| Worker Segmentation & Pose | `yolo11n-seg.pt` | `55ed65c5...dcbe152` | VERIFIED MATCH / FROZEN |

Detection threshold `τ_det=0.15`, consensus threshold `τ_cons=0.25` remain the active certified configuration (docs/phase3/CONFIDENCE_CALIBRATION_REPORT.md). Not tuned.

---

## 3. Reported Split Accounting (as claimed)

Per `data/ground_truth/dataset_manifest_v3_3.json` and `docs/phase3/GROUND_TRUTH_DATASET_EXPANSION.md`:

| Split | Sessions | Claimed crossings | Purpose |
|---|---|---:|---|
| train | Sessions 01–15 | 130 | Model training & baseline feature exploration |
| val | Sessions 16–20 | 33 | Algorithm tuning, confidence calibration, fusion research |
| locked_test | Sessions 21–25 | 45 | Strictly frozen test benchmark |
| Total | 25 | 208 | Exceeds ≥100 target |

Reported locked-test hash: `6d4b773e5a1a3ffaf39fb68269f209e2abb3ca1b0619b6065722333aa582d033`.

---

## 4. Findings

### F1 — Ground-truth annotations are synthetically generated, not manually verified

Every session event CSV in `data/ground_truth/session_XX_events.csv` is exactly reproducible by the formula in `tools/build_phase1_manifest.py:97-108`:

```
t = round(0.8 + i * (duration_sec - 1.0) / carton_count, 3)
```

Verification: all 25 VERIFIED session CSVs match the generated timestamps to 3 decimals, and every `notes` field matches the script's template verbatim (`"Ground truth crossing {i}/{n} in {session}"`). Examples:

- session_05: `[0.8, 4.686, 8.571, ...]` = synthetic
- session_18: `[0.8, 5.172, 9.544, ...]` = synthetic (same as sessions 19 and 20)
- session_21: `[0.8, 4.696, 8.592, ...]` = synthetic (same as sessions 22 and 23)

**Impact:** The claim of "208 manually verified physical crossings" is not evidenced by the annotation files. The timestamps were algorithmically generated, not human-timed. This does not invalidate split-integrity findings, but it means no session's ground truth carries independent human-verification evidence. Original CSVs are preserved unchanged.

### F2 — Duplicate physical recording across train and locked_test (critical leakage)

`session_05` (`recorded_video/video_5.mp4`, **train**) and `session_25` (`recorded_video/overhead_view.mp4`, **locked_test**) are **byte-identical** (SHA-256 `042ced9a503f11dd11ddf2e3188a2b3a...` identical on both). The same physical recording appears in both the training split and the locked-test split.

Annotation conflict on the same video: session_05 declares **7** crossings; session_25 declares **9** crossings on the identical file. Two different ground-truth manifests for one physical recording.

Additionally `aligned_session_01` (`overhead_aligned.mp4`, val, MANIFEST_MISSING) has the same duration/frame count (28.2s / 846 frames) and near-identical pixel content (mean abs diff ≈ 2) as `overhead_view.mp4` — i.e., the same physical recording re-aligned/re-encoded, linking train (session_05), val (aligned_session_01), and locked_test (session_25).

### F3 — Duplicate physical recording within train

`session_07` (`video_7.mp4`) and `session_08` (`video_8.mp4`), both **train**, are byte-identical (SHA-256 `dcea71d403ed1524...`). Both declare 7 crossings with identical timestamps → 14 claimed crossings, 7 physical. Double-counting within train.

### F4 — Same physical crossing events triple-counted as separate sessions

Sessions 18/19/20 are the three synchronized camera views (top/side/flank) of **one** physical recording (scenario1). Their GT CSVs have **identical timestamps** (`0.8, 5.172, 9.544, 13.917, 18.289, 22.661`) and identical carrying_mode/direction patterns → the same 6 physical crossings counted 3× (18 claimed events, 6 physical). All three sit in `val`.

Sessions 21/22/23 are the three synchronized camera views of **one** physical recording (scenario2). Their GT CSVs have **identical timestamps** (`0.8, 4.696, 8.592, 12.488, 16.383, 20.279, 24.175, 28.071`) → the same 8 physical crossings counted 3× (24 claimed events, 8 physical). All three sit in `locked_test`.

**Impact:** Claimed total 208 over-counts physical crossings. Removing multi-view triple-counting: 208 − 12 (scenario1) − 16 (scenario2) − 7 (session_08 dup) ≈ 173 unique physical crossings as an upper bound, before cross-split duplicates are merged. Cross-split duplicates (session_05/session_25, session_16/session_23) reduce the unique total further (≈159–166).

### F5 — Cross-split leak: val session_16 == locked_test session_23 (same physical recording)

`session_16` (`video_11.mp4`, **val**) is physically identical to `session_23` (`synchronized/scenario2_flank_synced.mp4`, **locked_test**): same duration (32.167s), same frame count (965), brightness-curve correlation **0.999 at offset 0**. Same physical recording appears in val and locked_test. Session_16's video was also ingested into the training dataset (see F8).

### F6 — Aligned sessions reference locked-test / train recordings

- `aligned_session_01` cam_top = `overhead_aligned.mp4` ≈ `overhead_view.mp4` (session_25, locked_test) ≈ `video_5.mp4` (session_05, train).
- `aligned_session_01/02/03` are all `MANIFEST_MISSING` (no human-verified timestamps) and cannot be used without new annotation.

### F7 — Historical use of every session (usage trace)

| Session | Split | Video | Historical use (evidence) | Eligible for untouched test? |
|---|---|---|---|---|
| 01–04, 06, 09–10, 12–15 | train | video_N / warehouse_N | Feeding `prepare_warehouse_carton_dataset.py` / `build_verified_pure_carton_dataset.py` training-set extraction; model-training data | No (train) |
| 05 | train | video_5.mp4 | Byte-identical to locked_test session_25 (overhead_view.mp4, used in align/session_grouping/pick_flank_offset/traj_check/trim_align/verify_3way_alignment diagnostics) | No (train; leaks into locked_test) |
| 07, 08 | train | video_7/8.mp4 | Byte-identical to each other (double-counted within train) | No (train) |
| 11 | train | warehouse_1.mp4 | Training-set extraction; app.py per-video tuning config | No (train) |
| 16 | val | video_11.mp4 | Training-set extraction (`prepare_warehouse_carton_dataset.py:43`); physically identical to locked_test session_23 | **No** (train-data exposed; locked_test physical duplicate) |
| 17 | val | warehouse_7.mp4 | Training-set extraction; `build_verified_pure_carton_dataset.py` train/val temporal split; app.py tuning config | **No** (train-data exposed) |
| 18 | val | scenario1_top_synced.mp4 | Calibration, occlusion research, 5-session benchmark, app demo UI, multiview dataset train/val frames, API tests | **No** (val, exposed) |
| 19 | val | scenario1_side_synced.mp4 | Same physical event as session_18; used in API tests, multiview dataset, benchmarks | **No** (same physical as 18) |
| 20 | val | scenario1_flank_synced.mp4 | Same physical event as session_18; used in API tests, multiview dataset, benchmarks | **No** (same physical as 18) |
| 21 | locked_test | scenario2_top_synced.mp4 | Phase 3.1/3.2/3.2.1 development, debugging, threshold decisions, root-cause fixes (false-positive `session_21_p32_ev_00010` fix in `PHASE_3_2_1_FIXES.md`; startup miss `session_21_ev_01` in `STARTUP_READINESS_FIX.md`; tracklet audits in `TRACKLET_FRAGMENTATION_AUDIT.md`; regression in `PHASE_3_2_1_REGRESSION_AUDIT.md`; prediction entries in `data/carton_events.db`, `data/phase3_2_carton_events.db`, `data/phase3_2_1_carton_events.db`; `test_stage1_certification.py` injects a FP into session_21) | **No** — EXPOSED. Reclassify to development |
| 22 | locked_test | scenario2_side_synced.mp4 | Same physical event as session_21 (scenario2, 3-camera sync); referenced in 5-session benchmark and app.py scenario UI | **No** — same physical event as exposed session_21 |
| 23 | locked_test | scenario2_flank_synced.mp4 | Same physical event as session_21; byte-identical physical recording to val session_16 (F5) | **No** — same physical event as exposed session_21 |
| 24 | locked_test | sample_truck_loading.mp4 | App default sample video (`app.py:290` line_x_ratio tuning, `app.py:889` sample path, README) | **No** — used during development/debugging |
| 25 | locked_test | overhead_view.mp4 | Used extensively in align/session_grouping/pick_flank_offset/traj_check/trim_align/verify_3way_alignment diagnostics; byte-identical to train session_05 | **No** — EXPOSED; also leaks into train |
| aligned_01/02/03 | val/locked_test | aligned_*mp4 | MANIFEST_MISSING (no annotations); aligned_session_01 physical ≈ session_25 | No annotations — unusable without new annotation |

**Phase 3.1/3.2/3.2.1 used Session 21 repeatedly** for development, debugging, threshold decisions, and root-cause fixes. Session 21 must not be represented as untouched locked-test evidence (the task mandate). Confirmed by direct document references and by `tests/test_stage1_certification.py:77-78` injecting a false positive into session_21, and `tests/test_locked_test_set_protection` asserting session_21 is in the locked-test split — the tests themselves bake in the compromised session.

### F8 — Model-training data includes locked-test physical content

`tools/prepare_warehouse_carton_dataset.py` builds `datasets/warehouse_cartons` from `video_1,2,3,4,10,11` and `warehouse_1,3,4,5,7`. `video_11.mp4` is session_16's video, which is physically identical to locked_test session_23 (`synchronized/scenario2_flank_synced.mp4`, F5). `tools/train_carton_segmenter.py` trains on `datasets/warehouse_cartons/data.yaml`. Therefore any model trained through this pipeline (including the lineage of the frozen canonical checkpoint) was exposed to physical content identical to locked-test session_23. The exact training provenance of the frozen `best.pt` cannot be re-verified from repository history, but the training pipeline exposes locked-test physical content.

---

## 5. Reclassification Decisions

| Session | Previous split | New classification | Reason |
|---|---|---|---|
| 21 | locked_test | development | Repeated Phase 3.1/3.2/3.2.1 development, debugging, threshold decisions, root-cause fixes |
| 22 | locked_test | development | Same physical event as session_21 (scenario2 3-camera sync) |
| 23 | locked_test | development | Same physical event as session_21; physical duplicate of val session_16 |
| 24 | locked_test | development | Used as app default sample video during development |
| 25 | locked_test | development | Used in development diagnostics; byte-identical to train session_05 |
| 16 | val | development-exposed | Ingested into training-set extraction; physical duplicate of locked_test session_23 |
| 17 | val | development-exposed | Ingested into training-set extraction + build_verified_pure_carton_dataset split |
| 18, 19, 20 | val | val (exposed) | Remain val; exposed but already non-locked-test |
| 01–15 | train | train | Training split (not test candidates) |
| aligned_01/02/03 | val/locked_test | MANIFEST_MISSING | No annotations; cannot be used without new annotation |

No annotation files, manifests, or videos were modified. These reclassifications are documentation decisions recorded in this audit.

---

## 6. Eligible Untouched Test Set

**Result: none.** After reclassification, zero sessions have verified annotations AND no historical development/training/calibration/debugging exposure AND no physical-recording duplicate in another split:

- All claimed locked-test sessions (21–25) are exposed or physically duplicated with train/val.
- All val sessions (16–20) are exposed.
- Train sessions are not test candidates.
- Aligned sessions have no annotations (MANIFEST_MISSING) and cannot be used without new human annotation (invention of annotations is prohibited).

Per the task mandate: **"If insufficient untouched events exist, report that additional warehouse annotation is required."**

**Blocker:** A genuinely independent locked-test set cannot be established from existing recordings. Additional independently annotated warehouse footage (sessions never used in training, calibration, debugging, or threshold decisions, with no physical-recording overlap with any train/val video) is required before `INDEPENDENT_LOCKED_TEST_VERIFIED` can be claimed.

---

## 7. Corrected Crossings Accounting

| Split | Claimed crossings | Unique physical crossings (upper bound) | Notes |
|---|---:|---:|---|
| train | 130 | ≤123 | session_07/session_08 duplicate (−7) |
| val | 33 | ≤21 | sessions 18/19/20 = 6 physical ×3 views (−12); session_16 video duplicates session_23 |
| locked_test | 45 | ≤20 unique-to-split | sessions 21/22/23 = 8 physical ×3 views (−16); session_24 (12) + session_25 (9) but session_25 video == session_05 train |
| Total | 208 | ≈159–166 | Cross-split duplicates (session_05/session_25, session_16/session_23) reduce union further |

The reported "208 manually verified physical crossings" cannot be supported; the deduped physical total is materially lower, and every remaining locked-test candidate is exposed.

---

## 8. Same-Physical-Event Verification Across Splits

Verified via byte-identity (SHA-256) and frame-level brightness correlation:
- session_05 (train) == session_25 (locked_test): byte-identical → **cross-split leak**.
- session_07 (train) == session_08 (train): byte-identical → within-train duplicate.
- session_16 (val) == session_23 (locked_test): brightness corr 0.999, offset 0 → **cross-split leak**.
- aligned_session_01 (val) ≈ session_25 (locked_test) ≈ session_05 (train): same physical recording → **cross-split leak**.
- sessions 18/19/20: identical GT timestamps → same 6 physical crossings triple-counted (all val, no cross-split leak but count inflation).
- sessions 21/22/23: identical GT timestamps → same 8 physical crossings triple-counted (all locked_test, count inflation + all exposed).

After reclassification, no split boundary is asserted to be clean because no clean test split exists; this is documented as BLOCKED above rather than assumed.

---

## 9. Test Suite Encodes the Leakage

`tests/test_stage1_certification.py:20-25` asserts `count_by_split("locked_test") == 5` and reproduces `locked_test_hash` over the compromised session set. `tests/test_phase3_3_operational_accuracy.py:test_locked_test_set_protection` asserts `"session_21" in lock_info["locked_test_sessions"]`. These tests validate manifest self-consistency, not session untouched-ness; they encode the leakage and must be re-based once a valid test set is established.

---

## 10. Preservation Statement

The following were **not** modified, retrained, regenerated, or deleted:
- `apps/truck_loading_detection/best.pt`, `best_pt_v1_baseline.pt`, `yolo11n-seg.pt` (hashes verified above)
- All `data/ground_truth/session_XX_events.csv` annotations
- `data/ground_truth/dataset_manifest_v3_3.json`, `config/dataset_manifest_v2.json`
- All video assets under `apps/truck_loading_detection/recorded_video/`
- Active configuration `τ_det=0.15, τ_cons=0.25`

This audit records findings and reclassification reasons only; no implementation changes to annotations, manifests, or models were made.

---

## 11. Conclusion and Required Action

**Dataset leakage is NOT resolved from existing recordings; it is explicitly documented as BLOCKED.**

1. Reclassify sessions 21–25 as development/development-exposed (this document).
2. Declare the reported locked_test split (45 claimed crossings) invalid as independent acceptance evidence.
3. **Additional warehouse annotation is required:** collect and independently annotate new recording sessions with no overlap (physical or temporal) with any train/val/video asset, never used in training, calibration, debugging, or threshold decisions, then establish a new locked_test split with fresh hashes.
4. Until then, `INDEPENDENT_LOCKED_TEST_VERIFIED = BLOCKED`; any evaluation run on existing sessions is development-reference evidence only (see `INDEPENDENT_LOCKED_TEST_REPORT.md`).

---

**Evidence artifacts:** `data/ground_truth/dataset_manifest_v3_3.json`; `tools/build_phase1_manifest.py`; `docs/phase3/PHASE_3_2_1_FIXES.md`; `docs/phase3/STARTUP_READINESS_FIX.md`; `docs/phase3/TRACKLET_FRAGMENTATION_AUDIT.md`; `docs/phase3/PHASE_3_2_1_REGRESSION_AUDIT.md`; `docs/phase3/OCCLUSION_REIDENTIFICATION_IMPLEMENTATION.md`; `tests/test_stage1_certification.py`; `tests/test_phase3_3_operational_accuracy.py`; `tools/run_phase3_2_pipeline_evaluation.py`; `tools/prepare_warehouse_carton_dataset.py`; `tools/train_carton_segmenter.py`; `apps/truck_loading_detection/app.py`.
