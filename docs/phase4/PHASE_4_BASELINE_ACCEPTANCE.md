# PHASE 4 — BASELINE ACCEPTANCE (Step 0)

**Repository:** `/home/muhammadtayyab/projects/carton-counter`  
**Phase:** 4 — Controlled Production Deployment  
**Step:** 0 — Verify Phase 3.3.1 Completion Before Deployment Modifications  
**Status:** COMPLETED  
**Related report:** `docs/phase3/PHASE_3_3_1_FINAL_ACCEPTANCE_REPORT.md`

---

## 1. Purpose

Before any Phase 4 deployment engineering changes are made, verify that the Phase 3.3.1 acceptance milestone is complete and document its verified findings, blockers, and operational limitations. Phase 4 must not weaken, retrain, or retune anything certified (or explicitly blocked) in Phase 3.3.1.

This document is the gate: if any Phase 3.3.1 acceptance requirement is still blocked, it is either resolved here or explicitly documented before deployment modifications proceed.

---

## 2. Phase 3.3.1 Deliverables — Presence Verification

All Phase 3.3.1 deliverables exist in `docs/phase3/`:

| Deliverable | Path | Status |
|---|---|---|
| Dataset Split Integrity Audit | `docs/phase3/DATASET_SPLIT_INTEGRITY_AUDIT.md` | Present — BLOCKED (leakage documented) |
| Independent Locked-Test Report | `docs/phase3/INDEPENDENT_LOCKED_TEST_REPORT.md` | Present — BLOCKED (no untouched test set) |
| Human Review Security Audit | `docs/phase3/HUMAN_REVIEW_SECURITY_AUDIT.md` | Present — BLOCKED (auth not secure) |
| Full Regression Test Report | `docs/phase3/FULL_REGRESSION_TEST_REPORT.md` | Present — COMPLETED |
| Shadow Mode Operational Acceptance | `docs/phase3/SHADOW_MODE_OPERATIONAL_ACCEPTANCE.md` | Present — PRERECORDED_VIDEO_VALIDATED |
| Final Acceptance Report | `docs/phase3/PHASE_3_3_1_FINAL_ACCEPTANCE_REPORT.md` | Present |

Evaluation evidence artifacts present:
- `docs/phase3/phase3_3_1_eval_metrics.json`
- `docs/phase3/phase3_3_1_frozen_inputs.json`
- `docs/phase3/phase3_3_1_reconciliation.csv`
- `docs/phase3/phase3_3_1_unmatched_events.csv`
- `docs/phase3/phase3_3_1_security_test_results.json`
- `docs/phase3/phase3_3_1_shadow_eval.json`

---

## 3. Baseline Model & Configuration — Verified Against Repository

| Artifact | Path | Expected SHA-256 | Actual SHA-256 | Match |
|---|---|---|---|---|
| Canonical Carton Detector | `apps/truck_loading_detection/best.pt` | `98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33` | `98c6164c...ad33` | ✅ VERIFIED |
| Rollback Baseline Model | `apps/truck_loading_detection/best_pt_v1_baseline.pt` | `2054d9e2093bd73315c37b3fbe21a9b3c93f0fc30a4d6eb7f755cb75e3c428b4` | `2054d9e2...428b4` | ✅ VERIFIED |
| Worker Segmentation & Pose | `yolo11n-seg.pt` | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` | `55ed65c5...dcbe152` | ✅ VERIFIED |

**Active configuration (Phase 3.3 certified, preserved):**
- Detection threshold `τ_det = 0.15`
- Consensus threshold `τ_cons = 0.25`

**Note on repository discrepancy:** `apps/truck_loading_detection/app.py` defaults `CONF_THRESHOLD` env to `0.24` and the `Dockerfile` sets `CONF_THRESHOLD=0.24`, while the three-camera `pipeline.py` uses `conf_threshold=0.15` and `consensus min_two_camera_conf=0.25`. Phase 4 Workstream A aligns deployment environment defaults to the certified Phase 3.3 active configuration (`τ_det=0.15`, `τ_cons=0.25`) — this is a deployment-config alignment, **not** threshold tuning using locked-test data. No model weights, class mappings, or inference behavior are changed.

---

## 4. Phase 3.3.1 Acceptance Requirements — Verification

| # | Requirement | Verified | Evidence |
|---|---|---|---|
| 1 | Dataset train/validation/locked-test separation | ⚠️ PARTIAL | `docs/phase3/DATASET_SPLIT_INTEGRITY_AUDIT.md` documents leakage: sessions 21–25 exposed/duplicated; sessions 05/25 byte-identical across train/locked_test; sessions 18/19/20 and 21/22/23 triple-count same physical events. Separation is documented as violated, not assumed clean. |
| 2 | Previously exposed sessions not claimed as untouched locked-test evidence | ✅ | Sessions 21–25 reclassified to `development`; audit states zero genuinely untouched verified sessions exist. |
| 3 | Operator authentication and authorization | ❌ BLOCKED | `docs/phase3/HUMAN_REVIEW_SECURITY_AUDIT.md`: `X-Operator-ID` independently authenticates; hardcoded plaintext tokens; no role auth; `/api/3angle/review_queue` unauthenticated. Security tests: 12 passed, 3 failed, 1 blocked. |
| 4 | Database persistence and idempotency | ✅ | Event-store `record_manual_review` idempotent (`ALREADY_CONFIRMED_IDEMPOTENT`), transactional (single SQLite WAL commit), append-only audit trail. Concurrent-approval test: no double count. |
| 5 | Inventory isolation | ✅ | No ERP/WMS client or inventory-write integration in repo; `ShadowModeManager` writes only to local SQLite tables. Proved by code inspection. |
| 6 | Frozen model hashes | ✅ | All three checkpoints match expected SHA-256 (table above). |
| 7 | Complete regression results | ✅ | 277 passed, 3 failed, 4 skipped, 2 errors. Failures are environment/app-mismatch or obsolete suites, not logic regressions. No tests deleted; no coverage reduced. |
| 8 | Actual deployment & operational limitations documented | ✅ | Final acceptance report Section 11 explicitly rejects unattended production readiness; unattended operation prohibited (auto recall ~21%). |

---

## 5. Unresolved Blockers Carried Into Phase 4

These blockers are **documented, not resolved** in this Step 0. Phase 4 must not claim they are fixed:

| Blocker | Impact on Phase 4 | Phase 4 disposition |
|---|---|---|
| B1 — No valid locked-test set (sessions 21–25 exposed/duplicated) | `INDEPENDENT_LOCKED_TEST_VERIFIED = BLOCKED` | Additional independently annotated warehouse footage required. Phase 4 deployment proceeds on prerecorded-video validation only; no locked-test acceptance claim. |
| B2 — Human-review authentication not secure | `HUMAN_REVIEW_AUTHENTICATION_SECURE = BLOCKED` | Phase 4 must not expose `/api/3angle/audit_event` or `/api/3angle/review_queue` to untrusted callers without remediation. Deployment docs mark human-review endpoints as requiring auth remediation before exposure. |
| B3 — No live shadow deployment | `LIVE_SHADOW_VERIFIED = NOT_IMPLEMENTED` | Shadow mode validated on prerecorded video only. Phase 4 docs distinguish prerecorded-video vs live-camera vs real-warehouse deployment. |
| B4 — Three full-suite failures + 2 errors (environment/obsolete suites) | Regression suite not 100% green | Documented; not logic regressions. Phase 4 re-runs suite and records actual results. |

---

## 6. Operational Limitations (Explicitly Carried Forward)

Per the Phase 3.3.1 final acceptance report Section 11, the following do **not** justify unattended production readiness, and none are claimed by Phase 4:

- Unit tests alone.
- A small number of automatically confirmed events.
- Human-approved event totals.
- Recorded-video processing alone.
- A health endpoint returning HTTP 200.

**Unattended production operation remains prohibited.** Automatic recall is low (~21% on development evidence); most crossings require human review. The system is suitable only for human-in-the-loop / shadow-mode operation until blockers B1–B3 are resolved.

---

## 7. Gate Decision

**Phase 3.3.1 acceptance milestone: COMPLETED (with documented blockers).**

Phase 4 deployment engineering may proceed, subject to these constraints:
1. Do not retrain or replace frozen models (`best.pt`, `best_pt_v1_baseline.pt`, `yolo11n-seg.pt`).
2. Do not change model class mappings.
3. Do not weaken multi-camera consensus to increase automatic recall.
4. Do not mix human-approved counts with automatic counts.
5. Do not remove rollback checkpoints.
6. Do not destroy uncommitted user changes; do not use destructive Git commands.
7. Preserve existing API compatibility.
8. Do not remove Phase 1–3 deliverables.

All unresolved blockers (B1–B4) are documented above and carried into Phase 4 as known limitations.

---

## 8. Evidence

- `docs/phase3/PHASE_3_3_1_FINAL_ACCEPTANCE_REPORT.md` — full Phase 3.3.1 findings
- `docs/phase3/DATASET_SPLIT_INTEGRITY_AUDIT.md` — split integrity
- `docs/phase3/INDEPENDENT_LOCKED_TEST_REPORT.md` — locked-test evaluation
- `docs/phase3/HUMAN_REVIEW_SECURITY_AUDIT.md` — auth/security
- `docs/phase3/FULL_REGRESSION_TEST_REPORT.md` — regression results
- `docs/phase3/SHADOW_MODE_OPERATIONAL_ACCEPTANCE.md` — shadow acceptance
- Baseline checkpoint SHA-256 verified at Phase 4 Step 0 (table in Section 3).
