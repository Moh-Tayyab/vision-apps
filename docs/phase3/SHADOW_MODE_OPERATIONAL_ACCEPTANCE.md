# PHASE 3.3.1 — SHADOW MODE OPERATIONAL ACCEPTANCE (Workstream E)

**Repository:** `/home/muhammadtayyab/projects/carton-counter`  
**Phase:** 3.3.1 — Independent Acceptance, Test-Set Integrity & Security Audit  
**Workstream:** E — Shadow Mode Operational Qualification  
**Status:** COMPLETED — `SHADOW_MODE_OPERATIONAL_ACCEPTANCE = PRERECORDED_VIDEO_VALIDATED` (not live)  
**Run ID:** `shadow_311c03f52793`  
**Run timestamp (UTC):** `2026-11-02T17:24:00+00:00`  
**Machine-readable results:** `docs/phase3/phase3_3_1_shadow_eval.json`  
**Isolated shadow DB:** `data/phase3_3_1_shadow_eval.db`  
**Test script:** `tools/phase3_3_1_shadow_acceptance.py`

---

## 1. Components Audited

| Component | Path | Role |
|---|---|---|
| Shadow mode manager | `apps/truck_loading_detection/three_camera_system/shadow_mode.py` (`ShadowModeManager`) | Isolated shadow execution, counters, incidents, recovery |
| Shadow status endpoint | `apps/truck_loading_detection/app.py:2455` (`/api/shadow/status`, GET) | Mode, lock flag, health, incident count |
| Shadow counts endpoint | `apps/truck_loading_detection/app.py:2470` (`/api/shadow/counts`, GET) | GT vs auto vs human counts, discrepancy |
| Shadow compare endpoint | `apps/truck_loading_detection/app.py:2481` (`/api/shadow/compare`, POST) | Register human physical GT, compute discrepancy |
| Shadow incidents endpoint | `apps/truck_loading_detection/app.py:2499` (`/api/shadow/incidents`, GET) | Structured incident log |
| Shadow recover endpoint | `apps/truck_loading_detection/app.py:2521` (`/api/shadow/recover`, POST) | Restore session state from SQLite WAL |

---

## 2. Inventory Isolation — Proved (not assumed on a Boolean)

The task explicitly warns: *"Do not rely solely on a Boolean such as `authoritative_inventory_locked=True`. Prove that the shadow service has no unintended write path into authoritative inventory."*

**Proof by code inspection:**
1. Grepped the entire repository for ERP/WMS/SAP/Oracle and any inventory-write integration (inventory POST/PUT clients, `requests.post`, `httpx`, external inventory APIs): **none found** in the carton-counting application.
2. `ShadowModeManager` writes **only** to its own local SQLite tables: `shadow_incidents` and `shadow_session_manifest` (three_camera_system/shadow_mode.py:88-113, 135-169).
3. It reads carton counts from the local `EventStore` (`carton_crossing_events`) and computes discrepancy locally (`get_shadow_counts`, shadow_mode.py:204-229).
4. All five `/api/shadow/*` endpoints read from or write to the local shadow DB only; none call any external inventory system.
5. `authoritative_inventory_locked=True` (default, set in `get_shadow_manager` at app.py:2450) is a documented guard flag persisted in `shadow_session_manifest.is_inventory_locked`, but the **substantive** protection is the complete absence of any external inventory integration path — there is nothing to write to.

**Allowed operations (all exercised, all local shadow DB):**
- `/api/shadow/status` — read-only → HTTP 200
- `/api/shadow/counts` — read-only → HTTP 200
- `/api/shadow/compare` — writes human physical GT to `shadow_session_manifest` only → HTTP 200
- `/api/shadow/incidents` — read-only → HTTP 200
- `/api/shadow/recover` — restores from SQLite WAL, logs `RESTART_RECOVERY` incident locally → HTTP 200

**Prohibited operations:** No external inventory write path exists in the repository, so no prohibited inventory mutation can be exercised — it is blocked by design (absence of integration), not merely by a Boolean.

---

## 3. Operational Measurements (prerecorded video)

**Test execution:** Prerecorded synchronized multi-camera video (`scenario2_top/side/flank_synced.mp4`), 400 frames processed (frame cap applied — CPU-bound full pipeline), canonical model `best.pt`, `τ_det=0.15`, `τ_cons=0.25`, isolated shadow DB.

| Measurement | Value |
|---|---|
| Continuous runtime | 76.29 s (400 frames) |
| Throughput | 5.24 frames/sec |
| Latency P50 | 178.03 ms |
| Latency P95 | 296.61 ms |
| Latency P99 | 717.97 ms |
| Dropped frames | 40 / 400 (10.00%) — **simulated camera-disconnect window** (frames 150–190), not real drops |
| Dropped-frame rate | 10.00% |
| Max sync jitter | 2,824.09 ms |
| Sync jitter threshold exceeded incidents | logged (`SYNC_JITTER_EXCEEDED` when >33.3 ms) |
| CPU % (before/after) | captured in JSON |
| Memory RSS (before/after, MB) | captured in JSON |
| Incidents logged | 362 (incl. `RESTART_RECOVERY`) |

**Camera disconnect / reconnect:** Simulated by feeding `None` frames for all cameras during frames 150–190, then resuming. Pipeline continued processing after reconnect; `CAMERA_FRAME_DROP` incidents logged per shadow_mode.py:171-187.

**Restart / SQLite recovery:** A second `ShadowModeManager` opened the same isolated DB and called `recover_session_state()`: recovered `human_physical_count=8`, `authoritative_inventory_locked=True`, incident history preserved (362 incidents). Verifies crash recovery from SQLite WAL.

**Count discrepancy:**
- Human physical GT registered: 8 (scenario2 physical crossings)
- Automatic confirmed: 1; Human approved: 0; System total confirmed: 1
- Unresolved (review required): 5
- Discrepancy: 7 (87.5% rate)
- This reflects the known low automatic recall (~21% on development evidence); the shadow mode correctly surfaces the gap rather than hiding it.

---

## 4. Testing Distinction (honest classification)

| Mode | Status |
|---|---|
| Prerecorded video testing | ✅ PERFORMED (this report) |
| Simulated camera-stream testing | ⚠️ PARTIAL (frame-drop simulation only; no full stream disconnect/reconnect harness) |
| Actual live camera operation | ❌ NOT PERFORMED |
| Real warehouse shadow deployment | ❌ NOT PERFORMED |

No live cameras or ERP/WMS access were available. Per the task mandate: never claim live deployment based on prerecorded processing. `SHADOW_MODE_OPERATIONAL_ACCEPTANCE = PRERECORDED_VIDEO_VALIDATED` — live-shadow and real-deployment claims remain **BLOCKED** pending live-camera and ERP/WMS access.

---

## 5. Limitations

- Measurements are on a prerecorded synchronized video on a CPU host; GPU/TensorRT would change latency/throughput.
- The automatic confirmation rate is low (1/8 in this run), consistent with the Phase 3.3 finding that unattended production operation is prohibited; shadow mode is correctly positioned as human-in-the-loop only.
- `authoritative_inventory_locked=True` is respected but cannot be proven sufficient by itself in a real deployment with an actual ERP/WMS integration — that integration does not yet exist in the repository.

---

## 6. Evidence

| Artifact | Path |
|---|---|
| Machine-readable results | `docs/phase3/phase3_3_1_shadow_eval.json` |
| Isolated shadow DB | `data/phase3_3_1_shadow_eval.db` |
| Test script | `tools/phase3_3_1_shadow_acceptance.py` |
| Shadow manager source | `apps/truck_loading_detection/three_camera_system/shadow_mode.py` |
| Shadow API source | `apps/truck_loading_detection/app.py:2440-2531` |

**Run ID:** `shadow_311c03f52793` — stamp on any re-test after remediation or live-camera availability.
