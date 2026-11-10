# PHASE 4 — PRODUCTION DEPLOYMENT (Workstream A)

**Repository:** `/home/muhammadtayyab/projects/carton-counter`  
**Phase:** 4 — Controlled Production Deployment  
**Workstream:** A — Production Release Engineering  
**Status:** COMPLETED  
**Tested environment:** Python 3.12, CPU-only (no GPU detected; `torch.cuda.is_available() == False`), Debian/Ubuntu host. Docker 29.8.2 + Compose plugin.

---

## 1. Overview

This workstream prepares a reproducible, controlled production deployment of the
carton-counting (truck loading detection) service. It does **not** retrain or
replace frozen models, does not weaken multi-camera consensus, and does not tune
configuration using locked-test data.

**Fundamental invariant preserved:** one genuine physical carton crossing must
produce exactly one logical confirmed crossing event; automatic and human-approved
counts remain separately identifiable.

---

## 2. Deployment Artifacts

| Artifact | Path | Purpose |
|---|---|---|
| Dockerfile | `apps/truck_loading_detection/Dockerfile` | CPU-only production image; frozen-model hash validation at import; non-root; health check |
| docker-entrypoint.sh | `apps/truck_loading_detection/docker-entrypoint.sh` | Entrypoint: logs build/version/config, then execs gunicorn |
| docker-compose.yml | `docker-compose.yml` | Multi-service compose; persistent `carton_data` volume for SQLite + audit records |
| .env.example | `.env.example` | Runtime configuration template (no real secrets) |
| requirements.txt | `apps/truck_loading_detection/requirements.txt` | Pinned production dependency manifest |
| deploy.sh | `apps/truck_loading_detection/scripts/deploy.sh` | Build + start + health-verify the service |
| rollback.sh | `apps/truck_loading_detection/scripts/rollback.sh` | Roll back to `best_pt_v1_baseline.pt` via `CARTON_MODEL_PATH` |
| Production deployment doc | `docs/phase4/PRODUCTION_DEPLOYMENT.md` | This document |

---

## 3. Baseline & Configuration (Preserved)

### Frozen models (SHA-256 verified at container startup — fail-closed on mismatch)

| Artifact | Path | SHA-256 |
|---|---|---|
| Canonical Carton Detector | `apps/truck_loading_detection/best.pt` | `98c6164c83905feb9db6d9bee6f48ab239c5d68a00efbe0484dbf3448333ad33` |
| Rollback Baseline Model | `apps/truck_loading_detection/best_pt_v1_baseline.pt` | `2054d9e2093bd73315c37b3fbe21a9b3c93f0fc30a4d6eb7f755cb75e3c428b4` |
| Worker Segmentation & Pose | `yolo11n-seg.pt` | `55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152` |

### Active configuration (Phase 3.3 certified, preserved)

| Setting | Env var | Value | Source |
|---|---|---|---|
| Detection threshold `τ_det` | `CONF_THRESHOLD` | `0.15` | Certified Phase 3.3 active config |
| Consensus threshold `τ_cons` | `CARTON_CONSENSUS_THRESHOLD` | `0.25` | Certified Phase 3.3 active config |
| IOU threshold | `IOU_THRESHOLD` | `0.45` | — |
| Image size | `IMG_SIZE` | `640` | — |
| Frame stride | `FRAME_STRIDE` | `2` | — |
| Torch threads | `TORCH_THREADS` | `4` | CPU-only sizing |
| Workers | `WORKERS` | `1` | Inference thread is the bottleneck |

**Configuration source:** environment variables (`.env` → compose `environment` → container `ENV`). No production secrets are hardcoded. `.env.example` contains placeholders only; real keys (e.g. `ROBOFLOW_API_KEY`) must be injected via Docker secrets or a protected non-committed file at runtime.

**Runtime rollback:** setting `CARTON_MODEL_PATH` to a frozen checkpoint (e.g.
`best_pt_v1_baseline.pt`) overrides the canonical model at startup. Used by
`scripts/rollback.sh`.

---

## 4. Docker Image (`Dockerfile`)

- Base: `python:3.12-slim` (CPU-only; no GPU assumed or required).
- System deps: OpenCV headless libraries + `libgomp1` (torch CPU).
- Pinned dependencies installed from `requirements.txt` (cached layer).
- Build args `BUILD_VERSION` and `MODEL_VERSION` baked into `ENV` for config/version
  identification (visible in `/health` and entrypoint logs).
- Non-root user (`appuser`).
- `VOLUME ["/app/data"]` for persistent SQLite event store + audit records.
- `HEALTHCHECK` hits `/health` and requires `status == "healthy"` **and** `models_valid == true`.
- `STOPSIGNAL SIGTERM` + `--graceful-timeout 15` so the app's `state.cleanup()`
  (SIGTERM handler + `atexit`) runs on shutdown.
- `ENTRYPOINT ["/app/docker-entrypoint.sh"]` logs version/config, then execs gunicorn.
- `CMD`: gunicorn `--workers 1 --threads 4 --preload` bound to `0.0.0.0:5000`.

Startup validation: `apps/truck_loading_detection/app.py` verifies all frozen model
SHA-256 digests at import (`_MODELS_VALID`). A corrupted/substituted checkpoint
fails closed before traffic is served; `/health` reports `models_valid: false`.

---

## 5. Docker Compose (`docker-compose.yml`)

Services: `carton_counter`, `helmet_detection`, `face_authorization`, `face_auth_qdrant`.

`carton_counter` production settings:
- `volumes`: `carton_data` → `/app/data` (persistent SQLite + audit records),
  `carton_uploads` → `/app/uploads`, `recorded_video` read-only.
- `restart: unless-stopped`, `stop_grace_period: 15s`.
- `healthcheck`: `/health` with `models_valid` requirement.
- Environment injected explicitly (no secrets baked in).

**Persistent storage:** `carton_data` volume survives container recreation/upgrade.
Back up `/app/data/carton_events.db` (and WAL files) as part of operational
backups (see Workstream B / `DATABASE_DURABILITY_AND_RECOVERY.md`).

---

## 6. Startup, Health, and Shutdown

### Startup validation
1. Container entrypoint logs `BUILD_VERSION`, `MODEL_VERSION`, thresholds.
2. `gunicorn --preload` imports `app` → `StreamState` loads frozen models →
   `verify_frozen_models()` checks SHA-256 of `best.pt`,
   `best_pt_v1_baseline.pt`, `yolo11n-seg.pt` (`_MODELS_VALID`).
3. If any hash mismatches or a file is missing, startup logs `ERROR` and
   `_MODELS_VALID = False`; `/health` returns `status: degraded` /
   `models_valid: false`. The service does not silently serve with a bad model.

### Health endpoints
- `GET /health` — full status: `models_valid`, `model`, `model_sha256`,
  `detection_threshold`, `consensus_threshold`, counts, FPS, uptime.
- `GET /healthz` — liveness probe (capture opened).
- `GET /metrics` — Prometheus plaintext (FPS, total_in, total_out, net).

### Graceful shutdown
- `SIGTERM`/`SIGINT` → `_shutdown_handler` → `shutdown_event.set()` +
  `state.cleanup()` (releases capture, closes DB).
- `atexit.register(state.cleanup)`.
- Compose `STOPSIGNAL SIGTERM` + `stop_grace_period: 15s`.

---

## 7. Deployment Procedure

```bash
# 1. Review/copy environment (set real secrets via secrets manager, not committed)
cp .env.example .env   # only if .env does not exist; edit as needed

# 2. Deploy (build + start + health-verify)
cd apps/truck_loading_detection
./scripts/deploy.sh up

# 3. Check status
./scripts/deploy.sh status

# 4. View logs
./scripts/deploy.sh logs

# 5. Stop
./scripts/deploy.sh down
```

Or directly with compose:
```bash
docker compose -f docker-compose.yml up -d carton_counter
```

### Rollback (controlled)
If a regression is detected after a model/config change:

```bash
cd apps/truck_loading_detection
./scripts/rollback.sh
```

This restarts `carton_counter` with `CARTON_MODEL_PATH=best_pt_v1_baseline.pt`
and verifies `/health` reports `models_valid: true` and the rollback model name.
To roll forward, unset `CARTON_MODEL_PATH` (or point it at `best.pt`) and redeploy.

---

## 8. Monitoring & Observability

- `/health` JSON: service status, model validity + sha256, thresholds, counts,
  FPS, uptime, reconnect attempts, frame drops, active tracks.
- `/metrics`: Prometheus-scrapable plaintext counters/gauges.
- Container logs: gunicorn access/error + app structured logging (stdout).
- Compose healthcheck (30s interval) drives restart policy (`unless-stopped`).

---

## 9. GPU / Hardware Assumptions

- **No GPU assumed or required.** The host has no NVIDIA GPU (`nvidia-smi` absent,
  `torch.cuda.is_available() == False`). The image installs the CPU torch wheel
  (`torch==2.13.0+cpu`).
- If a GPU becomes available, switch to the CUDA torch wheel and tune
  `TORCH_THREADS`/workers; this is a configuration change outside this workstream.

---

## 10. Security Notes

- Non-root container user.
- CORS restricted to `CORS_ORIGIN` (default `localhost,127.0.0.1`; never `*` in prod).
- No production secrets in image or compose. `.env.example` uses `REPLACE_ME_*`
  placeholders; real keys via Docker secrets / secrets manager.
- Frozen-model hash verification fails closed on tampering/corruption.
- Human-review endpoints (`/api/3angle/audit_event`, `/api/3angle/review_queue`)
  are **not** exposed to untrusted callers until the Phase 3.3.1 authentication
  blocker (Workstream C) is remediated (see `docs/phase3/HUMAN_REVIEW_SECURITY_AUDIT.md`).

---

## 11. Limitations Carried From Phase 3.3.1

- `INDEPENDENT_LOCKED_TEST_VERIFIED = BLOCKED`: no valid untouched locked-test set;
  deployment validated on prerecorded video only.
- `HUMAN_REVIEW_AUTHENTICATION_SECURE = BLOCKED`: review endpoints not remediated.
- `LIVE_SHADOW_VERIFIED = NOT_IMPLEMENTED`: no live cameras / ERP-WMS integration.
- Unattended production operation remains **prohibited** (human-in-the-loop only).
