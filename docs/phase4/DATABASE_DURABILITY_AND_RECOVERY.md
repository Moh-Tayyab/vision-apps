# PHASE 4 — DATABASE DURABILITY AND RECOVERY (Workstream B)

**Repository:** `/home/muhammadtayyab/projects/carton-counter`  
**Phase:** 4 — Controlled Production Deployment  
**Workstream:** B — Database and Data Durability  
**Status:** COMPLETED  
**Implementation:** `apps/truck_loading_detection/three_camera_system/event_store.py`  
**Tests:** `tests/test_event_store_durability.py` (11 tests, all passing)

---

## 1. Purpose

Guarantee durability of the fundamental counting invariant:

> **One genuine physical carton crossing must produce exactly one logical confirmed crossing event.**

All crossing events, human review decisions, and audit history are persisted in a
SQLite WAL-backed `EventStore`. This workstream hardens that store for production:
schema migration tracking, WAL checkpoint management, backup/restore, corrupt-database
detection, disk-space monitoring, and recovery after restart / unexpected termination.

No historical production or evaluation records are deleted or modified by these changes.

---

## 2. Design & Invariants (Preserved)

| Invariant | Mechanism |
|---|---|
| Exactly-once persistence | `UNIQUE(session_id, global_carton_id, direction)` + `crossing_occurrence_id` duplicate window (3.5s). Duplicate replay returns `ALREADY_COMMITTED_IDEMPOTENT`. |
| Atomic commit | Event recording, state transitions, and ledger totals in a single SQLite transaction (`isolation_level="DEFERRED"`, `conn.commit()`). |
| Concurrency safety | WAL journal mode, `busy_timeout=5000`, internal re-entrant `threading.RLock`. |
| Automatic vs. human separation | `is_auto_confirmed` and `is_human_approved` strictly partitioned (`is_auto_confirmed=0` for manual approvals). Human-approved counts never back-fill automatic counts. |
| Full audit trail | `review_audit_log` append-only: every manual review records operator, timestamp, previous/new state, notes. |
| Restart & recovery | `recover_state()` repopulates committed IDs, events, and counts from the durable DB on process restart. |

---

## 3. New Durability Features

### 3.1 Schema migration tracking (`schema_migrations` table)
A new `schema_migrations` table records every applied schema migration (`version`, `description`, `applied_at`, `sql`).

- `record_migration(version, description, sql)` — insert/replace a migration record.
- `get_applied_migrations()` — list all recorded migrations, oldest first.
- `ensure_migration(version, description, sql)` — idempotent: applies `sql` only if `version` is not already recorded, then records it. Returns `True` if applied, `False` if already present.

This lets production upgrades track schema evolution safely and re-run without re-applying.

### 3.2 WAL checkpoint management
- `wal_checkpoint(mode="TRUNCATE")` — runs `PRAGMA wal_checkpoint(<mode>)` and returns `(busy, log, checkpointed)`. Use this to flush the WAL into the main database file before backups, restores, or scheduled maintenance.

### 3.3 Integrity / corruption detection
- `integrity_check()` — `PRAGMA integrity_check` result string.
- `quick_check()` — `PRAGMA quick_check` result string.
- `is_corrupt()` — `True` if `quick_check() != "ok"` or the file could not be initialized.

The `EventStore` constructor no longer hard-fails on a missing/corrupt database file: it catches `sqlite3.DatabaseError`, sets `self._corrupt = True`, and methods degrade gracefully (integrity checks report `"corrupt"` instead of raising).

### 3.4 Backup & restore
- `backup(backup_path)` — consistent snapshot via the SQLite online backup API (source stays readable/writable during copy). Returns the absolute backup path.
- `restore(backup_path)` — replaces the live database with a consistent backup (intended to be run while the service is stopped). Checkpoints the current WAL first, copies the backup over the live file, and removes stale `-wal`/`-shm` sidecars so stale frames are not replayed.

### 3.5 Disk-space monitoring
- `disk_free_bytes(path=None)` — free bytes on the volume holding the DB (or `path`).
- `check_disk_space(min_free_mb=100.0)` — `(ok, free_bytes, total_bytes)`; `ok` is `False` when free space falls below the threshold.

---

## 4. Test Coverage (`tests/test_event_store_durability.py`)

| Test | Verifies |
|---|---|
| `test_exactly_once_commit_duplicate_replay` | Duplicate crossing replay returns `ALREADY_COMMITTED_IDEMPOTENT`; counts do not double. |
| `test_legitimate_repeat_crossing` | Same carton, distinct occurrence (outside 3.5s window) commits as a new event; counts reflect 1 auto + 1 review. |
| `test_concurrent_review_approval_no_double_count` | 6 concurrent approvals → exactly 1 applied, 5 idempotent no-ops; single `human_approved` count; audit trail preserved (≥6 entries). |
| `test_concurrent_commit_lock_contention` | 8 threads × 10 commits (80 events) under lock contention → all persisted, no errors, counts correct. |
| `test_backup_and_restore` | `backup()` produces a readable file; restore after mutation returns DB to backed-up state; restored DB not corrupt. |
| `test_schema_migration_tracking` | `ensure_migration` applies once and is idempotent on repeat; `get_applied_migrations()` records version/description. |
| `test_wal_checkpoint` | `wal_checkpoint("TRUNCATE")` returns `(busy, log, checkpointed)`. |
| `test_disk_space_monitoring` | `check_disk_space` and `disk_free_bytes` report positive free/total bytes. |
| `test_corrupt_database_detection` | Valid DB → `is_corrupt() == False`; garbage file → `is_corrupt() == True` (constructor does not hard-fail). |
| `test_restart_recovery_persists_state` | Reopening the same DB after "process exit" recovers committed IDs, events, counts, and audit history. |
| `test_event_persistence_across_reopen` | Events persist across `EventStore` re-instantiation on the same DB file. |

**Result: 11 passed, 0 failed.**

---

## 5. Operational Procedures

### 5.1 Backup (while service is running)
```bash
python3 - <<'EOF'
from apps.truck_loading_detection.three_camera_system.event_store import EventStore
store = EventStore(db_path="data/carton_events.db")
print("Backup:", store.backup("/backups/carton_events_$(date +%s).db"))
print("Integrity:", store.integrity_check())
print("Disk OK:", store.check_disk_space(min_free_mb=500)[0])
EOF
```

### 5.2 Restore (service stopped)
```bash
# Stop the carton_counter service first, then:
python3 - <<'EOF'
from apps.truck_loading_detection.three_camera_system.event_store import EventStore
store = EventStore(db_path="data/carton_events.db")
store.restore("/backups/carton_events_<timestamp>.db")
print("Restored. Corrupt:", store.is_corrupt())
EOF
# Restart the service; recover_state() repopulates in-memory state from the DB.
```

### 5.3 Schema migration (production upgrade)
```python
from apps.truck_loading_detection.three_camera_system.event_store import EventStore
store = EventStore(db_path="data/carton_events.db")
store.ensure_migration(
    "v2_add_archive_flag",
    "Add archived flag to crossing events",
    "ALTER TABLE carton_crossing_events ADD COLUMN archived INTEGER NOT NULL DEFAULT 0;",
)
print(store.get_applied_migrations())
```

### 5.4 Corrupt database recovery
1. Stop the service.
2. Confirm: `store.is_corrupt() == True` (or `quick_check() != "ok"`).
3. Restore the most recent good backup via `restore()` (or copy the backup `.db` file over the live one and remove stale `-wal`/`-shm`).
4. Restart the service; `recover_state()` rebuilds in-memory state.

---

## 6. Persistent Storage Layout (Production)

| Path | Contents | Persistence |
|---|---|---|
| `/app/data/carton_events.db` | Crossing events + review audit log + schema migrations | Persistent Docker volume `carton_data` |
| `/app/data/carton_events.db-wal` | Write-ahead log | Rotated via `wal_checkpoint` |
| `/app/data/carton_events.db-shm` | Shared-memory index | Transient; removed on restore |
| `/app/uploads/` | Uploaded video files | Persistent Docker volume `carton_uploads` |

Back up `/app/data/` regularly. The Docker Compose file mounts `carton_data` → `/app/data` so the database survives container recreation/upgrade.

---

## 7. Relationship to Phase 3.3.1

Workstream B preserves and builds on the Phase 3.3.1-verified event store:
- Idempotent (`ALREADY_COMMITTED_IDEMPOTENT`) and transactional review decisions remain intact.
- `review_audit_log` append-only history is preserved.
- Automatic vs. human-approved count separation is preserved.

It does **not** alter the frozen models, class mappings, consensus thresholds, or counting logic.

---

## 8. Limitations

- Backups/restore assume the service is stopped (or at least quiesced) for `restore()` to avoid replacing a live, actively-written database file mid-write.
- `is_corrupt()` uses `quick_check` (fast, detects most corruption); `integrity_check()` is more thorough but slower on large databases.
- Disk-space monitoring reports the DB volume only; monitor application logs and container metrics separately for full observability.
