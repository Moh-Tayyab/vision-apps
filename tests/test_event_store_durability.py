"""
Phase 4 Workstream B — Database durability & recovery tests.

Covers:
  - Durable crossing-event identity & exactly-once commit
  - Safe repeated processing (duplicate crossing replay)
  - Transactional human approval/rejection
  - Schema migration tracking
  - Persistent event history / recovery after restart
  - Backup & restore
  - WAL checkpoint management
  - Disk-space monitoring
  - Corrupt database detection
  - Database lock contention
  - Concurrent review approval (no double count)
  - Legitimate repeat crossings (same carton, distinct occurrence)

Uses isolated SQLite databases under pytest tmp_path so no production or
evaluation records are touched.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from apps.truck_loading_detection.three_camera_system.event_store import EventStore


def _seed_event(store: EventStore, event_id: str, global_cid: str, ts: float,
                session_id: str = "s1", direction: str = "IN",
                state: str = "COUNT_COMMITTED", carton_count: int = 1,
                is_auto_confirmed: int = 1) -> Dict[str, Any]:
    requires_review = 0 if state == "COUNT_COMMITTED" else 1
    return store.commit_event(
        {
            "event_id": event_id,
            "global_carton_id": global_cid,
            "crossing_occurrence_id": f"{session_id}:{global_cid}:c1:{direction}",
            "timestamp_sec": ts,
            "commit_timestamp_sec": ts,
            "direction": direction,
            "carton_count": carton_count,
            "state": state,
            "is_auto_confirmed": is_auto_confirmed,
            "is_human_approved": 0,
            "requires_review": requires_review,
            "participating_cameras": ["cam_top"],
            "created_at": ts,
            "updated_at": ts,
        },
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Durable identity & exactly-once commit
# ---------------------------------------------------------------------------
def test_exactly_once_commit_duplicate_replay(tmp_path: Path):
    db = str(tmp_path / "dedup.db")
    store = EventStore(db_path=db)
    r1 = _seed_event(store, "ev_1", "gc_1", 1.0)
    assert r1[0] is True and r1[1] == "COMMITTED_NEW", r1
    # Replay the same crossing (same event_id / occurrence) -> idempotent no-op.
    # Note: exactly-once returns success=False with ALREADY_COMMITTED_IDEMPOTENT
    # for a duplicate replay (it is intentionally NOT a new commit).
    r2 = store.commit_event(
        {
            "event_id": "ev_1",
            "global_carton_id": "gc_1",
            "crossing_occurrence_id": "s1:gc_1:c1:IN",
            "timestamp_sec": 1.0,
            "commit_timestamp_sec": 1.0,
            "direction": "IN",
            "carton_count": 1,
            "state": "COUNT_COMMITTED",
            "is_auto_confirmed": 1,
            "is_human_approved": 0,
            "requires_review": 0,
            "participating_cameras": ["cam_top"],
            "created_at": 1.0,
            "updated_at": 1.0,
        },
        session_id="s1",
    )
    assert r2[0] is False and r2[1] == "ALREADY_COMMITTED_IDEMPOTENT", r2
    # Counts must not double.
    counts = store.get_counts(session_id="s1")
    assert counts.get("confirmed_in", 0) == 1, counts


def test_legitimate_repeat_crossing(tmp_path: Path):
    """Same physical carton crossing again later (new occurrence) is a distinct event."""
    db = str(tmp_path / "repeat.db")
    store = EventStore(db_path=db)
    # First crossing at t=1.0
    r1 = _seed_event(store, "ev_1", "gc_1", 1.0)
    assert r1[0] is True
    # Same carton, same direction, but well outside the 3.5s duplicate window -> new occurrence.
    r2 = store.commit_event(
        {
            "event_id": "ev_1_repeat",
            "global_carton_id": "gc_1",
            "timestamp_sec": 10.0,
            "commit_timestamp_sec": 10.0,
            "direction": "IN",
            "carton_count": 1,
            "state": "REVIEW_REQUIRED",
            "is_auto_confirmed": 0,
            "is_human_approved": 0,
            "requires_review": 1,
            "participating_cameras": ["cam_top"],
            "created_at": 10.0,
            "updated_at": 10.0,
        },
        session_id="s1",
    )
    assert r2[0] is True and r2[1] == "COMMITTED_NEW", r2
    counts = store.get_counts(session_id="s1")
    # ev_1 (auto-confirmed) + ev_1_repeat (review-required) = 2 committed,
    # but only 1 automatic confirmation (repeat crossing is held for review).
    assert counts.get("confirmed_in", 0) == 1, counts
    assert counts.get("review_required_in", 0) >= 1, counts
    assert counts.get("total_committed", 0) == 2, counts


# ---------------------------------------------------------------------------
# Transactional human approval / rejection & concurrency
# ---------------------------------------------------------------------------
def test_concurrent_review_approval_no_double_count(tmp_path: Path):
    db = str(tmp_path / "concurrent.db")
    store = EventStore(db_path=db)
    _seed_event(store, "ev_cc", "gc_cc", 2.0)

    results: Dict[str, Any] = {}
    errors: list = []

    def worker(tag: str) -> None:
        try:
            results[tag] = store.record_manual_review(
                "ev_cc", "CONFIRM_CARTON_CROSSING", f"op_{tag}", "concurrent", None
            )
        except Exception as exc:  # noqa: BLE001
            errors.append((tag, str(exc)))

    threads = [threading.Thread(target=worker, args=(f"t{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    successes = [v for v in results.values() if v and v[0]]
    idempotent = [v for v in results.values() if v and v[1] == "ALREADY_CONFIRMED_IDEMPOTENT"]
    assert len(successes) >= 1, results
    assert len(idempotent) >= 5, results  # exactly one applies, rest are idempotent no-ops

    # Single human-approved count (no double count).
    with store._get_connection() as conn:
        row = conn.execute(
            "SELECT is_human_approved FROM carton_crossing_events WHERE event_id = ?",
            ("ev_cc",),
        ).fetchone()
    assert row is not None and bool(row[0]) is True
    # Audit trail preserved for every attempt.
    log = store.get_review_audit_log("ev_cc")
    assert len(log) >= 6, log


def test_concurrent_commit_lock_contention(tmp_path: Path):
    db = str(tmp_path / "contention.db")
    store = EventStore(db_path=db)

    errors: list = []
    committed = []

    def worker(base: int) -> None:
        try:
            for i in range(10):
                res = _seed_event(
                    store, f"ev_lock_{base}_{i}", f"gc_lock_{base}_{i}",
                    float(base * 10 + i), session_id="s_lock",
                )
                committed.append(res)
        except Exception as exc:  # noqa: BLE001
            errors.append((base, str(exc)))

    threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(committed) == 80, len(committed)
    counts = store.get_counts(session_id="s_lock")
    assert counts.get("confirmed_in", 0) == 80, counts


# ---------------------------------------------------------------------------
# Backup & restore
# ---------------------------------------------------------------------------
def test_backup_and_restore(tmp_path: Path):
    db = str(tmp_path / "live.db")
    backup = str(tmp_path / "backup.db")
    store = EventStore(db_path=db)
    _seed_event(store, "ev_b", "gc_b", 5.0)

    backup_path = store.backup(backup)
    assert os.path.exists(backup_path)

    # Mutate live DB (add another event) then restore from backup.
    _seed_event(store, "ev_b2", "gc_b2", 6.0)
    assert store.get_counts(session_id="s1").get("confirmed_in", 0) == 2

    # Simulate service stopped: restore replaces live DB.
    store.restore(backup)
    counts = store.get_counts(session_id="s1")
    assert counts.get("confirmed_in", 0) == 1, counts
    # The restored DB must still be queryable / not corrupt.
    assert store.is_corrupt() is False


# ---------------------------------------------------------------------------
# Schema migration tracking
# ---------------------------------------------------------------------------
def test_schema_migration_tracking(tmp_path: Path):
    db = str(tmp_path / "migrate.db")
    store = EventStore(db_path=db)

    applied = store.ensure_migration(
        "v1_add_legacy_col",
        "Add legacy notes column",
        "ALTER TABLE carton_crossing_events ADD COLUMN legacy_col TEXT;",
    )
    assert applied is True
    # Idempotent: second call does not re-apply.
    applied_again = store.ensure_migration(
        "v1_add_legacy_col",
        "Add legacy notes column",
        "ALTER TABLE carton_crossing_events ADD COLUMN legacy_col TEXT;",
    )
    assert applied_again is False

    migrations = store.get_applied_migrations()
    versions = [m["version"] for m in migrations]
    assert "v1_add_legacy_col" in versions
    entry = [m for m in migrations if m["version"] == "v1_add_legacy_col"][0]
    assert entry["description"] == "Add legacy notes column"


# ---------------------------------------------------------------------------
# WAL checkpoint management
# ---------------------------------------------------------------------------
def test_wal_checkpoint(tmp_path: Path):
    db = str(tmp_path / "wal.db")
    store = EventStore(db_path=db)
    _seed_event(store, "ev_w", "gc_w", 7.0)
    result = store.wal_checkpoint("TRUNCATE")
    assert isinstance(result, tuple) and len(result) == 3, result


# ---------------------------------------------------------------------------
# Disk-space monitoring
# ---------------------------------------------------------------------------
def test_disk_space_monitoring(tmp_path: Path):
    db = str(tmp_path / "disk.db")
    store = EventStore(db_path=db)
    ok, free, total = store.check_disk_space(min_free_mb=1.0)
    assert ok is True
    assert free > 0 and total > 0
    assert store.disk_free_bytes() == free


# ---------------------------------------------------------------------------
# Corrupt database detection
# ---------------------------------------------------------------------------
def test_corrupt_database_detection(tmp_path: Path):
    # Valid DB is not corrupt.
    db = str(tmp_path / "valid.db")
    store = EventStore(db_path=db)
    assert store.is_corrupt() is False
    assert store.quick_check() == "ok"

    # Corrupt DB: overwrite file with garbage.
    corrupt_db = str(tmp_path / "corrupt.db")
    with open(corrupt_db, "wb") as f:
        f.write(b"NOT A SQLITE DATABASE" * 100)
    corrupt_store = EventStore(db_path=corrupt_db)
    assert corrupt_store.is_corrupt() is True


# ---------------------------------------------------------------------------
# Recovery after restart / unexpected termination
# ---------------------------------------------------------------------------
def test_restart_recovery_persists_state(tmp_path: Path):
    db = str(tmp_path / "restart.db")
    # First "process": commit events.
    store1 = EventStore(db_path=db)
    _seed_event(store1, "ev_r", "gc_r", 9.0)
    store1.record_manual_review(
        "ev_r", "CONFIRM_CARTON_CROSSING", "op_restart", "restart test", None
    )
    counts_before = store1.get_counts(session_id="s1")
    del store1

    # Second "process" (simulating restart / unexpected termination): open same DB.
    store2 = EventStore(db_path=db)
    committed_ids, events, counts = store2.recover_state(session_id="s1")
    assert "gc_r" in committed_ids, committed_ids
    assert any(e["event_id"] == "ev_r" for e in events), events
    assert counts.get("human_approved_in", 0) == 1, counts
    assert counts == counts_before, (counts, counts_before)

    # Audit history survived restart.
    log = store2.get_review_audit_log("ev_r")
    assert any(e["action"] == "CONFIRM_CARTON_CROSSING" for e in log), log
    assert store2.is_corrupt() is False


def test_event_persistence_across_reopen(tmp_path: Path):
    db = str(tmp_path / "persist.db")
    store = EventStore(db_path=db)
    _seed_event(store, "ev_p", "gc_p", 11.0)
    del store
    store2 = EventStore(db_path=db)
    rows = store2.get_events(session_id="s1")
    assert len(rows) == 1
    assert rows[0]["event_id"] == "ev_p"
    assert rows[0]["global_carton_id"] == "gc_p"
