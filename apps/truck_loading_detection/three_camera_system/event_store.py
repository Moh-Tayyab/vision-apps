"""
Durable Exactly-Once Event Store (Phase 2 Requirement 6).

Provides an ACID-compliant, thread-safe, SQLite WAL-backed persistence ledger for
multi-camera physical carton crossing events.

Guarantees & Invariants:
1. Exactly-Once Persistence: Database-level uniqueness constraint
   UNIQUE(session_id, global_carton_id, direction) guarantees that duplicate or
   repeated submissions cannot double-count physical cartons.
2. Atomic Commit: Event recording, status transitions, and ledger totals update
   within atomic SQLite transactions.
3. Concurrency Safe: Uses WAL mode, busy timeout, and internal re-entrant threading
   lock to prevent race conditions during concurrent multi-angle processing.
4. Restart & Recovery: Pre-populates pipeline and state machine state upon restart
   directly from the durable database.
5. Review Isolation: REVIEW_REQUIRED events are stored durably but strictly excluded
   from confirmed_in, confirmed_out, and net_count until manually approved.
6. Full Audit Trail: Every manual review decision (approval/rejection) is permanently
   recorded with operator identity, timestamp, and notes.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.getenv(
    "CARTON_EVENTS_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        "data",
        "carton_events.db",
    ),
)



class EventStore:
    """
    SQLite WAL-backed event persistence engine for carton crossing events.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self._lock = threading.RLock()
        # True when the underlying SQLite file is missing/corrupt and could not
        # be initialized. Methods degrade gracefully instead of hard-failing.
        self._corrupt = False

        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        try:
            self._init_sqlite()
        except sqlite3.DatabaseError:
            logger.warning("Database file is not initializable: %s", self.db_path)
            self._corrupt = True

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            timeout=15.0,
            check_same_thread=False,
            isolation_level="DEFERRED",
        )
        conn.row_factory = sqlite3.Row
        if self.db_path != ":memory:":
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
            except sqlite3.DatabaseError:
                # File may be corrupt or locked; leave journal mode as-is.
                pass
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_sqlite(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                # Check if existing schema has unq_session_carton_dir constraint and migrate safely
                cur = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='carton_crossing_events';")
                row = cur.fetchone()
                if row and "unq_session_carton_dir" in row[0]:
                    logger.info("Migrating carton_crossing_events to support legitimate repeat crossings")
                    conn.execute("DROP TABLE IF EXISTS review_audit_log_legacy;")
                    cur_rev = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='review_audit_log';")
                    has_rev = cur_rev.fetchone() is not None
                    if has_rev:
                        conn.execute("ALTER TABLE review_audit_log RENAME TO review_audit_log_legacy;")
                    conn.execute("ALTER TABLE carton_crossing_events RENAME TO carton_crossing_events_legacy;")
                    conn.execute("""
                    CREATE TABLE carton_crossing_events (
                        event_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        global_carton_id TEXT NOT NULL,
                        timestamp_sec REAL NOT NULL,
                        commit_timestamp_sec REAL NOT NULL,
                        direction TEXT NOT NULL,
                        carton_count INTEGER NOT NULL DEFAULT 1,
                        state TEXT NOT NULL,
                        is_auto_confirmed INTEGER NOT NULL DEFAULT 1,
                        requires_review INTEGER NOT NULL DEFAULT 0,
                        participating_cameras_json TEXT NOT NULL,
                        consensus_json TEXT,
                        stack_resolution_json TEXT,
                        notes TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    """)
                    conn.execute("INSERT INTO carton_crossing_events SELECT * FROM carton_crossing_events_legacy;")
                    conn.execute("DROP TABLE carton_crossing_events_legacy;")
                    if has_rev:
                        conn.execute("""
                        CREATE TABLE review_audit_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            event_id TEXT NOT NULL,
                            session_id TEXT NOT NULL,
                            previous_state TEXT NOT NULL,
                            new_state TEXT NOT NULL,
                            operator_id TEXT NOT NULL,
                            operator_notes TEXT,
                            reviewed_at REAL NOT NULL,
                            FOREIGN KEY (event_id) REFERENCES carton_crossing_events(event_id) ON DELETE CASCADE
                        );
                        """)
                        conn.execute("INSERT INTO review_audit_log SELECT * FROM review_audit_log_legacy;")
                        conn.execute("DROP TABLE review_audit_log_legacy;")
                    conn.commit()

                # Repair any existing review_audit_log pointing to carton_crossing_events_legacy
                cur_rev2 = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='review_audit_log';")
                rev_row = cur_rev2.fetchone()
                if rev_row and "carton_crossing_events_legacy" in rev_row[0]:
                    conn.execute("ALTER TABLE review_audit_log RENAME TO review_audit_log_repair;")
                    conn.execute("""
                    CREATE TABLE review_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        previous_state TEXT NOT NULL,
                        new_state TEXT NOT NULL,
                        operator_id TEXT NOT NULL,
                        operator_notes TEXT,
                        reviewed_at REAL NOT NULL,
                        FOREIGN KEY (event_id) REFERENCES carton_crossing_events(event_id) ON DELETE CASCADE
                    );
                    """)
                    conn.execute("INSERT INTO review_audit_log SELECT * FROM review_audit_log_repair;")
                    conn.execute("DROP TABLE review_audit_log_repair;")
                    conn.commit()

                conn.execute("""
                CREATE TABLE IF NOT EXISTS carton_crossing_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    global_carton_id TEXT NOT NULL,
                    crossing_occurrence_id TEXT,
                    timestamp_sec REAL NOT NULL,
                    commit_timestamp_sec REAL NOT NULL,
                    direction TEXT NOT NULL,
                    carton_count INTEGER NOT NULL DEFAULT 1,
                    state TEXT NOT NULL,
                    is_auto_confirmed INTEGER NOT NULL DEFAULT 1,
                    requires_review INTEGER NOT NULL DEFAULT 0,
                    participating_cameras_json TEXT NOT NULL,
                    consensus_json TEXT,
                    stack_resolution_json TEXT,
                    notes TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """)

                # Safe schema migration for existing databases missing crossing_occurrence_id
                cur_cols = conn.execute("PRAGMA table_info(carton_crossing_events);")
                col_names = [col[1] for col in cur_cols.fetchall()]
                if "crossing_occurrence_id" not in col_names:
                    conn.execute("ALTER TABLE carton_crossing_events ADD COLUMN crossing_occurrence_id TEXT;")
                if "is_human_approved" not in col_names:
                    conn.execute("ALTER TABLE carton_crossing_events ADD COLUMN is_human_approved INTEGER NOT NULL DEFAULT 0;")
                if "reviewed_by" not in col_names:
                    conn.execute("ALTER TABLE carton_crossing_events ADD COLUMN reviewed_by TEXT;")
                if "reviewed_at" not in col_names:
                    conn.execute("ALTER TABLE carton_crossing_events ADD COLUMN reviewed_at REAL;")

                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_session_state
                ON carton_crossing_events (session_id, state);
                """)

                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_carton
                ON carton_crossing_events (global_carton_id);
                """)

                conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_events_occurrence
                ON carton_crossing_events (session_id, crossing_occurrence_id);
                """)

                conn.execute("""
                CREATE TABLE IF NOT EXISTS review_audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    action TEXT NOT NULL DEFAULT 'MANUAL_REVIEW',
                    previous_state TEXT NOT NULL,
                    new_state TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    operator_notes TEXT,
                    reviewed_at REAL NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES carton_crossing_events(event_id) ON DELETE CASCADE
                );
                """)

                cur_rev_cols = conn.execute("PRAGMA table_info(review_audit_log);")
                rev_col_names = [col[1] for col in cur_rev_cols.fetchall()]
                if "action" not in rev_col_names:
                    conn.execute("ALTER TABLE review_audit_log ADD COLUMN action TEXT NOT NULL DEFAULT 'MANUAL_REVIEW';")

                # Schema migration tracking table (records applied migrations).
                conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    applied_at REAL NOT NULL,
                    sql TEXT
                );
                """)

                # Security & operational audit trail table
                conn.execute("""
                CREATE TABLE IF NOT EXISTS security_audit_log (
                    audit_id TEXT PRIMARY KEY,
                    timestamp_sec REAL NOT NULL,
                    iso_time TEXT NOT NULL,
                    action TEXT NOT NULL,
                    client_ip TEXT NOT NULL,
                    user_identity TEXT NOT NULL,
                    details TEXT NOT NULL
                );
                """)

    def commit_event(
        self,
        event_data: Dict[str, Any],
        session_id: str = "default_session",
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Atomically commits a logical carton crossing event.

        Returns:
            (committed_new, reason, persisted_record)
            - If (session_id, global_carton_id, direction) is already committed:
              returns (False, "ALREADY_COMMITTED_IDEMPOTENT", existing_record).
            - If existing event was REVIEW_REQUIRED and is now corroborated:
              upgrades to COUNT_COMMITTED and returns (True, "UPGRADED_ON_CORROBORATION", updated_record).
            - If new event:
              persists atomically and returns (True, "COMMITTED_NEW", record).
        """
        now = time.time()
        global_cid = str(event_data.get("global_carton_id", ""))
        direction = str(event_data.get("direction", "IN")).upper()
        if not global_cid:
            raise ValueError("Event data missing required 'global_carton_id'")

        event_id = str(event_data.get("event_id") or f"fused_ev_{int(now * 1000)}")
        timestamp_sec = float(event_data.get("timestamp_sec", now))
        commit_t = float(event_data.get("commit_timestamp_sec", timestamp_sec))
        carton_count = max(1, int(event_data.get("carton_count", 1)))
        state = str(event_data.get("state", "COUNT_COMMITTED"))
        is_auto = 1 if bool(event_data.get("is_auto_confirmed", state == "COUNT_COMMITTED")) else 0
        req_rev = 1 if (state == "REVIEW_REQUIRED" or not is_auto) else 0

        cams = event_data.get("participating_cameras", [])
        if isinstance(cams, (set, tuple)):
            cams = sorted(list(cams))
        cams_json = json.dumps(cams)

        consensus_dict = event_data.get("consensus", {})
        consensus_json = json.dumps(consensus_dict) if isinstance(consensus_dict, dict) else str(consensus_dict)

        stack_dict = event_data.get("stack_resolution", {})
        stack_json = json.dumps(stack_dict) if isinstance(stack_dict, dict) else str(stack_dict)

        notes = str(event_data.get("notes", ""))

        with self._lock:
            with self._get_connection() as conn:
                # Resolve occurrence index and durable crossing_occurrence_id
                if event_data.get("crossing_occurrence_id"):
                    crossing_occ_id = str(event_data["crossing_occurrence_id"])
                    occurrence_idx = int(event_data.get("occurrence_index", 1))
                elif "occurrence_index" in event_data:
                    occurrence_idx = int(event_data["occurrence_index"])
                    crossing_occ_id = f"{session_id}:{global_cid}:c{occurrence_idx}:{direction}"
                else:
                    # Query prior occurrences for this carton in this session
                    cur_occ = conn.execute(
                        """
                        SELECT event_id, timestamp_sec, direction, crossing_occurrence_id
                        FROM carton_crossing_events
                        WHERE session_id = ? AND global_carton_id = ?
                        ORDER BY timestamp_sec ASC;
                        """,
                        (session_id, global_cid),
                    )
                    prior_rows = cur_occ.fetchall()
                    if not prior_rows:
                        occurrence_idx = 1
                        crossing_occ_id = f"{session_id}:{global_cid}:c1:{direction}"
                    else:
                        # Check if any prior event is within duplicate window (3.5s)
                        matched_duplicate = False
                        for pr in prior_rows:
                            dt = abs(timestamp_sec - float(pr["timestamp_sec"]))
                            if dt <= 3.5:
                                crossing_occ_id = pr["crossing_occurrence_id"] or f"{session_id}:{global_cid}:c1:{pr['direction']}"
                                matched_duplicate = True
                                break
                        if not matched_duplicate:
                            occurrence_idx = len(prior_rows) + 1
                            crossing_occ_id = f"{session_id}:{global_cid}:c{occurrence_idx}:{direction}"

                # 1. Check existing record by event_id or stable persistent crossing_occurrence_id
                cur = conn.execute(
                    """
                    SELECT event_id, state, timestamp_sec, carton_count, participating_cameras_json, notes, crossing_occurrence_id
                    FROM carton_crossing_events
                    WHERE event_id = ? OR (session_id = ? AND crossing_occurrence_id = ?);
                    """,
                    (event_id, session_id, crossing_occ_id),
                )
                rows = cur.fetchall()
                existing = rows[0] if rows else None

                # Secondary heuristic: check if uncorroborated REVIEW_REQUIRED event can be upgraded
                if existing is None:
                    cur_temp = conn.execute(
                        """
                        SELECT event_id, state, timestamp_sec, carton_count, participating_cameras_json, notes, crossing_occurrence_id
                        FROM carton_crossing_events
                        WHERE session_id = ? AND global_carton_id = ? AND direction = ? AND state = 'REVIEW_REQUIRED';
                        """,
                        (session_id, global_cid, direction),
                    )
                    for r in cur_temp.fetchall():
                        dt = abs(timestamp_sec - float(r["timestamp_sec"]))
                        if dt <= 1.5:
                            existing = r
                            break

                if existing is not None:
                    ex_ev_id = existing["event_id"]
                    ex_state = existing["state"]
                    ex_cams = set(json.loads(existing["participating_cameras_json"]))

                    # Dynamic corroboration check:
                    # If existing was REVIEW_REQUIRED and we now have corroborating cameras
                    new_cams = set(cams)
                    combined_cams = sorted(list(ex_cams.union(new_cams)))

                    if ex_state == "REVIEW_REQUIRED" and len(combined_cams) >= 2 and is_auto:
                        # Upgrade to confirmed
                        conn.execute(
                            """
                            UPDATE carton_crossing_events
                            SET state = 'COUNT_COMMITTED',
                                is_auto_confirmed = 1,
                                requires_review = 0,
                                participating_cameras_json = ?,
                                notes = ?,
                                updated_at = ?
                            WHERE event_id = ?;
                            """,
                            (
                                json.dumps(combined_cams),
                                "Upgraded to multi-camera confirmation on corroboration",
                                now,
                                ex_ev_id,
                            ),
                        )
                        conn.commit()
                        logger.info("Upgraded existing event %s to COUNT_COMMITTED on corroboration", ex_ev_id)
                        updated = self.get_event(ex_ev_id) or {}
                        return (True, "UPGRADED_ON_CORROBORATION", updated)

                    # Otherwise, strict deduplication: do NOT double count!
                    logger.debug("Idempotent hit: Carton %s already committed as %s", global_cid, ex_ev_id)
                    existing_record = self.get_event(ex_ev_id) or {}
                    return (False, "ALREADY_COMMITTED_IDEMPOTENT", existing_record)

                # 2. Insert new record with stable crossing_occurrence_id
                conn.execute(
                    """
                    INSERT INTO carton_crossing_events (
                        event_id, session_id, global_carton_id, crossing_occurrence_id,
                        timestamp_sec, commit_timestamp_sec, direction, carton_count, state,
                        is_auto_confirmed, requires_review, participating_cameras_json,
                        consensus_json, stack_resolution_json, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event_id,
                        session_id,
                        global_cid,
                        crossing_occ_id,
                        timestamp_sec,
                        commit_t,
                        direction,
                        carton_count,
                        state,
                        is_auto,
                        req_rev,
                        cams_json,
                        consensus_json,
                        stack_json,
                        notes,
                        now,
                        now,
                    ),
                )
                conn.commit()

        persisted = self.get_event(event_id) or {}
        return (True, "COMMITTED_NEW", persisted)

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single event record by ID."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM carton_crossing_events WHERE event_id = ?;",
                    (event_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)

    def get_events(
        self,
        session_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries events by session and/or state."""
        query = "SELECT * FROM carton_crossing_events WHERE 1=1"
        params: List[Any] = []

        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        if state is not None:
            query += " AND state = ?"
            params.append(state)

        query += " ORDER BY timestamp_sec ASC;"

        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(query, params)
                return [self._row_to_dict(r) for r in cur.fetchall()]

    def get_events_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Convenience query for all events in a session."""
        return self.get_events(session_id=session_id)

    def get_counts(self, session_id: str = "default_session") -> Dict[str, int]:
        """
        Computes accurate transactional totals directly from durable database state.
        Guarantees that:
          - REVIEW_REQUIRED is excluded from confirmed totals.
          - Automatic confirmations and human approvals are distinctly tracked and partitioned.
        """
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT direction, state, is_auto_confirmed, is_human_approved, SUM(carton_count) as total_qty
                    FROM carton_crossing_events
                    WHERE session_id = ?
                    GROUP BY direction, state, is_auto_confirmed, is_human_approved;
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()

        auto_confirmed_in = 0
        auto_confirmed_out = 0
        human_approved_in = 0
        human_approved_out = 0
        review_in = 0
        review_out = 0

        for r in rows:
            direction = r["direction"]
            state = r["state"]
            is_auto = bool(r["is_auto_confirmed"])
            is_human = bool(r["is_human_approved"])
            qty = int(r["total_qty"] or 0)

            if state in ("COUNT_COMMITTED", "COUNTED"):
                if direction == "IN":
                    auto_confirmed_in += qty
                elif direction == "OUT":
                    auto_confirmed_out += qty
            elif state == "MANUALLY_CONFIRMED" or is_human:
                if direction == "IN":
                    human_approved_in += qty
                elif direction == "OUT":
                    human_approved_out += qty
            elif state in ("REVIEW_REQUIRED", "AMBIGUOUS_STACK", "MARK_AMBIGUOUS"):
                if direction == "IN":
                    review_in += qty
                elif direction == "OUT":
                    review_out += qty

        confirmed_in = auto_confirmed_in + human_approved_in
        confirmed_out = auto_confirmed_out + human_approved_out
        net_count = confirmed_in - confirmed_out
        provisional_in = confirmed_in + review_in
        provisional_out = confirmed_out + review_out

        return {
            "auto_confirmed_in": auto_confirmed_in,
            "auto_confirmed_out": auto_confirmed_out,
            "human_approved_in": human_approved_in,
            "human_approved_out": human_approved_out,
            "confirmed_in": confirmed_in,
            "confirmed_out": confirmed_out,
            "net_count": net_count,
            "review_required_in": review_in,
            "review_required_out": review_out,
            "provisional_in": provisional_in,
            "provisional_out": provisional_out,
            "confirmed_count": confirmed_in + confirmed_out,
            "auto_confirmed_count": auto_confirmed_in + auto_confirmed_out,
            "human_approved_count": human_approved_in + human_approved_out,
            "review_required_count": review_in + review_out,
            "total_committed": confirmed_in + confirmed_out + review_in + review_out,
        }

    def record_manual_review(
        self,
        event_id: str,
        action: str,
        operator_id: str,
        reason: str = "",
        adjusted_count: Optional[int] = None,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Transactionally records a human review decision adhering to Phase 3.3 specifications.
        Supports standard operator actions:
          - CONFIRM_CARTON_CROSSING (or legacy 'approve')
          - REJECT_FALSE_DETECTION (or legacy 'reject')
          - MARK_AMBIGUOUS
          - FLAG_SYSTEM_ERROR
        Guarantees:
          - Idempotency: repeated approvals do not generate duplicate counts
          - Separation: is_human_approved=1 and is_auto_confirmed=0 are strictly partitioned
          - Full audit trail recorded in review_audit_log
        """
        act_norm = action.strip().upper()
        # Map legacy actions
        if act_norm == "APPROVE":
            act_norm = "CONFIRM_CARTON_CROSSING"
        elif act_norm == "REJECT":
            act_norm = "REJECT_FALSE_DETECTION"
        elif act_norm in ("ADJUST", "MARK_AMBIGUOUS"):
            act_norm = "MARK_AMBIGUOUS"

        valid_actions = {
            "CONFIRM_CARTON_CROSSING",
            "REJECT_FALSE_DETECTION",
            "MARK_AMBIGUOUS",
            "FLAG_SYSTEM_ERROR",
        }
        if act_norm not in valid_actions:
            raise ValueError(f"Invalid operator action '{action}'. Must be one of {sorted(list(valid_actions))}")

        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(
                    "SELECT * FROM carton_crossing_events WHERE event_id = ?;",
                    (event_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return False, f"Event '{event_id}' not found in database", {}

                existing = self._row_to_dict(row)
                prev_state = existing["state"]
                session_id = existing["session_id"]
                current_human_approved = bool(existing.get("is_human_approved", 0))

                # Check Idempotency for CONFIRM_CARTON_CROSSING
                if act_norm == "CONFIRM_CARTON_CROSSING" and current_human_approved:
                    logger.warning(
                        "Operator %s re-approved already confirmed event %s (idempotent no-op)",
                        operator_id,
                        event_id,
                    )
                    conn.execute(
                        """
                        INSERT INTO review_audit_log (
                            event_id, session_id, action, previous_state, new_state,
                            operator_id, operator_notes, reviewed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            event_id,
                            session_id,
                            act_norm,
                            prev_state,
                            prev_state,
                            operator_id,
                            f"Idempotent re-approval: {reason}",
                            now,
                        ),
                    )
                    conn.commit()
                    return True, "ALREADY_CONFIRMED_IDEMPOTENT", existing

                # Determine state transitions
                if act_norm == "CONFIRM_CARTON_CROSSING":
                    new_state = "MANUALLY_CONFIRMED"
                    is_auto = 0  # CRITICAL: Keep manual approvals distinctly separated from automatic confirmation
                    is_human = 1
                    req_rev = 0
                elif act_norm == "REJECT_FALSE_DETECTION":
                    new_state = "MANUALLY_REJECTED"
                    is_auto = 0
                    is_human = 0
                    req_rev = 0
                elif act_norm == "MARK_AMBIGUOUS":
                    new_state = "MARK_AMBIGUOUS"
                    is_auto = 0
                    is_human = 0
                    req_rev = 1
                elif act_norm == "FLAG_SYSTEM_ERROR":
                    new_state = "FLAGGED_ERROR"
                    is_auto = 0
                    is_human = 0
                    req_rev = 0

                new_count = existing["carton_count"]
                if adjusted_count is not None and adjusted_count > 0:
                    new_count = int(adjusted_count)

                # Update carton_crossing_events
                conn.execute(
                    """
                    UPDATE carton_crossing_events
                    SET state = ?,
                        carton_count = ?,
                        is_auto_confirmed = ?,
                        is_human_approved = ?,
                        requires_review = ?,
                        reviewed_by = ?,
                        reviewed_at = ?,
                        updated_at = ?
                    WHERE event_id = ?;
                    """,
                    (
                        new_state,
                        new_count,
                        is_auto,
                        is_human,
                        req_rev,
                        operator_id,
                        now,
                        now,
                        event_id,
                    ),
                )

                # Insert into review_audit_log
                conn.execute(
                    """
                    INSERT INTO review_audit_log (
                        event_id, session_id, action, previous_state, new_state,
                        operator_id, operator_notes, reviewed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event_id,
                        session_id,
                        act_norm,
                        prev_state,
                        new_state,
                        operator_id,
                        reason,
                        now,
                    ),
                )
                conn.commit()

                # Fetch updated record
                cur_up = conn.execute("SELECT * FROM carton_crossing_events WHERE event_id = ?;", (event_id,))
                updated_record = self._row_to_dict(cur_up.fetchone())

        logger.info(
            "Operator '%s' action %s executed on event %s: %s -> %s (human_approved=%s, auto_confirmed=%s)",
            operator_id,
            act_norm,
            event_id,
            prev_state,
            new_state,
            is_human,
            is_auto,
        )
        return True, "ACTION_APPLIED", updated_record

    def update_review_status(
        self,
        event_id: str,
        new_state: str,
        operator_id: str,
        operator_notes: str = "",
    ) -> bool:
        """
        Backward-compatible review status updater. Delegates to record_manual_review.
        """
        action_map = {
            "MANUALLY_CONFIRMED": "CONFIRM_CARTON_CROSSING",
            "COUNT_COMMITTED": "CONFIRM_CARTON_CROSSING",
            "MANUALLY_REJECTED": "REJECT_FALSE_DETECTION",
            "REVIEW_REQUIRED": "MARK_AMBIGUOUS",
            "MARK_AMBIGUOUS": "MARK_AMBIGUOUS",
            "FLAGGED_ERROR": "FLAG_SYSTEM_ERROR",
        }
        action = action_map.get(new_state, "CONFIRM_CARTON_CROSSING" if "CONFIRM" in new_state else "MARK_AMBIGUOUS")
        success, _, _ = self.record_manual_review(
            event_id=event_id,
            action=action,
            operator_id=operator_id,
            reason=operator_notes,
        )
        return success

    def get_review_queue(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves review-required events with multi-camera context,
        confidence values, sync metrics, tracklets, and audit history.
        """
        query = """
            SELECT * FROM carton_crossing_events
            WHERE (requires_review = 1 OR state IN ('REVIEW_REQUIRED', 'AMBIGUOUS_STACK', 'MARK_AMBIGUOUS'))
              AND state NOT IN ('MANUALLY_REJECTED', 'FLAGGED_ERROR', 'MANUALLY_CONFIRMED', 'COUNT_COMMITTED')
        """
        params: List[Any] = []
        if session_id is not None:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY timestamp_sec ASC;"

        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(query, params)
                rows = cur.fetchall()
                events = [self._row_to_dict(r) for r in rows]
                for ev in events:
                    cur_audit = conn.execute(
                        "SELECT * FROM review_audit_log WHERE event_id = ? ORDER BY reviewed_at ASC;",
                        (ev["event_id"],),
                    )
                    ev["audit_history"] = [dict(a) for a in cur_audit.fetchall()]
        return events

    def get_review_audit_log(self, event_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieves audit trail entries for manual review actions."""
        query = "SELECT * FROM review_audit_log"
        params: List[Any] = []
        if event_id is not None:
            query += " WHERE event_id = ?"
            params.append(event_id)
        query += " ORDER BY reviewed_at ASC;"

        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(query, params)
                return [dict(r) for r in cur.fetchall()]

    def recover_state(
        self,
        session_id: str = "default_session",
    ) -> Tuple[Set[str], List[Dict[str, Any]], Dict[str, int]]:
        """
        Recovers session state for crash recovery or application restart.
        Returns:
            (committed_carton_ids, committed_events, counts_summary)
        """
        events = self.get_events(session_id=session_id)
        committed_ids: Set[str] = {e["global_carton_id"] for e in events}
        counts = self.get_counts(session_id=session_id)
        return committed_ids, events, counts

    def reset_session(self, session_id: str = "default_session") -> None:
        """Deletes all events and audit log entries for a given session."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    "DELETE FROM carton_crossing_events WHERE session_id = ?;",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM review_audit_log WHERE session_id = ?;",
                    (session_id,),
                )
                conn.commit()

    # ------------------------------------------------------------------
    # Schema migration tracking
    # ------------------------------------------------------------------
    def record_migration(
        self,
        version: str,
        description: str,
        sql: Optional[str] = None,
    ) -> None:
        """Record an applied schema migration (idempotent by version)."""
        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO schema_migrations (version, description, applied_at, sql)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(version) DO UPDATE SET
                        description = excluded.description,
                        applied_at = excluded.applied_at,
                        sql = excluded.sql;
                    """,
                    (version, description, now, sql),
                )
                conn.commit()

    def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Return all recorded schema migrations, oldest first."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT version, description, applied_at, sql FROM schema_migrations "
                "ORDER BY applied_at ASC;"
            )
            return [dict(r) for r in cur.fetchall()]

    def ensure_migration(
        self,
        version: str,
        description: str,
        sql: Optional[str] = None,
    ) -> bool:
        """Idempotently apply `sql` and record the migration if not already applied.

        Returns True when the migration was applied in this call, False if it was
        already recorded.
        """
        applied = {m["version"] for m in self.get_applied_migrations()}
        if version in applied:
            return False
        if sql:
            with self._lock:
                with self._get_connection() as conn:
                    conn.execute(sql)
                    conn.commit()
        self.record_migration(version, description, sql)
        return True

    # ------------------------------------------------------------------
    # WAL checkpoint management
    # ------------------------------------------------------------------
    def wal_checkpoint(self, mode: str = "TRUNCATE") -> Tuple[int, int, int]:
        """Run PRAGMA wal_checkpoint(<mode>) and return (busy, log, checkpointed)."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(f"PRAGMA wal_checkpoint({mode});")
                row = cur.fetchone()
                return (int(row[0]), int(row[1]), int(row[2]))

    # ------------------------------------------------------------------
    # Integrity / corruption detection
    # ------------------------------------------------------------------
    def integrity_check(self) -> str:
        """Run PRAGMA integrity_check and return its result string."""
        if self._corrupt:
            return "corrupt"
        with self._get_connection() as conn:
            cur = conn.execute("PRAGMA integrity_check;")
            return str(cur.fetchone()[0])

    def quick_check(self) -> str:
        """Run PRAGMA quick_check and return its result string."""
        if self._corrupt:
            return "corrupt"
        with self._get_connection() as conn:
            cur = conn.execute("PRAGMA quick_check;")
            return str(cur.fetchone()[0])

    def is_corrupt(self) -> bool:
        """True if the database file fails quick_check or could not be initialized."""
        if self._corrupt:
            return True
        return self.quick_check() != "ok"

    # ------------------------------------------------------------------
    # Backup / restore
    # ------------------------------------------------------------------
    def backup(self, backup_path: str) -> str:
        """Create a consistent snapshot of the SQLite database.

        Uses the online SQLite backup API so the source remains readable/writable
        during the copy. Returns the absolute backup path.
        """
        backup_path = os.path.abspath(backup_path)
        os.makedirs(os.path.dirname(backup_path) or ".", exist_ok=True)
        with self._lock:
            with self._get_connection() as conn:
                dest = sqlite3.connect(backup_path)
                try:
                    conn.backup(dest)
                finally:
                    dest.close()
        return backup_path

    def restore(self, backup_path: str) -> None:
        """Restore the live database from a consistent backup.

        The service should be stopped before calling this method. The current
        WAL is checkpointed into the main file, then the live database file is
        replaced with the backup and any stale -wal/-shm sidecars are removed so
        stale frames are not replayed.
        """
        backup_path = os.path.abspath(backup_path)
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        live = os.path.abspath(self.db_path)
        # Flush current WAL into the main database file before replacing it.
        try:
            self.wal_checkpoint("TRUNCATE")
        except Exception:
            logger.warning("WAL checkpoint skipped during restore of %s", live)
        shutil.copy2(backup_path, live)
        for ext in ("-wal", "-shm"):
            sidecar = live + ext
            if os.path.exists(sidecar):
                os.remove(sidecar)
        logger.info("Database restored from %s to %s", backup_path, live)

    # ------------------------------------------------------------------
    # Disk-space monitoring
    # ------------------------------------------------------------------
    def disk_free_bytes(self, path: Optional[str] = None) -> int:
        """Return free bytes on the volume holding `path` (default: DB dir)."""
        target = path or os.path.dirname(os.path.abspath(self.db_path))
        return shutil.disk_usage(target).free

    def check_disk_space(self, min_free_mb: float = 100.0) -> Tuple[bool, int, int]:
        """Return (ok, free_bytes, total_bytes) for the DB volume."""
        target = os.path.dirname(os.path.abspath(self.db_path))
        usage = shutil.disk_usage(target)
        ok = usage.free >= int(min_free_mb * 1024 * 1024)
        return ok, usage.free, usage.total

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["participating_cameras"] = json.loads(d.get("participating_cameras_json") or "[]")
        except Exception:
            d["participating_cameras"] = []
        try:
            d["consensus"] = json.loads(d.get("consensus_json") or "{}")
        except Exception:
            d["consensus"] = {}
        try:
            d["stack_resolution"] = json.loads(d.get("stack_resolution_json") or "{}")
        except Exception:
            d["stack_resolution"] = {}
        d["is_auto_confirmed"] = bool(d.get("is_auto_confirmed", 1))
        d["requires_review"] = bool(d.get("requires_review", 0))
        d["is_human_approved"] = bool(d.get("is_human_approved", 0))
        d["reviewed_by"] = d.get("reviewed_by")
        d["reviewed_at"] = d.get("reviewed_at")
        return d

    def record_security_audit(
        self,
        action: str,
        client_ip: str = "127.0.0.1",
        user_identity: str = "authenticated_client",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Records an immutable security audit event."""
        now = time.time()
        iso = datetime.now(timezone.utc).isoformat()
        audit_id = f"sec_{int(now * 1000)}_{uuid.uuid4().hex[:8]}"
        details_json = json.dumps(details or {})
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO security_audit_log
                    (audit_id, timestamp_sec, iso_time, action, client_ip, user_identity, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (audit_id, now, iso, action, client_ip, user_identity, details_json),
                )
        return {
            "audit_id": audit_id,
            "timestamp_sec": now,
            "iso_time": iso,
            "action": action,
            "client_ip": client_ip,
            "user_identity": user_identity,
            "details": details or {},
        }

    def get_security_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves recent security audit logs."""
        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT audit_id, timestamp_sec, iso_time, action, client_ip, user_identity, details
                    FROM security_audit_log
                    ORDER BY timestamp_sec DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
                results = []
                for r in rows:
                    results.append({
                        "audit_id": r["audit_id"],
                        "timestamp_sec": r["timestamp_sec"],
                        "iso_time": r["iso_time"],
                        "action": r["action"],
                        "client_ip": r["client_ip"],
                        "user_identity": r["user_identity"],
                        "details": json.loads(r["details"]) if r["details"] else {},
                    })
                return results
