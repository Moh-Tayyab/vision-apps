"""Database & Vector Persistence Layer for Face Authorization (App 3).

Supports:
- SQLite WAL (Write-Ahead-Logging) for concurrent, crash-safe relational storage.
- Qdrant Vector Engine (Self-hosted Docker or Cloud) with HNSW indexing and Cosine distance.
- Embedded NumPy vector search fallback if Qdrant container is not reachable.
- Persistent audit logs, person registry, and configuration store.
- Automatic migration from legacy embeddings.json.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("face_auth.db")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "face_embeddings")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "128"))  # Facenet is 128, Facenet512 is 512


def _vector_to_blob(vec: List[float] | np.ndarray) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


def _blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class DatabaseManager:
    """Manages SQLite WAL persistence and Qdrant vector index synchronization."""

    def __init__(self, db_path: str, qdrant_url: Optional[str] = None):
        self.db_path = db_path
        self.qdrant_url = qdrant_url or QDRANT_URL
        self._lock = threading.Lock()
        self._qdrant_client = None
        self._qdrant_available = False

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_sqlite()
        self._init_qdrant()
        self._migrate_legacy_json()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_sqlite(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    photo_path TEXT,
                    notes TEXT
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    person_name TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    dim INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
                );
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    camera_id TEXT NOT NULL,
                    status TEXT NOT NULL, -- authorized | unauthorized | spoof | unknown
                    matched_name TEXT,
                    confidence REAL,
                    distance REAL,
                    liveness_score REAL,
                    bbox_json TEXT,
                    snapshot_path TEXT
                );
                """)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp DESC);
                """)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_events(status);
                """)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """)
                conn.commit()

    def _init_qdrant(self) -> None:
        """Attempt connection to self-hosted or cloud Qdrant vector database."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models

            client = QdrantClient(url=self.qdrant_url, api_key=QDRANT_API_KEY, timeout=2.0)
            collections = [c.name for c in client.get_collections().collections]
            if QDRANT_COLLECTION not in collections:
                client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=models.VectorParams(
                        size=VECTOR_DIM,
                        distance=models.Distance.COSINE,
                    ),
                )
            self._qdrant_client = client
            self._qdrant_available = True
            logger.info(f"Connected to Qdrant at {self.qdrant_url} (collection: {QDRANT_COLLECTION})")
        except Exception as e:
            self._qdrant_available = False
            self._qdrant_client = None
            logger.warning(f"Qdrant unavailable at {self.qdrant_url} ({e}). Using embedded SQLite+NumPy vector search.")

    def _migrate_legacy_json(self) -> None:
        """Auto-import embeddings from old embeddings.json if present."""
        json_path = os.path.join(os.path.dirname(self.db_path), "embeddings.json")
        if not os.path.exists(json_path):
            return

        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute("SELECT COUNT(*) as cnt FROM persons;")
                if cur.fetchone()["cnt"] > 0:
                    return  # already populated

                try:
                    with open(json_path, "r") as f:
                        data = json.load(f)

                    for name, p_data in data.items():
                        embs = p_data.get("embeddings", [])
                        if not embs:
                            continue
                        p_id = str(uuid.uuid4())
                        now = float(p_data.get("updated_at", time.time()))
                        photo_file = os.path.join(os.path.dirname(self.db_path), "photos", name, "photo.jpg")
                        photo_path = photo_file if os.path.exists(photo_file) else None

                        conn.execute(
                            "INSERT OR IGNORE INTO persons (id, name, created_at, updated_at, photo_path) VALUES (?, ?, ?, ?, ?);",
                            (p_id, name, now, now, photo_path),
                        )
                        for vec in embs:
                            e_id = str(uuid.uuid4())
                            conn.execute(
                                "INSERT INTO embeddings (id, person_id, person_name, vector, dim, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                                (e_id, p_id, name, _vector_to_blob(vec), len(vec), now),
                            )
                    conn.commit()
                    logger.info(f"Successfully migrated {len(data)} person(s) from {json_path} to SQLite.")
                except Exception as e:
                    logger.warning(f"Failed to migrate legacy JSON: {e}")

    # ---------------- Person & Embedding CRUD ----------------

    def enroll_person(self, name: str, embeddings: List[List[float]], photo_path: Optional[str] = None) -> dict:
        """Enroll or add embeddings for an authorized person."""
        if not embeddings:
            raise ValueError("No embeddings provided for enrollment")

        now = time.time()
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT id FROM persons WHERE name = ?;", (name,)).fetchone()
                if row:
                    person_id = row["id"]
                    conn.execute(
                        "UPDATE persons SET updated_at = ?, photo_path = COALESCE(?, photo_path) WHERE id = ?;",
                        (now, photo_path, person_id),
                    )
                else:
                    person_id = str(uuid.uuid4())
                    conn.execute(
                        "INSERT INTO persons (id, name, created_at, updated_at, photo_path) VALUES (?, ?, ?, ?, ?);",
                        (person_id, name, now, now, photo_path),
                    )

                added_ids = []
                qdrant_points = []
                for vec in embeddings:
                    e_id = str(uuid.uuid4())
                    added_ids.append(e_id)
                    conn.execute(
                        "INSERT INTO embeddings (id, person_id, person_name, vector, dim, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                        (e_id, person_id, name, _vector_to_blob(vec), len(vec), now),
                    )
                    if self._qdrant_available and self._qdrant_client is not None:
                        try:
                            from qdrant_client.http import models
                            qdrant_points.append(
                                models.PointStruct(
                                    id=e_id,
                                    vector=list(vec),
                                    payload={"person_id": person_id, "name": name},
                                )
                            )
                        except Exception:
                            pass

                conn.commit()

                # Sync to Qdrant
                if qdrant_points and self._qdrant_available and self._qdrant_client is not None:
                    try:
                        self._qdrant_client.upsert(
                            collection_name=QDRANT_COLLECTION,
                            points=qdrant_points,
                        )
                    except Exception as e:
                        logger.warning(f"Qdrant sync failed: {e}")

                cur = conn.execute("SELECT COUNT(*) as total FROM embeddings WHERE person_id = ?;", (person_id,))
                total_embs = cur.fetchone()["total"]

        return {
            "name": name,
            "person_id": person_id,
            "new_embeddings": len(embeddings),
            "total_embeddings": total_embs,
        }

    def delete_person(self, name: str) -> bool:
        """Delete person and associated embeddings from DB and Qdrant."""
        with self._lock:
            with self._get_connection() as conn:
                row = conn.execute("SELECT id FROM persons WHERE name = ?;", (name,)).fetchone()
                if not row:
                    return False
                person_id = row["id"]
                conn.execute("DELETE FROM embeddings WHERE person_id = ?;", (person_id,))
                conn.execute("DELETE FROM persons WHERE id = ?;", (person_id,))
                conn.commit()

            if self._qdrant_available and self._qdrant_client is not None:
                try:
                    from qdrant_client.http import models
                    self._qdrant_client.delete(
                        collection_name=QDRANT_COLLECTION,
                        points_selector=models.FilterSelector(
                            filter=models.Filter(
                                must=[models.FieldCondition(key="name", match=models.MatchValue(value=name))]
                            )
                        ),
                    )
                except Exception as e:
                    logger.warning(f"Qdrant delete error: {e}")

        return True

    def list_persons(self) -> List[dict]:
        """List all enrolled persons with embedding counts and photo availability."""
        with self._get_connection() as conn:
            query = """
            SELECT p.id, p.name, p.created_at, p.updated_at, p.photo_path,
                   COUNT(e.id) as num_embeddings
            FROM persons p
            LEFT JOIN embeddings e ON p.id = e.person_id
            GROUP BY p.id
            ORDER BY p.name ASC;
            """
            rows = conn.execute(query).fetchall()
            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "num_embeddings": r["num_embeddings"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "has_photo": bool(r["photo_path"] and os.path.exists(r["photo_path"])),
                    "photo_path": r["photo_path"],
                }
                for r in rows
            ]

    def get_person_photo_path(self, name: str) -> Optional[str]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT photo_path FROM persons WHERE name = ?;", (name,)).fetchone()
            if row and row["photo_path"] and os.path.exists(row["photo_path"]):
                return row["photo_path"]
        return None

    def get_all_embeddings_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Return (names_list, embeddings_matrix) for local vector search."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT person_name, vector FROM embeddings;").fetchall()
            if not rows:
                return [], np.empty((0, 0), dtype=np.float32)

            names = [r["person_name"] for r in rows]
            vectors = [_blob_to_vector(r["vector"]) for r in rows]
            matrix = np.vstack(vectors).astype(np.float32)
            return names, matrix

    # ---------------- Vector Search ----------------

    def search_face(self, query_vector: np.ndarray, threshold: float = 0.48) -> Optional[dict]:
        """Match query face embedding against stored vectors.

        Uses Qdrant if available; otherwise performs vectorized NumPy cosine matching.
        """
        query_vec = np.asarray(query_vector, dtype=np.float32).flatten()
        if query_vec.size == 0:
            return None

        # Try Qdrant search
        if self._qdrant_available and self._qdrant_client is not None:
            try:
                hits = self._qdrant_client.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=query_vec.tolist(),
                    limit=1,
                    score_threshold=1.0 - threshold,  # Cosine similarity score is (1 - cosine_dist)
                )
                if hits:
                    best_hit = hits[0]
                    cosine_dist = round(1.0 - float(best_hit.score), 4)
                    return {
                        "name": best_hit.payload.get("name", "Unknown"),
                        "distance": cosine_dist,
                        "authorized": cosine_dist <= threshold,
                        "engine": "qdrant",
                    }
                else:
                    # Still find closest for display/audit
                    all_hits = self._qdrant_client.search(
                        collection_name=QDRANT_COLLECTION,
                        query_vector=query_vec.tolist(),
                        limit=1,
                    )
                    if all_hits:
                        best = all_hits[0]
                        dist = round(1.0 - float(best.score), 4)
                        return {
                            "name": best.payload.get("name", "Unknown"),
                            "distance": dist,
                            "authorized": False,
                            "engine": "qdrant",
                        }
                    return None
            except Exception as e:
                logger.warning(f"Qdrant query failed ({e}), falling back to SQLite+NumPy.")

        # Embedded NumPy Fallback Search
        names, mat = self.get_all_embeddings_matrix()
        if mat.size == 0 or len(names) == 0:
            return None

        # Normalized cosine distance
        norm_mat = np.linalg.norm(mat, axis=1) + 1e-8
        norm_q = np.linalg.norm(query_vec) + 1e-8
        dots = mat @ query_vec
        dists = 1.0 - (dots / (norm_mat * norm_q))
        min_idx = int(np.argmin(dists))
        best_dist = float(dists[min_idx])
        best_name = names[min_idx]

        return {
            "name": best_name,
            "distance": round(best_dist, 4),
            "authorized": best_dist <= threshold,
            "engine": "sqlite_numpy",
        }

    # ---------------- Audit Logging ----------------

    def log_audit_event(
        self,
        status: str,
        camera_id: str = "cam_01",
        matched_name: Optional[str] = None,
        confidence: float = 0.0,
        distance: Optional[float] = None,
        liveness_score: float = 1.0,
        bbox: Optional[List[int]] = None,
        snapshot_path: Optional[str] = None,
    ) -> int:
        """Record an access authorization or violation event to persistent database."""
        now = time.time()
        bbox_json = json.dumps(bbox) if bbox else None

        with self._lock:
            with self._get_connection() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO audit_events 
                    (timestamp, camera_id, status, matched_name, confidence, distance, liveness_score, bbox_json, snapshot_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (now, camera_id, status, matched_name, confidence, distance, liveness_score, bbox_json, snapshot_path),
                )
                conn.commit()
                return cur.lastrowid

    def list_audit_events(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        search_name: Optional[str] = None,
    ) -> Tuple[int, List[dict]]:
        """Query paginated audit logs with filtering."""
        with self._get_connection() as conn:
            where_clauses = []
            params = []
            if status:
                where_clauses.append("status = ?")
                params.append(status.lower())
            if search_name:
                where_clauses.append("matched_name LIKE ?")
                params.append(f"%{search_name}%")

            where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            count_cur = conn.execute(f"SELECT COUNT(*) as total FROM audit_events {where_str};", params)
            total = count_cur.fetchone()["total"]

            query = f"""
            SELECT id, timestamp, camera_id, status, matched_name, confidence, distance, liveness_score, bbox_json, snapshot_path
            FROM audit_events
            {where_str}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?;
            """
            rows = conn.execute(query, params + [limit, offset]).fetchall()

            events = []
            for r in rows:
                ev = dict(r)
                if ev["bbox_json"]:
                    try:
                        ev["bbox"] = json.loads(ev["bbox_json"])
                    except Exception:
                        ev["bbox"] = None
                else:
                    ev["bbox"] = None
                events.append(ev)

            return total, events

    def get_stats(self) -> dict:
        """Return system statistics for dashboard and monitoring."""
        with self._get_connection() as conn:
            p_cnt = conn.execute("SELECT COUNT(*) as cnt FROM persons;").fetchone()["cnt"]
            e_cnt = conn.execute("SELECT COUNT(*) as cnt FROM embeddings;").fetchone()["cnt"]
            ev_cnt = conn.execute("SELECT COUNT(*) as cnt FROM audit_events;").fetchone()["cnt"]
            unauth_cnt = conn.execute("SELECT COUNT(*) as cnt FROM audit_events WHERE status IN ('unauthorized', 'spoof');").fetchone()["cnt"]
            spoof_cnt = conn.execute("SELECT COUNT(*) as cnt FROM audit_events WHERE status = 'spoof';").fetchone()["cnt"]

            return {
                "enrolled_persons": p_cnt,
                "total_embeddings": e_cnt,
                "total_audit_events": ev_cnt,
                "total_violations": unauth_cnt,
                "total_spoofs_detected": spoof_cnt,
                "qdrant_active": self._qdrant_available,
                "qdrant_url": self.qdrant_url,
                "vector_engine": "qdrant" if self._qdrant_available else "sqlite_numpy_embedded",
            }
