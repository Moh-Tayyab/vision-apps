"""End-to-End Production Verification Test Suite for Face Authorization Service."""

import os
import sys
import tempfile
import shutil
import unittest
import numpy as np
import cv2

# Set test environment
os.environ["SQLITE_DB_PATH"] = ":memory:"
os.environ["QDRANT_URL"] = "http://localhost:6333"  # Will fallback to in-memory numpy gracefully
os.environ["FACE_AUTH_API_KEY"] = "test-secret-key-123"

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from apps.face_authorization.db import DatabaseManager
from apps.face_authorization.anti_spoof import AntiSpoofEngine
from apps.face_authorization.tracker import FaceTracker
from apps.face_authorization.security import verify_admin_access
from apps.face_authorization.metrics import MetricsCollector
from fastapi.testclient import TestClient
from apps.face_authorization.main import app


class TestFaceAuthorizationProduction(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_faces.db")
        self.db = DatabaseManager(db_path=self.db_path, qdrant_url=None)  # Test pure SQLite + NumPy mode
        self.anti_spoof = AntiSpoofEngine(default_threshold=0.50)
        self.tracker = FaceTracker(iou_threshold=0.30, max_missed_frames=5)
        self.client = TestClient(app)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_crud_and_vector_search(self):
        # 1. Enroll Person
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        
        enroll_res = self.db.enroll_person(
            name="Ali Khan",
            embeddings=[emb1.tolist()]
        )
        self.assertIn("person_id", enroll_res)
        self.assertEqual(enroll_res["name"], "Ali Khan")
        self.assertEqual(enroll_res["total_embeddings"], 1)
        
        # 2. List Persons
        persons = self.db.list_persons()
        self.assertEqual(len(persons), 1)
        self.assertEqual(persons[0]["name"], "Ali Khan")
        self.assertEqual(persons[0]["num_embeddings"], 1)

        # 3. Vector Match with close embedding (distance < 0.2)
        noisy_emb = emb1 + (np.random.randn(512) * 0.02).astype(np.float32)
        noisy_emb = noisy_emb / np.linalg.norm(noisy_emb)
        
        match = self.db.search_face(noisy_emb, threshold=0.48)
        self.assertIsNotNone(match)
        self.assertTrue(match["authorized"])
        self.assertEqual(match["name"], "Ali Khan")
        self.assertLess(match["distance"], 0.20)

        # 4. Vector Match with orthogonal embedding (distance ~ 1.0 -> no match)
        ortho_emb = -emb1
        no_match = self.db.search_face(ortho_emb, threshold=0.48)
        self.assertIsNotNone(no_match)
        self.assertFalse(no_match["authorized"])

        # 5. Audit Logging
        event_id = self.db.log_audit_event(
            status="authorized",
            camera_id="cam_gate_1",
            matched_name="Ali Khan",
            confidence=0.95,
            distance=0.05,
            liveness_score=0.92,
            bbox=[100, 100, 200, 200]
        )
        self.assertGreater(event_id, 0)
        total, logs = self.db.list_audit_events(limit=10)
        self.assertEqual(total, 1)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["matched_name"], "Ali Khan")
        self.assertEqual(logs[0]["status"], "authorized")

        # 6. Delete Person
        deleted = self.db.delete_person("Ali Khan")
        self.assertTrue(deleted)
        self.assertEqual(len(self.db.list_persons()), 0)

    def test_anti_spoofing_engine(self):
        # Create synthetic live-like face pattern (gradient texture + chrominance variance)
        real_crop = np.zeros((160, 160, 3), dtype=np.uint8)
        for y in range(160):
            for x in range(160):
                real_crop[y, x] = [120 + (x % 30), 140 + (y % 40), 200 - (x % 20)]
        
        # Add subtle natural noise
        noise = np.random.randint(-15, 15, (160, 160, 3)).astype(np.int16)
        real_crop = np.clip(real_crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        result_real = self.anti_spoof.check_liveness(real_crop)
        self.assertIsInstance(result_real.is_real, bool)
        self.assertIsInstance(result_real.liveness_score, float)
        self.assertIn("chroma_score", result_real.details)

        # Create flat blank screen / single color (definite spoof)
        flat_crop = np.full((160, 160, 3), 128, dtype=np.uint8)
        result_flat = self.anti_spoof.check_liveness(flat_crop)
        self.assertFalse(result_flat.is_real)
        self.assertLess(result_flat.liveness_score, 0.40)

    def test_temporal_face_tracker(self):
        # Frame 1: Detection at [100, 100, 200, 200]
        detections_f1 = [
            {
                "bbox": [100, 100, 200, 200],
                "status": "AUTHORIZED",
                "matched_name": "Ali Khan",
                "confidence": 0.95,
                "distance": 0.15,
                "liveness_score": 0.88,
            }
        ]
        tracked_f1 = self.tracker.update(detections_f1)
        self.assertEqual(len(tracked_f1), 1)
        track_id_1 = tracked_f1[0]["track_id"]
        self.assertEqual(tracked_f1[0]["matched_name"], "Ali Khan")
        self.assertEqual(tracked_f1[0]["status"], "AUTHORIZED")

        # Frame 2: Slight movement to [102, 99, 202, 199] with temporary 1-frame jitter name "Unknown"
        detections_f2 = [
            {
                "bbox": [102, 99, 202, 199],
                "status": "UNAUTHORIZED",
                "matched_name": "Unknown",
                "confidence": 0.40,
                "distance": 0.60,
                "liveness_score": 0.85,
            }
        ]
        tracked_f2 = self.tracker.update(detections_f2)
        self.assertEqual(len(tracked_f2), 1)
        self.assertEqual(tracked_f2[0]["track_id"], track_id_1)
        # Smoothing should retain "Ali Khan" and "AUTHORIZED" via majority voting
        self.assertEqual(tracked_f2[0]["matched_name"], "Ali Khan")
        self.assertEqual(tracked_f2[0]["status"], "AUTHORIZED")

    def test_fastapi_endpoints(self):
        # 1. Health check
        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json()["status"], "healthy")

        # 2. Metrics endpoint (Prometheus format)
        res_metrics = self.client.get("/metrics")
        self.assertEqual(res_metrics.status_code, 200)
        self.assertIn("face_auth_inference_total", res_metrics.text)
        self.assertIn("face_auth_up", res_metrics.text)

        # 3. DB Stats
        res_stats = self.client.get("/db/stats")
        self.assertEqual(res_stats.status_code, 200)
        stats = res_stats.json()
        self.assertIn("enrolled_persons", stats)
        self.assertIn("vector_engine", stats)

        # 4. Audit Events Query
        res_audit = self.client.get("/audit/events?limit=5")
        self.assertEqual(res_audit.status_code, 200)
        body = res_audit.json()
        self.assertIn("events", body)
        self.assertIsInstance(body["events"], list)


if __name__ == "__main__":
    unittest.main()
