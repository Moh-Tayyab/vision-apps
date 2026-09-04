"""Face Engine: DeepFace Feature Extractor with Database & Vector Persistence.

Uses the Python ``deepface`` library (Facenet embeddings, cosine distance)
backed by the thread-safe DatabaseManager (SQLite WAL + Qdrant Vector Engine).
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from db import DatabaseManager

logger = logging.getLogger("face_auth.engine")

MODEL_NAME = "Facenet"
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "retinaface")
COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.48"))


class FaceEngine:
    """Lazy-loaded DeepFace wrapper integrated with Database & Qdrant Vector Store."""

    def __init__(self, db_path: str, threshold: Optional[float] = None, qdrant_url: Optional[str] = None):
        self.db_path = db_path
        self.threshold = float(threshold if threshold is not None else COSINE_THRESHOLD)
        self._photos_dir = os.path.join(os.path.dirname(db_path), "photos")
        os.makedirs(self._photos_dir, exist_ok=True)

        self.db = DatabaseManager(db_path=db_path, qdrant_url=qdrant_url)
        self._model_loaded = False
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if not self._model_loaded:
            with self._lock:
                if not self._model_loaded:
                    import deepface  # noqa: F401

                    from deepface import DeepFace

                    # Warm up with a small blank image so first request is fast
                    DeepFace.represent(
                        img_path=np.zeros((160, 160, 3), dtype=np.uint8),
                        model_name=MODEL_NAME,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                    self._model_loaded = True
                    logger.info("DeepFace model warmed up successfully.")

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def enroll(self, name: str, images: List[np.ndarray]) -> dict:
        """Extract embeddings from images and persist to SQLite + Qdrant."""
        try:
            from deepface import DeepFace
        except ImportError as e:
            raise RuntimeError(f"deepface/tensorflow not available: {e}")

        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Name cannot be empty")

        new_embeddings: List[List[float]] = []
        for img in images:
            reps = DeepFace.represent(
                img_path=img,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
            )
            if not reps:
                continue
            new_embeddings.extend(r["embedding"] for r in reps)

        if not new_embeddings:
            raise ValueError(f"No face detected in any of the {len(images)} enrollment image(s)")

        # Save primary photo to disk
        photo_file = None
        if images:
            person_dir = os.path.join(self._photos_dir, clean_name)
            os.makedirs(person_dir, exist_ok=True)
            photo_file = os.path.join(person_dir, "photo.jpg")
            cv2.imwrite(photo_file, images[0])

        res = self.db.enroll_person(clean_name, new_embeddings, photo_path=photo_file)
        logger.info(f"Enrolled person '{clean_name}' with {len(new_embeddings)} new embeddings.")
        return res

    def get_person_photo_path(self, name: str) -> Optional[str]:
        return self.db.get_person_photo_path(name.strip())

    def remove(self, name: str) -> bool:
        clean_name = name.strip()
        ok = self.db.delete_person(clean_name)
        person_dir = os.path.join(self._photos_dir, clean_name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir, ignore_errors=True)
        return ok

    def list_persons(self) -> List[dict]:
        return self.db.list_persons()

    def identify_face(self, face_bgr: np.ndarray, threshold: Optional[float] = None) -> Optional[dict]:
        """Extract query vector and search in Database / Qdrant vector index."""
        from deepface import DeepFace

        eff_threshold = self.threshold if threshold is None else float(threshold)

        rep = DeepFace.represent(
            img_path=face_bgr,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        if not rep or not rep[0].get("embedding"):
            return None

        query = np.asarray(rep[0]["embedding"], dtype=np.float32)
        return self.db.search_face(query, threshold=eff_threshold)
