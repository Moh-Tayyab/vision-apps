"""Face embedding store: persists authorized-person embeddings as JSON.

Uses the Python ``deepface`` library (Facenet embeddings, cosine distance).

Photo storage:
    Each enrolled person gets a directory under ``<data_dir>/photos/<name>/``
    containing the front-facing enrollment photo(s).  MVP stores 1 photo;
    the structure is ready for 3-4 photos in a future update.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from typing import Dict, List, Optional

import cv2
import numpy as np

MODEL_NAME = "Facenet"
# "retinaface" is used instead of "opencv": OpenCV 5.x removed Haar cascade XML
# files from its pip package. RetinaFace (installed via the retinaface package)
# is more accurate and works out-of-the-box.
DETECTOR_BACKEND = "retinaface"
COSINE_THRESHOLD = 0.40  # deepface recommended threshold for Facenet


class FaceEngine:
    """Lazy-loaded deepface wrapper + JSON-backed embedding store."""

    def __init__(self, store_path: str):
        self.store_path = store_path
        # Photos live alongside the embeddings JSON: <data_dir>/photos/<name>/
        self._photos_dir = os.path.join(os.path.dirname(store_path), "photos")
        self._lock = threading.Lock()
        # {person_name: {"embeddings": [[float]], "updated_at": ts}}
        self._persons: Dict[str, dict] = {}
        self._model_loaded = False
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.store_path):
            with open(self.store_path, "r") as f:
                self._persons = json.load(f)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w") as f:
            json.dump(self._persons, f)

    def _ensure_model(self) -> None:
        if not self._model_locked():
            import deepface  # noqa: F401  (imports tensorflow; slow on first call)

            from deepface import DeepFace

            # Warm up with a tiny image so first real request isn't cold.
            DeepFace.represent(
                img_path=np.zeros((160, 160, 3), dtype=np.uint8),
                model_name=MODEL_NAME,
                detector_backend="skip",
                enforce_detection=False,
            )
            self._model_loaded = True

    def _model_locked(self) -> bool:
        return self._model_loaded

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    def enroll(self, name: str, images: List[np.ndarray]) -> dict:
        """Extract and store face embeddings for one person."""
        try:
            from deepface import DeepFace
        except ImportError as e:
            raise RuntimeError(f"deepface/tensorflow not available: {e}")

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
            raise ValueError(f"No face found in any of the {len(images)} image(s)")

        # Persist enrollment photo to data/photos/<name>/photo.jpg
        if images:
            person_dir = os.path.join(self._photos_dir, name)
            os.makedirs(person_dir, exist_ok=True)
            photo_file = os.path.join(person_dir, "photo.jpg")
            cv2.imwrite(photo_file, images[0])

        with self._lock:
            existing = self._persons.get(name, {"embeddings": []})
            existing["embeddings"].extend(new_embeddings)
            existing["updated_at"] = time.time()
            self._persons[name] = existing
            self._save()
        return {
            "name": name,
            "new_embeddings": len(new_embeddings),
            "total_embeddings": len(existing["embeddings"]),
        }

    def get_person_photo_path(self, name: str) -> Optional[str]:
        """Return absolute path to person's enrollment photo if present."""
        person_dir = os.path.join(self._photos_dir, name)
        photo_file = os.path.join(person_dir, "photo.jpg")
        if os.path.exists(photo_file):
            return photo_file
        return None

    def remove(self, name: str) -> bool:
        with self._lock:
            if name not in self._persons:
                return False
            del self._persons[name]
            self._save()

        person_dir = os.path.join(self._photos_dir, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir, ignore_errors=True)
        return True

    def list_persons(self) -> List[dict]:
        return [
            {"name": n, "num_embeddings": len(p["embeddings"]), "updated_at": p.get("updated_at")}
            for n, p in sorted(self._persons.items())
        ]

    def identify_face(self, face_bgr: np.ndarray) -> Optional[dict]:
        """Match one cropped face against stored embeddings.

        Returns {"name", "distance", "authorized"} or None if no embeddings stored.
        """
        from deepface import DeepFace

        with self._lock:
            persons = {n: list(p["embeddings"]) for n, p in self._persons.items()}
        if not persons:
            return None

        rep = DeepFace.represent(
            img_path=face_bgr,
            model_name=MODEL_NAME,
            detector_backend="skip",
            enforce_detection=False,
        )
        if not rep:
            return None
        query = np.asarray(rep[0]["embedding"], dtype=np.float32)

        best_name, best_dist = None, float("inf")
        for name, embs in persons.items():
            mat = np.asarray(embs, dtype=np.float32)
            # cosine distance per deepface convention
            dots = mat @ query
            dists = 1 - dots / (np.linalg.norm(mat, axis=1) * np.linalg.norm(query) + 1e-8)
            d = float(np.min(dists))
            if d < best_dist:
                best_name, best_dist = name, d

        return {
            "name": best_name,
            "distance": round(best_dist, 4),
            "authorized": best_dist <= COSINE_THRESHOLD,
        }
