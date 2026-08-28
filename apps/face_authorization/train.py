"""Train a face-authorization model from a folder of images.

Scans a folder of face images, detects faces with DeepFace, computes embeddings
for every detected face, and writes a single portable model file (``.pkl``)
ready to be loaded by :mod:`inference`.

Folder layout (Option A — preferred, folder-per-person)::

    <images_folder>/
        alice/
            img1.jpg
            img2.jpg
        bob/
            shot.png

Flat layout (Option B — name inferred from filename)::

    <images_folder>/
        alice_01.jpg     -> person "alice"
        alice_02.jpg     -> person "alice"
        bob_01.jpg       -> person "bob"

    Use ``--name-separator _`` (default).  If no separator is present in a
    filename the whole stem is treated as one person label (each file its own
    person), which is convenient for a quick single-person test.

Usage::

    python train.py --images_folder /path/to/images
    python train.py --images_folder /path/to/images --model face_embeddings.pkl \\
        --threshold 0.40 --detector retinaface
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

from face_engine import (
    COSINE_THRESHOLD,
    DETECTOR_BACKEND,
    MODEL_NAME,
    FaceDatabase,
    detect_faces,
    embed_face,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _load_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"unreadable / corrupt image: {path}")
    return img


def _discover_persons(images_folder: str, name_separator: str):
    """Return ``(layout, mapping)``.

    layout is "folders" or "flat".  mapping is ``person_name -> [image_paths]``.
    """
    entries = sorted(
        e for e in os.listdir(images_folder) if not e.startswith(".")
    )
    subdirs = [e for e in entries if os.path.isdir(os.path.join(images_folder, e))]

    if subdirs:
        # Option A: folder-per-person
        mapping: Dict[str, List[str]] = {}
        for d in subdirs:
            dpath = os.path.join(images_folder, d)
            files = [
                os.path.join(dpath, f)
                for f in sorted(os.listdir(dpath))
                if _is_image(os.path.join(dpath, f))
            ]
            if files:
                mapping[d] = files
        return "folders", mapping

    # Option B: flat folder, infer name from filename
    files = [os.path.join(images_folder, f) for f in entries if _is_image(os.path.join(images_folder, f))]
    mapping = {}
    for fpath in files:
        stem = os.path.splitext(os.path.basename(fpath))[0]
        if name_separator and name_separator in stem:
            name = stem.split(name_separator)[0].strip()
        else:
            name = stem.strip()
        if not name:
            name = stem
        mapping.setdefault(name, []).append(fpath)
    return "flat", mapping


def _process_person(db: FaceDatabase, name: str, image_paths: List[str]) -> Tuple[int, int, int]:
    """Embed every face in ``image_paths`` for ``name``.

    Returns (images_ok, faces_detected, embeddings_added).
    """
    embeddings: List[np.ndarray] = []
    images_ok = 0
    faces_detected = 0
    for path in image_paths:
        try:
            img = _load_image(path)
        except ValueError as e:
            print(f"  [warn] {e}")
            continue
        images_ok += 1
        try:
            faces = detect_faces(img, db.detector_backend)
        except Exception as e:
            print(f"  [warn] detection failed for {path}: {e}")
            continue
        for face in faces:
            faces_detected += 1
            emb = embed_face(face["face"], db.model_name)
            if emb is not None:
                embeddings.append(emb)

    added = 0
    if embeddings:
        added = db.add_person(
            name,
            embeddings,
            metadata={"source_images": len(image_paths), "faces_detected": faces_detected},
        )
    return images_ok, faces_detected, added


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a face-authorization model from a folder of images.")
    parser.add_argument("--images_folder", required=True, help="Folder containing face images (folder-per-person or flat).")
    parser.add_argument("--model", default="face_embeddings.pkl", help="Output model file path (default: face_embeddings.pkl).")
    parser.add_argument("--model_name", default=MODEL_NAME, help=f"DeepFace model (default: {MODEL_NAME}).")
    parser.add_argument("--detector", default=DETECTOR_BACKEND, help=f"Face detector backend (default: {DETECTOR_BACKEND}).")
    parser.add_argument("--threshold", type=float, default=COSINE_THRESHOLD, help=f"Authorization cosine-distance threshold (default: {COSINE_THRESHOLD}).")
    parser.add_argument("--name-separator", default="_", help="Flat-mode: split filename on this char to get the person name (default: _).")
    parser.add_argument("--min-face", type=int, default=40, help="Minimum face size in px to keep (default: 40).")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.images_folder):
        print(f"[error] images folder not found: {args.images_folder}", file=sys.stderr)
        return 2

    print("=" * 60)
    print("Face Authorization — Training")
    print("=" * 60)
    print(f"Images folder : {args.images_folder}")
    print(f"Model out     : {args.model}")
    print(f"Model name    : {args.model_name}")
    print(f"Detector      : {args.detector}")
    print(f"Threshold     : {args.threshold}")
    print("=" * 60)

    layout, mapping = _discover_persons(args.images_folder, args.name_separator)
    if not mapping:
        print("[error] no images found in the folder.", file=sys.stderr)
        return 1
    print(f"Layout detected: {layout}  ({len(mapping)} person(s))\n")

    db = FaceDatabase(
        model_name=args.model_name,
        detector_backend=args.detector,
        threshold=args.threshold,
    )

    total_images = total_faces = total_embeddings = 0
    for name, paths in mapping.items():
        print(f"[{name}] processing {len(paths)} image(s)...")
        try:
            ok, faces, added = _process_person(db, name, paths)
        except Exception as e:
            print(f"  [error] failed to process '{name}': {e}")
            ok, faces, added = 0, 0, 0
        total_images += ok
        total_faces += faces
        total_embeddings += added
        print(f"  -> images ok: {ok}, faces detected: {faces}, embeddings added: {added}")

    # Drop persons that ended up with zero embeddings.
    empty = [n for n, p in db._persons.items() if p["embeddings"].size == 0]
    for n in empty:
        db.remove(n)

    if db.is_empty():
        print("\n[error] no faces were detected in any image; model not saved.", file=sys.stderr)
        return 1

    db.save(args.model)

    print("\n" + "=" * 60)
    print("Training complete")
    print("=" * 60)
    print(f"Persons trained     : {len(db.list_persons())}")
    print(f"Images processed    : {total_images}")
    print(f"Faces detected      : {total_faces}")
    print(f"Total embeddings    : {total_embeddings}")
    print(f"Model saved to      : {args.model}")
    if empty:
        print(f"Skipped (no face)   : {', '.join(empty)}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
