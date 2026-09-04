"""Run face-authorization inference against a trained model.

Loads a model produced by :mod:`train`, then checks faces in:

  * a single image  (``--image path.jpg``)
  * a folder of images (``--folder dir/``)
  * a live webcam    (``--webcam``)

For every detected face it prints/overlays the match result:
``authorized`` / ``unauthorized`` + matched name + cosine distance (lower = closer).

Usage::

    python inference.py --model face_embeddings.pkl --image test.jpg
    python inference.py --model face_embeddings.pkl --folder ./test_images --output-dir ./out
    python inference.py --model face_embeddings.pkl --webcam
    python inference.py --model face_embeddings.pkl --image test.jpg --threshold 0.35 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import cv2
import numpy as np

from face_engine import COSINE_THRESHOLD, DETECTOR_BACKEND, FaceDatabase

COLOR_AUTHORIZED = (0, 200, 0)     # BGR green
COLOR_UNAUTHORIZED = (0, 0, 255)   # BGR red
COLOR_UNKNOWN = (0, 165, 255)      # BGR orange


def _draw(frame: np.ndarray, result: dict) -> np.ndarray:
    vis = frame.copy()
    for f in result["faces"]:
        x1, y1, x2, y2 = f["bbox"]
        status = f.get("status", "unknown")
        color = {
            "authorized": COLOR_AUTHORIZED,
            "unauthorized": COLOR_UNAUTHORIZED,
            "unknown": COLOR_UNKNOWN,
        }.get(status, (255, 255, 255))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if status == "authorized":
            label = f"AUTHORIZED: {f['matched_name']} ({f['distance']:.2f})"
        elif status == "unauthorized":
            label = f"UNAUTHORIZED: {f['matched_name']} ({f['distance']:.2f})"
        else:
            label = "UNKNOWN"
        cv2.putText(vis, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


def _print_result(source: str, result: dict, as_json: bool) -> None:
    if as_json:
        payload = {"source": source, "num_faces": result["num_faces"], "faces": result["faces"]}
        print(json.dumps(payload))
        return
    print(f"\n[{source}]  faces detected: {result['num_faces']}")
    for i, f in enumerate(result["faces"]):
        status = f.get("status", "unknown")
        if status == "authorized":
            print(f"  face {i + 1}: AUTHORIZED  -> {f['matched_name']}  (distance={f['distance']:.4f}, det_conf={f['confidence']:.2f})")
        elif status == "unauthorized":
            print(f"  face {i + 1}: UNAUTHORIZED -> {f['matched_name']}  (distance={f['distance']:.4f}, det_conf={f['confidence']:.2f})")
        else:
            print(f"  face {i + 1}: UNKNOWN ({f.get('reason', '')})  (det_conf={f['confidence']:.2f})")


def _run_image(db: FaceDatabase, path: str, args) -> None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[error] cannot read image: {path}", file=sys.stderr)
        return
    result = db.recognize_image(img)
    _print_result(os.path.basename(path), result, args.json)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, _draw(img, result))
        if not args.json:
            print(f"  annotated image -> {out_path}")


def _run_folder(db: FaceDatabase, folder: str, args) -> None:
    files = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )
    if not files:
        print(f"[error] no images found in folder: {folder}", file=sys.stderr)
        return
    for path in files:
        _run_image(db, path, args)


def _run_webcam(db: FaceDatabase, args) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[error] cannot open webcam (index 0).", file=sys.stderr)
        return
    print("Webcam started. Press 'q' to quit.\n")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[warn] frame read failed; stopping.")
                break
            result = db.recognize_image(frame)
            vis = _draw(frame, result)
            if not args.json:
                status = "UNAUTHORIZED" if result["any_unauthorized"] else (
                    "AUTHORIZED" if result["num_faces"] else "NO FACE"
                )
                cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Face Authorization", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Face-authorization inference (image / folder / webcam).")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Path to a single image.")
    src.add_argument("--folder", help="Path to a folder of images.")
    src.add_argument("--webcam", action="store_true", help="Use the default webcam (index 0).")
    parser.add_argument("--model", default="face_embeddings.pkl", help="Trained model file (default: face_embeddings.pkl).")
    parser.add_argument("--threshold", type=float, default=None, help="Override the model's authorization threshold.")
    parser.add_argument("--detector", default=DETECTOR_BACKEND, help=f"Detector backend (default: {DETECTOR_BACKEND}).")
    parser.add_argument("--output-dir", default=None, help="Where to save annotated images (image/folder modes).")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of human text.")
    args = parser.parse_args(argv)

    try:
        db = FaceDatabase.load(args.model)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    if args.threshold is not None:
        db.threshold = args.threshold
    db.detector_backend = args.detector

    if not args.json:
        print(f"Loaded model: {args.model}")
        print(f"Persons     : {len(db.list_persons())}  threshold={db.threshold:.2f}  detector={db.detector_backend}\n")

    if args.webcam:
        _run_webcam(db, args)
    elif args.folder:
        _run_folder(db, args.folder, args)
    elif args.image:
        _run_image(db, args.image, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
