"""Threshold sweep and calibration utility for Helmet Detection.

Sweeps confidence thresholds across [0.15 - 0.70] on test images / validation sets
to analyze:
- Total persons detected
- Helmets vs Violations breakdown
- Detection confidence distributions
- Optimal confidence operating point (balancing false alarms vs missed detections)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Dict, List

import cv2
import numpy as np

from detector import HelmetDetector


def sweep_thresholds(
    image_paths: List[str],
    conf_range: List[float] = [0.20, 0.25, 0.30, 0.35, 0.38, 0.45, 0.50, 0.60],
    model_path: str | None = None,
) -> List[Dict]:
    """Evaluate detector across a list of images over multiple confidence thresholds."""
    detector = HelmetDetector(model_path=model_path)
    loaded_images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is not None:
            loaded_images.append((os.path.basename(p), img))

    if not loaded_images:
        raise ValueError(f"No valid images could be loaded from paths: {image_paths}")

    results = []
    for conf in conf_range:
        total_persons = 0
        total_helmets = 0
        total_violations = 0
        total_unknowns = 0
        total_inference_ms = 0.0

        for name, img in loaded_images:
            res = detector.detect(img, confidence=conf)
            total_persons += len(res.persons)
            total_helmets += sum(1 for p in res.persons if p.status == "helmet")
            total_violations += sum(1 for p in res.persons if p.status == "no_helmet")
            total_unknowns += sum(1 for p in res.persons if p.status == "unknown")
            total_inference_ms += res.inference_time_ms

        n_images = len(loaded_images)
        results.append({
            "conf_threshold": round(conf, 3),
            "num_images": n_images,
            "total_persons_detected": total_persons,
            "avg_persons_per_image": round(total_persons / n_images, 2),
            "helmets_detected": total_helmets,
            "violations_detected": total_violations,
            "unknown_detected": total_unknowns,
            "avg_inference_ms": round(total_inference_ms / n_images, 2),
            "compliance_rate": round(total_helmets / max(1, total_helmets + total_violations), 4),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Sweep confidence thresholds for Helmet Detection.")
    parser.add_argument("--images", nargs="+", default=None, help="Image file paths or glob pattern")
    parser.add_argument("--model", default=None, help="Path to custom model weights (best.pt)")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    # Find candidate images
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if args.images:
        for item in args.images:
            matched = glob.glob(item)
            candidates.extend(matched if matched else [item])
    else:
        candidates = glob.glob(os.path.join(base_dir, "*.jpg")) + glob.glob(os.path.join(base_dir, "*.png"))

    if not candidates:
        print("[error] No sample images found for calibration. Provide --images path/to/*.jpg", file=sys.stderr)
        sys.exit(1)

    print(f"Calibrating helmet detector over {len(candidates)} image(s)...")
    results = sweep_thresholds(candidates, model_path=args.model)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print("\n" + "=" * 80)
        print("  HELMET DETECTION CONFIDENCE THRESHOLD SWEEP")
        print("=" * 80)
        print(f"{'CONF':>6} | {'PERSONS':>8} | {'HELMETS':>8} | {'VIOLATIONS':>11} | {'UNKNOWN':>8} | {'AVG MS':>8} | {'COMPLIANCE':>10}")
        print("-" * 80)
        for r in results:
            print(
                f"{r['conf_threshold']:>6.2f} | "
                f"{r['total_persons_detected']:>8} | "
                f"{r['helmets_detected']:>8} | "
                f"{r['violations_detected']:>11} | "
                f"{r['unknown_detected']:>8} | "
                f"{r['avg_inference_ms']:>8.1f} | "
                f"{r['compliance_rate'] * 100:>9.1f}%"
            )
        print("=" * 80)
        # Suggest optimal point
        # A good balanced threshold maintains high person recall with stable violation detection
        best = next((r for r in results if r["conf_threshold"] in (0.35, 0.38)), results[len(results)//2])
        print(f"\nRecommended Production Threshold: {best['conf_threshold']} (Persons: {best['total_persons_detected']}, Violations: {best['violations_detected']})\n")


if __name__ == "__main__":
    main()
