"""Comprehensive, fast cached accuracy benchmark for Top Camera Counter."""

import os
import glob
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../apps/top_camera_counter")))

from detector import CartonDetector, compute_iou, compute_ios, non_max_suppression
from live_counter import filter_top_row_cartons, LiveCartonCounter
from tracker import ByteTracker
from state_machine import CartonStateMachine, CartonState


CACHE_FILE = "/home/muhammadtayyab/projects/carton-counter/tests/inference_cache.json"


def parse_yolo_annotation(label_path: str, img_w: int, img_h: int) -> List[Tuple[float, float, float, float]]:
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            coords = [float(p) for p in parts[1:]]
            if len(coords) == 4:
                cx, cy, w, h = coords
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
            elif len(coords) >= 6:
                xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
                ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                continue

            x1 = max(0.0, min(float(img_w), x1))
            y1 = max(0.0, min(float(img_h), y1))
            x2 = max(0.0, min(float(img_w), x2))
            y2 = max(0.0, min(float(img_h), y2))

            if (x2 - x1) > 2 and (y2 - y1) > 2:
                boxes.append((x1, y1, x2, y2))
    return boxes


def match_detections(
    gt_boxes: List[Tuple[float, float, float, float]],
    pred_boxes: List[dict],
    iou_thresh: float = 0.5,
) -> Tuple[int, int, int, List[float]]:
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, []
    if not gt_boxes:
        return 0, len(pred_boxes), 0, []
    if not pred_boxes:
        return 0, 0, len(gt_boxes), []

    sorted_preds = sorted(pred_boxes, key=lambda p: p.get("confidence", 0.0), reverse=True)
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(sorted_preds)
    matched_ious = []

    for p_idx, pred in enumerate(sorted_preds):
        p_box = (pred["x1"], pred["y1"], pred["x2"], pred["y2"])
        best_iou = 0.0
        best_g_idx = -1

        for g_idx, gt in enumerate(gt_boxes):
            if gt_matched[g_idx]:
                continue
            iou = compute_iou(p_box, gt)
            if iou > best_iou:
                best_iou = iou
                best_g_idx = g_idx

        if best_iou >= iou_thresh and best_g_idx >= 0:
            gt_matched[best_g_idx] = True
            pred_matched[p_idx] = True
            matched_ious.append(best_iou)

    tp = sum(pred_matched)
    fp = len(sorted_preds) - tp
    fn = len(gt_boxes) - sum(gt_matched)

    return tp, fp, fn, matched_ious


def get_cached_raw_inferences(detector: CartonDetector, dataset_root: str) -> Dict[str, dict]:
    """Fetch raw predictions at min confidence (0.05) once and cache to disk."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cached = json.load(f)
                print(f"Loaded {len(cached)} cached inferences from {CACHE_FILE}")
                return cached
        except Exception:
            pass

    print("Fetching raw detections from Roboflow model for dataset images...")
    cached = {}
    splits = ["test", "valid", "train"]

    for split in splits:
        img_dir = os.path.join(dataset_root, split, "images")
        lbl_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.exists(img_dir):
            continue

        for img_path in sorted(glob.glob(os.path.join(img_dir, "*.*"))):
            img_name = os.path.basename(img_path)
            stem = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_dir, f"{stem}.txt")

            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_boxes = parse_yolo_annotation(lbl_path, w, h)

            # Detect at low confidence=0.05 without NMS to cache raw candidates
            boxes, inf_ms = detector.detect(img, confidence=0.05, apply_nms=False)
            print(f"  [{split}] {img_name}: {len(boxes)} raw boxes in {inf_ms:.1f}ms")

            cached[img_path] = {
                "split": split,
                "img_name": img_name,
                "width": w,
                "height": h,
                "gt_boxes": gt_boxes,
                "raw_boxes": boxes,
                "inf_ms": inf_ms,
            }

    with open(CACHE_FILE, "w") as f:
        json.dump(cached, f, indent=2)
    print(f"Saved {len(cached)} cached inferences to {CACHE_FILE}")
    return cached


def run_evaluation_on_cache(
    cached_data: Dict[str, dict],
    confidence_thresh: float = 0.36,
    iou_eval_thresh: float = 0.5,
    top_row_only: bool = False,
    nms_iou: float = 0.35,
    ios_thresh: float = 0.58,
) -> dict:
    all_images_results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt, total_pred = 0, 0
    abs_errors = []
    sq_errors = []
    all_ious = []
    by_split = {}

    for img_path, data in cached_data.items():
        split = data["split"]
        w = data["width"]
        h = data["height"]
        gt_boxes = data["gt_boxes"]
        raw_boxes = data["raw_boxes"]

        # 1. Filter by confidence
        conf_filtered = [b for b in raw_boxes if b["confidence"] >= confidence_thresh]

        # 2. Apply NMS and containment deduplication
        nms_boxes = non_max_suppression(conf_filtered, iou_thresh=nms_iou, ios_thresh=ios_thresh)

        # 3. Apply Top-Row / Top-Layer filter if requested
        if top_row_only:
            det_tuples = [(b["x1"], b["y1"], b["x2"], b["y2"], b["confidence"]) for b in nms_boxes]
            filtered_tuples, _ = filter_top_row_cartons(det_tuples, img_height=h, img_width=w, top_row_only=True)
            final_preds = [{"x1": d[0], "y1": d[1], "x2": d[2], "y2": d[3], "confidence": d[4]} for d in filtered_tuples]
        else:
            final_preds = nms_boxes

        tp, fp, fn, matched_ious = match_detections(gt_boxes, final_preds, iou_thresh=iou_eval_thresh)
        gt_cnt = len(gt_boxes)
        pred_cnt = len(final_preds)
        abs_err = abs(pred_cnt - gt_cnt)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += gt_cnt
        total_pred += pred_cnt
        abs_errors.append(abs_err)
        sq_errors.append((pred_cnt - gt_cnt) ** 2)
        all_ious.extend(matched_ious)

        if split not in by_split:
            by_split[split] = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0, "abs_errors": []}
        by_split[split]["tp"] += tp
        by_split[split]["fp"] += fp
        by_split[split]["fn"] += fn
        by_split[split]["gt"] += gt_cnt
        by_split[split]["pred"] += pred_cnt
        by_split[split]["abs_errors"].append(abs_err)

        all_images_results.append({
            "split": split,
            "image": data["img_name"],
            "gt_count": gt_cnt,
            "pred_count": pred_cnt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "mean_iou": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        })

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    mae = float(np.mean(abs_errors)) if abs_errors else 0.0
    rmse = math.sqrt(float(np.mean(sq_errors))) if sq_errors else 0.0
    exact = sum(1 for e in abs_errors if e == 0) / len(abs_errors) if abs_errors else 0.0
    within_1 = sum(1 for e in abs_errors if e <= 1) / len(abs_errors) if abs_errors else 0.0
    within_2 = sum(1 for e in abs_errors if e <= 2) / len(abs_errors) if abs_errors else 0.0

    split_summary = {}
    for s, sdata in by_split.items():
        sp = sdata["tp"] / (sdata["tp"] + sdata["fp"]) if (sdata["tp"] + sdata["fp"]) > 0 else 0.0
        sr = sdata["tp"] / (sdata["tp"] + sdata["fn"]) if (sdata["tp"] + sdata["fn"]) > 0 else 0.0
        sf1 = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0.0
        split_summary[s] = {
            "gt_cartons": sdata["gt"],
            "pred_cartons": sdata["pred"],
            "tp": sdata["tp"],
            "fp": sdata["fp"],
            "fn": sdata["fn"],
            "precision": round(sp, 4),
            "recall": round(sr, 4),
            "f1_score": round(sf1, 4),
            "mae": round(float(np.mean(sdata["abs_errors"])), 2) if sdata["abs_errors"] else 0.0,
        }

    return {
        "confidence_threshold": confidence_thresh,
        "top_row_only": top_row_only,
        "total_images": len(all_images_results),
        "total_gt_cartons": total_gt,
        "total_pred_cartons": total_pred,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "mean_iou_matched": round(float(np.mean(all_ious)), 4) if all_ious else 0.0,
        "counting_mae": round(mae, 2),
        "counting_rmse": round(rmse, 2),
        "exact_count_accuracy_pct": round(exact * 100.0, 2),
        "within_1_count_accuracy_pct": round(within_1 * 100.0, 2),
        "within_2_count_accuracy_pct": round(within_2 * 100.0, 2),
        "by_split": split_summary,
        "per_image": all_images_results,
    }


def main():
    detector = CartonDetector()
    dataset_root = "apps/top_camera_counter/dataset"

    cached = get_cached_raw_inferences(detector, dataset_root)

    print("\n" + "=" * 80)
    print("MODE A: RAW FULL-IMAGE CARTON DETECTIONS (All Layers vs GT Top Cartons)")
    print("=" * 80)
    raw_eval = run_evaluation_on_cache(cached, confidence_thresh=0.36, top_row_only=False)
    print(f"Total Images: {raw_eval['total_images']} | GT Cartons: {raw_eval['total_gt_cartons']} | Pred Cartons: {raw_eval['total_pred_cartons']}")
    print(f"Precision: {raw_eval['precision']:.4f} | Recall: {raw_eval['recall']:.4f} | F1: {raw_eval['f1_score']:.4f}")
    print(f"Matched IoU: {raw_eval['mean_iou_matched']:.4f} | MAE: {raw_eval['counting_mae']} | RMSE: {raw_eval['counting_rmse']}")
    print(f"Exact Count Acc: {raw_eval['exact_count_accuracy_pct']}% | Within +/-1 Acc: {raw_eval['within_1_count_accuracy_pct']}%")

    print("\n" + "=" * 80)
    print("MODE B: TOP-ROW FILTERED CARTON COUNTING (top_row_only=True)")
    print("=" * 80)
    top_eval = run_evaluation_on_cache(cached, confidence_thresh=0.36, top_row_only=True)
    print(f"Total Images: {top_eval['total_images']} | GT Cartons: {top_eval['total_gt_cartons']} | Pred Top-Row Cartons: {top_eval['total_pred_cartons']}")
    print(f"Precision: {top_eval['precision']:.4f} | Recall: {top_eval['recall']:.4f} | F1: {top_eval['f1_score']:.4f}")
    print(f"Matched IoU: {top_eval['mean_iou_matched']:.4f} | MAE: {top_eval['counting_mae']} | RMSE: {top_eval['counting_rmse']}")
    print(f"Exact Count Acc: {top_eval['exact_count_accuracy_pct']}% | Within +/-1 Acc: {top_eval['within_1_count_accuracy_pct']}% | Within +/-2 Acc: {top_eval['within_2_count_accuracy_pct']}%")

    print("\n" + "-" * 80)
    print("PER-IMAGE BREAKDOWN (Top-Row Filtered):")
    print(f"{'Split':<6} | {'Image Name':<38} | {'GT':<4} | {'Pred':<4} | {'TP':<3} | {'FP':<3} | {'FN':<3} | {'IoU':<6}")
    print("-" * 80)
    for row in top_eval["per_image"]:
        short_name = row["image"][:36]
        print(f"{row['split']:<6} | {short_name:<38} | {row['gt_count']:<4} | {row['pred_count']:<4} | {row['tp']:<3} | {row['fp']:<3} | {row['fn']:<3} | {row['mean_iou']:<6.4f}")

    print("\n" + "=" * 80)
    print("CONFIDENCE THRESHOLD SWEEP (Top-Row Filtered, 0.10 -> 0.80)")
    print("=" * 80)
    print(f"{'Conf':<6} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<8} | {'MAE':<6} | {'RMSE':<6} | {'Exact Acc':<10} | {'+/-1 Acc':<10}")
    print("-" * 80)
    sweep_data = []
    for c in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.36, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]:
        ev = run_evaluation_on_cache(cached, confidence_thresh=c, top_row_only=True)
        sweep_data.append(ev)
        print(f"{c:<6.2f} | {ev['precision']:<10.4f} | {ev['recall']:<8.4f} | {ev['f1_score']:<8.4f} | {ev['counting_mae']:<6.2f} | {ev['counting_rmse']:<6.2f} | {ev['exact_count_accuracy_pct']:<10.1f}% | {ev['within_1_count_accuracy_pct']:<10.1f}%")

    # Save complete benchmark results to json
    report = {
        "raw_full_evaluation": raw_eval,
        "top_row_filtered_evaluation": top_eval,
        "confidence_sweep": sweep_data,
    }
    with open("/home/muhammadtayyab/projects/carton-counter/tests/top_camera_benchmark_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nComplete benchmark report written to tests/top_camera_benchmark_results.json")


if __name__ == "__main__":
    main()
