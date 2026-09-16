"""K-Fold Cross-Validation for Optimal Distance Threshold Calibration in Face Biometrics.

Implements industry-standard biometrics evaluation (similar to LFW & ISO/IEC 19795):
1. Genuine & Imposter pair generation from enrolled embeddings / photo dataset.
2. Stratified K-Fold cross-validation splits.
3. Threshold sweeps over cosine distance range [0.15, 0.75].
4. Computation of:
   - False Acceptance Rate (FAR / FPR)
   - False Rejection Rate (FRR / FNR)
   - Equal Error Rate (EER: point where FAR == FRR)
   - Youden's J Index (Sensitivity + Specificity - 1)
   - Precision, Recall, and F1-Score
   - Area Under ROC Curve (AUC)
5. Selection and evaluation of optimal decision thresholds across folds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("face_auth.calibration")


@dataclass
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    far: float  # False Acceptance Rate = FP / (FP + TN)
    frr: float  # False Rejection Rate = FN / (TP + FN)
    tpr: float  # True Positive Rate (Recall) = TP / (TP + FN)
    fpr: float  # False Positive Rate = FP / (FP + TN)
    precision: float
    f1_score: float
    accuracy: float
    youden_j: float  # TPR - FPR


@dataclass
class FoldResult:
    fold_index: int
    train_optimal_threshold: float
    val_accuracy: float
    val_far: float
    val_frr: float
    val_f1: float
    val_tp: int
    val_fp: int
    val_tn: int
    val_fn: int


@dataclass
class CalibrationReport:
    target_metric: str
    k_folds: int
    num_persons: int
    num_genuine_pairs: int
    num_imposter_pairs: int
    recommended_threshold: float
    threshold_std: float
    mean_val_accuracy: float
    mean_val_far: float
    mean_val_frr: float
    mean_val_f1: float
    estimated_eer: float
    auc_roc: float
    fold_details: List[Dict]
    roc_curve: List[Dict[str, float]]

    def to_dict(self) -> dict:
        return asdict(self)


class ThresholdCalibrator:
    """Performs Stratified K-Fold Cross Validation to calibrate optimal match thresholds."""

    def __init__(
        self,
        k_folds: int = 5,
        target_metric: str = "eer",
        threshold_min: float = 0.15,
        threshold_max: float = 0.75,
        threshold_step: float = 0.005,
        random_seed: int = 42,
    ):
        self.k_folds = max(2, k_folds)
        self.target_metric = target_metric.lower()
        self.thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine distance: 1.0 - (u . v) / (|u| * |v|)."""
        v1 = np.asarray(vec1, dtype=np.float32)
        v2 = np.asarray(vec2, dtype=np.float32)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-8 or n2 < 1e-8:
            return 1.0
        cos_sim = float(np.dot(v1, v2) / (n1 * n2))
        return float(np.clip(1.0 - cos_sim, 0.0, 2.0))

    def generate_pairs(
        self, person_embeddings: Dict[str, List[np.ndarray]], max_pairs_per_type: int = 2500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate genuine pairs (label 1) and imposter pairs (label 0) with distance scores.

        Returns (distances, labels).
        """
        distances: List[float] = []
        labels: List[int] = []

        person_names = list(person_embeddings.keys())

        # 1. Genuine Pairs (Same person)
        genuine_dists = []
        for name, emb_list in person_embeddings.items():
            if len(emb_list) >= 2:
                for (e1, e2) in itertools.combinations(emb_list, 2):
                    genuine_dists.append(self.cosine_distance(e1, e2))
            elif len(emb_list) == 1:
                # Augment with slight simulated perturbations (scale/noise) to test identity self-stability
                base = emb_list[0]
                for _ in range(2):
                    jitter = base + self.rng.normal(0, 0.02, size=base.shape).astype(np.float32)
                    genuine_dists.append(self.cosine_distance(base, jitter))

        if len(genuine_dists) > max_pairs_per_type:
            genuine_dists = list(self.rng.choice(genuine_dists, size=max_pairs_per_type, replace=False))

        distances.extend(genuine_dists)
        labels.extend([1] * len(genuine_dists))

        # 2. Imposter Pairs (Different persons)
        imposter_dists = []
        for i in range(len(person_names)):
            for j in range(i + 1, len(person_names)):
                p1, p2 = person_names[i], person_names[j]
                for e1 in person_embeddings[p1]:
                    for e2 in person_embeddings[p2]:
                        imposter_dists.append(self.cosine_distance(e1, e2))

        if len(imposter_dists) > max_pairs_per_type:
            imposter_dists = list(self.rng.choice(imposter_dists, size=max_pairs_per_type, replace=False))

        # If imposter pairs are sparse, synthesize orthogonal samples
        if not imposter_dists and person_names:
            sample_dim = len(next(iter(person_embeddings.values()))[0])
            for _ in range(max(10, len(genuine_dists))):
                rnd_vec = self.rng.normal(0, 1, size=sample_dim).astype(np.float32)
                rnd_vec /= np.linalg.norm(rnd_vec)
                base = person_embeddings[person_names[0]][0]
                imposter_dists.append(self.cosine_distance(base, rnd_vec))

        distances.extend(imposter_dists)
        labels.extend([0] * len(imposter_dists))

        return np.array(distances, dtype=np.float32), np.array(labels, dtype=np.int32)

    def evaluate_threshold(self, distances: np.ndarray, labels: np.ndarray, threshold: float) -> ThresholdMetrics:
        """Compute verification performance metrics for a specific distance threshold."""
        preds = (distances <= threshold).astype(np.int32)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))

        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = far
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
        accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0.0
        youden = tpr - fpr

        return ThresholdMetrics(
            threshold=float(threshold),
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            far=float(far),
            frr=float(frr),
            tpr=float(tpr),
            fpr=float(fpr),
            precision=float(precision),
            f1_score=float(f1),
            accuracy=float(accuracy),
            youden_j=float(youden),
        )

    def find_optimal_threshold(self, distances: np.ndarray, labels: np.ndarray) -> Tuple[float, ThresholdMetrics]:
        """Sweep candidate thresholds and select optimal operating point based on target_metric."""
        best_t = 0.48
        best_metric = None
        best_val = -float("inf")
        min_eer_diff = float("inf")

        for t in self.thresholds:
            m = self.evaluate_threshold(distances, labels, t)

            if self.target_metric == "eer":
                diff = abs(m.far - m.frr)
                if diff < min_eer_diff:
                    min_eer_diff = diff
                    best_t = t
                    best_metric = m
            elif self.target_metric == "f1":
                if m.f1_score > best_val:
                    best_val = m.f1_score
                    best_t = t
                    best_metric = m
            elif self.target_metric == "youden":
                if m.youden_j > best_val:
                    best_val = m.youden_j
                    best_t = t
                    best_metric = m
            elif self.target_metric == "high_security":
                # Maximize TPR subject to FAR <= 0.01 (1%)
                if m.far <= 0.01 and m.tpr > best_val:
                    best_val = m.tpr
                    best_t = t
                    best_metric = m
            else:  # default accuracy
                if m.accuracy > best_val:
                    best_val = m.accuracy
                    best_t = t
                    best_metric = m

        if best_metric is None:
            best_metric = self.evaluate_threshold(distances, labels, best_t)

        return float(best_t), best_metric

    def run_kfold_calibration(
        self, person_embeddings: Dict[str, List[np.ndarray]]
    ) -> CalibrationReport:
        """Run Stratified K-Fold Cross Validation across genuine/imposter pairs."""
        distances, labels = self.generate_pairs(person_embeddings)
        total_samples = len(labels)
        if total_samples < 4:
            raise ValueError(f"Insufficient samples for K-Fold calibration: got {total_samples} pairs")

        # Stratified K-Fold partitioning
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        self.rng.shuffle(pos_idx)
        self.rng.shuffle(neg_idx)

        k = min(self.k_folds, len(pos_idx), len(neg_idx))
        if k < 2:
            k = 2

        pos_folds = np.array_split(pos_idx, k)
        neg_folds = np.array_split(neg_idx, k)

        fold_results: List[FoldResult] = []
        optimal_thresholds: List[float] = []

        for fold_i in range(k):
            # Validation indices
            val_idx = np.concatenate([pos_folds[fold_i], neg_folds[fold_i]])
            # Training indices
            train_pos = np.concatenate([pos_folds[j] for j in range(k) if j != fold_i])
            train_neg = np.concatenate([neg_folds[j] for j in range(k) if j != fold_i])
            train_idx = np.concatenate([train_pos, train_neg])

            train_dists, train_lbls = distances[train_idx], labels[train_idx]
            val_dists, val_lbls = distances[val_idx], labels[val_idx]

            # Find optimal threshold on Train split
            train_opt_t, _ = self.find_optimal_threshold(train_dists, train_lbls)
            optimal_thresholds.append(train_opt_t)

            # Evaluate on held-out Validation split
            val_m = self.evaluate_threshold(val_dists, val_lbls, train_opt_t)

            fold_results.append(
                FoldResult(
                    fold_index=fold_i + 1,
                    train_optimal_threshold=round(train_opt_t, 4),
                    val_accuracy=round(val_m.accuracy, 4),
                    val_far=round(val_m.far, 4),
                    val_frr=round(val_m.frr, 4),
                    val_f1=round(val_m.f1_score, 4),
                    val_tp=val_m.tp,
                    val_fp=val_m.fp,
                    val_tn=val_m.tn,
                    val_fn=val_m.fn,
                )
            )

        rec_threshold = float(np.mean(optimal_thresholds))
        thresh_std = float(np.std(optimal_thresholds))

        # Overall ROC curve points on all pairs
        roc_curve = []
        all_metrics = [self.evaluate_threshold(distances, labels, t) for t in self.thresholds]
        # Sort by FPR for AUC calculation
        sorted_m = sorted(all_metrics, key=lambda x: x.fpr)
        for m in sorted_m[::max(1, len(sorted_m) // 30)]:
            roc_curve.append({
                "threshold": round(m.threshold, 3),
                "fpr": round(m.fpr, 4),
                "tpr": round(m.tpr, 4),
                "far": round(m.far, 4),
                "frr": round(m.frr, 4),
                "f1": round(m.f1_score, 4),
            })

        # Calculate AUC via trapezoidal integration anchored at (0, 0) and (1, 1)
        roc_pts = [(0.0, 0.0)] + [(m.fpr, m.tpr) for m in sorted_m] + [(1.0, 1.0)]
        # Deduplicate and sort by fpr
        roc_pts = sorted(list(set(roc_pts)), key=lambda p: (p[0], p[1]))
        fpr_pts = [p[0] for p in roc_pts]
        tpr_pts = [p[1] for p in roc_pts]
        auc = float(np.trapezoid(tpr_pts, fpr_pts)) if hasattr(np, "trapezoid") else float(np.trapz(tpr_pts, fpr_pts))
        auc = round(float(np.clip(abs(auc), 0.0, 1.0)), 4)

        # Global EER estimation
        best_eer_t, eer_metric = self.find_optimal_threshold(distances, labels)
        est_eer = (eer_metric.far + eer_metric.frr) / 2.0

        return CalibrationReport(
            target_metric=self.target_metric,
            k_folds=k,
            num_persons=len(person_embeddings),
            num_genuine_pairs=int(len(pos_idx)),
            num_imposter_pairs=int(len(neg_idx)),
            recommended_threshold=round(rec_threshold, 4),
            threshold_std=round(thresh_std, 4),
            mean_val_accuracy=round(float(np.mean([f.val_accuracy for f in fold_results])), 4),
            mean_val_far=round(float(np.mean([f.val_far for f in fold_results])), 4),
            mean_val_frr=round(float(np.mean([f.val_frr for f in fold_results])), 4),
            mean_val_f1=round(float(np.mean([f.val_f1 for f in fold_results])), 4),
            estimated_eer=round(float(est_eer), 4),
            auc_roc=round(float(auc), 4),
            fold_details=[asdict(f) for f in fold_results],
            roc_curve=roc_curve,
        )


def calibrate_from_db(db_path: str, k_folds: int = 5, target_metric: str = "eer") -> CalibrationReport:
    """Load embeddings from SQLite Database and perform K-Fold calibration."""
    from db import DatabaseManager

    db = DatabaseManager(db_path=db_path)
    persons = db.list_persons()
    if not persons:
        raise ValueError("Database contains no enrolled persons.")

    person_embeddings: Dict[str, List[np.ndarray]] = {}
    with db.get_connection() as conn:
        for p in persons:
            p_id = p["id"]
            name = p["name"]
            cur = conn.execute("SELECT vector FROM face_embeddings WHERE person_id = ?", (p_id,))
            rows = cur.fetchall()
            embs = []
            for (blob,) in rows:
                if blob:
                    v = np.frombuffer(blob, dtype=np.float32)
                    norm = np.linalg.norm(v)
                    if norm > 1e-8:
                        v = v / norm
                    embs.append(v)
            if embs:
                person_embeddings[name] = embs

    calibrator = ThresholdCalibrator(k_folds=k_folds, target_metric=target_metric)
    return calibrator.run_kfold_calibration(person_embeddings)


def main():
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation Threshold Calibration for Face Authorization.")
    parser.add_argument("--db", default=os.getenv("SQLITE_DB_PATH", "data/face_auth.db"), help="Path to SQLite DB")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--metric", choices=["eer", "f1", "youden", "accuracy", "high_security"], default="eer",
                        help="Optimization objective (default: eer)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    try:
        report = calibrate_from_db(args.db, k_folds=args.k_folds, target_metric=args.metric)
    except Exception as e:
        print(f"[error] Calibration failed: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print("\n=======================================================")
        print("  FACE VERIFICATION K-FOLD THRESHOLD CALIBRATION")
        print("=======================================================")
        print(f"  Target Metric         : {report.target_metric.upper()}")
        print(f"  Cross-Validation Folds: {report.k_folds}")
        print(f"  Enrolled Persons      : {report.num_persons}")
        print(f"  Genuine Pairs         : {report.num_genuine_pairs}")
        print(f"  Imposter Pairs        : {report.num_imposter_pairs}")
        print(f"  ---------------------------------------------------")
        print(f"  RECOMMENDED THRESHOLD : {report.recommended_threshold:.4f} (std: {report.threshold_std:.4f})")
        print(f"  Mean Val Accuracy     : {report.mean_val_accuracy * 100:.2f}%")
        print(f"  Mean Val FAR          : {report.mean_val_far * 100:.2f}%")
        print(f"  Mean Val FRR          : {report.mean_val_frr * 100:.2f}%")
        print(f"  Estimated EER         : {report.estimated_eer * 100:.2f}%")
        print(f"  ROC AUC               : {report.auc_roc:.4f}")
        print("=======================================================\n")


if __name__ == "__main__":
    main()
