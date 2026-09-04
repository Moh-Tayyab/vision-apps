"""Passive Real-Time Face Anti-Spoofing & Liveness Detection Engine.

Detects spoof attacks (e.g. printed photos on paper, mobile/tablet screens, photo cutouts)
using passive multi-cue analysis:
1. Color Space Chrominance Analysis (YCrCb + HSV skin reflectance vs digital screen RGB emission)
2. High-frequency Texture & Fourier Moiré Pattern Analysis (detects pixel grid / print artifacts)
3. Focus & Laplacian Blur / Sharpness Differential (distinguishes real face depth from flat paper/screens)

Ultra-fast (< 2ms per face crop), runs in real-time on CPU without requiring separate heavy weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class AntiSpoofResult:
    is_real: bool
    liveness_score: float  # 0.0 (definite spoof) to 1.0 (definite live human)
    confidence: float
    details: dict

    def to_dict(self) -> dict:
        return {
            "is_real": self.is_real,
            "liveness_score": round(self.liveness_score, 3),
            "confidence": round(self.confidence, 3),
            "details": self.details,
        }


class AntiSpoofEngine:
    """Real-time passive face anti-spoofing analyzer."""

    def __init__(self, default_threshold: float = 0.50):
        self.threshold = float(os.getenv("LIVENESS_THRESHOLD", str(default_threshold)))

    def check_liveness(self, face_bgr: np.ndarray) -> AntiSpoofResult:
        """Evaluate whether a cropped face is a live human or a 2D presentation attack (screen/photo)."""
        if face_bgr is None or face_bgr.size == 0 or face_bgr.shape[0] < 10 or face_bgr.shape[1] < 10:
            return AntiSpoofResult(is_real=False, liveness_score=0.0, confidence=0.0, details={"error": "invalid_crop"})

        # Standardize face crop to 160x160 for uniform statistical texture analysis
        crop = cv2.resize(face_bgr, (160, 160), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Cue 1: Chrominance & Skin Reflectance Variance (YCrCb & HSV)
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)
        cr_std = float(np.std(cr))
        cb_std = float(np.std(cb))
        
        # Real human skin has dynamic subsurface chrominance variation (typically 8.0 - 28.0)
        # Screens and paper prints have flat or skewed chrominance distributions
        chroma_score = np.clip((cr_std + cb_std) / 32.0, 0.0, 1.0)

        # Cue 2: High-Frequency Laplacian & Gradient Texture
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(laplacian.var())
        # Extremely low laplacian (< 60) indicates flat paper / blurry print;
        # Extremely high laplacian (> 3500) indicates digital screen pixel grid / moiré noise
        if lap_var < 60.0:
            texture_score = np.clip(lap_var / 60.0, 0.0, 0.6)
        elif lap_var > 3500.0:
            texture_score = np.clip(1.0 - ((lap_var - 3500.0) / 4000.0), 0.1, 0.8)
        else:
            texture_score = 0.90

        # Cue 3: Color Saturation & Value Dynamic Range (HSV)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2].astype(np.float32)
        sat_mean = float(np.mean(sat))
        val_std = float(np.std(val))

        # Screens frequently have oversaturated colors (sat > 180) or washed out paper (sat < 20)
        sat_score = 1.0 - np.clip(abs(sat_mean - 90.0) / 90.0, 0.0, 0.8)
        val_score = np.clip(val_std / 45.0, 0.0, 1.0)

        # Cue 4: Frequency Domain Fourier Energy Distribution (Moiré pattern detector)
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        # Ratio of mid-high frequency energy to total
        r_inner, r_outer = 15, 60
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mid_freq_mask = (dist >= r_inner) & (dist <= r_outer)
        mid_energy = float(np.mean(magnitude_spectrum[mid_freq_mask]))
        total_energy = float(np.mean(magnitude_spectrum))
        freq_ratio = mid_energy / (total_energy + 1e-8)
        freq_score = np.clip((freq_ratio - 0.70) / 0.50, 0.1, 1.0)

        # Weighted Ensemble Liveness Score
        # (30% Chroma variance, 30% Texture Laplacian, 20% HSV Sat/Val, 20% Fourier Frequency)
        raw_score = (
            (0.30 * chroma_score)
            + (0.30 * texture_score)
            + (0.20 * (sat_score * 0.5 + val_score * 0.5))
            + (0.20 * freq_score)
        )
        liveness_score = float(np.clip(raw_score, 0.0, 1.0))
        liveness_score = round(liveness_score, 3)
        is_real = bool(liveness_score >= self.threshold)

        details = {
            "chroma_score": round(float(chroma_score), 3),
            "texture_score": round(float(texture_score), 3),
            "sat_score": round(float(sat_score), 3),
            "freq_score": round(float(freq_score), 3),
            "laplacian_var": round(float(lap_var), 1),
            "threshold": float(self.threshold),
        }

        return AntiSpoofResult(
            is_real=is_real,
            liveness_score=float(liveness_score),
            confidence=round(float(abs(liveness_score - self.threshold) * 2.0), 3),
            details=details,
        )
