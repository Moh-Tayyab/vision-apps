"""
Carton-Only Detector Adapter (Phase 1).

Wraps an Ultralytics segmentation or detection model to emit canonical,
strictly carton-only detections. Automatically rejects any non-carton classes
(such as person, vehicle, etc.), handles corrupted/empty frames gracefully,
and preserves mask contours and normalized bounding boxes.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from apps.truck_loading_detection.model_pipeline.mask_refinement import CartonMaskRefiner
except ImportError:
    from model_pipeline.mask_refinement import CartonMaskRefiner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CanonicalCartonDetection:
    """Canonical representation of a single carton detected in a camera frame."""

    camera_id: str
    frame_idx: int
    timestamp_sec: float
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in original frame pixel coords
    confidence: float
    class_name: str = "carton"
    mask_polygon: Optional[List[Tuple[float, float]]] = None  # Refined polygon vertices if available
    raw_mask_polygon: Optional[List[Tuple[float, float]]] = None  # Original raw model mask
    area_px: float = 0.0
    handling_state: str = "UNKNOWN"
    is_occluded_by_worker: bool = False

    def __post_init__(self) -> None:
        if self.class_name != "carton":
            raise ValueError(f"CanonicalCartonDetection only accepts 'carton', got '{self.class_name}'")
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Degenerate bounding box: {self.bbox}")

    @property
    def center(self) -> Tuple[float, float]:
        """Centroid of the detection bounding box (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class CartonDetectorAdapter:
    """
    Adapter for loading and running the existing carton-only model weights.
    
    Guarantees:
    - Never modifies checkpoint weights on disk.
    - Validates model task and class mappings.
    - Strictly filters out any class other than 'carton'.
    - Handles empty frames, None inputs, or inference exceptions without crashing.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: str = "cpu",
        preferred_profile: str = "auto",
        worker_model_path: Optional[str] = None,
        worker_conf: float = 0.25,
        enable_worker_disentanglement: bool = True,
        content_crop_margin_px: int = 4,
        content_crop_min_coverage: float = 0.92,
        enable_content_crop: bool = False,
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.device = str(os.environ.get("CARTON_DEVICE", device))
        self.preferred_profile = str(preferred_profile)

        # Telemetry metrics (Phase 2 Requirement 6)
        self.total_frames_received: int = 0
        self.total_frames_processed: int = 0
        self.dropped_frames_count: int = 0
        self.last_latency_ms: float = 0.0
        self._latencies: List[float] = []

        if not Path(self.model_path).exists():
            app_dir = Path(__file__).resolve().parent.parent
            for candidate in (
                app_dir / self.model_path,
                app_dir / Path(self.model_path).name,
                Path.cwd() / self.model_path,
                Path.cwd() / Path(self.model_path).name,
            ):
                if candidate.exists():
                    self.model_path = str(candidate)
                    break
            else:
                raise FileNotFoundError(f"Carton model checkpoint not found: {self.model_path}")

        # Lazy load Ultralytics YOLO & determine profile
        from ultralytics import YOLO
        import torch

        # Automatic hardware device resolution
        if self.device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "CUDA requested ('%s') but torch.cuda.is_available() is False. Falling back to CPU.",
                self.device,
            )
            self.device = "cpu"

        # Maximize PyTorch CPU intra-op parallelism across all available cores if running on CPU
        if self.device == "cpu":
            try:
                n_threads = max(4, min(8, os.cpu_count() or 4))
                if torch.get_num_threads() < n_threads:
                    torch.set_num_threads(n_threads)
            except Exception:
                pass

        self.active_profile = "pytorch_cpu"
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            self.active_profile = "gpu_tensorrt" if self.model_path.endswith(".engine") else "gpu_cuda"

        try:
            self._model = YOLO(self.model_path)
            self._task = getattr(self._model, "task", "segment")
            self._names = getattr(self._model, "names", {0: "carton"})
        except Exception as e:
            logger.error("Failed to load YOLO checkpoint '%s': %s", self.model_path, e)
            raise RuntimeError(f"Could not initialize detector from {self.model_path}: {e}") from e

        # Validate that 'carton' class exists
        self._carton_class_ids = [
            cid for cid, name in self._names.items() if str(name).lower() in ("carton", "box")
        ]
        if not self._carton_class_ids:
            logger.warning(
                "Model at '%s' has class names %s; none match 'carton' or 'box'. "
                "Any non-carton detections will be strictly rejected.",
                self.model_path,
                self._names,
            )

        self.refiner = CartonMaskRefiner()
        self._worker_model = None
        self._worker_model_failed = False
        self.worker_conf = float(worker_conf)
        self.enable_worker_disentanglement = bool(enable_worker_disentanglement)
        self.content_crop_margin_px = int(content_crop_margin_px)
        self.content_crop_min_coverage = float(content_crop_min_coverage)
        self.enable_content_crop = bool(enable_content_crop)
        self.worker_model_path = self._resolve_worker_model_path(worker_model_path)

    def _resolve_worker_model_path(self, worker_model_path: Optional[str]) -> Optional[str]:
        """Locate the COCO worker-silhouette segmentation checkpoint used for mask disentanglement."""
        if worker_model_path:
            return str(worker_model_path) if Path(worker_model_path).exists() else None

        app_dir = Path(__file__).resolve().parent.parent
        repo_root = app_dir.parent.parent
        for candidate in (
            repo_root / "yolo11n-seg.pt",
            app_dir / "yolo11n-seg.pt",
            Path.cwd() / "yolo11n-seg.pt",
        ):
            if candidate.exists():
                return str(candidate)
        return None

    def _worker_polygons(self, frame: np.ndarray) -> List[Any]:
        """Segment worker silhouettes so carton masks can be disentangled from people."""
        return self._worker_polygons_batch([frame])[0]

    def _worker_polygons_batch(self, frames: Sequence[np.ndarray]) -> List[List[Any]]:
        """Segment worker silhouettes for several frames in a single forward pass."""
        empty = [[] for _ in frames]
        if not self.enable_worker_disentanglement or self.worker_model_path is None:
            return empty
        if not frames:
            return empty

        if self._worker_model is None:
            if self._worker_model_failed:
                return empty
            try:
                from ultralytics import YOLO

                self._worker_model = YOLO(self.worker_model_path)
            except Exception as e:
                self._worker_model_failed = True
                logger.warning(
                    "Worker segmentation model unavailable at '%s': %s",
                    self.worker_model_path,
                    e,
                )
                return empty

        try:
            results = self._worker_model.predict(
                list(frames),
                classes=[0],
                conf=self.worker_conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.debug("Worker segmentation failed: %s", e)
            return empty

        polygons: List[List[Any]] = []
        for result in results or []:
            frame_polys: List[Any] = []
            masks = getattr(result, "masks", None)
            if masks is not None and getattr(masks, "xy", None) is not None:
                for poly in masks.xy:
                    if poly is not None and len(poly) >= 3:
                        frame_polys.append(poly)
            polygons.append(frame_polys)
        while len(polygons) < len(frames):
            polygons.append([])
        return polygons

    @staticmethod
    def _offset_worker_polygons(
        polygons: Optional[Sequence[Any]], x_offset: int
    ) -> Optional[List[Any]]:
        """Shift worker silhouettes from crop coordinates back to full-frame coordinates."""
        if not polygons or not x_offset:
            return list(polygons) if polygons else polygons
        shifted: List[Any] = []
        for poly in polygons:
            arr = np.asarray(poly, dtype=np.float32).copy()
            if arr.ndim == 2 and arr.shape[1] == 2:
                arr[:, 0] += x_offset
            shifted.append(arr)
        return shifted

    def _content_crop(self, frame: np.ndarray) -> Tuple[int, int]:
        """Horizontal bounds of the non-black image content, padded by a small margin.

        Normalized multi-camera videos are padded to a common canvas, so a
        portrait camera occupies only a narrow strip of the frame. Running
        inference on the padded frame squeezes that content into a tiny part
        of the model input and destroys detection confidence, so the content
        is cropped before inference and the coordinates are mapped back.
        """
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0 or not self.enable_content_crop:
            return 0, w
        try:
            gray = frame.max(axis=2) if frame.ndim == 3 else frame
            cols = np.where(gray.max(axis=0) > 12)[0]
        except Exception:
            return 0, w
        if cols.size == 0:
            return 0, w
        x0 = max(0, int(cols.min()) - self.content_crop_margin_px)
        x1 = min(w, int(cols.max()) + 1 + self.content_crop_margin_px)
        if (x1 - x0) >= (w * self.content_crop_min_coverage):
            return 0, w
        return x0, x1

    @property
    def memory_usage_mb(self) -> float:
        """Process resident memory footprint in megabytes."""
        try:
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
        except Exception:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    @property
    def mean_latency_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return float(np.mean(self._latencies[-100:]))

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns comprehensive Phase 2 hardware & execution profile telemetry."""
        return {
            "active_profile": self.active_profile,
            "device": self.device,
            "total_frames_received": self.total_frames_received,
            "total_frames_processed": self.total_frames_processed,
            "dropped_frames_count": self.dropped_frames_count,
            "last_latency_ms": round(self.last_latency_ms, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 1),
        }

    @property
    def task(self) -> str:
        return self._task

    @property
    def class_names(self) -> Dict[int, str]:
        return dict(self._names)

    def detect(
        self,
        frame: Optional[np.ndarray],
        camera_id: str,
        frame_idx: int,
        timestamp_sec: float,
        conf_override: Optional[float] = None,
        worker_polygons: Optional[Sequence[Any]] = None,
    ) -> List[CanonicalCartonDetection]:
        """
        Run carton-only inference on a single video frame.

        Returns an empty list safely if frame is empty, None, or on error.
        Tracks dropped frames and latency telemetry.
        """
        self.total_frames_received += 1

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.dropped_frames_count += 1
            return []

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            self.dropped_frames_count += 1
            return []

        conf = float(conf_override if conf_override is not None else self.conf_threshold)

        crop_x0, crop_x1 = self._content_crop(frame)
        infer_frame = frame[:, crop_x0:crop_x1] if crop_x1 > crop_x0 else frame

        t_start = time.perf_counter()
        try:
            results = self._model.predict(
                source=infer_frame,
                conf=conf,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            self.dropped_frames_count += 1
            logger.error("Detector inference failed for camera '%s' frame %d: %s", camera_id, frame_idx, e)
            return []

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        self.last_latency_ms = t_elapsed
        self._latencies.append(t_elapsed)
        self.total_frames_processed += 1

        if not results:
            return []

        if worker_polygons is None:
            worker_polygons = self._offset_worker_polygons(
                self._worker_polygons(infer_frame), crop_x0
            )

        return self._parse_results(
            results[0],
            camera_id,
            frame_idx,
            timestamp_sec,
            w,
            h,
            worker_polygons=worker_polygons,
            x_offset=crop_x0,
        )

    def detect_batch(
        self,
        frames: Dict[str, Optional[np.ndarray]],
        frame_idx: int,
        timestamps: Dict[str, float],
        conf_override: Optional[float] = None,
    ) -> Dict[str, List[CanonicalCartonDetection]]:
        """
        Executes a single multi-camera batched forward pass across all active camera frames.
        Avoids redundant per-camera Python/PyTorch inference invocations.
        """
        valid_cams = []
        frame_list = []
        dims = {}
        for cid, f in frames.items():
            self.total_frames_received += 1
            if f is not None and isinstance(f, np.ndarray) and f.size > 0:
                h, w = f.shape[:2]
                if h > 0 and w > 0:
                    valid_cams.append(cid)
                    frame_list.append(f)
                    dims[cid] = (w, h)
                else:
                    self.dropped_frames_count += 1
            else:
                self.dropped_frames_count += 1

        if not frame_list:
            return {cid: [] for cid in frames}

        conf = float(conf_override if conf_override is not None else self.conf_threshold)
        t_start = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame_list,
                conf=conf,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            self.dropped_frames_count += len(frame_list)
            logger.error("Batched detector inference failed on frame %d: %s", frame_idx, e)
            return {cid: [] for cid in frames}

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        self.last_latency_ms = t_elapsed
        self._latencies.append(t_elapsed)
        self.total_frames_processed += len(frame_list)

        detections: Dict[str, List[CanonicalCartonDetection]] = {cid: [] for cid in frames}
        for idx, cid in enumerate(valid_cams):
            if idx < len(results):
                w, h = dims[cid]
                ts = timestamps.get(cid, 0.0)
                detections[cid] = self._parse_results(results[idx], cid, frame_idx, ts, w, h)

        return detections

    def _parse_results(
        self,
        res: Any,
        camera_id: str,
        frame_idx: int,
        timestamp_sec: float,
        w: int,
        h: int,
        worker_polygons: Optional[Sequence[Any]] = None,
        x_offset: int = 0,
    ) -> List[CanonicalCartonDetection]:
        detections: List[CanonicalCartonDetection] = []
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        masks = getattr(res, "masks", None)
        has_masks = masks is not None and getattr(masks, "xy", None) is not None

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            cls_name = self._names.get(cls_id, "unknown").lower()

            # STRICT REJECTION: Reject any non-carton class (e.g. person, vehicle, etc.)
            if cls_name not in ("carton", "box"):
                continue

            score = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]) + x_offset, float(xyxy[1]), float(xyxy[2]) + x_offset, float(xyxy[3])

            # Clip to image boundaries
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))

            if (x2 - x1) <= 2.0 or (y2 - y1) <= 2.0:
                continue

            area = (x2 - x1) * (y2 - y1)

            poly: Optional[List[Tuple[float, float]]] = None
            raw_poly_pts: Optional[List[Tuple[float, float]]] = None
            handling_state = "UNKNOWN"
            is_occluded = False

            if has_masks and idx < len(masks.xy):
                raw_poly = masks.xy[idx]
                if raw_poly is not None and len(raw_poly) >= 3:
                    raw_poly_pts = [
                        (float(pt[0]) + x_offset, float(pt[1])) for pt in raw_poly
                    ]
                    if self.refiner.worker_false_positive(
                        raw_polygon=raw_poly_pts,
                        image_shape=(h, w),
                        worker_polygons=worker_polygons,
                        confidence=score,
                    ):
                        continue
                    refined = self.refiner.refine_carton_mask(
                        raw_polygon=raw_poly_pts,
                        image_shape=(h, w),
                        worker_polygons=worker_polygons,
                        confidence=score,
                        timestamp_sec=timestamp_sec,
                    )
                    if refined is not None:
                        poly = [(float(pt[0]), float(pt[1])) for pt in refined.polygon_pts]
                        x1, y1, x2, y2 = refined.bbox
                        area = refined.area_px
                        handling_state = refined.handling_state
                        is_occluded = refined.is_occluded_by_worker
                    else:
                        poly = raw_poly_pts

            det = CanonicalCartonDetection(
                camera_id=str(camera_id),
                frame_idx=int(frame_idx),
                timestamp_sec=float(timestamp_sec),
                bbox=(x1, y1, x2, y2),
                confidence=score,
                class_name="carton",
                mask_polygon=poly,
                raw_mask_polygon=raw_poly_pts,
                area_px=area,
                handling_state=handling_state,
                is_occluded_by_worker=is_occluded,
            )
            detections.append(det)

        return detections

    def detect_batch(
        self,
        frames: Dict[str, Optional[np.ndarray]],
        frame_idx: int,
        timestamps_sec: Dict[str, float],
        conf_override: Optional[float] = None,
        worker_polygons_by_cam: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> Dict[str, List[CanonicalCartonDetection]]:
        """
        Runs batched multi-camera inference across synchronized frame inputs in a single forward pass.
        Dramatically improves throughput and GPU tensor core utilization compared to sequential single calls.
        """
        output: Dict[str, List[CanonicalCartonDetection]] = {cid: [] for cid in frames}
        valid_cids: List[str] = []
        valid_frames: List[np.ndarray] = []
        dims: Dict[str, Tuple[int, int]] = {}

        for cid, f in frames.items():
            self.total_frames_received += 1
            if f is None or not isinstance(f, np.ndarray) or f.size == 0:
                self.dropped_frames_count += 1
                continue
            h, w = f.shape[:2]
            if h <= 0 or w <= 0:
                self.dropped_frames_count += 1
                continue
            valid_cids.append(cid)
            valid_frames.append(f)
            dims[cid] = (w, h)

        if not valid_frames:
            return output

        conf = float(conf_override if conf_override is not None else self.conf_threshold)

        crop_bounds: List[Tuple[int, int]] = []
        infer_frames: List[np.ndarray] = []
        for f in valid_frames:
            x0, x1 = self._content_crop(f)
            crop_bounds.append((x0, x1))
            infer_frames.append(f[:, x0:x1] if x1 > x0 else f)

        t_start = time.perf_counter()
        try:
            results = self._model.predict(
                source=infer_frames,
                conf=conf,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            logger.error("Batched detector inference failed for frame %d: %s. Falling back to serial.", frame_idx, e)
            # Fallback to serial
            for cid in valid_cids:
                output[cid] = self.detect(
                    frame=frames[cid],
                    camera_id=cid,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamps_sec.get(cid, 0.0),
                    conf_override=conf_override,
                )
            return output

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        self.last_latency_ms = t_elapsed / max(1, len(valid_frames))
        self._latencies.append(self.last_latency_ms)
        self.total_frames_processed += len(valid_frames)

        worker_polys_by_cam: Dict[str, Sequence[Any]] = {}
        if worker_polygons_by_cam is None and infer_frames:
            batched = self._worker_polygons_batch(infer_frames)
            worker_polys_by_cam = {
                cid: self._offset_worker_polygons(batched[i], crop_bounds[i][0])
                for i, cid in enumerate(valid_cids)
            }

        for i, cid in enumerate(valid_cids):
            w, h = dims[cid]
            t_sec = timestamps_sec.get(cid, 0.0)
            res = results[i] if i < len(results) else None
            w_polys = worker_polys_by_cam.get(cid) if worker_polys_by_cam else None
            if w_polys is None:
                w_polys = worker_polys_by_cam.get(cid, [])
            if res is not None:
                output[cid] = self._parse_results(
                    res,
                    cid,
                    frame_idx,
                    t_sec,
                    w,
                    h,
                    worker_polygons=w_polys,
                    x_offset=crop_bounds[i][0],
                )

        return output

