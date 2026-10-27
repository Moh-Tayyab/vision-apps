#!/usr/bin/env python3
"""
Multi-Angle Video Processing & Fusion Pipeline (N-camera, offset-synchronised).

Architecture (3 phases, so YOLO never runs twice on the same frames):

  Phase A - ANALYSE : each camera is decoded independently. YOLO11 tracking +
                      carton-holding detection produce a per-frame detection
                      list (original pixel coordinates). A local tripwire pass
                      over that list yields the camera's LOCAL event sequence.
                      Results are cached on disk.

  Phase B - ALIGN   : the local event sequences of every camera are aligned
                      against a reference camera (multi_angle_sync). This
                      recovers the recording start offset of each feed. Offsets
                      can be overridden from the command line.

  Phase C - RENDER  : a single global timeline is rendered. At every global
                      tick each camera contributes the frame that corresponds
                      to that instant (offset applied), detections are drawn,
                      and the fusion engine fuses the angles. Output: a
                      side-by-side / N-way split MP4 with a live HUD plus a
                      full audit-trail JSON report.

Coordinate convention:
  All detection boxes and tripwire positions are kept in ORIGINAL video pixels.
  Pane resizing for rendering is a display-only transform, so the drawn
  tripwire always matches the tripwire the tracker actually used.
"""

import os
import sys
import json
import time
import pickle
import hashlib
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_angle_fusion_engine import (
    CameraAngleConfig,
    AngleTripwireTracker,
    MultiAngleFusionEngine
)
from multi_angle_sync import resolve_offsets
from carton_preprocessor import CartonHoldingDetector

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 6)))

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratch", "analysis_cache")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraSpec:
    """One camera feed participating in the fusion."""
    camera_id: str
    video_path: str
    role: str = "corroborator"           # "width" | "height" | "corroborator"
    angle_type: str = "side_profile"
    line_orientation: str = "vertical"   # "vertical" (X line) | "horizontal" (Y line)
    line_pos_frac: float = 0.5           # tripwire position as a fraction of the frame
    direction: str = "left_to_right"     # left_to_right | right_to_left | up_to_down | down_to_up
    hysteresis: int = 15                # pixels (absolute, legacy)
    hysteresis_frac: float = 0.02       # fraction of the counted axis (resolution independent)
    cooldown_frames: int = 15
    roi_mode: str = "wide"              # "wide" | "column" | "row" | "tight"
    slice_frac: float = 0.30
    reference_carton_area: Optional[int] = None   # px area of ONE carton in this camera
    roi_y_frac: Optional[Tuple[float, float]] = None  # (lo, hi) fraction of person height for ROI
    count_directions: Optional[List[str]] = None   # e.g. ["IN"] for a one-way flow
    start_offset_sec: float = 0.0        # global time at which this feed started
    offset_locked: bool = False         # True = do not auto-overwrite the offset
    imgsz: int = 640                    # YOLO inference resolution; 640 for door-lage CCTV (small cartons)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CameraSpec":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class CameraAnalysis:
    """Phase A output for a single camera."""
    camera_id: str
    video_path: str
    width: int
    height: int
    fps: float
    frame_count: int
    line_pos: int = 0
    hysteresis: int = 15
    frames: List[List[Tuple[int, Tuple[float, float, float, float], str, float, int]]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


# ---------------------------------------------------------------------------
# Phase A - per camera analysis
# ---------------------------------------------------------------------------

def _cache_key(spec: CameraSpec, stride: int, conf: float, model_tag: str = "") -> str:
    """Cache key covers detection inputs and model, so tripwire tweaks are free."""
    st = os.stat(spec.video_path)
    raw = (f"{os.path.abspath(spec.video_path)}|{st.st_mtime}|{st.st_size}|{stride}|{conf}"
           f"|{spec.roi_mode}|{spec.slice_frac}|{spec.direction}|{spec.reference_carton_area}"
           f"|{spec.roi_y_frac}|{spec.imgsz}|{model_tag}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def analyse_camera(
    spec: CameraSpec,
    model: YOLO,
    stride: int = 2,
    conf: float = 0.22,
    iou: float = 0.45,
    imgsz: int = 640,
    use_cache: bool = True,
) -> CameraAnalysis:
    """Runs YOLO tracking + carton detection over one video (Phase A)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    is_carton_model = False
    if hasattr(model, "names") and isinstance(model.names, dict):
        if model.names.get(0, "").lower() in ["carton", "box"]:
            is_carton_model = True

    model_tag = "carton_model" if is_carton_model else "person_model"
    key = _cache_key(spec, stride, conf, model_tag)
    stem = os.path.splitext(os.path.basename(spec.video_path))[0]
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    cache_path = os.path.join(CACHE_DIR, f"{spec.camera_id}_{safe_stem}_{key}.pkl")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            blob = pickle.load(fh)
        # The cache is stored as a plain dict so it can be loaded from any
        # entry point (a pickled dataclass would tie the cache to the module
        # it was created in).
        analysis = CameraAnalysis(**blob)
        print(f"[Phase A] {spec.camera_id}: loaded cached detections "
              f"({analysis.frame_count} frames, mode={model_tag})")
    else:
        cap = cv2.VideoCapture(spec.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {spec.video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        detector = CartonHoldingDetector(roi_mode=spec.roi_mode, slice_frac=spec.slice_frac,
                                         reference_carton_area=spec.reference_carton_area,
                                         roi_y_frac=spec.roi_y_frac)
        frames: List[List[Tuple[int, Tuple[float, float, float, float], str, float, int]]] = []
        last_objects: List[Tuple[int, Tuple[float, float, float, float], str, float, int]] = []

        frame_idx = 0
        t0 = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if (frame_idx - 1) % stride == 0:
                    res = model.track(
                        frame,
                        persist=True,
                        classes=[0],
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        tracker="bytetrack.yaml",
                        verbose=False
                    )
                    objects: List[Tuple[int, Tuple[float, float, float, float], str, float, int]] = []
                    if res[0].boxes is not None and res[0].boxes.id is not None:
                        boxes = res[0].boxes.xyxy.cpu().numpy()
                        tids = res[0].boxes.id.cpu().numpy()
                        cfs = res[0].boxes.conf.cpu().numpy()
                        for b, tid, cf in zip(boxes, tids, cfs):
                            if is_carton_model:
                                # DIRECT CARTON TRACKING:
                                # Detected object is a physical carton instance.
                                # Each carton has qty=1 and is tracked individually across the tripwire.
                                objects.append((int(tid), tuple(float(v) for v in b), "carton", float(cf), 1))
                            else:
                                is_holding, _h_conf, _cbox, qty, _dbg = detector.detect_carton(
                                    frame, tuple(b), spec.direction
                                )
                                objects.append((int(tid), tuple(float(v) for v in b), "person", float(cf),
                                                int(qty) if is_holding else 0))
                    last_objects = objects
                frames.append(list(last_objects))
        finally:
            cap.release()

        analysis = CameraAnalysis(
            camera_id=spec.camera_id,
            video_path=spec.video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            frames=frames,
        )
        with open(cache_path, "wb") as fh:
            pickle.dump(asdict(analysis), fh)
        print(f"[Phase A] {spec.camera_id}: {frame_count} frames detected in {time.time() - t0:.1f}s ({model_tag})")

    # Tripwire geometry + local event pass (cheap, recomputed every run).
    line_pos = int((spec.line_pos_frac * analysis.height)
                   if spec.line_orientation == "horizontal"
                   else (spec.line_pos_frac * analysis.width))
    axis_px = analysis.height if spec.line_orientation == "horizontal" else analysis.width
    hysteresis = max(4, int(spec.hysteresis_frac * axis_px))
    analysis.line_pos = line_pos
    analysis.hysteresis = hysteresis

    cfg = CameraAngleConfig(
        camera_id=spec.camera_id,
        angle_type=spec.angle_type,
        line_orientation=spec.line_orientation,
        line_pos=line_pos,
        direction=spec.direction,
        hysteresis=hysteresis,
        cooldown_frames=spec.cooldown_frames,
        role=spec.role,
    )
    tracker = AngleTripwireTracker(cfg)
    for i, objects in enumerate(analysis.frames):
        tracker.update(objects, frame_idx=i + 1, timestamp_sec=i / analysis.fps)
    analysis.events = tracker.events

    print(f"[Phase A] {spec.camera_id}: tripwire@{line_pos}px "
          f"({spec.line_orientation}, hyst {hysteresis}px), "
          f"{len(analysis.events)} local events")
    return analysis


# ---------------------------------------------------------------------------
# Phase C - aligned rendering + fusion
# ---------------------------------------------------------------------------

class _CameraReader:
    """Sequential frame reader that can be addressed on a global timeline."""

    def __init__(self, path: str, fps: float):
        self.cap = cv2.VideoCapture(path)
        self.fps = fps
        self.next_idx = 0
        self.eof = False
        self.frame: Optional[np.ndarray] = None
        self.frame_idx: int = -1

    def seek_to(self, target_idx: int) -> Optional[np.ndarray]:
        if self.eof or target_idx < 0 or target_idx < self.next_idx - 1:
            return self.frame
        while self.next_idx <= target_idx:
            if not self.cap.grab():
                self.eof = True
                break
            if self.next_idx == target_idx:
                ok, fr = self.cap.retrieve()
                if ok:
                    self.frame = fr
                    self.frame_idx = self.next_idx
            self.next_idx += 1
        return self.frame

    def release(self) -> None:
        self.cap.release()


class MultiAnglePipeline:
    """N-camera synchronised fusion pipeline."""

    def __init__(
        self,
        camera_specs: List[CameraSpec],
        model_path: str = "best.pt",
        conf: float = 0.15,
        iou: float = 0.45,
        frame_stride: int = 2,
        imgsz: int = 640,
        target_fps: float = 30.0,
        pane_height: int = 720,
        time_window_sec: float = 0.85,
        use_cache: bool = True,
        hud_metric: str = "count",
        count_directions: Optional[List[str]] = None,
        fusion_mode: Optional[str] = None,
    ):
        self.camera_specs = camera_specs
        app_dir = os.path.dirname(os.path.abspath(__file__))
        cand_best = os.path.join(app_dir, "best.pt")
        if (not model_path or model_path in ("best.pt", "yolo11n.pt")) and os.path.exists(cand_best):
            model_path = cand_best

        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            local = os.path.join(app_dir, model_path)
            if os.path.exists(local):
                model_path = local
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.frame_stride = frame_stride
        self.imgsz = imgsz
        self.target_fps = target_fps
        self.pane_height = pane_height
        self.time_window_sec = time_window_sec
        self.use_cache = use_cache
        self.hud_metric = hud_metric
        if count_directions is None:
            # A per-camera spec may declare the counting directions; the first
            # camera that declares one wins for the whole run.
            for s in camera_specs:
                if s.count_directions:
                    count_directions = list(s.count_directions)
                    break
        self.count_directions = count_directions

        # Detect carton-centric model (class 0 = carton) vs person model
        self.is_carton_model = False
        try:
            _probe = YOLO(model_path)
            if hasattr(_probe, "names") and isinstance(_probe.names, dict):
                if _probe.names.get(0, "").lower() in ("carton", "box"):
                    self.is_carton_model = True
        except Exception:
            pass
        del _probe

        # Default fusion_mode: "carton" for carton models, "wh" for person models
        if fusion_mode is None:
            self.fusion_mode = "carton" if self.is_carton_model else "wh"
        else:
            self.fusion_mode = fusion_mode

        # Carton models typically run at lower confidence; auto-lower if the
        # caller passed a person-model default (>=0.15) and didn't override.
        if self.is_carton_model and conf >= 0.15:
            self.conf = 0.05

        self.model = YOLO(model_path)
        self.analyses: Dict[str, CameraAnalysis] = {}
        self.offsets: Dict[str, float] = {s.camera_id: s.start_offset_sec for s in camera_specs}
        self.sync_report: Dict[str, Any] = {}

    # -- Phase A + B --------------------------------------------------------

    def analyse(self) -> None:
        for spec in self.camera_specs:
            analysis = analyse_camera(
                spec, self.model, stride=self.frame_stride,
                conf=self.conf, iou=self.iou, imgsz=self.imgsz, use_cache=self.use_cache
            )
            self.analyses[spec.camera_id] = analysis

    def align(self, reference: Optional[str] = None, max_offset_sec: float = 30.0,
              tol_sec: float = 1.5, auto: bool = True) -> None:
        per_cam_events = {c: a.events for c, a in self.analyses.items()}
        resolved = resolve_offsets(
            per_cam_events,
            reference_camera_id=reference,
            max_offset_sec=max_offset_sec,
            tol_sec=tol_sec,
            fps=self.target_fps,
        )
        self.sync_report = resolved

        specs = {s.camera_id: s for s in self.camera_specs}
        for cam_id, est in resolved.get("diagnostics", {}).items():
            spec = specs[cam_id]
            if spec.offset_locked:
                print(f"[Phase B] {cam_id}: offset {spec.start_offset_sec:+.2f}s "
                      f"(LOCKED, auto estimate {est['offset_sec']:+.2f}s ignored)")
                self.offsets[cam_id] = spec.start_offset_sec
                continue
            if not auto:
                self.offsets[cam_id] = spec.start_offset_sec
                continue
            self.offsets[cam_id] = est["offset_sec"]
            print(f"[Phase B] {cam_id}: offset {est['offset_sec']:+.2f}s "
                  f"(matched {est['matched_events']} events, "
                  f"confidence {est['confidence']:.2f})")

        ref = resolved.get("reference")
        if ref:
            self.offsets[ref] = specs[ref].start_offset_sec if specs[ref].offset_locked else 0.0
            print(f"[Phase B] reference camera: {ref}")

    # -- Phase C ------------------------------------------------------------

    def _pane_dims(self) -> List[Tuple[int, int, float]]:
        dims = []
        for spec in self.camera_specs:
            a = self.analyses[spec.camera_id]
            scale = self.pane_height / max(1, a.height)
            dims.append((max(1, int(a.width * scale)), self.pane_height, scale))
        return dims

    def _global_timeline(self) -> Tuple[float, float]:
        starts, ends = [], []
        for spec in self.camera_specs:
            a = self.analyses[spec.camera_id]
            off = self.offsets[spec.camera_id]
            starts.append(off)
            ends.append(off + a.duration())
        return min(starts), max(ends)

    @staticmethod
    def _metric_totals(events: List[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Two readings of the same fusion:
          product = W x H  - the load is a stack (the truck-loading case)
          count   = max(W, H) - the cartons sit side by side in one row
        """
        prod = {"in": 0, "out": 0}
        cnt = {"in": 0, "out": 0}
        for ev in events:
            w = max(0, int(ev.get("top_camera_width", 0)))
            h = max(0, int(ev.get("side_camera_height", 0)))
            key = "in" if ev.get("direction") == "IN" else "out"
            prod[key] += w * h
            cnt[key] += max(w, h)
        return prod, cnt

    def render(
        self,
        output_video_path: str = "output_multi_angle_fused.mp4",
        output_json_path: str = "multi_angle_report.json",
        render_video: bool = True,
        title: str = "",
    ) -> Dict[str, Any]:
        pane_dims = self._pane_dims()
        total_width = sum(p[0] for p in pane_dims)
        hud_height = 140
        title_height = 52 if title else 0
        canvas_h = self.pane_height + hud_height + title_height
        canvas_w = total_width

        engine = MultiAngleFusionEngine(time_window_sec=self.time_window_sec,
                                        count_directions=self.count_directions,
                                        fusion_mode=self.fusion_mode)
        for spec in self.camera_specs:
            a = self.analyses[spec.camera_id]
            engine.add_camera(CameraAngleConfig(
                camera_id=spec.camera_id,
                angle_type=spec.angle_type,
                line_orientation=spec.line_orientation,
                line_pos=a.line_pos,
                direction=spec.direction,
                hysteresis=a.hysteresis,
                cooldown_frames=spec.cooldown_frames,
                role=spec.role,
            ))

        readers = {s.camera_id: _CameraReader(s.video_path, self.analyses[s.camera_id].fps)
                   for s in self.camera_specs}

        g_start, g_end = self._global_timeline()
        n_frames = int((g_end - g_start) * self.target_fps)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        temp_raw = (output_video_path[:-4] + "_raw.mp4"
                    if output_video_path.endswith(".mp4")
                    else output_video_path + "_raw.mp4")
        out_writer = None
        if render_video:
            out_writer = cv2.VideoWriter(temp_raw, fourcc, self.target_fps, (canvas_w, canvas_h))

        print(f"\n[Phase C] Rendering {n_frames} global frames "
              f"({canvas_w}x{canvas_h}) from {len(self.camera_specs)} cameras"
              f"{'' if render_video else '  [fusion only, no video]'}")

        active_toast = None
        toast_expiry = -1.0
        start_time = time.time()
        g = 0

        while g < n_frames:
            g_time = g_start + g / self.target_fps
            g += 1

            processed_panes = []
            for spec, (pw, ph, pscale) in zip(self.camera_specs, pane_dims):
                analysis = self.analyses[spec.camera_id]
                off = self.offsets[spec.camera_id]
                local_t = g_time - off
                local_idx = int(round(local_t * analysis.fps))

                state = "LIVE"
                frame = None
                if local_idx < 0:
                    state = "WAITING"
                elif local_idx >= analysis.frame_count:
                    state = "ENDED"
                elif render_video:
                    frame = readers[spec.camera_id].seek_to(local_idx)
                    if frame is None:
                        state = "ENDED"

                # Detections for this local frame (ORIGINAL pixel coords).
                objs = self._objects_at(analysis, local_idx)
                scaled_objs = []
                for tid, bbox, cls_name, conf, qty in objs:
                    x1, y1, x2, y2 = bbox
                    scaled_objs.append((tid, (x1 * pscale, y1 * pscale, x2 * pscale, y2 * pscale),
                                        conf, qty))

                if state == "LIVE":
                    # The tracker works in ORIGINAL pixel space so that the
                    # tripwire it uses is exactly the one drawn below.
                    engine.update_camera_frame(spec.camera_id, objs, g, g_time)

                if not render_video:
                    continue

                if frame is None:
                    pane = np.zeros((ph, pw, 3), dtype=np.uint8)
                    pane[:] = (25, 28, 38)
                    cv2.putText(pane, state, (20, ph // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (120, 130, 150), 2)
                else:
                    pane = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_LINEAR)

                # Detections for this local frame (ORIGINAL pixel coords).
                objs = self._objects_at(analysis, local_idx)
                scaled_objs = []
                for tid, bbox, cls_name, conf, qty in objs:
                    x1, y1, x2, y2 = bbox
                    scaled_objs.append((tid, (x1 * pscale, y1 * pscale, x2 * pscale, y2 * pscale),
                                        cls_name, conf, qty))

                if state == "LIVE":
                    # The tracker works in ORIGINAL pixel space so that the
                    # tripwire it uses is exactly the one drawn below.
                    engine.update_camera_frame(spec.camera_id, objs, g, g_time)

                # Draw tripwire at its true (scaled) position.
                line_px = int(analysis.line_pos * pscale)
                if spec.role == "width_sampler":
                    cv2.putText(pane, "WIDTH SAMPLER", (12, ph - 14), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (245, 158, 11), 1)
                elif spec.line_orientation == "horizontal":
                    cv2.line(pane, (0, line_px), (pw, line_px), (0, 0, 255), 2)
                    cv2.putText(pane, f"LINE Y={line_px}", (12, max(24, line_px - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    cv2.line(pane, (line_px, 0), (line_px, ph), (0, 0, 255), 2)
                    cv2.putText(pane, f"LINE X={line_px}", (min(pw - 110, line_px + 8), 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                color = {"width": (16, 185, 129), "height": (6, 182, 212),
                         "width_sampler": (245, 158, 11),
                         "corroborator": (245, 158, 11)}.get(spec.role, (200, 200, 200))
                for tid, (bx1, by1, bx2, by2), cls_name, cf, qty in scaled_objs:
                    box_color = (60, 220, 100) if cls_name == "carton" else color
                    cv2.rectangle(pane, (int(bx1), int(by1)), (int(bx2), int(by2)), box_color, 2)
                    if cls_name == "carton":
                        tag = f"CARTON #{tid}"
                    else:
                        tag = f"WORKER {tid}"
                        if qty > 0:
                            tag += f"  [{qty} CARTON{'S' if qty > 1 else ''}]"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                    ty = max(20, int(by1) - 6)
                    tx = int(bx1)
                    cv2.rectangle(pane, (tx, ty - th - 5), (tx + tw + 6, ty + 3), (10, 14, 24), -1)
                    cv2.putText(pane, tag, (tx + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, box_color, 2)

                badge = f"{spec.camera_id.upper()} ({spec.role})"
                cv2.rectangle(pane, (0, 0), (pw, 34), (20, 28, 45), -1)
                cv2.putText(pane, badge, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (248, 250, 252), 2)
                cv2.putText(pane, f"{g_time:.1f}s {state}", (12, ph - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 170, 185), 1)

                processed_panes.append(pane)

            fused_events = engine.process_time_window_consensus(g_time)
            if fused_events:
                active_toast = fused_events[-1]
                toast_expiry = g_time + 2.0

            if not render_video:
                continue

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            curr_x = 0
            for pane, (pw, ph, _) in zip(processed_panes, pane_dims):
                canvas[title_height:title_height + ph, curr_x:curr_x + pw] = pane
                cv2.line(canvas, (curr_x + pw - 1, title_height),
                         (curr_x + pw - 1, title_height + ph), (70, 80, 100), 2)
                curr_x += pw

            if title_height:
                band = canvas[0:title_height, 0:canvas_w]
                band[:] = (12, 17, 28)
                cv2.putText(band, title, (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                            (248, 250, 252), 2)
                cv2.line(canvas, (0, title_height - 1), (canvas_w, title_height - 1),
                         (56, 189, 248), 2)

            hud = canvas[title_height + self.pane_height:canvas_h, 0:canvas_w]
            hud[:] = (15, 23, 42)
            cv2.line(canvas, (0, title_height + self.pane_height),
                     (canvas_w, title_height + self.pane_height), (51, 65, 85), 2)

            summary = engine.get_summary()
            prod, cnt = self._metric_totals(summary["events"])
            if self.hud_metric == "product":
                headline = prod["in"]
                sub = "W x H reading (stacked load)"
            else:
                headline = cnt["in"]
                sub = "count reading (cartons side by side)"

            cv2.putText(hud, "MULTI-ANGLE FUSION ENGINE", (25, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (6, 182, 212), 2)
            cv2.putText(hud, f"TOTAL LOADED: {headline} CARTONS",
                        (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (16, 185, 129), 3)
            cv2.putText(hud, f"Trips: {summary['total_worker_trips']} | "
                             f"Events: {summary['fused_events_count']} | "
                             f"t={g_time:.1f}s", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (148, 163, 184), 2)
            cv2.putText(hud, sub, (canvas_w - 8 - 11 * len(sub), 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (148, 163, 184), 1)

            # Legend: which colour belongs to which angle.
            lx = 25
            for spec in self.camera_specs:
                col = {"width": (16, 185, 129), "height": (6, 182, 212),
                       "width_sampler": (245, 158, 11),
                       "corroborator": (245, 158, 11)}.get(spec.role, (200, 200, 200))
                cv2.rectangle(hud, (lx, 116), (lx + 16, 128), col, -1)
                label = f"{spec.camera_id} ({spec.role})"
                cv2.putText(hud, label, (lx + 22, 127), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (203, 213, 225), 1)
                lx += 22 + 9 * len(label) + 26
            cv2.putText(hud, "red line = counting tripwire", (lx, 127),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1)

            toast_x = max(int(canvas_w * 0.42), 420)
            if active_toast and g_time <= toast_expiry:
                cv2.rectangle(hud, (toast_x, 15), (canvas_w - 25, 125), (6, 78, 59), -1)
                cv2.rectangle(hud, (toast_x, 15), (canvas_w - 125, 125), (16, 185, 129), 2)
                if self.fusion_mode == "carton":
                    txt = (f"+{active_toast['fused_cartons']} CARTON LOADED "
                           f"({active_toast['consensus_status']})")
                    eq = (f"Cameras: {', '.join(active_toast.get('corroborating_cameras', []))}")
                else:
                    txt = (f"+{active_toast['fused_cartons']} LOADED "
                           f"({active_toast['consensus_status']})")
                    eq = (f"W={active_toast['top_camera_width']} x H={active_toast['side_camera_height']}")
                cv2.putText(hud, txt, (toast_x + 18, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (209, 250, 229), 2)
                cv2.putText(hud, eq, (toast_x + 18, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (248, 250, 252), 1)
            else:
                cv2.rectangle(hud, (toast_x, 25), (canvas_w - 25, 115), (30, 41, 59), -1)
                cv2.putText(hud, "Monitoring synchronised feeds...", (toast_x + 22, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1)

            out_writer.write(canvas)

        for r in readers.values():
            r.release()
        if out_writer is not None:
            out_writer.release()
        elapsed = time.time() - start_time

        if render_video and os.path.exists(temp_raw):
            cmd = ["ffmpeg", "-y", "-i", temp_raw, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-preset", "fast", "-crf", "22", output_video_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(output_video_path):
                os.remove(temp_raw)
                print(f"[Phase C] Saved H.264 video: {output_video_path}")

        final_summary = engine.get_summary()
        diag = self.sync_report.get("diagnostics", {})

        # Carton-direct reading: each fused event = 1 carton (deduplicated across cameras)
        loaded_cartons_direct = sum(1 for e in final_summary["events"]
                                    if e.get("direction") == "IN")
        returned_cartons_direct = sum(1 for e in final_summary["events"]
                                      if e.get("direction") == "OUT")
        final_summary["loaded_cartons_direct"] = loaded_cartons_direct
        final_summary["returned_cartons_direct"] = returned_cartons_direct
        final_summary["net_cartons_direct"] = loaded_cartons_direct - returned_cartons_direct

        # Two readings of the same fusion, because the geometry of the load
        # decides which one is right:
        #   product (W x H)  - the carton block is W wide and H high (a stack).
        #                      This is the truck-loading case.
        #   count  (max)     - the cartons are side by side in one row, so the
        #                      side camera's ROI sees N blobs rather than a
        #                      height of N. Use this when the loads are flat.
        prod, cnt = self._metric_totals(final_summary["events"])
        final_summary["loaded_cartons_product"] = prod["in"]
        final_summary["returned_cartons_product"] = prod["out"]
        final_summary["net_cartons_product"] = prod["in"] - prod["out"]
        final_summary["loaded_cartons_count"] = cnt["in"]
        final_summary["returned_cartons_count"] = cnt["out"]
        final_summary["net_cartons_count"] = cnt["in"] - cnt["out"]
        if self.fusion_mode == "carton":
            final_summary["interpretation_note"] = (
                "direct = one carton per fused event (cross-camera deduplicated). "
                "product = W x H (stacked load, legacy). "
                "count = max(W, H) (cartons side by side in one row, legacy). "
                "For carton-centric models (best.pt), use 'direct' as the authoritative count."
            )
        else:
            final_summary["interpretation_note"] = (
                "product = W x H (stacked load, the truck-loading case). "
                "count = max(W, H) (cartons side by side in one row). "
                "Which one applies depends on how the cartons are stacked in the "
                "scene; check the fused events before quoting a number."
            )

        low_conf = {c: d for c, d in diag.items() if d.get("confidence", 0.0) < 0.70}

        warnings: List[str] = []
        if low_conf:
            warnings.append(
                "Sync confidence is low for: " +
                ", ".join(f"{c} (conf={d['confidence']:.2f}, matched={d['matched_events']})"
                          for c, d in low_conf.items()) +
                ". The fused count was verified stable across the plausible offset "
                "space (see output_videos/offset_sensitivity.json), but the exact "
                "per-event pairing is not reliable. Use --offset cam_id=seconds to "
                "pin the offsets, and capture a shared visual timecode in production."
            )

        report = {
            "sync": {
                "reference_camera": self.sync_report.get("reference"),
                "offsets_sec": self.offsets,
                "diagnostics": diag,
                "low_confidence_cameras": list(low_conf.keys()),
                "warnings": warnings,
            },
            "cameras": [
                {
                    "camera_id": s.camera_id,
                    "video": os.path.basename(s.video_path),
                    "role": s.role,
                    "line_orientation": s.line_orientation,
                    "line_pos_px": self.analyses[s.camera_id].line_pos,
                    "hysteresis_px": self.analyses[s.camera_id].hysteresis,
                    "direction": s.direction,
                    "resolution": f"{self.analyses[s.camera_id].width}x{self.analyses[s.camera_id].height}",
                    "duration_sec": round(self.analyses[s.camera_id].duration(), 2),
                    "local_events": len(self.analyses[s.camera_id].events),
                }
                for s in self.camera_specs
            ],
            "result": final_summary,
            "render_seconds": round(elapsed, 1),
        }
        if output_json_path:
            with open(output_json_path, "w") as fh:
                json.dump(report, fh, indent=2)
            print(f"[Phase C] Saved audit JSON: {output_json_path}")
        for w in warnings:
            print(f"[Phase C] WARNING: {w}")
        return report

    @staticmethod
    def _objects_at(analysis: CameraAnalysis, local_idx: int):
        """Detections for a frame index, carrying the last known set forward."""
        if not analysis.frames:
            return []
        if 0 <= local_idx < len(analysis.frames):
            objs = analysis.frames[local_idx]
            if objs:
                return objs
            if local_idx > 0:
                return analysis.frames[local_idx - 1]
            return []
        return []


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_spec(video_dir: str) -> List[Dict[str, Any]]:
    """
    Spec for the 3-angle synchronized multi-camera take.

    IMPORTANT - `top_view.mp4` is a DIFFERENT TAKE of the same overhead camera
    position as `overhead_view.mp4` (verified: the green-shirt worker is
    present in overhead_view 53% of frames and in top_view 0% of frames).
    The real 3-angle set is therefore:
        overhead = overhead_view.mp4   (green worker + light blue worker)
        side     = side_view.mp4       (green worker + light blue + tan)
        flank    = flank_view.mp4       (light blue worker)

    Tripwire positions were chosen empirically with scripts/tripwire_sweep.py
    against the cached YOLO trajectories.
    """
    return [
        {
            "camera_id": "cam_top",
            "video_path": os.path.join(video_dir, "overhead_view.mp4"),
            "role": "width_sampler",
            "angle_type": "top_overhead",
            "line_orientation": "horizontal",
            "line_pos_frac": 0.45,
            "direction": "up_to_down",
            "cooldown_frames": 45,
        },
        {
            "camera_id": "cam_side",
            "video_path": os.path.join(video_dir, "side_view.mp4"),
            "role": "height",
            "angle_type": "side_profile",
            "line_orientation": "vertical",
            "line_pos_frac": 0.55,
            "direction": "left_to_right",
            "cooldown_frames": 45,
        },
        {
            "camera_id": "cam_flank",
            "video_path": os.path.join(video_dir, "flank_view.mp4"),
            "role": "corroborator",
            "angle_type": "side_profile",
            "line_orientation": "vertical",
            "line_pos_frac": 0.70,
            "direction": "right_to_left",
            "cooldown_frames": 45,
        },
    ]


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Multi-Angle Fusion Video Processor")
    parser.add_argument("--spec", type=str, default=None,
                        help="JSON file describing the camera set")
    parser.add_argument("--video-dir", type=str, default=os.path.join(here, "recorded_video"))
    parser.add_argument("--output-video", type=str, default=os.path.join(here, "output_videos", "multi_angle_3view.mp4"))
    parser.add_argument("--output-json", type=str, default=os.path.join(here, "output_videos", "multi_angle_3view_report.json"))
    parser.add_argument("--reference", type=str, default=None, help="Reference camera id for alignment")
    parser.add_argument("--offset", type=str, action="append", default=[],
                        help="Manual offset override: cam_id=seconds (repeatable)")
    parser.add_argument("--no-auto-offset", action="store_true",
                        help="Disable automatic offset estimation")
    parser.add_argument("--stride", type=int, default=2, help="YOLO inference stride")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference resolution (640 for door-lage CCTV)")
    parser.add_argument("--conf", type=float, default=0.22)
    parser.add_argument("--pane-height", type=int, default=720)
    parser.add_argument("--time-window", type=float, default=0.85)
    parser.add_argument("--title", type=str, default="",
                        help="Title banner drawn above the panes")
    parser.add_argument("--hud-metric", choices=["count", "product"], default="count",
                        help="Which reading the HUD shows: count = max(W,H) for flat "
                             "loads, product = W x H for stacked loads")
    parser.add_argument("--count-directions", type=str, default=None,
                        help="Comma separated tripwire directions that count, e.g. IN")
    parser.add_argument("--fusion-mode", choices=["carton", "wh"], default=None,
                        help="Fusion mode: carton = direct per-carton dedup, wh = legacy W x H")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.spec:
        with open(args.spec) as fh:
            raw = json.load(fh)
        raw_specs = raw["cameras"] if isinstance(raw, dict) and "cameras" in raw else raw
    else:
        raw_specs = _default_spec(args.video_dir)

    specs = [CameraSpec.from_dict(d) for d in raw_specs]
    repo_root = os.path.dirname(os.path.dirname(here))
    for s in specs:
        if not os.path.isabs(s.video_path):
            # Resolve relative to the spec file, the repo root, then the CWD.
            for base in (os.path.dirname(os.path.abspath(args.spec)) if args.spec else "",
                         repo_root, here, os.getcwd()):
                cand = os.path.join(base, s.video_path) if base else s.video_path
                if os.path.exists(cand):
                    s.video_path = cand
                    break
        if not os.path.exists(s.video_path):
            raise SystemExit(f"Video not found: {s.video_path}")

    for ov in args.offset:
        if "=" not in ov:
            raise SystemExit(f"Invalid --offset '{ov}', expected cam_id=seconds")
        cam_id, val = ov.split("=", 1)
        for s in specs:
            if s.camera_id == cam_id:
                s.start_offset_sec = float(val)
                s.offset_locked = True
                print(f"[CLI] offset override: {cam_id} = {float(val):+.2f}s (locked)")

    pipeline = MultiAnglePipeline(
        camera_specs=specs,
        conf=args.conf,
        frame_stride=args.stride,
        imgsz=args.imgsz,
        pane_height=args.pane_height,
        time_window_sec=args.time_window,
        use_cache=not args.no_cache,
        hud_metric=args.hud_metric,
        count_directions=([d.strip().upper() for d in args.count_directions.split(",")]
                          if args.count_directions else None),
        fusion_mode=args.fusion_mode,
    )

    pipeline.analyse()
    pipeline.align(reference=args.reference, auto=not args.no_auto_offset)
    report = pipeline.render(output_video_path=args.output_video,
                             output_json_path=args.output_json,
                             title=args.title)

    result = report["result"]
    print("\n--- Summary Results ---")
    print(f"Fusion Mode          : {pipeline.fusion_mode}")
    if pipeline.fusion_mode == "carton":
        print(f"Total Loaded Cartons : {result.get('loaded_cartons_direct', result['total_loaded_cartons'])} (direct)")
        print(f"Total Returned       : {result.get('returned_cartons_direct', result['total_returned_cartons'])} (direct)")
        print(f"Net Cartons          : {result.get('net_cartons_direct', result['net_cartons'])} (direct)")
    else:
        print(f"Total Loaded Cartons : {result['total_loaded_cartons']}")
        print(f"Total Returned       : {result['total_returned_cartons']}")
        print(f"Net Cartons          : {result['net_cartons']}")
    print(f"Worker Trips         : {result['total_worker_trips']}")
    print(f"Fused Events         : {result['fused_events_count']}")
    print(f"Consensus Breakdown  : {result.get('consensus_breakdown')}")


if __name__ == "__main__":
    main()
