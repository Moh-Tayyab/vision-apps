#!/usr/bin/env python3
"""
Offline Video-to-Video Processing Pipeline for Truck Loading & Carton Detection.

Takes an input video file, runs full YOLO tracking, classical CV carton holding detection,
Worker-Carton bipartite association, and tripwire counting, and writes out a fully annotated
MP4 video with visual HUD, bounding boxes, trajectories, and crossing event alerts.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from typing import Dict, Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 6)))

from counter_engine import TripwireCounter
from visualizer import LoadingVisualizer

VIDEO_PRESETS = {
    "sample_truck_loading.mp4": {"line_x_ratio": 0.50, "direction": "left_to_right"},
    "warehouse_1.mp4": {"line_x_ratio": 0.50, "direction": "right_to_left"},
    "warehouse_2.mp4": {"line_x_ratio": 0.65, "direction": "right_to_left"},
    "warehouse_3.mp4": {"line_x_ratio": 0.50, "direction": "right_to_left"},
    "warehouse_4.mp4": {"line_x_ratio": 0.40, "direction": "left_to_right"},
    "warehouse_5.mp4": {"line_x_ratio": 0.45, "direction": "right_to_left"},
    "warehouse_6.mp4": {"line_x_ratio": 0.45, "direction": "right_to_left"},
    "warehouse_7.mp4": {"line_x_ratio": 0.22, "direction": "right_to_left"},
    "warehouse_8.mp4": {"line_x_ratio": 0.65, "direction": "left_to_right"},
}

TARGET_CLASSES = {"person", "cardboard box", "carton", "box", "package", "suitcase", "backpack"}


def convert_to_h264(input_video: str, output_video: str) -> bool:
    """Converts a video to browser-friendly H.264 mp4 using ffmpeg if available."""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "22",
            "-movflags", "+faststart",
            output_video
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception as e:
        print(f"ffmpeg conversion warning: {e}")
        return False


def process_video_file(
    input_path: str,
    output_path: Optional[str] = None,
    line_x_ratio: Optional[float] = None,
    direction: Optional[str] = None,
    conf: float = 0.18,
    iou: float = 0.45,
    imgsz: int = 384,
    frame_stride: int = 1,
    max_dimension: int = 720,
    model_path: Optional[str] = None,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Processes an input video file and writes an annotated output video.
    """
    if not os.path.exists(input_path):
        return {"error": f"File not found: {input_path}"}

    video_name = os.path.basename(input_path)
    preset = VIDEO_PRESETS.get(video_name, {"line_x_ratio": 0.50, "direction": "left_to_right"})

    eff_ratio = line_x_ratio if line_x_ratio is not None else preset["line_x_ratio"]
    eff_direction = direction if direction is not None else preset["direction"]
    eff_stride = frame_stride if frame_stride is not None and frame_stride >= 1 else 1

    if output_path is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_videos")
        os.makedirs(out_dir, exist_ok=True)
        base_name, _ = os.path.splitext(video_name)
        output_path = os.path.join(out_dir, f"output_{base_name}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {input_path}"}

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_dim = max(raw_w, raw_h)
    if max_dim > max_dimension:
        scale = float(max_dimension) / max_dim
        out_w = int(raw_w * scale)
        out_h = int(raw_h * scale)
    else:
        scale = 1.0
        out_w = raw_w
        out_h = raw_h

    # Ensure even dimensions for video encoders
    if out_w % 2 != 0:
        out_w -= 1
    if out_h % 2 != 0:
        out_h -= 1

    line_x = int(out_w * eff_ratio)

    # Initialize model (prefer YOLOv8s-worldv2 for true carton class)
    if model_path is None:
        app_dir = os.path.dirname(os.path.abspath(__file__))
        cand_world = os.path.abspath(os.path.join(app_dir, "..", "..", "yolov8s-worldv2.pt"))
        if os.path.exists(cand_world):
            model_path = cand_world
        else:
            cand_11 = os.path.join(app_dir, "yolo11n.pt")
            cand_8 = os.path.join(app_dir, "yolov8n.pt")
            model_path = cand_11 if os.path.exists(cand_11) else (cand_8 if os.path.exists(cand_8) else "yolov8s-worldv2.pt")

    model = YOLO(model_path)
    if hasattr(model, "set_classes"):
        try:
            model.set_classes(["person", "cardboard box", "carton"])
        except Exception:
            pass
    model_names = model.names
    target_cids = [cid for cid, name in model_names.items() if any(d in name.lower() for d in TARGET_CLASSES)]

    # Initialize counter & visualizer
    counter = TripwireCounter(
        line_x=line_x,
        loading_direction=eff_direction,
        cooldown_frames=15,
        min_displacement_px=max(12, int(out_w * 0.012)),
        hysteresis=max(12, int(out_w * 0.015)),
    )
    visualizer = LoadingVisualizer()

    # Video writer setup: write raw mp4v first
    temp_raw_output = output_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    effective_fps = max(1.0, fps / eff_stride)
    writer = cv2.VideoWriter(temp_raw_output, fourcc, effective_fps, (out_w, out_h))

    frame_idx = 0
    saved_frames = 0
    all_events = []
    start_time = time.time()
    last_objs = []

    print(f"[PROCESS] Starting: {video_name} ({total_frames} frames, {fps:.1f} fps, stride={eff_stride}) -> {output_path}", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if eff_stride > 1 and (frame_idx % eff_stride != 0):
            continue

        if scale != 1.0 or frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        results = model.track(
            frame,
            persist=True,
            classes=target_cids,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        tracked_objects = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()

            for box, tid, c_idx, c_val in zip(boxes, track_ids, classes, confs):
                c_name = model_names.get(int(c_idx), f"class_{int(c_idx)}")
                tracked_objects.append((int(tid), tuple(box), c_name, float(c_val)))

        events = counter.update(tracked_objects, frame_idx, frame=frame)
        if events:
            for ev in events:
                all_events.append({
                    "frame": ev["frame"],
                    "sec": round(ev["frame"] / fps, 2),
                    "track_id": ev["track_id"],
                    "class": ev["class_name"],
                    "delta": ev["delta"],
                    "direction": ev.get("direction", "IN" if ev["delta"] > 0 else "OUT"),
                    "total_in": ev["total_in"],
                    "total_out": ev.get("total_out", counter.total_out),
                    "note": ev.get("note", ""),
                })

        # Render full visual overlays
        visualizer.draw_trajectories(frame, counter.track_history)
        visualizer.draw_tripwire(
            frame=frame,
            line_x=line_x,
            loading_direction=eff_direction,
        )
        visualizer.draw_detections(
            frame=frame,
            tracked_objects=tracked_objects,
            track_side=counter.track_side,
            line_x=line_x,
            associations=counter.current_associations,
            carton_boxes=counter.track_carton_boxes,
            holding_scores=counter.track_holding_scores,
            worker_qty=counter.current_worker_qty,
        )
        visualizer.draw_hud(
            frame=frame,
            total_in=counter.total_in,
            total_out=counter.total_out,
            net_count=counter.net_count,
            fps=fps,
            active_count=len(tracked_objects),
            recent_event=counter.recent_event,
            recent_event_expiry=counter.recent_event_expiry,
            worker_trips=counter.worker_trips_in,
        )

        writer.write(frame)
        saved_frames += 1

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            pct = (frame_idx / max(1, total_frames)) * 100
            print(f"  [{video_name}] Frame {frame_idx}/{total_frames} ({pct:.1f}%) | Net: {counter.net_count} | Trips: {counter.worker_trips_in}", flush=True)
            if progress_callback:
                progress_callback(video_name, frame_idx, total_frames)

    cap.release()
    writer.release()

    elapsed = round(time.time() - start_time, 2)

    # Convert temp raw video to H.264 mp4 via ffmpeg for web playback
    h264_ok = convert_to_h264(temp_raw_output, output_path)
    if h264_ok and os.path.exists(output_path):
        os.remove(temp_raw_output)
    else:
        # Fallback to the raw mp4
        if os.path.exists(temp_raw_output):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(temp_raw_output, output_path)

    out_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2) if os.path.exists(output_path) else 0.0

    result = {
        "video": video_name,
        "input_path": input_path,
        "output_path": output_path,
        "output_size_mb": out_size_mb,
        "total_frames": total_frames,
        "processed_frames": frame_idx,
        "saved_frames": saved_frames,
        "duration_sec": round(total_frames / fps, 2),
        "fps": round(fps, 1),
        "line_x": line_x,
        "line_x_ratio": eff_ratio,
        "direction": eff_direction,
        "total_in": counter.total_in,
        "total_out": counter.total_out,
        "net_count": counter.net_count,
        "worker_trips_in": counter.worker_trips_in,
        "worker_trips_out": counter.worker_trips_out,
        "events": all_events,
        "processing_time_sec": elapsed,
    }

    print(f"[DONE] {video_name}: Saved to {output_path} ({out_size_mb} MB) in {elapsed}s | Net Count: {counter.net_count} | Worker Trips: {counter.worker_trips_in}")
    return result


def batch_process_all(input_dir: str = ".", output_dir: str = "output_videos") -> Dict[str, Any]:
    """Batch process all warehouse_*.mp4 videos."""
    os.makedirs(output_dir, exist_ok=True)
    all_videos = sorted([f for f in os.listdir(input_dir) if f.startswith("warehouse_") and f.endswith(".mp4")])
    if not all_videos:
        print(f"No warehouse_*.mp4 videos found in {input_dir}")
        return {}

    summary = {}
    total_start = time.time()

    for v_name in all_videos:
        v_path = os.path.join(input_dir, v_name)
        out_path = os.path.join(output_dir, f"output_{v_name}")
        stride = 1
        res = process_video_file(
            input_path=v_path,
            output_path=out_path,
            frame_stride=stride,
        )
        summary[v_name] = res

    total_elapsed = round(time.time() - total_start, 2)
    report_file = os.path.join(output_dir, "summary_report.json")
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========================================================")
    print(f"BATCH PROCESSING COMPLETE ({len(all_videos)} videos in {total_elapsed}s)")
    print(f"Report saved to: {report_file}")
    print("========================================================")
    for v_name, res in summary.items():
        if "error" in res:
            print(f"  {v_name:20s}: ERROR - {res['error']}")
        else:
            print(f"  {v_name:20s}: Loaded {res['total_in']:2d} | Returned {res['total_out']:2d} | Net {res['net_count']:2d} | Trips {res['worker_trips_in']:2d} | Time {res['processing_time_sec']:5.1f}s | Output: {os.path.basename(res['output_path'])}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video-to-Video Truck Loading Inference Pipeline")
    parser.add_argument("--input", type=str, default=None, help="Input video path")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--all", action="store_true", help="Batch process all warehouse_*.mp4 videos")
    parser.add_argument("--line-ratio", type=float, default=None, help="Virtual line X ratio (0.0 - 1.0)")
    parser.add_argument("--direction", type=str, default=None, help="Loading direction ('left_to_right' or 'right_to_left')")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    args = parser.parse_args()

    if args.all or (args.input is None):
        batch_process_all()
    else:
        process_video_file(
            input_path=args.input,
            output_path=args.output,
            line_x_ratio=args.line_ratio,
            direction=args.direction,
            frame_stride=args.stride,
        )
