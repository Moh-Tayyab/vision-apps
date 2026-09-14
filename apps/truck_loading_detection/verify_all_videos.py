import os
import sys
import cv2
import json
import time
import numpy as np
import torch
from ultralytics import YOLO
from counter_engine import TripwireCounter

torch.set_num_threads(6)

VIDEO_PRESETS = {
    "warehouse_1.mp4": {"line_x_ratio": 0.50, "direction": "right_to_left", "stride": 2},
    "warehouse_2.mp4": {"line_x_ratio": 0.50, "direction": "right_to_left", "stride": 2},
    "warehouse_3.mp4": {"line_x_ratio": 0.50, "direction": "right_to_left", "stride": 2},
    "warehouse_4.mp4": {"line_x_ratio": 0.40, "direction": "left_to_right", "stride": 2},
    "warehouse_5.mp4": {"line_x_ratio": 0.45, "direction": "right_to_left", "stride": 2},
    "warehouse_6.mp4": {"line_x_ratio": 0.45, "direction": "right_to_left", "stride": 1},
    "warehouse_7.mp4": {"line_x_ratio": 0.22, "direction": "right_to_left", "stride": 2},
    "warehouse_8.mp4": {"line_x_ratio": 0.65, "direction": "left_to_right", "stride": 2},
}

TARGET_CLASSES = {"person", "cardboard box", "carton", "box", "package", "suitcase", "backpack"}

def process_video(video_path):
    video_name = os.path.basename(video_path)
    if not os.path.exists(video_path):
        return {"error": f"File not found: {video_path}"}

    preset = VIDEO_PRESETS.get(video_name, {"line_x_ratio": 0.50, "direction": "left_to_right", "stride": 2})
    stride = preset.get("stride", 2)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1024
    raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 576
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    max_dim = max(raw_w, raw_h)
    if max_dim > 720:
        scale = 720.0 / max_dim
        video_w = int(raw_w * scale)
        video_h = int(raw_h * scale)
    else:
        scale = 1.0
        video_w = raw_w
        video_h = raw_h

    line_x = int(video_w * preset["line_x_ratio"])
    direction = preset["direction"]

    counter = TripwireCounter(
        line_x=line_x,
        loading_direction=direction,
        cooldown_frames=15,
        min_displacement_px=max(12, int(video_w * 0.012)),
        hysteresis=max(12, int(video_w * 0.015)),
    )

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "yolov8s-worldv2.pt"))
    if not os.path.exists(model_path):
        model_path = "yolov8s-worldv2.pt"
    model = YOLO(model_path)
    if hasattr(model, "set_classes"):
        try:
            model.set_classes(["person", "cardboard box", "carton"])
        except Exception:
            pass
    model_names = model.names
    target_cids = [cid for cid, name in model_names.items() if any(d in name.lower() for d in TARGET_CLASSES)]

    frame_idx = 0
    processed = 0
    all_events = []
    last_tracked = []
    
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if scale != 1.0:
            frame = cv2.resize(frame, (video_w, video_h), interpolation=cv2.INTER_LINEAR)

        run_detection = (frame_idx % stride == 0) or (len(last_tracked) == 0)
        tracked_objects = []
        if run_detection:
            processed += 1
            results = model.track(
                frame,
                persist=True,
                classes=target_cids,
                conf=0.18,
                iou=0.45,
                imgsz=384,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()

                for box, tid, c_idx, c_val in zip(boxes, track_ids, classes, confs):
                    c_name = model_names.get(int(c_idx), f"class_{int(c_idx)}")
                    tracked_objects.append((int(tid), tuple(box), c_name, float(c_val)))
                last_tracked = tracked_objects
            else:
                tracked_objects = last_tracked
        else:
            tracked_objects = last_tracked

        events = counter.update(tracked_objects, frame_idx, frame=frame)
        if events:
            for ev in events:
                all_events.append({
                    "frame": ev["frame"],
                    "sec": round(ev["frame"] / fps, 2),
                    "track_id": ev["track_id"],
                    "class": ev["class_name"],
                    "delta": ev["delta"],
                    "total_in": ev["total_in"],
                    "note": ev["note"],
                })

        if processed % 50 == 0:
            print(f"  Processed {processed} frames ({frame_idx}/{total_frames})...", flush=True)

    cap.release()
    elapsed = round(time.time() - start_time, 2)

    return {
        "video": video_name,
        "total_frames": total_frames,
        "processed_frames": processed,
        "stride": stride,
        "duration_sec": round(total_frames / fps, 2),
        "fps": round(fps, 1),
        "line_x": line_x,
        "direction": direction,
        "total_in": counter.total_in,
        "total_out": counter.total_out,
        "net_count": counter.net_count,
        "worker_trips_in": counter.worker_trips_in,
        "events": all_events,
        "processing_time_sec": elapsed,
    }

if __name__ == "__main__":
    import sys
    videos = [f"warehouse_{i}.mp4" for i in range(1, 9)]
    if len(sys.argv) > 1:
        videos = [sys.argv[1]]

    for v in videos:
        print(f"\n==========================================")
        print(f"PROCESSING: {v}")
        print(f"==========================================")
        res = process_video(v)
        if "error" in res:
            print(f"ERROR: {res['error']}")
            continue
        print(f"Duration: {res['duration_sec']}s | Frames: {res['processed_frames']} | Direction: {res['direction']} | Line X: {res['line_x']}")
        print(f"TOTAL IN: {res['total_in']} | TOTAL OUT: {res['total_out']} | NET COUNT: {res['net_count']} | Worker Trips: {res['worker_trips_in']}")
        print("Events Timeline:")
        for ev in res["events"]:
            print(f"  Frame {ev['frame']} ({ev['sec']}s) [ID #{ev['track_id']} {ev['class']}]: delta {ev['delta']:+d} -> Total {ev['total_in']} | {ev['note']}")
        print(json.dumps({k: v for k, v in res.items() if k != "events"}, indent=2))
