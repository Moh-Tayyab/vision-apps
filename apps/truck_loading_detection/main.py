#!/usr/bin/env python3
"""
Truck Loading Object Detection and Directional Tripwire Counter
Real-time tracking of cartons/workers crossing a virtual vertical tripwire using YOLO, ByteTrack, and OpenCV.
"""

import argparse
import sys
import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

from counter_engine import TripwireCounter
from visualizer import LoadingVisualizer

# Global state for interactive mouse dragging of virtual line
is_mouse_dragging = False
mouse_line_x = None


def mouse_callback(event, x, y, flags, param):
    global is_mouse_dragging, mouse_line_x
    line_state = param  # dict containing 'line_x' and 'counter'

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if clicked near the line (within 25px) or anywhere on top/bottom
        if abs(x - line_state['line_x']) < 35:
            is_mouse_dragging = True
            mouse_line_x = x
            line_state['line_x'] = x
            line_state['counter'].set_line_x(x)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_mouse_dragging:
            mouse_line_x = x
            line_state['line_x'] = x
            line_state['counter'].set_line_x(x)
    elif event == cv2.EVENT_LBUTTONUP:
        if is_mouse_dragging:
            is_mouse_dragging = False
            line_state['line_x'] = x
            line_state['counter'].set_line_x(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Truck Loading Carton & Worker Tripwire Counter with YOLO & ByteTrack"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="sample_truck_loading.mp4",
        help="Path to video file, RTSP stream URL, or camera index (e.g. 0).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or model name (e.g. yolov8n.pt, yolov8s.pt, custom_carton.pt).",
    )
    parser.add_argument(
        "--line-x",
        type=int,
        default=None,
        help="X coordinate of the vertical virtual line (default: frame center width // 2).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.30,
        help="YOLO detection confidence threshold (default: 0.30).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
        help="NMS IoU threshold (default: 0.50).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Class names or IDs to filter (default: ['person', 'carton', 'box', 'suitcase', 'backpack']).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker configuration file (default: bytetrack.yaml, or botsort.yaml).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=384,
        help="Inference image size (default: 384 for fast CPU real-time tracking).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Run detector every N frames (default: 2 for 30 FPS playback).",
    )
    parser.add_argument(
        "--hysteresis",
        type=int,
        default=15,
        help="Hysteresis pixel band around tripwire line to avoid boundary jitter (default: 15).",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=15,
        help="Minimum frame cooldown before a single track ID can trigger another crossing (default: 15).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the annotated output video.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for saved output video (default: output_truck_loading.mp4).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run in headless mode without displaying GUI window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine video source
    source = args.source
    if source.isdigit():
        source = int(source)

    if not os.path.exists(str(source)) and not isinstance(source, int) and not str(source).startswith("rtsp"):
        print(f"[ERROR] Source video '{source}' not found.")
        sys.exit(1)

    print(f"[INFO] Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video source: {source}")
        sys.exit(1)

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video Resolution: {frame_width}x{frame_height} @ {fps:.2f} FPS ({total_frames} total frames)")

    # Tripwire Line Coordinate
    line_x = args.line_x if args.line_x is not None else (frame_width // 2)
    print(f"[INFO] Virtual Tripwire initialized at X = {line_x}")

    # Load YOLO Model
    print(f"[INFO] Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    model_names = model.names

    # Determine target classes to track
    target_class_ids = None
    if args.classes:
        target_class_ids = []
        for c in args.classes:
            if c.isdigit():
                target_class_ids.append(int(c))
            else:
                for cid, cname in model_names.items():
                    if c.lower() in cname.lower():
                        target_class_ids.append(cid)
    else:
        # Default smart filter: person, carton, box, suitcase, backpack
        desired_names = {"person", "carton", "box", "package", "suitcase", "backpack"}
        matched = [cid for cid, name in model_names.items() if name.lower() in desired_names]
        if matched:
            target_class_ids = matched
            print(f"[INFO] Auto-selected target classes: {[model_names[cid] for cid in target_class_ids]}")
        else:
            print(f"[INFO] Tracking all detected classes from model: {list(model_names.values())}")

    # Initialize Counter Engine and Visualizer
    counter = TripwireCounter(
        line_x=line_x,
        hysteresis=args.hysteresis,
        cooldown_frames=args.cooldown,
    )
    visualizer = LoadingVisualizer()

    # Video Writer setup if saving is requested
    writer = None
    if args.save or args.output:
        out_path = args.output or "output_truck_loading.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
        print(f"[INFO] Saving processed video to: {out_path}")

    # Window and interactive mouse setup
    window_name = "Real-Time Truck Loading Detection & Tripwire Counter"
    line_state = {"line_x": line_x, "counter": counter}

    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(1280, frame_width), min(720, frame_height))
        cv2.setMouseCallback(window_name, mouse_callback, line_state)

    frame_idx = 0
    start_time = time.time()
    fps_calc = fps
    is_paused = False
    last_tracked_objects = []

    print("\n" + "=" * 60)
    print("  TRUCK LOADING TRIPWIRE COUNTER RUNNING")
    print("  - Left -> Right  : +1 LOADED (Towards Truck)")
    print("  - Right -> Left  : -1 RETURNED (Away from Truck)")
    print("  Controls:")
    print("    [Click & Drag Line]  : Reposition tripwire line live")
    print("    [Left / Right Arrow] : Move line left / right")
    print("    [SPACE]              : Pause / Play toggle")
    print("    [R]                  : Reset counters")
    print("    [S]                  : Save current frame screenshot")
    print("    [Q] / [ESC]          : Exit")
    print("=" * 60 + "\n")

    try:
        while cap.isOpened():
            if not is_paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n[INFO] End of video stream reached.")
                    break
                frame_idx += 1
                t_frame_start = time.time()

                run_detection = (frame_idx % args.frame_stride == 0) or (len(last_tracked_objects) == 0)

                # Run YOLO Detection + Tracking
                if run_detection:
                    results = model.track(
                        frame,
                        persist=True,
                        classes=target_class_ids,
                        conf=args.conf,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        tracker=args.tracker,
                        verbose=False,
                    )

                    tracked_objects = []
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        confs = results[0].boxes.conf.cpu().numpy()

                        for box, track_id, cls_idx, conf in zip(boxes, track_ids, classes, confs):
                            cls_name = model_names.get(int(cls_idx), f"class_{int(cls_idx)}")
                            tracked_objects.append((int(track_id), tuple(box), cls_name, float(conf)))
                    last_tracked_objects = tracked_objects
                else:
                    tracked_objects = last_tracked_objects

                # Update Tripwire Counter State
                if run_detection:
                    new_events = counter.update(tracked_objects, frame_idx)
                    for ev in new_events:
                        direction_str = "+1 LOADED (Left->Right)" if ev["direction"] == "IN" else "-1 RETURNED (Right->Left)"
                        print(f"Frame {frame_idx:04d} | Object #{ev['track_id']} ({ev['class_name']}) crossed line: {direction_str} | Net: {ev['net_count']}")

                # Draw Visuals on Frame
                # 1. Motion Trajectory Trails
                visualizer.draw_trajectories(frame, counter.track_history)

                # 2. Virtual Tripwire Line
                visualizer.draw_tripwire(frame, line_state["line_x"], is_dragging=is_mouse_dragging)

                # 3. Object Bounding Boxes & Tags
                visualizer.draw_detections(frame, tracked_objects, counter.track_side, line_state["line_x"])

                # 4. Dashboard HUD Overlay & Toast Alert
                visualizer.draw_hud(
                    frame=frame,
                    total_in=counter.total_in,
                    total_out=counter.total_out,
                    net_count=counter.net_count,
                    fps=fps_calc,
                    active_count=len(tracked_objects),
                    recent_event=counter.recent_event,
                    recent_event_expiry=counter.recent_event_expiry,
                )

                # 5. Instructions footer
                if not args.no_show:
                    visualizer.draw_instructions(frame)

                # Write frame to video file
                if writer:
                    writer.write(frame)

                # Calculate live FPS
                frame_duration = time.time() - t_frame_start
                fps_calc = 0.9 * fps_calc + 0.1 * (1.0 / max(frame_duration, 1e-4))

            # GUI display & interaction handling
            if not args.no_show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1 if not is_paused else 30) & 0xFF

                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("[INFO] Quitting application.")
                    break
                elif key == ord(' '):  # SPACE: pause toggle
                    is_paused = not is_paused
                    print(f"[INFO] Video {'Paused' if is_paused else 'Resumed'}.")
                elif key == ord('r') or key == ord('R'):  # R: reset count
                    counter.reset_counts()
                    print("[INFO] Counters reset to zero.")
                elif key == ord('s') or key == ord('S'):  # S: save screenshot
                    snap_name = f"snapshot_frame_{frame_idx:04d}.png"
                    cv2.imwrite(snap_name, frame)
                    print(f"[INFO] Saved screenshot: {snap_name}")
                elif key == 81 or key == ord('-') or key == ord('_'):  # Left Arrow or '-'
                    line_state["line_x"] = max(20, line_state["line_x"] - 10)
                    counter.set_line_x(line_state["line_x"])
                elif key == 83 or key == ord('+') or key == ord('='):  # Right Arrow or '+'
                    line_state["line_x"] = min(frame_width - 20, line_state["line_x"] + 10)
                    counter.set_line_x(line_state["line_x"])

    finally:
        total_time = time.time() - start_time
        cap.release()
        if writer:
            writer.release()
        if not args.no_show:
            cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("                 PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Frames Processed : {frame_idx}")
        print(f"Total Time Taken       : {total_time:.2f} seconds ({frame_idx / max(total_time, 1e-3):.1f} FPS avg)")
        print(f"Total Loaded (+1)      : {counter.total_in}")
        print(f"Total Returned (-1)    : {counter.total_out}")
        print(f"Net Loaded Count       : {counter.net_count}")
        print(f"Total Crossing Events  : {len(counter.events)}")
        if writer:
            print(f"Annotated Video Saved  : {args.output or 'output_truck_loading.mp4'}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
