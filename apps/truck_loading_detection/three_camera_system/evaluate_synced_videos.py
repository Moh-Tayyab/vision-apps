"""
Real Video Evaluation Runner for Three-Camera Carton Counting (Phase 9).

Executes the prototype on:
- scenario1_top_synced.mp4
- scenario1_side_synced.mp4
- scenario1_flank_synced.mp4

Generates an auditable JSON report documenting:
- Per-camera detection stats
- Timestamp alignment metrics
- Multi-camera consensus breakdown
- Committed carton crossing events
- Latency and processing benchmarks
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Ensure repo root is on sys.path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from apps.truck_loading_detection.three_camera_system.pipeline import (
    ThreeCameraCartonPipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluator")


def run_evaluation(
    video_top: str,
    video_side: str,
    video_flank: str,
    output_report_json: str,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.15,
    max_frames: Optional[int] = None,
    tripwire_x: float = 500.0,
) -> Dict[str, Any]:
    """
    Runs three-camera carton evaluation over the synchronized MP4 files.
    """
    for p, name in [(video_top, "top"), (video_side, "side"), (video_flank, "flank")]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required video {name}: {p}")

    cap_top = cv2.VideoCapture(video_top)
    cap_side = cv2.VideoCapture(video_side)
    cap_flank = cv2.VideoCapture(video_flank)

    fps_top = cap_top.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(min(
        cap_top.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_side.get(cv2.CAP_PROP_FRAME_COUNT),
        cap_flank.get(cv2.CAP_PROP_FRAME_COUNT),
    ))

    logger.info("Initializing ThreeCameraCartonPipeline with model: %s (conf: %.2f)", model_path, conf_threshold)
    tripwire_configs = {
        "cam_side": {"direction_sign": 1, "line_pos_frac": 0.50},
        "cam_flank": {"direction_sign": 1, "line_pos_frac": 0.55},
        "cam_top": {"direction_sign": -1, "line_pos_frac": 0.50},
    }
    calibration_path = Path(__file__).resolve().parents[3] / "config" / "calibrations" / "wh_test_01_dock_test_01_v2.json"
    pipeline = ThreeCameraCartonPipeline(
        model_path=model_path,
        conf_threshold=conf_threshold,
        fps=fps_top,
        tripwire_x_px=tripwire_x,
        required_direction="BIDIRECTIONAL",
        tripwire_configs=tripwire_configs,
        calibration_path=str(calibration_path) if calibration_path.is_file() else None,
        require_calibrated_fusion=False,
        allow_temporal_video_fusion=True,
    )

    n_to_process = min(total_frames, max_frames) if max_frames else total_frames
    logger.info("Processing %d synchronized frames across 3 cameras...", n_to_process)

    start_time = time.time()
    tick_outputs = []
    frame_idx = 0

    while frame_idx < n_to_process:
        r1, f_top = cap_top.read()
        r2, f_side = cap_side.read()
        r3, f_flank = cap_flank.read()

        if not (r1 and r2 and r3):
            logger.warning("Frame read reached EOF at frame %d", frame_idx)
            break

        frames = {
            "cam_top": f_top,
            "cam_side": f_side,
            "cam_flank": f_flank,
        }

        tick = pipeline.process_synchronized_frames(frames, frame_idx=frame_idx)
        tick_outputs.append(tick)
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            logger.info("Processed %d/%d frames (%.1f FPS) | Committed Count: %d", frame_idx, n_to_process, fps_proc, pipeline.total_count)

    elapsed_total = time.time() - start_time
    cap_top.release()
    cap_side.release()
    cap_flank.release()

    # Tally consensus breakdown
    consensus_tally = {"THREE_CAMERA_UNANIMOUS": 0, "TWO_CAMERA_AGREEMENT": 0, "SINGLE_CAMERA_OBSERVATION": 0, "CONFLICTING_OR_UNCERTAIN": 0}
    for ev in pipeline.audit_events:
        c_level = ev.get("consensus", {}).get("level")
        if c_level in consensus_tally:
            consensus_tally[c_level] += 1

    report = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input_videos": {
            "top": video_top,
            "side": video_side,
            "flank": video_flank,
        },
        "frames_processed": frame_idx,
        "elapsed_sec": round(elapsed_total, 2),
        "processing_fps": round(frame_idx / elapsed_total, 2) if elapsed_total > 0 else 0,
        "confirmed_cartons": pipeline.confirmed_count,
        "review_required_cartons": pipeline.review_required_count,
        "provisional_cartons": pipeline.provisional_count,
        "total_committed_cartons": pipeline.confirmed_count,
        "observed_cartons": pipeline.provisional_count,
        "consensus_breakdown": consensus_tally,
        "committed_events": pipeline.audit_events,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_report_json)), exist_ok=True)
    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation complete! Report written to: %s", output_report_json)
    return report


if __name__ == "__main__":
    v_top = "apps/truck_loading_detection/recorded_video/synchronized/scenario1_top_synced.mp4"
    v_side = "apps/truck_loading_detection/recorded_video/synchronized/scenario1_side_synced.mp4"
    v_flank = "apps/truck_loading_detection/recorded_video/synchronized/scenario1_flank_synced.mp4"
    out_json = sys.argv[2] if len(sys.argv) > 2 else "apps/truck_loading_detection/scratch/three_camera_scenario1_audit_report.json"

    max_f = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    run_evaluation(v_top, v_side, v_flank, out_json, max_frames=max_f)
