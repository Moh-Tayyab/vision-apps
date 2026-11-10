import json
import os
import shutil
import subprocess
import tempfile
import unittest
from typing import List, Tuple, Dict, Optional, Any
from unittest import mock

from tools.multicam_normalizer import (
    MalformedVideoError,
    probe_video_metadata,
    SyncPoint,
    fit_alignment_model,
    escape_ffmpeg_filter_path,
    detect_visual_flash_events,
    process_warehouse_scenarios,
)


class TestMultiCamNormalizerUnit(unittest.TestCase):

    def _make_mock_ffprobe_frames(
        self, pts_list, pkt_dur=0.033333, r_frame_rate="30/1", duration=None, nb_frames=None
    ):
        stream = {"width": 320, "height": 240, "r_frame_rate": r_frame_rate}
        if duration is not None:
            stream["duration"] = str(duration)
        if nb_frames is not None:
            stream["nb_frames"] = str(nb_frames)
        frames = []
        for t in pts_list:
            f_entry = {"best_effort_timestamp_time": f"{t:.6f}"}
            if pkt_dur is not None:
                f_entry["pkt_duration_time"] = f"{pkt_dur:.6f}"
            frames.append(f_entry)
        return json.dumps({"streams": [stream], "format": {}, "frames": frames})

    def test_1_metadata_duration_non_zero_start_pts_and_malformed_policies(self):
        """Priority 1 & 4: Non-zero source start_pts, measured vs estimated duration, and duplicate/monotonic checks."""
        # Non-zero start_pts = 100.500s, 90 frames @ 30fps -> relative timestamps [0.0, 2.966667]
        pts_nonzero = [100.500 + (i / 30.0) for i in range(90)]
        json_nz = self._make_mock_ffprobe_frames(pts_nonzero, pkt_dur=1.0 / 30.0, duration=3.0, nb_frames=90)

        with mock.patch("os.path.exists", return_value=True):
            meta = probe_video_metadata(
                "nonzero_start.mp4",
                runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=json_nz)
            )
            self.assertAlmostEqual(meta.start_pts, 100.500, places=4)
            self.assertAlmostEqual(meta.decoded_rel_timestamps[0], 0.0, places=6)
            self.assertAlmostEqual(meta.decoded_rel_timestamps[-1], 89 / 30.0, places=5)
            self.assertEqual(meta.duration_source, "measured_from_decoded_pts_and_frame_duration")
            self.assertAlmostEqual(meta.duration, 3.0, places=4)

        # Duplicate timestamp policy: 1 duplicate in 90 frames (1.11%) passes under warn_under_threshold (2%),
        # but raises MalformedVideoError under reject_all.
        pts_dup = list(pts_nonzero)
        pts_dup[10] = pts_dup[9]
        json_dup = self._make_mock_ffprobe_frames(pts_dup)
        with mock.patch("os.path.exists", return_value=True):
            meta_warn = probe_video_metadata(
                "dup.mp4",
                duplicate_policy="warn_under_threshold",
                max_duplicate_ratio=0.02,
                runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=json_dup)
            )
            self.assertEqual(meta_warn.duplicate_pts_count, 1)

            with self.assertRaises(MalformedVideoError):
                probe_video_metadata(
                    "dup.mp4",
                    duplicate_policy="reject_all",
                    runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=json_dup)
                )

    def test_2_alignment_model_validation_and_bounds(self):
        """Priority 5: Test fit vs held-out validation points, uniqueness, range bounds, and scale limits."""
        # Fit on 2 points, validate on 2 independent held-out points
        fit_pts = [
            SyncPoint("pick_1", cam_rel_sec=1.500, ref_rel_sec=1.000),
            SyncPoint("pick_3", cam_rel_sec=4.500, ref_rel_sec=4.000),
        ]
        val_pts = [
            SyncPoint("pick_2", cam_rel_sec=3.002, ref_rel_sec=2.500),
            SyncPoint("pick_4", cam_rel_sec=5.498, ref_rel_sec=5.000),
        ]
        model = fit_alignment_model(
            "side.mp4", "top.mp4",
            sync_points=fit_pts,
            validation_sync_points=val_pts,
            cam_duration=6.0,
            ref_duration=6.0,
            target_fps=30
        )
        self.assertEqual(model.model_type, "constant_offset")
        self.assertEqual(model.scale_a, 1.0)
        self.assertAlmostEqual(model.offset_b, -0.500, places=5)
        self.assertTrue(model.model_fit_verified)
        self.assertTrue(model.independent_validation_verified)
        self.assertLess(model.validation_rmse_ms, 3.0)

        # Reject overlapping fit and validation event names
        with self.assertRaises(ValueError):
            fit_alignment_model(
                "side.mp4", "top.mp4",
                sync_points=fit_pts,
                validation_sync_points=[SyncPoint("pick_1", 1.5, 1.0)]
            )

        # Reject excessive scale deviation (> 10% clock drift)
        bad_drift = [
            SyncPoint("p1", cam_rel_sec=1.0, ref_rel_sec=1.0),
            SyncPoint("p2", cam_rel_sec=10.0, ref_rel_sec=13.0),  # scale_a = 1.333
        ]
        with self.assertRaises(ValueError):
            fit_alignment_model("side.mp4", "top.mp4", sync_points=bad_drift)

        # Reject out-of-range sync point exceeding camera duration
        with self.assertRaises(ValueError):
            fit_alignment_model(
                "side.mp4", "top.mp4",
                sync_points=[SyncPoint("p1", cam_rel_sec=7.5, ref_rel_sec=7.0)],
                cam_duration=6.0, ref_duration=8.0
            )

    def test_3_escape_ffmpeg_filter_path(self):
        """Priority 4: Verify special character escaping for lavfi movie= filter."""
        escaped = escape_ffmpeg_filter_path("/tmp/dir with spaces/cam:1_[top],test's.mp4")
        self.assertTrue(escaped.startswith("'") and escaped.endswith("'"))
        self.assertIn(r"\:", escaped)
        self.assertIn(r"\,", escaped)
        self.assertIn(r"\[", escaped)
        self.assertIn(r"\'", escaped)


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg/FFprobe binaries required for integration tests"
)
class TestMultiCamNormalizerIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="multicam_audit_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_test_video(
        self,
        path: str,
        fps: int,
        duration: float,
        flash_times: List[float],
        base_color: str = "gray",
        start_pts_offset: float = 0.0,
        vfr_jitter: bool = False
    ):
        filters = []
        for ft in flash_times:
            filters.append(
                f"drawbox=x=20:y=20:w=280:h=200:color=white:t=fill:"
                f"enable='between(t,{ft:.4f},{ft + 0.100:.4f})'"
            )
        if vfr_jitter:
            filters.append("setpts='PTS + (0.006 * mod(N, 2))/TB'")
        elif abs(start_pts_offset) > 1e-6:
            filters.append(f"setpts='PTS + ({start_pts_offset:.6f}/TB)'")

        vf_str = ",".join(filters) if filters else "null"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c={base_color}:s=320x240:r={fps}:d={duration}",
            "-vf", vf_str,
            "-fps_mode", "vfr" if (vfr_jitter or abs(start_pts_offset) > 1e-6) else "cfr",
            "-output_ts_offset", f"{start_pts_offset:.6f}",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
            path
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    def _get_luma_profile(self, path: str) -> List[Tuple[int, float, float]]:
        escaped = escape_ffmpeg_filter_path(path)
        cmd = [
            "ffprobe", "-v", "error",
            "-f", "lavfi",
            "-i", f"movie={escaped},signalstats",
            "-show_entries", "frame=best_effort_timestamp_time:frame_tags=lavfi.signalstats.YAVG",
            "-of", "json"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        frames = json.loads(res.stdout).get("frames", [])
        t0 = float(frames[0]["best_effort_timestamp_time"])
        return [
            (idx, round(float(f["best_effort_timestamp_time"]) - t0, 6), float(f["tags"]["lavfi.signalstats.YAVG"]))
            for idx, f in enumerate(frames)
        ]

    def test_1_union_black_pad_leading_and_trailing_boundaries(self):
        """
        Priority 1, 2, 3, 6:
        Tests `common_union_black_pad` when `side` camera both starts AFTER `common_start` AND
        ends BEFORE `common_end` (which previously failed due to `shortest=1`), with spaces in filenames
        and non-zero source PTS offset on `flank`.
        Verifies exact active frame boundaries [first_active, last_active], held-out validation points,
        visual alignment, and carton-counting ground truth.
        """
        top_in = os.path.join(self.temp_dir, "top cam 1.mp4")
        side_in = os.path.join(self.temp_dir, "side cam [2].mp4")
        flank_in = os.path.join(self.temp_dir, "flank cam 3.mp4")

        # 4 physical carton crossings on master reference timeline:
        # Fit points: event 1 (0.80s) and event 4 (3.80s)
        # Held-out independent validation points: event 2 (1.80s) and event 3 (2.80s)
        ref_events = [0.800, 1.800, 2.800, 3.800]
        # side starts at +0.50s on ref timeline and has duration 3.80s -> mapped [0.50s, 4.30s]
        side_events = [t - 0.500 for t in ref_events]  # [0.30, 1.30, 2.30, 3.30]
        # flank starts at -0.40s on ref timeline and has duration 5.40s -> mapped [-0.40s, 5.00s]
        flank_events = [t + 0.400 for t in ref_events]  # [1.20, 2.20, 3.20, 4.20]

        self._generate_test_video(top_in, fps=30, duration=4.6, flash_times=ref_events)
        self._generate_test_video(side_in, fps=24, duration=3.8, flash_times=side_events)
        self._generate_test_video(
            flank_in, fps=60, duration=5.4, flash_times=flank_events, start_pts_offset=12.500
        )

        # Independently detect visual events in raw inputs
        in_top = detect_visual_flash_events(top_in)
        in_side = detect_visual_flash_events(side_in)
        in_flank = detect_visual_flash_events(flank_in)
        self.assertEqual(len(in_top), 4)
        self.assertEqual(len(in_side), 4)
        self.assertEqual(len(in_flank), 4)

        # Fit ONLY on events 0 and 3; hold out events 1 and 2 for independent validation
        fit_cfg = {
            "sc_1": {
                top_in: [SyncPoint(f"ev_{i+1}", in_top[i].onset_rel_sec, in_top[i].onset_rel_sec) for i in (0, 3)],
                side_in: [SyncPoint(f"ev_{i+1}", in_side[i].onset_rel_sec, in_top[i].onset_rel_sec) for i in (0, 3)],
                flank_in: [SyncPoint(f"ev_{i+1}", in_flank[i].onset_rel_sec, in_top[i].onset_rel_sec) for i in (0, 3)],
            }
        }
        val_cfg = {
            "sc_1": {
                top_in: [SyncPoint(f"ev_{i+1}", in_top[i].onset_rel_sec, in_top[i].onset_rel_sec) for i in (1, 2)],
                side_in: [SyncPoint(f"ev_{i+1}", in_side[i].onset_rel_sec, in_top[i].onset_rel_sec) for i in (1, 2)],
                flank_in: [SyncPoint(f"ev_{i+1}", in_flank[i].onset_rel_sec, in_top[i].onset_rel_sec) for i in (1, 2)],
            }
        }

        out_dir = os.path.join(self.temp_dir, "out_union")
        rep_path = os.path.join(self.temp_dir, "rep_union.json")

        report = process_warehouse_scenarios(
            scenarios={"sc_1": [top_in, side_in, flank_in]},
            sync_config=fit_cfg,
            validation_sync_config=val_cfg,
            carton_ground_truth={"sc_1": [ev.onset_rel_sec for ev in in_top]},
            output_dir=out_dir,
            report_path=rep_path,
            target_fps=30,
            duration_policy="common_union_black_pad",
            verify_visual_markers=True
        )

        sc = report["scenarios"]["sc_1"]
        self.assertTrue(sc["all_verifications_passed"], msg=json.dumps(sc, indent=2))
        self.assertEqual(sc["synchronization_status"], "INDEPENDENTLY_AND_VISUALLY_VERIFIED")
        self.assertTrue(sc["carton_counting_verification"]["passed"])

        # Inspect `side cam [2].mp4` pixel luma at exact leading and trailing boundaries:
        # Common union timeline is [-0.40s, 5.00s] (duration = 5.40s = 162 frames).
        # `side` active interval on ref is [0.50s, 4.30s] -> on output timeline [0.90s, 4.70s] -> frames [27..140].
        side_out = os.path.join(out_dir, os.path.basename(side_in))
        luma = self._get_luma_profile(side_out)
        self.assertEqual(len(luma), 162)

        first_act, last_act = sc["output_verification"][side_in]["coverage"]["active_output_frame_range"]
        self.assertEqual(first_act, 26)
        self.assertEqual(last_act, 140)

        # Leading boundary check: frame 26 is black (YAVG ~ 16), frame 27 is active gray (YAVG > 100)
        self.assertAlmostEqual(luma[first_act - 1][2], 16.0, delta=1.0)
        self.assertGreater(luma[first_act][2], 100.0)

        # Trailing boundary check: frame 140 is active gray (YAVG > 100), frames 141..161 are black (YAVG ~ 16)
        self.assertGreater(luma[last_act][2], 100.0)
        for idx in range(last_act + 1, 162):
            self.assertAlmostEqual(luma[idx][2], 16.0, delta=1.0, msg=f"Trailing frame {idx} not black")

    def test_2_common_overlap_trimming_and_linear_drift(self):
        """
        Priority 1, 5, 6:
        Tests `common_overlap` trimming out-of-window events and genuine linear drift (scale_a = 1.04)
        with VFR input.
        """
        ref_in = os.path.join(self.temp_dir, "ref.mp4")
        drift_in = os.path.join(self.temp_dir, "drift.mp4")

        ref_events = [1.000, 3.000, 5.000, 7.000]
        scale_true = 1.04
        drift_events = [t / scale_true for t in ref_events]

        self._generate_test_video(ref_in, fps=30, duration=8.5, flash_times=ref_events)
        self._generate_test_video(drift_in, fps=30, duration=8.0, flash_times=drift_events, vfr_jitter=True)

        in_ref = detect_visual_flash_events(ref_in)
        in_drift = detect_visual_flash_events(drift_in)

        fit_cfg = {
            "sc_drift": {
                ref_in: [SyncPoint(f"ev_{i+1}", in_ref[i].onset_rel_sec, in_ref[i].onset_rel_sec) for i in (0, 3)],
                drift_in: [SyncPoint(f"ev_{i+1}", in_drift[i].onset_rel_sec, in_ref[i].onset_rel_sec) for i in (0, 3)],
            }
        }
        val_cfg = {
            "sc_drift": {
                ref_in: [SyncPoint(f"ev_{i+1}", in_ref[i].onset_rel_sec, in_ref[i].onset_rel_sec) for i in (1, 2)],
                drift_in: [SyncPoint(f"ev_{i+1}", in_drift[i].onset_rel_sec, in_ref[i].onset_rel_sec) for i in (1, 2)],
            }
        }

        out_dir = os.path.join(self.temp_dir, "out_drift")
        rep_path = os.path.join(self.temp_dir, "rep_drift.json")

        report = process_warehouse_scenarios(
            scenarios={"sc_drift": [ref_in, drift_in]},
            sync_config=fit_cfg,
            validation_sync_config=val_cfg,
            output_dir=out_dir,
            report_path=rep_path,
            target_fps=30,
            duration_policy="common_overlap",
            verify_visual_markers=True
        )

        sc = report["scenarios"]["sc_drift"]
        self.assertTrue(sc["all_verifications_passed"], msg=json.dumps(sc, indent=2))
        self.assertEqual(sc["synchronization_models"][drift_in]["model_type"], "linear_drift")
        self.assertEqual(sc["synchronization_status"], "INDEPENDENTLY_AND_VISUALLY_VERIFIED")


if __name__ == "__main__":
    unittest.main()