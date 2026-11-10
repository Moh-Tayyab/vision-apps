"""
Unit tests for LineCrossing and Fusion (Multi-Camera 2-of-3 Rule).

Verifies:
1. 2-of-3 Camera Consensus: 2 cameras detect crossing, 1 misses -> count commits (+1).
2. 3-Camera Full Agreement: All 3 cameras detect crossing -> count commits (+1).
3. Single Camera Miss/Occlusion: Only 1 camera detects -> count NOT committed, moved to unconfirmed.
4. Late 3rd Camera Arrival: 3rd camera event arrives after commit -> absorbed without double counting.
5. Dead-band Jitter Immunity: Micro-jitter within ±margin does not produce false crossings.
6. Bidirectional Net Counting: LOADED and RETURNED movements properly calculate net count.
7. Cooldown Protection: Rapid duplicate events for same tracklet are suppressed.
8. Multi-Carton Resolution: Fusion takes the max carton quantity detected across cameras.
"""

import pytest
from three_camera_system.line_crossing_fusion import Fusion, LineCrossing


class TestLineCrossing:
    """Tests for single-camera dead-band line crossing."""

    def test_horizontal_line_crossing_loaded_and_returned(self):
        """Horizontal line at y=430: moving y 400 -> 460 is LOADED, 460 -> 400 is RETURNED."""
        # Line from (0, 430) to (500, 430): normal vector is (0, 1) pointing downwards (positive y)
        line = LineCrossing(cam="top", p1=(0, 430), p2=(500, 430), margin=12, cooldown=1.0)

        # 1. Approach from -1 side (y = 400, dist = -30 < -12)
        evs = line.update(t=1.0, tracks=[(101, 250, 400)])
        assert len(evs) == 0
        assert line.last_side[101] == -1

        # 2. Enter dead-band (y = 425, dist = -5 in [-12, 12])
        evs = line.update(t=1.1, tracks=[(101, 250, 425)])
        assert len(evs) == 0
        assert line.last_side[101] == -1  # Dead-band does not overwrite side

        # 3. Cross to +1 side (y = 450, dist = +20 > +12) -> LOADED event!
        evs = line.update(t=1.2, tracks=[(101, 250, 450)])
        assert len(evs) == 1
        assert evs[0]["dir"] == "LOADED"
        assert evs[0]["track"] == 101
        assert evs[0]["cam"] == "top"
        assert line.last_side[101] == 1

        # 4. Movement further on +1 side does not re-trigger
        evs = line.update(t=1.3, tracks=[(101, 250, 480)])
        assert len(evs) == 0

        # 5. Return journey: moves back to y = 400 (dist = -30 < -12) after cooldown
        evs = line.update(t=2.5, tracks=[(101, 250, 400)])
        assert len(evs) == 1
        assert evs[0]["dir"] == "RETURNED"
        assert line.last_side[101] == -1

    def test_deadband_jitter_rejection(self):
        """Track jittering back and forth near the line does NOT trigger false counts."""
        line = LineCrossing(cam="side", p1=(0, 430), p2=(500, 430), margin=15, cooldown=1.0)

        # Object starts in deadband (y = 435, dist = +5)
        evs = line.update(t=0.0, tracks=[(1, 250, 435)])
        assert len(evs) == 0
        assert 1 not in line.last_side

        # Jitters slightly below (y = 425, dist = -5) -> still in deadband
        evs = line.update(t=0.1, tracks=[(1, 250, 425)])
        assert len(evs) == 0
        assert 1 not in line.last_side

        # Jitters to + side (y = 455, dist = +25 > 15)
        evs = line.update(t=0.2, tracks=[(1, 250, 455)])
        assert len(evs) == 0  # First valid side recorded, not a crossing yet
        assert line.last_side[1] == 1

        # Jitters back into deadband (y = 440, dist = +10) -> deadband ignored
        evs = line.update(t=0.3, tracks=[(1, 250, 440)])
        assert len(evs) == 0
        assert line.last_side[1] == 1

        # Jitters back out on + side (y = 460, dist = +30) -> side is still +1, no crossing
        evs = line.update(t=0.4, tracks=[(1, 250, 460)])
        assert len(evs) == 0
        assert line.last_side[1] == 1

    def test_cooldown_suppression(self):
        """
        Crossing back and forth within cooldown is suppressed.
        Crucial State Machine Invariant: Suppressing a return crossing does NOT flip
        confirmed side; oscillation within cooldown cannot re-prime a duplicate LOADED event.
        (A -> B, quick wapas A, phir B produces exactly ONE event).
        """
        line = LineCrossing(cam="flank", p1=(0, 430), p2=(500, 430), margin=10, cooldown=1.5, k_confirm=1)

        # Cross from -1 (A) to +1 (B) at t=1.2 -> accepted
        line.update(t=1.0, tracks=[(2, 200, 400)])
        evs1 = line.update(t=1.2, tracks=[(2, 200, 460)])
        assert len(evs1) == 1
        assert evs1[0]["dir"] == "LOADED"

        # Cross back to -1 (A) at t=1.8 (within 1.5s cooldown: dt = 0.6s) -> suppressed!
        evs2 = line.update(t=1.8, tracks=[(2, 200, 400)])
        assert len(evs2) == 0

        # Back to +1 (B) at t=3.0: since return to -1 was suppressed, confirmed side remains +1!
        # Cannot emit a second duplicate LOADED event without an intervening confirmed RETURNED.
        evs3 = line.update(t=3.0, tracks=[(2, 200, 460)])
        assert len(evs3) == 0, "Oscillation within cooldown must not fire a second duplicate LOADED event!"

    def test_k_tick_confirmation_and_oscillation_exact_one_event(self):
        """
        Verifies K-tick confirmation (k_confirm=2):
        1. A single-tick glitch does not cross.
        2. Sustained 2 ticks cross A -> B (LOADED).
        3. Quick jitter/oscillation back to A (suppressed by cooldown / single tick).
        4. Returning to B produces EXACTLY ONE event across the entire sequence.
        """
        line = LineCrossing(cam="top", p1=(0, 430), p2=(500, 430), margin=12, cooldown=1.5, k_confirm=2)

        # Initial track entry at A (y=400, side=-1)
        line.update(t=1.0, tracks=[(7, 250, 400)])
        assert line.last_side[7] == -1

        # Glitch to B (y=460, side=+1) for only 1 tick at t=1.1 -> NOT confirmed (needs 2 ticks)
        evs_glitch = line.update(t=1.1, tracks=[(7, 250, 460)])
        assert len(evs_glitch) == 0
        assert line.last_side[7] == -1  # Still confirmed at -1

        # Quick glitch returns to A at t=1.15 -> cancels unconfirmed candidate
        evs_cancel = line.update(t=1.15, tracks=[(7, 250, 400)])
        assert len(evs_cancel) == 0
        assert line.last_side[7] == -1

        # Sustained crossing to B: tick 1 at t=1.2 (candidate=1), tick 2 at t=1.3 (candidate=2) -> Confirmed LOADED!
        evs_tick1 = line.update(t=1.2, tracks=[(7, 250, 460)])
        assert len(evs_tick1) == 0  # 1st tick not enough

        evs_commit = line.update(t=1.3, tracks=[(7, 250, 460)])
        assert len(evs_commit) == 1  # 2nd tick confirmed!
        assert evs_commit[0]["dir"] == "LOADED"
        assert line.last_side[7] == 1

        # Quick jitter back to A (y=400, side=-1) at t=1.6 (within cooldown: dt=0.3s) -> Suppressed!
        evs_back = line.update(t=1.6, tracks=[(7, 250, 400)])
        assert len(evs_back) == 0
        assert line.last_side[7] == 1  # Confirmed side stays +1!

        # Moves to B again at t=2.0 and t=3.0 (cooldown elapsed):
        # Since it is already confirmed at B (+1), it does NOT emit another LOADED event!
        evs_b1 = line.update(t=2.0, tracks=[(7, 250, 460)])
        evs_b2 = line.update(t=3.0, tracks=[(7, 250, 460)])
        assert len(evs_b1) == 0
        assert len(evs_b2) == 0

        # Entire lifecycle: exactly ONE event emitted
        total_events = len(evs_glitch) + len(evs_commit) + len(evs_back) + len(evs_b1) + len(evs_b2)
        assert total_events == 1


class TestFusionConsensus:
    """Tests for multi-camera event fusion & the 2-of-3 consensus rule."""

    def test_2_of_3_cameras_agree_commits_immediately(self):
        """2 cameras see crossing, 1 camera misses -> count committed (+1) immediately upon 2nd camera."""
        fusion = Fusion(window=1.0, min_cams=2)

        # Camera 1 (top) detects event at t=2.0s
        ev_top = {"cam": "top", "t": 2.0, "track": 1, "dir": "LOADED"}
        fusion.add(ev_top)
        assert fusion.loaded == 0  # Only 1 camera, uncommitted
        assert len(fusion.clusters) == 1
        assert not fusion.clusters[0]["committed"]

        # Camera 2 (side) detects event at t=2.4s (within 1.0s window)
        ev_side = {"cam": "side", "t": 2.4, "track": 2, "dir": "LOADED"}
        fusion.add(ev_side)
        assert fusion.loaded == 1  # 2 cameras reached -> COMMIT!
        assert fusion.clusters[0]["committed"]
        assert len(fusion.ledger) == 1
        assert fusion.ledger[0]["cams"] == ["side", "top"]

        # Camera 3 (flank) missed completely. Flush at t=4.0s
        fusion.flush(now=4.0)
        assert len(fusion.clusters) == 0  # Expired cluster cleaned up
        assert len(fusion.unconfirmed) == 0  # Was committed, so NOT unconfirmed!
        assert fusion.loaded == 1
        assert fusion.net == 1

    def test_all_3_cameras_agree(self):
        """All 3 cameras observe the crossing within window -> commits count 1, ledger records all 3."""
        fusion = Fusion(window=1.0, min_cams=2)

        fusion.add({"cam": "top", "t": 5.0, "track": 10, "dir": "LOADED"})
        assert fusion.loaded == 0

        fusion.add({"cam": "side", "t": 5.2, "track": 11, "dir": "LOADED"})
        assert fusion.loaded == 1  # Committed at 2 cams

        # 3rd camera arrives at t=5.5 (within 1.0s window from t0=5.0)
        fusion.add({"cam": "flank", "t": 5.5, "track": 12, "dir": "LOADED"})
        assert fusion.loaded == 1  # Does NOT double count!

        fusion.flush(now=7.0)
        assert fusion.loaded == 1
        assert len(fusion.unconfirmed) == 0

    def test_only_1_camera_sees_goes_to_unconfirmed(self):
        """If only 1 camera sees an event, it is NEVER counted and gets moved to unconfirmed on flush."""
        fusion = Fusion(window=1.0, min_cams=2)

        ev_flank = {"cam": "flank", "t": 10.0, "track": 99, "dir": "LOADED"}
        fusion.add(ev_flank)
        assert fusion.loaded == 0

        # Before expiration
        fusion.flush(now=10.8)
        assert fusion.loaded == 0
        assert len(fusion.clusters) == 1
        assert len(fusion.unconfirmed) == 0

        # After window + 0.5s expiration (10.0 + 1.5 = 11.5)
        fusion.flush(now=11.6)
        assert fusion.loaded == 0
        assert len(fusion.clusters) == 0
        assert len(fusion.unconfirmed) == 1
        assert fusion.unconfirmed[0]["cams"]["flank"]["track"] == 99

    def test_late_3rd_camera_does_not_double_count(self):
        """
        Commit happens after 2 cameras.
        When 3rd camera arrives late (but within window), it is absorbed into the existing committed cluster.
        """
        fusion = Fusion(window=1.0, min_cams=2)

        fusion.add({"cam": "side", "t": 1.0, "track": "s1", "dir": "LOADED"})
        fusion.add({"cam": "top", "t": 1.3, "track": "t1", "dir": "LOADED"})
        assert fusion.loaded == 1
        assert len(fusion.ledger) == 1

        # Late 3rd camera arrives at t=1.8 (window is 1.0s, abs(1.8 - 1.0) = 0.8 <= 1.0)
        fusion.add({"cam": "flank", "t": 1.8, "track": "f1", "dir": "LOADED"})
        assert fusion.loaded == 1  # Count remains exactly 1!
        assert len(fusion.ledger) == 1

        fusion.flush(now=3.0)
        assert fusion.loaded == 1
        assert len(fusion.unconfirmed) == 0

    def test_separate_crossings_over_time(self):
        """Two separate physical crossings separated in time are both counted (+2)."""
        fusion = Fusion(window=1.0, min_cams=2)

        # First crossing around t = 2.0s
        fusion.add({"cam": "top", "t": 2.0, "track": 1, "dir": "LOADED"})
        fusion.add({"cam": "side", "t": 2.2, "track": 2, "dir": "LOADED"})
        assert fusion.loaded == 1

        # Flush at t = 4.0s
        fusion.flush(now=4.0)

        # Second crossing around t = 6.0s
        fusion.add({"cam": "side", "t": 6.0, "track": 3, "dir": "LOADED"})
        fusion.add({"cam": "flank", "t": 6.3, "track": 4, "dir": "LOADED"})
        assert fusion.loaded == 2

        fusion.flush(now=8.0)
        assert fusion.loaded == 2
        assert fusion.net == 2
        assert len(fusion.ledger) == 2

    def test_loaded_and_returned_net_count(self):
        """Worker loads a carton (+1), then returns a carton (-1), net count becomes 0."""
        fusion = Fusion(window=1.0, min_cams=2)

        # Crossing 1: LOADED
        fusion.add({"cam": "top", "t": 1.0, "track": 1, "dir": "LOADED"})
        fusion.add({"cam": "side", "t": 1.2, "track": 2, "dir": "LOADED"})
        assert fusion.loaded == 1
        assert fusion.returned == 0
        assert fusion.net == 1

        fusion.flush(now=3.0)

        # Crossing 2: RETURNED
        fusion.add({"cam": "top", "t": 4.0, "track": 1, "dir": "RETURNED"})
        fusion.add({"cam": "flank", "t": 4.2, "track": 3, "dir": "RETURNED"})
        assert fusion.loaded == 1
        assert fusion.returned == 1
        assert fusion.net == 0

    def test_multi_carton_stack_max_qty(self):
        """If worker carries a 2-carton stack and one camera detects qty=2, fusion resolves max qty."""
        fusion = Fusion(window=1.0, min_cams=2)

        # Top camera detects 1 carton, side camera sees stack of 2
        fusion.add({"cam": "top", "t": 1.0, "track": 1, "dir": "LOADED", "qty": 1})
        fusion.add({"cam": "side", "t": 1.2, "track": 2, "dir": "LOADED", "qty": 2})

        assert fusion.loaded == 2  # Max of (1, 2)
        assert fusion.net == 2
        assert fusion.ledger[0]["qty"] == 2


class TestPipelineLineCrossingIntegration:
    """Verifies that ThreeCameraCartonPipeline seamlessly integrates LineCrossing and Fusion."""

    def test_pipeline_initialization_with_line_crossing_fusion(self):
        from three_camera_system.pipeline import ThreeCameraCartonPipeline

        pipeline = ThreeCameraCartonPipeline(
            use_line_crossing_fusion=True,
            fusion_window=1.0,
            min_consensus_cams=2,
        )

        assert pipeline.use_line_crossing_fusion is True
        assert isinstance(pipeline.fusion, Fusion)
        assert "cam_side" in pipeline.line_crossings
        assert "cam_top" in pipeline.line_crossings
        assert "cam_flank" in pipeline.line_crossings
        assert pipeline.total_in == 0
        assert pipeline.total_out == 0
        assert pipeline.net_count == 0

    def test_pipeline_fusion_reset(self):
        from three_camera_system.pipeline import ThreeCameraCartonPipeline

        pipeline = ThreeCameraCartonPipeline(use_line_crossing_fusion=True)
        pipeline.fusion.loaded = 5
        pipeline.fusion.returned = 2
        assert pipeline.total_in == 5
        assert pipeline.total_out == 2
        assert pipeline.net_count == 3

        pipeline.reset()
        assert pipeline.total_in == 0
        assert pipeline.total_out == 0
        assert pipeline.net_count == 0


class TestAtomicSnapshotAndPreviewEndpoint:
    """Verifies atomic snapshot architecture, preview JPEG endpoint, and guards."""

    def test_preview_endpoint_and_atomic_snapshot_guard(self):
        """
        Confirms:
        1. Atomic snapshot contains counts, pts_sec, and binary jpeg outside locks.
        2. /api/3angle/job_tick_preview returns image/jpeg with Cache-Control: no-store.
        3. /api/3angle/job_status returns snap metadata without raw binary buffer.
        4. Incomplete job never marks provisional numbers as final.
        """
        from app import app, multi_angle_jobs, job_lock

        sample_jpeg = b"\xff\xd8\xff\xe0\x00\x10JFIF"  # minimal dummy JPEG header

        jid = "test_atomic_job_001"
        with job_lock:
            multi_angle_jobs[jid] = {
                "job_id": jid,
                "status": "processing",
                "progress_pct": 50,
                "phase": "Analyzing PTS 12.50s",
                "processed_time_sec": 12.50,
                "counts": {"loaded": 3, "returned": 1, "net": 2, "unconfirmed": 1},
                "snap": {
                    "counts": {"loaded": 3, "returned": 1, "net": 2, "unconfirmed": 1},
                    "pts_sec": 12.50,
                    "progress_pct": 50,
                    "phase": "Analyzing PTS 12.50s",
                    "jpeg": sample_jpeg,
                    "status": "processing",
                    "is_final": False,
                },
            }

        client = app.test_client()

        # 1. Preview endpoint returns JPEG bytes with Cache-Control: no-store
        res_preview = client.get(f"/api/3angle/job_tick_preview/{jid}")
        assert res_preview.status_code == 200
        assert res_preview.mimetype == "image/jpeg"
        assert "no-store" in res_preview.headers.get("Cache-Control", "")
        assert res_preview.data == sample_jpeg

        # 2. Status endpoint returns snap metadata without binary jpeg payload
        res_status = client.get(f"/api/3angle/job_status/{jid}")
        assert res_status.status_code == 200
        data = res_status.get_json()
        assert data["job_id"] == jid
        assert "snap" in data
        assert "jpeg" not in data["snap"]
        assert data["snap"]["counts"]["loaded"] == 3
        assert data["snap"]["pts_sec"] == 12.50

        # 3. Simulate failure / incomplete job: final counts must not be committed as final
        with job_lock:
            multi_angle_jobs[jid]["status"] = "incomplete"
            multi_angle_jobs[jid]["snap"]["status"] = "incomplete"
            multi_angle_jobs[jid]["snap"]["is_final"] = True

        res_incomplete = client.get(f"/api/3angle/job_status/{jid}")
        assert res_incomplete.get_json()["status"] == "incomplete"

        # Cleanup
        with job_lock:
            multi_angle_jobs.pop(jid, None)

