import time

import numpy as np

from apps.truck_loading_detection.three_camera_system.live_cameras import (
    LiveCameraManager,
    parse_camera_source,
)


class FakeCapture:
    def __init__(self, source, *args):
        self.source = source
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def read(self):
        time.sleep(0.01)
        if self.released:
            return False, None
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_camera_sources_accept_rtsp_urls_and_usb_indices():
    assert parse_camera_source("rtsp://camera/stream") == "rtsp://camera/stream"
    assert parse_camera_source("2") == 2


def test_camera_sources_reject_unsupported_values():
    try:
        parse_camera_source("camera-name")
    except ValueError as exc:
        assert "numeric USB device indices" in str(exc)
    else:
        raise AssertionError("Unsupported camera source was accepted")


def test_live_camera_manager_captures_three_frames_and_reports_health(monkeypatch):
    monkeypatch.setattr(
        "apps.truck_loading_detection.three_camera_system.live_cameras.cv2.VideoCapture",
        FakeCapture,
    )
    manager = LiveCameraManager()
    manager.start({
        "cam_top": "rtsp://top/stream",
        "cam_side": 0,
        "cam_flank": "rtsp://flank/stream",
    })
    try:
        deadline = time.monotonic() + 3.0
        status = manager.status()
        while not status["ready"] and time.monotonic() < deadline:
            time.sleep(0.02)
            status = manager.status()

        assert status["ready"] is True
        assert status["connected_cameras"] == 3
        assert status["counting_enabled"] is False
        assert status["cameras"]["cam_top"]["width"] == 64
        assert status["cameras"]["cam_side"]["height"] == 48
        assert manager.latest_jpeg("cam_flank").startswith(b"\xff\xd8")
    finally:
        manager.stop()

    assert manager.status()["active"] is False
    assert manager.status()["ready"] is False


def test_live_camera_manager_jitter_buffer_and_counting_worker(monkeypatch):
    monkeypatch.setattr(
        "apps.truck_loading_detection.three_camera_system.live_cameras.cv2.VideoCapture",
        FakeCapture,
    )
    manager = LiveCameraManager()
    manager.start({
        "cam_top": "rtsp://top/stream",
        "cam_side": 0,
        "cam_flank": "rtsp://flank/stream",
    })

    class FakePipeline:
        def __init__(self):
            self.confirmed_in = 2
            self.confirmed_out = 0
            self.net_count = 2
            self.review_required_count = 0
            self.calls = 0

        def process_synchronized_frames(self, frames, frame_idx, override_time_sec):
            self.calls += 1
            assert "cam_top" in frames
            assert "cam_side" in frames
            assert "cam_flank" in frames
            return type("Tick", (), {"active_tracklets": {}})()

    try:
        # Wait for cameras to connect
        deadline = time.monotonic() + 3.0
        while not manager.status()["ready"] and time.monotonic() < deadline:
            time.sleep(0.02)

        # 1. Test Jitter Buffer Triplet Extraction
        synced = manager.get_synchronized_triplet(max_jitter_sec=0.10)
        assert synced is not None
        ref_time, frame_dict = synced
        assert set(frame_dict.keys()) == {"cam_top", "cam_side", "cam_flank"}
        for f in frame_dict.values():
            assert isinstance(f, np.ndarray)

        # 2. Test Live Counting Activation
        fake_pipe = FakePipeline()
        manager.start_counting(fake_pipe, target_fps=30.0)

        # Allow worker thread to process at least a couple ticks
        tick_deadline = time.monotonic() + 2.0
        while fake_pipe.calls < 2 and time.monotonic() < tick_deadline:
            time.sleep(0.03)

        assert fake_pipe.calls >= 2
        st = manager.status()
        assert st["counting_enabled"] is True
        assert st["counting_blocker"] is None
        assert st["live_counts"] is not None
        assert st["live_counts"]["confirmed_loaded"] == 2
        assert st["live_counts"]["total_ticks"] >= 2

        # 3. Test Live Counting Deactivation
        manager.stop_counting()
        st_stopped = manager.status()
        assert st_stopped["counting_enabled"] is False
    finally:
        manager.stop()
