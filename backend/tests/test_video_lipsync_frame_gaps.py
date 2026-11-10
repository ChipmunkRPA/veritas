"""Missing source pictures must not become fabricated lip-sync motion evidence."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.video import lipsync
from app.detectors.video.adapters.lipsync import AVInputs, StableSyncNetDetector
from app.detectors.video.adapters.lipsync_xcorr import LandmarkXcorrDetector
from app.detectors.video.decode import FrameTable


def speech(n):
    return lipsync.Speech(np.zeros(n), np.ones(n), np.zeros(n), np.ones(n, dtype=bool), -50)


def test_long_timestamp_gap_cannot_supply_complete_speech_windows():
    times = np.r_[np.arange(10) / 25, 1.6 + np.arange(10) / 25]
    grid = lipsync.tick_grid(times, 2, 32000, audio_start_s=0)
    assert grid.frames[10] == 9  # 40-ms allowance includes its exact boundary.
    assert grid.frames[11:39] == (-1,) * 28
    frames = {i: {"bbox": {"w": 100, "h": 100}} for i in range(len(times))}
    assert lipsync.track_windows("track:0", frames, grid, speech(50)) == []
    assert lipsync.track_windows("track:0", frames, grid, speech(50), hop=12.5) == []


@pytest.mark.parametrize("fps", [24, 25, 30, 60])
def test_regular_cadence_keeps_nearest_frame_routing(fps):
    times = 5 + np.arange(fps * 2) / fps
    grid = lipsync.tick_grid(times, 2, 32000, audio_start_s=5)
    assert grid.ticks == 50 and all(i >= 0 for i in grid.frames)
    for tick, index in enumerate(grid.frames):
        distances = np.abs(times - grid.seconds(tick))
        assert distances[index] == pytest.approx(distances.min())


def test_a_short_gap_keeps_only_windows_with_actual_picture_support():
    times = np.r_[np.arange(20) / 25, np.arange(25, 75) / 25]
    grid = lipsync.tick_grid(times, 3, 48000, audio_start_s=0)
    frames = {i: {"bbox": {"w": 100, "h": 100}} for i in range(len(times))}
    windows = lipsync.track_windows("track:0", frames, grid, speech(75))
    assert [w.index for w in windows] == [1, 2]
    assert all(grid.frames[k] >= 0 for w in windows for k in w.ticks)


def test_single_picture_is_not_extended_through_a_long_display_duration():
    grid = lipsync.tick_grid(np.array([5.0]), 2, 32000, audio_start_s=5)
    assert grid.frames[:2] == (0, 0) and grid.frames[2:] == (-1,) * 48


def test_both_adapters_skip_gap_windows_before_store_or_model_access(monkeypatch):
    pts = tuple([5000 + 40 * i for i in range(10)] + [6600 + 40 * i for i in range(10)])
    table = FrameTable("1/1000", pts, (), "I" * len(pts), 96, 64, 96, 64)
    pcm = np.ones(32000, np.float32)
    mapping = {
        "samples": len(pcm),
        "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        "timeline_time_base": "1/16000",
        "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 80000}],
    }
    face = {"bbox": {"x": 0, "y": 0, "w": 96, "h": 64}, "landmarks": [[1, 1]] * 5}
    frames = [{**face, "frame_index": i, "pts": pts[i]} for i in range(len(pts))]
    inputs = AVInputs(
        table,
        ({"track_id": "track:0"},),
        {"track:0": frames},
        SimpleNamespace(status="analysed", pcm=pcm, samples=len(pcm)),
        None,
        mapping,
    )
    monkeypatch.setattr(lipsync, "vad", lambda pcm: speech(50))
    assert "timestamp-matched" in StableSyncNetDetector._windows(inputs)
    ctx = SimpleNamespace(outputs=SimpleNamespace(av=inputs))
    for detector in (StableSyncNetDetector(), LandmarkXcorrDetector()):
        result = detector.run(ctx)  # No frame store, model lease, exhibit sink or deadline is supplied.
        assert result.status == "not_applicable" and not result.unit_scores
