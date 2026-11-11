"""Missing source pictures must not become fabricated lip-sync motion evidence."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.video import lipsync, xcorr
from app.detectors.video.adapters.lipsync import AVInputs, StableSyncNetDetector
from app.detectors.video.adapters.lipsync_xcorr import LandmarkXcorrDetector, _mouth_status
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


def test_mouth_scope_distinguishes_unchanged_pixels_flat_proxy_and_missing_input():
    window = lipsync.Window("track:0", 0, 1., 100.)
    constant = {tick: np.full((24, 32), 100, dtype=np.float32) for tick in window.ticks}
    moving_luminance = {tick: pixels + tick for tick, pixels in constant.items()}
    articulating = {tick: pixels.copy() for tick, pixels in constant.items()}
    for tick, pixels in articulating.items():
        pixels[5:19, 8:24] -= tick

    def status(patches):
        return _mouth_status(window, patches, {tick: xcorr.aperture(patch) for tick, patch in patches.items()})

    assert status(constant) == "unchanged_mouth_pixels"
    assert status(moving_luminance) == "flat_mouth_proxy"
    assert status(articulating) == "measurable"
    del articulating[5]
    assert status(articulating) == "missing_mouth_input"


def test_offset_change_does_not_compare_across_omitted_mouth_windows():
    def measured(index, offset):
        window = lipsync.Window("track:0", index, 1., 100., hop=xcorr.HOP_TICKS)
        distances = np.ones((1, 2 * lipsync.VSHIFT + 1))
        distances[0, offset + lipsync.VSHIFT] = .1
        return lipsync.WindowSync(window, (window.start_tick,), distances, offset, .1, .9)

    windows = [measured(index, offset) for index, offset in ((2, 0), (3, 0), (6, 6), (7, 6))]
    assert lipsync.track_sync(windows, xcorr.SETTINGS).steps
    separated = lipsync.track_sync(windows, xcorr.SETTINGS, unmeasured_indices=(4, 5))
    assert not separated.steps and not separated.collapses
    assert tuple(len(run) for run in separated.runs) == (2, 2)


@pytest.mark.parametrize("gap", ["missing_source_pictures", "insufficient_speech"])
def test_adapter_does_not_bridge_windows_missing_before_mouth_admission(monkeypatch, gap):
    frame_count = 130
    missing = range(62, 75)
    picture_ticks = [i for i in range(frame_count) if gap != "missing_source_pictures" or i not in missing]
    pts = tuple(40 * i for i in picture_ticks)
    table = FrameTable("1/1000", pts, (), "I" * len(pts), 80, 80, 80, 80)
    face = {"bbox": {"x": 0, "y": 0, "w": 80, "h": 80},
            "landmarks": [[26, 21], [44, 21], [35, 30], [25, 40], [43, 40]]}
    frames = [{**face, "frame_index": i, "pts": point} for i, point in enumerate(pts)]
    rng = np.random.default_rng(37)
    pictures = []
    for level in rng.integers(12, 86, len(pts)):
        rgb = np.full((80, 80, 3), 100, dtype=np.uint8)
        rgb[35:45, 26:43] = level
        pictures.append(rgb)

    activity = np.ones(frame_count, dtype=bool)
    if gap == "insufficient_speech":
        activity[list(missing)] = False
    detected = lipsync.Speech(np.zeros(frame_count), np.ones(frame_count), np.zeros(frame_count), activity, -50.)
    pcm = np.zeros(frame_count * lipsync.HOP, dtype=np.float32)
    mapping = {"samples": len(pcm), "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
               "timeline_time_base": "1/16000",
               "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}]}
    store = SimpleNamespace(iter_many=lambda indices: ((i, pictures[i]) for i in indices))
    inputs = AVInputs(table, ({"track_id": "track:0"},), {"track:0": frames},
                      SimpleNamespace(status="analysed", pcm=pcm), store, mapping)
    exhibits = SimpleNamespace(series=lambda *args, **kwargs: "series", segments=lambda *args, **kwargs: "segments")
    ctx = SimpleNamespace(outputs=SimpleNamespace(av=inputs), deadline=None, exhibits=exhibits)
    monkeypatch.setattr(lipsync, "vad", lambda _pcm: detected)
    monkeypatch.setattr(lipsync, "speaker_score", lambda *_: .9)
    measured = []

    def window_sync(window, *_):
        offset = 0 if window.index < 4 else 6
        distances = np.ones((1, 2 * lipsync.VSHIFT + 1))
        distances[0, offset + lipsync.VSHIFT] = .1
        row = lipsync.WindowSync(window, (window.start_tick,), distances, offset, .1, .9)
        measured.append(row)
        return row

    monkeypatch.setattr(xcorr, "window_sync", window_sync)
    result = LandmarkXcorrDetector().run(ctx)
    assert [row.window.index for row in measured] == [2, 3, 6, 7]
    assert lipsync.track_sync(measured, xcorr.SETTINGS).steps
    assert result.status == "ok" and result.metrics["windows"] == 4
    assert result.metrics["offset_steps"] == 0 and result.aggregate is None
    track = next(u for u in result.unit_scores if u.unit_ref == "track:0")
    assert track.quality["longest_measured_run"] == 2
    assert "Offset changes unassessed" in result.observed
    assert "longest measured run had 2 consecutive windows" in result.observed
    assert "0 offset step" not in result.observed


def test_attributed_track_omits_held_windows_without_reporting_a_collapse(monkeypatch):
    frame_count = 200
    pts = tuple(40 * i for i in range(frame_count))
    table = FrameTable("1/1000", pts, (), "I" * frame_count, 80, 80, 80, 80)
    face = {"bbox": {"x": 0, "y": 0, "w": 80, "h": 80},
            "landmarks": [[26, 21], [44, 21], [35, 30], [25, 40], [43, 40]]}
    frames = [{**face, "frame_index": i, "pts": pts[i]} for i in range(frame_count)]
    rng = np.random.default_rng(41)
    originals = []
    for intensity in rng.integers(10, 85, frame_count):
        rgb = np.full((80, 80, 3), 100, dtype=np.uint8)
        rgb[35:45, 26:43] = intensity
        originals.append(rgb)
    pictures = [originals[50] if 50 <= i < 125 else frame for i, frame in enumerate(originals)]
    original_levels = np.array([.4 + xcorr.aperture(lipsync.mouth_patch(frame, face)) for frame in originals])
    detected_speech = lipsync.Speech(np.zeros(frame_count), original_levels, np.zeros(frame_count),
                                     np.ones(frame_count, dtype=bool), -50.)
    pcm = np.zeros(frame_count * lipsync.HOP, dtype=np.float32)
    mapping = {"samples": len(pcm), "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
               "timeline_time_base": "1/16000",
               "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}]}
    store = SimpleNamespace(iter_many=lambda indices: ((i, pictures[i]) for i in indices))
    inputs = AVInputs(table, ({"track_id": "track:0"},), {"track:0": frames},
                      SimpleNamespace(status="analysed", pcm=pcm), store, mapping)
    exhibits = SimpleNamespace(series=lambda *args, **kwargs: "series", segments=lambda *args, **kwargs: "segments")
    ctx = SimpleNamespace(outputs=SimpleNamespace(av=inputs), deadline=None, exhibits=exhibits)
    monkeypatch.setattr(lipsync, "vad", lambda _pcm: detected_speech)
    # Hold speaker attribution fixed so the test isolates window eligibility.
    monkeypatch.setattr(lipsync, "speaker_score", lambda *_: .9)

    grid = lipsync.tick_grid(table.times(), 8., len(pcm), audio_start_s=0.)
    windows = [window for window in lipsync.track_windows("track:0", dict(enumerate(frames)), grid, detected_speech,
                                                          hop=xcorr.HOP_TICKS)
               if xcorr.full_search(window, frame_count)]
    values = {tick: xcorr.aperture(lipsync.mouth_patch(pictures[grid.frames[tick]], face))
              for window in windows for tick in window.ticks}
    previous = lipsync.track_sync([xcorr.window_sync(window, values, original_levels) for window in windows], xcorr.SETTINGS)
    assert previous.collapses

    result = LandmarkXcorrDetector().run(ctx)
    assert result.status == "ok"
    assert result.metrics["unchanged_mouth_pixels"] == 5
    assert result.metrics["windows"] == len(windows) - 5
    assert result.metrics["correlation_collapses"] == 0
    assert result.aggregate is None
    assert all(2. <= row["start_s"] < row["end_s"] <= 5.
               for row in result.details["unmeasured_mouth_windows"])
    assert "mouth crops stayed unchanged" in result.reasoning
