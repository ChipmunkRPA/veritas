"""Real mux timestamps must survive decoding and address the same video/audio instant."""
from __future__ import annotations

import hashlib
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from app.detectors.video import lipsync, sampling
from app.detectors.video.adapters.common import AudioTrack
from app.detectors.video.adapters.lipsync import AVInputs, StableSyncNetDetector, source_audio_origin
from app.detectors.video.decode import probe_frames
from app.engine.stages.video import _audio


@pytest.fixture(params=["audio_later", "video_later"])
def offset_source(tmp_path, request):
    video = tmp_path / "different-stream-origins.mkv"
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if request.param == "video_later":
        cmd += ["-itsoffset", "0.4"]
    cmd += ["-f", "lavfi", "-i", "color=red:s=96x64:r=25:d=2"]
    if request.param == "audio_later":
        cmd += ["-itsoffset", "0.4"]
    cmd += ["-f", "lavfi", "-i", "sine=frequency=400:sample_rate=16000:duration=2",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "ffv1", "-c:a", "pcm_s16le", "-output_ts_offset", "5", str(video)]
    subprocess.run(cmd, check=True, timeout=30)
    source_sha = hashlib.sha256(video.read_bytes()).hexdigest()
    table = probe_frames(video)
    audio, mapping = _audio(video, True, source_sha256=source_sha)
    assert audio.status == "analysed"
    return video, source_sha, table, audio, mapping, request.param


def test_av_ticks_address_mux_source_times(offset_source):
    _, _, table, audio, mapping, side = offset_source
    inputs = AVInputs(table, (), {}, audio, None, mapping)
    origin = source_audio_origin(inputs)
    assert isinstance(origin, float)
    assert origin == pytest.approx(5.4 if side == "audio_later" else 5.0)
    grid = lipsync.tick_grid(table.times(), sampling.duration_s(table), len(audio.pcm), audio_start_s=origin)
    assert grid.seconds(0) == origin
    if side == "audio_later":
        assert grid.frames[0] == 10
        assert table.seconds(grid.frames[0]) == pytest.approx(origin)
    else:
        assert grid.frames[:10] == (-1,) * 10
        assert grid.frames[10] == 0
        assert table.seconds(grid.frames[10]) == pytest.approx(grid.seconds(10))
    valid = [(tick, frame) for tick, frame in enumerate(grid.frames) if frame >= 0]
    assert all(table.seconds(frame) == pytest.approx(grid.seconds(tick)) for tick, frame in valid)
    assert grid.seconds(grid.ticks) <= table.seconds(0) + sampling.duration_s(table) + 1e-6
    # A successful voice decode without the same source object must not supply AV timing.
    assert _audio(offset_source[0], True, source_sha256="0" * 64)[0].status == "unreadable"


def test_windows_never_include_audio_only_prefix():
    times = 5.4 + np.arange(50) / lipsync.FPS
    grid = lipsync.tick_grid(times, 2, 2 * lipsync.SR, audio_start_s=5)
    n = 50
    speech = lipsync.Speech(np.zeros(n), np.ones(n), np.zeros(n), np.ones(n, dtype=bool), -50)
    frames = {i: {"bbox": {"x": 0, "y": 0, "w": 100, "h": 100}} for i in range(50)}
    windows = lipsync.track_windows("t", frames, grid, speech)
    assert [w.index for w in windows] == [1]
    assert all(grid.frames[k] >= 0 for w in windows for k in w.ticks)


@pytest.mark.parametrize("damage", ["missing", "changed_pcm", "invalid_partition", "discontinuous"])
def test_ambiguous_audio_clock_is_unavailable_before_scoring(offset_source, damage):
    _, _, table, audio, original, _ = offset_source
    mapping = deepcopy(original)
    if damage == "missing":
        mapping = None
    elif damage == "changed_pcm":
        mapping["pcm_sha256"] = "0" * 64
    elif damage == "invalid_partition":
        mapping["timeline"][0]["start_sample"] = 1
    else:
        n, start = mapping["samples"], mapping["timeline"][0]["source_pts"]
        mapping["timeline"] = [{"start_sample": 0, "end_sample": n // 2, "source_pts": start},
                               {"start_sample": n // 2, "end_sample": n, "source_pts": start + n // 2 + 1600}]
    inputs = AVInputs(table, (), {}, audio, None, mapping)
    reason = StableSyncNetDetector._windows(inputs)
    assert isinstance(reason, str)
    if damage == "discontinuous":
        before = (start + n // 2) / lipsync.SR
        after = (start + n // 2 + 1600) / lipsync.SR
        assert reason == ("Audio/video timing was unassessed. The decoded audio clock jumps forward "
                          f"from {before:.7f} s to {after:.7f} s on the source clock.")
    else:
        assert reason.startswith("audio-visual timing unavailable:")


def test_absent_audio_remains_absent(tmp_path):
    audio, mapping = _audio(tmp_path / "unused", False, source_sha256="0" * 64)
    assert audio == AudioTrack("absent") and mapping is None
