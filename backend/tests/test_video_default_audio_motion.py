# SPDX-License-Identifier: MIT
import hashlib
import shutil
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from app.core.canonical import canonical_hash
from app.detectors.audio.decode import decode_audio, decode_default_second_aac441
from app.detectors.video.decode import probe_frames
from app.detectors.video.viseme_review import supported_aac441_motion
from app.evidence import video_lip_agreement as agreement
from app.evidence import video_viseme_binding as binding

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *args], check=True, timeout=25)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def two_soundtracks(tmp_path):
    path = tmp_path / "two-soundtracks.mov"
    ffmpeg("-f", "lavfi", "-i", "color=size=720x480:rate=24:duration=2.1",
           "-f", "lavfi", "-i", "sine=frequency=400:sample_rate=44100:duration=2.1",
           "-f", "lavfi", "-i", "sine=frequency=900:sample_rate=44100:duration=2.1",
           "-map", "0:v", "-map", "1:a", "-map", "2:a",
           "-c:v", "libx264", "-threads:v", "1", "-c:a", "aac",
           "-disposition:a:0", "0", "-disposition:a:1", "default", str(path))
    return path


def test_unique_default_stream_decodes_native_pcm_without_altering_first_track(two_soundtracks):
    path = two_soundtracks
    first = decode_audio(path, source_sha256=sha(path))
    second = decode_audio(path, source_sha256=sha(path), stream_position=1)
    selected = decode_default_second_aac441(path, source_sha256=sha(path))
    assert selected is not None
    assert first.mapping["source_stream_index"] == 1
    assert selected.mapping["source_stream_index"] == 2 and not np.array_equal(selected.pcm, first.pcm)
    assert selected.wav == second.wav and np.array_equal(selected.pcm, second.pcm)
    assert first.mapping["transform"].startswith("First audio stream;")
    assert selected.mapping["transform"].startswith("Default second audio stream;")
    assert selected.mapping[agreement.AAC_44100_CLOCK] is True
    assert agreement.AAC_44100_CLOCK not in second.mapping
    assert binding.audio_origin(selected.mapping) == 0
    assert binding.presented_samples(selected.mapping) == 33_600
    assert binding.selected_audio(selected.mapping)["default_audio_dispositions"] == [0, 1]
    table = probe_frames(path)
    assert supported_aac441_motion(table, selected.mapping, [[0, len(table)]], len(table))
    assert not supported_aac441_motion(table, first.mapping, [[0, len(table)]], len(table))
    assert not supported_aac441_motion(table, second.mapping, [[0, len(table)]], len(table))


@pytest.mark.parametrize(("defaults", "extra_track"), [(("default", "0"), False),
                                                       (("default", "default"), False), (("0", "default"), True)])
def test_ambiguous_or_outside_scope_tracks_are_withheld(two_soundtracks, tmp_path, defaults, extra_track):
    output = tmp_path / "other-disposition.mov"
    streams = ["-map", "0"] + (["-map", "0:a:0", "-disposition:a:2", "0"] if extra_track else [])
    ffmpeg("-i", str(two_soundtracks), *streams, "-c", "copy",
           "-disposition:a:0", defaults[0], "-disposition:a:1", defaults[1], str(output))
    assert decode_default_second_aac441(output, source_sha256=sha(output)) is None


def test_selected_track_record_replays_and_cannot_change_the_ordinary_audio_inventory(two_soundtracks):
    path = two_soundtracks
    decoded = decode_default_second_aac441(path, source_sha256=sha(path))
    table = probe_frames(path)
    source = {"width": table.width, "height": table.height, "time_base": table.time_base, "frame_count": len(table)}
    raw = {"frames": [{"index": i, "pts": pts, "gap_units": None} for i, pts in enumerate(table.pts)],
           "phones": [], agreement.AAC_44100_MOTION_STATUS: "coverage"}
    record = binding.measured(raw, timestamps=table.pts, time_base=table.time_base, source=source, mapping=decoded.mapping)
    assert record["schema"] == binding.SCHEMA_WITH_DEFAULT_AAC441_MOTION
    assert record["selected_audio"] == binding.selected_audio(decoded.mapping)
    assert record["status"] == "insufficient_events" and record["affects_overall_confidence"] is False
    ordinary = decode_audio(path, source_sha256=sha(path))
    document = {binding.KEY: record, "source": source | {binding.COMMITMENT: canonical_hash(record)},
                "audio": {"source": ordinary.mapping}, "packets": []}
    binding.validate(document, timestamps=table.pts)

    changed = deepcopy(document)
    changed[binding.KEY]["selected_audio"]["default_audio_dispositions"] = [1, 0]
    changed["source"][binding.COMMITMENT] = canonical_hash(changed[binding.KEY])
    with pytest.raises(ValueError, match="original audio inventory"):
        binding.validate(changed, timestamps=table.pts)
    changed = deepcopy(document)
    changed["audio"]["source"]["source_stream_index"] = 2
    with pytest.raises(ValueError, match="original audio inventory"):
        binding.validate(changed, timestamps=table.pts)
