import hashlib
import subprocess

import pytest

from app.detectors.audio.decode import decode_audio, decode_default_second_aac441
from app.evidence import video_viseme_binding as binding
from app.reports.verify_video_source import _decode_lip_soundtrack


def _two_aac(path, *, default_second):
    first, second = ("0", "default") if default_second else ("default", "0")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
         "sine=frequency=410:sample_rate=44100:duration=2.1", "-f", "lavfi", "-i",
         "sine=frequency=810:sample_rate=44100:duration=2.1", "-map", "0:a", "-map", "1:a",
         "-c:a", "aac", "-disposition:a:0", first, "-disposition:a:1", second, str(path)],
        check=True, timeout=20,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_original_timing_replay_reproduces_the_exact_default_second_soundtrack(tmp_path):
    source = tmp_path / "default-second.mov"
    digest = _two_aac(source, default_second=True)
    first = decode_audio(source, source_sha256=digest)
    second = decode_default_second_aac441(source, source_sha256=digest)
    assert second is not None
    selected = binding.selected_audio(second.mapping)
    assert selected is not None and selected["source_stream_position"] == 1
    record = {"schema": binding.SCHEMA_WITH_DEFAULT_AAC441_MOTION, binding.SELECTED_AUDIO: selected, "inputs": {}}

    decoded, replay_source = _decode_lip_soundtrack(source, digest, record, first.mapping)
    assert replay_source == selected
    assert decoded.pcm.tobytes() == second.pcm.tobytes()
    assert selected["pcm_sha256"] != first.mapping["pcm_sha256"]
    assert first.mapping["source_stream_index"] == selected["audio_stream_indices"][0]
    assert selected["source_time_base"] == "1/44100"
    assert selected["source_start_pts"] == 0 and selected["source_duration_pts"] == 92610

    mismatches = {"source_stream_index": first.mapping["source_stream_index"],
                  "pcm_sha256": first.mapping["pcm_sha256"],
                  "source_duration_pts": selected["source_duration_pts"] - 1}
    for key, value in mismatches.items():
        with pytest.raises(ValueError, match="selected default soundtrack"):
            _decode_lip_soundtrack(source, digest, record | {binding.SELECTED_AUDIO: selected | {key: value}}, first.mapping)

    alternative = tmp_path / "default-first.mov"
    alternative_digest = _two_aac(alternative, default_second=False)
    with pytest.raises(ValueError, match="selected default soundtrack"):
        _decode_lip_soundtrack(alternative, alternative_digest, record, first.mapping)

    historical, historical_source = _decode_lip_soundtrack(
        source, digest, {"schema": binding.SCHEMA_WITH_AAC441_MOTION, "inputs": {}}, first.mapping,
    )
    assert historical.pcm.tobytes() == first.pcm.tobytes()
    assert historical_source is first.mapping
