# SPDX-License-Identifier: MIT
import hashlib
import subprocess

import pytest

from app.detectors.audio.decode import decode_audio
from app.evidence import video_viseme_binding as binding


@pytest.fixture
def aac_mapping(tmp_path):
    path = tmp_path / "sound.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=400:sample_rate=48000:duration=2.1", "-c:a", "aac", str(path)],
                   check=True, timeout=20)
    return decode_audio(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()).mapping


def test_48khz_timing_stops_at_the_presented_end(aac_mapping):
    assert aac_mapping["source_duration_pts"] == 100_800
    assert aac_mapping["samples"] > 33_600
    assert binding.audio_origin(aac_mapping) == 0
    assert binding.presented_samples(aac_mapping) == 33_600

    with pytest.raises(ValueError, match="invalid speech event"):
        binding.measured({"frames": [], "phones": [["closure", 33_800, 0.9]]}, timestamps=[],
                         time_base="1/24", source={}, mapping=aac_mapping)


@pytest.mark.parametrize("change", [
    {"source_duration_pts": None},
    {"source_duration_pts": 99_000},
    {"source_duration_pts": 110_000},
    {"source_start_pts": 3},
    {"source_time_base": "1/90000"},
    {"source_channels": 2},
    {"timeline": [{"start_sample": 0, "end_sample": 32000, "source_pts": 0}]},
])
def test_48khz_timing_declines_unsupported_or_incomplete_clocks(aac_mapping, change):
    assert binding.audio_origin(aac_mapping | change) is None
