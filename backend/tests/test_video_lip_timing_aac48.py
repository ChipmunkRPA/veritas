# SPDX-License-Identifier: MIT
import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from app.detectors.audio.decode import decode_audio
from app.detectors.audio.phone_events import MODEL_BLOBS
from app.detectors.video import viseme_review, viseme_stereo
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


def test_articulation_uses_presented_extent_on_48khz_aac(aac_mapping):
    raw = {"frames": [], "phones": [], "articulation_phones": [["lower_lip", 33_480, 0.9]]}
    result = binding.measured(raw, timestamps=[], time_base="1/24", source={}, mapping=aac_mapping)
    assert result["schema"] == binding.SCHEMA_WITH_ARTICULATION
    assert result["inputs"]["samples"] == 33_600

    raw["articulation_phones"][0][1] = 33_800
    with pytest.raises(ValueError, match="invalid speech event"):
        binding.measured(raw, timestamps=[], time_base="1/24", source={}, mapping=aac_mapping)


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


def test_timing_reports_unavailable_assets_when_the_checkpoint_drive_is_missing(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in MODEL_BLOBS:
        (checkpoint / name).touch()
    (checkpoint / "pytorch_model.bin").symlink_to(tmp_path / "disconnected" / "pytorch_model.bin")
    geometry = tmp_path / "geometry.task"
    geometry.touch()
    monkeypatch.setattr(viseme_review.settings, "video_lip_timing_model_dir", checkpoint)
    monkeypatch.setattr(viseme_review.settings, "video_lip_timing_face_asset", geometry)
    monkeypatch.setattr(viseme_review, "supported", lambda *_: True)

    result = viseme_review.assess(None, SimpleNamespace(table=None), {}, {}, [])

    assert result == binding.skipped("assets")


@pytest.mark.parametrize(("scale", "admitted"), [(1, True), (-1, False)])
def test_stereo_48khz_timing_checks_both_channels_and_the_entire_clock(tmp_path, monkeypatch, scale, admitted):
    path = tmp_path / "two-channels.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=400:sample_rate=48000:duration=2.1", "-af", f"pan=stereo|c0=c0|c1={scale}*c0",
                    "-c:a", "aac", str(path)], check=True, timeout=20)
    decoded = decode_audio(path, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), inspect_stereo=True, retain_stereo=True)
    mapping = decoded.mapping
    scope = viseme_stereo.capture(decoded)
    assert scope["blocks"][-1][1] == 33600
    assert binding.audio_origin(mapping) is None
    assert binding.audio_origin(mapping, scope) == (0 if admitted else None)
    assert binding.audio_origin(mapping, scope | {"blocks": scope["blocks"][:-1]}) is None
    assert binding.audio_origin(mapping | {"pcm_sha256": "9" * 64}, scope) is None
    if admitted:
        record = binding.measured({"frames": [], "phones": []}, timestamps=[], time_base="1/24", source={},
                                  mapping=mapping, stereo_scope=scope)
        assert record["schema"] == binding.SCHEMA_WITH_STEREO and record["stereo_scope"] == scope
        assert record["affects_overall_confidence"] is False
    else:
        monkeypatch.setattr(viseme_review, "use_stereo", lambda: True)
        record = viseme_review.assess(None, None, mapping, {}, [], stereo_scope=scope)
        assert record == binding.skipped("stereo_unresolved")
