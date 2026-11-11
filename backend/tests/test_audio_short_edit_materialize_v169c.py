from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.audio.decode import read_wav
from scripts import audio_short_edit_materialize_v169c as stage


def audio(pcm, raw="raw", timeline=None):
    return SimpleNamespace(
        pcm=pcm,
        mapping={
            "source_sha256": raw,
            "samples": len(pcm),
            "sample_rate": 16000,
            "sample_format": "s16le",
            "channels": 1,
            "pcm_sha256": stage.pcm_digest(pcm),
            "timeline_time_base": "1/16000",
            "timeline": timeline or [{"start_sample": 0, "end_sample": len(pcm), "source_pts": -12}],
        },
    )


def test_three_second_prefix_keeps_exact_original_source_origin():
    pcm = np.full(stage.SOURCE_SAMPLES + 12, 0.25, dtype=np.float32)
    result, reason = stage.qualification(audio(pcm), "raw", set())
    prefix, mapping, commitments, identities = result
    assert reason is None and np.array_equal(prefix, pcm[:48_000])
    assert mapping["source_start_pts"] == -12 and mapping["source_end_pts"] == 47_988
    assert commitments["grid_length_prefix_pcm_sha256"] == stage.pcm_digest(pcm[:47_648])
    assert identities == {
        "bytes:raw",
        *(f"pcm:{value}" for name, value in commitments.items() if name != "original_sha256"),
    }


def test_short_pcm_is_ineligible_without_padding():
    pcm = np.full(stage.SOURCE_SAMPLES - 1, 0.25, dtype=np.float32)
    assert stage.qualification(audio(pcm), "raw", set()) == (None, "shorter_than_exact_three_second_prefix")


def test_grid_length_prefix_overlap_excludes_longer_clip():
    pcm = np.linspace(0.1, 0.5, stage.SOURCE_SAMPLES, dtype=np.float32)
    key = "pcm:" + stage.pcm_digest(pcm[: stage.GRID_PREFIX])
    assert stage.qualification(audio(pcm), "raw", {key}) == (
        None,
        "recorded_original_or_pcm_identity_overlap",
    )


def test_original_clock_discontinuity_stays_ineligible():
    pcm = np.full(stage.SOURCE_SAMPLES, 0.25, dtype=np.float32)
    blocks = [
        {"start_sample": 0, "end_sample": 24_000, "source_pts": 0},
        {"start_sample": 24_000, "end_sample": 48_000, "source_pts": 24_004},
    ]
    assert stage.qualification(audio(pcm, timeline=blocks), "raw", set()) == (
        None,
        "discontinuous_original_prefix_clock",
    )


def test_materialize_verifies_zero_rebase_and_rejects_second_role_duplicate(tmp_path, monkeypatch):
    pcm = np.full(stage.SOURCE_SAMPLES, 0.25, dtype=np.float32)
    temp = tmp_path / "private.audio"
    target = tmp_path / "development.wav"
    raw = b"authored source fixture"

    def decode(path, *, source_sha256, max_seconds=600):
        assert stage.sha(path) == source_sha256
        if path == temp:
            assert max_seconds == 600
            return audio(pcm, source_sha256)
        assert max_seconds == 3
        samples = read_wav(path.read_bytes())
        return audio(
            samples, source_sha256, [{"start_sample": 0, "end_sample": len(samples), "source_pts": 0}]
        )

    monkeypatch.setattr(stage, "decode_audio", decode)
    used = set()
    result, reason = stage.materialize(raw, temp, target, used)
    assert reason is None and result["source_range"]["source_start_pts"] == -12
    assert result["wav_range"]["source_start_pts"] == 0 and result["wav_sha256"] == stage.sha(target)
    assert not temp.exists()
    other = tmp_path / "holdout.wav"
    assert stage.materialize(raw, temp, other, used) == (None, "recorded_original_or_pcm_identity_overlap")
    assert not other.exists()


def test_corrupt_production_mapping_fails_before_materialization():
    pcm = np.full(stage.SOURCE_SAMPLES, 0.25, dtype=np.float32)
    decoded = audio(pcm)
    decoded.mapping["channels"] = 2
    with pytest.raises(ValueError, match="mapping or sample commitment differs"):
        stage.qualification(decoded, "raw", set())
