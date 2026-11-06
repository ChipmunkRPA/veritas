import numpy as np
import pytest

from app.bench.audio_continuity import candidates


def test_level_change_is_not_confused_with_spectral_content_change():
    time = np.arange(64000) / 16000
    pcm = 0.2 * np.sin(2 * np.pi * 300 * time)
    pcm[32000:] *= 0.05
    row = next(r for r in candidates(pcm) if r["boundary_sample"] == 32000)
    assert row["energy_change_db"] > 25
    assert row["spectral_distance"] < 1e-5


def test_level_matched_splice_has_exact_source_interval():
    time = np.arange(32000) / 16000
    pcm = np.concatenate([0.2 * np.sin(2 * np.pi * 300 * time), 0.2 * np.sin(2 * np.pi * 3000 * time)])
    row = next(r for r in candidates(pcm) if r["boundary_sample"] == 32000)
    assert row["start_sample"] == 24000 and row["end_sample"] == 40000
    assert row["boundary_s"] == 2
    assert row["energy_change_db"] < 1e-5
    assert row["stationary_spectral_distance"] > 0.99


def test_quiet_or_too_short_input_abstains_instead_of_inventing_boundaries():
    assert candidates(np.zeros(64000)) == []
    assert candidates(np.ones(7999) * 0.1) == []
    time = np.arange(64000) / 16000
    pcm = 0.2 * np.sin(2 * np.pi * 300 * time)
    rows = candidates(pcm)
    assert rows[0]["boundary_sample"] == 8000
    assert rows[-1]["boundary_sample"] == 56000
    assert all(0 <= r["start_sample"] < r["end_sample"] <= len(pcm) for r in rows)


def test_generated_stationary_tone_is_not_an_origin_detection_test():
    time = np.arange(64000) / 16000
    rows = candidates(0.2 * np.sin(2 * np.pi * 300 * time))
    assert max(r["spectral_distance"] for r in rows) < 1e-5
    assert all("probability" not in r and "ai_generated" not in r for r in rows)


@pytest.mark.parametrize("value", [np.array([np.nan]), np.zeros((2, 8000))])
def test_invalid_inputs_are_refused(value):
    with pytest.raises(ValueError):
        candidates(value)
