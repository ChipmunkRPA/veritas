import numpy as np
import pytest

from scripts.audio_source_qualification_v72 import eligibility, prefix_hash


def test_short_speech_is_not_padded_or_repeated_into_scope():
    pcm = np.ones(40000, np.float32)
    assert eligibility(pcm, 64000, "full", "prefix", "raw", set(), set(), set()) == "shorter_than_frozen_minimum"


def test_historical_reuse_is_explicit_for_music_only_when_caller_allows_it():
    pcm = np.ones(64000, np.float32)
    args = (pcm, 64000, "full", "prefix", "raw", {"pcm:prefix"}, set(), set())
    assert eligibility(*args) == "recorded_historical_bytes_or_pcm"
    assert eligibility(*args, allow_history=True) is None


def test_selected_duplicate_is_rejected_even_if_historical_music_is_allowed():
    pcm = np.ones(64000, np.float32)
    assert eligibility(pcm, 64000, "full", "prefix", "raw", set(), {"raw"}, set(), allow_history=True) == "duplicate_selected_bytes_or_pcm"


def test_nonfinite_or_silent_input_cannot_qualify():
    pcm = np.zeros(64000, np.float32)
    assert eligibility(pcm, 64000, "full", "prefix", "raw", set(), set(), set()) == "silent_fixed_prefix"
    pcm[0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        eligibility(pcm, 64000, "full", "prefix", "raw", set(), set(), set())


def test_source_prefix_hash_uses_exact_float32_samples():
    pcm = np.arange(64010, dtype=np.float32)
    assert prefix_hash(pcm, 64000) == prefix_hash(pcm[:64000], 64000)
    changed = pcm.copy()
    changed[63999] += 1
    assert prefix_hash(changed, 64000) != prefix_hash(pcm, 64000)
