from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.audio_generator_checks_v54 import (
    checked_phonemes,
    checkpoint_keys,
    strict_loaded_state,
    waveform_summary,
)


def test_checkpoint_prefix_mapping_requires_complete_exact_coverage():
    assert checkpoint_keys({"a": 1}, {"a": 2}) == {"a": "a"}
    assert checkpoint_keys({"module.a": 1}, {"a": 2}) == {"a": "module.a"}
    for saved in ({"module.a": 1, "extra": 2}, {"module.b": 1}, {}):
        with pytest.raises(ValueError):
            checkpoint_keys(saved, {"a": 2})


def test_a_partially_initialized_generator_is_rejected_even_if_constructor_did_not_raise():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2))
    saved = {name: {key: value.clone() for key, value in module.state_dict().items()} for name, module in model.named_children()}
    assert strict_loaded_state(model, saved)["all_tensors_exact"]
    saved["0"]["weight"][0, 0] += 1
    with pytest.raises(ValueError, match="partially loaded"):
        strict_loaded_state(model, saved)


def test_g2p_dropped_words_unknown_symbols_and_truncation_are_not_silent():
    token = SimpleNamespace(text="test", phonemes="tɛst", whitespace=" ")
    vocab = dict.fromkeys("tɛs ")
    assert checked_phonemes("test", [token], vocab) == "tɛst"
    with pytest.raises(ValueError, match="dropped"):
        checked_phonemes("test", [SimpleNamespace(text="test", phonemes=None, whitespace="")], vocab)
    with pytest.raises(ValueError, match="Unknown"):
        checked_phonemes("test", [token], {"t": 0})
    with pytest.raises(ValueError, match="overlong"):
        checked_phonemes("test", [SimpleNamespace(text="test", phonemes="t" * 511, whitespace="")], vocab)


@pytest.mark.parametrize("bad", [np.zeros(20, np.float32), np.array([np.nan], np.float32),
                                 np.ones((2, 4), np.float32), np.ones(20, np.float64)])
def test_silence_invalid_numbers_layout_and_dtype_are_explicit_generation_failures(bad):
    with pytest.raises(ValueError):
        waveform_summary(bad, 24000)


def test_native_clock_and_clipping_are_reported_without_hiding_peaks():
    result = waveform_summary(np.array([0, .1, 1.2, -1.1], np.float32), 24000)
    assert result["samples"] == 4 and result["clipped_samples_if_pcm16"] == 2
    with pytest.raises(ValueError):
        waveform_summary(np.ones(20, np.float32), 16000)
