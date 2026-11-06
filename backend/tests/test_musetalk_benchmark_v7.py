import numpy as np
import pytest
import torch

from scripts.musetalk_benchmark_v7 import audio_chunks, image_inputs, positional_encoding


def test_masking_precedes_normalization_and_rgb_channels_are_preserved():
    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[:, :, 0] = 255
    masked, original = image_inputs(rgb)
    assert original.shape == (1, 3, 256, 256)
    assert torch.all(original[:, 0] == 1) and torch.all(original[:, 1:] == -1)
    assert torch.all(masked[:, :, 128:] == -1)
    assert torch.equal(masked[:, :, :128], original[:, :, :128])


def test_exact_audio_frame_clock_and_padding_do_not_wrap_source_ticks():
    hidden = torch.arange(20, dtype=torch.float32)[:, None, None].expand(20, 5, 384).clone()
    result = audio_chunks(hidden, 10).reshape(10, 10, 5, 384)
    assert result.shape == (10, 10, 5, 384)
    assert result[0, :4].count_nonzero() == 0
    assert torch.equal(result[0, 4:, 0, 0], torch.arange(6, dtype=torch.float32))
    assert torch.equal(result[4, :, 0, 0], torch.arange(4, 14, dtype=torch.float32))
    assert torch.equal(result[-1, :6, 0, 0], torch.arange(14, 20, dtype=torch.float32))
    assert result[-1, 6:].count_nonzero() == 0


def test_fixed_positional_encoding_preserves_all_five_encoder_states():
    values = positional_encoding()
    assert values.shape == (1, 50, 384)
    assert torch.all(values[0, 0, 0::2] == 0) and torch.all(values[0, 0, 1::2] == 1)
    assert values[0, 1, 0] == pytest.approx(np.sin(1.0))
    with pytest.raises(ValueError):
        positional_encoding(width=512)
