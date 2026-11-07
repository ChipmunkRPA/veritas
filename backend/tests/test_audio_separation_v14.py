"""Source clock and unsafe/incomplete separator outputs cannot become evidence."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.audio_separation_v14 import GlobalNorm, separate


def test_global_norm_uses_population_variance_and_learned_channel_gain():
    module = GlobalNorm(2)
    with torch.no_grad():
        module.gamma.copy_(torch.tensor([2., 3.]))
        module.beta.copy_(torch.tensor([1., -1.]))
    actual = module(torch.tensor([[[1., 2.], [3., 4.]]]))
    scale = (1.25 + 1e-8)**.5
    expected = torch.tensor([[[1 - 3 / scale, 1 - 1 / scale], [-1 + 1.5 / scale, -1 + 4.5 / scale]]])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("array", [np.zeros(64000), np.zeros(63999, np.float32), np.zeros((1, 64000), np.float32),
                                   np.full(64000, np.nan, np.float32)])
def test_wrong_clock_dtype_or_nonfinite_source_refused_before_forward(array):
    with pytest.raises(ValueError):
        separate(array, SimpleNamespace())


@pytest.mark.parametrize("shape,dtype,finite", [((1, 1, 64000), torch.float32, True),
    ((1, 2, 63999), torch.float32, True), ((1, 2, 64000), torch.float64, True),
    ((1, 2, 64000), torch.float32, False)])
def test_incomplete_or_nonfinite_estimates_cannot_supply_confidence(shape, dtype, finite):
    class Broken:
        def __call__(self, values):
            return torch.full(shape, 0 if finite else float("nan"), dtype=dtype)
    with pytest.raises(ValueError):
        separate(np.zeros(64000, np.float32), Broken())


def test_both_sources_retained_and_caller_cannot_be_mutated():
    pcm = np.full(64000, .25, np.float32)

    class Synthetic:
        def __call__(self, value):
            value.zero_()
            return torch.stack([torch.zeros(64000), torch.ones(64000)], dim=0)[None]

    result = separate(pcm, Synthetic())
    assert np.all(pcm == .25)
    assert result.shape == (2, 64000)
    assert np.all(result[0] == 0) and np.all(result[1] == 1)
