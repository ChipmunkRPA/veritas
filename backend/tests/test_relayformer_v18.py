"""Source coverage, padding, integer readout and missing-weight refusals."""

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.inspect_relayformer_v18 import check_state
from scripts.relayformer_native_v18 import prepare, source_map


@pytest.mark.parametrize("size", [(515, 389), (1401, 773), (1024, 1024), (31, 77), (773, 1401)])
def test_every_source_is_preserved_without_silent_crop(size):
    tensor, geometry = prepare(Image.new("RGB", size, (73, 155, 212)))
    w, h = size
    fw, fh = geometry["fit_size"]
    assert tensor.shape == (1, 3, 1024, 1024) and tensor.dtype == torch.float32
    assert 0 < fw <= 1024 and 0 < fh <= 1024
    assert geometry["source_size"] == [w, h] and geometry["source_cropped"] is False
    if max(size) <= 1024:
        assert (fw, fh) == size
    else:
        assert max(fw, fh) == 1024
    canvas = torch.zeros((1, 1, 1024, 1024))
    assert source_map(canvas, geometry).shape == (h, w)


def test_padded_positive_responses_cannot_be_source_evidence():
    _, geometry = prepare(Image.new("RGB", (515, 389)))
    canvas = torch.ones((1, 1, 1024, 1024))
    fw, fh = geometry["fit_size"]
    canvas[:, :, :fh, :fw] = 0.2
    mapped = source_map(canvas, geometry)
    assert np.all(mapped == np.float32(0.2))


def test_integer_readout_preserves_first_and_last_source_centers():
    _, geometry = prepare(Image.new("RGB", (1401, 773)))
    fw, fh = geometry["fit_size"]
    values = torch.arange(1024 * 1024, dtype=torch.float32).reshape(1, 1, 1024, 1024) / (1024 * 1024)
    mapped = source_map(values, geometry)
    assert mapped[0, 0] == float(values[0, 0, 0, 0])
    assert mapped[-1, -1] == float(values[0, 0, fh - 1, fw - 1])


def test_unhandled_orientation_cannot_be_assessed():
    image = Image.new("RGB", (37, 91))
    image.getexif()[274] = 6
    with pytest.raises(ValueError, match="upright"):
        prepare(image)


@pytest.mark.parametrize("fault", ["empty", "missing_decoder", "nan", "non_tensor", "zero_tensor"])
def test_incomplete_or_invalid_state_cannot_substitute_initialized_weights(fault):
    state = {"vit.a": torch.ones(3), "mask_decoder.a": torch.ones(2)}
    if fault == "empty":
        state = {}
    elif fault == "missing_decoder":
        state.pop("mask_decoder.a")
    elif fault == "nan":
        state["vit.a"][0] = float("nan")
    elif fault == "non_tensor":
        state["vit.a"] = None
    else:
        state["vit.a"] = torch.empty(0)
    with pytest.raises(ValueError):
        check_state(state)
