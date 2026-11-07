"""Strict research loading, deterministic positions, and recurrence arithmetic."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from scripts.video_demamba_v13 import bind_keys, restore_constant_positions, serial_scan


def test_prefix_mapping_preserves_tensor_identity_and_all_expected_names():
    tensor = object()
    source = {"module.encoder.vision_model.weight": tensor, "module.head": tensor}
    bound, names = bind_keys(source, {"encoder.weight", "head"})
    assert bound == {"encoder.weight": tensor, "head": tensor}
    assert bound["encoder.weight"] is tensor
    assert names == [("module.encoder.vision_model.weight", "encoder.weight"), ("module.head", "head")]


@pytest.mark.parametrize("state,expected", [
    ({"head": 1}, {"head"}),
    ({"module.head": 1}, {"head", "missing"}),
    ({"module.extra": 1}, {"head"}),
    ({"module.encoder.x": 1, "module.encoder.vision_model.x": 2}, {"encoder.x"}),
])
def test_missing_unknown_or_colliding_tensors_are_refused(state, expected):
    with pytest.raises(ValueError):
        bind_keys(state, expected)


def test_direct_recurrence_has_hand_computed_outputs():
    a = torch.full((1, 4, 1, 1), .5, dtype=torch.float64)
    x = torch.ones_like(a)
    assert serial_scan(a, x).flatten().tolist() == [1., 1.5, 1.75, 1.875]
    assert torch.equal(a, torch.full_like(a, .5)) and torch.equal(x, torch.ones_like(x))
    with pytest.raises(ValueError):
        serial_scan(a[:, :0], x[:, :0])


def fixture():
    model = nn.Module()
    model.encoder = nn.Module()
    model.encoder.embeddings = nn.Module()
    model.encoder.embeddings.num_positions = 197
    model.encoder.embeddings.register_buffer("position_ids", torch.empty((1, 197), dtype=torch.int64, device="meta"), persistent=False)
    return model


def test_meta_constant_is_initialized_even_though_it_is_absent_from_state_dict():
    model = fixture()
    assert model.state_dict() == {}
    assert model.encoder.embeddings.position_ids.is_meta
    receipt = restore_constant_positions(model)
    assert receipt["shape"] == [1, 197]
    assert torch.equal(model.encoder.embeddings.position_ids, torch.arange(197).expand(1, -1))
    assert not any(buffer.is_meta for buffer in model.buffers())


def test_other_uninitialized_state_cannot_be_hidden_by_constant_initialization():
    model = fixture()
    model.register_buffer("unknown", torch.empty(1, device="meta"), persistent=False)
    with pytest.raises(ValueError, match="Uninitialized"):
        restore_constant_positions(model)
    corrupt = SimpleNamespace(encoder=SimpleNamespace(embeddings=SimpleNamespace(num_positions=197, position_ids=torch.zeros((1, 197), dtype=torch.int64))))
    with pytest.raises(ValueError, match="silently repaired"):
        restore_constant_positions(corrupt)
