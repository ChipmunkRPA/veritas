"""Fixed cached positional values must never be silently selected for training."""

import sys

import pytest
import torch

from scripts.relayformer_learning_v22 import trainable_name as previous
from scripts.relayformer_learning_v23 import trainable_name
from scripts.relayformer_native_v18 import RUNTIME


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_declared_fixed_rotary_frequency_parameters_stay_frozen(layer):
    name = f"mask_decoder.layers.{layer}.self_attn.token_rotary_emb.freqs"
    assert previous(name) and not trainable_name(name)


@pytest.mark.parametrize("name", ["mask_decoder.input_proj.weight", "mask_decoder.layers.0.self_attn.q_proj.weight", "vit.blocks.0.mlp.lora1_loc.lora_A", "vit.blocks.11.attn.qkv_lora_glo.lora_B"])
def test_actual_decoder_and_backbone_adapters_remain_selected(name):
    assert trainable_name(name) == previous(name) is True


def test_actual_fixed_frequency_cache_drops_its_gradient_after_first_call():
    sys.path.insert(0, str(RUNTIME))
    from rotary_embedding_torch import RotaryEmbedding

    rotation = RotaryEmbedding(32, theta=10)
    assert rotation.learned_freq is False and rotation.freqs.requires_grad is False
    rotation.freqs.requires_grad_(True)  # Reproduce the incorrect old selection.
    query = torch.linspace(0, 1, 256).reshape(1, 1, 8, 32).requires_grad_()
    rotation.rotate_queries_or_keys(query).sum().backward()
    assert rotation.freqs.grad is not None
    rotation.freqs.grad = None
    query.grad = None
    rotation.rotate_queries_or_keys(query).sum().backward()
    assert query.grad is not None and rotation.freqs.grad is None
