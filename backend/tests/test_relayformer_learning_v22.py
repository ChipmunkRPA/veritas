"""Meaningful loss gradients and exact boundaries for the new learning path."""

import math

import pytest
import torch

from scripts.relayformer_learning_v22 import regional_loss, trainable_name


@pytest.mark.parametrize("path", ["vit.blocks.0.attn.qkv_lora_loc.lora_A", "vit.blocks.11.attn.proj_lora_glo.lora_B", "vit.blocks.7.mlp.lora1_loc.lora_A", "vit.blocks.1.mlp.lora2_glo.lora_B", "mask_decoder.input_proj.weight", "mask_decoder.layers.2.norm_ffn.bias"])
def test_used_adapters_and_decoder_can_learn(path):
    assert trainable_name(path)


@pytest.mark.parametrize("path", ["vit.blocks.0.attn.qkv.weight", "vit.patch_embed.proj.weight", "vit.norm.weight", "vit.blocks.0.mlp.lora1_glo.lora_A", "vit.blocks.0.attn.proj_lora_glo.lora_B", "mask_decoder.final_conv.0.weight", "unrelated.lora_A"])
def test_dense_base_and_unused_paths_stay_frozen(path):
    assert not trainable_name(path)


def test_negative_control_mean_loss_and_each_pixel_gradient_have_analytic_values():
    values = torch.zeros((1, 1, 4, 4), requires_grad=True)
    loss = regional_loss(values, torch.zeros_like(values))
    assert float(loss.detach()) == pytest.approx(math.log(2), abs=1e-7)
    loss.backward()
    assert torch.equal(values.grad, torch.full_like(values, 1 / 32))


def test_balanced_positive_loss_has_analytic_value_and_correct_gradient_directions():
    values = torch.zeros((1, 1, 4, 4), requires_grad=True)
    target = torch.zeros_like(values)
    target[:, :, :2] = 1
    loss = regional_loss(values, target)
    assert float(loss.detach()) == pytest.approx(math.log(2) + 4 / 17, abs=1e-7)
    loss.backward()
    assert bool((values.grad[target.bool()] < 0).all())
    assert bool((values.grad[~target.bool()] > 0).all())


def test_confident_wrong_predictions_still_get_finite_corrective_gradients():
    values = torch.full((1, 1, 4, 4), 1000.0, requires_grad=True)
    loss = regional_loss(values, torch.zeros_like(values))
    assert float(loss.detach()) == 1000
    loss.backward()
    assert torch.equal(values.grad, torch.full_like(values, 1 / 16))


@pytest.mark.parametrize("kind", ["all_positive", "fractional", "nonfinite", "wrong_shape"])
def test_unusable_learning_targets_are_refused(kind):
    values = torch.zeros((1, 1, 4, 4))
    target = torch.zeros_like(values)
    if kind == "all_positive":
        target[:] = 1
    elif kind == "fractional":
        target[0, 0, 0, 0] = 0.5
    elif kind == "nonfinite":
        target[0, 0, 0, 0] = float("nan")
    else:
        target = target[:, :, :2]
    with pytest.raises(ValueError):
        regional_loss(values, target)
