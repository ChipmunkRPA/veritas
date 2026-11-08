"""Small authored networks verify recomputation, exact updates and parameter identity."""

import copy

import pytest
import torch
from torch import nn

from scripts.relayformer_checkpointing_v24 import install, saved_activation_payload


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(32, 128).requires_grad_(False)
        self.loc = nn.Linear(32, 128, bias=False)
        self.glo = nn.Linear(32, 128, bias=False)
        self.output = nn.Linear(128, 32)

    def forward(self, values, *, use_lora, grid_size):
        branch = self.glo if use_lora == 1 else self.loc
        return values + self.output(torch.nn.functional.gelu((self.base(values) + branch(values)) / grid_size)) / 8


class Authored(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = nn.Module()
        self.vit.blocks = nn.ModuleList([Block() for _ in range(12)])

    def forward(self, values):
        for block in self.vit.blocks:
            values = block(values, use_lora=0, grid_size=2)
            values = block(values, use_lora=1, grid_size=3)
        return values


def step(model, values):
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    with saved_activation_payload(model) as payload:
        output = model(values)
        loss = output.square().mean()
    loss.backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer.step()
    return output.detach(), gradients, {n: p.detach().clone() for n, p in model.named_parameters()}, payload


def test_checkpointed_outputs_all_gradients_and_adam_updates_repeat_exactly():
    torch.set_num_threads(2)
    torch.manual_seed(241)
    original = Authored().eval()
    candidate = copy.deepcopy(original)
    before = {n: (id(p), p.detach().clone()) for n, p in candidate.named_parameters()}
    trace = install(candidate)
    assert set(candidate.state_dict()) == set(original.state_dict())
    assert all(id(p) == before[n][0] and torch.equal(p, before[n][1]) for n, p in candidate.named_parameters())
    # A frozen embedding can supply an input without gradients. The non-reentrant
    # path must still calculate every trainable adapter/decoder gradient.
    values = torch.linspace(-1, 1, 16 * 32).reshape(16, 32)
    assert not values.requires_grad
    normal, recomputed = step(original, values), step(candidate, values)
    assert torch.equal(normal[0], recomputed[0])
    assert normal[1].keys() == recomputed[1].keys()
    assert all(torch.equal(normal[1][name], recomputed[1][name]) for name in normal[1])
    assert all(torch.equal(normal[2][name], recomputed[2][name]) for name in normal[2])
    assert trace["checkpointed_block_calls"] == 24
    assert trace["body_entries"] > trace["checkpointed_block_calls"]
    assert recomputed[3]["packed_nonparameter_bytes"] < normal[3]["packed_nonparameter_bytes"]
    assert all(torch.equal(p, before[n][1]) for n, p in candidate.named_parameters() if not p.requires_grad)


def test_no_grad_inference_keeps_values_exact_and_avoids_checkpointing():
    torch.manual_seed(242)
    original = Authored().eval()
    candidate = copy.deepcopy(original)
    trace = install(candidate)
    with torch.no_grad():
        values = torch.randn(8, 32)
        assert torch.equal(original(values), candidate(values))
    assert trace["no_grad_bypasses"] == 24 and trace["checkpointed_block_calls"] == 0


def test_installation_cannot_be_silently_repeated():
    model = Authored().eval()
    install(model)
    with pytest.raises(ValueError, match="already"):
        install(model)


@pytest.mark.parametrize("kind", ["training", "partial"])
def test_incomplete_or_stochastic_backbone_is_refused(kind):
    model = Authored()
    if kind == "partial":
        model.eval()
        model.vit.blocks = nn.ModuleList(list(model.vit.blocks)[:2])
    with pytest.raises(ValueError, match="complete"):
        install(model)
