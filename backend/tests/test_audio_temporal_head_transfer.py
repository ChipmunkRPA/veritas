import numpy as np
import torch

from scripts.audio_temporal_head_transfer import fixed_point, frame_logits, repeated_block_score, supplement


class ToyBackbone(torch.nn.Module):
    def forward(self, *, attention_mask):
        steps = torch.tensor([[[1.0, 4.0], [2.0, 3.0], [9.0, 0.0], [100.0, 100.0]]])
        return type("Features", (), {"hidden_states": (steps, 2 * steps)})()


class ToyVoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ToyBackbone()
        self.layer_weights = torch.nn.Parameter(torch.tensor([-0.25, 0.75]))
        self.head = torch.nn.Linear(2, 1)
        self.head.weight.data[:] = torch.tensor([[0.20, -0.10]])
        self.head.bias.data[:] = 0.25
        self.forwards = 0

    @torch.inference_mode()
    def predict(self, _chunks):
        self.forwards += 1
        valid = torch.tensor([[True, True, True, False]])
        features = self.backbone(attention_mask=valid)
        weights = torch.softmax(self.layer_weights, dim=0)
        pooled = (weights[0] * features.hidden_states[0] + weights[1] * features.hidden_states[1])[
            valid
        ].mean(0)
        return [float(self.head(pooled).sigmoid().item())]


def test_frame_logits_mask_padding_and_reconstruct_one_ordinary_forward():
    voice = ToyVoice()
    raw, trace = frame_logits(voice, np.zeros(16, np.float32))
    assert voice.forwards == 1 and len(trace) == 3
    assert np.isclose(raw, 1 / (1 + np.exp(-trace.mean())))
    assert not voice.backbone._forward_hooks


def test_repeated_peak_requires_two_complete_blocks_and_does_not_change_existing_directions():
    assert repeated_block_score(np.repeat([1, 4, 2, 15], [20, 20, 20, 19])) == 2
    assert repeated_block_score(np.ones(39)) is None
    observations = [
        {"genuine": True, "decision": "inconclusive", "block": 2},
        {"genuine": True, "decision": "ai", "block": 7},
        {"genuine": False, "decision": "inconclusive", "block": 6},
    ]
    point = fixed_point(observations, "block")
    assert point == 2
    assert supplement("inconclusive", 2, point) == "inconclusive"
    assert supplement("inconclusive", 3, point) == "ai"
    assert supplement("original", 3, point) == "original"
    assert supplement("unassessed", None, point) == "unassessed"
