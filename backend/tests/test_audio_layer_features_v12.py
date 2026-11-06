from types import SimpleNamespace

import pytest
import torch

from scripts.audio_layer_features_v12 import LayerCapture, pool_layers


def states():
    return tuple(
        (torch.arange(3, dtype=torch.float32)[None, :, None] + i).expand(1, 3, 1024).clone()
        for i in range(25)
    )


def test_original_attention_mask_excludes_padding_at_preselected_layer_indices():
    values = states()
    for value in values:
        value[:, 2] = 99999
    result = pool_layers(values, torch.tensor([[1, 1, 0]]))
    assert result["layer_ids"] == [8, 16, 24]
    assert [r[0] for r in result["layer_embeddings"]] == [8.5, 16.5, 24.5]
    assert result["feature_frames"] == 3 and result["valid_feature_frames"] == 2
    assert len(result["layer_embedding_sha256"]) == 64


@pytest.mark.parametrize(
    "mask", [torch.tensor([[0, 0, 0]]), torch.tensor([[1.0, 0.5, 0]]), torch.ones(2, 3), torch.ones(3)]
)
def test_unknown_or_empty_attention_clock_is_refused(mask):
    with pytest.raises(ValueError, match="original binary attention mask"):
        pool_layers(states(), mask)


def test_wrong_layer_count_clock_and_nonfinite_values_are_refused():
    values = states()
    with pytest.raises(ValueError, match="Exact25"):
        pool_layers(values[:-1], torch.ones(1, 3))
    with pytest.raises(ValueError, match="clock/width"):
        pool_layers(values, torch.ones(1, 2))
    values[16][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Finite original"):
        pool_layers(values, torch.ones(1, 3))


def test_hook_preserves_original_output_and_tensors_and_removes_only_its_own_handle():
    original = SimpleNamespace(hidden_states=states())
    before = [value.clone() for value in original.hidden_states]

    class Backbone(torch.nn.Module):
        def forward(self, **kwargs):
            return original

    voice = SimpleNamespace(backbone=Backbone())
    observed = []
    other = voice.backbone.register_forward_hook(lambda *args: observed.append(True))
    with LayerCapture(voice) as capture:
        output = voice.backbone(attention_mask=torch.tensor([[1, 1, 0]]))
        assert output is original
        assert capture.take()["layer_embeddings"][0][0] == 8.5
        with pytest.raises(ValueError, match="Exactly one"):
            capture.take()
        assert all(torch.equal(a, b) for a, b in zip(before, original.hidden_states, strict=True))
    assert len(voice.backbone._forward_hooks) == 1
    voice.backbone(attention_mask=torch.ones(1, 3))
    assert len(observed) == 2
    other.remove()
