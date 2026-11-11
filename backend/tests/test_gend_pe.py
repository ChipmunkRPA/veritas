import numpy as np
import pytest
import timm
import torch
from safetensors.torch import save_file

from app.detectors.image import gend_pe


class TinyEncoder(torch.nn.Module):
    num_features = 3
    pretrained_cfg = {"input_size": (3, 336, 336), "mean": (0.5,) * 3, "std": (0.5,) * 3, "crop_pct": 1.0}

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, pixels):
        return self.head(self.conv(pixels).mean(dim=(2, 3)))


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    def construct(name, *, pretrained, dynamic_img_size):
        assert name == gend_pe.BACKBONE and not pretrained and dynamic_img_size
        return TinyEncoder()

    monkeypatch.setattr(timm, "create_model", construct)
    state = {
        "feature_extractor.backbone.conv.weight": torch.eye(3).reshape(3, 3, 1, 1),
        "model.linear.weight": torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "model.linear.bias": torch.tensor([0.25, -0.25]),
    }
    return tmp_path / "model.safetensors", state


def test_checkpoint_loads_pooling_and_normalized_classifier_without_a_projection(checkpoint):
    path, state = checkpoint
    save_file(state, path)
    model = gend_pe.load_model(path, "cpu")
    crop = np.broadcast_to(np.array([255, 255, 0], dtype=np.uint8), (144, 110, 3)).copy()
    batch = model.to_batch([crop])
    assert batch.shape == (1, 3, 224, 224)
    assert not model.training and isinstance(model.backbone.head, torch.nn.Identity)
    expected = torch.tensor([[1 / np.sqrt(3) + 0.25, 2 / np.sqrt(3) - 0.25]], dtype=torch.float32)
    torch.testing.assert_close(model(batch), expected)


def test_unexpected_or_missing_checkpoint_tensors_are_rejected(checkpoint):
    path, state = checkpoint
    save_file({**state, "foreign.weight": torch.ones(1)}, path)
    with pytest.raises(ValueError, match="Unexpected GenD PE tensor"):
        gend_pe.load_model(path, "cpu")
    del state["model.linear.bias"]
    save_file(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        gend_pe.load_model(path, "cpu")
