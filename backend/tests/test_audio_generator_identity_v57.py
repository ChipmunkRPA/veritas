import pytest
import torch

from scripts.audio_generator_checks_v57 import normalized_keys, state_fingerprint, strict_loaded_state


class AdaIN1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(4, affine=True)


AdaIN1d.__module__ = "kokoro.istftnet"


def fixture():
    model = torch.nn.Sequential(AdaIN1d(), torch.nn.Linear(4, 2)).eval().requires_grad_(False)
    saved = {"0": {}, "1": {"module." + key: value.clone() for key, value in model[1].state_dict().items()}}
    return model, saved


def test_only_exact_export_identity_pairs_supplement_the_complete_checkpoint():
    model, saved = fixture()
    result = strict_loaded_state(model, saved, expected_identity_tensors=2)
    assert result["checkpoint_tensors"] == 2 and result["extra_identity_tensors_verified"] == 2
    model[0].norm.bias[0] = .001
    with pytest.raises(ValueError, match="not exact identity"):
        strict_loaded_state(model, saved, expected_identity_tensors=2)


def test_missing_learned_tensor_and_unexpected_saved_tensor_are_rejected():
    model, saved = fixture()
    saved["1"].pop("module.weight")
    with pytest.raises(ValueError, match="missing"):
        strict_loaded_state(model, saved, expected_identity_tensors=2)
    model, saved = fixture()
    saved["1"]["module.unexpected"] = torch.zeros(1)
    with pytest.raises(ValueError, match="Unexpected saved"):
        strict_loaded_state(model, saved, expected_identity_tensors=2)


def test_affine_identity_normalization_matches_the_non_affine_operation():
    generator = torch.Generator().manual_seed(5701)
    x = torch.randn((2, 4, 31), generator=generator)
    a = torch.nn.InstanceNorm1d(4, affine=True).eval()
    b = torch.nn.InstanceNorm1d(4, affine=False).eval()
    with torch.inference_mode():
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


def test_no_generic_normalization_exception_or_mixed_prefix_is_allowed():
    model, saved = fixture()
    type(model[0]).__module__ = "unqualified_module"
    try:
        with pytest.raises(ValueError, match="pinned upstream"):
            strict_loaded_state(model, saved, expected_identity_tensors=2)
    finally:
        type(model[0]).__module__ = "kokoro.istftnet"
    with pytest.raises(ValueError, match="Mixed"):
        normalized_keys({"module.a": 1, "b": 2})


def test_model_state_fingerprint_detects_any_parameter_change():
    model, _ = fixture()
    before = state_fingerprint(model)
    model[1].weight[0, 0] += .01
    assert state_fingerprint(model) != before
