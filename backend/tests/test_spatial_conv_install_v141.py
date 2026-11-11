import pytest
import torch

from scripts import spatial_conv_install_v141 as installation


@pytest.mark.parametrize("invalid", ["transpose", "padding", "dtype", "custom_forward"])
def test_later_incompatible_module_leaves_earlier_forward_untouched(invalid):
    first = torch.nn.Conv2d(2, 3, 3, padding=1).eval()
    later = torch.nn.Conv2d(3, 4, 3, padding=1).eval()
    if invalid == "transpose":
        later = torch.nn.ConvTranspose2d(3, 4, 3).eval()
    elif invalid == "padding":
        later.padding = "same"
    elif invalid == "dtype":
        later.double()
    else:
        later.forward = lambda value: value
    previous = first.forward
    with pytest.raises(ValueError):
        installation.install({"authored": torch.nn.Sequential(first, later)}, None)
    assert first.forward == previous and "forward" not in first.__dict__


def test_late_assignment_failure_rolls_back_even_the_module_that_mutated_then_raised():
    class RejectAssignmentOnce(torch.nn.Conv2d):
        def __setattr__(self, name, value):
            fail = name == "forward" and self.__dict__.get("reject_assignment_once", False)
            super().__setattr__(name, value)
            if fail:
                super().__setattr__("reject_assignment_once", False)
                raise RuntimeError("authored assignment failure after mutation")

    first = torch.nn.Conv2d(2, 3, 3, padding=1).eval()
    later = RejectAssignmentOnce(3, 4, 3, padding=1).eval()
    later.reject_assignment_once = True
    before = (first.forward, later.forward)
    with pytest.raises(RuntimeError, match="assignment failure"):
        installation.install({"authored": torch.nn.Sequential(first, later)}, None)
    assert (first.forward, later.forward) == before
    assert "forward" not in first.__dict__ and "forward" not in later.__dict__


@pytest.mark.parametrize(
    "problem",
    [
        "grad",
        "training",
        "input_device",
        "weight_device",
        "input_dtype",
        "weight_dtype",
        "padding",
        "autocast",
    ],
)
def test_fast_path_rejects_unsupported_modes_before_calling_original_convolution(problem):
    module = torch.nn.Conv2d(2, 3, 1).eval()
    value = torch.randn((1, 2, 2, 3))
    # Tiny input is below the workspace threshold: every rejection must come
    # from the common guard, rather than the original v139 tiled branch.
    records = installation.install({"authored": module}, None)
    try:
        if problem == "training":
            module.train()
        elif problem == "input_device":
            value = value.to("meta")
        elif problem == "weight_device":
            module.to("meta")
        elif problem == "input_dtype":
            value = value.double()
        elif problem == "weight_dtype":
            module.double()
        elif problem == "padding":
            module.padding = "same"
        if problem == "grad":
            with torch.enable_grad(), pytest.raises(ValueError, match="inference/no-grad"):
                module(value)
        elif problem == "autocast":
            with (
                torch.inference_mode(),
                torch.autocast("cpu", dtype=torch.bfloat16),
                pytest.raises(ValueError, match="autocast"),
            ):
                module(value)
        else:
            with torch.inference_mode(), pytest.raises(ValueError):
                module(value)
    finally:
        installation.restore(records)


@pytest.mark.parametrize("kind", ["2d", "3d"])
def test_bounded_output_matches_normal_convolution_and_restores_exact_lookup(kind):
    torch.set_num_threads(1)
    torch.manual_seed(141)
    if kind == "2d":
        module = torch.nn.Conv2d(
            4, 6, (3, 2), stride=(2, 1), padding=(2, 1), dilation=(2, 1), groups=2
        ).eval()
        value = torch.randn((1, 4, 11, 13))
    else:
        module = torch.nn.Conv3d(
            4, 6, (3, 2, 2), stride=(2, 1, 2), padding=(2, 1, 0), dilation=(2, 1, 1), groups=2
        ).eval()
        value = torch.randn((1, 4, 8, 9, 11))
    previous = module.forward
    parameters = {name: value.clone() for name, value in module.state_dict().items()}
    with torch.inference_mode():
        expected = module(value)
    events = []
    # Installation itself is metadata-only and must allow ordinary global grad.
    with torch.enable_grad():
        records = installation.install({"authored": module}, events.append, 4096)
    try:
        with torch.inference_mode():
            actual = module(value)
            repeated = module(value)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        assert torch.equal(actual, repeated)
        assert events and all(
            r["estimated_tile_columns_bytes"] <= 4096 for r in events if r["event"] == "spatial_conv_begin"
        )
        assert all(torch.equal(parameters[name], value) for name, value in module.state_dict().items())
    finally:
        installation.restore(records)
    assert module.forward == previous and "forward" not in module.__dict__


def test_spectral_normalization_hooks_parameters_and_buffers_are_preserved():
    torch.manual_seed(141)
    module = torch.nn.utils.spectral_norm(torch.nn.Conv2d(4, 6, 3, padding=1)).eval()
    value = torch.randn((1, 4, 12, 13))
    calls = []
    observer = module.register_forward_pre_hook(lambda *args: calls.append("existing_pre_hook"))
    hooks = dict(module._forward_pre_hooks)
    with torch.inference_mode():
        expected = module(value)
    state = {name: value.clone() for name, value in module.state_dict().items()}
    parameter_ids = [id(parameter) for parameter in module.parameters()]
    records = installation.install({"authored": module}, None, 4096)
    try:
        with torch.inference_mode():
            actual = module(value)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        assert calls == ["existing_pre_hook", "existing_pre_hook"]
        assert dict(module._forward_pre_hooks) == hooks
        assert [id(parameter) for parameter in module.parameters()] == parameter_ids
        assert all(torch.equal(state[name], value) for name, value in module.state_dict().items())
    finally:
        installation.restore(records)
        observer.remove()


def test_shared_module_is_patched_once_and_existing_instance_lookup_is_restored():
    module = torch.nn.Conv2d(2, 3, 1).eval()
    module.forward = module.forward
    previous = module.__dict__["forward"]
    records = installation.install({"first": module, "shared": module}, None)
    assert len(records) == 1
    installation.restore(records)
    assert module.__dict__["forward"] is previous
