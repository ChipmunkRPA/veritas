import numpy as np
import pytest
import torch

from scripts import spatial_conv_cpu_v139 as tiling


def test_large_seven_kernel_workspace_is_partitioned_without_reducing_full_output():
    spec = tiling.plan((1, 142, 16, 64, 64), (7, 7, 7), (1, 1, 1), (3, 3, 3), (1, 1, 1), 4)
    assert spec["output"] == (16, 64, 64)
    assert spec["estimated_full_columns_bytes"] > 10 * 1024**3
    assert spec["estimated_tile_columns_bytes"] <= tiling.WORKSPACE
    assert np.prod(spec["tile"]) < np.prod(spec["output"])


@pytest.mark.parametrize("kind", ["2d7", "3d7", "2d_stride_dilation_groups", "3d_stride_dilation_groups"])
def test_tiles_match_full_cpu_convolution_including_halos_and_repeat(kind):
    torch.set_num_threads(1)
    torch.manual_seed(139)
    if kind == "2d7":
        module, shape = torch.nn.Conv2d(4, 5, 7, padding=3), (1, 4, 13, 15)
    elif kind == "3d7":
        module, shape = torch.nn.Conv3d(2, 3, 7, padding=3), (1, 2, 8, 9, 10)
    elif kind == "2d_stride_dilation_groups":
        module = torch.nn.Conv2d(4, 6, (3, 2), stride=(2, 1), padding=(2, 1), dilation=(2, 1), groups=2)
        shape = (1, 4, 11, 13)
    else:
        module = torch.nn.Conv3d(
            4, 6, (3, 2, 2), stride=(2, 1, 2), padding=(2, 1, 0), dilation=(2, 1, 1), groups=2
        )
        shape = (1, 4, 8, 9, 11)
    module.eval()
    value = torch.randn(shape)
    events = []
    budget = 4096 if kind == "2d_stride_dilation_groups" else 16384
    with torch.inference_mode():
        expected = module(value)
        actual = tiling.tiled_forward(module, value, budget, events.append)
        repeated = tiling.tiled_forward(module, value, budget)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert torch.equal(actual, repeated)
    assert events[0]["estimated_tile_columns_bytes"] <= budget and events[-1]["tiles"] > 1


def test_install_preserves_parameters_and_restores_original_forward():
    module = torch.nn.Conv2d(4, 6, 3, padding=1).eval()
    before = {k: v.clone() for k, v in module.state_dict().items()}
    original = module.forward
    events = []
    patches = tiling.install({"authored": module}, events.append, 4096)
    with torch.inference_mode():
        value = torch.randn((1, 4, 12, 13))
        result = module(value)
        torch.testing.assert_close(result, original(value), rtol=2e-5, atol=2e-5)
    assert events and all(torch.equal(before[k], v) for k, v in module.state_dict().items())
    tiling.restore(patches)
    assert module.forward == original


def test_training_or_precision_changes_are_rejected_instead_of_falling_back():
    module = torch.nn.Conv2d(2, 3, 3, padding=1).eval()
    value = torch.randn((1, 2, 6, 7))
    with pytest.raises(ValueError):
        tiling.tiled_forward(module, value)
    module.train()
    with torch.inference_mode(), pytest.raises(ValueError):
        tiling.tiled_forward(module, value)
    module.eval().double()
    with torch.inference_mode(), pytest.raises(ValueError):
        tiling.tiled_forward(module, value.double())
