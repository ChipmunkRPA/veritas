"""Inference-only spatial tiling of large CPU convolutions; same full outputs."""

from __future__ import annotations

import itertools
import math
import types

WORKSPACE = 128 * 1024**2


def plan(shape, kernel, stride, padding, dilation, itemsize, budget=WORKSPACE):
    dimensions = len(shape) - 2
    if (
        dimensions not in (2, 3)
        or shape[0] != 1
        or itemsize != 4
        or any(len(v) != dimensions for v in (kernel, stride, padding, dilation))
        or any(type(n) is not int or n <= 0 for n in (*shape, *kernel, *stride, *dilation))
        or any(type(n) is not int or n < 0 for n in padding)
        or type(budget) is not int
        or budget <= 0
    ):
        raise ValueError("Bounded single-input float32 convolution dimensions required")
    effective = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel, strict=True))
    output = tuple(
        (n + 2 * p - k) // s + 1 for n, p, k, s in zip(shape[2:], padding, effective, stride, strict=True)
    )
    if min(output) <= 0:
        raise ValueError("Positive convolution output required")
    per_cell = shape[1] * math.prod(kernel) * itemsize
    cells = budget // per_cell
    if cells < 1:
        raise ValueError("Workspace cannot hold even one output cell")
    tile = [0] * dimensions
    for axis in reversed(range(dimensions)):
        tile[axis] = min(output[axis], cells)
        cells = max(1, cells // tile[axis])
    return {
        "output": output,
        "tile": tuple(tile),
        "effective_kernel": effective,
        "estimated_full_columns_bytes": per_cell * math.prod(output),
        "estimated_tile_columns_bytes": per_cell * math.prod(tile),
    }


def tiled_forward(module, value, budget=WORKSPACE, observer=None):
    import torch
    import torch.nn.functional as functional

    if (
        torch.is_grad_enabled()
        or module.training
        or module.padding_mode != "zeros"
        or value.device.type != "cpu"
        or module.weight.device.type != "cpu"
        or value.dtype != torch.float32
        or module.weight.dtype != torch.float32
        or not isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d))
    ):
        raise ValueError("Explicit eval/inference CPU float32 zero-padded convolution required")
    shape = tuple(value.shape)
    spec = plan(
        shape,
        module.kernel_size,
        module.stride,
        module.padding,
        module.dilation,
        value.element_size(),
        budget,
    )
    if observer:
        observer({"event": "spatial_conv_begin", "input_shape": shape, **spec})
    # Padding is applied ONCE to the complete input. Normalization/nonlinear
    # layers remain outside this function and see the complete assembled output.
    pad = tuple(p for p in reversed(module.padding) for _ in range(2))
    padded = functional.pad(value, pad)
    result = torch.empty((1, module.out_channels, *spec["output"]), dtype=value.dtype, device=value.device)
    operation = functional.conv2d if value.ndim == 4 else functional.conv3d
    tiles = 0
    for starts in itertools.product(
        *(range(0, n, t) for n, t in zip(spec["output"], spec["tile"], strict=True))
    ):
        stops = tuple(min(a + t, n) for a, t, n in zip(starts, spec["tile"], spec["output"], strict=True))
        input_slices = tuple(
            slice(a * s, (b - 1) * s + k)
            for a, b, s, k in zip(starts, stops, module.stride, spec["effective_kernel"], strict=True)
        )
        output_slices = tuple(slice(a, b) for a, b in zip(starts, stops, strict=True))
        block = operation(
            padded[(slice(None), slice(None), *input_slices)],
            module.weight,
            module.bias,
            stride=module.stride,
            padding=0,
            dilation=module.dilation,
            groups=module.groups,
        )
        if block.shape[2:] != tuple(b - a for a, b in zip(starts, stops, strict=True)):
            raise ValueError("Spatial tile output shape differs")
        result[(slice(None), slice(None), *output_slices)] = block
        tiles += 1
    if observer:
        observer({"event": "spatial_conv_end", "tiles": tiles, "output_shape": tuple(result.shape)})
    return result


def install(models, observer, budget=WORKSPACE):
    import torch

    restored = []
    for root, model in models.items():
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                raise ValueError("Transposed convolution is outside this qualification")
            if not isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                continue
            if module.training or module.padding_mode != "zeros" or isinstance(module.padding, str):
                raise ValueError("Only explicit eval numeric zero-padded convolutions are qualified")
            previous = module.forward

            def make_forward(original, label):
                def forward(current, value):
                    spec = plan(
                        tuple(value.shape),
                        current.kernel_size,
                        current.stride,
                        current.padding,
                        current.dilation,
                        value.element_size(),
                        budget,
                    )
                    if spec["estimated_full_columns_bytes"] <= budget:
                        return original(value)
                    return tiled_forward(
                        current, value, budget, lambda row: observer({"module": label, **row})
                    )

                return forward

            module.forward = types.MethodType(make_forward(previous, root + "." + name), module)
            restored.append((module, previous))
    return restored


def restore(records):
    for module, previous in records:
        module.forward = previous
