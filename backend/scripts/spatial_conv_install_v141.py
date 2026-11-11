"""Transactional inference-only installation over the frozen v139 spatial math."""

from __future__ import annotations

import types
from dataclasses import dataclass

from scripts import spatial_conv_cpu_v139 as arithmetic

WORKSPACE = arithmetic.WORKSPACE
plan = arithmetic.plan
_ABSENT = object()


@dataclass(frozen=True)
class ForwardPatch:
    module: object
    previous: object
    instance_forward: object


def _validate_module(module):
    import torch

    if isinstance(module, torch.nn.Conv2d):
        kind, dimensions = torch.nn.Conv2d, 2
    elif isinstance(module, torch.nn.Conv3d):
        kind, dimensions = torch.nn.Conv3d, 3
    else:
        raise ValueError("Only regular Conv2d/Conv3d modules are qualified")
    if (
        module.training
        or module.padding_mode != "zeros"
        or type(module).forward is not kind.forward
        or type(module)._conv_forward is not kind._conv_forward
    ):
        raise ValueError("Explicit eval standard-forward numeric zero-padded convolution required")
    for name in ("kernel_size", "stride", "padding", "dilation"):
        values = getattr(module, name)
        minimum = 0 if name == "padding" else 1
        if (
            not isinstance(values, tuple)
            or len(values) != dimensions
            or any(type(n) is not int or n < minimum for n in values)
        ):
            raise ValueError("Explicit numeric convolution dimensions required")
    if (
        type(module.groups) is not int
        or module.groups <= 0
        or type(module.in_channels) is not int
        or module.in_channels <= 0
        or type(module.out_channels) is not int
        or module.out_channels <= 0
        or module.in_channels % module.groups
        or module.out_channels % module.groups
    ):
        raise ValueError("Valid fixed convolution channel groups required")
    tensors = [module.weight, *module.parameters(recurse=False)]
    if module.bias is not None:
        tensors.append(module.bias)
    if any(
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.layout != torch.strided
        for value in tensors
    ):
        raise ValueError("Explicit CPU float32 convolution parameters required")
    for value in module.buffers(recurse=False):
        if value.device.type != "cpu" or (value.is_floating_point() and value.dtype != torch.float32):
            raise ValueError("CPU-compatible convolution buffers required")
    expected = (module.out_channels, module.in_channels // module.groups, *module.kernel_size)
    if tuple(module.weight.shape) != expected or (
        module.bias is not None and tuple(module.bias.shape) != (module.out_channels,)
    ):
        raise ValueError("Convolution parameter shapes differ from declared dimensions")
    return dimensions, kind


def _validate_forward(module, value):
    import torch

    dimensions, _ = _validate_module(module)
    if torch.is_grad_enabled() or torch.is_autocast_enabled("cpu"):
        raise ValueError("Explicit inference/no-grad with CPU autocast disabled required")
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.layout != torch.strided
        or value.ndim != dimensions + 2
        or value.shape[0] != 1
        or value.shape[1] != module.in_channels
    ):
        raise ValueError("One corresponding strided CPU float32 convolution input required")


def tiled_forward(module, value, budget=WORKSPACE, observer=None):
    _validate_forward(module, value)
    return arithmetic.tiled_forward(module, value, budget, observer)


def _forward(original, label, observer, budget):
    def forward(module, value):
        # All constraints apply before the fast path as well as before tiling.
        # Existing Module.__call__ pre-hooks have already run, so normalized
        # weights are checked/used without replacing spectral-normalization hooks.
        _validate_forward(module, value)
        spec = plan(
            tuple(value.shape),
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            value.element_size(),
            budget,
        )
        if spec["estimated_full_columns_bytes"] <= budget:
            return original(value)
        observe = None if observer is None else lambda row: observer({"module": label, **row})
        return arithmetic.tiled_forward(module, value, budget, observe)

    return forward


def restore(records):
    # forward is an ordinary Python attribute, not a parameter/buffer. Restoring
    # its exact dictionary entry avoids custom assignment failures and preserves
    # whether lookup originally used the class descriptor or an instance override.
    for record in reversed(records):
        namespace = object.__getattribute__(record.module, "__dict__")
        if record.instance_forward is _ABSENT:
            namespace.pop("forward", None)
        else:
            namespace["forward"] = record.instance_forward


def install(models, observer, budget=WORKSPACE):
    import torch

    if type(budget) is not int or budget <= 0 or (observer is not None and not callable(observer)):
        raise ValueError("Positive fixed workspace budget and optional callable observer required")
    candidates, seen = [], set()
    # Validate the complete graph before changing ANY forward method. Global grad
    # may be enabled during installation; the inference constraint is per forward.
    for root, model in models.items():
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
                raise ValueError("Transposed convolution is outside this qualification")
            if not isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)) or id(module) in seen:
                continue
            _, kind = _validate_module(module)
            previous = module.forward
            if (
                getattr(previous, "__func__", None) is not kind.forward
                or getattr(previous, "__self__", None) is not module
            ):
                raise ValueError("Previously replaced/custom convolution forwards are unsupported")
            seen.add(id(module))
            candidates.append(
                (
                    root + "." + name,
                    ForwardPatch(
                        module, previous, object.__getattribute__(module, "__dict__").get("forward", _ABSENT)
                    ),
                )
            )
    records = []
    try:
        for label, record in candidates:
            records.append(record)  # Include an assignment that mutates and THEN raises.
            record.module.forward = types.MethodType(
                _forward(record.previous, label, observer, budget), record.module
            )
    except BaseException:
        restore(records)
        raise
    return records
