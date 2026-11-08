"""Activation recomputation without changing parameters, spatial input or graph math."""

from collections import Counter
from contextlib import contextmanager

import torch
from torch.utils.checkpoint import checkpoint


def install(model):
    blocks = list(model.vit.blocks)
    if model.training or len(blocks) != 12 or any(block.training for block in blocks):
        raise ValueError("The complete12-block eval-mode backbone is required")
    if hasattr(model, "_chipmunk_checkpoint_trace_v24"):
        raise ValueError("Activation checkpointing is already installed")
    trace = Counter()

    def wrap(original):
        def body(*args, **kwargs):
            trace["body_entries"] += 1
            result = original(*args, **kwargs)
            trace["body_completions"] += 1
            return result

        def forward(*args, **kwargs):
            trace["external_block_calls"] += 1
            if not torch.is_grad_enabled():
                trace["no_grad_bypasses"] += 1
                return original(*args, **kwargs)
            trace["checkpointed_block_calls"] += 1
            return checkpoint(body, *args, use_reentrant=False, preserve_rng_state=True, determinism_check="default", **kwargs)

        return forward

    # Each closure keeps that block's original bound method. No registered module
    # or parameter is replaced, and the checkpoint state_dict keys stay identical.
    for block in blocks:
        block.forward = wrap(block.forward)
    model._chipmunk_checkpoint_trace_v24 = trace
    return trace


@contextmanager
def saved_activation_payload(model):
    """Count packed nonparameter bytes, not physical memory or a true peak."""
    parameter_storage = {p.untyped_storage().data_ptr() for p in model.parameters()}
    measured = {"packed_nonparameter_tensors": 0, "packed_nonparameter_bytes": 0}

    def pack(value):
        if value.untyped_storage().data_ptr() not in parameter_storage:
            measured["packed_nonparameter_tensors"] += 1
            measured["packed_nonparameter_bytes"] += value.numel() * value.element_size()
        return value.detach()

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        yield measured
