"""Offline readout of the fixed voice head at every valid encoder step."""

import math

import numpy as np

BLOCK_FRAMES = 20
MIN_BLOCKS = 2


def frame_logits(model, pcm):
    """Return the ordinary raw response and its frame logit decomposition from one forward."""
    traces = []

    def capture(_backbone, _args, kwargs, output):
        mask = kwargs["attention_mask"]
        if mask.shape[0] != 1:
            raise ValueError("The readout requires exactly one voice window")
        weights = model.layer_weights.softmax(dim=0)
        scores = None
        for weight, states in zip(weights, output.hidden_states, strict=True):
            contribution = weight * model.head(states).squeeze(-1)
            scores = contribution if scores is None else scores + contribution
        traces.append(scores[mask.bool()].detach().cpu().numpy())

    handle = model.backbone.register_forward_hook(capture, with_kwargs=True)
    try:
        (raw,) = model.predict([pcm])
    finally:
        handle.remove()
    (trace,) = traces
    if trace.ndim != 1 or not len(trace) or not np.isfinite(trace).all():
        raise ValueError("Invalid frame logit trace")
    reconstructed = 1.0 / (1.0 + math.exp(-float(np.mean(trace, dtype=np.float64))))
    if abs(reconstructed - raw) > 2e-6:
        raise ValueError("Frame logits do not reproduce the ordinary voice score")
    return float(raw), trace


def repeated_block_score(trace):
    """Second-highest mean across complete, nonoverlapping 20-step blocks."""
    blocks = len(trace) // BLOCK_FRAMES
    if blocks < MIN_BLOCKS:
        return None
    means = trace[: blocks * BLOCK_FRAMES].reshape(blocks, BLOCK_FRAMES).mean(axis=1, dtype=np.float64)
    return float(np.partition(means, -MIN_BLOCKS)[-MIN_BLOCKS])


def fixed_point(observations, feature):
    """Use the maximum feature among assessable, inconclusive genuine fitting inputs."""
    eligible = [
        row[feature]
        for row in observations
        if row["genuine"] and row["decision"] == "inconclusive" and row[feature] is not None
    ]
    if not eligible:
        raise ValueError("No genuine inconclusive examples to set a fixed point")
    return max(eligible)


def supplement(decision, feature, point):
    if decision == "inconclusive" and feature is not None and feature > point:
        return "ai"
    return decision
