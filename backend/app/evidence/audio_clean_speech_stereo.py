"""Corroborate a twelve-second clean-copy comparison with the stereo channels."""

from __future__ import annotations

CHANNELS = ("left", "right")
CORRELATION_MARGIN = .04
OTHER_CHANNEL_SUPPORT = .15


def overlap(a, b):
    return 2 * (min(a[1], b[1]) - max(a[0], b[0])) >= max(a[1] - a[0], b[1] - b[0])


def select(readings, support):
    left, right = (readings[side]["range"] for side in CHANNELS)
    agreement = None
    if left and right and overlap(left, right):
        if left == right == readings["downmix"]["range"]:
            agreement = "downmix"
        else:
            agreement = "left" if left[0] <= right[0] else "right"
    best = None
    for side, other in (CHANNELS, CHANNELS[::-1]):
        sections = zip(support[side], support[other], strict=True)
        if all(part["admitted"] and part["upper_correlation"] - (opposite["upper_correlation"] or 0.) >= CORRELATION_MARGIN
               for part, opposite in sections):
            best = side
            break
    return {"agreement": agreement, "reference_selected": best or agreement}


def corroborated_view(support, choices, *, point=OTHER_CHANNEL_SUPPORT):
    preferred = choices["reference_selected"]
    if preferred in CHANNELS:
        other = "right" if preferred == "left" else "left"
        if any((part["upper_correlation"] or 0.) < point for part in support[other]):
            return choices["agreement"]
    return preferred


def supporting_channels(readings, support, choices):
    downmix = readings["downmix"]["range"]
    preferred = corroborated_view(support, choices)
    channel = readings[preferred]["range"] if preferred else None
    if downmix and channel and overlap(downmix, channel):
        return list(CHANNELS) if preferred == "downmix" else [preferred]
    return []


def observe(earlier, views):
    import numpy as np

    from app.evidence import audio_clean_speech, audio_clean_speech_long

    readings, observations, support = {}, {}, {}
    half = audio_clean_speech.SAMPLES
    weight = np.hanning(audio_clean_speech.WIDTH)
    for name, pcm in views.items():
        reason, span, profile = audio_clean_speech_long.locate(earlier, pcm)
        observations[name] = reason, span, profile
        readings[name] = {"reason": reason, "range": span["samples"] if span else None}
        if name in CHANNELS:
            support[name] = [audio_clean_speech.match_support(earlier[start:start + half], pcm[start:start + half], weight)
                             for start in (0, half)]
    return observations, readings, support


def locate(earlier, downmix, stereo):
    observations, readings, support = observe(earlier, {"downmix": downmix, "left": stereo[:, 0], "right": stereo[:, 1]})
    choices = select(readings, support)
    witnesses = supporting_channels(readings, support, choices)
    proof = {"views": readings, "witness_channels": witnesses}
    reason, span, profile = observations["downmix"]
    if witnesses:
        return reason, span, profile, proof
    if any(view["range"] for view in readings.values()):
        return "stereo_range_not_corroborated", None, None, proof
    return reason, None, profile, proof
