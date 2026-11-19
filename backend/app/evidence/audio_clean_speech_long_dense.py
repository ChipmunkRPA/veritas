"""Corroborate a dense fallback between two native twelve-second copies."""

from __future__ import annotations

import numpy as np

from app.evidence import audio_clean_speech, audio_clean_speech_dense, audio_clean_speech_long

BASIS = "dense_long_clean_reference_v1"
SAMPLES = audio_clean_speech_long.SAMPLES
SCOPE = (
    "This check compares two twelve-second recordings on the same native sample clock. "
    "It retains an existing supported interval and can check 960 ms intervals when neighboring windows agree. "
    "Matching sound is needed on both sides. Ordinary edits, level changes and overlapping sound can produce findings. "
    "The finding contributes zero weight to the AI estimate. Speaker identity requires separate evidence."
)


def locate(earlier, submitted):
    """Keep the current result and its admission before the frozen dense retry."""
    if any(not isinstance(pcm, np.ndarray) or pcm.shape != (SAMPLES,) or pcm.dtype != np.dtype("<i2")
           for pcm in (earlier, submitted)):
        raise ValueError("Expected two twelve-second native mono PCM16 recordings on the same sample clock")
    reason, selected, profile = audio_clean_speech_long.locate(earlier, submitted)
    if selected is not None:
        return reason, selected, profile, "retained"
    if reason in {"reference_match_not_supported", "consistent_frequency_balance_change"}:
        return reason, None, profile, None
    values = []
    for start in audio_clean_speech_long.WINDOWS:
        end = start + audio_clean_speech.SAMPLES
        state, result, _, _ = audio_clean_speech_dense.locate_with_match_support(earlier[start:end], submitted[start:end])
        if result is not None:
            result = result | {"samples": [sample + start for sample in result["samples"]]}
        values.append({"start": start, "reason": state, "result": result})
    if (selected := audio_clean_speech_long.corroborate_windows(values)) is not None:
        return "local_change", selected, profile, "dense_fallback"
    return reason, None, profile, None
