"""Lazy execution of the frozen original-only sensitivity rule.

This is an execution optimization, not a new detection method: the band view
cannot affect calibration, or an AI/inconclusive identity-guard conclusion.
Only an otherwise original conclusion requires that probe. No labels, source
identities, generator names or SNR annotations are accepted by this interface.
"""
from __future__ import annotations

import functools
import hashlib
from typing import Any

import numpy as np

import audio_calibration_v3_methods as method

EXECUTION_VERSION = "audio-original-veto-lazy/1"


def score_candidate(pcm: np.ndarray, voice_model: Any, speech_activity: Any,
                    calibration: dict | None = None) -> dict:
    """Equivalent to score_views followed by decide, avoiding unused probes.

    ``calibration`` is the full artifact with parameters['new'] and
    guards['new']. Without it, only original-view fitting/calibration scores
    are requested. When supplied, probabilities use the unrounded mean while
    the frozen sensitivity check uses six-decimal aggregate scores.
    """
    from app.detectors.audio.content import window_in_speech
    from app.detectors.video.audio import MAX_VOICE_WINDOWS, VOICE_WINDOW_S, voice_windows
    from app.engine.detector import Profile
    from app.engine.profiles import SAMPLING

    values = np.asarray(pcm, dtype=np.float32)
    if values.ndim != 1 or len(values) < method.SR or not np.isfinite(values).all():
        raise ValueError("At least one second of finite mono 16 kHz PCM is required")
    spans = speech_activity.spans(values)
    params = SAMPLING[Profile("full")]
    windows = voice_windows(values, params.voice_hop_s or VOICE_WINDOW_S,
                            max_windows=params.audio_windows or MAX_VOICE_WINDOWS,
                            keep=functools.partial(window_in_speech, spans))
    intervals = [(round(start * method.SR), round(start * method.SR) + len(chunk))
                 for start, chunk in windows]
    raw_scores = [float(p) for p in voice_model.predict([chunk for _, chunk in windows])] if windows else []
    if len(raw_scores) != len(windows):
        raise ValueError("Original prediction count differs from selected source windows")
    raw = method._aggregate(raw_scores)
    raw_unrounded = float(np.mean(raw_scores)) if raw_scores else None
    p, initial = None, None
    if calibration is not None:
        p = method.probability(raw_unrounded, calibration["parameters"]["new"])
        # Equal-view input is a no-op sensitivity check. The identity guard
        # and probability threshold are therefore exactly the frozen ones.
        initial = method.decide(p, raw, raw, calibration["guards"]["new"])
    needs_probe = initial is not None and initial["identity_guard_decision"] == "original"
    band_scores, band, filtered_hash, probe_error = [], None, None, None
    probe_status = "not_required_for_decision"
    skip_reason = ("Only original-view scores are used for fitting and class-set calibration."
                   if calibration is None else "The identity-guard conclusion is not original.")
    if needs_probe:
        skip_reason = None
        try:
            filtered = method.band_view(values)
            filtered_hash = hashlib.sha256(filtered.astype("<f4").tobytes()).hexdigest()
            band_scores = [float(p) for p in voice_model.predict([filtered[a:b] for a, b in intervals])]
            if len(band_scores) != len(windows):
                raise ValueError("Probe prediction count differs from selected source windows")
            band = method._aggregate(band_scores)
            probe_status = "computed"
        except Exception as error:
            band_scores, band = [], None
            probe_status = "failed"
            probe_error = f"{type(error).__name__}: {error}"
    result = {
        "method": method.METHOD["version"], "execution": EXECUTION_VERSION,
        "device": "cpu", "sample_rate": method.SR,
        "source_pcm_sha256": hashlib.sha256(values.astype("<f4").tobytes()).hexdigest(),
        "band_pcm_sha256": filtered_hash,
        "status": "ok" if windows else "not_applicable", "probe_error": probe_error,
        "probe_status": probe_status, "probe_skip_reason": skip_reason,
        "speech_spans_s": [list(s) for s in spans],
        "raw": raw, "band": band, "raw_unrounded": raw_unrounded,
        "band_unrounded": float(np.mean(band_scores)) if band_scores else None,
        "windows": [{"start_sample": a, "end_sample": b,
                     "raw": raw_scores[i], "band": band_scores[i] if band_scores else None}
                    for i, (a, b) in enumerate(intervals)],
        "sensitivity": method.sensitivity(raw, band),
    }
    if calibration is not None:
        result["decision_detail"] = method.decide(p, raw, band, calibration["guards"]["new"])
    return result
