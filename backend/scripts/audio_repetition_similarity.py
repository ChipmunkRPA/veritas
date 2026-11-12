"""Compatibility imports for the frozen waveform comparison scripts."""

from app.evidence.audio_waveform_similarity import (
    MAX_LAGS,
    MAX_SAMPLES,
    MIN_SAMPLES,
    MIN_WINDOW_SIMILARITY,
    SAMPLE_RATE,
    WINDOW,
    SimilarityResult,
    SimilaritySpan,
    find_similar_spans,
)

__all__ = [
    "MAX_LAGS", "MAX_SAMPLES", "MIN_SAMPLES", "MIN_WINDOW_SIMILARITY", "SAMPLE_RATE", "WINDOW",
    "SimilarityResult", "SimilaritySpan", "find_similar_spans",
]
