"""Portable source and asset validation for similar repeated audio."""
from __future__ import annotations

from app.evidence.audio_exact_repetition import MAX_SAMPLES, MIN_REPEAT_SAMPLES, MIN_SAMPLES, SAMPLE_RATE
from app.evidence.audio_repetition_binding import _sample_range, _timeline, _valid_hash

SCHEMA = "chipmunk.audio-repeat-similarity/1"
WAVEFORM_SCHEMA = "chipmunk.audio-repeat-similarity/2"
INTERPRETATION = (
    "The ranges contain similar decoded audio. Loops, recurring sounds, and ordinary editing can produce this pattern. "
    "The source and direction of the repetition remain undetermined. This observation carries zero AI confidence weight."
)
BANDS = ((180, 900), (900, 2500), (2500, 6000))
FILTER_ORDER = 4
WINDOW = 640
HOP = 160
MIN_BAND_RMS = 16
MIN_CORRELATION = 0.65
MIN_SUPPORTING_BANDS = 2
MIN_LAG_RMS = 8
MIN_LAG_PEAK = 0.035
LAGS_PER_BAND = 8
LAG_MERGE_SAMPLES = 4
MAX_PAIRS = 8
METHOD = {
    "min_input_samples": MIN_SAMPLES,
    "max_input_samples": MAX_SAMPLES,
    "bands_hz": [list(band) for band in BANDS],
    "filter": "forward_reverse_butterworth",
    "filter_order": FILTER_ORDER,
    "lags_per_band": LAGS_PER_BAND,
    "lag_merge_samples": LAG_MERGE_SAMPLES,
    "min_lag_rms": MIN_LAG_RMS,
    "min_lag_peak": MIN_LAG_PEAK,
    "window_samples": WINDOW,
    "hop_samples": HOP,
    "min_band_rms": MIN_BAND_RMS,
    "min_correlation": MIN_CORRELATION,
    "min_supporting_bands": MIN_SUPPORTING_BANDS,
    "min_repeat_samples": MIN_REPEAT_SAMPLES,
    "max_pairs": MAX_PAIRS,
}
WAVEFORM_METHOD = {**METHOD, "fallback": "preemphasized_waveform"}
_BAND_UNMEASURED = {("unavailable", "insufficient_active_bands"), ("ambiguous", "span_budget")}
_WAVEFORM_UNMEASURED = {
    ("unavailable", "insufficient_signal_variation"),
    ("ambiguous", "periodic_signal"),
    ("ambiguous", "span_budget"),
    ("ambiguous", "competing_lags"),
}


def validate(context: dict, source: dict, repetition: dict | None = None) -> None:
    """Check source bounds and independent audio asset commitments without DSP."""
    try:
        _validate(context, source, repetition)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Repeated-audio similarity evidence contains invalid fields") from error


def _validate(context: dict, source: dict, repetition: dict | None) -> None:
    samples = source["samples"]
    schema = context["schema"]
    waveform = schema == WAVEFORM_SCHEMA
    if (type(samples) is not int or not 0 < samples <= 600 * SAMPLE_RATE or source["sample_rate"] != SAMPLE_RATE
            or not all(_valid_hash(source[key]) for key in ("sha256", "pcm_sha256"))
            or schema not in {SCHEMA, WAVEFORM_SCHEMA} or context["method"] != (WAVEFORM_METHOD if waveform else METHOD)
            or context["source_sha256"] != source["sha256"] or context["pcm_sha256"] != source["pcm_sha256"]
            or context["sample_rate"] != SAMPLE_RATE or context["input_samples"] != samples
            or context["evidence_type"] != "context" or context["affects_overall_confidence"] is not False
            or context["interpretation"] != INTERPRETATION):
        raise ValueError("Repeated-audio similarity differs from its source or fixed method")

    clock = _timeline(source, samples)
    status, reason, pairs = (context[key] for key in ("status", "reason", "pairs"))
    eligible = MIN_SAMPLES <= samples <= MAX_SAMPLES
    if not eligible or len(clock) != 1:
        expected = "source_clock_discontinuous" if eligible else "duration_out_of_scope"
        if waveform or (status, reason, pairs, context["searched_interval"]) != ("unavailable", expected, [], None):
            raise ValueError("Repeated-audio similarity claims an unsupported search")
        return
    if waveform:
        primary = context["frequency_band_search"]
        if (repetition is None or repetition["pairs"] != []
                or any(repetition[key] != source[key] for key in ("pcm_sha256", "sample_rate"))
                or repetition["source_sha256"] != source["sha256"]
                or set(primary) != {"status", "reason"}
                or (primary["status"], primary["reason"]) not in _BAND_UNMEASURED | {("measured", None)}):
            raise ValueError("Waveform fallback requires two earlier searches without a reported pair")
    if context["searched_interval"] != {"start_sample": 0, "end_sample": samples}:
        raise ValueError("Repeated-audio similarity search range differs from its PCM")
    if status != "measured":
        if pairs != [] or (status, reason) not in (_WAVEFORM_UNMEASURED if waveform else _BAND_UNMEASURED):
            raise ValueError("Unmeasured repeated-audio similarity contains an unsupported claim")
        return
    if reason is not None or not isinstance(pairs, list) or len(pairs) > MAX_PAIRS:
        raise ValueError("Invalid repeated-audio similarity pair count or status")

    seen = set()
    for index, pair in enumerate(pairs, 1):
        first = _sample_range(pair["first"], samples=samples, source_pts=clock[0]["source_pts"])
        second = _sample_range(pair["second"], samples=samples, source_pts=clock[0]["source_pts"])
        count = first[1] - first[0]
        windows = pair["support_windows"]
        if (pair["id"] != f"RS{index:02d}" or first[0] % HOP or first[1] > second[0]
                or (first, second) in seen or count != second[1] - second[0]
                or count < MIN_REPEAT_SAMPLES or pair["sample_count"] != count or type(windows) is not int
                or windows < 1 or WINDOW + (windows - 1) * HOP != count
                or pair["lag_samples"] != second[0] - first[0]):
            raise ValueError("Repeated-audio similarity pair has invalid ranges or support")
        seen.add((first, second))

        clips = pair["artifacts"]
        if set(clips) != {"first_clip", "second_clip"}:
            raise ValueError("Repeated-audio similarity lacks both playback clips")
        for interval in ("first", "second"):
            clip = clips[f"{interval}_clip"]
            if (not _valid_hash(pair[interval]["pcm_s16le_sha256"]) or not _valid_hash(clip["sha256"])
                    or (clip["media_type"], clip["samples"], clip["sample_rate"], clip["channels"])
                    != ("audio/wav", count, SAMPLE_RATE, 1)):
                raise ValueError("Repeated-audio similarity clip differs from its source interval")
        identical_samples = pair["first"]["pcm_s16le_sha256"] == pair["second"]["pcm_s16le_sha256"]
        if identical_samples != (clips["first_clip"]["sha256"] == clips["second_clip"]["sha256"]):
            raise ValueError("Repeated-audio similarity playback clips contradict their PCM commitments")
