"""Standard-library validation of exact decoded-audio repetition evidence."""

from __future__ import annotations

import re
from typing import Any

from app.evidence import audio_exact_repetition as exact

SCHEMA = "chipmunk.audio-exact-repetition/1"
METHOD = {
    "min_input_samples": exact.MIN_SAMPLES,
    "max_input_samples": exact.MAX_SAMPLES,
    "min_repeat_samples": exact.MIN_REPEAT_SAMPLES,
    "max_pairs": exact.MAX_SPANS,
    "max_anchor_pair_checks": exact.MAX_ANCHOR_PAIR_CHECKS,
    "max_sample_comparisons": exact.MAX_SAMPLE_COMPARISONS,
}
_HASH = re.compile(r"[a-f0-9]{64}")
_AMBIGUOUS = {
    "anchor_pair_budget",
    "sample_comparison_budget",
    "span_budget",
}


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(_HASH.fullmatch(value))


def _timeline(source: dict[str, Any], samples: int) -> list[dict[str, int]]:
    timeline = source["timeline"]
    if not isinstance(timeline, list) or source["timeline_time_base"] != "1/16000":
        raise ValueError("Invalid repetition source clock")
    cursor = 0
    for block in timeline:
        start, end, pts = (block[key] for key in ("start_sample", "end_sample", "source_pts"))
        if any(type(value) is not int for value in (start, end, pts)) or not start == cursor < end <= samples:
            raise ValueError("Repetition source clock does not partition decoded audio")
        cursor = end
    if cursor != samples:
        raise ValueError("Repetition source clock omits decoded audio")
    return timeline


def _sample_range(span: dict[str, Any], *, samples: int, source_pts: int) -> tuple[int, int]:
    start, end = span["start_sample"], span["end_sample"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= samples:
        raise ValueError("Repetition interval is outside decoded audio")
    expected = [{"start_sample": start, "end_sample": end, "source_start_pts": source_pts + start,
                 "source_end_pts": source_pts + end, "time_base": "1/16000"}]
    if span["source_timestamp_ranges"] != expected:
        raise ValueError("Repetition intervals differ from the recorded source timestamps")
    return start, end


def validate(context: dict[str, Any], source: dict[str, Any]) -> None:
    """Validate the bounded search scope, exact source spans and shared WAV identity."""
    try:
        _validate(context, source)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError("Repetition evidence contains missing fields or invalid types") from error


def _validate(context: dict[str, Any], source: dict[str, Any]) -> None:
    samples = source["samples"]
    if (type(samples) is not int or not 0 < samples <= 600 * exact.SAMPLE_RATE
            or source["sample_rate"] != exact.SAMPLE_RATE
            or not all(_valid_hash(source[key]) for key in ("sha256", "pcm_sha256"))
            or context["schema"] != SCHEMA or context["method"] != METHOD
            or context["source_sha256"] != source["sha256"] or context["pcm_sha256"] != source["pcm_sha256"]
            or context["sample_rate"] != exact.SAMPLE_RATE or context["input_samples"] != samples
            or context["evidence_type"] != "context" or context["affects_overall_confidence"] is not False):
        raise ValueError("Repetition evidence differs from its source audio or fixed method")
    clock = _timeline(source, samples)
    status, reason, pairs = (context[key] for key in ("status", "reason", "pairs"))
    eligible = exact.MIN_SAMPLES <= samples <= exact.MAX_SAMPLES
    if not eligible or len(clock) != 1:
        expected = "source_clock_discontinuous" if eligible else "duration_out_of_scope"
        if (status, reason, pairs, context["searched_interval"]) != ("unavailable", expected, [], None):
            raise ValueError("Repetition evidence claims an unsupported search")
        return
    if context["searched_interval"] != {"start_sample": 0, "end_sample": samples}:
        raise ValueError("Repetition search range differs from the supported decoded audio")
    if status != "measured":
        if pairs != [] or not ((status == "unavailable" and reason == "insufficient_signal_variation")
                               or (status == "ambiguous" and reason in _AMBIGUOUS)):
            raise ValueError("Unmeasured repetition evidence contains an unsupported claim")
        return
    if reason is not None or not isinstance(pairs, list) or len(pairs) > exact.MAX_SPANS:
        raise ValueError("Invalid repeated-audio pair count or status")

    ordering = []
    for index, pair in enumerate(pairs, 1):
        first = _sample_range(pair["first"], samples=samples, source_pts=clock[0]["source_pts"])
        second = _sample_range(pair["second"], samples=samples, source_pts=clock[0]["source_pts"])
        count = first[1] - first[0]
        if (pair["id"] != f"R{index:02d}" or first[1] > second[0] or count != second[1] - second[0]
                or count < exact.MIN_REPEAT_SAMPLES or pair["sample_count"] != count
                or pair["lag_samples"] != second[0] - first[0] or not _valid_hash(pair["pcm_s16le_sha256"])):
            raise ValueError("Repeated-audio pair has invalid source ranges")
        ordering.append((-count, first, second))
        clips = pair["artifacts"]
        if set(clips) != {"first_clip", "second_clip"}:
            raise ValueError("Repeated-audio pair lacks its two playback clips")
        first_clip, second_clip = clips["first_clip"], clips["second_clip"]
        if first_clip != second_clip or not _valid_hash(first_clip["sha256"]) or (
            first_clip["media_type"], first_clip["samples"], first_clip["sample_rate"], first_clip["channels"]
        ) != ("audio/wav", count, exact.SAMPLE_RATE, 1):
            raise ValueError("Exact repetition clips must commit the same bounded WAV")
    if ordering != sorted(set(ordering)):
        raise ValueError("Repeated-audio pairs are duplicated or out of order")
