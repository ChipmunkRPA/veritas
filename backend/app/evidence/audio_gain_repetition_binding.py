"""Standard-library source and signed-clip validation for gain-related repeats."""

from __future__ import annotations

from struct import iter_unpack

from app.evidence import audio_exact_repetition as exact
from app.evidence.audio_repetition_binding import _sample_range, _valid_hash

GAIN_INPUT = "stereo_repetition_gain_version"
GAIN_VERSION = 1
SCHEMA = "chipmunk.audio-gain-repetition/1"
GAINS = ((-1, 1), (-3, 2), (-5, 4), (-3, 4), (-2, 3), (-1, 2), (1, 2), (2, 3), (3, 4), (5, 4), (3, 2))


def validate(context, source, channel, pcm_sha256):
    if (context["schema"] != SCHEMA or context["source_sha256"] != source["sha256"]
            or context["pcm_sha256"] != pcm_sha256 or context["sample_rate"] != exact.SAMPLE_RATE
            or context["input_samples"] != source["samples"]):
        raise ValueError("Gain repeat coverage differs from the recorded channel")
    status, reason, pairs = context["status"], context["reason"], context["pairs"]
    if status != "measured":
        if pairs != [] or not (
            (status == "unavailable" and reason == "insufficient_signal_variation")
            or (status == "ambiguous" and reason in {"anchor_pair_budget", "candidate_budget", "span_budget"})
        ):
            raise ValueError("Unmeasured gain repetition includes unsupported evidence")
        return
    if reason is not None or not isinstance(pairs, list) or len(pairs) > exact.MAX_SPANS:
        raise ValueError("Invalid gain repetition status or pair count")
    order = []
    for index, pair in enumerate(pairs, 1):
        source_pts = source["timeline"][0]["source_pts"]
        first = _sample_range(pair["first"], samples=source["samples"], source_pts=source_pts)
        second = _sample_range(pair["second"], samples=source["samples"], source_pts=source_pts)
        count = first[1] - first[0]
        if (pair["id"] != f"{channel[0].upper()}G{index:02d}" or first[1] > second[0]
                or count != second[1] - second[0] or count < exact.MIN_REPEAT_SAMPLES
                or pair["sample_count"] != count or pair["lag_samples"] != second[0] - first[0]
                or type(pair["gain_numerator"]) is not int or type(pair["gain_denominator"]) is not int
                or (pair["gain_numerator"], pair["gain_denominator"]) not in GAINS
                or pair["scaled_occurrence"] not in {"first", "second"}):
            raise ValueError("Invalid gain repeat source ranges or ratio")
        order.append((-count, first, second))
        assets = pair["artifacts"]
        if set(assets) != {"first_clip", "second_clip", "first_waveform", "second_waveform"}:
            raise ValueError("Gain repetition lacks playback or waveform evidence")
        if assets["first_clip"]["sha256"] == assets["second_clip"]["sha256"]:
            raise ValueError("Gain repetition requires two distinct WAV clips")
        for side in ("first", "second"):
            clip, waveform = assets[f"{side}_clip"], assets[f"{side}_waveform"]
            if (not _valid_hash(pair[side]["pcm_s16le_sha256"]) or not _valid_hash(clip["sha256"])
                    or (clip["media_type"], clip["samples"], clip["sample_rate"], clip["channels"]) != ("audio/wav", count, exact.SAMPLE_RATE, 1)
                    or not _valid_hash(waveform["sha256"])
                    or (waveform["media_type"], waveform["width"], waveform["height"]) != ("image/png", 960, 280)):
                raise ValueError("Invalid gain repeat media commitment")
    if order != sorted(set(order)):
        raise ValueError("Gain repeat pairs are duplicated or out of order")


def matches_clips(pair, first, second):
    if first == second or len(first) != len(second) or len(first) != 2 * pair["sample_count"]:
        return False
    donor, target = (first, second) if pair["scaled_occurrence"] == "second" else (second, first)
    numerator, denominator = pair["gain_numerator"], pair["gain_denominator"]
    return all(round(a * numerator / denominator) == b for (a,), (b,) in zip(
        iter_unpack("<h", donor), iter_unpack("<h", target), strict=True,
    ))
