"""Public source and evidence constraints for the background-reduced speech review."""

from app.evidence.audio_long_review_binding import RATE, SHA, SPAN

KEY = "background_reduced_voice_review"
SCHEMA = "chipmunk.background-reduced-voice-review/1"
DETECTOR_ID = "aud.background_reduced_voice_review"
TITLE = "Speech review with reduced background sound"
LIMITATION = (
    "This review uses the same voice check on a copy with reduced background sound. Any response applies to the full "
    "three-second interval. It cannot establish how the speech was made or locate an exact edit. "
    "This review carries no weight in the overall assessment."
)
REASONS = {
    "flagged": "The speech response leaned AI-generated after background sound was reduced.",
    "no_flag": "The speech response did not lean clearly toward AI generation after background sound was reduced.",
    "unassessed": "The processed copy did not contain enough detected speech for this review.",
}
SKIPPED_DIRECTION = "The primary voice check already gave a direction on this interval."
SKIPPED_UNAVAILABLE = "The optional background-reduced speech review could not complete."
SKIPPED_BUDGET = "The optional background-reduced speech review did not have enough analysis time."
SKIPPED_RECEIPT = "The optional background-reduced speech response could not be saved."
SKIP_REASONS = {SKIPPED_DIRECTION, SKIPPED_UNAVAILABLE, SKIPPED_BUDGET, SKIPPED_RECEIPT}
INTERVAL = {"start_sample": 0, "end_sample": SPAN}
TIMESTAMPS = [{**INTERVAL, "source_start_pts": 0, "source_end_pts": SPAN, "time_base": "1/16000"}]


def supported_source(source):
    return (type(source.get("samples")) is int and source["samples"] == SPAN
            and (source.get("source_codec"), source.get("source_container")) in {("pcm_s16le", "wav"), ("flac", "flac")}
            and source.get("source_bit_depth") == 16
            and source.get("source_channels") == source.get("channels") == 1
            and source.get("source_sample_rate") == source.get("sample_rate") == RATE
            and source.get("source_stream_index") == 0 and source.get("audio_stream_indices", [0]) == [0]
            and source.get("timeline_time_base") == "1/16000"
            and source.get("timeline") == [{**INTERVAL, "source_pts": 0}])


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(review, source):
    try:
        _validate(review, source)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete background-reduced speech evidence") from error


def _validate(review, source):
    assets = review["artifacts"]
    _require(supported_source(source) and review["schema"] == SCHEMA and review["title"] == TITLE
             and review["limitation"] == LIMITATION and review["affects_overall_confidence"] is False
             and review["interval"] == INTERVAL and review["source_timestamp_ranges"] == TIMESTAMPS,
             "Background-reduced speech evidence differs from its native source interval")
    fields = {"schema", "title", "limitation", "affects_overall_confidence", "status", "reason",
              "interval", "source_timestamp_ranges", "artifacts"}
    status, reason = review["status"], review["reason"]
    if status != "flagged":
        valid_reason = reason in SKIP_REASONS if status == "skipped" else status in REASONS and reason == REASONS[status]
        _require(set(review) == fields and assets == {} and valid_reason,
                 "Invalid background-reduced speech disposition")
        return
    _require(set(review) == fields | {"processed_pcm_sha256"} and reason == REASONS["flagged"]
             and SHA.fullmatch(review["processed_pcm_sha256"]) is not None
             and set(assets) == {"selected_clip", "processed_clip", "waveform"},
             "Incomplete flagged background-reduced speech evidence")
    for role in ("selected_clip", "processed_clip"):
        clip = assets[role]
        _require(clip["media_type"] == "audio/wav" and clip["samples"] == SPAN and clip["sample_rate"] == RATE
                 and clip["channels"] == 1 and SHA.fullmatch(clip["sha256"]) is not None,
                 "Invalid background-reduced speech playback asset")
    waveform = assets["waveform"]
    _require(waveform["media_type"] == "image/png" and (waveform["width"], waveform["height"]) == (960, 280)
             and SHA.fullmatch(waveform["sha256"]) is not None, "Invalid background-reduced speech waveform")
