"""Optional source-time review of six-second mono speech."""

from __future__ import annotations

import hashlib

from app.detectors.audio import calibration
from app.detectors.video.adapters.audio import PellaVoiceDetector
from app.detectors.video.adapters.common import video_spec
from app.engine.detector import ActivationUse, Applicability, Detector, DetectorResult, InputUnit, Modality
from app.evidence.audio_short_review_binding import (
    AFTER_COVERAGE_SKIP,
    COVERAGE_SCHEMA,
    DETECTOR_ID,
    GRID,
    NEARBY_ID,
    NO_SPEECH_REASON,
    RATE,
    SAMPLES,
    SCHEMA,
    TITLE,
    nearby_windows,
    selected_range,
    speech_counts,
    supported_aac_length,
)

BUDGET_S = 15.0
PIPELINE_RESERVE_S = 25.0
NEARBY_BUDGET_S = 7.0
ACTIVITY_MODEL = "silero.speech_activity"


def skipped(reason: str) -> dict:
    return {"schema": SCHEMA, "status": "skipped", "reason": reason, "responses": [], "selected": None}


def scan(pcm, spans, classifier, deadline, *, codec=None) -> dict:
    counts = speech_counts(spans)
    responses = []
    for (a, b), speech in zip(GRID, counts, strict=True):
        if 2 * speech < b - a:
            continue
        deadline.check("short voice review")
        probability, = classifier.predict([pcm[a:b]])
        responses.append({"start_sample": a, "end_sample": b, "logit": calibration.logit(probability)})
    deadline.check("short voice review")
    selected = selected_range(responses, codec)
    if not responses:
        status, reason = "skipped", NO_SPEECH_REASON
    elif selected:
        status, reason = "flagged", "A two-second speech window returned an elevated response for review."
    else:
        status, reason = "no_flag", "No checked two-second speech window was flagged. The recording could still contain an edit."
    return {"schema": COVERAGE_SCHEMA, "status": status, "reason": reason, "responses": responses,
            "selected": selected, "speech_counts": counts}


class ShortVoiceReview(Detector):
    """Invoked after the main audio checks, outside the primary detector and fuser."""

    spec = video_spec(DETECTOR_ID, TITLE, InputUnit.AUDIO_WINDOW, (),
                      family="voice-review", model_ids=(PellaVoiceDetector.MODEL_ID, ACTIVITY_MODEL),
                      needs=("audio",), modality=Modality.AUDIO, devices=("cpu",), timeout_s=BUDGET_S, version="1.2.0")

    def activation_plan(self, ctx):
        return (ActivationUse(PellaVoiceDetector.MODEL_ID, 1, "audio window", "One padded two-second speech window"),
                ActivationUse(ACTIVITY_MODEL, 1, "audio stream", "One six-second speech-activity pass"))

    def applicable(self, ctx):
        track = ctx.outputs.audio
        if track is None or track.pcm is None or track.clock_breaks:
            return Applicability(False, "No eligible continuous six-second source.")
        codec = (ctx.outputs.probe or {}).get("source_codec")
        ok = supported_aac_length(len(track.pcm)) if codec == "aac" else len(track.pcm) == SAMPLES
        return Applicability(ok, None if ok else "No eligible continuous six-second source.")

    def run(self, ctx):
        pcm = ctx.outputs.audio.pcm
        presentation = pcm[:SAMPLES]
        spans = ctx.lease.get(ACTIVITY_MODEL).obj.spans(presentation)
        model = ctx.lease.get(PellaVoiceDetector.MODEL_ID).obj
        probe = ctx.outputs.probe or {}
        review = scan(presentation, spans, model, ctx.deadline, codec=probe.get("source_codec"))
        if probe.get("primary_coverage_skip") is True:
            review["reason"] = AFTER_COVERAGE_SKIP[review["status"]]
        return DetectorResult(status="ok", details={
            "short_voice_review": review,
            "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        })


class NearbyVoiceReview(ShortVoiceReview):
    """A separately timed comparison of the reported short speech interval."""

    spec = video_spec(NEARBY_ID, "Nearby speech windows", InputUnit.AUDIO_WINDOW, (),
                      family="voice-review", model_ids=(PellaVoiceDetector.MODEL_ID, ACTIVITY_MODEL),
                      needs=("audio",), modality=Modality.AUDIO, devices=("cpu",), timeout_s=NEARBY_BUDGET_S, version="1.0.0")

    def run(self, ctx):
        pcm = ctx.outputs.audio.pcm
        anchor = ctx.outputs.probe["selected"]
        windows = nearby_windows(anchor["start_sample"])
        spans = ctx.lease.get(ACTIVITY_MODEL).obj.spans(pcm)
        model = ctx.lease.get(PellaVoiceDetector.MODEL_ID).obj
        rows = []
        for (start, end), speech in zip(windows, speech_counts(spans, windows), strict=True):
            ctx.deadline.check("nearby speech comparison")
            chunk = pcm[start:end]
            logit = calibration.logit(model.predict([chunk])[0]) if speech >= RATE else None
            rows.append({"start_sample": start, "end_sample": end, "speech_samples": speech, "logit": logit})
        return DetectorResult(status="ok", details={
            "anchor": anchor, "windows": rows, "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        })
