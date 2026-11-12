"""Optional source-time review of six-second mono speech."""

from __future__ import annotations

import hashlib

from app.detectors.audio import calibration
from app.detectors.audio.content import window_in_speech
from app.detectors.video.adapters.audio import PellaVoiceDetector
from app.detectors.video.adapters.common import video_spec
from app.engine.detector import ActivationUse, Applicability, Detector, DetectorResult, InputUnit, Modality
from app.evidence.audio_short_review_binding import (
    DETECTOR_ID,
    GRID,
    RATE,
    SAMPLES,
    SCHEMA,
    TITLE,
    selected_range,
)

BUDGET_S = 15.0
PIPELINE_RESERVE_S = 25.0
ACTIVITY_MODEL = "silero.speech_activity"


def skipped(reason: str) -> dict:
    return {"schema": SCHEMA, "status": "skipped", "reason": reason, "responses": [], "selected": None}


def scan(pcm, spans, classifier, deadline, *, codec=None) -> dict:
    intervals = [(a, b) for a, b in GRID if window_in_speech(spans, a / RATE, b / RATE)]
    if not intervals:
        return skipped("No two-second interval contained enough detected speech.")
    responses = []
    for a, b in intervals:
        deadline.check("short voice review")
        probability, = classifier.predict([pcm[a:b]])
        responses.append({"start_sample": a, "end_sample": b, "logit": calibration.logit(probability)})
    deadline.check("short voice review")
    selected = selected_range(responses, codec)
    if selected:
        status, reason = "flagged", "A two-second speech window returned an elevated response for review."
    else:
        status, reason = "no_flag", "No checked two-second speech window was flagged. The recording could still contain an edit."
    return {"schema": SCHEMA, "status": status, "reason": reason, "responses": responses, "selected": selected}


class ShortVoiceReview(Detector):
    """Invoked after the main audio checks, outside the primary detector and fuser."""

    spec = video_spec(DETECTOR_ID, TITLE, InputUnit.AUDIO_WINDOW, (),
                      family="voice-review", model_ids=(PellaVoiceDetector.MODEL_ID, ACTIVITY_MODEL),
                      needs=("audio",), modality=Modality.AUDIO, devices=("cpu",), timeout_s=BUDGET_S, version="1.0.0")

    def activation_plan(self, ctx):
        return (ActivationUse(PellaVoiceDetector.MODEL_ID, 1, "audio window", "One padded two-second speech window"),
                ActivationUse(ACTIVITY_MODEL, 1, "audio stream", "One six-second speech-activity pass"))

    def applicable(self, ctx):
        track = ctx.outputs.audio
        ok = track is not None and track.pcm is not None and len(track.pcm) == SAMPLES and not track.clock_breaks
        return Applicability(ok, None if ok else "No eligible continuous six-second source.")

    def run(self, ctx):
        pcm = ctx.outputs.audio.pcm
        spans = ctx.lease.get(ACTIVITY_MODEL).obj.spans(pcm)
        model = ctx.lease.get(PellaVoiceDetector.MODEL_ID).obj
        codec = (ctx.outputs.probe or {}).get("source_codec")
        review = scan(pcm, spans, model, ctx.deadline, codec=codec)
        return DetectorResult(status="ok", details={
            "short_voice_review": review,
            "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        })
