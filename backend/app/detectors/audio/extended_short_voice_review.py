"""Run the calibrated two-second speech check in overlapping six-second excerpts."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from app.detectors.audio import short_voice_review
from app.engine.detector import Applicability, DetectorResult
from app.evidence import audio_two_second_binding as binding
from app.evidence.audio_short_review_binding import SAMPLES

BUDGET_S = 35.0
BASE_SPEC = short_voice_review.ShortVoiceReview.spec


def pcm_sha256(pcm):
    return hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()


def scan(pcm, activity, classifier, deadline):
    readings = []
    for start in binding.excerpt_starts(len(pcm)):
        deadline.check("two-second speech scan")
        excerpt = pcm[start:start + SAMPLES]
        row = short_voice_review.scan(excerpt, activity.spans(excerpt), classifier, deadline)
        readings.append({"start_sample": start, "pcm_sha256": pcm_sha256(excerpt),
                         "speech_samples": row["speech_counts"], "responses": row["responses"]})
    return readings


class ExtendedShortVoiceReview(short_voice_review.ShortVoiceReview):
    spec = replace(BASE_SPEC, id=binding.DETECTOR_ID, name=binding.TITLE, version="1.0.0",
                   requirements=replace(BASE_SPEC.requirements, timeout_s=BUDGET_S))

    def __init__(self, *, schema=binding.SCHEMA):
        self.schema = schema
        if schema == binding.CONFIRMED_SCHEMA:
            self.spec = replace(self.spec, version="1.1.0")

    def applicable(self, ctx):
        track = ctx.outputs.audio
        ok = track is not None and track.pcm is not None and not track.clock_breaks and bool(binding.excerpt_starts(len(track.pcm)))
        return Applicability(ok, None if ok else "The two-second scan needs a continuous recording from nine to fourteen seconds.")

    def run(self, ctx):
        pcm = ctx.outputs.audio.pcm
        activity = ctx.lease.get(short_voice_review.ACTIVITY_MODEL).obj
        voice = ctx.lease.get(short_voice_review.PellaVoiceDetector.MODEL_ID).obj
        readings = scan(pcm, activity, voice, ctx.deadline)
        return DetectorResult(status="ok", details={
            binding.KEY: {"schema": self.schema, "excerpts": readings},
            "pcm_sha256": pcm_sha256(pcm),
        })
