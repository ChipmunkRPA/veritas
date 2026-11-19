"""Evaluate the frozen 750 ms speech grid on the continuous original samples."""

from __future__ import annotations

from dataclasses import replace

from app.detectors.audio import short_voice_review as short
from app.detectors.audio.extended_short_voice_review import pcm_sha256
from app.engine.detector import ActivationUse, Applicability, DetectorResult
from app.evidence import audio_brief_voice_binding as binding

BUDGET_S = 38.0


def scan(pcm, speech, classifier, deadline) -> list[dict]:
    spans = [(round(a * binding.RATE), round(b * binding.RATE)) for a, b in speech]
    rows = []
    for start in binding.GRID:
        deadline.check("brief speech review")
        end = start + binding.WIDTH
        width = sum(max(0, min(b, end) - max(a, start)) for a, b in spans)
        excerpt = pcm[start:end]
        logit = float(short.voice_logit(classifier, excerpt)).hex() if width * 2 >= binding.WIDTH else None
        rows.append({"start_sample": start, "speech_samples": width, "logit_hex": logit,
                     "pcm_sha256": pcm_sha256(excerpt)})
    deadline.check("brief speech review")
    return rows


class BriefVoiceReview(short.ShortVoiceReview):
    base = short.ShortVoiceReview.spec
    spec = replace(base, id=binding.DETECTOR_ID, name=binding.TITLE, version="1.0.0",
                   requirements=replace(base.requirements, timeout_s=BUDGET_S))

    def activation_plan(self, ctx):
        return (ActivationUse(short.PellaVoiceDetector.MODEL_ID, 1, "audio window", "One padded 750 ms speech window"),
                ActivationUse(short.ACTIVITY_MODEL, 1, "audio stream", "One twelve-second speech-activity pass"))

    def applicable(self, ctx):
        track = ctx.outputs.audio
        ok = track is not None and track.pcm is not None and not track.clock_breaks and len(track.pcm) == binding.SAMPLES
        return Applicability(ok, None if ok else "The brief speech check needs twelve continuous seconds of audio.")

    def run(self, ctx):
        pcm = ctx.outputs.audio.pcm
        speech = ctx.lease.get(short.ACTIVITY_MODEL).obj.spans(pcm)
        classifier = ctx.lease.get(short.PellaVoiceDetector.MODEL_ID).obj
        rows = scan(pcm, speech, classifier, ctx.deadline)
        binding.raw_checks(rows)
        return DetectorResult(status="ok", details={
            binding.KEY: {"schema": binding.SCHEMA, "readings": rows},
            "pcm_sha256": pcm_sha256(pcm),
        })
