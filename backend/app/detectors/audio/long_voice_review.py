"""Optional, non-fusing review of disjoint source-bound speech windows."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from app.detectors.video.adapters.audio import PellaVoiceDetector
from app.detectors.video.adapters.common import AudioTrack, video_spec
from app.engine.detector import Applicability, Detector, DetectorResult, InputUnit, Modality
from app.evidence import audio_long_review_binding as binding

BUDGET_S = 20.0
PIPELINE_RESERVE_S = 25.0


def skipped(reason: str) -> dict:
    return {"schema": binding.SCHEMA, "status": "skipped", "reason": reason, "windows": []}


class _UnpublishedSeries:
    def series(self, *args, **kwargs):
        return "long-voice-review"


class LongVoiceReview(Detector):
    spec = video_spec(binding.DETECTOR_ID, binding.TITLE, InputUnit.AUDIO_WINDOW, (), family="voice-review",
                      model_ids=PellaVoiceDetector.spec.requirements.model_ids, needs=("audio",),
                      modality=Modality.AUDIO, devices=("cpu",), timeout_s=BUDGET_S, version="1.0.0")

    def activation_plan(self, ctx):
        return PellaVoiceDetector().activation_plan(ctx)

    def applicable(self, ctx):
        track = ctx.outputs.audio
        ok = track is not None and track.pcm is not None and not track.clock_breaks and bool(binding.intervals(len(track.pcm)))
        return Applicability(ok, None if ok else "No eligible continuous long speech source.")

    def run(self, ctx):
        source = ctx.outputs.audio.pcm
        checker = PellaVoiceDetector()
        windows = []
        for start, end in binding.intervals(len(source)):
            ctx.deadline.check("long voice review")
            pcm = source[start:end]
            context = replace(ctx, outputs=replace(ctx.outputs, audio=AudioTrack("analysed", pcm, len(pcm))),
                              exhibits=_UnpublishedSeries())
            result = checker.run(context)
            if result.status == "not_applicable":
                direction = "unassessed"
            else:
                if result.status != "ok":
                    raise ValueError("Unavailable scoped voice response")
                screening = result.details["audio_screening"]
                if (not screening["scope_supported"] or len(result.unit_scores) != 1
                        or screening["decision"] not in binding.DIRECTIONS - {"unassessed"}):
                    raise ValueError("Unexpected scoped voice response")
                direction = screening["decision"]
            windows.append({"start_sample": start, "end_sample": end, "direction": direction,
                            "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()})
        ctx.deadline.check("long voice review")
        status = binding.status(windows)
        return DetectorResult(status="ok", details={
            "long_voice_review": {"schema": binding.SCHEMA, "status": status, "reason": binding.REASONS[status], "windows": windows},
            "pcm_sha256": hashlib.sha256(source.astype("<f4").tobytes()).hexdigest(),
        })
