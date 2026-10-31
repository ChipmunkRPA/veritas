"""v0.3 synthetic-voice classifiers (`video/audio.py`) as v2 plugins on the sound track, on V-7 voice windows.

Each model scores ≤ 4 s windows of 16 kHz mono PCM every `voice_hop_s` seconds of the profile (2 s in full and deep; quick
scores one window, the loudest) and reports one window score per window; the aggregate is the window mean. The temporal
stage turns the window scores into voice-lane segments. The legacy fuser fuses the models with the v0.3 weights and
applies the hypothesis rule (a bona fide voice is weak evidence about the visuals).
"""
from __future__ import annotations

import functools
from typing import ClassVar

import numpy as np

from app.detectors.audio.content import window_in_speech
from app.detectors.audio.coverage import coverage_seconds
from app.detectors.image.adapters.common import revisions
from app.detectors.video.adapters.common import DEEP_ONLY, H2, video_spec
from app.detectors.video.audio import (
    CLASSIFIER_BATCH,
    MAX_VOICE_WINDOWS,
    SR,
    VOICE_WINDOW_S,
    AudioClassifier,
    voice_windows,
)
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    Modality,
    Profile,
    RunContext,
    UnitScore,
)
from app.engine.profiles import SAMPLING

NOT_ANALYSABLE = {"absent": "no audio track", "unreadable": "the sound track could not be decoded", "silent": "silent sound track",
                  "short": "the sound track is shorter than 1 s"}


class _VoiceClassifier(Detector):
    MODEL_ID: ClassVar[str]
    REPO: ClassVar[str]
    POSITIVE: ClassVar[str] = "fake"

    def activation_plan(self, ctx: RunContext) -> tuple[ActivationUse, ...]:
        return (ActivationUse(self.MODEL_ID, CLASSIFIER_BATCH, "audio window",
                              f"At most four padded mono {SR} Hz windows, each at most {VOICE_WINDOW_S:g} seconds"),)

    def applicable(self, ctx: RunContext) -> Applicability:
        audio = ctx.outputs.audio
        if audio is None:
            return Applicability(False, "no audio track")
        if audio.status != "analysed" or audio.pcm is None:
            return Applicability(False, NOT_ANALYSABLE.get(audio.status, "the sound track could not be analysed"))
        # the windows are cut from the sound track alone, so a track without one takes no model (BUILD_SPEC N-14c, E-10)
        return Applicability(True) if self._windows(ctx) else Applicability(False, self._no_window(ctx))

    @staticmethod
    def _windows(ctx: RunContext) -> list[tuple[float, np.ndarray]]:
        params = SAMPLING[Profile(ctx.profile)]
        audio = ctx.outputs.audio
        gate = audio.gate
        return voice_windows(audio.pcm, params.voice_hop_s or VOICE_WINDOW_S,
                             max_windows=params.audio_windows or audio.max_windows or MAX_VOICE_WINDOWS,
                             keep=None if gate is None else functools.partial(window_in_speech, gate))

    @staticmethod
    def _no_window(ctx: RunContext) -> str:
        return ("no window of at least 1 s is at least half speech" if ctx.outputs.audio.gate is not None
                else "the sound track is shorter than 1 s")

    def run(self, ctx: RunContext) -> DetectorResult:
        hop = SAMPLING[Profile(ctx.profile)].voice_hop_s or VOICE_WINDOW_S
        windows = self._windows(ctx)
        if not windows:
            return DetectorResult.not_applicable(self._no_window(ctx))
        feature_extractor, model = ctx.lease.get(self.MODEL_ID).obj
        clf = AudioClassifier.from_parts(feature_extractor, model, self.POSITIVE, ctx.device, repo=self.REPO)
        preds = [float(p) for p in clf.predict([c for _, c in windows])]
        units = [UnitScore(f"window:{k}", p, "probability", quality={"start_s": start, "end_s": start + len(c) / SR})
                 for k, (p, (start, c)) in enumerate(zip(preds, windows, strict=True))]
        mean = float(np.mean(preds))
        seconds = coverage_seconds((start, start + len(chunk) / SR) for start, chunk in windows)
        decoded_seconds = ctx.outputs.audio.samples / SR
        exhibit = ctx.exhibits.series(
            [{"name": "p_synthetic", "points": [(round((u.quality["start_s"] + u.quality["end_s"]) / 2.0, 6), u.raw) for u in units]}],
            x_unit="s", title=f"{self.spec.name} per window", media_ref="audio",
            caption=f"Raw classifier output per ≤ {VOICE_WINDOW_S:g} s window at a {hop:g} s hop, plotted at the window centre; uncalibrated.")
        side = "synthetic" if mean >= 0.5 else "bona fide"
        return DetectorResult(
            status="ok", unit_scores=units,
            aggregate=UnitScore("global", mean, "probability", quality={"windows": float(len(units)), "seconds": seconds,
                                                                     "decoded_seconds": decoded_seconds}),
            observed=(f"mean p(synthetic)={mean:.3f} over {len(units)} window(s) of ≤ {VOICE_WINDOW_S:g} s at a {hop:g} s hop "
                      f"({seconds:.1f} s scored of {decoded_seconds:.1f} s decoded audio)"),
            reference="Anti-spoofing classifier; decision boundary at 0.5 per window",
            reasoning=f"On average the classifier places the voice on the {side} side of its decision boundary.",
            metrics={"mean": mean, "windows": float(len(units)), "seconds": seconds, "decoded_seconds": decoded_seconds,
                     "hop_s": float(hop)}, exhibit_ids=[exhibit],
            model_revisions=revisions(ctx, self.MODEL_ID))


class GustkingVoiceDetector(_VoiceClassifier):
    MODEL_ID, REPO = "hf.gustking.wav2vec2_xlsr_deepfake", "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
    spec = video_spec("aud.gustking_xlsr", "Gustking wav2vec2-XLSR deepfake audio classifier", InputUnit.AUDIO_WINDOW, (H2,), ("series",),
                      family="voice", model_ids=(MODEL_ID,), needs=("audio",), modality=Modality.AUDIO, timeout_s=120.0, version="0.4.0")


class MoTheCreatorVoiceDetector(_VoiceClassifier):
    MODEL_ID, REPO = "hf.mo_thecreator.deepfake_audio_detection", "mo-thecreator/Deepfake-audio-detection"
    spec = video_spec("aud.mo_thecreator_w2v2", "mo-thecreator wav2vec2-base deepfake audio classifier", InputUnit.AUDIO_WINDOW, (H2,),
                      ("series",), family="voice", model_ids=(MODEL_ID,), needs=("audio",), modality=Modality.AUDIO, timeout_s=60.0,
                      profiles=DEEP_ONLY, version="0.4.0")
