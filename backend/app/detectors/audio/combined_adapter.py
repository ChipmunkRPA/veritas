"""Optional engine adapter for the separately calibrated speech combination.

Registration and selection remain explicit. Existing audio checks stay installed.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from app.detectors.audio import combined
from app.detectors.video.adapters.common import H2, video_spec
from app.engine.detector import (
    ActivationUse,
    Applicability,
    Detector,
    DetectorResult,
    InputUnit,
    Modality,
    UnitScore,
)

DETECTOR_ID = "aud.combined_speech_v1"
ENCODER_ID = "microsoft.wavlm_base_plus"
CALIBRATION_ID = "chipmunk.audio_global_v1"
BASELINE_ID, ACTIVITY_ID = "pella.voice_v2", "silero.speech_activity"
MODEL_IDS = (BASELINE_ID, ACTIVITY_ID, ENCODER_ID, CALIBRATION_ID)
BUNDLE_SHA256 = "e7de202c42b80fc81d8ab7faa0ce3538c6af5a4c28a9830e4b9dfbcaa6b63641"
VERSION = "combined-speech/1"


def load_encoder(model_id, device, dtype):
    from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file, _revision

    combined.require(device == "cpu" and dtype == "float32", "Qualified CPU float32 speech encoder required")
    entry = _entry(model_id)
    return LoadedModel(combined.load_encoder(_file(entry, "config.json").parent), _revision(entry))


def load_calibration(model_id, device, dtype):
    from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file

    entry = _entry(model_id)
    return LoadedModel(
        combined.load_calibration(_file(entry, "serving-bundle.json"), BUNDLE_SHA256), BUNDLE_SHA256
    )


def calibration_descriptor():
    """A report can bind the private calibration without distributing its arrays."""
    return json.dumps(
        {
            "schema": "chipmunk.speech-calibration-reference/1",
            "version": VERSION,
            "sha256": BUNDLE_SHA256,
            "reference_prior_ai": 0.5,
            "scope": "One complete four-second English speech reference window, 16 kHz mono",
            "limitations": combined.NOTE,
        },
        sort_keys=True,
    ).encode()


class CalibrationReference:
    """Private inference bundle with a separate, shareable report descriptor."""

    SHA256 = BUNDLE_SHA256
    VERSION = VERSION
    NOTE = combined.NOTE
    DETECTOR_ID = DETECTOR_ID
    SLICE_ID = "audio/combined-speech-v1"
    BENCH_VERSION = "audio-v84-held-role-confirmation"
    METHOD = "monotone-logistic-and-platt"
    EXTRA_CAVEAT = "The combined estimate has limited four-second English reference support; it is not a whole-recording authenticity guarantee."

    @property
    def PATH(self):
        from app.detectors.image.adapters.loaders import _entry, _file

        return _file(_entry(CALIBRATION_ID), "serving-bundle.json")

    def load(self):
        from app.detectors.image.adapters.loaders import _entry
        from app.engine.fetchers import ensure_verified

        ensure_verified(_entry(CALIBRATION_ID))
        return combined.load_calibration(self.PATH, self.SHA256)

    def report_payload(self):
        return calibration_descriptor()


class CombinedSpeechDetector(Detector):
    spec = video_spec(
        DETECTOR_ID,
        "Combined speech inspection",
        InputUnit.AUDIO_WINDOW,
        (H2,),
        ("series",),
        family="voice",
        model_ids=MODEL_IDS,
        needs=("audio",),
        modality=Modality.AUDIO,
        devices=("cpu",),
        timeout_s=300.0,
        version="1.0.0",
    )

    def activation_plan(self, ctx):
        return (
            ActivationUse(BASELINE_ID, 1, "audio window", "One source-bound four-second voice window"),
            ActivationUse(ACTIVITY_ID, 1, "audio stream", "Speech activity on the complete source window"),
            ActivationUse(ENCODER_ID, 1, "audio window", "One original four-second speech representation"),
            ActivationUse(CALIBRATION_ID, 1, "score pair", "Fixed inference parameters; no fitting"),
        )

    def applicable(self, ctx):
        audio = ctx.outputs.audio
        if audio is None or audio.status != "analysed" or not combined.source_supported(audio.pcm):
            return Applicability(
                False,
                "The combined speech reference requires exactly four seconds of decoded finite 16 kHz mono audio.",
            )
        mapping = getattr(audio, "source_mapping", None) or {}
        timeline = mapping.get("timeline") or []
        if (
            mapping.get("sample_rate") != 16000
            or mapping.get("samples") != 64000
            or mapping.get("pcm_sha256") != hashlib.sha256(np.asarray(audio.pcm, "<f4").tobytes()).hexdigest()
            or mapping.get("timeline_time_base") != "1/16000"
            or type(mapping.get("limit_samples")) is not int
            or mapping["limit_samples"] <= 64000
            or len(timeline) != 1
            or timeline[0].get("start_sample") != 0
            or timeline[0].get("end_sample") != 64000
            or type(timeline[0].get("source_pts")) is not int
        ):
            return Applicability(
                False,
                "A complete continuous source sample mapping is required; gaps or truncated input cannot supply a combined confidence estimate.",
            )
        return Applicability(True)

    def run(self, ctx):
        from app.detectors.audio.decode import source_ranges
        from app.detectors.image.adapters.common import revisions

        applicable = self.applicable(ctx)
        if not applicable.ok:
            return DetectorResult.not_applicable(applicable.reason)

        pcm = ctx.outputs.audio.pcm
        calibration = ctx.lease.get(CALIBRATION_ID).obj
        runtime = combined.CombinedSpeechRuntime(calibration, ctx.lease.get(ENCODER_ID).obj)
        screening = runtime.analyze(
            pcm, ctx.lease.get(BASELINE_ID).obj, ctx.lease.get(ACTIVITY_ID).obj, ctx.deadline
        )
        if screening["status"] != "ok":
            return DetectorResult.not_applicable(screening["reason"])
        screening = {
            **screening,
            "checked_source_ranges": source_ranges(ctx.outputs.audio.source_mapping, 0, 64000),
        }
        p = screening["probability_ai"]
        quality = {"start_s": 0.0, "end_s": 4.0, "start_sample": 0.0, "end_sample": 64000.0}
        exhibit = ctx.exhibits.series(
            [{"name": "speech_reference_estimate_ai", "points": [(2.0, p)]}],
            x_unit="s",
            title="Speech screening estimate",
            media_ref="audio",
            caption="One complete source window from 0–4 seconds. Balanced-reference estimate, not a measured accuracy rate or proof about background sounds.",
        )
        return DetectorResult(
            status="ok",
            unit_scores=[UnitScore("window:0", p, "probability", quality=quality)],
            aggregate=UnitScore(
                "global", p, "probability", quality={"windows": 1.0, "seconds": 4.0, "decoded_seconds": 4.0}
            ),
            observed=f"Speech screening estimate {p:.3f} towards AI for the complete 0–4 second source window.",
            reference="Separately calibrated combination under a balanced AI/original reference prior.",
            reasoning=screening["reason"] + " " + screening["limitation"],
            metrics={"windows": 1.0, "seconds": 4.0, "decoded_seconds": 4.0, "hop_s": 2.0},
            exhibit_ids=[exhibit],
            model_revisions=revisions(ctx, *MODEL_IDS),
            details={
                "audio_screening": screening,
                "audio_input": {
                    "version": "audio-input/1",
                    "sample_rate": combined.SR,
                    "samples": len(pcm),
                    "pcm_sha256": hashlib.sha256(np.asarray(pcm, "<f4").tobytes()).hexdigest(),
                },
            },
        )
