"""Review a second view of speech while preserving the primary voice decision."""

from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.detectors.audio import calibration
from app.detectors.audio.content import window_in_speech
from app.detectors.audio.short_voice_review import ACTIVITY_MODEL
from app.detectors.video.adapters.audio import NO_ADMITTED_VOICE, PellaVoiceDetector
from app.detectors.video.adapters.common import video_spec
from app.engine.detector import ActivationUse, Applicability, Detector, DetectorResult, InputUnit, Modality
from app.evidence import audio_background_review_binding as binding

BUDGET_S = 20.0
PIPELINE_RESERVE_S = 25.0
OPERATING_POINT = 7.422796281843017
PRIMARY_ID = PellaVoiceDetector.spec.id
_ENHANCER = threading.BoundedSemaphore()


def primary_direction(entry):
    if entry.get("detector_id") != PRIMARY_ID:
        return None
    if entry.get("status") == "not_applicable" and entry.get("reason") == NO_ADMITTED_VOICE:
        return "unassessed"
    if entry.get("status") != "ok":
        return None
    units = entry.get("unit_scores") or []
    screening = (entry.get("details") or {}).get("audio_screening") or {}
    if (len(units) == 1 and units[0]["quality"].get("start_sample") == 0
            and units[0]["quality"].get("end_sample") == binding.SPAN and screening.get("scope_supported") is True):
        return screening.get("decision")
    return None


def disposition(admitted, response):
    if not admitted:
        return "unassessed"
    return "flagged" if response > OPERATING_POINT else "no_flag"


def skipped(reason):
    return {"status": "skipped", "reason": reason}


def configured():
    model, packages = settings.audio_background_review_model, settings.audio_background_review_packages
    return (settings.environment == "dev" and sys.platform == "darwin" and model is not None and packages is not None
            and model.is_absolute() and packages.is_absolute())


def _enhance(pcm, deadline):
    model, packages = settings.audio_background_review_model, settings.audio_background_review_packages
    if not configured() or not _ENHANCER.acquire(blocking=False):
        raise ValueError("Local speech enhancement is unavailable")
    try:
        worker = Path(__file__).with_name("background_review_worker.py").resolve()
        env = {"PATH": os.defpath, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
        data = np.rint(pcm * 32768.0).astype("<i2").tobytes()
        deadline.check("background-reduced speech review")
        with subprocess.Popen([sys.executable, "-I", str(worker), str(model), str(packages)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env) as proc:
            try:
                while True:
                    deadline.check("background-reduced speech review")
                    try:
                        result, _ = proc.communicate(data, timeout=0.15)
                        break
                    except subprocess.TimeoutExpired:
                        data = None
            finally:
                if proc.poll() is None:
                    proc.kill()
            if proc.returncode != 0 or len(result) != binding.SPAN * 2:
                raise ValueError("Local speech enhancement did not return the expected PCM")
        return result
    finally:
        _ENHANCER.release()


class BackgroundVoiceReview(Detector):
    spec = video_spec(binding.DETECTOR_ID, binding.TITLE, InputUnit.AUDIO_WINDOW, (),
                      family="voice-review", model_ids=(PellaVoiceDetector.MODEL_ID, ACTIVITY_MODEL),
                      needs=("audio",), modality=Modality.AUDIO, devices=("cpu",), timeout_s=BUDGET_S, version="1.0.0")

    def activation_plan(self, ctx):
        return (ActivationUse(PellaVoiceDetector.MODEL_ID, 1, "audio window", "One processed three-second speech window"),
                ActivationUse(ACTIVITY_MODEL, 1, "audio stream", "One processed three-second speech-activity pass"))

    def applicable(self, ctx):
        track = ctx.outputs.audio
        eligible = (track is not None and track.pcm is not None and len(track.pcm) == binding.SPAN
                    and not track.clock_breaks and (ctx.outputs.probe or {}).get("primary_direction") in {"inconclusive", "unassessed"})
        return Applicability(eligible, None if eligible else "No eligible source interval.")

    def run(self, ctx):
        pcm = ctx.outputs.audio.pcm
        processed = _enhance(pcm, ctx.deadline)
        wave = np.frombuffer(processed, dtype="<i2").astype(np.float32) / 32768.0
        vad = ctx.lease.get(ACTIVITY_MODEL).obj
        admitted = window_in_speech(vad.spans(wave), 0.0, 3.0)
        response = None
        if admitted:
            ctx.deadline.check("background-reduced speech review")
            probability, = ctx.lease.get(PellaVoiceDetector.MODEL_ID).obj.predict([wave])
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Invalid speech response")
            response = calibration.logit(probability)
        status = disposition(admitted, response)
        ctx.deadline.check("background-reduced speech review")
        details = {"source_pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
                   "primary_direction": ctx.outputs.probe["primary_direction"], "status": status,
                   "processed_pcm_sha256": hashlib.sha256(processed).hexdigest(),
                   "admitted": admitted, "response_hex": None if response is None else response.hex()}
        if status == "flagged":
            details["processed_pcm"] = base64.b64encode(processed).decode("ascii")
        return DetectorResult(status="ok", details=details)
