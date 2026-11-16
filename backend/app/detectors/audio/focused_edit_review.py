"""Optional CPU review of short speech windows in one six-second mono recording."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import torch
from torch import nn

from app.detectors.audio.content import window_in_speech
from app.detectors.audio.short_voice_review import ACTIVITY_MODEL, BUDGET_S
from app.detectors.video.adapters.common import video_spec
from app.engine.detector import ActivationUse, Applicability, Detector, DetectorResult, InputUnit, Modality
from app.evidence import audio_focused_edit_binding as binding

MODEL_ID = "local.focused_speech_edit_v1"
HEAD_FILE = "joint-speech-head.safetensors"
CENTERED_MODEL_ID = "local.centered_speech_edit_v1"
CENTERED_HEAD_FILE = "centered-speech-head.safetensors"
ENCODER_FILE = "pytorch_model.bin"
PARAMETERS = 94381936
HOP = 160
FRAMES = binding.SAMPLES // HOP
LAYERS = (4, 8, 12)
WIDTH = 768
PROJECTED_WIDTHS = (22, 21, 21)


def _head():
    # This is the graph used by the frozen two-generator fit in the published comparison.
    return nn.Sequential(
        nn.Conv1d(sum(PROJECTED_WIDTHS), 24, 1), nn.GELU(),
        nn.Conv1d(24, 24, 5, padding=2, groups=24), nn.Conv1d(24, 24, 1), nn.GELU(),
        nn.Conv1d(24, 24, 5, dilation=4, padding=8, groups=24), nn.Conv1d(24, 24, 1), nn.GELU(),
        nn.Conv1d(24, 24, 5, dilation=12, padding=24, groups=24), nn.Conv1d(24, 16, 1), nn.GELU(),
        nn.Conv1d(16, 2, 1),
    )


class FocusedSpeechModel:
    def __init__(self, encoder, head, center, scale):
        self.encoder, self.head, self.center, self.scale = encoder, head, center, scale
        rng = np.random.default_rng(611407)
        self.projections = [torch.from_numpy(np.linalg.qr(rng.choice([-1., 1.], size=(WIDTH, channels)))[0].astype(np.float32))
                            for channels in PROJECTED_WIDTHS]
        self.source_clock = 200 + 320 * np.arange(299)
        self.head_clock = HOP * np.arange(FRAMES)

    @torch.inference_mode()
    def predict(self, pcm):
        waveform = torch.from_numpy(pcm)[None]
        result = self.encoder(waveform, attention_mask=torch.ones_like(waveform, dtype=torch.long),
                              output_hidden_states=True, return_dict=True)
        if result.last_hidden_state.shape != (1, len(self.source_clock), WIDTH):
            raise ValueError("Focused speech encoder clock differs from its calibration")
        projected = torch.cat([result.hidden_states[layer][0] @ projection
                               for layer, projection in zip(LAYERS, self.projections, strict=True)], dim=1).numpy()
        features = np.asarray([np.interp(self.head_clock, self.source_clock, column) for column in projected.T], dtype="<f4")
        normalized = (features - self.center[:, None]) / self.scale[:, None]
        readings = self.head(torch.from_numpy(normalized[None]))[0].numpy()
        if not np.isfinite(readings).all():
            raise ValueError("Nonfinite focused speech readings")
        return readings


def load_model(model_id: str, device: str, dtype: str):
    from safetensors.torch import load_file
    from transformers import WavLMConfig, WavLMModel

    from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file
    from app.engine.models import ModelUnavailable

    head_file = {MODEL_ID: HEAD_FILE, CENTERED_MODEL_ID: CENTERED_HEAD_FILE}.get(model_id)
    if head_file is None or device != "cpu" or dtype != "float32":
        raise ModelUnavailable("Focused speech review requires its pinned CPU float32 state")
    entry = _entry(model_id)
    for weight in entry.source.files:
        path = _file(entry, weight.path)
        if not path.is_file() or path.stat().st_size != weight.bytes:
            raise ModelUnavailable("A pinned focused speech asset is missing")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != weight.sha256:
                raise ModelUnavailable("A pinned focused speech asset differs")
    preprocessing = json.loads(_file(entry, "preprocessor_config.json").read_text())
    if preprocessing.get("sampling_rate") != binding.RATE or preprocessing.get("do_normalize") is not False:
        raise ModelUnavailable("Focused speech preprocessing differs from its calibration")
    encoder_weights = _file(entry, ENCODER_FILE)
    if torch.serialization.get_unsafe_globals_in_checkpoint(encoder_weights):
        raise ModelUnavailable("Unexpected globals in the pinned focused speech checkpoint")
    config = WavLMConfig.from_json_file(_file(entry, "config.json"))
    config._attn_implementation = "eager"
    encoder = WavLMModel(config).eval().requires_grad_(False)
    encoder.load_state_dict(torch.load(encoder_weights, map_location="cpu", weights_only=True), strict=True)
    if (sum(p.numel() for p in encoder.parameters()) != PARAMETERS
            or any(p.dtype != torch.float32 or not torch.isfinite(p).all().item() for p in encoder.parameters())):
        raise ModelUnavailable("Focused speech encoder state differs from its calibration")
    tensors = load_file(str(_file(entry, head_file)), device="cpu")
    norm = [tensors.pop(key).numpy() for key in ("feature_mean", "feature_scale")]
    if (any(v.shape != (sum(PROJECTED_WIDTHS),) or v.dtype != np.float32 or not np.isfinite(v).all() for v in norm)
            or np.any(norm[1] <= 0)):
        raise ModelUnavailable("Focused speech normalization is invalid")
    head = _head().eval().requires_grad_(False)
    head.load_state_dict(tensors, strict=True)
    pin = next(file.sha256 for file in entry.source.files if file.path == head_file)
    return LoadedModel(FocusedSpeechModel(encoder, head, *norm), pin)


def scan(pcm, spans, model, deadline):
    ranges = [(start, start + width) for width in binding.WIDTHS
              for start in range(0, binding.SAMPLES - width + 1, binding.STEP)
              if window_in_speech(spans, start / binding.RATE, (start + width) / binding.RATE, fraction=.5)]
    if not ranges:
        return {"checked_windows": 0, "best": None}
    deadline.check("focused speech edit review")
    local, edge = model.predict(pcm)
    peak = None
    for start, end in ranges:
        a, b = start // HOP, end // HOP
        onset = max(edge[max(0, a - 4):min(FRAMES, a + 5)])
        tail = max(edge[max(0, b - 4):min(FRAMES, b + 5)])
        score = float(local[a:b].mean()) + .075 * float(onset + tail)
        candidate = (score, -start, -end)
        if peak is None or candidate > peak:
            peak = candidate
    deadline.check("focused speech edit review")
    score, start, end = peak
    return {"checked_windows": len(ranges), "best": {"start_sample": -start, "end_sample": -end,
            "score_units": round(round(score, 8) * binding.SCORE_SCALE)}}


class FocusedSpeechEditReview(Detector):
    spec = video_spec(binding.DETECTOR_ID, binding.TITLE, InputUnit.AUDIO_WINDOW, (), family="voice-review",
                      model_ids=(MODEL_ID, ACTIVITY_MODEL), needs=("audio",), modality=Modality.AUDIO,
                      devices=("cpu",), timeout_s=BUDGET_S, version="1.0.0")
    model_id = MODEL_ID

    def activation_plan(self, ctx):
        return (ActivationUse(self.model_id, 1, "audio stream", "One six-second speech edit review"),
                ActivationUse(ACTIVITY_MODEL, 1, "audio stream", "One six-second speech-activity pass"))

    def applicable(self, ctx):
        track = ctx.outputs.audio
        eligible = track is not None and track.pcm is not None and len(track.pcm) == binding.SAMPLES and not track.clock_breaks
        return Applicability(eligible, None if eligible else "No eligible continuous six-second recording")

    def run(self, ctx):
        pcm = ctx.outputs.audio.pcm
        spans = ctx.lease.get(ACTIVITY_MODEL).obj.spans(pcm)
        response = scan(pcm, spans, ctx.lease.get(MODEL_ID).obj, ctx.deadline)
        return DetectorResult(status="ok", details={binding.KEY: response,
                              "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()})


def centered_readings(pcm, spans, model, deadline):
    clock = np.arange(FRAMES) * HOP
    speech = np.zeros(FRAMES, dtype=np.float64)
    for start, stop in spans:
        speech += np.clip(np.minimum(clock + HOP, stop * binding.RATE) - np.maximum(clock, start * binding.RATE), 0, HOP)
    mass = np.r_[0., np.cumsum(np.minimum(speech, HOP))]
    readings = [None] * len(binding.CENTERED_WIDTHS)
    admitted = []
    for width in binding.CENTERED_WIDTHS:
        cells = width // HOP
        left = np.arange(0, FRAMES - cells + 1, binding.CENTERED_STEP // HOP)
        right = left + cells
        valid = 2 * (mass[right] - mass[left]) >= width - .00001
        admitted.append((left[valid], right[valid], cells))
    count = sum(len(left) for left, _, _ in admitted)
    if count:
        deadline.check("focused speech edit review")
        local, edge = model.predict(pcm).astype(np.float64)
        content = np.r_[0., np.cumsum(local, dtype=np.float64)]
        edges = np.lib.stride_tricks.sliding_window_view(np.pad(edge, (4, 5), constant_values=-np.inf), 9).max(axis=1)
        for i, (left, right, cells) in enumerate(admitted):
            if len(left):
                scores = (content[right] - content[left]) / cells + .075 * (edges[left] + edges[right])
                peak = int(np.argmax(scores))
                readings[i] = {"score": round(float(scores[peak]), 8), "range": [int(left[peak]) * HOP, int(right[peak]) * HOP]}
    deadline.check("focused speech edit review")
    return count, readings


class CenteredSpeechEditReview(FocusedSpeechEditReview):
    model_id = CENTERED_MODEL_ID
    spec = replace(FocusedSpeechEditReview.spec, version="1.1.0",
                   requirements=replace(FocusedSpeechEditReview.spec.requirements,
                                        model_ids=(CENTERED_MODEL_ID, ACTIVITY_MODEL)))

    def applicable(self, ctx):
        track = ctx.outputs.audio
        eligible = track is not None and track.pcm is not None and not track.clock_breaks and binding.centered_scope(len(track.pcm)) is not None
        return Applicability(eligible, None if eligible else "The middle interval requires continuous audio from nine to fourteen seconds.")

    def run(self, ctx):
        pcm = ctx.outputs.audio.pcm
        start, end = binding.centered_scope(len(pcm))
        window = pcm[start:end]
        activity = ctx.lease.get(ACTIVITY_MODEL).obj
        model = ctx.lease.get(self.model_id).obj
        count, readings = centered_readings(window, activity.spans(window), model, ctx.deadline)
        return DetectorResult(status="ok", details={
            "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
            binding.KEY: {"schema": binding.CENTERED_SCHEMA, "scope": binding.interval_dict(start, end),
                          "scope_pcm_sha256": hashlib.sha256(window.astype("<f4").tobytes()).hexdigest(),
                          "checked_windows": count, "readings": readings},
        })
