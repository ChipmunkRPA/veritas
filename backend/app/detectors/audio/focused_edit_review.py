"""Optional CPU review of short speech windows in one six-second mono recording."""

from __future__ import annotations

import hashlib
import json

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

    if model_id != MODEL_ID or device != "cpu" or dtype != "float32":
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
    tensors = load_file(str(_file(entry, HEAD_FILE)), device="cpu")
    norm = [tensors.pop(key).numpy() for key in ("feature_mean", "feature_scale")]
    if (any(v.shape != (sum(PROJECTED_WIDTHS),) or v.dtype != np.float32 or not np.isfinite(v).all() for v in norm)
            or np.any(norm[1] <= 0)):
        raise ModelUnavailable("Focused speech normalization is invalid")
    head = _head().eval().requires_grad_(False)
    head.load_state_dict(tensors, strict=True)
    pin = next(file.sha256 for file in entry.source.files if file.path == HEAD_FILE)
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

    def activation_plan(self, ctx):
        return (ActivationUse(MODEL_ID, 1, "audio stream", "One six-second speech edit review"),
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
