"""Inference-only four-second speech combination; not enabled by default.

No fitting, research-script imports, downloads or customer-data persistence.
The caller owns the baseline and speech-activity models and the job deadline.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit

SCHEMA = "chipmunk.audio-combined-serving/1"
SAMPLES, SR, WIDTH, FRAMES = 64000, 16000, 4608, 199
LAYERS = (4, 8, 12)
ASSETS = {
    "pytorch_model.bin": "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51",
    "config.json": "fea6df1c2700a3954fc07e70588aecc9055eeb28db2ff57151a2db0d19180ed4",
    "preprocessor_config.json": "99272fe8ccfab114b68b478681ea47ee3a1ce62bb788cb92dd6e4f69fb1f1da2",
    "README.md": "d417fc16b08e2f66f0af6b371e47a5f9f1f596f6c1c593f1214d360c41d7a27e",
    "UniSpeech-LICENSE": "74295d561f60f770a4aa7525b71c0d119ec70422e9f5c601ee3c77e1b7822c91",
    "unilm-LICENSE": "904dc4d8749877f1dba1cda48200d2462dccbeb7c134d5e4ef6fa75e0198c8fe",
}
NOTE = (
    "A screening estimate under a balanced reference prior, not a measured accuracy rate for this recording. "
    "Validation used four-second English speech with controlled backgrounds. Strong noise, overlapping voices "
    "and unfamiliar generators can cause errors. This does not assess the authenticity of background sounds, "
    "establish a per-file confidence interval or provide a court conclusion."
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def vector(value, size):
    require(
        isinstance(value, list)
        and len(value) == size
        and all(type(x) in (int, float) and math.isfinite(x) for x in value),
        "Complete finite numeric calibration vector required",
    )
    result = np.asarray(value, np.float64)
    result.setflags(write=False)
    return result


def raw_logit(raw):
    require(
        isinstance(raw, (int, float, np.floating))
        and not isinstance(raw, (bool, np.bool_))
        and math.isfinite(raw)
        and 0 <= raw <= 1,
        "Finite component probability required",
    )
    p = float(np.clip(raw, 1e-6, 1 - 1e-6))
    return math.log(p) - math.log1p(-p)


def source_supported(pcm):
    return (
        isinstance(pcm, np.ndarray)
        and pcm.dtype == np.float32
        and pcm.shape == (SAMPLES,)
        and bool(np.isfinite(pcm).all())
    )


class CombinedCalibration:
    """The fixed complementary head followed by its separately fitted global map."""

    def __init__(self, bundle, checksum):
        require(
            set(bundle) == {"schema", "source_bindings", "complement", "global"}
            and bundle["schema"] == SCHEMA,
            "Exact inference bundle schema required",
        )
        require(
            isinstance(bundle["source_bindings"], dict)
            and set(bundle["source_bindings"])
            == {
                "study_freeze_sha256",
                "selected_global_sha256",
                "trained_fusion_sha256",
                "complement_head_sha256",
            }
            and all(
                isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)
                for v in bundle["source_bindings"].values()
            ),
            "Whole-artifact source bindings required",
        )
        head, fusion = bundle["complement"], bundle["global"]
        require(
            set(head) == {"center", "scale", "variance_factor", "coefficients"}
            and set(fusion) == {"coefficients", "calibration"},
            "Inference parameters only; no source identifiers or fitted-role records",
        )
        self.center, self.scale = vector(head["center"], WIDTH), vector(head["scale"], WIDTH)
        self.coefficients = vector(head["coefficients"], WIDTH + 2)
        require(
            bool((self.scale >= 1e-6).all()) and self.coefficients[0] >= 0, "Invalid complementary transform"
        )
        factor = head["variance_factor"]
        require(
            type(factor) is float and factor == float(np.sqrt(16 / WIDTH)),
            "Frozen representation variance factor required",
        )
        self.variance_factor = factor
        self.fusion, self.calibration = vector(fusion["coefficients"], 3), vector(fusion["calibration"], 2)
        require(
            bool(((self.fusion[:2] >= 0) & (self.fusion[:2] <= 5)).all())
            and abs(self.fusion[-1]) <= 10
            and 0 <= self.calibration[0] <= 5
            and abs(self.calibration[1]) <= 10,
            "Invalid frozen monotone combination or calibration",
        )
        self.sha256 = checksum

    def predict(self, raw, features):
        x = np.asarray(features)
        require(
            x.shape == (WIDTH,) and x.dtype == np.float32 and bool(np.isfinite(x).all()),
            "Finite float32 speech representation required",
        )
        logit = raw_logit(raw)
        normalized = (x.astype(np.float64) - self.center) / self.scale * self.variance_factor
        complement = float(expit(np.r_[logit, normalized, 1.0] @ self.coefficients))
        # Match the frozen float64 array-logit path for the joint inputs.
        values = np.clip(np.asarray([float(raw), complement]), 1e-6, 1 - 1e-6)
        logits = np.clip(np.log(values) - np.log1p(-values), -8.0, 8.0)
        combined = float(np.r_[logits, 1.0] @ self.fusion)
        return float(expit(np.r_[combined, 1.0] @ self.calibration))


def load_calibration(path, expected_sha256):
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    require(
        isinstance(expected_sha256, str) and len(expected_sha256) == 64 and actual == expected_sha256,
        "Combined speech calibration integrity check failed",
    )
    return CombinedCalibration(json.loads(raw), actual)


def load_encoder(root):
    """Pinned local CPU encoder. Attribution/share-alike terms stay with its assets."""
    import torch
    from transformers import WavLMConfig, WavLMModel

    root = Path(root)
    require(
        all(sha(root / name) == checksum for name, checksum in ASSETS.items()),
        "Speech encoder or license bytes changed",
    )
    preprocessing = json.loads((root / "preprocessor_config.json").read_text())
    require(
        preprocessing["sampling_rate"] == SR and preprocessing["do_normalize"] is False,
        "Original sample preprocessing required",
    )
    require(
        not torch.serialization.get_unsafe_globals_in_checkpoint(root / "pytorch_model.bin"),
        "Unexpected checkpoint globals",
    )
    config = WavLMConfig.from_json_file(root / "config.json")
    config._attn_implementation = "eager"
    model = WavLMModel(config).eval().requires_grad_(False)
    state = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    require(
        sum(p.numel() for p in model.parameters()) == 94381936
        and all(
            not p.requires_grad
            and p.device.type == "cpu"
            and p.dtype == torch.float32
            and torch.isfinite(p).all().item()
            for p in model.parameters()
        ),
        "Frozen finite CPU float32 encoder required",
    )
    return model


def extract(pcm, encoder):
    import torch

    require(source_supported(pcm), "Exact original finite float32 four-second PCM required")
    values = torch.from_numpy(np.ascontiguousarray(pcm).copy())[None]
    with torch.inference_mode():
        output = encoder(
            values,
            attention_mask=torch.ones_like(values, dtype=torch.long),
            output_hidden_states=True,
            return_dict=True,
        )
        states = output.hidden_states
        require(
            len(states) == 13 and output.last_hidden_state.shape == (1, FRAMES, 768),
            "Original feature clock required",
        )
        selected = []
        for layer in LAYERS:
            state = states[layer]
            require(
                state.shape == (1, FRAMES, 768) and torch.isfinite(state).all().item(),
                "Complete finite selected feature frames required",
            )
            value = state.detach().float()[0]
            selected.extend((value.mean(0), value.std(0, unbiased=False)))
        return torch.stack(selected).reshape(-1).cpu().numpy().astype(np.float32, copy=True)


def unassessed(reason):
    return {
        "status": "not_applicable",
        "probability_ai": None,
        "reference_estimate_ai": None,
        "decision": "inconclusive",
        "scope_supported": False,
        "reason": reason,
        "limitation": NOTE,
        "checked_windows": [],
        "reference_prior_ai": 0.5,
    }


class CombinedSpeechRuntime:
    def __init__(self, calibration, encoder):
        self.calibration, self.encoder = calibration, encoder

    def analyze(self, pcm, baseline, activity, deadline):
        if not source_supported(pcm):
            return unassessed(
                "The combined speech estimate requires exactly four seconds of finite 16 kHz mono float32 audio. No padding, truncation or whole-file score was applied."
            )
        from app.detectors.audio.content import window_in_speech

        deadline.check("combined speech activity")
        spans = activity.spans(pcm.copy())
        previous_end = 0.0
        for start, end in spans:
            require(
                math.isfinite(start) and math.isfinite(end) and previous_end <= start < end <= 4.0,
                "Finite ordered nonoverlapping source speech spans required",
            )
            previous_end = end
        # Exactly 64000 samples yields only [0,4) in the incumbent four-second
        # window builder, at its two-second full-profile hop. Keep the same gate
        # without importing its unrelated neural classifiers here.
        if not window_in_speech(spans, 0.0, 4.0):
            return unassessed(
                "No complete source window contains enough detected speech; no authenticity conclusion can be made."
            )
        deadline.check("combined voice component")
        scores = baseline.predict([pcm.copy()])
        require(len(scores) == 1, "One actual voice component response required")
        raw_logit(scores[0])
        deadline.check("combined speech representation")
        features = extract(pcm, self.encoder)
        deadline.check("combined speech calibration")
        p = self.calibration.predict(scores[0], features)
        direction = "ai" if p > 0.5 else "original" if p < 0.5 else "inconclusive"
        reason = {
            "ai": "Leaning towards AI-generated speech.",
            "original": "Leaning towards original speech.",
            "inconclusive": "The speech evidence does not favor either direction.",
        }[direction]
        return {
            "status": "ok",
            "probability_ai": p,
            "reference_estimate_ai": p,
            "decision": direction,
            "scope_supported": True,
            "reason": reason,
            "limitation": NOTE,
            "reference_prior_ai": 0.5,
            "calibration_sha256": self.calibration.sha256,
            "checked_windows": [{"start_sample": 0, "end_sample": SAMPLES, "start_s": 0.0, "end_s": 4.0}],
        }
