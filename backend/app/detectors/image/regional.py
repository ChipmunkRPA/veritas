"""Offline CPU regional screening, using a complete verified safetensors state.

The graph preserves the evaluated native geometry. A response is an editing
cue, not a calibrated probability of AI authorship. No downloads, global
import shims, pickle deserialization, thread changes or LoRA merges occur here.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL_ID = "local.native_regional_v1"
CHECKPOINT_FILE = "regional-model.safetensors"
CHECKPOINT_SHA256 = "587d553353ddab1c81f45e86fbc9ad2326dee9086c6b566d9abcacc221380590"
CHECKPOINT_BYTES = 367679412
PIXEL_THRESHOLD = 0.90
MINIMUM_AREA_FRACTION = 0.01


def require(value: bool, reason: str) -> None:
    if not value:
        raise ValueError(reason)


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare(image: Image.Image) -> tuple[torch.Tensor, dict]:
    """Use all upright RGB pixels; fit without stretching, pad only right/bottom."""
    require(isinstance(image, Image.Image) and image.mode == "RGB", "Upright RGB source required")
    w, h = image.size
    require(min(w, h) > 0 and image.getexif().get(274, 1) == 1, "Nonempty upright source required")
    fw, fh = (w, h) if max(w, h) <= 1024 else (
        max(1, w * 1024 // max(w, h)), max(1, h * 1024 // max(w, h)))
    fit = image if (fw, fh) == (w, h) else image.resize((fw, fh), Image.Resampling.BILINEAR)
    canvas = np.zeros((1024, 1024, 3), np.uint8)
    canvas[:fh, :fw] = np.asarray(fit)
    mean = np.asarray([0.485, 0.456, 0.406], np.float32)
    std = np.asarray([0.229, 0.224, 0.225], np.float32)
    values = (canvas.astype(np.float32) / np.float32(255) - mean) / std
    reference = (canvas.astype(np.float64) / 255 - np.asarray([0.485, 0.456, 0.406])) / np.asarray(
        [0.229, 0.224, 0.225])
    error = float(np.max(np.abs(values - reference)))
    require(error <= 1e-6, "Image normalization reference differs")
    tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)[None]))
    return tensor, {
        "source_size": [w, h], "fit_size": [fw, fh], "canvas_size": [1024, 1024],
        "source_rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
        "input_float32_sha256": tensor_sha(tensor),
        "normalization_reference_maximum_difference": error, "source_cropped": False,
        "padding_position": "top_left", "padding_rgb": [0, 0, 0],
        "resize": "Only if larger than1024; bilinear, aspect preserved with integer floor sizes",
        "readout": "Nearest source-center index, exact integer ratios; retain float32 response bits, exclude padding",
        "response_meaning": "Uncalibrated localizer response, not pixel-authorship probability",
    }


def source_map(output: torch.Tensor, geometry: dict) -> np.ndarray:
    w, h = geometry["source_size"]
    fw, fh = geometry["fit_size"]
    require(output.shape == (1, 1, 1024, 1024) and output.dtype == torch.float32
            and output.device.type == "cpu" and bool(torch.isfinite(output).all()),
            "Complete finite CPU float32 response canvas required")
    values = output[0, 0, :fh, :fw].detach().numpy()
    require(float(values.min()) >= 0 and float(values.max()) <= 1, "Response outside sigmoid range")
    rows = np.minimum(fh - 1, ((2 * np.arange(h, dtype=np.int64) + 1) * fh) // (2 * h))
    cols = np.minimum(fw - 1, ((2 * np.arange(w, dtype=np.int64) + 1) * fw) // (2 * w))
    return values[rows[:, None], cols[None, :]].copy()


@dataclass(frozen=True)
class RegionalPrediction:
    raw_mask: np.ndarray
    geometry: dict
    binding: dict
    candidate_eligibility: dict


class RegionalModel:
    def __init__(self, model: torch.nn.Module):
        self.model = model.eval().requires_grad_(False)
        # RoPE caches and input-binding hooks belong to this instance. Never
        # modify process-global imports/network/model factories in requests.
        self._lock = threading.RLock()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> RegionalPrediction:
        from app.evidence.image_eligibility import area_gate

        tensor, geometry = prepare(image)
        fw, fh = geometry["fit_size"]
        gh, gw = (2 * fh + 1023) // 1024, (2 * fw + 1023) // 1024
        expected = torch.stack([
            tensor[0, :, r * 496:r * 496 + 528, c * 496:c * 496 + 528]
            for r in range(gh) for c in range(gw)])
        calls = []

        def bind(_module, args):
            require(torch.equal(args[0], expected), "Actual image patches differ from source geometry")
            calls.append(tensor_sha(args[0]))

        with self._lock:
            hook = self.model.vit.patch_embed.register_forward_pre_hook(bind)
            try:
                output = self.model(image=tensor, origin_shape=torch.tensor([[fh, fw]], dtype=torch.int64),
                                    clip_len=torch.tensor([1], dtype=torch.int64))
            finally:
                hook.remove()
            require(len(calls) == 1 and tensor_sha(tensor) == geometry["input_float32_sha256"],
                    "Image input binding incomplete")
            raw = source_map(output, geometry)
            binding = {"actual_patch_count": gh * gw, "actual_patch_float32_sha256": calls[0],
                       "complete_canvas_float32_sha256": tensor_sha(output), "merged_lora": False}
        gate = area_gate(int(np.count_nonzero(raw > PIXEL_THRESHOLD)),
                         int(np.ceil(MINIMUM_AREA_FRACTION * raw.size)), raw.size)
        return RegionalPrediction(raw, geometry, binding, gate)


def load_model(checkpoint: Path) -> RegionalModel:
    """Reject unexpected bytes before parsing, then strictly validate every tensor."""
    from safetensors.torch import load_file

    require(checkpoint.is_file() and checkpoint.stat().st_size == CHECKPOINT_BYTES,
            "Complete pinned regional weights required")
    with checkpoint.open("rb") as stream:
        require(hashlib.file_digest(stream, "sha256").hexdigest() == CHECKPOINT_SHA256,
                "Regional weights digest differs")
    state = load_file(str(checkpoint), device="cpu")
    require(len(state) == 410 and all(v.dtype == torch.float32 and bool(torch.isfinite(v).all())
                                    for v in state.values()), "Complete finite float32 state required")
    from app.detectors.image.regional_graph.RelayFormer import RelayFormer

    model = RelayFormer(vit_pretrain_path="complete-local-state")
    template = model.state_dict()
    require(set(template) == set(state) and all(template[k].shape == v.shape and template[k].dtype == v.dtype
                                               for k, v in state.items()), "Regional graph/state schema differs")
    model.load_state_dict(state, strict=True)
    require(all(torch.equal(model.state_dict()[k], v) for k, v in state.items()),
            "Loaded regional tensors differ")
    return RegionalModel(model)


def load_regional(model_id: str, device: str, dtype: str):
    from app.detectors.image.adapters.loaders import LoadedModel, _entry, _file
    from app.engine.models import ModelUnavailable

    if model_id != MODEL_ID or device != "cpu" or dtype != "float32":
        raise ModelUnavailable("Regional screening requires its verified CPU float32 state")
    return LoadedModel(load_model(_file(_entry(model_id), CHECKPOINT_FILE)), CHECKPOINT_SHA256)
