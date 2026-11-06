"""Pinned, offline speech-enhancement view for detector development only.

The enhanced signal is a transformed inspection input, never a recovered original
or ground truth. No serving module imports this file and no score is calibrated
by the enhancer. Original source timing and PCM remain authoritative.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch

REVISION = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
PINS = {
    "LICENSE": "c467165e5860b4a7494ef4a6e2788e4115d95b5b8989bb5f86287089adddf794",
    "gtcrn.py": "20bf0610702506fa45cd543454b84bbeb862d73c7c20a3f8d820fdc04da6099b",
    "infer.py": "1061bb6a9a6f1a27423f7603871e683aaf0cd3ffd40a75efa9f9ba0965330d7b",
    "checkpoints/model_trained_on_dns3.tar": "a630d992cf792daf4ce2bb5bcf9c4d389f740a8f09c6e0971184697fe6371b79",
}
METHOD = {
    "version": "audio-enhanced-view-development/4.0",
    "upstream_revision": REVISION,
    "checkpoint": "DNS3",
    "publisher_license": "MIT",
    "device": "cpu",
    "dtype": "float32",
    "sample_rate": 16000,
    "fft_samples": 512,
    "hop_samples": 256,
    "window": "square-root periodic Hann",
    "center": True,
    "output_length": "explicitly equal to original input length",
    "original_gate_and_windows": True,
    "post_normalization": False,
    "post_clipping": False,
    "deployment_eligible": False,
}


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(root: Path):
    for name, expected in PINS.items():
        if file_sha(root / name) != expected:
            raise ValueError("Noise-view source or checkpoint differs from its reviewed pin: " + name)
    module_spec = importlib.util.spec_from_file_location("chipmunk_gtcrn_v4", root / "gtcrn.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    state = torch.load(root / "checkpoints/model_trained_on_dns3.tar", map_location="cpu", weights_only=True)
    model = module.GTCRN().float().eval()
    model.load_state_dict(state["model"], strict=True)
    return model


@torch.inference_mode()
def enhance(values: np.ndarray, model) -> np.ndarray:
    values = np.asarray(values)
    if (
        values.dtype != np.float32
        or values.ndim != 1
        or not 16000 <= len(values) <= 64000
        or not np.isfinite(values).all()
    ):
        raise ValueError("Noise-view development requires 1–4 seconds of finite float32 mono 16 kHz PCM")
    window = torch.hann_window(512).sqrt()
    spec = torch.stft(torch.from_numpy(values.copy()), 512, 256, 512, window, return_complex=True)
    output = model(torch.view_as_real(spec)[None])[0]
    if output.shape != (*spec.shape, 2) or output.dtype != torch.float32:
        raise ValueError("Enhancer output differs from the pinned real/imaginary spectrum contract")
    output = torch.view_as_complex(output.contiguous())
    result = torch.istft(output, 512, 256, 512, window, length=len(values)).numpy().astype(np.float32)
    if result.shape != values.shape or not np.isfinite(result).all():
        raise ValueError("Enhanced input lost source sample positions or finite values")
    return result
