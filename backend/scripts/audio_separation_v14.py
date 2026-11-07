"""Research-only two-source separator with exact pinned tensor/clock contracts.

Architecture adapted from Asteroid0.7.0 and asteroid-filterbanks0.4.0,
Copyright (c)2019 Pariente Manuel, MIT. Full notice accompanies this file at
docs/third-party/asteroid-v14-MIT.txt. This code license is separate from the
checkpoint's retained CC-BY-SA3.0/4.0 publisher statements.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REVISION = "e1ef95ab7a037950f3a606b9a56760cf94701d3d"
WEIGHT_SHA = "8d97f012f7b2f22bb79cb0d0983a7ba27a52c1796ee3f63cbf25b4d28630adce"
CARD_SHA = "c34341c446d0c5abd1603ee3cb9e4c0093a7b2422389fdeceeea7625e280fce1"
ARGS = {
    "fb_name": "FreeFB", "n_filters": 512, "kernel_size": 32, "stride": 16, "sample_rate": 8000,
    "in_chan": 512, "out_chan": 512, "bn_chan": 128, "hid_chan": 512, "skip_chan": 128,
    "conv_kernel_size": 3, "n_blocks": 8, "n_repeats": 3, "n_src": 2, "norm_type": "gLN",
    "mask_act": "relu", "encoder_activation": None,
}
SPEC = {
    "repository": "JorisCos/ConvTasNet_Libri2Mix_sepclean_16k", "revision": REVISION,
    "weight_sha256": WEIGHT_SHA, "card_sha256": CARD_SHA, "parameters": 5066929, "tensor_entries": 345,
    "clock": "Declared training16kHz; serialized default8000 retained as metadata only. Free learned filters have no samplerate-dependent arithmetic. No resampling.",
    "input": "Exact mono16kHz64000 float32 samples, no input normalization, denoising, padding or clipping",
    "output": "Two unordered64000-sample float32 source estimates; not identified people or guaranteed clean voices",
    "terms": "Card metadata CC-BY-SA-4.0, explicit author notice CC-BY-SA-3.0, LibriSpeech source CC-BY-4.0. Version conflict retained; no unambiguous license or complete training-rights warranty asserted.",
    "runtime": "CPUfloat32, frozen weights, eval. Inspected local architecture, no dynamic remote code or unsafe pickle globals.",
    "serving_admitted": False,
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class GlobalNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, value):
        dims = list(range(1, value.ndim))
        normalized = (value - value.mean(dims, keepdim=True)) / torch.sqrt(
            torch.var(value, dim=dims, keepdim=True, unbiased=False) + 1e-8)
        return (self.gamma * normalized.transpose(1, -1) + self.beta).transpose(1, -1)


class Filters(nn.Module):
    def __init__(self):
        super().__init__()
        self._filters = nn.Parameter(torch.ones(512, 1, 32))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.filterbank = Filters()

    def forward(self, pcm):
        return F.conv1d(pcm, self.filterbank._filters, stride=16)


class Decoder(Encoder):
    def forward(self, features):
        shape = features.shape
        decoded = F.conv_transpose1d(features.reshape(-1, *shape[-2:]), self.filterbank._filters, stride=16)
        return decoded.reshape(*shape[:-2], -1)


class Block(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.shared_block = nn.Sequential(nn.Conv1d(128, 512, 1), nn.PReLU(), GlobalNorm(512),
            nn.Conv1d(512, 512, 3, padding=dilation, dilation=dilation, groups=512), nn.PReLU(), GlobalNorm(512))
        self.res_conv = nn.Conv1d(512, 128, 1)
        self.skip_conv = nn.Conv1d(512, 128, 1)

    def forward(self, value):
        shared = self.shared_block(value)
        return self.res_conv(shared), self.skip_conv(shared)


class Masker(nn.Module):
    def __init__(self):
        super().__init__()
        self.bottleneck = nn.Sequential(GlobalNorm(512), nn.Conv1d(512, 128, 1))
        self.TCN = nn.ModuleList(Block(2**index) for _ in range(3) for index in range(8))
        self.mask_net = nn.Sequential(nn.PReLU(), nn.Conv1d(128, 1024, 1))
        self.output_act = nn.ReLU()

    def forward(self, features):
        value = self.bottleneck(features)
        skip = torch.tensor([0.0], device=value.device)
        for layer in self.TCN:
            residual, addition = layer(value)
            skip = skip + addition
            value = value + residual
        return self.output_act(self.mask_net(skip).view(features.shape[0], 2, 512, features.shape[-1]))


class Separator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.masker, self.decoder = Encoder(), Masker(), Decoder()

    def forward(self, pcm):
        if (pcm.ndim != 3 or pcm.shape[1:] != (1, 64000) or pcm.dtype != torch.float32
                or pcm.device.type != "cpu" or not torch.isfinite(pcm).all().item()):
            raise ValueError("Exact finite CPUfloat32 four-second mono inputs required")
        encoded = self.encoder(pcm)
        return self.decoder(self.masker(encoded) * encoded[:, None])


def load(root):
    root = Path(root)
    if digest(root / "pytorch_model.bin") != WEIGHT_SHA or digest(root / "README.md") != CARD_SHA:
        raise ValueError("Pinned separation checkpoint/card bytes changed")
    if torch.serialization.get_unsafe_globals_in_checkpoint(root / "pytorch_model.bin"):
        raise ValueError("Unexpected checkpoint pickle globals")
    release = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
    if (release["model_name"] != "ConvTasNet" or release["model_args"] != ARGS
            or release["dataset"] != "Libri2Mix" or release["task"] != "sep_clean"):
        raise ValueError("Exact published construction and serialization metadata required")
    model = Separator().eval().requires_grad_(False)
    model.load_state_dict(release["state_dict"], strict=True)
    if (len(model.state_dict()) != 345 or sum(p.numel() for p in model.parameters()) != 5066929
            or any(not torch.isfinite(p).all() or p.requires_grad or p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("Exact finite frozen5,066,929-parameter architecture required")
    return model


def separate(pcm, model):
    pcm = np.asarray(pcm)
    if pcm.shape != (64000,) or pcm.dtype != np.float32 or not np.isfinite(pcm).all():
        raise ValueError("Exact original finite64000-sample float32 input required")
    with torch.inference_mode():
        output = model(torch.from_numpy(pcm.copy())[None, None])
    if output.shape != (1, 2, 64000) or output.dtype != torch.float32 or not torch.isfinite(output).all():
        raise ValueError("Two complete finite source estimates required")
    return output[0].cpu().numpy().copy()
