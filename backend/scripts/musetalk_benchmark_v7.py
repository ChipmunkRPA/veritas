"""Adapted research-only MuseTalk inference; no application detector is installed.

Inference equations are derived from Tencent Music's MIT MuseTalk source
0a89dec45a0192b824e3cf4daf96c239440c5ed8 (unet.py, vae.py,
audio_processor.py). Preserve the publisher notice in the asset documentation.
Downloaded upstream Python and shell files are never imported or executed.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from scripts.fetch_mouth_generator_v7 import digest, verify

ASSET_MANIFEST_SHA = "bc823fb8f21e21c8c1cd722d225c927b0102b724b76942457f166ef047caf9fc"
METHOD = "musetalk-adapted-benchmark/7.0"


def positional_encoding(length=50, width=384):
    if width != 384 or not 0 < length <= 5000:
        raise ValueError("Unsupported publisher positional-encoding shape")
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(torch.arange(0, width, 2, dtype=torch.float32) * (-np.log(10000.0) / width))
    values = torch.zeros((length, width), dtype=torch.float32)
    values[:, 0::2] = torch.sin(position * divisor)
    values[:, 1::2] = torch.cos(position * divisor)
    return values.unsqueeze(0)


def audio_chunks(hidden, frames):
    """At 25 Hz, each frame takes ten 50 Hz ticks and all five encoder states."""
    if hidden.ndim != 3 or hidden.shape[1:] != (5, 384) or frames <= 0 or hidden.shape[0] < frames * 2:
        raise ValueError("Unsupported exact-clock speech conditioning")
    indices = torch.arange(frames)[:, None] * 2 + torch.arange(-4, 6)[None, :]
    valid = (indices >= 0) & (indices < len(hidden))
    chunks = hidden[indices.clamp(0, len(hidden) - 1)].clone()
    chunks[~valid] = 0
    return chunks.reshape(frames, 50, 384).contiguous()


def image_inputs(rgb):
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError("A native 256x256 RGB benchmark crop is required")
    # The publisher divides in NumPy float64, casts to torch float32, masks,
    # then applies Normalize(0.5,0.5). Keep that order and its rounding.
    pixels = torch.from_numpy(np.ascontiguousarray((rgb / 255.0).transpose(2, 0, 1))).float().unsqueeze(0)
    mask = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
    mask[:, :, :128] = 1
    return (pixels * mask - 0.5) / 0.5, (pixels - 0.5) / 0.5


def tensor_sha(value):
    return hashlib.sha256(value.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()).hexdigest()


def load(manifest_path, assets, device="mps"):
    from accelerate import init_empty_weights
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import WhisperConfig, WhisperFeatureExtractor
    from transformers.models.whisper.modeling_whisper import WhisperEncoder

    if digest(manifest_path) != ASSET_MANIFEST_SHA:
        raise ValueError("Benchmark assets differ from the reviewed permitted plan")
    manifest = json.loads(manifest_path.read_text())
    receipt = json.loads((assets / "verified-assets.json").read_text())
    actual = [verify(assets / row["path"], row) for row in manifest["assets"]]
    if receipt["manifest_sha256"] != ASSET_MANIFEST_SHA or receipt["assets"] != actual:
        raise ValueError("Asset acquisition receipt differs from exact file verification")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Explicit MPS generation is unavailable; no silent fallback")
    if device not in {"mps", "cpu"}:
        raise ValueError("This local benchmark port supports explicit CPU or MPS only")
    config = json.loads((assets / "musetalk/musetalk.json").read_text())
    if (config["in_channels"], config["out_channels"], config["cross_attention_dim"]) != (8, 4, 384):
        raise ValueError("Unexpected publisher image/audio architecture")
    checkpoint = assets / "musetalk/unet.pth"
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):
        raise ValueError("Publisher checkpoint requires an unreviewed pickle global")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    with init_empty_weights():
        unet = UNet2DConditionModel(**config)
    unet.load_state_dict(state, strict=True, assign=True)
    unet.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    del state
    vae = AutoencoderKL.from_config(json.loads((assets / "vae/config.json").read_text()))
    state = load_file(assets / "vae/diffusion_pytorch_model.safetensors")
    before = {id(value): key for key, value in state.items()}
    state = vae._fix_state_dict_keys_on_load(state)
    after = {id(value): key for key, value in state.items()}
    if before.keys() != after.keys() or len(before) != len(state):
        raise ValueError("Legacy VAE conversion changed or duplicated tensor objects")
    renamed = [{"from": name, "to": after[key]} for key, name in before.items() if name != after[key]]
    if len(renamed) != 16:
        raise ValueError("Unexpected legacy VAE attention key conversion")
    vae.load_state_dict(state, strict=True)
    if not all(torch.equal(value, vae.state_dict()[key]) for key, value in state.items()):
        raise ValueError("Strict VAE loading changed a publisher tensor")
    del state
    vae.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    speech_config = WhisperConfig.from_dict(json.loads((assets / "speech/config.json").read_text()))
    speech_config._attn_implementation = "eager"
    encoder = WhisperEncoder(speech_config)
    with safe_open(assets / "speech/model.safetensors", framework="pt", device="cpu") as file:
        keys = list(file.keys())
        if any(not key.startswith(("model.encoder.", "model.decoder.", "proj_out.")) for key in keys):
            raise ValueError("Unexpected speech-checkpoint tensor family")
        state = {
            key.removeprefix("model.encoder."): file.get_tensor(key)
            for key in keys
            if key.startswith("model.encoder.")
        }
    encoder.load_state_dict(state, strict=True)
    encoder.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    extractor = WhisperFeatureExtractor.from_pretrained(assets / "speech", local_files_only=True)
    if extractor.feature_size != 80 or extractor.sampling_rate != 16000:
        raise ValueError("Unexpected speech preprocessing configuration")
    return Renderer(
        unet,
        vae,
        encoder,
        extractor,
        device,
        {
            "vae_attention_key_mapping": renamed,
            "vae_all_cpu_tensors_exact": True,
            "all_three_modules_strict_loaded": True,
        },
    )


class Renderer:
    def __init__(self, unet, vae, encoder, extractor, device, load_audit):
        self.unet, self.vae, self.encoder, self.extractor, self.device = unet, vae, encoder, extractor, device
        self.positions = positional_encoding().to(device)
        self.load_audit = load_audit

    @torch.inference_mode()
    def conditioning(self, pcm):
        if pcm.ndim != 1 or not 640 <= len(pcm) <= 480000 or not np.isfinite(pcm).all():
            raise ValueError("Benchmark speech must be finite mono 16 kHz PCM, at most 30 seconds")
        frames, ticks = len(pcm) // 640, len(pcm) // 320
        inputs = self.extractor(pcm, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        states = self.encoder(inputs, output_hidden_states=True).hidden_states
        hidden = torch.stack(states, dim=2)[0, :ticks].detach().to("cpu", torch.float32)
        return audio_chunks(hidden, frames)

    @torch.inference_mode()
    def render(self, rgb, conditioning, seed):
        if (
            conditioning.shape != (50, 384)
            or not torch.isfinite(conditioning).all()
            or not isinstance(seed, int)
            or not 0 <= seed < 2**63
        ):
            raise ValueError("Invalid frozen speech conditioning or benchmark seed")
        masked, original = image_inputs(rgb)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        a = self.vae.encode(masked.to(self.device)).latent_dist.sample(generator=generator)
        b = self.vae.encode(original.to(self.device)).latent_dist.sample(generator=generator)
        latents = self.vae.config.scaling_factor * torch.cat((a, b), dim=1)
        audio = conditioning.unsqueeze(0).to(self.device) + self.positions
        predicted = self.unet(
            latents, torch.tensor([0], device=self.device), encoder_hidden_states=audio
        ).sample
        pixels = self.vae.decode(predicted / self.vae.config.scaling_factor).sample
        if pixels.shape != (1, 3, 256, 256) or not torch.isfinite(pixels).all():
            raise ValueError("Invalid or nonfinite neural benchmark output")
        result = (
            ((pixels[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        )
        return result, {
            "method": METHOD,
            "seed": seed,
            "masked_input_f32_sha256": tensor_sha(masked),
            "original_input_f32_sha256": tensor_sha(original),
            "sampled_latents_f32_sha256": tensor_sha(latents),
            "condition_with_position_f32_sha256": tensor_sha(audio),
            "predicted_latents_f32_sha256": tensor_sha(predicted),
            "output_rgb_sha256": hashlib.sha256(result.tobytes()).hexdigest(),
        }
