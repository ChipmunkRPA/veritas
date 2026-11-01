"""Fixed-checkpoint image localisation for limited local inspection.

This is our independent inference-only tensor interpreter, promoted from the
hash-recorded research evaluator after bounded numerical reference parity. It
does not import or redistribute the separate, unlicensed upstream source. The
released checkpoint declares MIT; the installed timm implementation and pinned
backbone declare Apache-2.0. See docs/fused-local-research.md for their boundaries.

The map and whole-image head are uncalibrated measurements. The fixed image gate
selects candidate regions; it is neither a quality mask nor proof of editing.
No training, dynamic configuration, remote loading or alternate weights exist.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_ID = "fused.opensdi"
BACKBONE_MODEL_ID = "fused.convnext_xxlarge"
CHECKPOINT_FILE = "fused_opensdi.pth"
BACKBONE_FILE = "convnext.safetensors"
CHECKPOINT_SHA256 = "7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c"
BACKBONE_SHA256 = "1ddf6dbc7f0753fc4dfec5db5fa474acfd1ac6f67f466d6a976780c30b3a1b23"
CHECKPOINT_REVISION = "b7dfe0cb2d2f8fc8fd9e6ebea5f7185cfa8e1af7"
BACKBONE_ARCHITECTURE = "convnext_xxlarge.clip_laion2b_soup"
STATE_SCHEMA_SHA256 = "bb4155467f97b68dc4862388796bd75e019d0fc15f017b9469d5a85c6d3eeb1c"
INPUT_SIZE = 512
IMAGE_GATE_THRESHOLD = 0.5

# Data-only validation. `_target_` is never imported or instantiated.
EXPECTED_CONFIG = {
    "_target_": "fused.model.model.FUSED", "num_classes": 2, "dim": 512,
    "num_latents": 8, "num_heads": 8, "moe_num_experts": 8, "moe_top_k": 2,
    "moe_expansion": 4, "ffn_type": "moe", "seg_bridge": True,
    "enable_sparsevit_branch": True, "enable_convnext_branch": True,
    "convnext_model": BACKBONE_ARCHITECTURE, "depth": [5, 8, 20, 7],
    "embed_dim": [64, 128, 320, 512], "head_dim": 64, "img_size": 512,
    "s_blocks3": [8, 4, 2, 1], "s_blocks4": [2, 1], "mlp_ratio": 4,
    "qkv_bias": True, "drop_path_rate": 0.2, "pretrained_path": None,
    "enable_seg": True, "fusion_mode": "self_attn", "moe_before_sa": True,
    "seg_decoder_mode": "joint", "convnext_input_size": 512,
}


def candidate_eligibility(image_score: float) -> dict:
    """Return selection metadata without altering any pixel measurement.

    Hex values preserve the exact operating side across JSON canonical rounding.
    The adapter must additionally bind this record to the source and scalar asset.
    """
    if isinstance(image_score, bool) or not math.isfinite(image_score) or not 0 <= image_score <= 1:
        raise ValueError("The whole-image response must be finite and between zero and one")
    score = float(image_score)
    return {
        "schema": "image-candidate-eligibility/1",
        "status": "evaluated",
        "scope": "whole_image",
        "score_kind": "uncalibrated_response",
        "score": score,
        "score_hex": score.hex(),
        "threshold": IMAGE_GATE_THRESHOLD,
        "threshold_hex": IMAGE_GATE_THRESHOLD.hex(),
        "comparison": "greater_or_equal",
        "eligible": score >= IMAGE_GATE_THRESHOLD,
        "role": "candidate_selection_only",
        "raw_mask_retained": True,
    }


@dataclass(frozen=True)
class FusedPrediction:
    raw_mask: np.ndarray  # float32, 512 x 512; never changed by the image gate
    image_score: float
    source_size: tuple[int, int]  # width, height before the square resize

    @property
    def candidate_eligibility(self) -> dict:
        return candidate_eligibility(self.image_score)


def _execution_target(device: str, dtype: str | torch.dtype) -> torch.device:
    target = torch.device(device)
    if target.type not in {"cpu", "mps"} or target.index is not None:
        raise ValueError("Only unindexed CPU or MPS execution is supported")
    if dtype not in ("float32", torch.float32):
        raise ValueError("Only float32 execution has been evaluated")
    if target.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; choose CPU explicitly")
    return target


def _check_hash(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("The local checkpoint does not match its pinned SHA-256")


def _validated_state(saved: dict) -> dict[str, torch.Tensor]:
    if not isinstance(saved, dict) or set(saved) != {"state_dict", "cfg"}:
        raise ValueError("Unexpected checkpoint structure")
    config = saved["cfg"]
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Missing static architecture configuration")
    # JSON comparison also rejects bools substituted for integer dimensions.
    if json.dumps(config["model"], sort_keys=True) != json.dumps(EXPECTED_CONFIG, sort_keys=True):
        raise ValueError("Only the pinned architecture configuration is supported")
    state = saved["state_dict"]
    if (not isinstance(state, dict) or len(state) != 879
            or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items())):
        raise ValueError("Unexpected checkpoint tensor collection")
    schema = [[key, list(tensor.shape), str(tensor.dtype)] for key, tensor in sorted(state.items())]
    digest = hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
    if digest != STATE_SCHEMA_SHA256:
        raise ValueError("Checkpoint tensor keys, shapes or dtypes differ from the pinned schema")
    return state


def lattice_tokens(image: torch.Tensor, side: int) -> torch.Tensor:
    """Group spatial positions by their residue in a strided square lattice."""
    batch, channels, height, width = image.shape
    if height != width or height % side:
        raise ValueError("Only divisible square grids are supported")
    stride = height // side
    return (image.reshape(batch, channels, side, stride, side, stride)
            .permute(0, 3, 5, 2, 4, 1).reshape(batch * stride * stride, side * side, channels))


def lattice_image(tokens: torch.Tensor, side: int, height: int) -> torch.Tensor:
    stride = height // side
    channels = tokens.shape[-1]
    return (tokens.reshape(-1, stride, stride, side, side, channels)
            .permute(0, 5, 3, 1, 4, 2).reshape(-1, channels, height, height))


class TensorEvaluator:
    """Inference-only tensor interpreter. Every required parameter must exist."""

    def __init__(self, tensors: dict[str, torch.Tensor]):
        self.tensors = tensors
        self.used: set[str] = set()

    def p(self, key: str) -> torch.Tensor:
        self.used.add(key)
        return self.tensors[key]

    def linear(self, x, name):
        bias = self.p(name + ".bias") if name + ".bias" in self.tensors else None
        return F.linear(x, self.p(name + ".weight"), bias)

    def conv(self, x, name, *, stride=1, depthwise=False):
        weight = self.p(name + ".weight")
        return F.conv2d(x, weight, self.p(name + ".bias"), stride=stride,
                        padding=weight.shape[-1] // 2 if stride == 1 else 0,
                        groups=x.shape[1] if depthwise else 1)

    def norm(self, x, name, *, eps=1e-5):
        return F.layer_norm(x, (x.shape[-1],), self.p(name + ".weight"), self.p(name + ".bias"), eps)

    def image_norm(self, x, name, *, eps=1e-6):
        return self.norm(x.permute(0, 2, 3, 1), name, eps=eps).permute(0, 3, 1, 2).contiguous()

    def batch_norm(self, x, name):
        return F.batch_norm(x, self.p(name + ".running_mean"), self.p(name + ".running_var"),
                            self.p(name + ".weight"), self.p(name + ".bias"), training=False, eps=1e-5)

    def forensic_features(self, x):
        outputs = []
        for stage, count in enumerate((5, 8, 20, 7), start=1):
            prefix = f"encoder.patch_embed{stage}"
            x = self.image_norm(self.conv(x, prefix + ".proj", stride=4 if stage == 1 else 2),
                                prefix + ".norm", eps=1e-5)
            for index in range(count):
                prefix = f"encoder.blocks{stage}.{index}"
                if stage <= 2:
                    x = x + self.conv(x, prefix + ".pos_embed", depthwise=True)
                    y = self.conv(self.batch_norm(x, prefix + ".norm1"), prefix + ".conv1")
                    x = x + self.conv(self.conv(y, prefix + ".attn", depthwise=True), prefix + ".conv2")
                    y = self.conv(self.batch_norm(x, prefix + ".norm2"), prefix + ".mlp.fc1")
                    x = x + self.conv(F.gelu(y), prefix + ".mlp.fc2")
                else:
                    size = (4, 8, 16, 32)[index // 5] if stage == 3 else (8, 16)[index // 4]
                    x = self.sparse_block(x, prefix, size)
                if stage == 3 and index in (4, 9, 14) or stage == 4 and index == 3:
                    outputs.append(x)
            if stage >= 3:
                outputs.append(self.image_norm(x, f"encoder.norm{stage}"))
        return outputs

    def sparse_block(self, image, name, side):
        height = image.shape[-1]
        x = self.norm(lattice_tokens(image, side), name + ".norm1", eps=1e-6)
        batch, length, channels = x.shape
        qkv = self.linear(x, name + ".attn.qkv").reshape(batch, length, 3, channels // 64, 64)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = ((q @ k.transpose(-1, -2)) * (64 ** -0.5)).softmax(-1)
        x = (attention @ v).transpose(1, 2).reshape(batch, length, channels)
        x = lattice_image(self.linear(x, name + ".attn.proj"), side, height)
        image = image + self.p(name + ".gamma_1")[None, :, None, None] * x
        x = self.linear(self.norm(image.permute(0, 2, 3, 1), name + ".norm2", eps=1e-6), name + ".mlp.fc1")
        x = self.conv(x.permute(0, 3, 1, 2), name + ".mlp.dwconv.dwconv", depthwise=True)
        x = self.linear(F.gelu(x.permute(0, 2, 3, 1)), name + ".mlp.fc2").permute(0, 3, 1, 2)
        return image + self.p(name + ".gamma_2")[None, :, None, None] * x

    def fusion_map(self, features):
        result = None
        size = features[0].shape[-2:]
        for i, feature in enumerate(features, start=1):
            if i <= 4:
                feature = self.conv(feature, f"lmu.conv_layer{i}")
            else:
                feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
            weighted = feature * self.p(f"lmu.gamma{i}")[None, :, None, None]
            result = weighted if result is None else result + weighted
        return self.image_norm(result, "lmu.norm")

    def pool(self, x, name):
        # Functional PyTorch MHA uses sequence-first layout and its documented
        # packed projection format; eval removes dropout from the saved module.
        query = self.p(name + ".latents").expand(x.shape[0], -1, -1).transpose(0, 1)
        values = x.transpose(0, 1)
        output, _ = F.multi_head_attention_forward(
            query, values, values, x.shape[-1], 8,
            self.p(name + ".mha.in_proj_weight"), self.p(name + ".mha.in_proj_bias"),
            None, None, False, 0.0, self.p(name + ".mha.out_proj.weight"),
            self.p(name + ".mha.out_proj.bias"), training=False, need_weights=True)
        return self.linear(self.norm(output.transpose(0, 1), name + ".norm"), name + ".out_proj")

    def gated_ffn(self, x, name):
        value, gate = self.linear(x, name + ".fc").chunk(2, dim=-1)
        return self.linear(value * F.silu(gate), name + ".proj")

    def mixture(self, x):
        shape = x.shape
        tokens = x.reshape(-1, shape[-1])
        router = self.linear(tokens, "moe_ffn.router")
        chosen = router.topk(2, dim=-1).indices
        weights = router.softmax(-1).gather(-1, chosen)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)
        output = torch.zeros_like(tokens)
        for expert in range(8):
            rows, slots = torch.where(chosen == expert)
            if rows.numel():
                transformed = self.gated_ffn(tokens[rows], f"moe_ffn.experts.{expert}.ffn")
                output.index_add_(0, rows, transformed * weights[rows, slots, None])
        return output.reshape(shape)

    def shared_attention(self, x):
        q, k, v = self.linear(self.norm(x, "self_attn.norm"), "self_attn.qkv").chunk(3, dim=-1)
        q, k = self.norm(q, "self_attn.q_norm"), self.norm(k, "self_attn.k_norm")
        shape = (x.shape[0], x.shape[1], 8, 64)
        attended = F.scaled_dot_product_attention(
            q.reshape(shape).transpose(1, 2), k.reshape(shape).transpose(1, 2),
            v.reshape(shape).transpose(1, 2), dropout_p=0.0)
        return x + self.linear(attended.transpose(1, 2).reshape_as(x), "self_attn.out_proj")

    def finish(self, forensic, semantic):
        forensic_tokens = self.pool(forensic.flatten(2).transpose(1, 2), "svit_pooling")
        semantic_tokens = self.conv(semantic, "convnext_proj").flatten(2).transpose(1, 2)
        tokens = torch.cat((forensic_tokens, semantic_tokens), dim=1)
        tokens = self.norm(tokens + self.mixture(self.norm(tokens, "concat_norm2")), "post_moe_norm")
        tokens = self.norm(self.shared_attention(tokens), "pre_pool_norm")
        grid = semantic.shape[-1]
        context = tokens[:, forensic_tokens.shape[1]:].transpose(1, 2).reshape(1, 512, grid, grid)
        context = F.interpolate(context, forensic.shape[-2:], mode="bilinear", align_corners=False)
        mask = torch.cat((forensic, context), dim=1)
        for conv_index, norm_index in ((0, 1), (3, 4)):
            mask = self.conv(mask, f"joint_seg_decoder.{conv_index}")
            prefix = f"joint_seg_decoder.{norm_index}"
            mask = F.gelu(F.group_norm(mask, 32, self.p(prefix + ".weight"), self.p(prefix + ".bias"), eps=1e-5))
        mask = self.conv(mask, "joint_seg_decoder.6")
        mask = F.interpolate(mask, (512, 512), mode="bilinear", align_corners=False).sigmoid()
        pooled = self.pool(tokens, "final_attn_pool").squeeze(1)
        detection = self.linear(self.gated_ffn(pooled, "head.0"), "head.1").softmax(-1)[:, 1]
        return mask, detection



class FusedModel:
    """One leased composite model; dropping this object releases both branches.

    `predict` accepts exactly one already-oriented PIL image. Batch inference,
    automatic EXIF transforms and native tile inference are intentionally absent.
    """

    def __init__(self, ops: TensorEvaluator, backbone: torch.nn.Module, device: torch.device):
        self.ops, self.backbone, self.device = ops, backbone, device

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> FusedPrediction:
        if not isinstance(image, Image.Image) or image.width <= 0 or image.height <= 0:
            raise ValueError("Pass exactly one nonempty PIL image")
        source_size = image.size
        rgb = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        pixels = pixels.to(device=self.device, dtype=torch.float32)
        mean = pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        semantic_input = (pixels - self.ops.p("clip_mean")) / self.ops.p("clip_std")
        semantic = self.backbone.forward_features(semantic_input)
        forensic = self.ops.fusion_map(self.ops.forensic_features((pixels - mean) / std))
        mask, score = self.ops.finish(forensic, semantic)
        if mask.shape != (1, 1, INPUT_SIZE, INPUT_SIZE) or score.shape != (1,):
            raise RuntimeError("Inference output shape differs from the fixed single-image contract")
        if not torch.isfinite(mask).all() or not torch.isfinite(score).all():
            raise RuntimeError("Inference produced non-finite output")
        if bool((mask < 0).any()) or bool((mask > 1).any()):
            raise RuntimeError("Inference produced values outside the raw response range")
        image_score = float(score.item())
        candidate_eligibility(image_score)  # fail closed without zeroing the raw map
        raw_mask = mask[0, 0].float().cpu().numpy().copy()
        raw_mask.setflags(write=False)
        return FusedPrediction(raw_mask, image_score, source_size)


def load_model(checkpoint: Path, backbone_checkpoint: Path, device: str, dtype: str = "float32") -> FusedModel:
    """Load only the two pinned local files, without importing upstream code."""
    target = _execution_target(device, dtype)
    checkpoint, backbone_checkpoint = Path(checkpoint), Path(backbone_checkpoint)
    # Both files are checked before any pickle is inspected or deserialized.
    _check_hash(checkpoint, CHECKPOINT_SHA256)
    _check_hash(backbone_checkpoint, BACKBONE_SHA256)
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):
        raise ValueError("Checkpoint requires unsupported pickle globals")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = _validated_state(saved)

    import timm
    from safetensors.torch import load_file

    # Meta construction avoids a second multi-GiB allocation and never downloads.
    with torch.device("meta"):
        backbone = timm.create_model(BACKBONE_ARCHITECTURE, pretrained=False, num_classes=0)
    backbone_state = load_file(str(backbone_checkpoint), device="cpu")
    # These exact classification tensors are not called by forward_features.
    for key in ("head.fc.weight", "head.fc.bias"):
        del backbone_state[key]
    backbone.load_state_dict(backbone_state, strict=True, assign=True)
    backbone.eval().requires_grad_(False).to(device=target, dtype=torch.float32)
    ops = TensorEvaluator({key: value.detach().to(device=target, dtype=torch.float32)
                           for key, value in state.items()})
    return FusedModel(ops, backbone, target)
