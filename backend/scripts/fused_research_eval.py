"""Isolated FUSED checkpoint compatibility research; never a product loader.

This functional evaluator expresses tensor operations for the published OpenSDI
checkpoint. It does not import, execute, or vendor the upstream Python package.
Architecture inspected at AntonNuzhdin/FUSED e8580ebad33decf0787a96133af4fafee640e024.
Upstream source has no LICENSE file at that revision. Keep this experiment out of
the application until licensing and numerical/benchmark validation are resolved.
Training, alternate configurations and arbitrary checkpoint paths are unsupported.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "backend/data/localization-eval"
WEIGHTS = REPO / "models/weights/fused"
HASHES = {
    "fused_opensdi.pth": "7e527d49080a773641cf96915b4b3fa12d571c5b87f35c527f13ab82f148364c",
    "convnext.safetensors": "1ddf6dbc7f0753fc4dfec5db5fa474acfd1ac6f67f466d6a976780c30b3a1b23",
}
SOURCE_REVISION = "e8580ebad33decf0787a96133af4fafee640e024"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


class FusedResearch:
    def __init__(self, device="cpu", dtype=torch.float32):
        import timm
        from safetensors.torch import load_file

        print("Verifying pinned weight hashes", flush=True)
        for name, expected in HASHES.items():
            if file_sha(WEIGHTS / name) != expected:
                raise ValueError(f"Unexpected checkpoint bytes: {name}")
        unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(WEIGHTS / "fused_opensdi.pth")
        if unsafe:
            raise ValueError(f"Unsafe checkpoint globals: {unsafe}")
        saved = torch.load(WEIGHTS / "fused_opensdi.pth", map_location="cpu", weights_only=True, mmap=True)
        self.config = saved["cfg"]["model"]
        expected_config = {"dim": 512, "num_heads": 8, "num_latents": 8, "img_size": 512,
                           "depth": [5, 8, 20, 7], "ffn_type": "moe", "seg_bridge": True,
                           "enable_sparsevit_branch": True, "enable_convnext_branch": True}
        if any(self.config.get(key) != value for key, value in expected_config.items()):
            raise ValueError("This evaluator supports only the pinned OpenSDI architecture")
        self.device, self.dtype = device, dtype
        self.ops = TensorEvaluator({k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
                                    for k, v in saved["state_dict"].items()})
        print("Constructing licensed timm backbone on meta device", flush=True)
        with torch.device("meta"):
            self.backbone = timm.create_model("convnext_xxlarge.clip_laion2b_soup", pretrained=False, num_classes=0)
        backbone_state = load_file(str(WEIGHTS / "convnext.safetensors"), device="cpu")
        # The downloaded classification head is explicitly unused by forward_features.
        for key in ("head.fc.weight", "head.fc.bias"):
            del backbone_state[key]
        self.backbone.load_state_dict(backbone_state, strict=True, assign=True)
        self.backbone.eval().requires_grad_(False).to(device=device, dtype=dtype)
        del backbone_state, saved
        print("Model loaded", flush=True)

    @torch.inference_mode()
    def predict(self, image: Image.Image):
        image = image.convert("RGB").resize((512, 512), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)[None]
        pixels = pixels.to(device=self.device, dtype=self.dtype)
        mean = pixels.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = pixels.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        semantic_input = (pixels - self.ops.p("clip_mean")) / self.ops.p("clip_std")
        semantic = self.backbone.forward_features(semantic_input)
        forensic = self.ops.fusion_map(self.ops.forensic_features((pixels - mean) / std))
        mask, score = self.ops.finish(forensic, semantic)
        if not torch.isfinite(mask).all() or not torch.isfinite(score).all():
            raise RuntimeError("Inference produced non-finite output")
        return mask[0, 0].float().cpu().numpy(), float(score.item())


def evaluate(args):
    manifest_path = Path(getattr(args, "manifest", DATA / "manifest.json")).resolve()
    source_root = manifest_path.parent
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    policy_sha256 = manifest.get("evaluation_policy_sha256")
    if policy_sha256:
        policy_bytes = (source_root / "evaluation-policy.json").read_bytes()
        if sha(policy_bytes) != policy_sha256:
            raise ValueError("The predeclared evaluation policy changed")
        policy = json.loads(policy_bytes)
        if (policy.get("mask_threshold"), policy.get("image_gate_threshold"), policy.get("threshold_tuning")) != (0.5, 0.5, False):
            raise ValueError("The manifest policy does not match this fixed-threshold evaluator")
    production_policy = source_root / "production-grid-policy.json"
    production_policy_sha256 = sha(production_policy.read_bytes()) if production_policy.exists() else None
    # First N examples from each existing family/label group. No score-based selection.
    selected, counts = [], {}
    for item in manifest["items"]:
        group = (item["family"], item["label"])
        if counts.get(group, 0) < args.per_group:
            selected.append(item)
            counts[group] = counts.get(group, 0) + 1
    if args.max_items:
        selected = selected[:args.max_items]
    output = (source_root / args.output).resolve()
    if output == source_root or not output.is_relative_to(source_root):
        raise ValueError("Output must be a new directory below the selected manifest directory")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    model = FusedResearch(args.device, torch.float32 if args.dtype == "float32" else torch.float16)
    result = {"purpose": "isolated research; not a product admission or proof of editing",
              "source_revision": SOURCE_REVISION, "weights_sha256": HASHES,
              "manifest_sha256": sha(manifest_bytes), "manifest_path": str(manifest_path),
              "evaluation_policy_sha256": policy_sha256, "device": args.device, "dtype": args.dtype,
              "production_grid_policy_sha256": production_policy_sha256,
              "torch_version": torch.__version__, "threshold": 0.5,
              "protocol": "RGB PIL bilinear 512 square; ImageNet forensic / checkpoint CLIP semantic normalization; "
                          "float32 bilinear mask projection to stored pixels; fixed 0.5 threshold; no EXIF rotation; no tuning",
              "selected_items": [item["id"] for item in selected], "rows": [],
              "load_seconds": time.monotonic() - started,
              "limitations": ["Source license unresolved", "Implementation parity is tracked separately and does not establish accuracy",
                              "Small deterministic convenience sample; not representative accuracy"]}
    for item in selected:
        data = (source_root / item["image"]).read_bytes()
        if sha(data) != item["image_sha256"]:
            raise ValueError("Source hash mismatch")
        with Image.open(io.BytesIO(data)) as original:
            original = original.convert("RGB")
            width, height = original.size
        truth = np.zeros((height, width), bool)
        if item["label"]:
            mask_data = (source_root / item["mask"]).read_bytes()
            if sha(mask_data) != item["mask_sha256"]:
                raise ValueError("Reference mask hash mismatch")
            with Image.open(io.BytesIO(mask_data)) as reference:
                if reference.size != original.size:
                    raise ValueError("Reference coordinate mismatch")
                truth = np.asarray(reference.convert("L")) > 127
        started = time.monotonic()
        native_values, score = model.predict(original)
        if native_values.shape != (512, 512) or native_values.dtype != np.float32:
            raise ValueError("Expected the original 512-square float32 probability map")
        values = F.interpolate(torch.from_numpy(native_values)[None, None], (height, width),
                               mode="bilinear", align_corners=False)[0, 0].numpy()
        chosen = values >= 0.5
        tp, fp, fn = int((chosen & truth).sum()), int((chosen & ~truth).sum()), int((~chosen & truth).sum())
        row = {"id": item["id"], "family": item["family"], "label": item["label"], "score": score,
               "seconds": time.monotonic() - started, "intersection": tp, "false_positive_pixels": fp,
               "false_negative_pixels": fn, "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
               "pixel_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
               "flagged_fraction": float(chosen.mean()), "max_probability": float(values.max()),
               "mean_probability": float(values.mean())}
        native_path = output / (item["id"] + ".native-probabilities.npy")
        projected_path = output / (item["id"] + ".probabilities.npy")
        np.save(native_path, native_values, allow_pickle=False)
        np.save(projected_path, values, allow_pickle=False)
        row["probabilities"] = {
            "native": {"path": native_path.name, "sha256": file_sha(native_path), "shape": list(native_values.shape)},
            "source": {"path": projected_path.name, "sha256": file_sha(projected_path), "shape": list(values.shape)}}
        Image.fromarray(chosen.astype(np.uint8) * 255).save(output / (item["id"] + ".mask.png"))
        result["rows"].append(row)
        result["unused_checkpoint_keys"] = sorted(set(model.ops.tensors) - model.ops.used)
        if args.device == "mps":
            result["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
            result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        (output / "results.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(row), flush=True)
    groups = {}
    for family in sorted({r["family"] for r in result["rows"]}):
        edited = [r for r in result["rows"] if r["family"] == family and r["label"]]
        real = [r for r in result["rows"] if r["family"] == family and not r["label"]]
        groups[family] = {"edited_n": len(edited), "authentic_n": len(real),
                          "edited_mean_iou": float(np.mean([r["iou"] for r in edited])) if edited else None,
                          "edited_mean_f1": float(np.mean([r["pixel_f1"] for r in edited])) if edited else None,
                          "authentic_with_any_flagged_pixel": sum(r["flagged_fraction"] > 0 for r in real)}
    result["groups"] = groups
    (output / "results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(groups, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--per-group", type=int, default=1)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--manifest", type=Path, default=DATA / "manifest.json",
                        help="Pinned source manifest; image/mask paths and output are relative to its directory")
    parser.add_argument("--output", default="fused-research")
    arguments = parser.parse_args()
    if not 1 <= arguments.per_group <= 20 or arguments.max_items is not None and arguments.max_items < 1:
        parser.error("per-group must be 1..20; max-items must be positive")
    evaluate(arguments)
