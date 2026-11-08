"""Strict offline upstream graph with full-source image geometry; research only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import socket
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scripts.inspect_relayformer_v18 import ASSETS, ROOT, load_state
from scripts.prepare_relayformer_v18 import digest

RUNTIME = ROOT / "backend/data/independent-evaluation/relayformer-v18-runtime/packages"
MAP_BOUND = 0.0003


def require(value, reason):
    if not value:
        raise ValueError(reason)


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def prepare(image):
    require(
        isinstance(image, Image.Image) and image.mode == "RGB",
        "Already oriented explicit RGB source required",
    )
    w, h = image.size
    require(min(w, h) > 0 and image.getexif().get(274, 1) == 1, "Nonempty upright source required")
    fw, fh = (w, h) if max(w, h) <= 1024 else (max(1, w * 1024 // max(w, h)), max(1, h * 1024 // max(w, h)))
    fit = image if (fw, fh) == (w, h) else image.resize((fw, fh), Image.Resampling.BILINEAR)
    canvas = np.zeros((1024, 1024, 3), np.uint8)
    canvas[:fh, :fw] = np.asarray(fit)
    mean, std = np.asarray([0.485, 0.456, 0.406], np.float32), np.asarray([0.229, 0.224, 0.225], np.float32)
    values = (canvas.astype(np.float32) / np.float32(255) - mean) / std
    reference = (canvas.astype(np.float64) / 255 - np.asarray([0.485, 0.456, 0.406])) / np.asarray(
        [0.229, 0.224, 0.225]
    )
    error = float(np.max(np.abs(values - reference)))
    require(error <= 1e-6, "ImageNet normalization reference differs")
    tensor = torch.from_numpy(np.ascontiguousarray(values.transpose(2, 0, 1)[None]))
    geometry = {
        "source_size": [w, h],
        "fit_size": [fw, fh],
        "canvas_size": [1024, 1024],
        "source_rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
        "input_float32_sha256": tensor_sha(tensor),
        "normalization_reference_maximum_difference": error,
        "source_cropped": False,
        "padding_position": "top_left",
        "padding_rgb": [0, 0, 0],
        "resize": "Only if larger than1024; bilinear, aspect preserved with integer floor sizes",
        "readout": "Nearest source-center index, exact integer ratios; retain float32 response bits, exclude padding",
        "response_meaning": "Uncalibrated localizer response, not pixel-authorship probability",
    }
    return tensor, geometry


def source_map(output, geometry):
    w, h = geometry["source_size"]
    fw, fh = geometry["fit_size"]
    require(
        output.shape == (1, 1, 1024, 1024)
        and output.dtype == torch.float32
        and bool(torch.isfinite(output).all()),
        "Complete finite native response canvas required",
    )
    values = output[0, 0, :fh, :fw].detach().cpu().numpy()
    require(float(values.min()) >= 0 and float(values.max()) <= 1, "Native sigmoid response range required")
    rows = np.minimum(fh - 1, ((2 * np.arange(h, dtype=np.int64) + 1) * fh) // (2 * h))
    cols = np.minimum(fw - 1, ((2 * np.arange(w, dtype=np.int64) + 1) * fw) // (2 * w))
    return values[rows[:, None], cols[None, :]].copy()


def upstream_class():
    require(RUNTIME.is_dir(), "Separate pinned rotary runtime required")
    sys.path.insert(0, str(RUNTIME))
    import rotary_embedding_torch

    require(
        Path(rotary_embedding_torch.__file__).is_relative_to(RUNTIME)
        and importlib.metadata.version("rotary-embedding-torch") == "0.8.9",
        "Pinned isolated helper required",
    )
    require("IMDLBenCo" not in sys.modules, "Registry shim is allowed only in an isolated research process")
    registry_calls = []

    class Registry:
        def register_module(self):
            def identity(cls):
                registry_calls.append(cls.__name__)
                return cls

            return identity

    package, registry = types.ModuleType("IMDLBenCo"), types.ModuleType("IMDLBenCo.registry")
    registry.MODELS = Registry()
    sys.modules["IMDLBenCo"], sys.modules["IMDLBenCo.registry"] = package, registry
    namespace = types.ModuleType("_chipmunk_relay_v18")
    namespace.__path__ = [str(ASSETS / "upstream/models")]
    sys.modules[namespace.__name__] = namespace
    name = namespace.__name__ + ".RelayFormer"
    spec = importlib.util.spec_from_file_location(name, ASSETS / "upstream/models/RelayFormer.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(
        registry_calls == ["RelayFormer"], "Only the inspected identity class-registration path is permitted"
    )
    return module.RelayFormer


class Native:
    def __init__(self):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        state, _, _ = load_state()
        base = upstream_class()
        self.skipped_base_loads = 0
        owner = self

        class FullCheckpoint(base):
            def _load_vit_checkpoint(self, path):
                require(
                    path == "full-checkpoint-loaded-strictly-below", "No base-checkpoint fallback allowed"
                )
                owner.skipped_base_loads += 1

        import timm

        original_create, original_connect = timm.create_model, socket.socket.connect
        creations = []

        def create(*args, **kwargs):
            require(kwargs.get("pretrained") is False, "Implicit base download prohibited")
            creations.append({"model": args[0], "pretrained": False})
            return original_create(*args, **kwargs)

        def denied(*_args, **_kwargs):
            raise ValueError("Network access prohibited during native graph construction")

        timm.create_model, socket.socket.connect = create, denied
        try:
            self.model = FullCheckpoint(vit_pretrain_path="full-checkpoint-loaded-strictly-below")
        finally:
            timm.create_model, socket.socket.connect = original_create, original_connect
        template = self.model.state_dict()
        require(set(template) == set(state), "Released checkpoint and upstream graph fields differ")
        require(
            all(template[k].shape == v.shape and template[k].dtype == v.dtype for k, v in state.items()),
            "Released checkpoint/graph shape or dtype differs",
        )
        self.model.load_state_dict(state, strict=True)
        require(
            all(torch.equal(self.model.state_dict()[k], v) for k, v in state.items()),
            "Every native tensor must exactly match the released checkpoint",
        )
        self.model.eval().requires_grad_(False)
        require(self.skipped_base_loads == 1 and len(creations) == 1, "Unexpected constructor fallback")
        self.parameters = sum(p.numel() for p in self.model.parameters())
        self.loaded_tensors = len(state)
        self.attempted_forwards = self.completed_forwards = 0
        self.merged = False

    @torch.inference_mode()
    def predict(self, tensor, geometry):
        fw, fh = geometry["fit_size"]
        require(
            tensor.shape == (1, 3, 1024, 1024)
            and tensor.dtype == torch.float32
            and tensor_sha(tensor) == geometry["input_float32_sha256"],
            "Exact prepared tensor required",
        )
        origin = torch.tensor([[fh, fw]], dtype=torch.int64)
        length = torch.tensor([1], dtype=torch.int64)
        gh, gw = (2 * fh + 1023) // 1024, (2 * fw + 1023) // 1024
        expected = torch.stack(
            [
                tensor[0, :, r * 496 : r * 496 + 528, c * 496 : c * 496 + 528]
                for r in range(gh)
                for c in range(gw)
            ]
        )
        calls = []

        def bind(_module, args):
            require(
                torch.equal(args[0], expected), "Actual native patch tensors differ from full-source geometry"
            )
            calls.append(tensor_sha(args[0]))

        hook = self.model.vit.patch_embed.register_forward_pre_hook(bind)
        self.attempted_forwards += 1
        try:
            output = self.model(image=tensor, origin_shape=origin, clip_len=length)
            self.completed_forwards += 1
        finally:
            hook.remove()
        require(
            len(calls) == 1 and tensor_sha(tensor) == geometry["input_float32_sha256"],
            "Native input binding incomplete",
        )
        return source_map(output, geometry), {
            "actual_patch_count": gh * gw,
            "actual_patch_float32_sha256": calls[0],
            "complete_canvas_float32_sha256": tensor_sha(output),
            "merged_lora": self.merged,
        }

    def merge(self):
        require(not self.merged, "Only one explicit LoRA merge permitted")
        self.model.merge_lora()
        self.merged = True


def smoke():
    from scripts.image_imdet_native_v17 import fixture

    native = Native()
    values = []
    retained = []
    for size in ((515, 389), (1401, 773)):
        tensor, geometry = prepare(fixture(*size))
        a, proof = native.predict(tensor, geometry)
        b, repeat = native.predict(tensor, geometry)
        require(np.array_equal(a, b) and proof == repeat, "Canonical native repeat must be exact")
        values.append(
            {
                "geometry": geometry,
                "native": proof,
                "source_map_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                "canonical_repeat_exact": True,
            }
        )
        retained.append((tensor, geometry, a))
    native.merge()
    for row, (tensor, geometry, canonical) in zip(values, retained, strict=True):
        merged, proof = native.predict(tensor, geometry)
        difference = float(np.max(np.abs(merged - canonical)))
        require(difference <= MAP_BOUND, "Prospective merged/unmerged response bound failed")
        row.update(merged_native=proof, merged_unmerged_maximum_difference=difference)
    return {
        "schema": "chipmunk.relayformer-authored-native/18",
        "authored_sources": 2,
        "records": values,
        "complete_checkpoint_tensors": native.loaded_tensors,
        "parameters": native.parameters,
        "attempted_model_forwards": native.attempted_forwards,
        "completed_model_forwards": native.completed_forwards,
        "project_media_forwards": 0,
        "detection_quality_evaluated": False,
        "map_difference_bound": MAP_BOUND,
        "serving_admitted": False,
        "customer_confidence_changed": False,
        "implementation_sha256": digest(Path(__file__)),
        "scope": "Strict full pretrained graph and authored geometry/execution only; no real edit detection or accuracy result.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), "New authored receipt required")
    value = smoke()
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "authored_sources": 2,
                "actual_model_forwards": value["completed_model_forwards"],
                "project_media_forwards": 0,
                "serving_admitted": False,
            }
        ),
        flush=True,
    )
