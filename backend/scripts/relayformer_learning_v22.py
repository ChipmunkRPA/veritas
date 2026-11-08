"""Differentiable regional adapter and authored backward probe; no source study."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from scripts.relayformer_native_v18 import Native, prepare, tensor_sha


def need(ok, message):
    if not ok:
        raise ValueError(message)


def trainable_name(name):
    """The decoder's unused final_conv and unused block0 global path stay frozen."""
    if name.startswith("mask_decoder."):
        return not name.startswith("mask_decoder.final_conv.")
    match = re.fullmatch(r"vit\.blocks\.([0-9]+)\.(?:attn\.(?:qkv|proj)_lora|mlp\.lora[12])_(loc|glo)\.lora_[AB]", name)
    return match is not None and (match[2] == "loc" or int(match[1]) > 0)


def configure(model):
    # Eval mode disables stochastic dropout, not differentiation. This also keeps
    # the exact inference reconstruction path used by the released checkpoint.
    model.eval().requires_grad_(False)
    parameters = {}
    for name, parameter in model.named_parameters():
        if trainable_name(name):
            parameter.requires_grad_(True)
            parameters[name] = parameter
    need(parameters and any("_loc." in n for n in parameters) and any("_glo." in n for n in parameters), "Both used backbone adaptation paths required")
    need(any(n.startswith("mask_decoder.") for n in parameters), "Mask decoder must learn alongside the backbone")
    return parameters


def logits(model, tensor, geometry):
    """Follow the native differentiable graph, returning pre-sigmoid valid logits."""
    fw, fh = geometry["fit_size"]
    need(tensor.shape == (1, 3, 1024, 1024) and tensor.dtype == torch.float32 and tensor_sha(tensor) == geometry["input_float32_sha256"], "Exact source-bound native tensor required")
    origin = torch.tensor([[fh, fw]], dtype=torch.int64)
    lengths = torch.tensor([1], dtype=torch.int64)
    divided = model.divide_patches(tensor)
    valid, _maximum, grids = model.get_valid_patches_info(origin, 1024, model.grid_size, clip_len=lengths)
    patches = model.select_valid_patches(valid, divided)
    gh, gw = (2 * fh + 1023) // 1024, (2 * fw + 1023) // 1024
    expected = torch.stack([tensor[0, :, r * 496:r * 496 + 528, c * 496:c * 496 + 528] for r in range(gh) for c in range(gw)])
    need(len(patches) == 1 and torch.equal(patches[0], expected), "Differentiable patch input differs from native rectangles")
    features, _unused = model.forward_features(patches, valid, grids, clip_len_list=lengths)
    canvas = model.assemble_and_decode(features[:, model.tpp:], tensor, valid, grids, model.grid_size)
    need(canvas.shape == (1, 1, 1024, 1024) and canvas.dtype == torch.float32 and bool(torch.isfinite(canvas).all()), "Finite full native logits required")
    return canvas[:, :, :fh, :fw], {"actual_patch_count": gh * gw, "actual_patch_float32_sha256": tensor_sha(expected), "full_logit_float32_sha256": tensor_sha(canvas)}


def regional_loss(values, target):
    """Equal positive/negative pixel terms, plus a soft overlap term for edits."""
    need(values.shape == target.shape and values.ndim == 4 and values.shape[:2] == (1, 1), "One matching spatial logit/target field required")
    need(bool(torch.isfinite(values).all()) and bool(torch.isfinite(target).all()) and bool(((target == 0) | (target == 1)).all()), "Finite logits and explicit binary targets required")
    positive = target.bool()
    terms = F.binary_cross_entropy_with_logits(values, target, reduction="none")
    if bool(positive.any()):
        need(bool((~positive).any()), "An edited training view must contain background as well")
        bce = (terms[positive].mean() + terms[~positive].mean()) / 2
        response = values.sigmoid()
        dice = 1 - (2 * (response * target).sum() + 1) / (response.sum() + target.sum() + 1)
        return bce + dice / 2
    return terms.mean()


def authored_image():
    y, x = np.mgrid[:1024, :1024]
    rgb = np.stack([(x * 7 + y * 3) % 256, (x // 9 * 17 + y * 5) % 256, (x * 11 + y // 7 * 19) % 256], axis=2).astype(np.uint8)
    target = np.zeros((1, 1, 1024, 1024), np.float32)
    target[:, :, 201:739, 307:813] = 1
    return Image.fromarray(rgb), torch.from_numpy(target)


def sha_state(parameters):
    return {name: tensor_sha(value) for name, value in parameters.items()}


def probe(folder):
    folder.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20261108)
    native = Native()
    model = native.model
    parameters = configure(model)
    frozen = {name: p for name, p in model.named_parameters() if name not in parameters}
    frozen_before = sha_state(frozen)
    initial = {name: p.detach().clone() for name, p in parameters.items()}
    initial_hashes = sha_state(initial)
    tensor, geometry = prepare(authored_image()[0])
    target = authored_image()[1]
    passes = []
    for repeat in range(2):
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(initial[name])
        model.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW(list(parameters.values()), lr=1e-4, weight_decay=0.01)
        values, binding = logits(model, tensor, geometry)
        with torch.no_grad():
            canonical = model(image=tensor, origin_shape=torch.tensor([[1024, 1024]]), clip_len=torch.tensor([1]))
            need(torch.equal(values.sigmoid(), canonical), "Differentiable logits differ from the released inference graph")
        loss = regional_loss(values, target)
        loss.backward()
        gradients = {name: p.grad for name, p in parameters.items() if p.grad is not None}
        need(set(gradients) == set(parameters) and all(bool(torch.isfinite(g).all()) for g in gradients.values()), "Every selected tensor needs a finite gradient in this full four-patch probe")
        need(all(p.grad is None for p in frozen.values()), "Frozen base parameters must receive no gradients")
        nonzero = {name for name, g in gradients.items() if bool(g.ne(0).any())}
        need(any("_loc." in n for n in nonzero) and any("_glo." in n for n in nonzero) and any(n.startswith("mask_decoder.") for n in nonzero), "Backbone local/global paths and decoder need actual nonzero gradients")
        gradient_hashes = sha_state(gradients)
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 1, error_if_nonfinite=True)
        optimizer.step()
        after = sha_state(parameters)
        need(any(after[n] != initial_hashes[n] for n in nonzero), "An optimizer step must actually change learned tensors")
        need(sha_state(frozen) == frozen_before, "Dense base or unused path changed")
        passes.append({"loss": float(loss.detach()), "gradient_norm_before_clipping": float(norm), "binding": binding, "gradient_sha256": gradient_hashes, "updated_sha256": after})
        print(json.dumps({"authored_backward_repeat": repeat, "loss": float(loss.detach()), "gradient_tensors": len(gradients), "nonzero_gradient_tensors": len(nonzero)}), flush=True)
        del optimizer, values, loss, canonical, gradients
    need(passes[0] == passes[1], "Independent reset to the released starting tensors must repeat logits, gradients and update bits exactly")
    for name, p in parameters.items():
        with torch.no_grad():
            p.copy_(initial[name])
    need(sha_state(parameters) == initial_hashes and sha_state(frozen) == frozen_before, "Authored probe must restore every starting tensor")
    result = {"schema": "chipmunk.regional-backward-authored-probe/22", "authored_model_forwards": 4, "authored_backward_optimizer_steps": 2, "project_source_forwards": 0, "project_fitting_steps": 0, "selected_tensors": len(parameters), "selected_parameters": sum(p.numel() for p in parameters.values()), "dense_base_unchanged": True, "restored_starting_tensors": True, "native_sigmoid_parity_exact": True, "reset_gradient_and_update_repeat_exact": True, "passes": passes, "serving_admitted": False, "scope": "An authored mathematical target tests differentiability, native graph parity, frozen-base preservation and exact repeat only. It is not an AI example, a quality test, training admission or customer confidence."}
    with (folder / "probe.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "passes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_folder", type=Path)
    args = parser.parse_args()
    probe(args.output_folder)
