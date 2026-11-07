"""Strict offline DeMamba research loader and synthetic compatibility review.

No source recordings, decoder, file selection, classifier fitting or admission.
Only inspected definitions from the checksum-pinned publisher source execute.
The full checkpoint includes its fine-tuned vision tower; no Hub calls occur.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import inspect
import json
import math
import time
import typing
from pathlib import Path

from scripts.review_av_candidate_v12 import definitions

REPOSITORY = "Ahsanz/DeMamba-GenVideo100K"
REVISION = "08664b1ea90a1de2bbc1f9a77e3ba0dd1dd4846e"
WEIGHT_SHA = "f89897c72b595aa85a524e59083919cebb0b92161d2a694a2cbffbdd8456ce17"
ASSETS = {
    "README.md": "a522f68fa52dc8dbc12b0fa585846e8f5378823beb0dadd52f5b9eee90ae56b6",
    "eval_demamba_standalone.py": "ffcb27191451e2a901720b9599315a6bd261a8b3261ffbeceafd504e9cee4fa6",
    "XCLIP-README.md": "7e49123d064919c3f64125331915af2ef61691c9e14284e984b1cc1e5e93a92f",
    "DeMamba-LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "Mamba-LICENSE": "011235acc8c9b140ed6013f9b855e82d24c855dbe7f26c4632ab9c9b969add37",
    "VideoX-LICENSE": "5c47310aaa4b66be539a2a85975ea02a900a83083d424b5f3e854931538890c1",
}
SELECTED = {"_npo2", "_pscan_inplace", "pscan", "MambaConfig", "RMSNorm", "MambaBlock", "ResidualBlock", "reorder_data", "XCLIP_DeMamba"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bind_keys(state, expected):
    """Bijective version-prefix mapping only; no dropped/missing tensors."""
    result, changes = {}, []
    for name, tensor in state.items():
        if not name.startswith("module."):
            raise ValueError("Every original DataParallel prefix required")
        key = name.removeprefix("module.")
        options = [key]
        if key.startswith("encoder.vision_model."):
            options.append("encoder." + key.removeprefix("encoder.vision_model."))
        elif key.startswith("encoder."):
            options.append("encoder.vision_model." + key.removeprefix("encoder."))
        matches = [k for k in options if k in expected]
        if len(matches) != 1 or matches[0] in result:
            raise ValueError("Non-bijective or unknown checkpoint tensor")
        result[matches[0]] = tensor
        changes.append((name, matches[0]))
    if set(result) != set(expected):
        raise ValueError("Complete original state required; no initialized fallback")
    return result, changes


def serial_scan(a, x):
    """Independent direct H[t] = A[t] H[t-1] + X[t], not a parallel scan."""
    import torch
    if a.shape != x.shape or x.ndim != 4 or x.shape[1] < 1:
        raise ValueError("Complete matching nonempty recurrence inputs required")
    previous = torch.zeros_like(x[:, 0])
    output = []
    for index in range(x.shape[1]):
        previous = a[:, index] * previous + x[:, index]
        output.append(previous)
    return torch.stack(output, dim=1)


def restore_constant_positions(model):
    """Rebuild the constructor's nonpersistent integer index, never weights."""
    import torch
    embeddings = model.encoder.embeddings
    old = embeddings.position_ids
    if old.shape != (1, 197) or old.dtype != torch.int64 or embeddings.num_positions != 197:
        raise ValueError("Only the inspected constant position-index buffer is supported")
    expected = torch.arange(197, dtype=torch.int64, device="cpu").expand(1, -1)
    if old.is_meta:
        embeddings.position_ids = expected
    elif not torch.equal(old, expected):
        raise ValueError("Changed nonpersistent positions cannot be silently repaired")
    if any(value.is_meta for value in [*model.parameters(), *model.buffers()]):
        raise ValueError("Uninitialized model parameter or buffer")
    return {"name": "encoder.embeddings.position_ids", "shape": [1, 197], "dtype": "int64",
            "rule": "Exact CPU constructor constant arange197; not a trained tensor or missing-state fallback",
            "sha256": hashlib.sha256(expected.numpy().astype("<i8").tobytes()).hexdigest()}


def load(root):
    for name, expected in ASSETS.items():
        if digest(root / name) != expected:
            raise ValueError("Changed declared source/notice: " + name)
    checkpoint = root / "best_acc.pth"
    if checkpoint.stat().st_size != 502986967 or digest(checkpoint) != WEIGHT_SHA:
        raise ValueError("Exact complete published checkpoint required")
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from transformers import XCLIPVisionConfig, XCLIPVisionModel

    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint)
    if unsafe:
        raise ValueError("Checkpoint requests unsafe globals")
    source = (root / "eval_demamba_standalone.py").read_bytes()
    node = next(n.value for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "XCLIP_CFG" for t in n.targets))
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "dict" or node.args:
        raise ValueError("Original constant vision configuration required")
    config = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    namespace = {"__name__": __name__, "math": math, "dataclass": dataclasses.dataclass,
                 "Union": typing.Union, "torch": torch, "nn": nn, "F": functional,
                 "XCLIPVisionModel": XCLIPVisionModel, "XCLIPVisionConfig": XCLIPVisionConfig, "XCLIP_CFG": config}
    definitions(source, SELECTED, namespace)
    with torch.device("meta"):
        model = namespace["XCLIP_DeMamba"]()
    expected = model.state_dict()
    body = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(body, dict) or not isinstance(body.get("model_state_dict"), dict):
        raise ValueError("Original declared checkpoint dictionary required")
    state, names = bind_keys(body["model_state_dict"], expected)
    for name, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype or not torch.isfinite(tensor).all()):
            raise ValueError("Finite shape/type-matched tensor required: " + name)
    model.load_state_dict(state, strict=True, assign=True)
    positions = restore_constant_positions(model)
    model.encoder.set_attn_implementation("eager")
    model.eval()
    if any(value.is_meta for value in model.state_dict().values()):
        raise ValueError("Uninitialized checkpoint state")
    return model, namespace, {"epoch": body.get("epoch"), "tensors": len(state),
        "parameters": sum(p.numel() for p in model.parameters()), "unsafe_globals": unsafe,
        "strict_state_load": True, "renamed_tensor_prefixes": len(names),
        "tensor_name_mapping_sha256": hashlib.sha256(json.dumps(names, separators=(",", ":")).encode()).hexdigest(),
        "constant_buffer_materialization": positions,
        "vision_config": config, "attention_implementation": model.encoder.config._attn_implementation,
        "transformer_source_sha256": digest(inspect.getsourcefile(XCLIPVisionModel))}


def review(root):
    import torch
    import transformers
    torch.set_num_threads(2)
    model, namespace, loaded = load(root)
    generator = torch.Generator().manual_seed(20261107)
    scan_cases = []
    for length in (1, 2, 3, 4, 5, 8, 9, 16, 31):
        a = torch.rand((2, length, 3, 4), generator=generator)
        x = torch.randn((2, length, 3, 4), generator=generator)
        original, direct = namespace["pscan"](a, x), serial_scan(a, x)
        error = float((original - direct).abs().max())
        if not torch.allclose(original, direct, atol=1e-5, rtol=1e-5):
            raise ValueError("Source scan differs from direct recurrence")
        scan_cases.append({"length": length, "max_absolute_difference": error})
    parallel = namespace["pscan"]
    cases = []
    for kind in ("zeros", "seeded_noise", "temporal_ramp"):
        if kind == "zeros":
            data = torch.zeros((1, 8, 3, 224, 224))
        elif kind == "seeded_noise":
            data = torch.randn((1, 8, 3, 224, 224), generator=generator) * .5
        else:
            data = torch.linspace(-1., 1., 8).reshape(1, 8, 1, 1, 1).expand(1, 8, 3, 224, 224).contiguous()
        started = time.monotonic()
        with torch.inference_mode():
            first, repeat = model(data), model(data)
            try:
                namespace["pscan"] = serial_scan
                direct = model(data)
            finally:
                namespace["pscan"] = parallel
        if (first.shape != (1, 1) or not torch.isfinite(first).all() or not torch.equal(first, repeat)
                or not torch.allclose(first, direct, atol=1e-4, rtol=1e-5)):
            raise ValueError("Synthetic graph repeat/recurrence correspondence failed")
        cases.append({"kind": kind, "input_sha256": hashlib.sha256(data.numpy().astype("<f4").tobytes()).hexdigest(),
                      "logit": first.item(), "repeat_exact": True,
                      "direct_recurrence_max_difference": float((first - direct).abs().max()),
                      "runtime_s": round(time.monotonic() - started, 4)})
    return {"schema": "chipmunk.demamba-synthetic-compatibility/13", "passed": True,
        "repository": REPOSITORY, "revision": REVISION, "weight_sha256": WEIGHT_SHA,
        "reviewer_sha256": digest(Path(__file__)), "assets_sha256": ASSETS, "loaded": loaded,
        "recurrence_tolerances": {"absolute": 1e-5, "relative": 1e-5},
        "graph_tolerances": {"absolute": 1e-4, "relative": 1e-5}, "scan_cases": scan_cases, "synthetic_cases": cases,
        "reference_scope": "Independent serial recurrence; vision library, trained tensors and remaining graph shared. Channel-axis reversal from publisher source is preserved. This does not prove historical training-runtime equivalence.",
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu", "threads": 2},
        "source_recordings_scored": 0, "classifier_fitted": False, "calibration_fitted": False,
        "predictive_accuracy_validated": False, "serving_admitted": False, "outside_reviewers": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    result = review(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": True, "synthetic_inputs": 3, "source_recordings_scored": 0}), flush=True)
