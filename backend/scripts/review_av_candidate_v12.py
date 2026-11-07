"""Pinned audio-visual candidate preflight; no full weights or media inference.

Acquisition reads public declarations/source and only the small SafeTensors
header. Review executes selected, checksum-verified definitions from inspected
source. It never initializes the publisher API, imports its package or downloads
backbones. This is structural/clock review, not detector validation.
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import struct
import typing
import urllib.request
from pathlib import Path

GITHUB = "bc911636a11a42fd0291bc31a5f092e1dd3dc811"
HUB = "a071972e6b9325da99a0d4ec2adce35a2446215b"
REPOSITORY = "MIHMahmudEli/david-net-av-v2"
HEADER_SHA = "0140fca0c20e573b9810370d1f3ac8d7f1246ac5b65d6b7cd88b78ae51b63b82"
WEIGHT_SHA = "0ce11de56b8fd1413ce2991e25b5edc56ae1e5ab7a4b577f97165e4b9ce17fbc"
WEIGHT_BYTES = 341484140
FILES = {
    "github/src/models/david_net.py": "8d317e6d581379ff39e76cd62fd4ea39e621173a249f8c81b6f5a4478faeb9ec",
    "github/src/pipeline/encoders.py": "8dbd9396decf43aeacf661a1135bd4788bc757b51a9f52c7093c6de43f182a45",
    "github/api/inference.py": "0f905f0da92d300224c96488ae793ab06737f192fa493133a2ffd8d7d4482a3b",
    "github/LICENSE": "3fd4f858876090a176a8483c4ce38d1a07b9cbf35deca2a97055fa334532371e",
    "hf/README.md": "3fcffb682c3f9a20e18da9eb89b6fff2429d15fce4244f46d24a8e9f5aa6460c",
    "hf/config.json": "94bf7388c72c51977ce53023510880cba34f78c8bbf5ebac3c4e29deb1c86ed6",
    "hf/metadata/experiments/CODE_REVISION.json": "00300b67f930457d2da0379054bcefd9dc403c9e575d6d06b83bb05e79460ff9",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verified(root, relative):
    data = (root / relative).read_bytes()
    if digest(data) != FILES[relative]:
        raise ValueError("Changed inspected source: " + relative)
    return data


def source_url(relative):
    if relative.startswith("github/"):
        return f"https://raw.githubusercontent.com/{REPOSITORY}/{GITHUB}/{relative.removeprefix('github/')}"
    name = relative.removeprefix("hf/")
    if name == "config.json":
        name = "models/config.json"
    return f"https://huggingface.co/{REPOSITORY}/resolve/{HUB}/{name}"


def acquire(root):
    root.mkdir(parents=True, exist_ok=True)
    for relative, expected in FILES.items():
        path = root / relative
        if path.exists():
            verified(root, relative)
            continue
        with urllib.request.urlopen(source_url(relative), timeout=40) as response:
            data = response.read(200001)
        if len(data) > 200000 or digest(data) != expected:
            raise ValueError("Published source bytes changed: " + relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    path = root / "weight-header.bin"
    if not path.exists():
        url = f"https://huggingface.co/{REPOSITORY}/resolve/{HUB}/models/david_net_best.safetensors"
        request = urllib.request.Request(url, headers={"Range": "bytes=0-1048575"})
        with urllib.request.urlopen(request, timeout=40) as response:
            prefix = response.read(8)
            if len(prefix) != 8 or struct.unpack("<Q", prefix)[0] != 17472:
                raise ValueError("Unexpected bounded tensor header")
            data = response.read(17472)
        if digest(data) != HEADER_SHA:
            raise ValueError("Published tensor header changed")
        with path.open("xb") as stream:
            stream.write(prefix + data)
    checked_header(root)


def checked_header(root):
    data = (root / "weight-header.bin").read_bytes()
    if len(data) != 17480 or struct.unpack("<Q", data[:8])[0] != 17472 or digest(data[8:]) != HEADER_SHA:
        raise ValueError("Exact original tensor header required")
    return json.loads(data[8:])


def definitions(data, names, namespace):
    """Select inspected definitions only; no top-level imports or API execution."""
    nodes = [n for n in ast.parse(data).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError("Complete inspected definitions required")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-candidate-source", "exec", dont_inherit=True), namespace)
    return namespace


def interval_function(api, torch):
    engine = next(n for n in ast.parse(api).body if isinstance(n, ast.ClassDef) and n.name == "DavidNetInference")
    predict = next(n for n in engine.body if isinstance(n, ast.FunctionDef) and n.name == "predict")
    nodes = [n for n in predict.body if isinstance(n, ast.FunctionDef) and n.name == "_get_intervals"]
    if len(nodes) != 1:
        raise ValueError("Inspected interval function absent")
    namespace = {"torch": torch, "List": list, "Dict": dict, "Any": typing.Any}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-candidate-interval", "exec", dont_inherit=True), namespace)
    return namespace["_get_intervals"]


def review(root):
    sources = {name: verified(root, name) for name in FILES}
    header = checked_header(root)
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    namespace = {"__name__": __name__, "dataclass": dataclasses.dataclass, "Optional": typing.Optional,
                 "math": math, "torch": torch, "nn": nn, "F": functional}
    definitions(sources["github/src/models/david_net.py"],
                {"DavidNetConfig", "DavidNet", "CrossModalBlock", "SyncModule", "_ff", "_resize_seq"}, namespace)
    config = json.loads(sources["hf/config.json"])
    with torch.device("meta"):
        graph = namespace["DavidNet"](namespace["DavidNetConfig"](**config["model"]), nn.Identity(), nn.Identity())
    state = graph.state_dict()
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if set(state) != set(tensors):
        raise ValueError("Header/declared-head parameter names differ")
    offset = 0
    for name, value in sorted(tensors.items(), key=lambda item: item[1]["data_offsets"][0]):
        parameter = state[name]
        size = math.prod(value["shape"]) * 4
        if (value["dtype"] != "F32" or list(parameter.shape) != value["shape"]
                or parameter.dtype != torch.float32 or value["data_offsets"] != [offset, offset + size]):
            raise ValueError("Header shape/type/offset correspondence differs")
        offset += size
    if offset + 17480 != WEIGHT_BYTES:
        raise ValueError("Header does not account for complete published file size")
    intervals = interval_function(sources["github/api/inference.py"], torch)
    examples = []
    for patch in (0, 15, 16, 31, 127):
        logits = torch.full((1, 128), -20.)
        logits[0, patch] = 20.
        actual = intervals(logits, 4.)
        time_bin = patch // 16
        support = [time_bin / 2, (time_bin + 1) / 2]
        expected = [{"start": round(4 * patch / 128, 2), "end": round(4 * (patch + 1) / 128, 2), "score": 1.}]
        if actual != expected:
            raise ValueError("Isolated stock interval arithmetic differs")
        examples.append({"flat_index": patch, "tubelet": time_bin, "spatial_patch": patch % 16,
                         "nominal_time_bin_s": support, "stock_reported_intervals": actual})
    historical = json.loads(sources["hf/metadata/experiments/CODE_REVISION.json"])
    return {"schema": "chipmunk.av-candidate-source-preflight/12", "structural_checks_passed": True,
        "repository": REPOSITORY, "github_revision": GITHUB, "hub_revision": HUB,
        "sources_sha256": FILES, "reviewer_sha256": digest(Path(__file__).read_bytes()),
        "weight_file_bytes_declared": WEIGHT_BYTES, "weight_file_sha256_declared": WEIGHT_SHA,
        "header_sha256_verified": HEADER_SHA, "header_only_bytes_acquired": 17480,
        "head_tensors_shape_compatible": len(state), "head_parameters_from_shapes": sum(p.numel() for p in state.values()),
        "graph_on_meta_device": True, "complete_weights_acquired_or_verified": False,
        "strict_trained_state_load_performed": False, "backbones_acquired_or_loaded": False,
        "source_media_inference_performed": False, "new_predictive_accuracy_result": False,
        "interval_examples": examples, "time_axis_issue_reproduced": True,
        "time_axis_scope": "Isolated supplied interval conversion maps each flattened spatial patch to a distinct time subdivision. There are eight nominal temporal bins and sixteen spatial positions per bin, not128 distinct acquisition times. Global transformer context is not confined to those bins. These synthetic bins are not verified native timestamps or causal edit boundaries, and this test does not measure learned localization accuracy.",
        "stock_api_adopted": False, "serving_changed": False, "overall_confidence_changed": False,
        "license_declaration": "MIT in publisher code license and weight card; underlying pretrained/data rights are separate and not fully established here.",
        "historical_source_declared": {k: historical[k] for k in ("code_repo", "code_revision")},
        "historical_source_equivalence_verified": False,
        "remaining_requirements": ["Verify complete trained head and exact backbone/source lineage before any recording comparison.",
            "Use a dedicated strict loader which aborts on missing weights/backbones; do not use stock API fallbacks.",
            "Preserve decoded picture/audio clocks and separate spatial axes from temporal support.",
            "Freeze a complete comparison before scoring; use all four audio/video authenticity combinations and ordinary controls.",
            "Obtain own calibration, location/false-flag/miss results and fresh confirmation before admission."],
        "outside_reviewers": 0, "runtime": {"torch": torch.__version__, "numerical_device": "cpu"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("acquire", "review"))
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    if args.mode == "acquire":
        if args.output is not None:
            parser.error("Acquisition accepts no report path")
        acquire(args.root)
        print(json.dumps({"acquired_source_files": len(FILES), "tensor_header_bytes": 17480, "full_weights_downloaded": False}))
    else:
        if args.output is None or args.output.exists():
            parser.error("Review requires a new output path")
        result = review(args.root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"structural_checks_passed": True, "time_axis_issue_reproduced": True, "predictive_result": False}))


if __name__ == "__main__":
    main()
