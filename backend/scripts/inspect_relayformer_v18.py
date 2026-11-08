"""Inspect permitted checkpoint tensors with weights-only loading, never a forward."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from scripts.prepare_relayformer_v18 import MODEL_SHA, digest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "models/research/image-v18-relayformer-v2"


def check_state(state):
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items())
    ):
        raise ValueError("Complete string-keyed tensor state required")
    if not all(v.numel() > 0 and bool(torch.isfinite(v).all()) for v in state.values()):
        raise ValueError("Every released tensor must be nonempty and finite")
    if not any(k.startswith("vit.") for k in state) or not any(k.startswith("mask_decoder.") for k in state):
        raise ValueError("Both backbone and localization decoder required")
    return state


def load_state():
    path = ASSETS / "image-only.pth"
    acquisition = json.loads(
        (ROOT / "docs/evaluation-panels/relayformer-v18-acquisition-v2.json").read_text()
    )
    for name, asset in acquisition["assets"].items():
        if digest(ASSETS / name) != asset["sha256"]:
            raise ValueError("Exact acquired checkpoint/source/license asset changed")
    if digest(path) != MODEL_SHA:
        raise ValueError("Publisher checkpoint commitment changed")
    unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    if any(name != "argparse.Namespace" for name in unsafe):
        raise ValueError("Unreviewed serialized globals: " + repr(unsafe))
    # Namespace is a known standard-library metadata container, not downloaded
    # executable code. No other extra global is admitted and weights_only stays on.
    with torch.serialization.safe_globals([argparse.Namespace] if unsafe else []):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("Publisher model-state field required; no alternate/random fallback")
    return check_state(checkpoint["model"]), sorted(checkpoint), unsafe


def inspect():
    torch.set_num_threads(2)
    state, fields, globals_used = load_state()
    rows = {}
    for key, value in state.items():
        array = value.detach().contiguous().cpu().numpy()
        rows[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return {
        "schema": "chipmunk.relayformer-checkpoint-inspection/18",
        "model_sha256": MODEL_SHA,
        "checkpoint_fields": fields,
        "weights_only": True,
        "additional_metadata_globals": globals_used,
        "tensors": len(state),
        "model_tensor_elements": sum(v.numel() for v in state.values()),
        "tensor_dtypes": dict(Counter(str(v.dtype) for v in state.values())),
        "finite_tensors": True,
        "tensor_shapes_and_sha256": rows,
        "upstream_model_code_executed": False,
        "model_forwards": 0,
        "project_media_forwards": 0,
        "serving_admitted": False,
        "customer_confidence_changed": False,
        "scope": "Released tensor/metadata inspection only; graph correspondence, native execution and detection quality are unqualified.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("New structural receipt path required")
    result = inspect()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("tensor_shapes_and_sha256",)}), flush=True)
