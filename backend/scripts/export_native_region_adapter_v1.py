"""Export a private, inference-only regional adapter without optimizer state.

This verifies a completed research state; it does not admit a production model,
change its calibration, distribute weights or make an authorship probability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(chain_path: Path, output: Path):
    import torch
    from safetensors.torch import load, save

    chain = json.loads(chain_path.read_text())
    source = Path(chain["final_checkpoint"])
    expected = chain["final_checkpoint_sha256"]
    if output.exists() or chain["completed_updates"] != 1278 or digest(source) != expected:
        raise ValueError("A new destination and exact completed checkpoint are required")
    state = torch.load(source, map_location="cpu", weights_only=True)
    parameters = state["parameters"]
    if state["completed_updates"] != 1278 or len(parameters) != 246:
        raise ValueError("The complete accepted regional adaptation is required")
    if sum(value.numel() for value in parameters.values()) != 6_009_089:
        raise ValueError("Unexpected parameter count")
    if not all(
        value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in parameters.values()
    ):
        raise ValueError("Finite CPU float32 tensors are required")
    tensors = {name: value.detach().cpu().contiguous() for name, value in parameters.items()}
    hashes = {name: hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in tensors.items()}
    if hashes != chain["final_parameter_sha256"]:
        raise ValueError("Accepted learned tensors changed")
    blob = save(
        tensors, metadata={"schema": "chipmunk.regional-adapter/1", "source_checkpoint_sha256": expected}
    )
    restored = load(blob)
    if set(restored) != set(tensors) or not all(
        torch.equal(restored[name], value) for name, value in tensors.items()
    ):
        raise ValueError("Compact adapter failed exact tensor roundtrip")
    output.mkdir(parents=True, exist_ok=False)
    path = output / "regional-adapter.safetensors"
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    if digest(path) != hashlib.sha256(blob).hexdigest():
        raise ValueError("Saved adapter readback failed")
    result = {
        "schema": "chipmunk.private-regional-inference-artifact/1",
        "source_checkpoint_sha256": expected,
        "source_chain_sha256": digest(chain_path),
        "adapter_file": path.name,
        "adapter_sha256": digest(path),
        "bytes": path.stat().st_size,
        "tensors": len(tensors),
        "parameters": sum(value.numel() for value in tensors.values()),
        "tensor_sha256": hashes,
        "exact_tensor_roundtrip": True,
        "optimizer_state_included": False,
        "fitting_or_neural_calls": 0,
        "serving_admitted": False,
        "authorship_probability_calibrated": False,
        "scope": "Private compact artifact for production-loader integration and parity testing. Same learned tensors, no new training. Base checkpoint and reviewed inference graph are still required. Existing bounded-local-study permissions do not imply a blanket production-training grant.",
    }
    with (output / "manifest.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "adapter_sha256",
                    "bytes",
                    "tensors",
                    "parameters",
                    "exact_tensor_roundtrip",
                    "serving_admitted",
                )
            }
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chain", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.chain, args.output)
