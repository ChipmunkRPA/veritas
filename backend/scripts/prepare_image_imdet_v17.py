"""Acquire immutable permitted research assets; never run provider code or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVISION = "17e493fd263f9195999d01dde1eee78ea8956638"
REPOSITORY = "Deckard618/imdet"
MODEL_SHA = "87cf796d2ca7cd20039e17c4be40ff4559f58d16d29410e38e3fb8234e60c10e"
MODEL_BYTES = 12509961
FILES = ("README.md", "LICENSE", "NOTICE", "pe_tiny/LICENSE-pe_tiny.txt", "pe_tiny/config.json")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(name, folder, maximum):
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as stream:
        while block := response.read(1024 * 1024):
            size += len(block)
            if size > maximum or shutil.disk_usage(folder).free < 1536 * 1024**2:
                raise ValueError("Acquisition size/storage guard: partial file retained")
            stream.write(block)
    return {"url": url, "sha256": digest(path), "bytes": size}


def acquire(folder, output):
    if folder.exists() or output.exists():
        raise ValueError("New acquisition paths required; never overwrite an attempt")
    folder.mkdir(parents=True)
    records = {name: fetch(name, folder, 100000) for name in FILES}
    declaration = (folder / "LICENSE").read_text()
    license_text = (folder / "pe_tiny/LICENSE-pe_tiny.txt").read_text()
    if not (
        "Each license covers the model.onnx and config.json" in declaration
        and "pe_tiny/ and pe_small/" in declaration
        and "Apache License, Version 2.0" in declaration
        and "Version 2.0, January 2004" in license_text
    ):
        raise ValueError("Explicit checkpoint-folder license correspondence required before weights")
    config = json.loads((folder / "pe_tiny/config.json").read_text())
    if not (
        config["labels"] == ["fake", "real"]
        and config["output"] == "logits"
        and config["preprocess"]
        == [{"op": "resize", "size": 256}, {"op": "center_crop", "size": 256}, {"op": "to_tensor"}]
    ):
        raise ValueError("Published output/preprocessing contract changed")
    if shutil.disk_usage(folder).free - MODEL_BYTES < 1536 * 1024**2:
        raise ValueError("Insufficient reserved disk for this exact model; weights not started")
    records["pe_tiny/model.onnx"] = fetch("pe_tiny/model.onnx", folder, MODEL_BYTES)
    model = records["pe_tiny/model.onnx"]
    if model["sha256"] != MODEL_SHA or model["bytes"] != MODEL_BYTES:
        raise ValueError("Exact publisher LFS object/size required; failed acquisition retained")
    result = {
        "schema": "chipmunk.image-imdet-acquisition/17",
        "repository": REPOSITORY,
        "revision": REVISION,
        "assets": records,
        "checkpoint_license": "Apache-2.0",
        "permission_basis": "Publisher LICENSE explicitly applies folder license to model.onnx and config.json; full Apache text and NOTICE retained.",
        "new_model_forwards": 0,
        "project_media_forwards": 0,
        "fitting_started": False,
        "serving_admitted": False,
        "customer_confidence_changed": False,
        "limitations": [
            "Publisher-reported accuracy is not our validation.",
            "Training-source overlap and training protocol remain unverified.",
            "Whole-image classifier; no native pixel-origin map or demonstrated video capability.",
            "Only tiny checkpoint acquired; larger alternatives not silently substituted.",
        ],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "revision": REVISION,
                "model_sha256": MODEL_SHA,
                "model_bytes": MODEL_BYTES,
                "new_model_forwards": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acquire(args.folder, args.output)
