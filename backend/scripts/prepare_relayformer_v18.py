"""Acquire the explicitly licensed official image localizer, without executing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path

REVISION = "348c5f7c9069336846cd74bdbbd5cdc85693555d"
CODE_REVISION = "23c973fc956195537d85148d124d6a146243f6c9"
MODEL_BYTES = 1102625388
MODEL_SHA = "00a0f145ae4a98e66cad95aa79d2ce470d77821ee4262d6b803b3705c11c2090"
SOURCES = {
    "README.md": "ed8acec78a66f764a4150bf867b60a298559ce42",
    "LICENSE": "feab6d301e74d102ecbbd3b0f9f7b4809f5896c7",
    "models/RelayFormer.py": "de98b88bbed87465c5bcd22495203fa840d8ee44",
    "models/GLoRA_vit.py": "93778927e2f0b419bc162b1b62802ea78730c32a",
    "infer.py": "4fd1362b9656061a0d7c82445d54ee17a488f8ce",
    "datasets/inference_dataset.py": "414b2ef70daee1bb52117660816a074ec168f811",
    "requirements.txt": "9f7f1201464883fdc2b323cd34579c2453500484",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def acquire(folder, output):
    if folder.exists() or output.exists():
        raise ValueError("New acquisition paths required; preserve existing attempts")
    if shutil.disk_usage(folder.parent).free - MODEL_BYTES < 1536 * 1024**2:
        raise ValueError("Insufficient reserved disk; model download not started")
    folder.mkdir(parents=True)
    records = {}
    for path, blob in SOURCES.items():
        url = f"https://raw.githubusercontent.com/WenOOI/RelayFormer/{CODE_REVISION}/{path}"
        with urllib.request.urlopen(url, timeout=30) as response:
            value = response.read(300000)
        actual = hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()
        if actual != blob:
            raise ValueError("Exact inspected upstream source blob required")
        destination = folder / "upstream" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(value)
        records["upstream/" + path] = {
            "url": url,
            "git_blob": blob,
            "sha256": digest(destination),
            "bytes": len(value),
        }
    card_url = f"https://huggingface.co/Wenn11/RelayFormer/resolve/{REVISION}/README.md"
    with urllib.request.urlopen(card_url, timeout=30) as response:
        card = response.read(10000)
    if hashlib.sha256(card).hexdigest() != "98b45ea81164d1e1a1dd82255207053b15cd6c69d922a1c5cf3387ce604d4b74":
        raise ValueError("Exact explicit official checkpoint-license declaration required")
    with (folder / "CHECKPOINT_README.md").open("xb") as stream:
        stream.write(card)
    if "MIT License" not in (folder / "upstream/LICENSE").read_text():
        raise ValueError("Inspected MIT code license changed")
    records["CHECKPOINT_README.md"] = {
        "url": card_url,
        "sha256": digest(folder / "CHECKPOINT_README.md"),
        "bytes": len(card),
    }
    model_url = f"https://huggingface.co/Wenn11/RelayFormer/resolve/{REVISION}/image-only.pth"
    started, received, next_progress = time.monotonic(), 0, 128 * 1024**2
    path = folder / "image-only.pth"
    try:
        with urllib.request.urlopen(model_url, timeout=60) as response, path.open("xb") as stream:
            while value := response.read(4 * 1024**2):
                if (
                    received + len(value) > MODEL_BYTES
                    or time.monotonic() - started > 900
                    or shutil.disk_usage(folder).free - len(value) < 1536 * 1024**2
                ):
                    raise ValueError("Download size/disk/time guard; partial artifact retained")
                stream.write(value)
                received += len(value)
                if received >= next_progress:
                    print(
                        json.dumps({"downloaded_bytes": received, "expected_bytes": MODEL_BYTES}), flush=True
                    )
                    next_progress += 128 * 1024**2
        if received != MODEL_BYTES or digest(path) != MODEL_SHA:
            raise ValueError("Released image-checkpoint bytes do not match publisher LFS commitment")
    except BaseException:
        with (folder / "failed-download.json").open("x") as stream:
            json.dump(
                {
                    "received_bytes": received,
                    "expected_bytes": MODEL_BYTES,
                    "partial_retained": True,
                    "project_media_forwards": 0,
                },
                stream,
            )
        raise
    records["image-only.pth"] = {"url": model_url, "sha256": MODEL_SHA, "bytes": received}
    result = {
        "schema": "chipmunk.relayformer-acquisition/18",
        "checkpoint_revision": REVISION,
        "code_revision": CODE_REVISION,
        "code_license": "MIT",
        "checkpoint_license_declared": "Apache-2.0",
        "assets": records,
        "upstream_model_code_executed": False,
        "new_model_forwards": 0,
        "project_media_forwards": 0,
        "serving_admitted": False,
        "customer_confidence_changed": False,
        "limitations": [
            "Publisher label is an explicit checkpoint grant, not independent clearance of training photographs.",
            "Only image checkpoint acquired; video and application-oriented checkpoints are untested.",
            "Sample preprocessing crops oversized images and sample video iteration may drop trailing frames.",
            "Implicit timm base-weight downloads and non-strict checkpoint fallback must be disabled before any forward.",
            "No detection-quality comparison or outside human review yet.",
        ],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"model_sha256": MODEL_SHA, "model_bytes": received, "project_media_forwards": 0}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    acquire(args.folder, args.output)
