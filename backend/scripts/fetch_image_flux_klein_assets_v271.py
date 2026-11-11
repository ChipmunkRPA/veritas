"""Verify or acquire the frozen, public FLUX.2 Klein reference-edit assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

from scripts.fetch_mouth_generator_v7 import acquire, digest

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "docs/evaluation-panels/image-flux-klein-assets-v271.json"
RECEIPT_SHA256 = "f27d41114e0853b407a794f683a9151566f2a415502cb91beea50c86c6b374b3"


def check_assets(directory: Path, *, download: bool) -> dict:
    if digest(RECEIPT) != RECEIPT_SHA256:
        raise ValueError("The reviewed asset receipt changed")
    manifest = json.loads(RECEIPT.read_bytes())
    directory = directory.expanduser().absolute()
    if directory.resolve() != directory:
        raise ValueError("Use a direct asset directory without symlinks")
    snapshot = f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}"
    verified = []
    for entry in manifest["verified_assets"]:
        path = directory / entry["path"]
        if not path.is_relative_to(directory) or path.resolve() != path:
            raise ValueError("Asset path must stay inside the direct directory")
        url = f"{snapshot}/{quote(entry['path'])}"
        actual = acquire(directory, {**entry, "url": url}, verify_only=not download)
        if actual != entry:
            raise ValueError(f"Asset disagrees with the frozen receipt: {entry['path']}")
        verified.append(actual)
    return {
        "status": "byte_verified",
        "verified_assets": len(verified),
        "verified_bytes": sum(entry["bytes"] for entry in verified),
        "receipt_sha256": RECEIPT_SHA256,
        "neural_forwards": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--download", action="store_true", help="Fetch missing pinned files from the publisher"
    )
    args = parser.parse_args()
    print(json.dumps(check_assets(args.directory, download=args.download), indent=2))
