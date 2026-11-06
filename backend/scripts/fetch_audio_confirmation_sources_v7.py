"""Bounded public-source acquisition; never run a detector or unpack an archive."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

REVISION = "7b7904b73b10ba272145606a06afb6e65bfac985"


def plan(metadata, output):
    tree = json.loads((metadata / "duplex-pln-tree.json").read_text())
    selected = sorted([r for r in tree if r["path"].endswith(".tar")], key=lambda r: r["path"])[:4]
    if len(selected) != 4 or any(not r["path"].startswith("shards/PLN/PLN-0000") for r in selected):
        raise ValueError("Keep the four fixed planning shards")
    assets = [
        {
            "name": Path(r["path"]).name,
            "url": f"https://huggingface.co/datasets/DuplexGen/duplexgen-spoken/resolve/{REVISION}/{r['path']}",
            "bytes": r["size"],
            "sha256": r["lfs"]["oid"],
            "max_bytes": r["size"],
            "license": "publisher per-scenario MIT",
            "source": "DuplexGen/duplexgen-spoken",
            "revision": REVISION,
        }
        for r in selected
    ]
    checksums = {
        line.split()[1]: line.split()[0]
        for line in (metadata / "librispeech-md5sum.txt").read_text().splitlines()
    }
    for name in ("test-clean.tar.gz", "test-other.tar.gz"):
        assets.append(
            {
                "name": name,
                "url": "https://openslr.trmal.net/resources/12/" + name,
                "md5": checksums[name],
                "max_bytes": 600000000,
                "source": "OpenSLR12 LibriSpeech",
                "license": "CC-BY-4.0",
            }
        )
    body = {
        "schema": "chipmunk.audio-fresh-source-acquisition/1",
        "assets": assets,
        "metadata_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (metadata / "duplex-pln-tree.json", metadata / "librispeech-md5sum.txt")
        },
        "selection_scope": "Only acquire fixed archives for metadata/source eligibility. No detector inference, final case selection or confirmation claim.",
        "parallel_downloads": 2,
        "deployment_eligible": False,
    }
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "planned_archives": 6,
                "maximum_bytes": sum(r["max_bytes"] for r in assets),
                "plan_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        ),
        flush=True,
    )


def download(asset, root):
    if (
        Path(asset["name"]).name != asset["name"]
        or urlparse(asset["url"]).scheme != "https"
        or urlparse(asset["url"]).hostname not in ("huggingface.co", "openslr.trmal.net")
    ):
        raise ValueError("Unsafe source filename or publisher origin")
    target = root / asset["name"]
    partial = root / (asset["name"] + ".partial")
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    start = time.monotonic()
    existing = target if target.exists() else partial if partial.exists() else None
    offset = 0
    if existing:
        with existing.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                sha.update(block)
                md5.update(block)
                offset += len(block)
    complete_partial = asset.get("bytes") is not None and offset == asset["bytes"]
    if not target.exists() and not complete_partial:
        headers = {"User-Agent": "Chipmunk-Verify-public-evaluation/1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(asset["url"], headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            if offset and (
                response.status != 206
                or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
            ):
                raise ValueError("Server refused exact partial-file resume; retained partial source")
            mode = "ab" if offset else "xb"
            reported = offset
            with partial.open(mode) as stream:
                while block := response.read(4 * 1024 * 1024):
                    offset += len(block)
                    if offset > asset["max_bytes"]:
                        raise ValueError("Source archive exceeds frozen byte budget")
                    stream.write(block)
                    sha.update(block)
                    md5.update(block)
                    if offset - reported >= 128 * 1024 * 1024:
                        print(json.dumps({"asset": asset["name"], "downloaded_bytes": offset}), flush=True)
                        reported = offset
    if asset.get("bytes") is not None and offset != asset["bytes"]:
        raise ValueError("Publisher archive size differs")
    if asset.get("sha256") is not None and sha.hexdigest() != asset["sha256"]:
        raise ValueError("Publisher SHA256 differs")
    if asset.get("md5") is not None and md5.hexdigest() != asset["md5"]:
        raise ValueError("Publisher MD5 differs")
    if not target.exists():
        partial.rename(target)
    result = {
        "name": asset["name"],
        "status": "verified",
        "bytes": offset,
        "sha256": sha.hexdigest(),
        "md5": md5.hexdigest(),
        "publisher_sha256_verified": "sha256" in asset,
        "publisher_md5_verified": "md5" in asset,
        "seconds": time.monotonic() - start,
    }
    print(json.dumps(result), flush=True)
    return result


def fetch(plan_path, root):
    recipe = json.loads(plan_path.read_text())
    root.mkdir(parents=True, exist_ok=True)
    budget = sum(r["max_bytes"] for r in recipe["assets"])
    if shutil.disk_usage(root).free < budget + 6000000000:
        raise ValueError("Insufficient free disk for bounded archives plus6GB reserve")
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(download, a, root): a for a in recipe["assets"]}
        for future in concurrent.futures.as_completed(futures):
            try:
                records.append(future.result())
            except Exception as error:
                # Do not echo redirected presigned CDN URLs or authorization query strings.
                record = {
                    "name": futures[future]["name"],
                    "status": "failed",
                    "error_type": type(error).__name__,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    receipt = {
        "schema": "chipmunk.audio-fresh-archive-receipt/1",
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "archives": sorted(records, key=lambda r: r["name"]),
        "detector_inference_run": False,
        "scope": "Checksummed source archives only; not eligibility approval, completed confirmation or outside testing.",
    }
    with (root / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    if any(r["status"] != "verified" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "fetch"])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.stage == "plan":
        plan(args.input, args.output)
    else:
        fetch(args.input, args.output)
