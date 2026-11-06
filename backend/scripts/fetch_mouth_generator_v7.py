"""Acquire only pinned permitted assets for research lip-edit generation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, row):
    if path.stat().st_size != row["bytes"]:
        raise ValueError("Asset size differs from publisher metadata: " + row["path"])
    sha = digest(path)
    if row["sha256"]:
        if sha != row["sha256"]:
            raise ValueError("Asset differs from publisher LFS digest: " + row["path"])
    else:
        body = path.read_bytes()
        if hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest() != row["git_blob"]:
            raise ValueError("Config differs from publisher Git object: " + row["path"])
    return {"path": row["path"], "bytes": row["bytes"], "sha256": sha}


def acquire(root, row, verify_only):
    path = root / row["path"]
    if path.exists():
        return verify(path, row)
    if verify_only:
        raise ValueError("Required permitted asset is absent: " + row["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    created = False
    try:
        request = urllib.request.Request(row["url"], headers={"User-Agent": "ChipmunkVerify-research"})
        with partial.open("xb") as out:
            created = True
            with urllib.request.urlopen(request, timeout=60) as response:
                size, reported = 0, 0
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > row["bytes"]:
                        raise ValueError("Asset exceeds its publisher byte bound")
                    out.write(block)
                    if size // (128 * 1024 * 1024) > reported:
                        reported = size // (128 * 1024 * 1024)
                        print(
                            json.dumps(
                                {"path": row["path"], "received_bytes": size, "total_bytes": row["bytes"]}
                            ),
                            flush=True,
                        )
        result = verify(partial, row)
        partial.rename(path)
        return result
    finally:
        if created:
            partial.unlink(missing_ok=True)


def fetch(manifest, expected, output, verify_only=False):
    if digest(manifest) != expected:
        raise ValueError("Selected asset manifest changed")
    data = json.loads(manifest.read_text())
    if data["schema"] != "chipmunk.mouth-generator-assets/7" or len(data["assets"]) != 7:
        raise ValueError("Unknown permitted asset plan")
    for row in data["assets"]:
        if (
            row["role"] not in {"musetalk", "vae", "speech"}
            or row["license"] not in {"mit", "apache-2.0"}
            or Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
        ):
            raise ValueError("Unreviewed role, permission or output path")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(acquire, output, row, verify_only) for row in data["assets"]]
        results = [job.result() for job in pending]
    body = {
        "schema": "chipmunk.mouth-generator-assets-receipt/7",
        "manifest_sha256": expected,
        "assets": results,
        "purpose": data["purpose"],
        "restricted_components_loaded": 0,
    }
    receipt = output / "verified-assets.json"
    if receipt.exists():
        if json.loads(receipt.read_text()) != body:
            raise ValueError("Existing acquisition receipt differs")
    elif verify_only:
        raise ValueError("Required acquisition receipt is absent")
    else:
        with receipt.open("x") as stream:
            json.dump(body, stream, indent=2)
            stream.write("\n")
    print(
        json.dumps(
            {
                "verified_files": len(results),
                "bytes": sum(r["bytes"] for r in results),
                "manifest_sha256": expected,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--sha256", required=True)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    fetch(a.manifest, a.sha256, a.output, a.verify_only)
