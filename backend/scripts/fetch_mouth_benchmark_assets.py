"""Fetch/verify the exact permitted research assets, excluding restricted detectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

CATALOGUE_SHA = "f1448f43338ba4a8e47145b0ab56f8f8e5a0b2546a48dbdc6f5ea994da7766a3"
MESH_SHA = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
LICENCE_SHA = "75fcad03ad21369e9eeb369b97b73fc0bba7f7f3243d0a661f968bd1478207bb"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(catalogue, root, verify_only=False):
    if digest(catalogue) != CATALOGUE_SHA:
        raise ValueError("Asset catalogue differs from the reviewed public record")
    m = json.loads(catalogue.read_text())
    jobs = []
    for row in m["code"]:
        url = f"https://raw.githubusercontent.com/KlingAIResearch/LivePortrait/{m['source_revision']}/{row['path']}"
        jobs.append(("liveportrait-source/" + row["path"], url, row["sha256"], 2 * 1024 * 1024))
    for row in m["weights"]:
        jobs.append(("liveportrait-weights/" + row["path"], row["url"], row["sha256"], row["bytes"]))
    jobs += [
        (
            "permissions/face_landmarker.task",
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            MESH_SHA,
            4 * 1024 * 1024,
        ),
        (
            "permissions/liveportrait-source-LICENSE.txt",
            f"https://raw.githubusercontent.com/KlingAIResearch/LivePortrait/{m['source_revision']}/LICENSE",
            LICENCE_SHA,
            65536,
        ),
    ]
    total = 0
    for name, url, sha, limit in jobs:
        path = root / name
        if not path.exists():
            if verify_only:
                raise ValueError("Required benchmark asset is missing: " + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".part")
            created = False
            try:
                h, size = hashlib.sha256(), 0
                with urllib.request.urlopen(url, timeout=60) as response, partial.open("xb") as out:
                    created = True
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            raise ValueError("Benchmark asset exceeds its reviewed bound")
                        h.update(chunk)
                        out.write(chunk)
                if h.hexdigest() != sha:
                    raise ValueError("Downloaded benchmark asset differs from its expected SHA-256")
                partial.replace(path)
            finally:
                if created:
                    partial.unlink(missing_ok=True)
        if digest(path) != sha:
            raise ValueError("Cached benchmark asset changed: " + name)
        total += path.stat().st_size
    target = root / "permissions/liveportrait-verified-assets.json"
    if target.exists():
        if digest(target) != CATALOGUE_SHA:
            raise ValueError("Existing asset receipt differs from the reviewed catalogue")
    elif not verify_only:
        shutil.copyfile(catalogue, target)
    else:
        raise ValueError("Required benchmark asset receipt is missing")
    print(
        json.dumps(
            {
                "verified_files": len(jobs),
                "bytes": total,
                "catalogue_sha256": CATALOGUE_SHA,
                "restricted_face_checkpoints": 0,
                "role": "Local benchmark generation only; no customer model enabled",
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("catalogue", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--verify-only", action="store_true")
    a = p.parse_args()
    fetch(a.catalogue, a.output, a.verify_only)
