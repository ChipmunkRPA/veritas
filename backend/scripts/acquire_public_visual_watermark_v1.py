"""Bootstrap the published authored visual test from pinned public assets only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import acquire_public_audioseal_v2 as archives
from scripts import public_visual_watermark_v1 as method


def acquire(root):
    lock = method.read(method.LOCK)
    root.mkdir(exist_ok=False)
    source = root / "source"
    files = source / "watermark-anything"
    notice_urls = {row["path"]: row["source"] for row in lock["third_party_notices"]}
    for row in lock["source_files"]:
        archives.relative(row["path"])
        url = (
            notice_urls.get(row["path"])
            or f"https://raw.githubusercontent.com/{lock['source_repo']}/{lock['source_revision']}/{row['path']}"
        )
        target = files / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        archives.download({**row, "source": url}, target)
    dependency_lock = method.read(Path(__file__).with_name("public_audioseal_assets_v2.json"))
    dep_root = root / "backend/data/independent-evaluation/public-watermark-v2-source"
    dep_root.mkdir(parents=True)
    names = {row["archive"] for row in dependency_lock["dependency_files"]}
    notice = next(r for r in dependency_lock["assets"] if r["path"] == "antlr4-LICENSE.txt")
    archives.download(notice, dep_root / notice["path"])
    for row in dependency_lock["assets"]:
        if row["path"] not in names:
            continue
        archive = dep_root / row["path"]
        archives.download(row, archive)
        archives.extract(
            archive,
            [r for r in dependency_lock["dependency_files"] if r["archive"] == row["path"]],
            dep_root / "isolated-python",
        )
    method.write(
        dep_root / "isolated-python-files.json",
        {"files": [{k: r[k] for k in ("path", "sha256")} for r in dependency_lock["dependency_files"]]},
    )
    method.verify(root, source)
    receipt = {
        "schema": "chipmunk.public-visual-watermark-acquisition/1",
        "source_files": len(lock["source_files"]),
        "dependency_files": len(dependency_lock["dependency_files"]),
        "source_imports": 0,
        "package_installations": 0,
        "checkpoint_downloads": 0,
        "model_constructions": 0,
        "neural_forwards": 0,
        "customer_inputs": 0,
        "model_asset_lock_sha256": method.digest(method.LOCK),
        "dependency_lock_sha256": method.digest(Path(__file__).with_name("public_audioseal_assets_v2.json")),
        "source_license_notice_count": len(lock["third_party_notices"]) + 1,
        "next": "Freeze using main=<this root>, source=<this root>/source and a new trial path. Native checkpoint is verified/loaded in RAM only.",
    }
    method.write(root / "acquisition.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(acquire(args.root), indent=2))
