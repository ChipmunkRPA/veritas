# SPDX-License-Identifier: MIT
"""Reserve previously unused publisher captures for an eye-region hold test."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import re
from pathlib import Path

from scripts import video_common_sources_v89 as io
from scripts.acquire_video_common_sources_v91 import hf_object
from scripts.video_face_temporal_panel import BASE, RECIPE, RULE

SCHEMA = "chipmunk.eye-region-hold-sources/1"
COUNTS = {"development": 3, "held": 10}
CAPTURE_KEY = re.compile(rb"[\w-]{11}(?=_\d{5}_\d{5}\.mp4)")


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def reserve(base, panels, output):
    cache = base / "video-common-sources-v89"
    permission = cache / "permissions"
    receipt, history = io.read(permission / "receipt.json"), io.read(cache / "plan.json")
    if receipt["rtfs_revision"] != io.RTFS_REV or io.sha(permission / "receipt.json") != history["metadata_receipt_sha256"]:
        raise ValueError("Prior source metadata is unverified")
    videos = permission / "rtfs-videos.csv"
    if any(io.sha(path) != receipt["files"][path.name]["sha256"] for path in (videos, permission / "rtfs-README.md")):
        raise ValueError("The publisher's source or license data changed")
    publisher = base / "video-common-acquisition-v91" / "publisher-file-info.json"
    if io.sha(publisher) != io.read(publisher.parent / "preparation-summary.json")["publisher_file_info_sha256"]:
        raise ValueError("The publisher asset index changed")
    index = io.read(publisher)
    if index["sha"] != io.RTFS_REV:
        raise ValueError("The publisher metadata revision changed")
    siblings = {row["rfilename"]: row for row in index["siblings"]}
    used_captures, previous_groups = set(), set()
    for path in (*sorted(panels.glob("video*.json*")), cache / "plan.json"):
        if "eye-hold" in path.name:
            continue
        body = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        if b"RTFS" in body or b"rtfs" in body or b"original_videos/" in body:
            used_captures.update(value.decode() for value in CAPTURE_KEY.findall(body))
            previous_groups.update(value.decode() for value in re.findall(rb'[0-9a-f]{64}', body))
    new = {}
    for item in sorted(io.table(videos), key=lambda row: digest(SCHEMA + "/" + row["filename"])):
        filename, capture = item["filename"], item["id"]
        normalized = filename.split("_", 1)[0]
        group = digest(RULE + "/" + capture)
        url_key = digest(io.source_key(item["original_link"]))
        if (group in previous_groups or url_key in previous_groups or {capture, normalized} & used_captures
                or capture in new or "child" in item["tags"].lower()
                or not 5 <= float(item["end_time"]) - float(item["start_time"]) <= 18):
            continue
        path = "original_videos/" + filename
        if path not in siblings:
            continue
        sha, size = hf_object(siblings[path])
        if sha in previous_groups or not 0 < size <= 3_000_000:
            continue
        new[capture] = {"group": group, "capture_key_sha256": url_key, "files": {
            "ordinary": {"path": path, "size": size, "sha256": sha},
        }}
    groups = list(new.values())[:sum(COUNTS.values())]
    if len(groups) != sum(COUNTS.values()) or len({row["capture_key_sha256"] for row in groups}) != len(groups):
        raise ValueError("Unable to reserve a complete disjoint capture set")
    for index, row in enumerate(groups):
        row.update(index=index, role="development" if index < COUNTS["development"] else "held")
    io.write(output, {"purpose": SCHEMA, "rules": RULE, "recipe": RECIPE, "revision": io.RTFS_REV,
                      "license": "CC-BY-SA-4.0", "permission_card_sha256": receipt["files"]["rtfs-README.md"]["sha256"],
                      "publisher_index_sha256": io.sha(publisher), "groups": groups, "selected_groups": COUNTS,
                      "excluded_capture_filenames": len(used_captures)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--panels", type=Path, default=BASE.parents[2] / "docs/evaluation-panels")
    args = parser.parse_args()
    reserve(args.base, args.panels, args.output)


if __name__ == "__main__":
    main()
