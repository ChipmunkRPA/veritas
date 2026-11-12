# SPDX-License-Identifier: Apache-2.0
"""Freeze a publisher-verified, actor-disjoint panel for speech-content replacement."""

from __future__ import annotations

import argparse
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from scripts.prepare_video_lipsync_local_delay_transfer import acquire_one
from scripts.video_common_sources_v89 import CREMA_REV, read, require, sha, write

RECORDINGS = {"target": "DFA_NEU_XX", "same_words": "DFA_HAP_XX", "different_words": "IWL_NEU_XX"}
SALT = "chipmunk-mouth-content-independent/1"
VIDEO_PATH = re.compile(r"VideoFlash/(\d+)_(.+)\.flv")


def freeze(folder, tree_path, prior_freeze):
    tree = read(tree_path)
    require(tree["sha"] == CREMA_REV and not tree["truncated"], "Complete pinned publisher tree required")
    excluded = {row["actor"] for row in read(prior_freeze)["sources"]}
    groups = {}
    for row in tree["tree"]:
        match = VIDEO_PATH.fullmatch(row["path"])
        if match and match[1] not in excluded and match[2] in RECORDINGS.values():
            groups.setdefault(match[1], {})[match[2]] = row
    qualified = [actor for actor, group in groups.items() if len(group) == len(RECORDINGS)]
    require(len(qualified) >= 24, "Complete same-speaker publisher recording triplets required")
    ordered = sorted(qualified, key=lambda actor: hashlib.sha256(f"{SALT}/{actor}".encode()).digest())
    rows = []
    for index, actor in enumerate(ordered[:24]):
        split = "fit" if index < 8 else "calibration" if index < 12 else "held"
        for role, suffix in RECORDINGS.items():
            pinned = groups[actor][suffix]
            rows.append({"actor": actor, "group": index + 1, "split": split, "role": role, "ordinal": len(rows) + 1,
                         "source_path": pinned["path"], "pointer_git_sha1": pinned["sha"],
                         "pointer_bytes": pinned["size"], "file": f"speech-{index + 1:02d}-{role}.flv"})
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "intake-freeze.json", {"schema": "chipmunk.video-mouth-content-intake/1", "publisher_revision": CREMA_REV,
          "tree_sha256": sha(tree_path), "excluded_roster_sha256": sha(prior_freeze), "roster_salt": SALT, "sources": rows})


def acquire(folder):
    roster = folder / "intake-freeze.json"
    plan = read(roster)
    require(plan["publisher_revision"] == CREMA_REV, "Pinned source plan required")
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda row: acquire_one(folder, row), plan["sources"]))
    write(folder / "intake-receipt.json", {"schema": "chipmunk.video-mouth-content-acquisition/1",
          "roster_sha256": sha(roster), "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acquire"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--prior-freeze", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        if args.tree is None or args.prior_freeze is None:
            parser.error("freeze requires --tree and --prior-freeze")
        freeze(args.folder, args.tree, args.prior_freeze)
    else:
        acquire(args.folder)


if __name__ == "__main__":
    main()
