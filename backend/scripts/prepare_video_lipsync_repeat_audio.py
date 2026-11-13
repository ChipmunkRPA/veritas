# SPDX-License-Identifier: Apache-2.0
"""Select separate, unused CREMA-D actors before testing soundtrack repeat evidence."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from scripts.prepare_video_lipsync_local_delay_transfer import acquire
from scripts.video_common_sources_v89 import CREMA_REV, historical_actors, read, require, sha, write

SALT = "chipmunk-single-video-soundtrack-repeat/1"
SENTENCES = ("DFA", "IOM", "ITH", "ITS", "IWL", "IWW", "MTI", "TAI", "TIE", "TSI", "WSI", "IEO")
HISTORY = ("video-lipsync-local-delay-recovery-2026-11-11", "video-mouth-content-transfer-2026-11-11")


def order(value):
    return hashlib.sha256(f"{SALT}/{value}".encode()).hexdigest()


def freeze(folder, archive):
    tree_path = archive / "video-v3/av/crema-tree.json"
    tree = read(tree_path)
    require(tree["sha"] == CREMA_REV and not tree["truncated"], "Complete pinned publisher tree required")
    used, commitments = historical_actors(archive)
    for relative in (
        "video-common-acquisition-v91/asset-plan.json",
        *(f"{name}/intake-freeze.json" for name in HISTORY),
    ):
        path = archive / relative
        data = read(path)
        used.update(row["actor"] for row in data.get("sources", []) if "actor" in row)
        used.update(
            int(row["group"].split(":")[1])
            for row in data.get("assets", [])
            if row["source_family"] == "recorded_audiovisual_speech"
        )
        commitments[relative] = sha(path)
    members = {}
    for row in tree["tree"]:
        match = re.fullmatch(r"VideoFlash/(\d+)_([A-Z]{3})_NEU_XX\.flv", row["path"])
        if match and int(match[1]) not in used and match[2] in SENTENCES and row["type"] == "blob":
            members.setdefault(int(match[1]), {})[match[2]] = row
    actors = sorted((actor for actor, files in members.items() if len(files) >= 10), key=order)
    require(len(actors) >= 15, "Need fifteen actors distinct from documented historical sources")
    rows = []
    for i, actor in enumerate(actors[:16]):
        phrases = sorted(members[actor], key=lambda phrase: order(f"{actor}/{phrase}"))
        for j, phrase in enumerate(phrases[:6]):
            row = members[actor][phrase]
            ordinal = len(rows) + 1
            rows.append(
                {
                    "actor": actor,
                    "phrase": phrase,
                    "group": i + 1,
                    "choice": j,
                    "ordinal": ordinal,
                    "split": "development" if i < 4 else "held",
                    "source_path": row["path"],
                    "pointer_git_sha1": row["sha"],
                    "pointer_bytes": row["size"],
                    "file": f"source-{ordinal:02d}.flv",
                }
            )
    folder.mkdir(mode=0o700)
    write(
        folder / "intake-freeze.json",
        {
            "schema": "chipmunk.video-lipsync-repeat-audio-intake/1",
            "publisher_revision": CREMA_REV,
            "tree_sha256": sha(tree_path),
            "history_sha256": commitments,
            "previous_actor_count": len(used),
            "ordering_salt": SALT,
            "sources": rows,
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "acquire"))
    p.add_argument("folder", type=Path)
    p.add_argument(
        "--archive", type=Path, default=Path(__file__).resolve().parents[1] / "data/independent-evaluation"
    )
    args = p.parse_args()
    if args.phase == "freeze":
        freeze(args.folder, args.archive)
    else:
        acquire(args.folder)


if __name__ == "__main__":
    main()
