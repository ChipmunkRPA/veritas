# SPDX-License-Identifier: Apache-2.0
"""Freeze and acquire source-verified originals for the local timing comparison."""

from __future__ import annotations

import argparse
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from scripts.acquire_video_common_sources_v91 import lfs_pointer
from scripts.video_common_sources_v89 import CREMA_REV, historical_actors, read, require, sha, write

PUBLISHER = "CheyneyComputerScience/CREMA-D"
ROSTER_SALT = "chipmunk-local-lipsync-recovery/1"
CLIP = re.compile(r"VideoFlash/(\d+)_ITS_NEU_XX\.flv")


def freeze(folder, archive):
    tree_path = archive / "video-v3/av/crema-tree.json"
    earlier_acquisition = archive / "video-common-acquisition-v91/asset-plan.json"
    tree = read(tree_path)
    require(tree["sha"] == CREMA_REV and not tree["truncated"], "Complete pinned publisher tree required")
    earlier, history_digests = historical_actors(archive)
    acquisition = read(earlier_acquisition)
    earlier.update(int(row["group"].split(":")[1]) for row in acquisition["assets"]
                   if row["source_family"] == "recorded_audiovisual_speech")
    qualified = {}
    for row in tree["tree"]:
        match = CLIP.fullmatch(row["path"])
        if match and row["type"] == "blob" and int(match[1]) not in earlier:
            qualified[match[1]] = row
    require(len(qualified) >= 15, "At least fifteen unused known actor codes with the shared sentence required")
    actor_codes = sorted(qualified, key=lambda actor: hashlib.sha256(f"{ROSTER_SALT}/{actor}".encode()).digest())
    sources = [{"actor": actor, "ordinal": index + 1, "split": "development" if index < 4 else "held",
                "source_path": qualified[actor]["path"], "pointer_git_sha1": qualified[actor]["sha"],
                "pointer_bytes": qualified[actor]["size"], "file": f"source-{index + 1:02d}.flv"}
               for index, actor in enumerate(actor_codes[:15])]
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "intake-freeze.json", {"schema": "chipmunk.video-lipsync-local-delay-intake/1",
          "publisher_revision": CREMA_REV, "publisher_tree_sha256": sha(tree_path),
          "prior_acquisition_sha256": sha(earlier_acquisition), "history_sha256": history_digests,
          "previous_actor_count": len(earlier), "qualified_actor_count": len(qualified),
          "ordering_salt": ROSTER_SALT, "sources": sources})


def fetch(url, limit):
    request = Request(url, headers={"User-Agent": "Chipmunk-Verify-source-acquisition"})
    with urlopen(request, timeout=30) as response:
        return response.read(limit + 1)


def acquire_one(folder, row):
    source = f"{PUBLISHER}/{CREMA_REV}/{row['source_path']}"
    pointer = fetch(f"https://raw.githubusercontent.com/{source}", row["pointer_bytes"])
    git_blob = f"blob {len(pointer)}\0".encode() + pointer
    require(len(pointer) == row["pointer_bytes"] and hashlib.sha1(git_blob).hexdigest() == row["pointer_git_sha1"],
            "Publisher pointer differs from the pinned Git tree")
    digest, size = lfs_pointer(pointer)
    path = folder / row["file"]
    if not path.exists():
        payload = fetch(f"https://media.githubusercontent.com/media/{source}", size)
        require(len(payload) == size and hashlib.sha256(payload).hexdigest() == digest,
                "Source bytes differ from publisher LFS pointer")
        with path.open("xb") as stream:
            stream.write(payload)
    require(path.stat().st_size == size and sha(path) == digest, "Stored source differs from publisher LFS pointer")
    print("verified", row["ordinal"], flush=True)
    return {**row, "sha256": digest, "bytes": size}


def acquire(folder):
    roster = folder / "intake-freeze.json"
    plan = read(roster)
    require(plan["publisher_revision"] == CREMA_REV, "Pinned source plan required")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda row: acquire_one(folder, row), plan["sources"]))
    write(folder / "intake-receipt.json", {"schema": "chipmunk.video-lipsync-local-delay-acquisition/1",
          "roster_sha256": sha(roster), "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acquire"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--archive", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data/independent-evaluation")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.folder, args.archive)
    else:
        acquire(args.folder)


if __name__ == "__main__":
    main()
