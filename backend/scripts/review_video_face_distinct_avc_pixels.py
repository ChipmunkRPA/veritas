# SPDX-License-Identifier: Apache-2.0
"""Independently decode original pictures and presentation timestamps for the AVC two-range study."""

import argparse
import json
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import native, require
from scripts.video_face_distinct_avc import PANELS, review

ROOT = Path(__file__).resolve().parents[2]


def file_paths(folder, panel):
    banks = (
        (folder,)
        if panel == "fresh"
        else tuple(ROOT / "backend/data/independent-evaluation" / name for name in PANELS)
    )
    paths = {}
    for bank in banks:
        manifest = json.loads((bank / "media.json").read_text())
        for row in manifest["rows"]:
            paths[row["sha256"]] = bank / "media" / row["file"]
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("run", type=Path, help="Private directory containing the frozen run")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    panel = receipt["freeze"]["panel"]
    rows = receipt["rows"]
    require(review(rows, panel) == receipt["result"], "Public outcome differs")
    paths = file_paths(args.run, panel)
    counts = {"native_clips": 0, "native_ticks": 0, "selected_original_frames": 0}
    for row in rows:
        exhibits = {
            anchor["frame_index"]: anchor
            for record in (row["bounded_first"], *row["extras"].values())
            if record
            for anchor in record["anchors"]
        }
        counts["selected_original_frames"] += native(
            paths[row["sha256"]],
            {
                "delivery_sha256": row["sha256"],
                "time_base": row["time_base"],
                "first_pts": row["first_pts"],
                "proposal": {"anchors": list(exhibits.values())},
            },
            "avc",
        )
        counts["native_clips"] += 1
        counts["native_ticks"] += 120
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
