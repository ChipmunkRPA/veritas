# SPDX-License-Identifier: Apache-2.0
"""Reproduce moving-shortlist witnesses using an independent FFmpeg decode."""

import argparse
import json
from pathlib import Path

from scripts.review_video_brief_face_120_carrier import native, require
from scripts.video_face_moving_shortlist import review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument(
        "banks", nargs="+", type=Path, help="Directories containing native media and media.json"
    )
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    require(review(receipt["rows"], receipt["freeze"]["panel"]) == receipt["result"], "Outcomes differ")
    paths = {}
    for bank in args.banks:
        manifest = json.loads((bank / "media.json").read_text())
        paths.update({item["sha256"]: bank / "media" / item["file"] for item in manifest["rows"]})
    cases = frames = 0
    for row in receipt["rows"]:
        anchors = {
            anchor["frame"]: {
                "frame_index": anchor["frame"],
                "pts": anchor["pts"],
                "decoded_rgb_sha256": anchor["rgb_sha256"],
            }
            for method in (row["candidates"] or {}).values()
            if method["choice"]
            for anchor in method["choice"]["anchors"]
        }
        frames += native(
            paths[row["sha256"]],
            {
                "time_base": row["time_base"],
                "first_pts": row["first_pts"],
                "delivery_sha256": row["sha256"],
                "proposal": {"anchors": list(anchors.values())},
            },
            "avc",
        )
        cases += 1
    print(
        json.dumps(
            {"clips": cases, "presentation_ticks": 120 * cases, "selected_original_frames": frames},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
