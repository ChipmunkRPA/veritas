# SPDX-License-Identifier: Apache-2.0
"""Recount the 720p WebM transport study and verify the retained range geometry."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.video-scene-return-webm-transfer/1"
ROLES = ("unchanged", "layout_cut", "speaker_and_layout_cut", "speaker_only", "camera_shift", "grade", "full_camera")
SHA = re.compile(r"[0-9a-f]{64}")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def report(document, prior):
    require(document["schema"] == SCHEMA and document["detector_sha256"] == prior["bindings"]["scene_return.py"], "Unrecognized check")
    expected = {f"{row['pair']}-{row['direction']}-{row['role']}": row for row in prior["rows"]}
    counts = Counter()
    seen = set()
    for row in document["rows"]:
        name, role, truth, found = (row[key] for key in ("name", "role", "truth", "candidate"))
        require(name not in seen and all(SHA.fullmatch(row[key]) for key in ("source_sha256", "webm_sha256")), "Invalid case commitment")
        seen.add(name)
        if role in ROLES:
            source = expected.pop(name)
            require((role, truth, row["expected"], row["source_sha256"]) ==
                    (source["role"], source["truth"], source["candidate"], source["sha256"]), "Prior source binding differs")
        else:
            require(role in ("one_way_hard_cut", "reflected_camera_retime") and row["expected"] is None and truth is None,
                    "Unknown control")
        if found is not None:
            start, end = found
            tiles, anchors = row["tiles"], row["anchors"]
            require(4 <= start < end < 117 and 6 <= end - start <= 24, "Invalid candidate range")
            require(tiles == sorted(set(tiles)) and len(tiles) >= 24 and all(0 <= tile < 60 for tile in tiles), "Invalid grid support")
            quadrants = Counter((tile // 10 >= 3, tile % 10 >= 5) for tile in tiles)
            require(len(quadrants) == 4 and min(quadrants.values()) >= 4, "Invalid tile distribution")
            require([a["index"] for a in anchors] == [start - 1, (start + end) // 2, end]
                    and all(a["pts"] == (100 * a["index"] + 1) // 3 and SHA.fullmatch(a["sha256"]) for a in anchors),
                    "Invalid WebM source anchor")
        else:
            require(not row["anchors"] and not row["tiles"], "An unreported case contains an evidence region")
        counts[f"{role}_files"] += 1
        counts["controls"] += truth is None
        counts["control_flags"] += truth is None and found is not None
        counts["prior_exact"] += truth is not None and truth == row["expected"]
        counts["prior_exact_preserved"] += truth is not None and truth == row["expected"] == found
        counts["new_exact"] += truth is not None and truth == found and row["expected"] is None
        counts["exact"] += truth is not None and truth == found
        counts["wrong_timed_flags"] += truth is not None and found is not None and found != truth
        counts["anchors"] += len(row["anchors"])
    require(not expected and len(seen) == 74 and counts["controls"] == 58 and counts["prior_exact"] == 14
            and counts["one_way_hard_cut_files"] == 12 and counts["reflected_camera_retime_files"] == 6, "Incomplete panel")
    require(document["independent_replay"] == {"frames": 120 * len(seen), "anchors": counts["anchors"]}, "Independent audit count differs")
    return dict(counts) | {"passes": counts["prior_exact_preserved"] == 14 and counts["control_flags"] == 0
                           and counts["wrong_timed_flags"] == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("prior", type=Path)
    args = parser.parse_args()
    document = json.loads(args.ledger.read_bytes())
    prior = args.prior.read_bytes()
    require(hashlib.sha256(prior).hexdigest() == document["parent_ledger_sha256"], "Earlier study commitment differs")
    print(json.dumps(report(document, json.loads(prior)), sort_keys=True))


if __name__ == "__main__":
    main()
