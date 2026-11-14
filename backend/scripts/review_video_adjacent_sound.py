# SPDX-License-Identifier: Apache-2.0
"""Recount source-bound adjacent soundtrack measurements without media dependencies."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.video-adjacent-sound/1"
ARMS = ("existing_audio", "adjacent_audio", "existing_mouth", "adjacent_mouth")
EDITS = ("local_early", "local_late")
CONTROLS = ("original", "gain", "lowpass", "noise", "global_early", "global_late",
            "matched_picture_early", "matched_picture_late", "repeated_bed")
SHA = re.compile(r"[0-9a-f]{64}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def useful(pair, truth):
    return truth is not None and all(2 * (min(end, right) - max(start, left)) >= max(end, right) - min(start, left)
                                     for (start, end), (left, right) in zip(pair, truth, strict=True))


def _validate_pair(pair, samples):
    (a, b), (c, d) = pair
    require(all(type(x) is int for x in (a, b, c, d)) and d <= samples and 0 <= a < b <= c < d
            and b - a == d - c, "A repetition pair exceeds its source sample range")


def review(doc):
    sources = doc["freeze"]["sources"]
    rows = doc["rows"]
    require(doc["freeze"]["schema"] == SCHEMA and len(sources) == len(rows) == 26
            and len({row["sha256"] for row in sources}) == len(sources), "The 26-source plan is incomplete")
    output = {}
    added, lost, new_controls, new_groups = set(), set(), set(), set()
    for bank, split in (("acted", "development"), ("acted", "held"), ("conversation", "held")):
        label = f"{bank}_{split}"
        counts = {arm: Counter() for arm in ARMS}
        scope = Counter()
        for source, row in zip(sources, rows, strict=True):
            require((row["bank"], row["group"], row["source_sha256"]) == (source["bank"], source["group"], source["sha256"])
                    and SHA.fullmatch(source["sha256"]), "Source identity does not match its result")
            if (source["bank"], source["split"]) != (bank, split):
                continue
            scope.update(groups=1, edits=len(EDITS), controls=len(CONTROLS))
            if "unavailable" in row:
                scope["unavailable_groups"] += 1
                continue
            scope["measured_groups"] += 1
            samples = row["samples"]
            require(len(row["cases"]) == len(EDITS) + len(CONTROLS)
                    and {case["id"] for case in row["cases"]} == set(EDITS + CONTROLS), "Incomplete matched conditions")
            for case in row["cases"]:
                role, truth = case["id"], case["truth_samples"]
                require((role in EDITS) == (truth is not None) and SHA.fullmatch(case["pcm_sha256"]),
                        "Truth or waveform commitment differs from its condition")
                if truth:
                    _validate_pair(truth, samples)
                    first, second = truth
                    require(first[1] == second[0] and first[1] - first[0] in (3200, 4480), "Invalid known repeat")
                hits = {}
                alerts = {}
                for arm in ARMS:
                    records = case["arms"][arm]
                    counts[arm]["checked_edits" if truth else "checked_controls"] += records is not None
                    pairs = [] if records is None else records if arm.endswith("audio") else [item["ranges"] for item in records]
                    for pair in pairs:
                        _validate_pair(pair, samples)
                    if arm.endswith("mouth") and records:
                        for item in records:
                            require(item["ranges"] in (case["arms"][arm.replace("mouth", "audio")] or [])
                                    and len(item["frame_pairs"]) >= 3
                                    and len(item["pcm_sha256"]) == 2 and all(SHA.fullmatch(sha) for sha in item["pcm_sha256"]),
                                    "A visual result lacks its measured sound or picture support")
                    hits[arm] = any(useful(pair, truth) for pair in pairs)
                    alerts[arm] = bool(pairs)
                    if truth:
                        counts[arm]["useful"] += hits[arm]
                        counts[arm]["wrong_pairs"] += bool(pairs) and not hits[arm]
                    elif pairs:
                        counts[arm]["control_flags"] += 1
                        counts[arm][f"flags_{role}"] += 1
                if split == "held":
                    key = (bank, row["group"], role)
                    if hits["adjacent_mouth"] and not hits["existing_mouth"]:
                        added.add(key)
                        new_groups.add(key[:2])
                    if hits["existing_mouth"] and not hits["adjacent_mouth"]:
                        lost.add(key)
                    if not truth and alerts["adjacent_mouth"] and not alerts["existing_mouth"]:
                        new_controls.add(key)
        output[label] = {"scope": dict(scope), "arms": {arm: dict(counts[arm]) for arm in ARMS}}
    output["decision"] = {
        "added_useful": len(added), "lost_useful": len(lost), "additional_groups": len(new_groups),
        "new_control_flags": len(new_controls),
        "combined_useful": sum(output[label]["arms"]["existing_mouth"].get("useful", 0)
                               for label in ("acted_held", "conversation_held")) + len(added),
        "synchronized_flags": sum(output[label]["arms"]["adjacent_mouth"].get(f"flags_matched_picture_{side}", 0)
                                  for label in ("acted_held", "conversation_held") for side in ("early", "late")),
        "advances": (len(added) >= 4 and len(new_groups) >= 3 and len({bank for bank, _ in new_groups}) == 2
                     and not lost and len(new_controls) <= 2
                     and all(not output[label]["arms"]["adjacent_mouth"].get(f"flags_{control}", 0)
                             for label in ("acted_held", "conversation_held")
                             for control in ("matched_picture_early", "matched_picture_late", "global_early", "global_late"))),
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    doc = json.loads(parser.parse_args().receipt.read_text())
    result = review(doc)
    require(result == doc["outcome"], "Published outcomes differ from the replay")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
