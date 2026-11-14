# SPDX-License-Identifier: Apache-2.0
"""Replay the source bindings and carrier-specific adjacent-sound outcomes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts.review_video_adjacent_sound import ARMS, CONTROLS, EDITS, SHA, _validate_pair, require, useful

SCHEMA = "chipmunk.video-adjacent-sound-delivery/1"
CARRIERS = ("PCM", "AAC")
CONDITIONS = {"clean": EDITS + CONTROLS,
              "competing": EDITS + ("original", "matched_picture_early", "matched_picture_late")}


def review(document):
    rows, sources = document["rows"], document["freeze"]["sources"]
    require(document["freeze"]["schema"] == SCHEMA and len(rows) == len(sources) == 4
            and len({row["sha256"] for row in sources}) == 4, "Incomplete public transfer roster")
    output = {}
    added_genuine, new_flags, prohibited = set(), set(), set()
    planned = {(carrier, condition, role) for carrier in CARRIERS for condition, roles in CONDITIONS.items() for role in roles}
    for carrier in CARRIERS:
        for condition, roles in CONDITIONS.items():
            counts = Counter()
            for source, row in zip(sources, rows, strict=True):
                require((row["id"], row["sha256"]) == (source["id"], source["sha256"]) and SHA.fullmatch(row["sha256"]),
                        "A source commitment differs from the freeze")
                require(len(row["cases"]) == len(planned)
                        and {(c["carrier"], c["condition"], c["role"]) for c in row["cases"]} == planned,
                        "A transfer condition was omitted or duplicated")
                clock = Fraction(row["picture_time_base"])
                frames = row["pictures"]
                require(len(frames) == 63 and all(pic["index"] == index
                        and (pic["pts"] - frames[0]["pts"]) * clock == Fraction(index, 25) and SHA.fullmatch(pic["rgb_sha256"])
                        for index, pic in enumerate(frames)), "Invalid verified video picture clock")
                synchronized = row["synchronized_picture_sha256"]
                require(set(synchronized) == {"matched_picture_early", "matched_picture_late"}
                        and all(len(items) == len(frames) and all(SHA.fullmatch(item) for item in items)
                                and items != [pic["rgb_sha256"] for pic in frames] for items in synchronized.values()),
                        "A synchronized control lacks its independently decoded retimed pictures")
                for case in row["cases"]:
                    if (case["carrier"], case["condition"]) != (carrier, condition):
                        continue
                    truth, role = case["truth"], case["role"]
                    require(role in roles and (role in EDITS) == (truth is not None)
                            and SHA.fullmatch(case["decoded_pcm_sha256"]) and set(case["arms"]) == set(ARMS),
                            "Case binding differs")
                    if truth:
                        _validate_pair(truth, row["audio_samples"])
                    hits = {}
                    alerts = {}
                    for method in ("existing", "adjacent"):
                        observed = case["arms"][method + "_mouth"]
                        counts[method + "_checked"] += observed is not None
                        pairs = []
                        for result in observed or []:
                            pair = result["ranges"]
                            _validate_pair(pair, row["audio_samples"])
                            require(pair in case["arms"][method + "_audio"]
                                    and all(type(i) is int and 0 <= i < len(frames) for pair in result["frame_pairs"] for i in pair)
                                    and all(SHA.fullmatch(item) for item in result["pcm_sha256"]), "Invalid repeat witness")
                            pairs.append(pair)
                        hits[method] = any(useful(pair, truth) for pair in pairs)
                        alerts[method] = bool(pairs)
                        if truth:
                            counts[method + "_useful"] += hits[method]
                            counts[method + "_misplaced"] += alerts[method] and not hits[method]
                        else:
                            counts[method + "_control_flags"] += alerts[method]
                    key = (row["id"], carrier, condition, role)
                    if truth:
                        counts["combined_useful"] += any(hits.values())
                        if hits["adjacent"] and not hits["existing"]:
                            counts["added_useful"] += 1
                            if carrier == "AAC" and condition == "clean":
                                added_genuine.add(row["id"])
                    elif alerts["adjacent"]:
                        if not alerts["existing"]:
                            new_flags.add(key)
                            counts["added_control_flags"] += 1
                        if role.startswith(("matched_picture_", "global_")):
                            prohibited.add(key)
                            counts["adjacent_synchronized_or_global_flags"] += 1
            output[f"{carrier}_{condition}"] = dict(counts) | {
                "planned_edits": len(rows) * len(EDITS), "planned_controls": len(rows) * (len(roles) - len(EDITS))}
    output["decision"] = {"distinct_added_clean_aac_recordings": len(added_genuine), "added_control_flags": len(new_flags),
                          "candidate_synchronized_or_global_flags": len(prohibited),
                          "supports_later_local_trial": len(added_genuine) >= 2 and not new_flags and not prohibited}
    return output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    document = json.loads(cli.parse_args().receipt.read_text())
    result = review(document)
    require(result == document["outcome"], "Published transfer results do not match")
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
