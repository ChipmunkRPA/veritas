# SPDX-License-Identifier: Apache-2.0
"""Replay the published mirrored-preview localization and admission counts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MIRRORS = (2, 4, 5, 7)
AXES = ("raw_axes", "display_axes")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(receipt):
    require(receipt["schema"] == "chipmunk.image-thumbnail-mirror-transfer/1"
            and receipt["confidence_weight"] == 0, "Unexpected evidence policy")
    ledger, families, kept, compact_partial = {}, {}, Counter(), 0
    for entry in receipt["cases"]:
        key = tuple(entry["key"])
        family, name, tag, axes, role = key
        require(key not in ledger and family in {"feathered", "compact", "genuine"}, "Duplicate or unknown case")
        families.setdefault(family, set()).add(name)
        target, selected, intersection = entry["pixels"]
        require(all(type(n) is int and n >= 0 for n in entry["pixels"]) and intersection <= min(target, selected)
                and bool(target) == (role == "edit"), "Invalid native pixel counts")
        useful = target > 0 and intersection >= (target + 1) // 2 and intersection * 10 >= 3 * (target + selected - intersection)
        require(entry["useful"] == useful and (entry["regions"] == 0 or entry["flag"]), "Incorrect case arithmetic")
        require(not (entry["flag"] and role != "edit") and not (entry["comparison"] and role in {"missing", "foreign"})
                and not entry["wrong_match"], "Unexpected clean flag, unsupported pairing, or wrong orientation")
        if role == "edit" and family == "compact" and tag != 1 and entry["flag"] and not useful:
            compact_partial += 1
        ledger[key] = entry
        if role == "edit":
            kept[family, tag, axes] += useful

    expected = set()
    for family in ("feathered", "compact"):
        for name in families[family]:
            expected.add((family, name, 1, "display_axes", "edit"))
            for tag in MIRRORS:
                expected.update((family, name, tag, axis, role)
                                for axis in AXES for role in ("edit", "unedited", "global_tone"))
                expected.update((family, name, tag, "none", role) for role in ("missing", "foreign"))
    expected.update(("genuine", name, tag, axis, "unedited")
                    for name in families["genuine"] for tag in MIRRORS for axis in AXES)
    require(set(ledger) == expected and {family: len(names) for family, names in families.items()}
            == {"feathered": 12, "compact": 6, "genuine": 5}, "Missing source groups or views")
    upright = sum(kept[family, 1, "display_axes"] for family in ("feathered", "compact"))
    branches = {}
    for tag in MIRRORS:
        for axis in AXES:
            counts = {family: kept[family, tag, axis] for family in ("feathered", "compact")}
            shared = sum(ledger[family, name, 1, "display_axes", "edit"]["useful"]
                         and ledger[family, name, tag, axis, "edit"]["useful"]
                         for family in counts for name in families[family])
            total = sum(counts.values())
            require(total >= 7 and shared * 4 >= upright * 3, "Mirrored localization gate failed")
            branches[f"{tag}/{axis}"] = {**counts, "useful": total, "upright_retained": shared}
    expected_summary = {"branches": branches, "upright_useful": upright, "native_cases": len(ledger),
                        "mapped_regions": sum(entry["regions"] for entry in ledger.values()),
                        "compact_mirrored_flags_below_overlap_rule": compact_partial}
    require(receipt["summary"] == expected_summary, "Published aggregate differs from the case ledger")
    return expected_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    print(json.dumps(verify(json.loads(parser.parse_args().receipt.read_text())), sort_keys=True))
