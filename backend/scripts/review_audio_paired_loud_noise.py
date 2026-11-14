"""Verify and replay two paired voice comparisons without scores or model files."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_paired_competing_speech as competing
from scripts import audio_paired_loud_background as environmental

SCHEMA = "chipmunk.audio-paired-loud-noise-public/1"


def anonymize(first, second):
    groups = {}
    for (source, expected), row in zip(environmental.parent(), first, strict=True):
        foreground = source["source"]
        yield {"panel": "environmental", "scene": source["scene"],
               "label": foreground["label"] if foreground else "background",
               "family": None, "group": groups.setdefault(foreground["id"], len(groups)) if foreground else None,
               "primary": row["primary"], "paired_assessed": row["paired"] is not None,
               "reduced_flag": expected["enhanced"] is not None and expected["enhanced"] > environmental.BACKGROUND_POINT,
               "paired_direction": direction(row["paired"])}
    for row in second:
        family = row["family"]
        yield {"panel": "speech", "scene": row["condition"], "family": family, "group": row["group"],
               "label": "spoof" if family in competing.FAMILIES else "bonafide" if family == "genuine" else "background",
               "primary": row["primary"], "paired_assessed": row["paired"] is not None,
               "reduced_flag": False, "paired_direction": direction(row["paired"])}


def direction(score):
    if score is None:
        return "unassessed"
    if score > environmental.PAIRED_POINT:
        return "ai"
    return "original" if score < competing.ORIGINAL_POINT else "inconclusive"


def summarize(rows):
    allowed = {"ai", "original", "inconclusive", "unassessed"}
    counts, matches = {}, {"environmental": Counter(), "speech": Counter()}
    matched_groups = defaultdict(Counter)
    group_labels = {}
    added = {(panel, label): set() for panel in matches for label in ("bonafide", "spoof", "background")}
    families = set()
    for row in rows:
        panel, scene, label, group, family = (row[key] for key in ("panel", "scene", "label", "group", "family"))
        primary, paired, reduced = (row[key] for key in ("primary", "paired_direction", "reduced_flag"))
        if (panel not in matches or (panel, label) not in added or primary not in allowed or paired not in allowed
                or row["paired_assessed"] is not (paired != "unassessed") or type(reduced) is not bool
                or reduced and (panel != "environmental" or primary not in ("inconclusive", "unassessed"))
                or (paired != "unassessed") != (primary == "inconclusive" or panel == "speech" and primary == "ai")):
            raise ValueError("Unexpected decision or speech coverage")
        if panel == "environmental":
            valid_group = group is None if label == "background" else type(group) is int and group >= 0
            valid_family = family is None
        else:
            valid_group = type(group) is int and group >= 0
            valid_family = family in competing.FAMILIES if label == "spoof" else family == (
                "genuine" if label == "bonafide" else "background")
        if not valid_group or not valid_family:
            raise ValueError("Unexpected group or generator label")
        if label != "background":
            identity = (panel, group)
            if group_labels.setdefault(identity, (label, family)) != (label, family):
                raise ValueError("Matched deliveries changed their source label")
            matched_groups[identity][scene] += 1
        matches[panel][(scene, label)] += 1
        tally = counts.setdefault(f"{panel}/{scene}/{label}", Counter())
        gained = primary == "inconclusive" and paired == "ai"
        withheld = panel == "speech" and primary == "ai" and paired == "original"
        tally.update((f"primary_{primary}", "deliveries"))
        tally["paired_assessed"] += row["paired_assessed"]
        tally["primary_with_reduction"] += primary == "ai" or reduced
        tally["primary_with_paired"] += primary == "ai" or gained
        tally["union"] += primary == "ai" or reduced or gained
        tally["gained_by_union"] += gained and not reduced
        tally["primary_withheld"] += withheld
        if gained and not reduced and scene != "clean":
            added[panel, label].add(group)
            if panel == "speech" and label == "spoof":
                families.add(family)
    expected = {
        "environmental": {(scene, label): 18 for scene in ("clean", "street", "washroom") for label in ("bonafide", "spoof")}
        | {(scene, "background"): 4 for scene in ("street", "washroom")},
        "speech": {(scene, label): number for scene in ("clean", *competing.NOISY)
                   for label, number in (("bonafide", 16), ("spoof", 9))}
        | {(f"isolated_{scene}", "background"): 1 for scene in competing.NOISY},
    }
    group_profiles = Counter((panel, *group_labels[panel, group], tuple(sorted(scenes.values())))
                             for (panel, group), scenes in matched_groups.items())
    expected_profiles = Counter({
        ("environmental", "bonafide", None, (1, 1, 1)): 18,
        ("environmental", "spoof", None, (1, 1, 1)): 18,
        ("speech", "bonafide", "genuine", (2, 2, 2)): 8,
        **{("speech", "spoof", family, (1, 1, 1)): 3 for family in competing.FAMILIES},
    })
    if (matches != expected or group_profiles != expected_profiles
            or any(set(scenes) != ({"clean", "street", "washroom"} if panel == "environmental" else {"clean", *competing.NOISY})
                   for (panel, _), scenes in matched_groups.items())):
        raise ValueError("Published deliveries do not match the frozen cells")
    return {"counts": counts, "added_noisy_groups": {panel: {label: len(added[panel, label])
            for label in ("bonafide", "spoof", "background")} for panel in matches},
            "added_competing_speech_families": sorted(families)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--from-native", nargs=2, type=Path, metavar=("ENVIRONMENTAL", "SPEECH"))
    args = parser.parse_args()
    if args.from_native:
        a, b = map(environmental.read, args.from_native)
        if (a["lock"] != environmental.lock() or b["freeze"] != competing.identity()
                or a["summary"] != environmental.summarize(a["rows"]) or b["summary"] != competing.summarize(b["rows"])):
            raise ValueError("The native results or frozen points differ")
        rows = list(anonymize(a["rows"], b["rows"]))
        result = {"schema": SCHEMA, "environmental_lock": a["lock"], "competing_speech_lock": b["freeze"],
                  "native_receipt_sha256": [environmental.digest(p) for p in args.from_native],
                  "rows": rows, "summary": summarize(rows)}
        # Compact one row per delivery permits diff review without revealing scalar scores or voice representations.
        with args.receipt.open("x") as stream:
            header = {key: value for key, value in result.items() if key != "rows"}
            stream.write(json.dumps(header, indent=2, allow_nan=False)[:-2])
            stream.write(',\n  "rows": [\n    ' + ",\n    ".join(json.dumps(row, allow_nan=False) for row in rows) + "\n  ]\n}\n")
    else:
        result = environmental.read(args.receipt)
        if (result["schema"] != SCHEMA or result["environmental_lock"] != environmental.lock()
                or result["competing_speech_lock"] != competing.identity() or summarize(result["rows"]) != result["summary"]):
            raise ValueError("Published decisions do not reproduce their counts")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
