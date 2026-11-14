"""Freeze and score the earlier-copy check for separate revisions using retained source excerpts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.evidence import document_revision as draft
from app.evidence import document_revision_writing as paired

BANKS = {
    "ordinary": "writing-typed-decision-transfer-2026-11-13",
    "generated_1": "writing-context-margin-transfer-2026-11-13",
    "generated_2": "writing-typed-decision-transfer-2026-11-13",
}
GENERATORS = ("gemini-2.5-flash", "gpt-5.4-nano")
METHOD = (
    "Join two source-native paragraphs with a blank line. Combine a historical journal revision with each known "
    "generated revision on a different work. Alternate the order by sorted group rank. Form additional journal-only "
    "pairs from successive groups. Use every registered case, and count unsupported input separately. Keep the "
    "existing decision of at least double the odds on both complete responses. Count exact matches to independently "
    "recorded center offsets and deduplicate ordinary flags by journal work. Advance to optional local use only with "
    "at least four exact added generated edits, across at least three works and both generators, and no more than "
    "20 percent of assessed ordinary works flagged."
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def windows(old, new, idx):
    return tuple(
        (row["text"][row["spans"][idx - 1][0]:row["spans"][idx + 1][1]],
         [pos - row["spans"][idx - 1][0] for pos in row["spans"][idx]])
        for row in (old, new)
    )


def join(parts):
    earlier, later = ("\n\n".join(part[2][i][0] for part in parts) for i in (0, 1))
    offset = 0
    truth = []
    for group, family, (_, (text, span)) in parts:
        truth.append({"group": group, "family": family, "later": [pos + offset for pos in span]})
        offset += len(text) + 2
    return earlier, later, truth


def load_cases(root):
    available = {
        label: (root / directory / "roster.json").read_text()
        for label, directory in BANKS.items()
    }
    banks = {}
    for label, raw in available.items():
        groups = defaultdict(dict)
        for row in json.loads(raw):
            if (label == "ordinary") == (row["cohort"] == "calibration"):
                groups[row["group"]][row["family"]] = row
        banks[label] = groups
    ordinary = [
        (group, "ordinary", windows(variants["before"], variants["edited"], 2))
        for group, variants in sorted(banks["ordinary"].items()) if "edited" in variants
    ]
    cases = []
    for left, right in zip(ordinary[::2], ordinary[1::2]):
        cases.append(("journal_pair", *join((left, right))))
    for label in ("generated_1", "generated_2"):
        for index, (group, variants) in enumerate(sorted(banks[label].items())):
            control = ordinary[(index + (label == "generated_2") * 7) % len(ordinary)]
            for family in GENERATORS:
                old, new = variants["original"], variants[family]
                target = new.get("target_index")
                if target is None:
                    target = new["spans"].index(new["truth"])
                edit = (group, family, windows(old, new, target))
                parts = (edit, control) if index % 2 else (control, edit)
                cases.append((label, *join(parts)))
    return cases, {key: sha(raw) for key, raw in available.items()}


def map_cases(cases):
    for bank, earlier, later, truth in cases:
        comparison, old, new = draft.compare_with_text(earlier.encode(), later.encode(), ".txt")
        candidates = paired.separate_revisions(comparison, old, new)
        row = {
            "bank": bank, "draft_sha256": [sha(earlier), sha(later)], "truth": truth,
            "incumbent_scope": paired.paired_input(comparison, old, new, allow_interior=True) is not None,
            "mapping": [
                {side: {key: value for key, value in pos.items() if key != "text"}
                 for side, pos in candidate.geometry.items() if side != "change_ids"}
                for candidate in candidates
            ],
        }
        if candidates:
            positions = [[item["later"]["start"], item["later"]["end"]] for item in row["mapping"]]
            if positions != [item["later"] for item in truth]:
                raise ValueError("Mapped sentence differs from the registered edit")
        yield row, candidates


def summarize(rows):
    counts = defaultdict(Counter)
    controls, false_works, hit_works, hit_families = set(), set(), set(), set()
    for row in rows:
        truth = row["truth"]
        statuses = row.get("statuses", ["unavailable"] * len(truth))
        for label, status in zip(truth, statuses, strict=True):
            family, group = label["family"], label["group"]
            stats = counts[row["bank"] + "/" + family]
            stats["total"] += 1
            stats[status] += 1
            stats["incumbent_in_scope"] += row["incumbent_scope"]
            if family == "ordinary":
                if status != "unavailable":
                    controls.add(group)
                if status == "joint_increase":
                    false_works.add(group)
            elif status == "joint_increase" and not row["incumbent_scope"]:
                hit_works.add(group)
                hit_families.add(family)
    added = sum(stats["joint_increase"] for key, stats in counts.items() if not key.endswith("/ordinary"))
    return {
        "counts": counts, "unique_assessed_ordinary_works": len(controls),
        "unique_flagged_ordinary_works": len(false_works), "exact_added_ranges": added,
        "unique_edited_works_hit": len(hit_works), "generators_with_hits": sorted(hit_families),
        "passes_local_trial_rule": (bool(controls) and len(false_works) * 5 <= len(controls)
                                    and added >= 4 and len(hit_works) >= 3 and len(hit_families) >= 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "score"))
    parser.add_argument("root", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases, inputs = load_cases(args.root)
    mapped = list(map_cases(cases))
    metadata = {"schema": "chipmunk.writing-separate-revision-transfer/1", "method": METHOD, "inputs": inputs,
                "runner_sha256": sha(Path(__file__).read_text()), "rows": [row for row, _ in mapped]}
    if args.mode == "freeze":
        with args.lock.open("x") as target:
            json.dump(metadata, target, indent=2)
        print("Frozen cases:", Counter((row["bank"], bool(items)) for row, items in mapped))
        return

    if metadata != json.loads(args.lock.read_text()):
        raise ValueError("The roster or runner differs from the freeze")
    if args.output is None:
        parser.error("score requires --output")
    import torch

    torch.set_num_threads(2)
    cache = {}
    for row, candidates in mapped:
        if not candidates:
            continue
        unseen = [item for item in candidates if (item.earlier, item.later) not in cache]
        if unseen:
            readings = paired.score_separate(unseen) or [None] * len(unseen)
            for item, reading in zip(unseen, readings, strict=True):
                cache[item.earlier, item.later] = reading
        responses = [cache[item.earlier, item.later] for item in candidates]
        result = paired.multiple_result(candidates, responses)
        row["responses"] = responses
        row["statuses"] = [item["status"] for item in result["revisions"]]
    metadata["summary"] = summarize(metadata["rows"])
    with args.output.open("x") as target:
        json.dump(metadata, target, indent=2)
        target.write("\n")
    print(json.dumps(metadata["summary"], indent=2))


if __name__ == "__main__":
    main()
