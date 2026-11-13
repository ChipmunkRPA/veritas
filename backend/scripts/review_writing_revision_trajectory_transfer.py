"""Verify the sealed writing comparison and optionally rebuild the publisher source positions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_revision_trajectory_transfer as study


def exact_trajectory(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    stages = {(family, version) for family in study.SHARDS for version in study.STAGES}
    unchanged = Counter()
    for values in groups.values():
        originals = [row for row in values if (row["family"], row["version"]) == ("original", "v0")]
        study.require(len(originals) == 1 and not originals[0]["gold"], "An ordinary source is missing or labeled")
        original = originals[0]
        edited = [row for row in values if row is not original]
        study.require(all(row["role"] == original["role"] and row["source"] == original["source"] for row in values),
                      "A source trajectory crosses roles")
        expected = stages if original["role"] == "held" else set()
        study.require(len(edited) == len(expected) and {(r["family"], r["version"]) for r in edited} == expected,
                      "Incomplete matched source trajectory")
        for row in edited:
            study.require(row["truth_characters"] > 0, "Missing annotated operation span")
            if row["text_sha256"] == original["text_sha256"]:
                unchanged[row["family"], row["version"]] += 1
    return {f"{a}/{b}": count for (a, b), count in sorted(unchanged.items())}


def overlaps_and_input_replay(receipt, cache):
    rebuilt, admissions = study.select_roster(cache)
    study.require(study.sha(study.serialize(rebuilt)) == receipt["freeze"]["roster_sha256"]
                  and admissions == receipt["freeze"]["admissions"], "Publisher roster differs from the freeze")
    saved = {r["record_id"]: r for r in receipt["rows"]}
    study.require(set(saved) == {r["record_id"] for r in rebuilt}, "Scored roster differs from publisher sources")
    by_role = defaultdict(list)
    focuses = 0
    for row in rebuilt:
        score = saved[row["record_id"]]
        native = {key: value for key, value in row.items() if key not in {"text", "context", "isolated"}}
        public = {key: value for key, value in score.items() if key not in {"primary", "independent"}}
        study.require(public == native, "Source range, role or per-span intersection differs from the publisher replay")
        for name, view in (("primary", "context"), ("independent", "isolated")):
            study.require([study.sha(text) for text in row[view]] == [digest for digest, _ in score[name]],
                          "Recorded model input does not match the native source view")
        if row["family"] == "original":
            by_role[row["role"]].append(row["text"])
        focuses += len(row["ranges"])
    # Check that nearby report variants did not leak verbatim prose across roles.
    import re

    shingles = {}
    for role, texts in by_role.items():
        fragments = set()
        for text in texts:
            words = re.findall(r"\b[\w'-]+\b", text.casefold())
            fragments.update(tuple(words[start:start + 12]) for start in range(len(words) - 11))
        shingles[role] = fragments
    shared = len(shingles["fit"] & shingles["held"])
    study.require(shared == 0, "Identical 12-word sequences cross the fit and held roles")
    return {"reconstructed_rows": len(rebuilt), "native_focuses": focuses, "cross_role_shared_12_word_sequences": shared,
            "roster_sha256": receipt["freeze"]["roster_sha256"]}


def verify(receipt, cache=None):
    output = study.verify_receipt(receipt)
    duplicates = exact_trajectory(receipt["rows"])
    original_hashes = {r["group"]: r["text_sha256"] for r in receipt["rows"] if r["family"] == "original"}
    changed_rows = [r for r in receipt["rows"] if r["family"] == "original" or r["text_sha256"] != original_hashes[r["group"]]]
    changed_results = study.all_outcomes(changed_rows, receipt["points"]["values"], receipt["freeze"]["prior_isolated_odds"])
    output.update({"identical_to_original": duplicates, "text_changed_only": changed_results["summary"]})
    if cache:
        output["publisher_replay"] = overlaps_and_input_replay(receipt, cache)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--cache", type=Path, help="Directory containing both pinned CSVs and their publisher metadata")
    args = parser.parse_args()
    print(json.dumps(verify(study.read(args.receipt), args.cache), indent=2, sort_keys=True))
