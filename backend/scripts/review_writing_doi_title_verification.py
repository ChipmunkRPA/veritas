# SPDX-License-Identifier: Apache-2.0
"""Recount the text-free local DOI and title study using the saved measurements."""

from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from collections import Counter
from gzip import open as open_gzip
from pathlib import Path

SCHEMA = "chipmunk.writing-doi-title-study/1"
POLICY = {
    "title_minimum_words": 6,
    "title_maximum_words": 65,
    "minimum_citation_characters": 32,
    "maximum_citation_characters": 640,
    "required_visible_dois": 1,
    "start_title_before_token": 48,
    "maximum_word_set_overlap": [9, 20],
    "affects_overall_confidence": False,
}
SPLITS = ("development", "held")
ROLES = ("ordinary", "replacement", "unknown", "multiple", "no_doi")
HASH = re.compile(r"[0-9a-f]{64}")


def require(valid, message):
    if not valid:
        raise ValueError(message)


def sha(value):
    return isinstance(value, str) and HASH.fullmatch(value) is not None


def span(value, size):
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(type(pos) is int for pos in value)
        and 0 <= value[0] < value[1] <= size
    )


def classify(row):
    fields = row["numeric"]
    chars, visible, resolved = (fields[key] for key in ("characters", "visible_dois", "resolved"))
    identities, related, overlap = (
        fields[key] for key in ("candidate_identity_sha256s", "linked_version", "destination_overlap")
    )
    observed = row["observed_doi_sha256"]
    require(
        type(chars) is int and chars > 0 and type(visible) is int and visible >= 0 and type(resolved) is bool,
        "Invalid measured citation geometry",
    )
    require(
        identities == sorted(set(identities)) and all(map(sha, identities)),
        "Invalid title and author match commitments",
    )
    supported = 32 <= chars <= 640 and visible == 1
    if supported:
        require(sha(observed) and span(row["observed_doi_range"], chars), "Missing measured DOI position")
    else:
        require(
            not resolved and observed is None and row["observed_doi_range"] is None,
            "Unsupported citation claims a DOI",
        )
    if not resolved:
        require(
            not identities and related is None and not overlap, "Unresolved citation recorded comparisons"
        )
        return "unknown_in_snapshot" if supported else "unsupported_geometry"
    if identities == [observed] or len(identities) != 1:
        require(
            related is None and not overlap, "Undetermined citation recorded a competing title comparison"
        )
        return "consistent" if identities == [observed] else "no_unique_visible_reference"
    require(
        type(related) is bool
        and all(
            len(pair) == 2
            and all(type(n) is int for n in pair)
            and 0 <= pair[0] <= pair[1] <= 130
            and pair[1] > 0
            for pair in overlap
        ),
        "Invalid relationship or title comparisons",
    )
    if related:
        return "version_relation"
    if not overlap or any(20 * common > 9 * total for common, total in overlap):
        return "destination_not_distinct"
    return "different_registered_work"


def count_group(group, rows, total):
    ordinary = set()
    keyed = {}
    for row in rows:
        ident, role = row["citation"], row["role"]
        require(
            type(ident) is int and 1 <= ident <= total and role in ROLES and (ident, role) not in keyed,
            "Invalid citation identity",
        )
        keyed[ident, role] = row
        state = classify(row)
        require(state == row["state"], "Recorded decision differs from the saved measurements")
        chars = row["numeric"]["characters"]
        reported = state == "different_registered_work"
        if reported:
            require(
                row["numeric"]["candidate_identity_sha256s"] == [row["matched_doi_sha256"]]
                and span(row["matched_title_range"], chars)
                and row["matched_title_range"][1] <= row["observed_doi_range"][0],
                "Unbound registered-work location",
            )
        else:
            require(
                row["matched_doi_sha256"] is None and row["matched_title_range"] is None,
                "An unreported comparison contains a claimed source range",
            )
        entry = group[role]
        entry["cases"] += 1
        entry["registry_only_resolved"] += row["numeric"]["resolved"]
        entry["candidate_status"][state] += 1
        if role == "ordinary":
            ordinary.add(ident)
        if role != "replacement":
            require(row["truth"] is None, "An unmodified or unsuitable citation has replacement ground truth")
            continue
        printed, intended, observed_span = row["truth"]
        require(
            sha(printed) and sha(intended) and printed != intended and span(observed_span, chars),
            "Invalid frozen citation replacement",
        )
        entry["candidate_exact_replacements"] += int(
            reported
            and observed_span == row["observed_doi_range"]
            and printed == row["observed_doi_sha256"]
            and intended == row["matched_doi_sha256"]
        )
    require(ordinary == set(range(1, total + 1)), "Incomplete genuine citation control group")
    return keyed


def native_replay(items, groups):
    pairs = set()
    counts = {"docx_files": 0, "doi_positions": 0, "matched_title_positions": 0}
    for item in items:
        code, variant = item["group"], item["variant"]
        require(
            code in groups
            and variant in {"ordinary", "replacement"}
            and (code, variant) not in pairs
            and sha(item["docx_sha256"])
            and sha(item["txt_sha256"]),
            "Invalid DOCX comparison identity",
        )
        pairs.add((code, variant))
        rows = groups[code]
        expected = {ident: row for (ident, role), row in rows.items() if role == "ordinary"}
        if variant == "replacement":
            expected.update({ident: row for (ident, role), row in rows.items() if role == "replacement"})
        pending = {ident for ident, row in expected.items() if row["observed_doi_range"]}
        for location in item["native_positions"]:
            ident = location["citation"]
            require(
                ident in pending and location["body_path"] == [ident + 1], "Unexpected original Word passage"
            )
            pending.remove(ident)
            row = expected[ident]
            doi, title = row["observed_doi_range"], row["matched_title_range"]
            require(
                location["state"] == row["state"]
                and location["docx_doi_ranges"] == [doi]
                and location["txt_doi_range_in_paragraph"] == doi
                and location["txt_title_range_in_paragraph"] == title
                and location["docx_title_ranges"] == ([title] if title else None),
                "Native DOCX and TXT ranges differ",
            )
            counts["doi_positions"] += 1
            counts["matched_title_positions"] += int(title is not None)
        require(not pending, "Missing native Word positions")
    counts["docx_files"] = len(pairs)
    return counts


def replay(receipt):
    require(receipt["schema"] == SCHEMA and receipt["policy"] == POLICY, "Unknown citation study policy")
    registry = receipt["crossref"]
    batches = registry["batches"]
    require(
        all(
            sha(row["request_sha256"])
            and sha(row["response_sha256"])
            and 0 <= row["matched_records"] <= row["requests"] <= 24
            for row in batches
        )
        and sum(row["requests"] for row in batches) == registry["lookup_count"]
        and sum(row["matched_records"] for row in batches) == registry["matched_records"],
        "Crossref coverage counts differ",
    )
    results = {
        split: {
            role: {
                "cases": 0,
                "registry_only_resolved": 0,
                "candidate_status": Counter(),
                "candidate_exact_replacements": 0,
            }
            for role in ROLES
        }
        for split in SPLITS
    }
    located_groups = {split: 0 for split in SPLITS}
    groups = {}
    for item in receipt["groups"]:
        code, split = item["group"], item["split"]
        require(
            split in SPLITS
            and code not in groups
            and sha(item["publisher_jats_sha256"])
            and sha(item["source_doi_sha256"]),
            "Invalid citing work group",
        )
        group = results[split]
        previous = group["replacement"]["candidate_exact_replacements"]
        groups[code] = count_group(group, item["rows"], item["citations"])
        located_groups[split] += int(group["replacement"]["candidate_exact_replacements"] > previous)
    for split in SPLITS:
        results[split]["groups_with_exact_replacement"] = located_groups[split]
    require(results == receipt["counts"], "Published citation results differ from the independent recount")
    return {
        "counts": results,
        "crossref_registered": registry["matched_records"],
        "crossref_queried": registry["lookup_count"],
        **native_replay(receipt["docx_replays"], groups),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    with open_gzip(args.receipt, "rt", encoding="utf-8") as source:
        print(json.dumps(replay(json.load(source)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
