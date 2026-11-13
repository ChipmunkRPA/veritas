"""Replay the public forum writing ledger from score and source commitments."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from scripts import writing_uswds_revision_transfer as earlier

CC_TERMS = {
    "CC BY-SA 3.0": "https://creativecommons.org/licenses/by-sa/3.0/",
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
}
SCHEMA = "chipmunk.writing-source-local-forum-transfer/1"
FAMILIES = {"before", "after", *earlier.SEEDS}
SITES = {"english", "travel", "superuser", "askubuntu"}
HASH = re.compile(r"[0-9a-f]{64}")
CUTOFF = datetime.fromisoformat("2020-01-01T00:00:00+00:00")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def proposals(row, before):
    methods = earlier.proposal(row, before)
    methods["union"] = sorted({tuple(span) for method in ("primary", "paired_direction") for span in methods[method]})
    methods["agreement"] = [
        window["span"] for window in row["native"]
        if window["span"] in methods["primary"] and float.fromhex(window["modernbert"]) >= earlier.NATIVE_CUTOFF
    ]
    return methods


def valid_span(span, length):
    return (isinstance(span, list) and len(span) == 2 and all(type(i) is int for i in span)
            and 0 <= span[0] < span[1] <= length)


def review_sources(records, frozen):
    groups = {}
    threads = set()
    for source in records:
        group, site, thread = source["group"], source["site"], source["thread_id"]
        require(site in SITES and group.startswith(site + "-") and type(thread) is int and thread > 0,
                "Unknown or invalid source group")
        require(group not in groups and (site, thread) not in threads, "Duplicate source thread")
        require(source["kind"] in {"question", "answer"} and set(source["revisions"]) == {"before", "after"},
                "Unknown source kind or incomplete revisions")
        before, after = (source["revisions"][role] for role in ("before", "after"))
        require(datetime.fromisoformat(before["date"]) < datetime.fromisoformat(after["date"]) < CUTOFF,
                "Revision chronology differs")
        require(before["revision"] != after["revision"] and before["body_sha256"] != after["body_sha256"],
                "Expected a different exact body")
        require(len(source["positions"]) == 2 and all(
            len(position) == 2 and all(type(i) is int and i >= 0 for i in position) for position in source["positions"]
        ), "Invalid paragraph and sentence positions")
        for revision in (before, after):
            require(revision["license"] in CC_TERMS and revision["license_url"] == CC_TERMS[revision["license"]]
                    and revision["attribution"] and HASH.fullmatch(revision["body_sha256"])
                    and revision["publisher_record_sha256"] == frozen["publisher_records"].get(revision["record"]),
                    "Missing exact publisher or license commitment")
        groups[group] = source
        threads.add((site, thread))
    require(len(groups) >= 7, "Too few source groups")
    return groups


def review_inputs(rows, groups, expected_inputs):
    cases = {}
    windows = {}
    for row in rows:
        group, role, length = row["group"], row["family"], row["characters"]
        require(group in groups and role in FAMILIES and (group, role) not in cases, "Unknown or repeated case")
        require(type(length) is int and length > 0 and HASH.fullmatch(row["text_sha256"]), "Invalid case")
        require(bool(row["locations"]) == (role != "before") and all(
            valid_span(span, length) for span in row["locations"]
        ), "Exact changed tokens are absent or out of range")
        spaces, truth = row["spaces"], row["truth_indices"]
        require(spaces == sorted(set(spaces)) and truth == sorted(set(truth))
                and all(type(i) is int and 0 <= i < length for i in (*spaces, *truth))
                and not set(spaces) & set(truth) and bool(truth) == (role in earlier.SEEDS),
                "Invalid known replacement positions")
        native = row["native"]
        require(len(native) >= 3 and native[0]["span"][0] == 0 and native[-1]["span"][1] == length,
                "Missing edge context")
        for window in native:
            digest = window["sha256"]
            require(valid_span(window["span"], length) and HASH.fullmatch(digest), "Native input is outside its file")
            scores = tuple(float.fromhex(window[mode]) for mode in ("tropa", "modernbert"))
            require(all(math.isfinite(value) and 0 <= value <= 1 for value in scores)
                    and windows.get(digest, scores) == scores, "Repeated native input response differs")
            windows[digest] = scores
        cases[group, role] = row
    require(all((group, role) in cases for group in groups for role in ("before", "after")),
            "Missing exact before or after control")
    require(all(sum(role == family for _, role in cases) >= 3 for family in earlier.SEEDS), "Missing generator coverage")
    require(len(windows) == expected_inputs, "Unique native input count differs")
    return cases


def replay(cases, groups):
    outcomes = defaultdict(lambda: defaultdict(Counter))
    transitions = Counter()
    for (group, role), row in cases.items():
        original = cases[group, "before"]
        spaces, truth = set(row["spaces"]), set(row["truth_indices"])
        original_proposals = proposals(original, original)
        for method, spans in proposals(row, original).items():
            combined = [spans] if method == "draft_comparison" else [[span] for span in spans]
            candidates = [{i for start, stop in item for i in range(start, stop)} - spaces for item in combined]
            selected = set().union(*candidates)
            useful = bool(truth) and any(
                2 * len(candidate & truth) >= len(truth)
                and 10 * len(candidate & truth) >= 3 * len(candidate | truth)
                for candidate in candidates
            )
            matched = len(selected & truth)
            outcomes[role][method].update(files=1, flags=bool(selected), useful=useful,
                                          ordinary_characters=len(selected) - matched, labeled_characters=matched)
            transitions[method] += role == "after" and bool(selected) and not original_proposals[method]
    readout = {"false_flag_transitions": dict(transitions), "groups": dict(Counter(row["site"] for row in groups.values()))}
    return outcomes, readout


def review_product_scope(record, cases):
    counts = defaultdict(Counter)
    covered = set()
    for item in record["rows"]:
        group, role = key = item["group"], item["family"]
        require(key in cases and key not in covered and type(item["eligible"]) is bool,
                "Unknown or duplicated product replay row")
        covered.add(key)
        counts[role].update(total=1, eligible=item["eligible"])
        if not item["eligible"]:
            require(set(item) == {"group", "family", "eligible"} and role not in earlier.SEEDS,
                    "Unexpected scope abstention or geometry")
            continue
        before, row = cases[group, "before"], cases[key]
        native = row["native"]
        center, focus = item["legacy_context_index"], item["legacy_focus"]
        require(type(center) is int and 0 < center < len(native) - 1 and valid_span(focus, row["characters"])
                and native[center]["span"][0] < focus[0] < focus[1] < native[center]["span"][1]
                and all(focus[0] <= span[0] < span[1] <= focus[1] for span in row["locations"]),
                "Legacy focus no longer contains the exact changed tokens inside its context")
        paired = {tuple(span) for span in proposals(row, before)["paired_direction"]}
        selected = [index for index, window in enumerate(native) if tuple(window["span"]) in paired]
        interior = [index for index in selected if 0 < index < len(native) - 1 and abs(index - center) <= 1]
        positive = center in selected
        require(item["legacy_status"] == ("joint_increase" if positive else "no_joint_increase")
                and item["interior_joint_contexts"] == interior, "Product replay score arithmetic differs")
        truth, spaces = set(row["truth_indices"]), set(row["spaces"])
        if truth:
            candidate = set(range(*focus)) - spaces
            require(2 * len(candidate & truth) >= len(truth)
                    and 10 * len(candidate & truth) >= 3 * len(candidate | truth),
                    "The shared changed-sentence anchor does not localize the known replacement")
        counts[role].update(positive=positive, verified=1, paired_any=bool(selected), paired_any_interior=bool(interior))
    require(covered == cases.keys() and counts == record["aggregates"], "Product replay counts or coverage differ")
    return counts


def review(path):
    value = json.loads(path.read_bytes())
    frozen = value["freeze"]
    require(value["schema"] == earlier.SCHEMA and frozen["schema"] == SCHEMA
            and earlier.sha(json.dumps(frozen, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                == value["freeze_sha256"]
            and frozen["native_cutoff"] == earlier.NATIVE_CUTOFF
            and frozen["paired_log_odds"] == earlier.PAIRED_LOG_ODDS and frozen["prompt"] == earlier.PROMPT
            and frozen["seeds"] == earlier.SEEDS and frozen["attempts"] == 2
            and all(frozen["protocol"].get(key) == item for key, item in earlier.PROTOCOL.items()),
            "Unknown frozen operating rules")
    groups = review_sources(value["sources"], frozen)
    cases = review_inputs(value["rows"], groups, value["inputs"])
    results, readout = replay(cases, groups)
    require(results == value["results"] and readout == value["readout"], "Aggregate does not match replay")
    return {"source_groups": len(groups), "cases": len(cases), "native_inputs": value["inputs"],
            "results": results, "readout": readout, "product_scope_replay": review_product_scope(value["product_scope_replay"], cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(args.ledger), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
