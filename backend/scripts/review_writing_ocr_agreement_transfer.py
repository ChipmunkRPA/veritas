"""Replay the held scan agreement comparison from public, text-free receipts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCHEMA = "chipmunk.writing-ocr-agreement-transfer/1"
ORDINARY = {"original", "human_splice"}
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probability(encoded):
    value = float.fromhex(encoded)
    require(0 <= value <= 1 and encoded == value.hex(), "Invalid exact classifier response")
    return value


def positive(response, digest):
    require(response["input_sha256"] == digest and type(response["complete"]) is bool, "Response binding changed")
    return probability(response["score"]) >= .5 and response["complete"]


def useful(counts):
    truth, mapped, read, intersection = (counts[key] for key in ("truth", "mapped_truth", "checked", "intersection"))
    union = truth + read - intersection
    require(all(type(count) is int for count in (truth, mapped, read, intersection))
            and 0 <= intersection <= min(mapped, read) and mapped <= truth and truth > 0,
            "Invalid location overlap")
    expected = 10 * intersection >= 3 * union and 2 * intersection >= truth
    require(counts["useful"] is expected, "Useful location changed")
    return expected


def verify_windows(case):
    windows = case["windows"]
    previous = 0
    eligible, assessed, primary_marks, second_marks = [], [], [], []
    for index, window in enumerate(windows):
        focus, spans, digest = window["focus"], window["spans"], window["sha256"]
        require(focus in spans and len(spans) in {2, 3} and previous <= focus[0] < focus[1]
                and all(0 <= start < end for start, end in spans)
                and all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))
                and HEX_DIGEST.fullmatch(digest), "Invalid source window")
        previous = focus[1]
        page = window["page"]
        if page is not None:
            require(case["qualified"] and type(page) is int and 1 <= page <= len(case["page_qualities"]),
                    "Source page changed")
        location = useful(window["truth_overlap"])
        first, second = window["primary"], window["second"]
        require((first is not None) == case["qualified"], "Primary scope changed")
        if first is not None and positive(first, digest):
            primary_marks.append(location)
            if page is not None:
                eligible.append(index)
        if second is not None:
            assessed.append(index)
            if positive(second, digest):
                second_marks.append(location)
    require(assessed == eligible[:6], "Second check did not use the frozen eligible contexts")
    return primary_marks, second_marks


def review(receipt):
    require(receipt["schema"] == SCHEMA, "Unknown review schema")
    cases = receipt["cases"]
    by_index = {case["index"]: case for case in cases}
    groups = {case["group"] for case in cases}
    earlier = set(receipt["earlier_scan_groups"])
    require(len(cases) == len(by_index) == 52 and len(groups) == len(earlier) == 14 and not groups & earlier,
            "Held scan roster differs from the frozen split")
    commits = {item["case"]: item for item in receipt["source_commitments"]}
    require(set(commits) == set(by_index), "Source commitments do not cover every scan")
    seen_scans = set()
    outcome = dict.fromkeys(("scope_eligible", "exact_transcripts", "source_bounded_contexts", "primary_inputs", "second_inputs",
                             "second_assessed_cases", "generated_scans", "ordinary_scans", "generated_useful_primary",
                             "generated_useful_second", "generated_useful_added", "generated_flagged_only_elsewhere_second",
                             "ordinary_flags_primary", "ordinary_flags_second", "native_generated_useful", "native_ordinary_flags"), 0)
    native_false, second_false, second_useful, families = set(), set(), set(), set()
    for case in cases:
        scan_hash = case["scan_sha256"]
        require(scan_hash not in seen_scans and all(HEX_DIGEST.fullmatch(value) for value in (
            scan_hash, case["text_sha256"], case["ocr_text_sha256"], case["source_map_sha256"],
            commits[case["index"]]["native_pdf_sha256"], *commits[case["index"]]["native_contexts"])),
            "Invalid or repeated scan commitment")
        seen_scans.add(scan_hash)
        primary, second = verify_windows(case)
        outcome["scope_eligible"] += case["qualified"]
        outcome["exact_transcripts"] += case["text_sha256"] == case["ocr_text_sha256"]
        outcome["source_bounded_contexts"] += sum(window["page"] is not None for window in case["windows"])
        outcome["primary_inputs"] += sum(window["primary"] is not None for window in case["windows"])
        outcome["second_inputs"] += sum(window["second"] is not None for window in case["windows"])
        outcome["second_assessed_cases"] += any(window["second"] is not None for window in case["windows"])
        if case["kind"] in ORDINARY:
            outcome["ordinary_scans"] += 1
            outcome["ordinary_flags_primary"] += bool(primary)
            outcome["ordinary_flags_second"] += bool(second)
            outcome["native_ordinary_flags"] += bool(case["native_flags"])
            if case["native_flags"]:
                native_false.add(case["group"])
            if second:
                second_false.add(case["group"])
        else:
            outcome["generated_scans"] += 1
            outcome["generated_useful_primary"] += any(primary)
            outcome["generated_useful_second"] += any(second)
            outcome["generated_useful_added"] += any(second) and not any(primary)
            outcome["generated_flagged_only_elsewhere_second"] += bool(second) and not any(second)
            outcome["native_generated_useful"] += case["native_useful"]
            if any(second):
                second_useful.add(case["group"])
                families.add(case["kind"])

    cases_with_dash, newly_positive = set(), 0
    for case in receipt["dash_cases"]:
        item = by_index[case["index"]]
        window = item["windows"][case["window"]]
        require(case["role"] == item["kind"] and case["ocr_sha256"] == window["sha256"]
                and case["native_sha256"] in commits[case["index"]]["native_contexts"]
                and case["ocr_sha256"] not in commits[case["index"]]["native_contexts"]
                and case["index"] not in cases_with_dash, "Dash counterfactual input changed")
        first, second = (probability(case[key]) for key in ("native_response", "ocr_response"))
        before_second, after_second = (probability(case[key]) for key in ("second_native_response", "second_ocr_response"))
        require(second >= .5 and after_second >= .5 and abs(second - probability(window["primary"]["score"])) <= .02
                and abs(after_second - probability(window["second"]["score"])) <= .02,
                "Dash counterfactual disagrees with the recorded scan decision")
        cases_with_dash.add(case["index"])
        newly_positive += first < .5 and before_second >= .5

    new_ordinary = len(second_false - native_false)
    require(len(receipt["controls"]) == 5 and all(control["qualified"] is False for control in receipt["controls"].values()),
            "An excluded scan entered the second check")
    outcome.update(case_count=len(cases), groups=len(groups), ordinary_groups_second=len(second_false),
                   native_ordinary_groups=len(native_false), new_ordinary_groups_second=new_ordinary,
                   second_useful_groups=len(second_useful), second_useful_families=len(families),
                   dash_only_corroborated_cases=len(cases_with_dash), dash_created_primary_positives=newly_positive,
                   numeric_gate=len(second_useful) >= 2 and len(families) >= 2 and new_ordinary <= 1, adopted=False)
    require(outcome == receipt["outcome"], "Held aggregate differs from the recorded decisions")
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True))


if __name__ == "__main__":
    main()
