"""Replay a frozen F1000Research longer-passage comparison without model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from .review_writing_long_position import compare, read, select_case

SCHEMA = "chipmunk.writing-f1000-position-transfer/1"
ROOT = Path(__file__).resolve().parents[2]
PANELS = ROOT / "docs/evaluation-panels"
SOURCE = PANELS / "writing-f1000-revision-spans-2026-11-13.json"
PRIOR = PANELS / "writing-long-position-transfer-2026-11-12.json"
POINTS = {"short_passage": 2.7579683180027486, "long_document": 3.6821095168341915}
GENERATORS = {"qwen": 2845958, "granite": 1845115}
ORDINARY = {"before", "after"}
SHA = re.compile(r"[a-f0-9]{64}")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check(valid, reason):
    if not valid:
        raise ValueError(reason)


def control_matches(case, view):
    if case["text_sha256"] != view["text_sha256"] or case["positions"] != view["positions"]:
        return False
    if [row["input_sha256"] for row in case["native_current"]] != view["native_input_sha256"]:
        return False
    if [row["span"] for row in case["review"]] != [row["focus"] for row in view["position_contexts"]]:
        return False
    return all(
        reading is None or reading["input_sha256"] == expected
        for focus, native in zip(case["review"], view["position_contexts"], strict=True)
        for reading, expected in zip(focus["responses"], native["inputs_sha256"], strict=True)
    )


def generated_matches(case, earlier):
    truth = case["truth"]
    if not truth or truth[0] != earlier["target"][0] or case["text_sha256"] == earlier["text_sha256"]:
        return False
    old = earlier["positions"]
    new = case["positions"]
    if len(old) != len(new):
        return False
    idx = next(i for i, unit in enumerate(old) if unit[:2] == earlier["target"])
    delta = truth[1] - earlier["target"][1]
    return (
        new[idx][:3] == [*truth, old[idx][2]]
        and new[:idx] == old[:idx]
        and new[idx + 1 :] == [[lo + delta, hi + delta, p, n] for lo, hi, p, n in old[idx + 1 :]]
        and len(case["review"]) >= 1
        and any(row["span"] == truth for row in case["review"])
    )


def outcomes(cases):
    result = {}
    for name, point in POINTS.items():
        totals = compare(cases, point, allow_punctuation_unit=True)["f1000-2018"]
        locations = []
        for case in cases:
            selection = select_case(case, point, allow_punctuation_unit=True)
            index = selection["position"]
            positions = case["positions"]
            if index is None and not selection["current"]:
                continue
            locations.append(
                {
                    "group": case["group"],
                    "variant": case["variant"],
                    "known_edit": case["truth"],
                    "candidate": positions[index][:2] if index is not None else None,
                    "current_windows": [
                        [positions[max(i - 1, 0)][0], positions[min(i + 1, len(positions) - 1)][1]]
                        for i in selection["current"]
                    ],
                }
            )
        result[name] = {"counts": totals, "locations": locations}
    return result


def verify(receipt):
    source = read(SOURCE)
    cases = receipt["cases"]
    check(receipt["schema"] == SCHEMA, "Unknown receipt format")
    check(receipt["source_sha256"] == digest(SOURCE), "Archived source receipt differs")
    check(receipt["prior_sha256"] == digest(PRIOR), "Previous comparison receipt differs")
    check(
        receipt["points"] == source["long_study_coverage"]["frozen_comparison_points"] == POINTS,
        "The frozen comparison points differ",
    )
    check(
        set(receipt["commitments"]) == {"generation_freeze", "score_freeze", "raw_cases", "measurements"}
        and all(SHA.fullmatch(sha) for sha in receipt["commitments"].values()),
        "Invalid private commitments",
    )
    admission = receipt["admission"]
    check(
        admission["detector_runs"] == 0
        and admission["case_sha256"] == receipt["commitments"]["raw_cases"]
        and admission["freeze_sha256"] == receipt["commitments"]["generation_freeze"],
        "Invalid generation freeze",
    )
    by_case = {(row["group"], row["variant"]): row for row in cases}
    dispositions = {(row["group"], row["family"]): row for row in admission["dispositions"]}
    check(
        len(by_case) == len(cases) == admission["total_views"]
        and len(dispositions) == len(admission["dispositions"]) == 18,
        "Repeated or missing cases",
    )
    admitted_counts = dict.fromkeys(GENERATORS, 0)
    admitted_groups = set()
    expected_keys = set()
    for idx, item in enumerate(source["sources"]):
        group = item["group"]
        for side in ORDINARY:
            key = group, side
            expected_keys.add(key)
            case = by_case[key]
            check(
                case["truth"] is None and control_matches(case, item["views"][side]),
                "Control source inputs differ",
            )
        for family, seed_base in GENERATORS.items():
            key = group, family
            disposition = dispositions[key]
            check(SHA.fullmatch(disposition["generator_packet_sha256"]), "Invalid generation packet hash")
            if not disposition["admitted"]:
                check(
                    key not in by_case and disposition["selected_seed"] is None,
                    "Excluded generation was scored",
                )
                continue
            expected_keys.add(key)
            check(
                disposition["selected_seed"] in range(seed_base + 3 * idx, seed_base + 3 * idx + 3),
                "A scored rewrite used a seed outside the frozen schedule",
            )
            check(
                generated_matches(by_case[key], item["views"]["before"]), "Generated source positions differ"
            )
            admitted_groups.add(group)
            admitted_counts[family] += 1
    check(set(by_case) == expected_keys, "Unexpected measured cases")
    check(
        len(admitted_groups) == admission["held_distinct"] >= 8
        and admission["generator_counts"] == admitted_counts
        and min(admitted_counts.values()) >= 6,
        "The frozen admission minimum was not satisfied",
    )
    for row in cases:
        check(
            row["publisher"] == "f1000-2018" and row["role"] == "held" and SHA.fullmatch(row["text_sha256"]),
            "Unexpected measurement source",
        )
        select_case(row, POINTS["long_document"], allow_punctuation_unit=True)
    measured_inputs = {entry["input_sha256"] for row in cases for entry in row["native_current"]}
    measured_inputs.update(
        entry["input_sha256"]
        for row in cases
        for focus in row["review"]
        for entry in focus["responses"]
        if entry is not None
    )
    check(
        len(measured_inputs) == receipt["unique_measured_inputs"],
        "The number of distinct model inputs differs",
    )
    result = outcomes(cases)
    counts = result["long_document"]["counts"]
    advance = len(counts["additional_useful_works"]) >= 2 and len(counts["additional_ordinary_groups"]) <= 1
    check(
        receipt["advance"] is advance and receipt["outcomes"] == result, "Saved outcomes differ from replay"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(read(args.receipt)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
