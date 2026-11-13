"""Recompute the article position trial from the frozen measurements."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from math import isfinite
from pathlib import Path

SCHEMA = "chipmunk.writing-long-position-public/1"
SHORT_PASSAGE_POINT = 2.7579683180027486
SHA256 = re.compile(r"[0-9a-f]{64}")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def select(case, point):
    measured = sorted(
        (x for x in case["review"] if x["contrast"] is not None), key=lambda x: x["contrast"], reverse=True
    )
    if not measured or measured[0]["contrast"] <= point:
        return None
    if len(measured) > 1 and measured[0]["contrast"] == measured[1]["contrast"]:
        return None
    return measured[0]["idx"]


def baseline(case):
    return [i for i, row in enumerate(case["native_current"]) if row["score"] >= 0.5][:6]


def useful(case, idx):
    if not case["truth"]:
        return False
    hit = next(i for i, row in enumerate(case["positions"]) if row[:2] == case["truth"])
    indices = range(max(0, idx - 1), min(len(case["positions"]), idx + 2))
    if hit not in indices:
        return False
    words = case["nonspace"]
    return 10 * words[hit] >= 3 * sum(words[i] for i in indices)


def select_case(case, point, *, allow_punctuation_unit=False, allow_touching=False):
    positions = case["positions"]
    assert 9 <= len(positions) <= 25 and 3 <= len({p for _, _, p, _ in positions}) <= 5
    assert 160 <= sum(n for *_, n in positions) <= 680 or case["truth"] is not None
    assert len(case["native_current"]) == len(positions) == len(case["nonspace"])
    assert all(
        0 <= a < b <= case["characters"]
        and (0 < n <= b - a or (allow_punctuation_unit and n == 0 and m == 1))
        and 0 < m <= b - a
        for (a, b, _, n), m in zip(positions, case["nonspace"])
    )
    assert all(
        (
            x[1] < y[0]
            or (x[1] == y[0] and (allow_touching or (allow_punctuation_unit and (x[3] == 0 or y[3] == 0))))
        )
        and x[2] <= y[2]
        for x, y in zip(positions, positions[1:])
    )
    assert all(
        isfinite(r["score"]) and 0 <= r["score"] <= 1 and SHA256.fullmatch(r["input_sha256"])
        for r in case["native_current"]
    )
    seen = set()
    for x in case["review"]:
        i = x["idx"]
        assert 0 < i < len(positions) - 1 and i not in seen
        seen.add(i)
        assert positions[i][:2] == x["span"] and all(p[2] == x["paragraph"] for p in positions[i - 1 : i + 2])
        assert 12 <= positions[i][3] <= 65 and all(7 <= positions[k][3] <= 80 for k in (i - 1, i + 1))
        assert len(x["responses"]) == 3
        for r in x["responses"]:
            if r is not None:
                assert isfinite(r["raw"]) and r["focus_tokens"] > 0 and r["neighbor_tokens"] > 0
                assert r["focus_tokens"] + r["neighbor_tokens"] <= 512
                assert SHA256.fullmatch(r["input_sha256"])
        expected = None
        if all(x["responses"]):
            a, b, c = (r["raw"] for r in x["responses"])
            expected = b - max(a, c)
        assert x["contrast"] == expected
    truth = case["truth"]
    if truth is not None:
        assert any(positions[i][:2] == truth for i in seen)
    current = baseline(case)
    new = select(case, point)
    return {
        "current": current,
        "current_useful": any(useful(case, i) for i in current),
        "position": new,
        "position_useful": new is not None and positions[new][:2] == truth,
    }


def compare(cases, point, *, allow_punctuation_unit=False, allow_touching=False):
    result = {}
    for publisher in sorted({c["publisher"] for c in cases}):
        rows = [
            (
                c,
                select_case(
                    c, point, allow_punctuation_unit=allow_punctuation_unit, allow_touching=allow_touching
                ),
            )
            for c in cases
            if c["publisher"] == publisher
        ]
        controls = [(c, o) for c, o in rows if c["variant"] in ("before", "after")]
        positives = [(c, o) for c, o in rows if c["truth"]]
        ordinary = {c["group"] for c, _ in controls}
        current_groups = {c["group"] for c, o in controls if o["current"]}
        new_groups = {c["group"] for c, o in controls if o["position"] is not None}
        useful_groups = {c["group"] for c, o in positives if o["position_useful"] and not o["current_useful"]}
        counts = {
            "ordinary_work_groups": len(ordinary),
            "current_ordinary_groups": len(current_groups),
            "position_ordinary_groups": len(new_groups),
            "union_ordinary_groups": len(current_groups | new_groups),
            "additional_ordinary_groups": sorted(new_groups - current_groups),
            "additional_useful_works": sorted(useful_groups),
            "by_variant": {},
        }
        for variant in sorted({c["variant"] for c, _ in rows}):
            values = [o for c, o in rows if c["variant"] == variant]
            counts["by_variant"][variant] = {
                "views": len(values),
                "current_flagged": sum(bool(o["current"]) for o in values),
                "position_flagged": sum(o["position"] is not None for o in values),
                "current_useful": sum(o["current_useful"] for o in values),
                "position_useful": sum(o["position_useful"] for o in values),
                "union_useful": sum(o["current_useful"] or o["position_useful"] for o in values),
                "additional_useful": sum(o["position_useful"] and not o["current_useful"] for o in values),
            }
        result[publisher] = counts
    return result


def verify(receipt):
    assert receipt["schema"] == SCHEMA
    commitments = {row["file"]: row["sha256"] for row in receipt["commitments"]}
    assert len(commitments) == len(receipt["commitments"]) and all(
        SHA256.fullmatch(v) for v in commitments.values()
    )
    admission = receipt["admission"]
    assert admission["detector_runs"] == 0 and admission["held_distinct"] == 12
    assert admission["case_sha256"] == commitments["cases.json"]
    assert admission["freeze_sha256"] == commitments["generation-freeze.json"]
    assert admission["generator_counts"] == {"granite": 12, "qwen": 12}
    assert len({s["group"] for s in receipt["sources"]}) == 26
    by_case = {(r["group"], r["variant"]): r for r in receipt["cases"]}
    assert len(by_case) == len(receipt["cases"]) == 76
    for row in receipt["sources"]:
        group = row["group"]
        for version in ("before", "after"):
            r = by_case[group, version]
            assert r["publisher"] == row["publisher"] and r["role"] == row["role"]
            assert r["text_sha256"] == row["views"][version]["text_sha256"] and r["truth"] is None
        if row["role"] == "held":
            for name in ("qwen", "granite"):
                candidate = by_case[group, name]
                assert candidate["role"] == "held" and candidate["publisher"] == row["publisher"]
                assert candidate["truth"][0] == row["views"]["before"]["focus"][0]
    phases = defaultdict(list)
    for row in receipt["cases"]:
        phases[row["role"]].append(row)
    frozen = receipt["point"]["original_point"]
    assert frozen == receipt["design"]["candidate_point"] == SHORT_PASSAGE_POINT
    assert receipt["point"]["scores_sha256"] == commitments["calibration.json"]
    assert receipt["point"]["freeze_sha256"] == commitments["score-freeze.json"]
    per = defaultdict(list)
    for case in phases["calibration"]:
        select_case(case, frozen)
        per[case["group"]].extend(v["contrast"] for v in case["review"] if v["contrast"] is not None)
    maxima = {group: max(value) for group, value in per.items()}
    calibrated = sorted(maxima.values())[-2]
    assert calibrated == receipt["point"]["second_highest_genuine_work_max"]
    assert maxima == receipt["point"]["per_work_maxima"]
    p = max(calibrated, frozen)
    assert p == receipt["point"]["operating_point"]
    held = phases["held"]
    outcomes = {"original_point": compare(held, frozen), "document_point": compare(held, p)}
    assert outcomes == receipt["outcomes"]
    positive = sum(len(group["additional_useful_works"]) for group in outcomes["document_point"].values())
    negative = sum(len(group["additional_ordinary_groups"]) for group in outcomes["document_point"].values())
    assert receipt["advance"] == (positive >= 2 and negative <= 1)
    return outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--native-packets", type=Path, help="Verify the retained frozen packet digests")
    args = parser.parse_args()
    receipt = read(args.receipt)
    if args.native_packets is not None:
        for row in receipt["commitments"]:
            name = row["file"]
            assert Path(name).name == name and sha(args.native_packets / name) == row["sha256"], name
    print(json.dumps(verify(receipt), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
