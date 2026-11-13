"""Build and independently verify a portable twelve-second audio result."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.audio-clean-reference-duration-review/1"
KINDS = ("synthetic_tts", "synthetic_conversion", "ordinary_replacement")
MISALIGNED = ("wrong_reader", "clock_shifted", "partial_match")
NORMAL = ("untouched", "local_gain", "ordinary_tts", "ordinary_conversion", "global_eq")
REASONS = ("reference_match_not_supported", "local_support_not_available", "no_local_change", "local_change")


def require(test, reason):
    if not test:
        raise ValueError(reason)


def useful(span, truth):
    a, b = span
    c, d = truth
    overlap = max(0, min(b, d) - max(a, c))
    return overlap / (d - c) >= .5 and overlap / (max(b, d) - min(a, c)) >= .3


def outcomes(windows, point):
    require(len(windows) == 6 and [w[0] for w in windows] == list(range(0, 96001, 19200)),
            "Unexpected source windows")
    for source, reason, result in windows:
        require(reason in REASONS and bool(result) == (reason == "local_change"), "Invalid assessment status")
        if result is not None:
            score, a, b = result
            require(source + 11520 <= a < b <= source + 84480 and b - a == 11520
                    and (a - source) % 1280 == 0 and score > point, "Window location or score differs")
    admission = [reason != "reference_match_not_supported" for _, reason, _ in windows]
    admitted = sum(admission) >= 4 and admission[0] and admission[-1]
    candidates = [result for _, _, result in windows if result]
    strongest = max(candidates, key=lambda row: row[0], default=None) if admitted else None
    support = []
    if admitted:
        for before, after in zip(windows, windows[1:]):
            left, right = before[2], after[2]
            if not (left and right):
                continue
            earlier, later = sorted((left, right), key=lambda row: row[1])
            if earlier[2] - later[1] < 5760:
                continue
            retained = left if left[0] >= right[0] else right
            support.append((min(left[0], right[0]), -retained[1], retained, [before[0], after[0]]))
    best = max(support, key=lambda entry: entry[:2], default=None)
    return admission[0], admitted, windows[0][2], strongest, None if best is None else best[2], None if best is None else best[3]


def verify(document):
    require(document["schema"] == SCHEMA and document["point"] == .12831161, "Unexpected frozen point")
    rows = document["rows"]
    require(len(rows) == 495 and sum(r["role"] == "assessment" for r in rows) == 297, "Unexpected source roster")
    summary = {}
    for role in ("development", "assessment"):
        summary[role] = {}
        for level in (0, -5, -10):
            selected = [row for row in rows if row["role"] == role and row["level"] == level]
            counts = {arm: Counter() for arm in ("first_only", "single_window", "corroborated")}
            readers = {arm: set() for arm in counts}
            zones = {arm: set() for arm in counts}
            for row in selected:
                first_ok, admitted, head, single, paired, witnesses = outcomes(row["windows"], document["point"])
                require([first_ok, admitted, head, single, paired, witnesses] == row["readout"], "Native window decisions differ")
                kind = row["kind"]
                for arm, eligible, observation in zip(counts, (first_ok, admitted, admitted), (head, single, paired), strict=True):
                    total = counts[arm]
                    total[f"planned_{kind}"] += 1
                    total[f"admitted_{kind}"] += int(eligible)
                    if observation is None:
                        continue
                    total[f"flag_{kind}"] += 1
                    if kind in KINDS:
                        if useful(observation[1:], row["truth"]):
                            total[f"localized_{kind}"] += 1
                            if kind.startswith("synthetic"):
                                readers[arm].add(row["reader"])
                                zones[arm].add(row["zone"])
                        else:
                            total["misplaced_positive"] += 1
            summary[role][str(level)] = {arm: {"counts": dict(total), "synthetic_reader_groups": len(readers[arm]),
                                              "synthetic_edit_zones": len(zones[arm])} for arm, total in counts.items()}
    gate = summary["assessment"]["-5"]["corroborated"]
    counts = gate["counts"]
    summary["gate_pass"] = (sum(counts.get(f"localized_{kind}", 0) for kind in KINDS[:2]) >= 4
                            and gate["synthetic_reader_groups"] >= 2 and gate["synthetic_edit_zones"] >= 2
                            and sum(counts.get(f"flag_{kind}", 0) for kind in NORMAL) <= 2
                            and counts.get("misplaced_positive", 0) == 0
                            and all(summary["assessment"][str(level)]["corroborated"]["counts"][f"admitted_{kind}"] == 0
                                    for level in (0, -5, -10) for kind in MISALIGNED))
    return summary


def package(path):
    raw = path.read_bytes()
    source = json.loads(raw)
    rows = []
    for row in source["rows"]:
        data = row["comparison"]

        def compact(result):
            return [result["relative_drop"], *result["samples"]] if result else None

        windows = [[w["start"], w["reason"], compact(w["result"])] for w in data["windows"]]
        first_ok = windows[0][1] != "reference_match_not_supported"
        seen = (data["first_only"], data["single_window"], data["corroborated"])
        starts = None if seen[-1] is None else seen[-1]["supporting_window_starts"]
        rows.append({k: row[k] for k in ("role", "level", "kind", "zone", "truth", "earlier_sha256", "submitted_sha256")}
                    | {"reader": row["group"], "windows": windows,
                       "readout": [first_ok, data["admitted"], *(compact(r) for r in seen), starts]})
    return {"schema": SCHEMA, "point": source["freeze"]["point"], "method": source["freeze"]["method"],
            "freeze_sha256": hashlib.sha256(path.with_name("freeze.json").read_bytes()).hexdigest(),
            "native_receipt_sha256": hashlib.sha256(raw).hexdigest(), "rows": rows, "summary": source["summary"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--from-native", type=Path)
    args = parser.parse_args()
    try:
        doc = package(args.from_native) if args.from_native else json.loads(args.receipt.read_bytes())
        require(doc["summary"] == verify(doc), "Independent localization or admission accounting differs")
        if args.from_native:
            args.receipt.write_text(json.dumps(doc, sort_keys=True, indent=2, allow_nan=False) + "\n")
        print(f"Verified {len(doc['rows'])} source-bound window schedules, decisions and location counts.")
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
