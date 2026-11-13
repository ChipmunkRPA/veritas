"""Build or check a portable scalar receipt for the blind pitch comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "chipmunk.audio-overlap-pitch-portable/1"


def serial(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def require(test, reason):
    if not test:
        raise ValueError(reason)


def useful(span, truth):
    if span is None or truth is None:
        return False
    start, stop = span
    expected_start, expected_stop = truth
    intersection = max(0, min(stop, expected_stop) - max(start, expected_start))
    return (intersection >= .5 * (expected_stop - expected_start)
            and intersection >= .3 * (max(stop, expected_stop) - min(start, expected_start)))


def scan(rows, arm, point):
    counts = dict(synthetic=0, ordinary=0, synthetic_unassessed=0, ordinary_unassessed=0, useful_before_point=0,
                  useful_flags=0, misplaced_flags=0, ordinary_flags=0, background_flags=0, available_matched_pairs=0,
                  synthetic_above_matched_ordinary=0)
    genuine = {(r["group"], r["kind"]): r["scores"][arm] for r in rows if not r["kind"].startswith("synthetic")}
    for row in rows:
        positive = row["kind"].startswith("synthetic")
        counts["synthetic" if positive else "ordinary"] += 1
        best = row["scores"][arm]
        if best is None:
            counts["synthetic_unassessed" if positive else "ordinary_unassessed"] += 1
            continue
        hit = positive and useful(best[1], row["truth"])
        counts["useful_before_point"] += int(hit)
        if best[0] > point:
            counts["useful_flags"] += int(hit)
            counts["misplaced_flags"] += int(positive and not hit)
            counts["ordinary_flags"] += int(not positive)
            counts["background_flags"] += int(row["kind"] == "background")
        if positive:
            matched = genuine.get((row["group"], row["kind"].replace("synthetic_", "ordinary_")))
            if matched is not None:
                counts["available_matched_pairs"] += 1
                counts["synthetic_above_matched_ordinary"] += int(best[0] > matched[0])
    return counts


def verify(receipt):
    require(receipt["schema"] == SCHEMA, "Unexpected schema")
    rows = receipt["rows"]
    expected = {"development": 54, "assessment": 54, "transfer": 72}
    require(all(sum(row["role"] == role for row in rows) == size for role, size in expected.items()) and len(rows) == 180,
            "Panel sizes differ")
    development = [r["scores"]["candidate"][0] for r in rows if r["role"] == "development"
                   and not r["kind"].startswith("synthetic") and r["scores"]["candidate"] is not None]
    points = receipt["points"]
    require(max(development, default=1.) == points["candidate"], "Development operating point differs")
    output = {"eligible_development_controls": len(development)}
    for role in expected:
        scoped = [row for row in rows if row["role"] == role]
        levels = sorted({str(row["level"]) for row in scoped})
        output[role] = {}
        for level in levels:
            selected = [row for row in scoped if str(row["level"]) == level]
            output[role][level] = {arm: scan(selected, arm, points[arm]) for arm in selected[0]["scores"]}
    current = output["assessment"]["-5"]["short"]
    candidate = output["assessment"]["-5"]["candidate"]
    output["gate_pass"] = (
        candidate["useful_flags"] >= current["useful_flags"] + 2
        and candidate["ordinary_flags"] <= current["ordinary_flags"] + 1 and candidate["background_flags"] == 0
        and all(output["transfer"][level]["candidate"]["useful_flags"] >= 2
                and output["transfer"][level]["candidate"]["ordinary_flags"] <= 1 for level in ("-5", "-10"))
    )
    return output


def package(path):
    native = json.loads(path.read_bytes())
    rows = []
    groups = {}
    points = {"candidate": native["point"]}
    for row in native["development"] + native["assessment"]:
        role = row["role"]
        group = groups.setdefault((role, row["case"]), len(groups))
        entries = {"candidate": row["candidate"]["best"]}
        for arm, result in row.get("incumbent", {}).items():
            require(points.setdefault(arm, result["point"]) == result["point"], "Incumbent point differs across deliveries")
            entries[arm] = result["best"]
        rows.append({"role": role, "group": group, "level": row["level"], "kind": row["kind"], "truth": row["truth"],
                     "pcm_sha256": row["sha256"], "dual_pitch_frames": row["candidate"]["dual_pitch_frames"],
                     "eligible_ranges": row["candidate"]["eligible_ranges"],
                     "scores": {arm: None if value is None else [value["score"], value["range"]] for arm, value in entries.items()}})
    result = {"schema": SCHEMA, "freeze": native["freeze"], "points": points, "rows": rows,
              "native_receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    result["summary"] = verify(result)
    for role in ("development", "assessment", "transfer"):
        for level, arms in native["summary"][role].items():
            for arm, original in arms.items():
                reviewed = result["summary"][role][level][arm]
                fields = {"synthetic": "positives", "ordinary": "controls", "synthetic_unassessed": "positive_unassessed",
                          "ordinary_unassessed": "control_unassessed", "useful_before_point": "candidate_range_hits",
                          "useful_flags": "localized", "misplaced_flags": "misplaced",
                          "ordinary_flags": "false_flags", "background_flags": "false_background"}
                require(all(reviewed[k] == original.get(v, 0) for k, v in fields.items()), "Native and portable accounting differ")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--from-native", type=Path)
    args = parser.parse_args()
    try:
        if args.from_native:
            args.receipt.write_bytes(serial(package(args.from_native)))
        receipt = json.loads(args.receipt.read_bytes())
        require(receipt["summary"] == verify(receipt), "Portable score or localization accounting differs")
        print(f"Verified operating point, coverage, ranges and false flags across {len(receipt['rows'])} deliveries.")
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
