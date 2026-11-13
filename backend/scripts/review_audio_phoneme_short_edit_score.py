"""Replay the short-edit study's window admission, point and location arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

LEVELS = (None, 5, 0, -5)
FAMILIES = ("tts", "conversion")
GRIDS = {"full": ((0, 64000), (32000, 96000)), "short": tuple((a, a + 32000) for a in range(0, 64001, 16000))}
GRIDS["reduced_short"] = GRIDS["short"]
SCHEMA = "chipmunk.audio-phoneme-short-edit-score/1"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    body = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(body.encode()).hexdigest()


def pin(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def verify_reads(section, role):
    admitted, tested, scores = set(), set(), {}
    for row in section["rows"]:
        require(row["role"] == role and set(row["readings"]) == set(row["input_windows"]) == set(GRIDS),
                "Unexpected condition or measurement grid")
        original = row["pcm_sha256"]
        for arm, bundle in row["input_windows"].items():
            view = bundle["view_sha256"]
            require(pin(view) and (arm == "reduced_short" or view == original), "Original comparison input differs")
            tested.add(view)
            readings = row["readings"][arm]
            require(tuple(tuple(cell["range"]) for cell in bundle["grid"]) == GRIDS[arm]
                    and [{key: value for key, value in reading.items() if key != "score"}
                         for reading in readings] == bundle["grid"], "Clock-bound window schedule differs")
            for reading in readings:
                sha, value = reading["window_sha256"], reading["score"]
                require(pin(sha) and (value is None or type(value) in (int, float) and math.isfinite(value)),
                        "Unbound voice response")
                if value is not None:
                    admitted.add(sha)
                    require(scores.setdefault(sha, value) == value, "The same exact classifier input has different scores")
    require(len(section["rows"]) == 54 and section["classifier_forwards"] == len(admitted)
            and section["vad_passes"] == len(tested), "Model forward counts differ")


def strongest(row, arm):
    values = [item for item in row["readings"][arm] if item["score"] is not None]
    return max(values, key=lambda x: (x["score"], -x["range"][0]), default=None)


def hit(truth, proposed):
    a, b = truth
    c, d = proposed
    overlap = max(0, min(b, d) - max(a, c))
    broad = 2 * overlap >= b - a
    return broad and 10 * overlap >= 3 * (b - a + d - c - overlap), broad


def cells(rows, arm, point):
    out = {}
    for kind in ("untouched", "local_gain", "ordinary_tts", "ordinary_conversion", "background", "synthetic_tts", "synthetic_conversion"):
        entries = [row for row in rows if row["kind"] == kind]
        assessed = [(row, peak) for row in entries if (peak := strongest(row, arm)) is not None]
        flagged = [(row, peak) for row, peak in assessed if peak["score"] > point]
        cell = {"total": len(entries), "assessed": len(assessed), "flagged": len(flagged),
                "unassessed": len(entries) - len(assessed), "flagged_reader_groups": len({row["case"] for row, _ in flagged})}
        if kind.startswith("synthetic"):
            timed = [(row, peak, *hit(row["source_edit_range"], peak["range"])) for row, peak in flagged]
            cell.update({"strict_localized": sum(strict for _, _, strict, _ in timed),
                         "broad_coverage": sum(broad for _, _, _, broad in timed),
                         "misplaced_flags": sum(not broad for _, _, _, broad in timed),
                         "strongest_strict": sum(hit(row["source_edit_range"], peak["range"])[0] for row, peak in assessed),
                         "strongest_broad": sum(hit(row["source_edit_range"], peak["range"])[1] for row, peak in assessed),
                         "locations": [{"case": row["case"], "level_db": row["level"], "found_range": peak["range"],
                                        "true_edit": row["source_edit_range"], "strict": strict, "broad": broad}
                                       for row, peak, strict, broad in timed]})
        out[kind] = cell
    return out


def paired_rank(rows, arm):
    index = {(r["case"], r["level"], r["kind"]): r for r in rows}
    counts = {}
    for family in FAMILIES:
        pairs, aligned = [], []
        for (case, level, kind), fake in index.items():
            if kind != f"synthetic_{family}":
                continue
            ordinary = index[case, level, f"ordinary_{family}"]
            first, other = strongest(fake, arm), strongest(ordinary, arm)
            if first is not None and other is not None:
                pairs.append(first["score"] - other["score"])
            right = {tuple(item["range"]): item["score"] for item in ordinary["readings"][arm]}
            windows = [(item["score"], right[tuple(item["range"])]) for item in fake["readings"][arm]
                       if hit(fake["source_edit_range"], item["range"])[1] and item["score"] is not None
                       and right[tuple(item["range"])] is not None]
            if windows:
                aligned.append(max(a for a, _ in windows) - max(b for _, b in windows))
        counts[family] = {"paired_assessed": len(pairs), "synthetic_best_higher": sum(x > 0 for x in pairs),
                          "matched_edit_covering_windows_assessed": len(aligned),
                          "synthetic_edit_covering_best_higher": sum(x > 0 for x in aligned)}
    return counts


def review(receipt, source_bytes):
    lock, dev, held = (receipt[key] for key in ("freeze", "development", "assessment"))
    require(lock["schema"] == SCHEMA and lock["native_rate"] == 16000
            and hashlib.sha256(source_bytes).hexdigest() == lock["source_intake_sha256"]
            and lock["grid"] == {arm: [list(window) for window in grid] for arm, grid in GRIDS.items()},
            "Frozen input ancestry differs")
    verify_reads(dev, "development")
    verify_reads(held, "assessment")
    combined = dev["rows"] + held["rows"]
    bound = [{k: v for k, v in row.items() if k != "readings"} for row in combined]
    require(digest(bound) == lock["source_windows_sha256"], "Source window commitments differ")
    require([{k: v for k, v in row.items() if k not in ("id", "input_windows")} for row in bound]
            == json.loads(source_bytes)["deliveries"] and [row["id"] for row in bound] == list(range(108)),
            "Input order or exact delivery scope differs")
    operating = {arm: max(window["score"] for row in dev["rows"] if not row["kind"].startswith("synthetic")
                          for window in row["readings"][arm] if window["score"] is not None) for arm in GRIDS}
    require(receipt["operating_points"] == {"freeze_sha256": digest(lock), "development_sha256": digest(dev), "points": operating},
            "Development-only points differ")
    rows = held["rows"]
    return {"points": operating,
            "forwards": {name: {key: part[key] for key in ("classifier_forwards", "vad_passes")}
                         for name, part in (("development", dev), ("assessment", held))},
            "pooled": {arm: cells(rows, arm, operating[arm]) for arm in GRIDS},
            "by_level": {str(level): {arm: cells([row for row in rows if row["level"] == level], arm, operating[arm])
                                      for arm in GRIDS} for level in LEVELS},
            "matched_ranking": {str(level): {arm: paired_rank([row for row in rows if row["level"] == level], arm)
                                            for arm in GRIDS} for level in LEVELS},
            "production_short_diagnostic": {str(level): cells([row for row in rows if row["level"] == level], "short",
                                                              lock["production_short_point"]) for level in LEVELS}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("intake", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes()), args.intake.read_bytes()), sort_keys=True, indent=2))
