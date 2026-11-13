"""Recompute matched speech continuity outcomes from the public scalar receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

ARMS = ("correlation", "projection")
CONTROLS = ("untouched", "local_gain", "ordinary_tts", "ordinary_conversion", "global_eq")
POSITIVES = ("ordinary_replacement", "synthetic_tts", "synthetic_conversion")
UNSUPPORTED = ("different_reader_reference", "same_words_rerecorded_reference")
LEVELS = (0, -5, -10)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(value):
    body = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    return hashlib.sha256(body.encode()).hexdigest()


def useful(truth, proposed):
    a, b = truth
    c, d = proposed
    overlap = max(0, min(b, d) - max(a, c))
    return 2 * overlap >= b - a and 10 * overlap >= 3 * (b - a + d - c - overlap)


def check_rows(rows, role):
    cases = [f"dev-{i}" for i in range(2)] if role == "development" else [f"held-{i}" for i in range(3)]
    wanted = {(case, level, kind) for case in cases for level in LEVELS
              for kind in (*CONTROLS, *POSITIVES, *UNSUPPORTED)}
    range_by_case = {}
    for row in rows:
        case, level, kind = (row[k] for k in ("case", "level", "kind"))
        key = (case, level, kind)
        require(key in wanted and row["role"] == role and abs(row["measured_snr_db"] - level) < .02,
                "Missing or invalid paired delivery")
        wanted.remove(key)
        require(set(row["results"]) == set(ARMS) and 0 <= row["active_windows"] <= 73, "Measurement scope differs")
        if kind in POSITIVES:
            a, b = row["truth"]
            require(20800 <= a < b <= 96000 - 20800 and 10240 <= b - a <= 16320
                    and range_by_case.setdefault(case, row["truth"]) == row["truth"], "Source edit geometry differs")
        else:
            require(row["truth"] is None, "Control carries an edit label")
        for reading in row["results"].values():
            if reading is None:
                continue
            a, b = reading["range"]
            before, after = reading["flanks"]
            middle = reading["middle"]
            require(row["supported_pair"] and b - a == 11520 and a % 1280 == 0 and 0 <= a < b <= 96000
                    and all(math.isfinite(x) for x in (before, after, middle, reading["score"]))
                    and abs(reading["score"] - (min(before, after) - middle - .25 * abs(before - after))) < 1e-7,
                    "Paired range or independent score arithmetic differs")
    require(not wanted, "Incomplete paired roster")


def totals(rows, points):
    out = {}
    for level in LEVELS:
        local = [row for row in rows if row["level"] == level]
        out[str(level)] = {}
        for arm, point in points.items():
            cells = {}
            for kind in (*CONTROLS, *POSITIVES, *UNSUPPORTED):
                entries = [row for row in local if row["kind"] == kind]
                seen = [(row, candidate) for row in entries if (candidate := row["results"][arm]) is not None]
                flags = [(row, candidate) for row, candidate in seen if candidate["score"] > point]
                true = [(row, candidate) for row, candidate in flags
                        if row["truth"] is not None and useful(row["truth"], candidate["range"])]
                cells[kind] = {"eligible": len(entries), "pair_supported": sum(row["supported_pair"] for row in entries),
                               "range_assessed": len(seen), "flags": len(flags), "localized": len(true),
                               "best_range_overlaps": sum(row["truth"] is not None and useful(row["truth"], candidate["range"])
                                                          for row, candidate in seen),
                               "locations": [{"case": row["case"], "truth": row["truth"], "range": candidate["range"]}
                                             for row, candidate in true]}
            out[str(level)][arm] = cells
    return out


def review(result):
    lock, dev, held, operating = (result[k] for k in ("freeze", "development", "assessment", "operating_points"))
    require(lock["schema"] == "chipmunk.audio-paired-speech-continuity/1" and lock["clock_hz"] == 16000
            and lock["reference_level_db"] == 5 and lock["submission_levels_db"] == list(LEVELS), "Unexpected source scope")
    check_rows(dev, "development")
    check_rows(held, "assessment")
    fields = ("case", "role", "level", "kind", "truth", "earlier_sha256", "submitted_sha256", "measured_snr_db")
    require(digest([{field: row[field] for field in fields} for row in dev + held]) == lock["delivery_roster_sha256"],
            "Pair waveform commitments differ")
    points = {arm: max(row["results"][arm]["score"] for row in dev
                       if row["kind"] in CONTROLS and row["results"][arm] is not None) for arm in ARMS}
    require(operating == {"freeze_sha256": digest(lock), "development_sha256": digest(dev), "points": points},
            "Development-only point differs")
    baseline = Counter(row["exact_comparator_changed_bins"] for row in held)
    durations = [row["exact_comparator_extent"][1] - row["exact_comparator_extent"][0]
                 for row in held if row["exact_comparator_extent"]]
    return {"points": points, "development_pairs": len(dev), "held_pairs": len(held), "held": totals(held, points),
            "exact_baseline_changed_bins": dict(baseline),
            "exact_baseline_localized": sum(row["truth"] is not None and row["exact_comparator_extent"] is not None
                                            and useful(row["truth"], row["exact_comparator_extent"]) for row in held),
            "exact_baseline_median_changed_extent_ms": statistics.median(durations) / 16}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True, indent=2))
