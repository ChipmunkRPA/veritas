"""Verify the clean-reference audio comparison without source recordings."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import statistics
from pathlib import Path

from review_audio_paired_speech_continuity_transfer import (
    CONTROLS,
    LEVELS,
    POSITIVES,
    UNSUPPORTED,
    digest,
    require,
    useful,
)

ARMS = ("prior_correlation", "coherent_projection")
FIELDS = ("case", "role", "kind", "level", "truth", "earlier_sha256", "submitted_sha256", "measured_snr_db")


def rows_ok(rows, role):
    labels = [f"dev-{i}" for i in range(2)] if role == "development" else [f"held-{i}" for i in range(3)]
    schedule = set(itertools.product(labels, LEVELS, (*CONTROLS, *POSITIVES, *UNSUPPORTED)))
    for row in rows:
        key = tuple(row[name] for name in ("case", "level", "kind"))
        require(key in schedule and row["role"] == role and abs(row["measured_snr_db"] - row["level"]) < .02,
                "Unexpected pair or condition")
        schedule.remove(key)
        require(all(re.fullmatch(r"[a-f0-9]{64}", row[name]) for name in ("earlier_sha256", "submitted_sha256")),
                "Unbound source waveforms")
        require(0 <= row["active_windows"] <= 73 and set(row["results"]) == set(ARMS), "Measurement scope differs")
        for arm, result in row["results"].items():
            if result is None:
                continue
            a, b = result["range"]
            require(row["supported_pair"] and 0 <= a < b <= 96000 and b-a == 11520 and a % 1280 == 0
                    and math.isfinite(result["score"]), "Invalid reported range")
            if arm == "coherent_projection":
                gains = result["band_gains"]
                require(2 <= len(gains) <= 3 and all(len(band) == 3 and all(math.isfinite(x) for x in band)
                         and min(band[0], band[2]) > .249999 and max(band[0], band[2]) / min(band[0], band[2]) <= 2.200001
                         for band in gains), "Insufficient flank gain")
                estimated = statistics.median(1 - center / min(left, right) for left, center, right in gains)
                require(abs(estimated - result["score"]) < 1e-6, "Candidate score differs from source-bound gains")
        truth = row["truth"]
        if row["kind"] in POSITIVES:
            a, b = truth
            require(20800 <= a < b <= 75200 and 10240 <= b-a <= 16320, "Invalid replacement extent")
        else:
            require(truth is None, "Control carries edit truth")
    require(not schedule, "Incomplete source pairing")


def summary(rows, points):
    totals = {}
    for level in LEVELS:
        cases = [row for row in rows if row["level"] == level]
        totals[str(level)] = {}
        for arm, cutoff in points.items():
            output = {}
            for kind in (*CONTROLS, *POSITIVES, *UNSUPPORTED):
                group = [row for row in cases if row["kind"] == kind]
                measured = [row for row in group if row["results"][arm] is not None]
                flags = [row for row in measured if row["results"][arm]["score"] > cutoff]
                localized = [row for row in flags if row["truth"] is not None
                             and useful(row["truth"], row["results"][arm]["range"])]
                output[kind] = {"planned": len(group), "paired": sum(row["supported_pair"] for row in group),
                                "measured": len(measured), "flags": len(flags), "localized": len(localized),
                                "locations": [{"case": row["case"], "truth": row["truth"],
                                               "range": row["results"][arm]["range"]} for row in localized]}
            totals[str(level)][arm] = output
    return totals


def review(receipt):
    freeze, development, held = (receipt[key] for key in ("freeze", "development", "assessment"))
    require(freeze["schema"] == "chipmunk.audio-clean-reference-transfer/1" and freeze["clock_hz"] == 16000
            and freeze["source_samples"] == 96000 and freeze["levels"] == list(LEVELS), "Frozen scope differs")
    rows_ok(development, "development")
    rows_ok(held, "assessment")
    require(digest([{field: row[field] for field in FIELDS} for row in development + held]) == freeze["roster_sha256"],
            "Source roster differs")
    points = {arm: max(r["results"][arm]["score"] for r in development
                       if r["kind"] in CONTROLS and r["results"][arm] is not None) for arm in ARMS}
    require(receipt["points"] == {"freeze_sha256": digest(freeze), "development_sha256": digest(development),
                                  "points": points}, "Development-only points differ")
    data = summary(held, points)
    active = data["-5"]["coherent_projection"]
    localized = {entry["case"] for kind in ("synthetic_tts", "synthetic_conversion") for entry in active[kind]["locations"]}
    gate = (sum(active[kind]["localized"] for kind in ("synthetic_tts", "synthetic_conversion")) >= 4
            and sum(active[kind]["flags"] for kind in CONTROLS) <= 1 and len(localized) >= 2
            and sum(data[level]["coherent_projection"][kind]["flags"] for level in data for kind in UNSUPPORTED) == 0)
    return {"points": points, "held": data, "candidate_pass": gate, "full_span_exact_comparisons":
            sum(row["exact_range"] == [0, 96000] for row in held)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True, indent=2))
