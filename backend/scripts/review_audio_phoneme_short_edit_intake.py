"""Check scope and source-coordinate completeness of the public short-edit intake."""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import Counter
from pathlib import Path

KINDS = ("untouched", "local_gain", "ordinary_tts", "ordinary_conversion", "synthetic_tts", "synthetic_conversion")
LEVELS = (None, 5, 0, -5)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def review(receipt):
    rows = receipt["deliveries"]
    require(receipt["schema"] == "chipmunk.audio-phoneme-short-edit/1" and receipt["voice_classifier_forwards"] == 0
            and receipt["common_utterance_ids"] == 1054 and receipt["eligible_distinct_speakers"] == 7
            and receipt["eligible_speakers_outside_all_saved_v9_roles"] == 1 and len(rows) == 108,
            "Unexpected source scope")
    cases = {f"phoneme-reader-{index:02d}-prompt-0": "development" if index < 2 else "assessment" for index in range(4)}
    planned = {(case, level, kind) for case, level, kind in itertools.product(cases, LEVELS, KINDS)}
    planned |= {(case, level, "background") for case, level in itertools.product(cases, LEVELS[1:])}
    ranges, hashes, counts = {}, set(), Counter()
    for row in rows:
        case, level, kind = (row[key] for key in ("case", "level", "kind"))
        require((case, level, kind) in planned and row["role"] == cases[case] and row["clock_hz"] == 16000
                and re.fullmatch(r"[a-f0-9]{64}", row["pcm_sha256"]), "Unexpected role or delivery coordinate")
        planned.remove((case, level, kind))
        hashes.add(row["pcm_sha256"])
        counts[row["role"]] += 1
        span, measured = row["source_edit_range"], row["measured_snr_db"]
        require(measured is None if level is None or kind == "background" else abs(measured - level) < .02,
                "Reported measured signal ratio differs")
        if kind in ("untouched", "background"):
            require(span is None, "No-edit control claims an edit range")
        else:
            start, stop = span
            require(20800 <= start < stop <= 96000 - 20800 and 10240 <= stop - start <= 16320,
                    "Unsupported short edit geometry")
            require(ranges.setdefault(case, span) == span, "Paired origins use different destinations")
    require(not planned and len(hashes) == 96 and len(ranges) == 4 and set(counts.values()) == {54},
            "Missing data or unexpected control waveform aliases")
    overlap = receipt["background_edit_overlap_centiseconds"]
    require(len(overlap) == 4 and all(type(item) is int and item >= 20 for item in overlap), "Overlap coverage missing")
    return {"deliveries": len(rows), "unique_pcm_deliveries": len(hashes), "roles": dict(counts),
            "short_edit_ms": sorted(round((b - a) / 16, 2) for a, b in ranges.values()),
            "competing_voice_overlap_ms": [value * 10 for value in overlap], "voice_classifier_forwards": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    print(json.dumps(review(json.loads(args.receipt.read_bytes())), sort_keys=True, indent=2))
