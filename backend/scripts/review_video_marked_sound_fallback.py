# SPDX-License-Identifier: Apache-2.0
"""Replay the marked soundtrack fallback on the short native Lombard transfer."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from app.evidence import video_reference_soundtrack as automatic
from app.evidence import video_reference_timing as marked
from scripts import evaluate_short_lip_release_timing as study
from scripts.video_common_sources_v89 import read, require, sha, write


def replay(sources, observations):
    rows = {row["ordinal"]: row for row in study.sources(sources)}
    ledger = read(observations)
    require(ledger["source_receipt_sha256"] == sha(sources / "source-receipt.json"), "Source roster differs")
    results = []
    for item in ledger["rows"]:
        truth = item["truth"]
        if truth is None:
            continue
        row = rows[item["ordinal"]]
        raw, audio, _ = study.source_pcm(study.prior.source(sources, row, "audio"))
        require(study.prior.digest(raw) == item["audio_pcm_sha256"], "Source PCM differs")
        times = [tick * Fraction(item["time_base"]) for tick in item["pts"]]
        a = truth["visual_frames"][0]
        step = times[1] - times[0]
        motion = [(v, v, v) if v is not None else None for v in item["apertures"]]
        supported = (
            48 <= len(times) <= 75
            and Fraction(1, 25) <= step <= Fraction(2, 47)
            and times == [i * step for i in range(len(times))]
            and 24000 <= len(audio) <= 54400
            and 2 <= a < len(times) - 4
        )
        visual = marked.visual_shift(motion, motion, a) if supported else None
        eligible = visual is not None and visual["submitted_frame"] == a
        first = np.clip(np.rint(audio * 32768), -32768, 32767).astype("<i2")
        for role, variant, _, edit in study.variants(audio, item["apertures"], times, item):
            if role == "paired_edit":
                continue
            current, _ = automatic.measure(audio, variant)
            if current is not None and Fraction(max(current["earlier_samples"][1], current["submitted_samples"][1]), 16000) > times[-1]:
                current = None
            moved = None
            if eligible and current is None:
                last = np.clip(np.rint(variant * 32768), -32768, 32767).astype("<i2")
                match = marked.local_sound_match(first, last, round(times[a] * 16000))
                if match is not None:
                    candidate, _, changed = match
                    last_sample = max(candidate["source_samples"][1], candidate["submitted_samples"][1])
                    if Fraction(last_sample, 16000) <= times[-1]:
                        moved = {
                            "offset": candidate["offset_samples"], "earlier": candidate["source_samples"],
                            "analyzed": candidate["submitted_samples"], "changed": changed,
                        }
            results.append({"case": item["ordinal"], "group": item["group"], "role": role,
                            "expected": None if edit is None else round(edit["shift_s"] * 16000),
                            "automatic": current and current["shift_samples"], "fallback": moved})
    counts = Counter()
    for item in results:
        expected = item["expected"]
        found = item["automatic"]
        fallback = item["fallback"]
        if expected is None:
            counts["controls"] += 1
            counts["automatic_controls_flagged"] += found is not None
            counts["fallback_controls_flagged"] += fallback is not None
        else:
            counts["known_moves"] += 1
            counts["automatic_useful"] += found is not None and abs(found - expected) <= 16
            counts["fallback_useful"] += fallback is not None and fallback["offset"] == expected
    return {"schema": "chipmunk.marked-sound-transfer/1", "observations_sha256": sha(observations),
            "counts": dict(counts), "results": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", type=Path)
    p.add_argument("observations", type=Path)
    p.add_argument("receipt", type=Path)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    result = replay(args.sources, args.observations)
    if args.verify:
        require(result == read(args.receipt), "Timing replay differs")
    else:
        write(args.receipt, result)
    print(result["counts"])


if __name__ == "__main__":
    main()
