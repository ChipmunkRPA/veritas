"""Replay the local timing check on pinned WAV-to-AAC versions of held speech."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audio_source_isolation_diagnostic import PANELS, check, path_sha, read, write

SCHEMA = "chipmunk.audio-aac-timing-transfer/1"
PARENT = PANELS / "audio-reference-local-shift-2026-11-12.json"
PIN = "2d3826174fb5b6d582bb33823f23c633b39eb6517bde26bd8a2039c77de2db71"


def cases():
    check(path_sha(PARENT) == PIN, "Earlier timing evaluation differs")
    for row in read(PARENT)["rows"]:
        pcm = {case["role"]: case for case in row["cases"] if case["carrier"] == "pcm"}
        for case in row["cases"]:
            if case["carrier"] == "aac":
                yield row["group"], row["ordinal"], pcm[case["role"]], case


def paths(folder, ordinal, role):
    directory = folder / f"case-{ordinal:02}"
    name = role if role in {"unchanged", "silence_all", "preexisting_repeat"} else "unchanged"
    return directory / f"{name}-pcm-earlier.wav", directory / f"{role}-aac-submitted.m4a"


def summarize(rows):
    result = {}
    for row in rows:
        cell = result.setdefault(row["role"], Counter())
        candidate = row["range"]
        exact = False
        if candidate and row["truth"]:
            start, stop = candidate["submitted"]
            left, right = row["truth"]
            overlap = max(0, min(stop, right) - max(start, left))
            exact = 5 * overlap >= 3 * (max(stop, right) - min(start, left)) and abs(candidate["offset"] - row["offset"]) <= 16
        cell.update({"pairs": 1, "assessed": row["status"] != "unassessed",
                     "flags": candidate is not None, "exact": exact})
    return result


def score(folder):
    import base64
    from tempfile import TemporaryDirectory

    from app.evidence import audio_reference
    from scripts.verify_audio_local_timing_export import verify_export

    rows = []
    with TemporaryDirectory(prefix="review-aac-timing-") as temp:
        output = Path(temp) / "comparison.json"
        for group, ordinal, pcm, aac in cases():
            role = aac["role"]
            earlier, submitted = paths(folder, ordinal, role)
            pins = [pcm["earlier"]["file_sha256"], aac["submitted"]["file_sha256"]]
            check([path_sha(path) for path in (earlier, submitted)] == pins, "Pinned encoded input changed")
            record, assets = audio_reference.compare(earlier, submitted, include_local_timing=True, include_aac_timing=True)
            timing = record["local_timing"]
            span = timing.get("range")
            if span:
                output.write_text(json.dumps({"comparison": record, "assets": {
                    name: base64.b64encode(raw).decode() for name, raw in assets.items()}}))
                check(verify_export(earlier, submitted, output)["excerpts_verified"] == 2,
                      "Independent waveform and evidence check failed")
            rows.append({
                "group": group, "ordinal": ordinal, "role": role, "truth": aac["truth"], "offset": aac["offset"],
                "input_sha256": pins, "status": timing["status"], "presented": timing["aac_presentation"],
                "matched_windows": timing.get("matched_windows"),
                "range": {"earlier": span["earlier_samples"], "submitted": span["submitted_samples"],
                          "offset": span["shift_samples"], "pcm_sha256": [span[side]["pcm_sha256"]
                                                                        for side in ("earlier", "submitted")]} if span else None,
            })
    return {"schema": SCHEMA, "parent_sha256": PIN, "rows": rows, "summary": summarize(rows)}


def review(receipt):
    check(receipt["schema"] == SCHEMA and receipt["parent_sha256"] == PIN, "Different evaluation")
    rows = receipt["rows"]
    for row, (group, ordinal, pcm, aac) in zip(rows, cases(), strict=True):
        check((row["group"], row["ordinal"], row["role"], row["truth"], row["offset"])
              == (group, ordinal, aac["role"], aac["truth"], aac["offset"]), "Different source role or labels")
        clock = row["presented"]
        duration = pcm["earlier"]["native_readback_samples"]
        decoded = aac["submitted"]["native_readback_samples"]
        check(row["input_sha256"] == [pcm["earlier"]["file_sha256"], aac["submitted"]["file_sha256"]]
              and clock == {"side": "submitted", "sample_rate": 16000, "time_base": "1/16000",
                            "presentation_start_pts": 0, "presentation_end_exclusive_pts": duration,
                            "decoded_samples": decoded, "excluded_trailing_samples": decoded-duration}
              and 0 <= decoded-duration < 1024, "Different AAC presentation clock")
        span = row["range"]
        support = row["matched_windows"]
        check(row["status"] == ("unassessed" if support < 8 else "local_shift" if span else "no_local_shift")
              and (span is None or support >= 8), "Inconsistent assessed outcome")
        if span:
            check(all(0 <= a < b <= duration for a, b in (span["earlier"], span["submitted"]))
                  and all(b-a == span["offset"] for a, b in zip(span["earlier"], span["submitted"], strict=True))
                  and len(span["pcm_sha256"]) == 2 and all(len(pin) == 64 for pin in span["pcm_sha256"]),
                  "Invalid source timing or excerpt commitment")
    summary = summarize(rows)
    check(summary == receipt["summary"], "Published counts differ")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--prepared", type=Path)
    args = parser.parse_args()
    if args.prepared:
        value = score(args.prepared)
        review(value)
        write(args.receipt, value)
    else:
        value = read(args.receipt)
    print(json.dumps(review(value), sort_keys=True))


if __name__ == "__main__":
    main()
