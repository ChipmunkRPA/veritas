# SPDX-License-Identifier: Apache-2.0
"""Compare the fixed twelve-second clean-reference check on PCM and AAC."""

import argparse
import gzip
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from scripts import audio_clean_speech_aac_transfer as previous

SCHEMA = "chipmunk.audio-clean-speech-aac-long-transfer/1"
CARRIERS = ("pcm", "aac48_mono", "aac48_stereo")
SECOND_HALF = previous.SAMPLES
SAMPLES = 2 * SECOND_HALF
WRONG = {"unrelated", "half_matched"}


def pairs(bank, folder):
    source = previous.icsi_cases if bank == "icsi" else previous.ami_cases
    cases = defaultdict(dict)
    for group, kind, truth, earlier, submitted in source(folder):
        cases[group][kind] = truth, earlier, submitted
    groups = sorted(cases)
    for index, group in enumerate(groups):
        companion = groups[(index + 1) % len(groups)]
        _, context, noisy_context = cases[companion]["untouched"]
        _, wrong_context, _ = cases[companion]["unrelated"]
        late = index % 2 == 1
        for kind, (edit, earlier, submitted) in cases[group].items():
            truth = [tick + (SECOND_HALF if late else 0) for tick in edit] if edit else None
            # Unrelated controls use the matching source only in the companion half.
            yield {"group": group, "companion": companion, "half": "late" if late else "early",
                   "kind": "half_matched" if kind == "unrelated" else kind, "truth": truth}, (
                context + earlier if late else earlier + context,
                noisy_context + submitted if late else submitted + noisy_context,
            )
        _, earlier, submitted = cases[group]["unrelated"]
        yield {"group": group, "companion": companion, "half": "late" if late else "early",
               "kind": "unrelated", "truth": None}, (
            wrong_context + earlier if late else earlier + wrong_context,
            noisy_context + submitted if late else submitted + noisy_context,
        )


def assess(first, second):
    import numpy as np

    from app.evidence import audio_clean_speech_long as check

    a, b = (np.frombuffer(raw, dtype="<i2") for raw in (first, second))
    reason, range_, _profile = check.locate(a, b)
    return {"reason": reason, "range": range_["samples"] if range_ else None}


def delivered(path, submitted, carrier):
    from app.detectors.audio.decode import aac48_presentation, decode_audio

    previous.encode48(submitted, path, carrier)
    pin = previous.sha(path.read_bytes())
    clock = aac48_presentation(path)
    channels = previous.CARRIERS48[carrier][1]
    previous.require(clock == {"sample_rate": 48_000, "channels": channels, "time_base": "1/48000",
                               "presentation_end_exclusive_pts": 3 * SAMPLES}, "AAC presentation changed")
    decode = decode_audio(path, source_sha256=pin, max_seconds=13)
    extent = len(decode.pcm)
    previous.require(0 <= extent - SAMPLES <= 342 and decode.mapping["timeline"] == [
        {"start_sample": 0, "end_sample": extent, "source_pts": 0}], "AAC decoded timeline changed")
    return (decode.pcm[:SAMPLES] * 32768).astype("<i2").tobytes(), {
        "encoded_sha256": pin, "decoded_samples": extent, "native_clock": clock,
    }


def category(kind):
    if kind in WRONG:
        return "unrelated_reference"
    if kind.startswith("edited/") or kind in {"piper", "kitten"}:
        return "synthetic_edit"
    if "splice" in kind:
        return "genuine_edit"
    if kind == "background":
        return "background_only_half"
    return "ordinary"


def outcomes(rows):
    reports = defaultdict(Counter)
    for row in rows:
        kind = category(row["kind"])
        baseline = row["readings"]["pcm"]
        old_useful = previous.location(baseline["range"], row["truth"])
        for carrier, result in row["readings"].items():
            span = result["range"]
            useful = previous.location(span, row["truth"])
            counter = reports[f"{row['bank']}/{carrier}/{kind}"]
            counter.update(files=1, flagged=bool(span), useful=useful, changed_range=span != baseline["range"],
                           unresolved=result["reason"] in {"reference_match_not_supported", "local_support_not_available"},
                           gained=useful and not old_useful, lost=old_useful and not useful,
                           late_useful=useful and row["half"] == "late")
    return dict(reports)


def evaluate(icsi, ami):
    rows = []
    with tempfile.TemporaryDirectory(prefix="chipmunk-twelve-aac-") as directory:
        path = Path(directory) / "temporary.m4a"
        for bank, folder in (("icsi", icsi), ("ami", ami)):
            for row, (first, second) in pairs(bank, folder):
                readings = {"pcm": assess(first, second)}
                for carrier in CARRIERS[1:]:
                    delivered_pcm, presentation = delivered(path, second, carrier)
                    readings[carrier] = assess(first, delivered_pcm) | presentation
                rows.append(row | {"bank": bank, "earlier_sha256": previous.sha(first),
                                   "submitted_sha256": previous.sha(second), "readings": readings})
                if len(rows) % 20 == 0:
                    print(f"Checked {len(rows)} joined pairs at all three carriers.", flush=True)
    return {"schema": SCHEMA, "inputs": {bank: previous.source_hash(bank) for bank in ("icsi", "ami")},
            "rows": rows, "outcomes": outcomes(rows)}


def review(receipt):
    rows = receipt["rows"]
    previous.require(receipt["schema"] == SCHEMA and receipt["inputs"] == {
        bank: previous.source_hash(bank) for bank in ("icsi", "ami")}, "Benchmark sources differ")
    previous.require(Counter(row["bank"] for row in rows) == {"icsi": 216, "ami": 56}
                     and len({(row["bank"], row["group"], row["kind"]) for row in rows}) == len(rows),
                     "Joined pair roster differs")
    for row in rows:
        previous.require(set(row["readings"]) == set(CARRIERS), "Missing carrier")
        for carrier, reading in row["readings"].items():
            if carrier != "pcm":
                clock = reading["native_clock"]
                previous.require(clock == {"sample_rate": 48000, "time_base": "1/48000",
                                           "channels": previous.CARRIERS48[carrier][1],
                                           "presentation_end_exclusive_pts": 3 * SAMPLES}
                                 and 0 <= reading["decoded_samples"] - SAMPLES <= 342, "Invalid source playback clock")
    actual = outcomes(rows)
    previous.require(actual == receipt["outcomes"], "Held comparisons do not reproduce")
    return actual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--icsi", type=Path)
    parser.add_argument("--ami", type=Path)
    args = parser.parse_args()
    if not args.icsi and not args.ami:
        print(json.dumps(review(json.loads(gzip.decompress(args.receipt.read_bytes()))), sort_keys=True, indent=2))
    elif args.icsi and args.ami:
        document = evaluate(args.icsi, args.ami)
        review(document)
        with args.receipt.open("xb") as output:
            output.write(gzip.compress(json.dumps(document, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
        print(json.dumps(document["outcomes"], sort_keys=True), flush=True)
    else:
        parser.error("Provide both audio source banks for native replay.")
