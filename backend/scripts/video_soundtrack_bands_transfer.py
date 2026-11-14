# SPDX-License-Identifier: Apache-2.0
"""Measure agreement across frequency bands on a source-separated two-track panel."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scripts import video_soundtrack_pair_transfer as parent

SCHEMA = "chipmunk.video-soundtrack-bands-transfer/1"
PINNED = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/video-mouth-hd-hold-2026-11-12.json"
NAMES = ("diego", "jan", "mckensie", "paison", "priscilla", "sara", "simon")
LAGS = parent.LAGS | {"earlier_train_loud": -3200, "later_rain_loud": 3200}
EXTRA = ("earlier_train_loud", "later_rain_loud", "unchanged_train_loud", "unchanged_rain_loud",
         "same_masker_both", "different_masker_tracks", "constant_train_loud", "repeat_music_like")


def variants(original, advanced, delayed, beds, choose, primary):
    import numpy as np

    from scripts.video_lipsync_local_delay_transfer import shift

    scale = 3.95  # 0.45 times this value makes the masker 5 dB louder than the original.
    rain, train = (beds[key] * scale for key in ("rain", "train"))
    choose.update(zip(EXTRA, (advanced+train, delayed+rain, original+train, original+rain, delayed+train,
                               original+rain, shift(original, 3200)+train, original + np.resize(train[:8000], len(original))),
                      strict=True))
    # Both tracks carry identical timing in this control, including the same localized delay.
    primary.update(same_masker_both=delayed+train, different_masker_tracks=original+train)
    return choose, primary


def score(row, primary, secondary, path, output, examples):
    from app.evidence import video_soundtrack_bands as bands
    from app.evidence import video_soundtrack_pair as pair

    alternate = {"status": "unsupported_clock"}
    if secondary is not None and pair.scope(primary, secondary) and row["guarded"]["status"] != "local_shift":
        alternate = pair.select(list(bands.matches(primary.pcm, secondary.pcm)), len(primary.pcm))
    row["band_match"] = alternate
    if examples and alternate["status"] == "local_shift" and row["role"] in LAGS:
        target = output / f"{row['carrier']}-{row['group']}-{row['offset']}-{row['role']}{path.suffix}"
        target.write_bytes(path.read_bytes())


def useful(span, truth):
    if span is None or truth is None:
        return False
    a, b = span["submitted_samples"]
    c, d = truth["samples"]
    return abs(span["shift_samples"] - truth["shift"]) <= 16 and 2 * max(0, min(b, d) - max(a, c)) >= max(b, d) - min(a, c)


def review(receipt):
    freeze = receipt["freeze"]
    if freeze != plan():
        raise ValueError("Sources, roles or settings differ from the frozen plan")
    expected = [(group, offset, carrier, role) for group, row in enumerate(freeze["sources"], 1)
                for offset in row["offsets"] for carrier in parent.CARRIERS for role in freeze["roles"]]
    if expected != [(row["group"], row["offset"], row["carrier"], row["role"]) for row in receipt["rows"]]:
        raise ValueError("Incomplete or reordered held panel")
    out = {name: Counter() for name in parent.CARRIERS}
    gains = {name: set() for name in parent.CARRIERS}
    for row in receipt["rows"]:
        counter = out[row["carrier"]]
        role, truth = row["role"], row["truth"]
        if (truth is None) != (role not in LAGS) or truth and truth["shift"] != LAGS[role]:
            raise ValueError("Known source shift differs")
        offsets = freeze["sources"][row["group"] - 1]["offsets"]
        ordinal = offsets.index(row["offset"])
        if truth and truth["samples"] != parent.boundary(row["group"], parent.EDIT_CLOCK_OFFSETS[ordinal]):
            raise ValueError("Known edit boundaries differ from the frozen picture clock")
        base, candidate = (row[key].get("range") for key in ("guarded", "band_match"))
        if base and candidate:
            raise ValueError("Band check replaced a result from the first pass")
        counter[f"planned_{'edits' if truth else 'controls'}"] += 1
        if truth:
            counter[f"{role}_planned"] += 1
            if useful(base, truth):
                counter["base_useful"] += 1
                counter[f"{role}_base"] += 1
            if useful(candidate, truth):
                counter["added_useful"] += 1
                counter[f"{role}_added"] += 1
                gains[row["carrier"]].add(row["group"])
            elif candidate:
                counter["added_wrong"] += 1
        elif base or candidate:
            counter["added_control_flags" if candidate else "base_control_flags"] += 1
            if candidate:
                counter[f"flag_{role}"] += 1
    return {name: dict(counts) | {"sources_with_added_locations": len(gains[name]),
                                "passes": counts["added_useful"] >= 4 and len(gains[name]) >= 3
                                and counts["added_control_flags"] <= 2 and counts["added_wrong"] <= 1
                                and not any(counts[f"flag_{role}"] for role in ("constant", "constant_train_loud",
                                            "same_masker_both", "same_local_shift", "different_speech"))}
            for name, counts in out.items()}


def plan(bank=None):
    sources = {item["name"]: item for item in json.loads(PINNED.read_bytes())["source_plan"]["rows"]}
    roster = []
    for name in NAMES:
        item = sources[name]
        file = name + ".webm"
        if bank is not None and parent.sha(bank / file) != item["source_sha256"]:
            raise ValueError("A source copy differs from its published commitment")
        roster.append({"file": file, "sha256": item["source_sha256"], "source": item["commons"], "license": item["license"],
                       "credit": item["credit"], "offsets": [item["offset_s"] + 4, item["offset_s"] + 21]})
    return {"schema": SCHEMA, "sources": roster, "roles": list(parent.VARIANTS + EXTRA),
            "method": {"bands_hz": [[250, 1350], [1350, 3500], [3500, 7000]], "correlation": .82, "margin": .09,
                       "min_rms": .0005, "two_bands_within_samples": 16},
            "acceptance": "In each format add four useful locations on three different source recordings, with no more than "
                          "two added control flags and one wrong extra range, and no alerts on matched delays or different speech."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--sounds", type=Path)
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()
    if args.mode == "run":
        if not (args.bank and args.sounds):
            parser.error("Native comparison requires source recordings and background files")
        parent.run(args.bank, args.sounds, args.path, args.examples, plan=plan(args.bank), callback=sys.modules[__name__])
    else:
        receipt = json.loads(args.path.read_bytes())
        if review(receipt) != receipt["results"]:
            raise ValueError("Saved results do not recount")
        print(json.dumps(receipt["results"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
