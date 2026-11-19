# SPDX-License-Identifier: Apache-2.0
"""Recount the paired 44.1 kHz AAC speech-to-mouth comparison without detector dependencies."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.video-lip-motion-aac441-transfer/1"
POINT = 5_318_680
CARRIERS = ("aac48", "aac441")
ROLES = {"original", "gain", "same_words", "global_early", "global_late", "different_words"}
GROUP_FIELDS = {"id", "speaker_group", "time_base", "picture_pts", "truth", "cases"}
CASE_FIELDS = {"role", "presented", "native_44100_ticks", "sample_alignment", "observations", "prior_speech_event_range"}
OBSERVATION_FIELDS = {"status", "proposal", "selected"}
PROPOSAL_FIELDS = {"score_units", "audio_span", "picture_span", "bracket_frames", "lag_samples"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integers(values, length):
    return isinstance(values, list) and len(values) == length and all(type(x) is int for x in values)


def in_audio(span, presented):
    return integers(span, 2) and 0 <= span[0] < span[1] <= presented


def hit(truth, span):
    intersection = max(0, min(truth[1], span[1]) - max(truth[0], span[0]))
    union = truth[1] - truth[0] + span[1] - span[0] - intersection
    return intersection * 5 >= union * 2


def validate_observation(record, picture_ticks, presented):
    require(set(record) == OBSERVATION_FIELDS and record["status"] in
            ("coverage", "whole_track_delay", "no_local_dip", "measured"), "Invalid local comparison disposition")
    proposal = record["proposal"]
    require((record["status"] == "measured") is (proposal is not None), "A proposal must have a measured disposition")
    if proposal is None:
        require(record["selected"] is False, "Unmeasured input cannot have selected evidence")
        return None
    require(set(proposal) == PROPOSAL_FIELDS and in_audio(proposal["audio_span"], presented)
            and integers(proposal["picture_span"], 2) and integers(proposal["bracket_frames"], 3)
            and type(proposal["score_units"]) is int and 0 < proposal["score_units"] <= 20_000_000
            and type(proposal["lag_samples"]) is int and abs(proposal["lag_samples"]) <= 1920
            and proposal["lag_samples"] % 320 == 0, "Invalid local comparison measurement")
    start, stop = proposal["picture_span"]
    before, during, after = proposal["bracket_frames"]
    require(0 <= before < start <= during < stop <= after < len(picture_ticks)
            and proposal["audio_span"] == [picture_ticks[i] + proposal["lag_samples"] for i in (start, stop)],
            "Local interval or witness does not match the source clock")
    require(record["selected"] is (proposal["score_units"] > POINT), "Selection differs from the fixed comparison point")
    return proposal["audio_span"] if record["selected"] else None


def review(receipt):
    require(set(receipt) == {"schema", "point_score_units", "groups", "commitments", "verification", "frozen_gate"}
            and receipt["schema"] == SCHEMA and receipt["point_score_units"] == POINT,
            "Unexpected paired carrier receipt or comparison point")
    seals = receipt["commitments"]
    require(set(seals) == {"method_sha256", "native_intake_sha256", "measured_sha256", "independent_review_sha256"}
            and all(isinstance(pin, str) and re.fullmatch("[a-f0-9]{64}", pin) for pin in seals.values()),
            "Invalid private replay commitments")
    groups = receipt["groups"]
    require(isinstance(groups, list) and len(groups) == 18, "Expected all 18 matched recordings")
    counts = {carrier: Counter() for carrier in CARRIERS}
    successful_speakers = {carrier: set() for carrier in CARRIERS}
    pairs = Counter()
    speakers = Counter()
    source_anchors = Counter()
    timestamp_checks = 0
    for number, group in enumerate(groups, 1):
        require(set(group) == GROUP_FIELDS and type(group["id"]) is int and group["id"] == number
                and type(group["speaker_group"]) is int and group["speaker_group"] in range(1, 10),
                "Invalid anonymous group")
        speakers[group["speaker_group"]] += 1
        pts = group["picture_pts"]
        base = Fraction(group["time_base"])
        require(isinstance(pts, list) and 24 <= len(pts) <= 120 and integers(pts, len(pts))
                and pts[0] == 0 and all(Fraction(1, 28) <= (b - a) * base <= Fraction(1, 22)
                                        for a, b in zip(pts, pts[1:])), "Invalid source picture clock")
        cases = group["cases"]
        require(isinstance(cases, list) and len(cases) == 6 and {case["role"] for case in cases} == ROLES,
                "A matched treatment is missing or repeated")
        picture_ticks = [round(value * base * 16_000) for value in pts]
        for case in cases:
            require(set(case) == CASE_FIELDS, "Unexpected treatment fields")
            extent, native = case["presented"], case["native_44100_ticks"]
            alignment = case["sample_alignment"]
            require(set(extent) == set(CARRIERS) and all(type(x) is int and 20_000 <= x <= 64_000 for x in extent.values())
                    and type(native) is int and native * 16_000 // 44_100 == extent["aac441"]
                    and abs(extent["aac441"] - extent["aac48"]) <= 1
                    and set(alignment) == set(CARRIERS) and all(type(v) is int and abs(v) <= 1 for v in alignment.values()),
                    "Decoded audio disagrees with the native presentation")
            require(set(case["observations"]) == set(CARRIERS), "A carrier comparison is missing")
            require(in_audio(group["truth"], min(extent.values())), "Edited interval disagrees with the paired soundtracks")
            prior = case["prior_speech_event_range"]
            require(prior is None or in_audio(prior, extent["aac48"]), "Invalid earlier speech event range")
            timestamp_checks += 2 * len(pts)
            selected = {carrier: validate_observation(case["observations"][carrier], picture_ticks, extent[carrier])
                        for carrier in CARRIERS}
            outcomes = {carrier: bool(span and hit(group["truth"], span)) for carrier, span in selected.items()}
            role = case["role"]
            for carrier, span in selected.items():
                stats = counts[carrier]
                stats["deliveries"] += 1
                stats["selected"] += span is not None
                source_anchors[carrier] += 3 if span else 0
                if role == "different_words":
                    stats["edited"] += 1
                    stats["useful"] += outcomes[carrier]
                    stats["misplaced"] += bool(span and not outcomes[carrier])
                    if outcomes[carrier]:
                        successful_speakers[carrier].add(group["speaker_group"])
                else:
                    category = "ordinary" if role in ("original", "gain") else "same_word" if role == "same_words" else "whole_track"
                    stats[category] += 1
                    stats[category + "_flagged"] += span is not None
                    pairs["new_" + category] += bool(carrier == "aac441" and span and not selected["aac48"])
            if role == "different_words":
                before, after = (outcomes[carrier] for carrier in CARRIERS)
                pairs["useful_retained" if before and after else "useful_gained" if after else "useful_lost" if before
                      else "missed_both"] += 1
                pairs["new_misplaced"] += bool(selected["aac441"] and not after and not (selected["aac48"] and not before))
                pairs["prior_speech_event_useful"] += bool(prior and hit(group["truth"], prior))
    require(speakers == {group: 2 for group in range(1, 10)}, "The matched speaker roster has changed")
    require(receipt["verification"] == {"source_timestamps": timestamp_checks, "original_rgb_anchors": source_anchors},
            "Replay counts disagree with the bound observations")
    gate = {
        "counts": all(stats["deliveries"] == 108 for stats in counts.values()),
        "min_useful": counts["aac441"]["useful"] >= 2,
        "min_groups": len(successful_speakers["aac441"]) >= 2,
        "keeps_paired_useful": pairs["useful_lost"] == 0,
        "ordinary": pairs["new_ordinary"] <= 1,
        "same_word": pairs["new_same_word"] <= 1,
        "whole_track": pairs["new_whole_track"] == 0,
        "wrong": pairs["new_misplaced"] <= 1,
    }
    require(receipt["frozen_gate"] == gate, "Outcomes disagree with the locked comparison")
    return {"carriers": counts, "pairs": pairs, "gates": gate, "passed": all(gate.values())}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    args = cli.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as stream:
        outcome = review(json.load(stream))
    print(json.dumps(outcome, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
