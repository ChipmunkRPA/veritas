"""Recount the fixed unique-default soundtrack transfer without source media or models."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.video-default-audio-motion-transfer/1"
POINT = 5_318_680
MAIN_ROLES = ("original", "gain", "same_words", "global_early", "global_late", "different_words")
ROUTING = {
    "first": [1, 0], "muxed_without_explicit_default": [1, 0], "both": [1, 1],
    "third": [0, 1, 0], "authored_without_default": [0, 0],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def span(value, *, allow_none=True):
    return allow_none and value is None or (
        isinstance(value, list) and len(value) == 2 and all(type(x) is int for x in value)
        and 0 <= value[0] < value[1] <= 64_000
    )


def useful(prediction, target):
    intersection = max(0, min(prediction[1], target[1]) - max(prediction[0], target[0]))
    union = max(prediction[1], target[1]) - min(prediction[0], target[0])
    return 5 * intersection >= 2 * union


def recount(receipt):
    require(receipt["schema"] == SCHEMA and receipt["point_units"] == POINT, "Frozen method identity differs")
    cases = receipt["cases"]
    require(len(cases) == 126 and set(receipt["routing"]) == set(ROUTING), "The held roster changed")
    require(all(receipt["routing"][key] == {"flags": flags, "withheld": 3}
                for key, flags in ROUTING.items()), "Observed default-track flags differ")
    counts, groups = Counter(), {}
    correct_speakers, alternate_speakers = set(), set()
    for row in cases:
        group, speaker, role = row["group"], row["speaker_group"], row["role"]
        require(type(group) is int and 1 <= group <= 18 and type(speaker) is int and 1 <= speaker <= 9
                and role in {*MAIN_ROLES, "alternate_unrelated"}, "Unexpected source group or role")
        order, roles = groups.setdefault(group, (speaker, set()))
        require(order == speaker and role not in roles, "Repeated role or changed speaker assignment")
        roles.add(role)
        prior, measured, claim, score, truth = (row[key] for key in (
            "prior_span", "measured", "proposed_span", "score_units", "truth_span"))
        require(type(measured) is bool and span(prior) and span(claim)
                and span(truth, allow_none=role != "different_words")
                and (claim is None) == (score is None)
                and (score is None or (measured and type(score) is int and 0 < score <= 20_000_000)),
                "Invalid source interval or comparison score")
        selected = claim if score is not None and score > POINT else None
        if role == "alternate_unrelated":
            require(prior is None and truth is None, "An alternate-track control contains an edit truth or prior")
            counts.update(alternate_measured=int(measured), alternate_highlights=int(selected is not None))
            if selected is not None:
                alternate_speakers.add(speaker)
        else:
            require(measured and prior == selected, "The transfer changed a prior disposition or interval")
            counts.update(paired=int(measured))
            if role == "different_words":
                correct = selected is not None and useful(selected, truth)
                counts.update(useful=int(correct), misplaced=int(selected is not None and not correct))
                if correct:
                    correct_speakers.add(speaker)
            else:
                require(truth is None, "An ordinary matched control contains edit truth")
                counts[f"{role}_highlights"] += int(selected is not None)
        counts["selected_excerpts"] += int(selected is not None)
    require(len(groups) == 18 and Counter(s for s, _ in groups.values()) == Counter({i: 2 for i in range(1, 10)})
            and all(roles == {*MAIN_ROLES, "alternate_unrelated"} for _, roles in groups.values()),
            "Speaker-blocked role membership differs")
    counts.update(correct_speaker_groups=len(correct_speakers), alternate_flagged_speaker_groups=len(alternate_speakers),
                  routing_withheld=3 * len(ROUTING))
    require(dict(counts) == receipt["recount"], "The result totals differ from the held rows")
    native = receipt["native_bindings"]
    require(native["production_decode"] == {
        "all": len(cases) + counts["routing_withheld"], "abstention": counts["routing_withheld"],
        "qualified": len(cases), "pcm_match": len(cases),
        "source_clock_verified": counts["paired"] + counts["alternate_measured"],
    } and native["native_frame_anchors"] == 3 * counts["selected_excerpts"]
        and native["audio_excerpt_commitments"] == counts["selected_excerpts"]
        and len(native["input_manifests"]) == 7
        and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in native["input_manifests"].values()),
        "The native custody totals or commitments differ")
    require(receipt["advanced_gate"] == (len(correct_speakers) >= 2 and counts["useful"] >= 2
                                        and counts["alternate_highlights"] <= 1),
            "Frozen advancement decision differs")
    return dict(counts)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("receipt", type=Path)
    args = cli.parse_args()
    try:
        record = json.loads(gzip.decompress(args.receipt.read_bytes()))
        print(json.dumps(recount(record), sort_keys=True))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        cli.exit(1, f"Default soundtrack transfer review failed. {exc}\n")
