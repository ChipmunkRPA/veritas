# SPDX-License-Identifier: Apache-2.0
"""Recount the speech mismatch panels with standard-library Python."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from fractions import Fraction

POSITIVE = frozenset({"muted", "muted_faded"})


def overlap(prediction, truth):
    if prediction is None:
        return False
    start, end = prediction
    a, b = truth
    common = min(end, b) - max(start, a)
    full = max(end, b) - min(start, a)
    return common > 0 and Fraction(common, full) >= Fraction(2, 5)


def silent_mouth(document):
    assert document["schema"] == "chipmunk.video-silent-mouth-study/1"
    groups, totals, seen = set(), defaultdict(Counter), set()
    for row in document["rows"]:
        key = row["group"], row["delivery"], row["role"]
        assert key not in seen
        seen.add(key)
        groups.add(row["group"])
        result, truth = row["reading"], row["truth"]
        controls = row["role"] not in POSITIVE
        assert controls == (truth is None)
        flagged, current = result["added"], result["baseline"]
        if flagged is not None:
            assert current is None
            assert result["measurements"]["span"] == flagged
            begin, end = flagged
            timing = Fraction(row["picture_timescale"])
            before, middle, after = (round(row["pts"][result["frames"][part]] * timing * 16000)
                                     for part in ("before", "during", "after"))
            assert before < begin <= middle < end <= after
            assert result["measurements"]["mouth_frames"] >= 4 and result["measurements"]["peak_units"] <= 4
        a, b = [bool(value) if controls else overlap(value, truth) for value in (current, flagged)]
        audio = (bool(result["without_mouth"]) if controls else
                 any(overlap(span, truth) for span in result["without_mouth"]))
        totals[f'{row["delivery"]}/{row["role"]}'].update(views=1, current=a, combined=a or b,
                                                          new_flags=bool(flagged), sound_only=audio)
    assert document["totals"] == totals
    return {"recordings": len(groups), "deliveries": len(seen), "totals": totals}


def donor_contrast(document):
    assert document["freeze"]["schema"] == "chipmunk.video-lip-donor-contrast/1"
    rows = document["held"]
    assert len({(row["picture_sha256"], row["role"], row["delivery"]) for row in rows}) == len(rows)
    assert not {row["speaker"] for row in rows} & set(document["freeze"]["excluded_speakers"])
    totals = {}
    for key, cutoff in document["freeze"]["points"].items():
        delivery, visual, method, _ = key.split("/")
        features = visual + "/" + method
        counts = Counter()
        added, flags = set(), set()
        for row in rows:
            if row["delivery"] != delivery:
                continue
            baseline, trial = row["baseline"], row["measured"].get(features)
            addition = trial["audio"] if trial and trial["score"] > cutoff and baseline is None else None
            if row["role"] == "different_words":
                current, extra = (overlap(item, row["truth"]) for item in (baseline, addition))
                counts.update(edits=1, baseline=current, combined=current or extra, misplaced=bool(addition) and not extra)
                if extra:
                    added.add(row["speaker"])
            else:
                kind = ("controls" if row["role"] in {"original", "gain"} else
                        "same_words" if row["role"] == "same_words" else "whole_delay")
                counts.update({kind: 1, kind + "_baseline": bool(baseline), kind + "_combined": bool(baseline or addition)})
                if addition:
                    flags.add(row["speaker"])
        totals[key] = dict(counts) | {"additional_speakers": len(added), "additional_control_speakers": len(flags)}
    assert totals == document["total"]
    return {"recordings": len({row["picture_sha256"] for row in rows}), "deliveries": len(rows),
            "selected": {key: totals[value] for key, value in document["freeze"]["chosen"].items()}}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("method", choices=("donor", "silence"))
    cli.add_argument("receipt")
    flags = cli.parse_args()
    with gzip.open(flags.receipt, "rt") as stream:
        reader = donor_contrast if flags.method == "donor" else silent_mouth
        print(json.dumps(reader(json.load(stream)), sort_keys=True))
