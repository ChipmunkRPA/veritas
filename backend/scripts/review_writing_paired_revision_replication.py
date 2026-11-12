"""Audit the writing replication receipts and decisions without loading model weights."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.review_writing_reference_revision_transfer import native, read, sha, verify

CHECKS = ("current", "second")
GENERATED = {"qwen", "smol2"}
ARMS = ("current", "agreement", "paired", "combined")


def expected(row, key, previous, checks):
    focus = row["target"]
    scores = [[item[identity]["score"] for identity in key] for item in checks]
    verify(all(type(value) is float and 0 <= value <= 1 for readings in scores for value in readings),
           "Invalid response")
    current = [i for i, value in enumerate(scores[0]) if value >= .5][:6]
    agreed = [i for i in current if scores[1][i] >= .5]
    changed = [i for i in agreed if key[i] != previous[i]]
    unchanged = [i for i in agreed if key[i] == previous[i]]

    def odds(value):
        p = max(1e-7, min(1 - 1e-7, value))
        return p / (1 - p)

    comparable = previous[focus] != key[focus]
    paired = comparable and all(odds(record[key[focus]]["score"]) >= 2 * odds(record[previous[focus]]["score"])
                                for record in checks)
    results = {"unchanged_agreement": bool(unchanged), "paired_eligible": comparable,
               "draft_hunks": row["draft_hunks"], "hunks_within_focus": row["hunks_within_focus"]}
    a, b = row["focus"]
    mask = [int(not value.isspace()) for value in row["text"]]
    target = sum(mask[a:b])
    generated = row["role"] in GENERATED
    for arm, chosen in zip(ARMS, (current, agreed, [None] if paired else [], [*changed, None] if paired else changed),
                           strict=True):
        useful = False
        for index in chosen:
            ranges = [row["focus"]] if index is None else row["windows"][index]["spans"]
            hit = sum(sum(mask[max(a, x):min(b, y)]) for x, y in ranges)
            union = sum(sum(mask[x:y]) for x, y in ranges) + target - hit
            useful |= generated and hit * 2 >= target and hit * 10 >= union * 3
        results.update({f"{arm}_flag": bool(chosen), f"{arm}_useful": useful,
                        f"{arm}_exact_focus": focus in chosen or None in chosen,
                        f"{arm}_wrong_only": generated and bool(chosen) and not useful,
                        f"{arm}_no_flag": not chosen})
    return results


def audit(run, source):
    freeze, scoring, rows, observations, output = (read(run / f"{key}.json") for key in
                                                   ("freeze", "scoring-freeze", "cases", "private-outcomes", "aggregate"))
    paths = {name: sha(run / f"{name}.json") for name in ("freeze", "scoring-freeze", "cases", "private-outcomes", *CHECKS)}
    originals = {row["group"]: row for row in read(source / "originals.json")}
    verify(freeze["source_sha256"] == sha(source / "originals.json")
           and freeze["source_receipt_sha256"] == sha(source / "source-receipt.json")
           and scoring["freeze_sha256"] == paths["freeze"] and scoring["cases_sha256"] == paths["cases"]
           and output["score_freeze_sha256"] == paths["scoring-freeze"]
           and output["outcomes_sha256"] == paths["private-outcomes"], "Study receipt changed")
    generation = {}
    for family in GENERATED:
        path = run / f"generated-{family}.json"
        receipt = read(path)
        verify(sha(path) == scoring["generation"][family] and receipt["freeze_sha256"] == paths["freeze"]
               and receipt["results"].keys() == originals.keys()
               and all(reply["input_sha256"] == sha(originals[group]["before"]["focus_text"])
                       for group, reply in receipt["results"].items()), "A generator saw a different group or focus")
        generation[family] = receipt["results"]
    prior = {}
    if freeze["scope"] == "diagnostic":
        verify(all(sha(source / name) == value for name, value in freeze["prior"].items()), "Earlier source receipt changed")
        prior = {arm: read(source / f"{arm}.json")["responses"] for arm in CHECKS}
    seen, inputs, units, counts = set(), set(), {}, defaultdict(Counter)
    scores = []
    for arm in CHECKS:
        record = read(run / f"{arm}.json")
        values = record["responses"]
        cache = prior.get(arm, {})
        hits = values.keys() & cache.keys()
        verify(record["freeze_sha256"] == paths["scoring-freeze"] and output["score_receipts"][arm] == paths[arm]
               and record["cached"] == output["cached_responses"][arm] == len(hits)
               and record["forwards"] == output["native_forwards"][arm] == (len(values) - len(hits) + 3) // 4
               and all(values[key] == cache[key] for key in hits), "A reused or new classifier response changed")
        scores.append(values)
    for row, observation in zip(rows, observations, strict=True):
        group, role = row["group"], row["role"]
        source_row = originals[group]
        verify((group, role) not in seen, "Duplicate source group and role")
        seen.add((group, role))
        sentences, keys = native(row)
        inputs.update(keys)
        earlier = source_row["before"]
        if role in {"before", "edited"}:
            exact = earlier if role == "before" else source_row["after"]
            verify(exact["text"] == row["text"] and exact["focus"] == row["focus"], "Publisher excerpt changed")
        else:
            a, b = earlier["focus"]
            answer = generation[role][group]
            target = answer["text"].strip()
            verify(answer["eos"] and target == sentences[row["target"]] and target != source_row["after"]["focus_text"]
                   and row["text"] == earlier["text"][:a] + target + earlier["text"][b:],
                   "Generated replacement or preserved flank changed")
        if role == "before":
            units[group] = sentences, keys
        initial, original_keys = units[group]
        verify(initial[:2] == sentences[:2] and initial[3:] == sentences[3:] and row["reference"] == original_keys,
               "Exact matched native context changed")
        values = expected(row, keys, original_keys, scores)
        verify(observation == {key: row[key] for key in ("group", "publisher", "role", "text_sha256")} | values,
               "Case outcome differs")
        counts[f"{row['publisher']}/{role}"].update({"cases": 1, **values})
    verify(all((group, role) in seen for group in originals for role in ("before", "edited"))
           and all(score.keys() == inputs for score in scores) and len(inputs) == output["distinct_inputs"]
           and output["counts"] == counts, "Study populations or aggregate counts differ")
    controls = {row["group"] for row in observations if row["role"] == "edited" and row["agreement_flag"]}
    paired = {row["group"] for row in observations if row["role"] == "edited" and row["paired_flag"]}
    for family in GENERATED:
        cases = [row for row in observations if row["role"] == family]
        gains = sum(row["paired_useful"] and not row["current_useful"] for row in cases)
        losses = sum(row["current_useful"] and not row["paired_useful"] for row in cases)
        false = len(paired - controls)
        verify(output["admission"][family] == {"attempted": len(originals), "admitted": len(cases)}
               and output["decisions"][family] == {"admitted_groups": len(cases), "added_useful_groups": gains,
                  "lost_useful_groups": losses, "added_false_groups": false,
                  "replicated": len(cases) >= 6 and gains >= 2 and not losses and false <= 1},
               "Generator admission or study decision differs")
    return {"status": "verified_without_model_inference", "cases": len(rows), "groups": len(originals),
            "distinct_inputs": len(inputs), "aggregate_sha256": sha(run / "aggregate.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run, args.source), sort_keys=True))
