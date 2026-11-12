"""Independently reconstruct scoped writing outcomes from retained source and score receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("current", "agreement", "changed_agreement", "paired", "combined")
MODELS = ("current", "second")
GENERATORS = {"granite", "olmo"}


def sha(value):
    if isinstance(value, Path):
        with value.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return hashlib.sha256(value.encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def verify(condition, message):
    if not condition:
        raise ValueError(message)


def native(row):
    text = row["text"]
    focus = row["target"]
    windows = row["windows"]
    verify(len(windows) in (5, 7) and focus == len(windows) // 2 and sha(text) == row["text_sha256"],
           "Frozen text commitment differs")
    units = [tuple(window["spans"][0 if i == 0 else 1]) for i, window in enumerate(windows)]
    verify(all(a < b <= len(text) and a >= 0 for a, b in units)
           and all(a <= b and text[a:b].strip() == "" for (_, a), (b, _) in zip(units[:-1], units[1:], strict=True))
           and list(units[focus]) == row["focus"] and units[0][0] == 0 and units[-1][1] == len(text),
           "Frozen sentence coordinates differ")
    inputs = []
    for i, window in enumerate(windows):
        ranges = units[max(0, i - 1):i + 2]
        binding = sha(" ".join(text[a:b] for a, b in ranges))
        verify(window["spans"] == [list(span) for span in ranges] and window["sha256"] == binding, "Native input differs")
        inputs.append(binding)
    return [text[a:b] for a, b in units], inputs


def evaluate(row, keys, baseline_keys, scores):
    responses = [[score[key]["score"] for key in keys] for score in scores]
    verify(all(type(v) is float and 0 <= v <= 1 for values in responses for v in values), "Nonprobability raw response")
    marked = [i for i, value in enumerate(responses[0]) if value >= .5][:6]
    agreed = [i for i in marked if responses[1][i] >= .5]
    unchanged, changed = ([i for i in agreed if (keys[i] == baseline_keys[i]) == same] for same in (True, False))
    focus = row["target"]

    def odds(value):
        value = min(max(value, 1e-7), 1 - 1e-7)
        return value / (1 - value)

    comparable = keys[focus] != baseline_keys[focus]
    paired = comparable and all(odds(score[keys[focus]]["score"]) >= 2 * odds(score[baseline_keys[focus]]["score"])
                                for score in scores)
    mask = [not ch.isspace() for ch in row["text"]]
    a, b = row["focus"]
    output = {
        "windows_with_unchanged_agreement": len(unchanged), "windows_with_changed_agreement": len(changed),
        "unchanged_agreement_file": bool(unchanged), "changed_agreement_file": bool(changed),
        "both_agreement_types": bool(unchanged and changed), "agreement_at_focus": focus in agreed,
        "diff_located_focus": bool(row["diff"]) and all(a <= start < stop <= b for start, stop in row["diff"]),
        "reference_comparable_focus": comparable,
    }
    arms = (marked, agreed, changed, [None] if paired else [], [*changed, None] if paired else changed)
    truth = sum(mask[a:b])
    for arm, indices in zip(ARMS, arms, strict=True):
        hit = False
        for i in indices:
            spans = [row["focus"]] if i is None else row["windows"][i]["spans"]
            intersection = sum(sum(mask[max(a, lo):min(b, hi)]) for lo, hi in spans)
            selected = sum(sum(mask[lo:hi]) for lo, hi in spans)
            if row["role"] in GENERATORS and intersection * 2 >= truth and 10 * intersection >= 3 * (selected + truth - intersection):
                hit = True
        output.update({arm + "_flag": bool(indices), arm + "_focus": focus in indices or None in indices,
                       arm + "_useful": hit, arm + "_wrong_only": row["role"] in GENERATORS and bool(indices) and not hit,
                       arm + "_no_flag": not indices})
    return output


def audit(run):
    source, originals, generation, freeze, cases, aggregate, outcomes = (
        read(run / f"{name}.json") for name in ("source-receipt", "originals", "generation-freeze", "scoring-freeze",
                                               "cases", "aggregate", "private-outcomes")
    )
    verify(source["originals_sha256"] == generation["originals_sha256"] == sha(run / "originals.json")
           and generation["source_receipt_sha256"] == aggregate["source_receipt_sha256"] == sha(run / "source-receipt.json")
           and source["catalog_freeze_sha256"] == sha(run / "catalog-freeze.json")
           and source["source_code_sha256"] == sha(Path(__file__).with_name("writing_reference_revision_sources.py"))
           and generation["code_sha256"] == freeze["code_sha256"]
           == sha(Path(__file__).with_name("writing_reference_revision_transfer.py"))
           and freeze["generation_sha256"] == sha(run / "generation-freeze.json")
           and aggregate["freeze_sha256"] == sha(run / "scoring-freeze.json")
           and aggregate["case_sha256"] == freeze["case_sha256"] == sha(run / "cases.json")
           and aggregate["outcomes_sha256"] == sha(run / "private-outcomes.json"), "Receipt commitments differ")
    catalog = read(run / "catalog-freeze.json")
    verify(catalog["source_plan_sha256"] == sha(run / "source-plan.json")
           and catalog["catalog_sha256"] == sha(run / "publisher/catalog.json")
           and source["publisher_xml"].keys() == {doi.split(".")[-2] for doi in catalog["pool"]}
           and all(sha(run / "publisher" / f"{group}-{side}.xml") == value
                   for group, pair in source["publisher_xml"].items() for side, value in pair.items()),
           "Publisher XML byte commitments differ")
    scores = []
    for arm in MODELS:
        path = run / f"{arm}.json"
        record = read(path)
        verify(aggregate["score_sha256"][arm] == sha(path)
               and record["forwards"] == aggregate["classifier_forwards"][arm]
               and record["freeze_sha256"] == aggregate["freeze_sha256"]
               and record["case_sha256"] == aggregate["case_sha256"], "Score receipt changed")
        scores.append(record["responses"])
    provenance = {row["group"]: row for row in originals}
    generation_receipts = {}
    for family in sorted(GENERATORS):
        path = run / f"generated-{family}.json"
        packet = read(path)
        verify(freeze["generation_receipts"][family] == sha(path)
               and packet["generation_sha256"] == freeze["generation_sha256"]
               and packet["results"].keys() == provenance.keys()
               and all(packet["results"][key]["input_sha256"] == sha(row["before"]["focus_text"])
                       for key, row in provenance.items()), "Generated input or receipt differs")
        generation_receipts[family] = packet["results"]
    counts, groups, before, inputs = defaultdict(Counter), defaultdict(set), {}, set()
    for row, outcome in zip(cases, outcomes, strict=True):
        units, keys = native(row)
        group, role, focus = row["group"], row["role"], row["target"]
        inputs.update(keys)
        if role == "before":
            verify(group not in before, "Duplicate earlier version")
            before[group] = units, keys
        source_row = provenance[group]
        if role in ("before", "edited"):
            side = "before" if role == "before" else "after"
            exact = source_row[side]
            verify(exact["text"] == row["text"] and exact["focus"] == row["focus"], "Frozen publisher excerpt changed")
        else:
            generated = generation_receipts[role][group]
            a, b = source_row["before"]["focus"]
            revised = generated["text"].strip()
            original = source_row["before"]["text"]
            verify(generated["eos"] and units[focus] == revised and row["text"] == original[:a] + revised + original[b:],
                   "Generated replacement was not exact or changed extra text")
        prior, reference = before[group]
        verify(row["reference"] == reference and units[:focus] == prior[:focus] and units[focus+1:] == prior[focus+1:],
               "Reference changed away from the paired focus")
        observed = evaluate(row, keys, reference, scores)
        verify(outcome == {key: row[key] for key in ("group", "role", "text_sha256")} | observed, "Case outcomes differ")
        counts[role].update({"cases": 1, **observed})
        for metric, value in observed.items():
            if value:
                groups[role, metric].add(group)
    verify(set(before) == set(provenance) and aggregate["counts"] == counts
           and aggregate["group_counts"] == {f"{role}/{metric}": len(values) for (role, metric), values in groups.items()}
           and aggregate["distinct_inputs"] == len(inputs) and all(score.keys() == inputs for score in scores),
           "Aggregate populations or outcomes differ")
    verify(aggregate["admission"] == {family: {"attempted": len(before), "admitted": counts[family]["cases"]}
                                     for family in sorted(GENERATORS)}
           and all(value == (len(inputs) + 3) // 4 for value in aggregate["classifier_forwards"].values()),
           "Generator admission or classifier batch counts differ")
    genuine = [row for row in outcomes if row["role"] == "edited"]
    generated = [row for row in outcomes if row["role"] in GENERATORS]
    control = {row["group"] for row in genuine if row["agreement_flag"]}
    for arm in ARMS[2:]:
        flag = {row["group"] for row in genuine if row[f"{arm}_flag"]}
        gains = [row for row in generated if row[f"{arm}_useful"] and not row["current_useful"]]
        losses = sum(row["current_useful"] and not row[f"{arm}_useful"] for row in generated)
        n, m, added = len({row["group"] for row in gains}), len({row["role"] for row in gains}), len(flag - control)
        verify(aggregate["follow_up"][arm] == {
            "lost_current_useful": losses, "added_useful_groups": n, "added_useful_families": m,
            "added_false_groups": added, "follow_up": losses == 0 and n >= 2 and m == 2 and added <= 1,
        }, "Advance rules differ")
    print(json.dumps({"status": "verified_without_model_inference", "groups": len(before), "cases": len(cases),
                      "distinct_inputs_per_checkpoint": len(inputs), "outcome_fields": sum(map(len, counts.values())),
                      "aggregate_sha256": sha(run / "aggregate.json")}, sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    audit(cli.parse_args().run)
