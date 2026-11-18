# SPDX-License-Identifier: Apache-2.0
"""Check the sealed report localizer on new official reports and two offline editors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.bench.writing_entropy_prominence import compact_range
from scripts import writing_multi_passage_panel as data
from scripts import writing_token_supervision_data as intake
from scripts import writing_token_supervision_eval as evaluation

SCHEMA = "chipmunk.writing-token-supervision-transfer/1"
POINT = "tuned/joint/medium"
COUNT = 32
WIDTH = 12


def prepare(archive, previous, freeze, output, reference=None):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    if evaluation.bind(archive) != intake.reports.GAO_SHA256:
        raise ValueError("The government report edition differs from its verified archive")
    bound = data.read(reference) if reference else None
    excluded, known, history = intake.exposure.exposures(bound["prior_exposures"] if bound else None)
    phrases = set()
    plans = {}
    for path in sorted(previous.glob("*.json.gz")):
        roster = data.read(path)
        plans[roster["role"]] = evaluation.bind(path)
        for work in roster["works"]:
            excluded.add(work["document"].removeprefix("GAO_"))
            for case in work["cases"]:
                if case["kind"] == "ordinary":
                    phrases.update(intake.overlap_keys(case["text"]))
    if set(plans) != {"fit", "calibration", "held"}:
        raise ValueError("Exclude all work roles from the first comparison")
    if plans["held"] != data.read(freeze)["held_roster_sha256"]:
        raise ValueError("The first comparison differs from the sealed operating points")
    pool = data.read(archive)["reports"]
    by_model = {"qwen": [], "smol2": []}
    for key in sorted(pool, key=lambda record: intake.order(f"{SCHEMA}/{record}")):
        if all(len(works) == COUNT for works in by_model.values()):
            break
        row = pool[key]
        if (key in excluded or intake.group_key(f"GAO_{key}") in known or row["published_date"] > "2020-12-31"
                or intake.group_key(row["text"]) in known):
            continue
        doc = from_string(row["text"])
        source = segment(doc)
        if len(source) < WIDTH:
            continue
        opening = int(intake.order(key)[:8], 16) % (len(source) - WIDTH + 1)
        excerpt = doc.text[source[opening].start:source[opening + WIDTH - 1].end]
        revised = segment(from_string(excerpt))
        if len(revised) != WIDTH:
            continue
        choices = [(a.start, b.end, i) for i, (a, b) in enumerate(zip(revised, revised[1:]))
                   if 2 <= i < WIDTH - 3 and 10 <= a.n_words <= 44 and 10 <= b.n_words <= 44 and a.paragraph == b.paragraph]
        if not choices:
            continue
        overlap = intake.overlap_keys(row["text"])
        if phrases & overlap or intake.group_key(excerpt) in known:
            continue
        phrases.update(overlap)
        family = min(by_model, key=lambda label: len(by_model[label]))
        lo, hi, index = choices[int(intake.order(f"{key}/edit")[:8], 16) % len(choices)]
        by_model[family].append({"document": key, "source_sha256": row["source_record_sha256"], "text": excerpt,
                                 "edit_target_codepoints": [lo, hi], "sentence_index": index})
    if any(len(works) != COUNT for works in by_model.values()):
        raise ValueError("Too few new records for both source-separated generator groups")
    output.mkdir(mode=0o700)
    for family, rows in by_model.items():
        path = output / f"{family}.json.gz"
        data.save(path, {"schema": SCHEMA, "family": family, "plans": plans, "point": POINT, "prior_exposures": history,
                         "archive_sha256": evaluation.bind(archive), "freeze_sha256": evaluation.bind(freeze), "sources": rows})
        if bound and evaluation.bind(path) != bound["parents"][family]["seed_sha256"]:
            raise ValueError("The reconstructed generator input roster differs")
    print("Frozen", COUNT, "fresh reports per local generator", flush=True)


def edit_pair(source, completion):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts import text_elife_rewrites_v273 as generation

    replacement = " ".join(completion.split())
    if not replacement or generation.WRAPPER.match(replacement) or generation.CONTACT.search(replacement):
        return None
    a, b = source["edit_target_codepoints"]
    body = source["text"]
    prior = body[a:b]
    if not .33 <= len(replacement.split()) / len(prior.split()) <= 2.5:
        return None
    clean = data.native(body, ())
    edited = body[:a] + replacement + body[b:]
    observed = from_string(edited)
    delivered = segment(observed)
    if clean is None or clean["text"] != body or observed.text != edited or len(delivered) < WIDTH:
        return None
    index = source["sentence_index"]
    trailing = WIDTH - index - 2
    unchanged = [clean["sentences"][:index], clean["sentences"][-trailing:]]
    outside = [delivered[:index], delivered[-trailing:]]
    if any(body[lo:hi] != sentence.text for block, actual in zip(unchanged, outside, strict=True)
           for (lo, hi, *_), sentence in zip(block, actual, strict=True)):
        return None
    changed = delivered[index:-trailing]
    if not 2 <= len(changed) <= 5 or changed[0].start != a or changed[-1].end != a + len(replacement):
        return None
    copied = {data.compact(body[lo:hi]) for lo, hi, *_ in clean["sentences"]}
    edits = [(s.start, s.end) for s in changed if data.compact(s.text) not in copied]
    if not edits:
        return None
    candidate = data.native(edited, edits)
    if candidate is None:
        return None
    return [clean | {"kind": "ordinary", "changed_words": []}, candidate | {"kind": "mixed", "changed_words": []}]


def assemble(seeds, first, second, output):
    from scripts import text_elife_rewrites_v273 as generation

    works, eligible, rejected, parents = [], Counter(), Counter(), {}
    for file in (first, second):
        packet = data.read(file)
        family = packet["family"]
        plan_file = seeds / f"{family}.json.gz"
        plan = data.read(plan_file)
        pin = evaluation.bind(plan_file)
        source_rows = plan["sources"]
        _, rev, weights, _ = generation.GENERATORS[family]
        if (plan["schema"] != SCHEMA or plan["point"] != POINT or family != plan["family"] or family in parents
                or (packet["source_plan_sha256"], packet["revision"], packet["weights_sha256"]) != (pin, rev, weights)
                or [row["document"] for row in source_rows] != [row["document"] for row in packet["responses"]]):
            raise ValueError("Source plan, generation order or licensed editor bytes differ")
        parents[family] = {"seed_sha256": pin, "response_sha256": evaluation.bind(file), "revision": rev, "weights_sha256": weights}
        for row, reply in zip(source_rows, packet["responses"], strict=True):
            pair = edit_pair(row, reply["response"])
            if pair is None:
                rejected[family] += 1
                continue
            works.append({"group": intake.group_key(f"GAO_{row['document']}"), "document": f"GAO_{row['document']}",
                          "genre": "gao_report", "generator": family, "operation": "local_paraphrase", "prior_project_exposure": False,
                          "original_source_sha256": row["source_sha256"], "cases": pair})
            eligible[family] += 1
    if set(parents) != set(generation.GENERATORS) or not all(eligible[family] for family in parents):
        raise ValueError("Require at least one valid staged rewrite from each frozen local editor")
    data.save(output, {"schema": SCHEMA, "role": "transfer", "works": works, "point": POINT, "parents": parents,
                       "prior_exposures": plan["prior_exposures"],
                       "fit_plans": plan["plans"], "freeze_sha256": plan["freeze_sha256"], "generated": eligible, "unusable": rejected})
    print("Eligible", dict(eligible), "with unusable replies", dict(rejected), flush=True)


def summary(cases, seal):
    arm, view, mode = POINT.split("/")
    cut = seal["points"][f"{arm}/{view}"][mode]
    return {"incumbent": evaluation.tally(cases), "medium_report": evaluation.tally(cases, arm, view, cut),
            "combined": evaluation.tally(cases, arm, view, cut, combined=True)}


def run(plan, checkpoint, weights, freeze, comparator, output, device):
    runtime = evaluation.learner.configure(device)
    roster, seal, previous = (data.read(file) for file in (plan, freeze, comparator))
    fit_file = weights / "fit.json"
    fit = data.read(fit_file)
    if (roster["schema"] != SCHEMA or roster["point"] != POINT or roster["freeze_sha256"] != evaluation.bind(freeze)
            or fit["runtime"] != runtime or roster["fit_plans"] != fit["plans"] or seal["fit_sha256"] != evaluation.bind(fit_file)
            or previous["plan_sha256"] != evaluation.bind(plan)):
        raise ValueError("The transferred source plans, measurements and operating point must remain fixed")
    measured = evaluation.measure(roster, checkpoint, weights, fit, device)
    references = {(work["group"], case["version"]): case for work in previous["works"] for case in work["variants"]}
    for row in measured:
        earlier = references.pop((row["group"], row["kind"]))
        selected, checked = evaluation.existing({"native_text_sha256": row["native_sha256"], "sentences": row["sentences"]}, earlier)
        row["incumbent"] = [compact_range(row["sentences"], first) for first in selected]
        row["incumbent_assessed"] = checked
    if references:
        raise ValueError("The baseline files do not match the separate transfer deliveries")
    data.save(output, {**{key: value for key, value in roster.items() if key != "works"}, "plan_sha256": evaluation.bind(plan),
                       "comparator_sha256": evaluation.bind(comparator), "cases": measured, "results": summary(measured, seal)})


def review(freeze, results):
    seal, receipt = data.read(freeze), data.read(results)
    if receipt["schema"] != SCHEMA or receipt["freeze_sha256"] != evaluation.bind(freeze) or receipt["point"] != POINT:
        raise ValueError("Transfer result differs from the selected operating point")
    evaluation.verify_positions(receipt["cases"])
    totals = summary(receipt["cases"], seal)
    if totals != receipt["results"]:
        raise ValueError("Independent transfer counts differ")
    print(json.dumps(totals, sort_keys=True))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    for command, names in (("prepare", ("archive", "previous", "freeze", "output")),
                           ("assemble", ("seeds", "first", "second", "output")),
                           ("run", ("plan", "checkpoint", "weights", "freeze", "comparator", "output")),
                           ("review", ("freeze", "results"))):
        mode = sub.add_parser(command)
        for name in names:
            mode.add_argument(name, type=Path)
        if command == "prepare":
            mode.add_argument("--reference", type=Path)
        if command == "run":
            mode.add_argument("--device", required=True, choices=("cpu", "mps"))
    args = vars(cli.parse_args())
    {"prepare": prepare, "assemble": assemble, "run": run, "review": review}[args.pop("action")](**args)
