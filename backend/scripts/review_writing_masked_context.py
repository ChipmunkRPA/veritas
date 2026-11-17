# SPDX-License-Identifier: Apache-2.0
"""Export the fresh writing panel for a separate detector and verify its native input bytes."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from app.bench import independent
from app.bench import writing_masked_context as bench
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events, intervals
from scripts import writing_masked_context as trial
from scripts import writing_masked_word_reader as reader
from scripts import writing_multi_passage_panel as data
from scripts.verify_writing_entropy_prominence import matching


def prepare(shards, pack, output, previous, reference):
    from scripts import writing_entropy_sources as intake

    pinned = data.read(reference)["exposed"] if reference else None
    identities, hashes, exposure = intake.exposures(pinned)
    earlier = []
    for path in previous:
        earlier.append([path.name.removesuffix(".json.gz"), trial.binding(path)])
        for work in data.read(path)["works"]:
            identities.add(work["document"].removeprefix("GAO_").upper())
            hashes.add(work["group_sha256"])
            hashes.update(variant["native_text_sha256"] for variant in work["variants"])
    works = intake.report_sources(shards, pack, identities, hashes)
    groups = Counter(work["generator_backend"] for work in works)
    if len(works) < 24 or len(groups) < 2:
        raise ValueError("The fresh panel needs separate government reports from two generators")
    data.save(output, {"schema": intake.SCHEMA, "cohort": "new_report_holdout",
                       "exposed": exposure, "excluded_plans": earlier, "works": works})
    print(len(works), "new source groups", dict(groups), flush=True)


def location(sentences, bounds):
    first, stop = bounds
    return [sentences[first][0], sentences[stop - 1][1]]


def bound_inputs(measured, plan):
    expected = {(work["document"], version["version"]): (work, version) for work in plan["works"]
                for version in work["variants"] if version["layout"] == "published"}
    for case in measured["cases"]:
        work, version = expected.pop((case["document"], case["role"]))
        body = version["text"].encode("utf-8")
        if (case["native_sha256"] != data.digest(body) or case["group"] != work["group_sha256"]
                or case["sentences"] != version["sentences"] or case["truth"] != [row["compact"] for row in version["truth"]]):
            raise ValueError("A reading differs from its exact original delivered text, label or coordinates")
        yield case, work, version, body
    if expected:
        raise ValueError("There are planned files without a measured result")


def prediction(identity, assessed, regions):
    if assessed:
        return identity | {"status": "ok", "response": int(bool(regions)),
                           "localization": {"status": "ok", "regions": regions}}
    reason = "The check had no eligible writing comparison for this input."
    return identity | {"status": "not_applicable", "response": None, "reason": reason,
                       "localization": {"status": "not_applicable", "reason": reason}}


def audit(plan_file, measured_file, vocabulary):
    from tokenizers import BertWordPieceTokenizer

    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment

    measured, plan = data.read(measured_file), data.read(plan_file)
    if measured["plan_sha256"] != trial.binding(plan_file) or trial.binding(vocabulary) != reader.ASSETS["vocab.txt"]:
        raise ValueError("Provide the exact native files and vocabulary for these readings")
    tokenizer = BertWordPieceTokenizer(str(vocabulary), lowercase=True)
    totals = Counter()
    for case, _, version, _ in bound_inputs(measured, plan):
        doc = from_string(version["text"])
        prefix = data._prefix(doc.text)
        native = [[s.start, s.end, prefix[s.start], prefix[s.end], s.paragraph, s.n_words] for s in segment(doc)]
        if case["sentences"] != native or case["native_sha256"] != data.digest(doc.text.encode()):
            raise ValueError("The native extraction no longer binds to the reported source positions")
        encoded = tokenizer.encode(doc.text, add_special_tokens=False)
        if case["token_count"] != len(encoded.ids):
            raise ValueError("Tokenizer produced a different number of native positions")
        totals.update(deliveries=1, sentences=len(case["readings"]))
        for index, count, words, pieces, _, _, locations in case["readings"]:
            lo, hi, *_ = native[index]
            inside = [j for j, (a, b) in enumerate(encoded.offsets) if a >= lo and b <= hi]
            by_start = {encoded.offsets[j][0]: j for j in inside}
            candidates = []
            for word in reader.WORD.finditer(doc.text, lo, hi):
                first, stop = word.span()
                start = by_start.get(first)
                if start is None:
                    continue
                for last in range(start, min(start + reader.MAX_WORD_PIECES, len(encoded.ids))):
                    if encoded.offsets[last][1] > stop or encoded.tokens[last] == "[UNK]":
                        break
                    if encoded.offsets[last][1] == stop:
                        candidates.append(([first, stop], last - start + 1))
                        break
            sample = [candidates[(len(candidates) * (2*j+1)) // (2*words)] for j in range(words)]
            if count != len(inside) or pieces != sum(item[1] for item in sample) or locations != [item[0] for item in sample]:
                raise ValueError("The sampled words differ from the recorded model inputs")
            totals.update(words=words, pieces=pieces)
    print(dict(totals), flush=True)


def export(plan_file, measured_file, freeze_file, output):
    from scripts.writing_focus_v3 import BASELINE_ASSETS

    plan, measured, frozen = (data.read(path) for path in (plan_file, measured_file, freeze_file))
    if (plan.get("cohort") != "new_report_holdout"
            or measured["plan_sha256"] != trial.binding(plan_file) or measured["freeze_sha256"] != trial.binding(freeze_file)
            or trial.binding(plan_file) not in frozen["held_inputs"]):
        raise ValueError("Provide the source-separated, previously frozen government-report panel")
    trial.verify_ranges(measured["cases"])
    selected = frozen["choice"]
    arm, point = selected["arm"], frozen["thresholds"][selected["policy"]][selected["arm"]]
    ordinary = {work["document"]: variant["native_text_sha256"] for work in plan["works"]
                for variant in work["variants"] if variant["version"] == "ordinary"}
    output.mkdir(mode=0o700)
    files = output / "inputs"
    files.mkdir()
    roster, truth, index, existing, combined = [], [], [], [], []
    native_count = [0, 0]
    for case, work, version, body in bound_inputs(measured, plan):
        case_id = f"{work['document']}/{case['role']}"
        parent = ordinary[work["document"]]
        digest = case["native_sha256"]
        original = f"inputs/{parent}.txt"
        path = f"inputs/{digest}.txt"
        (output / path).write_bytes(body)
        identity = {"id": case_id, "input_sha256": digest}
        before = [location(version["sentences"], bounds) for bounds in case["existing"]]
        best, assessed = bench.candidates(case, frozen["scales"], frozen["weights"])[arm]
        extra = [location(version["sentences"], best[1:])] if best is not None and best[0] > point else []
        after = before + extra
        regions = [row["source"] for row in version["truth"]]
        extent = len(version["text"])
        roster.append(identity | {"modality": "text", "split": "confirmation", "source_sha256": parent,
                                  "group": case["group"], "lineage": ["gao/" + work["document"]],
                                  "label": int(bool(regions)), "condition": "government report " + case["role"],
                                  "family": work["generator_backend"] if regions else "dated government report transcript",
                                  "license": {"id": "GovReport CC BY 4.0 and OpAI-Bench Apache-2.0",
                                              "evaluation_permitted": True, "redistribution_permitted": True},
                                  "localization_truth": {"unit": "unicode_codepoints", "source_units": extent,
                                                         "target_units": sum(end - start for start, end in intervals(regions))}})
        truth.append(identity | {"extent": [extent], "regions": regions})
        index.append({"id": case_id, "input": path, "source": original})
        existing.append(prediction(identity, case["incumbent_assessed"], before))
        combined.append(prediction(identity, case["incumbent_assessed"] or assessed, after))
        native_count[0] += matching(before, regions)
        native_count[1] += matching(after, regions)
    manifest = {"schema": independent.SCHEMA, "cases": roster}
    commitment = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    truth = commitment | {"schema": TRUTH_SCHEMA, "cases": truth}
    private_index = {"schema": INDEX_SCHEMA, "manifest_sha256": commitment["manifest_sha256"], "cases": index}
    verified = verify_inputs(manifest, private_index, output)
    receipt = commitment | {"method_kind": "weighted", "score_kind": "raw",
                            "response_units": "A source interval was selected for review (0 or 1)"}
    counts = {}
    for i, (system, predictions) in enumerate((("existing", existing), ("combined", combined))):
        weights = [BASELINE_ASSETS["model.safetensors"]]
        code = trial.binding(trial.ROOT / "backend/app/evidence/document_passage_intervals.py")
        if i == 1:
            weights.append(measured["assets"]["pytorch_model.bin"])
            code = independent.digest(frozen["implementation"])
        reply = receipt | {"system": system, "predictions": predictions,
                           "implementation_sha256": code, "weight_sha256": weights}
        calculated = count_events(manifest, truth, reply, minimum_iou=.30, minimum_recall=.50)
        if sum(row["matched_events"] for row in calculated["strata"]) != native_count[i]:
            raise ValueError("The independent native character counts differ")
        counts[system] = {"matched_events": native_count[i], "details_sha256": independent.digest(calculated)}
        data.save(output / f"{system}-predictions.json", reply)
        data.save(output / f"{system}-events.json", calculated)
    data.save(output / "receipt.json", {"input_verification": verified, "localization_unit": "unicode_codepoints",
                                       "unicode_whitespace_counted": True, "counts": counts, "candidate": selected})
    data.save(output / "manifest.json", manifest)
    data.save(output / "location-truth.json", truth)
    data.save(output / "input-index.json", private_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    replay = sub.add_parser("export")
    for name in ("plan_file", "measured_file", "freeze_file", "output"):
        replay.add_argument(name, type=Path)
    intake = sub.add_parser("prepare")
    for name in ("shards", "pack", "output"):
        intake.add_argument(name, type=Path)
    intake.add_argument("previous", type=Path, nargs="+")
    intake.add_argument("--reference", type=Path)
    check = sub.add_parser("audit")
    for name in ("plan_file", "measured_file", "vocabulary"):
        check.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    {"prepare": prepare, "export": export, "audit": audit}[args.pop("action")](**args)
