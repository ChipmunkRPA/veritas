# SPDX-License-Identifier: Apache-2.0
"""Calibrate the supervised writing views and replay exact held positions independently."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from app.bench.writing_entropy_prominence import compact_range, matched
from scripts import writing_multi_passage_panel as data
from scripts import writing_token_supervision as learner
from scripts.verify_writing_entropy_prominence import matching
from scripts.writing_entropy_prominence import existing

SCHEMA = "chipmunk.writing-token-supervision-results/1"
ARMS = learner.ARMS
VIEWS = learner.VIEWS


def bind(path):
    return learner.pinned.hash_file(path)


def implementation():
    return {Path(path).name: bind(Path(path)) for path in (__file__, learner.__file__, learner.source.__file__)}


def baseline(plan, snapshot, output):
    roster = data.read(plan)
    prepared = [work | {"split": "test", "variants": [case | {"layout": "published", "version": case["kind"]}
                                                       for case in work["cases"]]} for work in roster["works"]]
    for work in prepared:
        del work["cases"]
    data.score({"plan_sha256": bind(plan), "works": prepared}, snapshot, output)


def measure(roster, checkpoint, weights, fit_result, device):
    import torch
    from safetensors.torch import load_file

    model, tokenizer, tokens = learner.initialization(checkpoint, preload=False)
    windows_by_file, cases = [], []
    for work in roster["works"]:
        for case in work["cases"]:
            windows_by_file.append(learner.windows(case | {"truth": [], "changed_words": []}, tokenizer, tokens))
            cases.append({"group": work["group"], "native_sha256": case["native_text_sha256"], "kind": case["kind"],
                          "generator": work["generator"], "genre": work["genre"], "operation": work["operation"],
                          "prior_project_exposure": work["prior_project_exposure"],
                          "sentences": case["sentences"], "truth": [row["compact"] for row in case["truth"]],
                          "extent": case["nonspace_characters"], "readings": {}})
    for arm in ARMS:
        snapshot = weights / f"{arm}.safetensors"
        if bind(snapshot) != fit_result["runs"][arm]["weights_sha256"]:
            raise ValueError("Local fitted weights differ from the calibrated checkpoint")
        model.load_state_dict(load_file(snapshot, device="cpu"), strict=True)
        model.eval().to(device)
        with torch.inference_mode():
            for case, windows in zip(cases, windows_by_file, strict=True):
                values = []
                for batch in learner.batches(windows):
                    labels, sentences, words = learner.forward(model, batch, device)
                    word_values = iter(words.cpu().tolist())
                    for (_, row), active in zip(labels, sentences.cpu().tolist(), strict=True):
                        readings = [(word.start, word.stop, next(word_values)) for word in row.words]
                        if not math.isfinite(active) or any(not math.isfinite(value) for _, _, value in readings):
                            raise ValueError("The detector produced a nonfinite local response")
                        values.append([row.index, round(active * learner.UNITS),
                                       [[lo, hi, round(value * learner.UNITS)] for lo, hi, value in readings]])
                    if next(word_values, None) is not None:
                        raise ValueError("Measured word positions do not align with the supplied sentences")
                case["readings"][arm] = sorted(values)
        print(f"Measured {len(cases)} {roster['role']} deliveries with {arm} encoder", flush=True)
    return cases


def candidates(case, arm, view):
    result = []
    for i, sentence, words in case["readings"][arm]:
        count = (len(words) + 3) // 4
        lexical = round(sum(sorted((row[-1] for row in words), reverse=True)[:count]) / count)
        score = {"sentence": sentence, "words": lexical, "joint": round((sentence + lexical) / 2)}[view]
        result.append((i, score))
    return result


def regions(case, arm, view, point):
    ranked = sorted(candidates(case, arm, view), key=lambda row: (-row[1], row[0]))
    selected = sorted(i for i, value in ranked[:learner.REGIONS] if value > point)
    return [compact_range(case["sentences"], (i, i + 1)) for i in selected]


def points(calibration):
    ordinary = [case for case in calibration if case["kind"] == "ordinary"]
    answer = {}
    for arm in ARMS:
        for view in VIEWS:
            maxima = sorted(max(value for _, value in indexed) for case in ordinary
                            if (indexed := candidates(case, arm, view)))
            if len(maxima) < 64:
                raise ValueError("The ordinary calibration coverage is insufficient")
            answer[f"{arm}/{view}"] = {name: maxima[math.ceil(fraction * len(maxima)) - 1]
                                      for name, fraction in learner.QUANTILES.items()}
    return answer


def cohorts(case):
    yield "all"
    yield case["genre"]
    if case["genre"] == "gao_report":
        yield case["generator"]
        yield f"operation/{case['operation']}"
        if not case["prior_project_exposure"]:
            yield "fresh_reports"


def tally(cases, arm=None, view=None, point=None, *, combined=False, alternate=False):
    rows = defaultdict(Counter)
    for case in cases:
        old = case.get("incumbent", [])
        if arm is None:
            selected = old
            assessed = case.get("incumbent_assessed", False)
        else:
            current = regions(case, arm, view, point)
            selected = list({tuple(span) for span in old + current}) if combined else current
            assessed = bool(case["readings"][arm]) or (combined and case.get("incumbent_assessed", False))
        truth = case["truth"]
        hits = matching(selected, truth) if alternate else len(matched(selected, truth))
        for cohort in cohorts(case):
            out = rows[cohort]
            out.update(files=1, assessed=assessed)
            if case["kind"] == "ordinary":
                out.update(ordinary=1, ordinary_flags=bool(selected), ordinary_ranges=len(selected))
            else:
                out.update(mixed=1, known_edits=len(truth), located=hits, missed=len(truth)-hits,
                           misplaced=len(selected)-hits, mixed_files_located=bool(hits))
    return dict(rows)


def decisions(calibration):
    frozen = points(calibration)
    tables = {}
    selected = {}
    for arm in ARMS:
        eligible = []
        for view in VIEWS:
            prefix = f"{arm}/{view}"
            for name, point in frozen[prefix].items():
                result = tally(calibration, arm, view, point)["all"]
                key = f"{prefix}/{name}"
                tables[key] = result
                utility = 2 * result["located"] - 4 * result["ordinary_flags"] - result["misplaced"]
                eligible.append((utility, -result["ordinary_flags"], learner.QUANTILES[name], key))
        selected[arm] = max(eligible)[-1]
    return {"points": frozen, "selection": selected, "development": tables}


def frozen_plan(fit_result, plan, role, runtime):
    if fit_result.get("schema") != learner.SCHEMA or fit_result["plans"][role] != bind(plan):
        raise ValueError("The supplied measurements would leave the frozen role separation")
    if fit_result["runtime"] != runtime:
        raise ValueError("Supply the same recorded inference runtime")
    roster = data.read(plan)
    if roster.get("role") != role:
        raise ValueError("Expected the predeclared source role")
    return roster


def calibrate(plan, checkpoint, weights, output, device):
    runtime = learner.configure(device)
    fit_file = weights / "fit.json"
    fit_result = data.read(fit_file)
    roster = frozen_plan(fit_result, plan, "calibration", runtime)
    readback = measure(roster, checkpoint, weights, fit_result, device)
    record = decisions(readback)
    data.save(output, {"schema": SCHEMA, "runtime": runtime, "fit_sha256": bind(fit_file),
                       "held_roster_sha256": fit_result["plans"]["held"], "implementation": implementation(),
                       "readings": readback, **record})
    print("Selected from separate calibration works", record["selection"], flush=True)


def outcomes(freeze, cases, *, alternate=False):
    result = {"incumbent": tally(cases, alternate=alternate)}
    for key in freeze["development"]:
        arm, view, name = key.split("/")
        point = freeze["points"][f"{arm}/{view}"][name]
        result[key] = tally(cases, arm, view, point, alternate=alternate)
        result[key + "/with_incumbent"] = tally(cases, arm, view, point, combined=True, alternate=alternate)
    return result


def held(plan, checkpoint, weights, freeze, comparator, output, device):
    runtime = learner.configure(device)
    seal, former = data.read(freeze), data.read(comparator)
    fit_file = weights / "fit.json"
    fit_result = data.read(fit_file)
    roster = frozen_plan(fit_result, plan, "held", runtime)
    if (seal["implementation"] != implementation() or seal["fit_sha256"] != bind(fit_file)
            or seal["held_roster_sha256"] != bind(plan) or former["plan_sha256"] != bind(plan)):
        raise ValueError("The fit, source plan or existing measurement differs from the predeclared held comparison")
    readback = measure(roster, checkpoint, weights, fit_result, device)
    existing_cases = {(work["group"], code["version"]): code for work in former["works"] for code in work["variants"]}
    for case in readback:
        previous = existing_cases.pop((case["group"], case["kind"]))
        located, checked = existing({"native_text_sha256": case["native_sha256"], "sentences": case["sentences"]}, previous)
        case["incumbent"] = [compact_range(case["sentences"], pair) for pair in located]
        case["incumbent_assessed"] = checked
    if existing_cases:
        raise ValueError("One or more original readings have no matched held input")
    result = outcomes(seal, readback)
    data.save(output, {"schema": SCHEMA, "freeze_sha256": bind(freeze), "plan_sha256": bind(plan),
                       "incumbent_sha256": bind(comparator), "readings": readback, "results": result})


def verify_positions(cases):
    seen = set()
    for case in cases:
        key = (case["group"], case["kind"])
        if key in seen or len(case["native_sha256"]) != 64:
            raise ValueError("Each delivery must keep one bound source identity")
        seen.add(key)
        for arm in ARMS:
            last = -1
            for index, value, words in case["readings"][arm]:
                if not last < index < len(case["sentences"]) or type(value) is not int or not words:
                    raise ValueError("A sentence measurement has invalid source positions")
                last = index
                left, right = case["sentences"][index][2:4]
                for start, stop, score in words:
                    if not left <= start < stop <= right <= case["extent"] or type(score) is not int:
                        raise ValueError("A word response falls outside its bound native sentence")
                    left = stop
        if any(not 0 <= a < b <= case["extent"] for a, b in case["truth"] + case.get("incumbent", [])):
            raise ValueError("Source labels and existing findings must fit the document")


def review(freeze, results):
    seal, receipt = data.read(freeze), data.read(results)
    if seal.get("schema") != SCHEMA or receipt.get("schema") != SCHEMA or bind(freeze) != receipt["freeze_sha256"]:
        raise ValueError("The held receipt refers to a different prior analysis")
    for packet in (seal, receipt):
        verify_positions(packet["readings"])
    if {row["group"] for row in seal["readings"]} & {row["group"] for row in receipt["readings"]}:
        raise ValueError("Calibration and held source groups overlap")
    recorded = decisions(seal["readings"])
    if any(recorded[name] != seal[name] for name in recorded):
        raise ValueError("Calibration decisions do not replay")
    result = outcomes(seal, receipt["readings"])
    if result != receipt["results"] or result != outcomes(seal, receipt["readings"], alternate=True):
        raise ValueError("Held outcomes do not agree with both independent matchers")
    keys = ["incumbent", *seal["selection"].values(), *(f"{key}/with_incumbent" for key in seal["selection"].values())]
    print(json.dumps({key: result[key] for key in keys}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    base = modes.add_parser("baseline")
    for name in ("plan", "snapshot", "output"):
        base.add_argument(name, type=Path)
    for name in ("calibrate", "held"):
        command = modes.add_parser(name)
        for path in ("plan", "checkpoint", "weights", "output"):
            command.add_argument(path, type=Path)
        command.add_argument("--device", required=True, choices=("cpu", "mps"))
        if name == "held":
            command.add_argument("--freeze", required=True, type=Path)
            command.add_argument("--comparator", required=True, type=Path)
    read = modes.add_parser("review")
    read.add_argument("freeze", type=Path)
    read.add_argument("results", type=Path)
    fields = vars(parser.parse_args())
    action = fields.pop("action")
    {"baseline": baseline, "calibrate": calibrate, "held": held, "review": review}[action](**fields)


if __name__ == "__main__":
    main()
