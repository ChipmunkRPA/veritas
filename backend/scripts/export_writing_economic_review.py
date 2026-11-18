# SPDX-License-Identifier: Apache-2.0
"""Verify economic writing files and export a detector-independent comparison."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events, intervals
from scripts import writing_economic_paraphrase_holdout as holdout
from scripts import writing_multi_passage_panel as data
from scripts import writing_token_supervision_eval as evaluation
from scripts import writing_token_supervision_transfer as transfer
from scripts.review_writing_masked_context import prediction
from scripts.writing_focus_v3 import BASELINE_ASSETS

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.economic-writing-independent-review/1"
SYSTEMS = ("incumbent", "medium_report", "combined")
POPULATION = {"publisher_excerpt": 22, "human_substitution": 17, "generated_rewrite": 15}


def _units(regions):
    return sum(b - a for a, b in intervals(regions))


def _compact_coordinates(extent, whitespace):
    prefix = list(range(extent + 1))
    previous = removed = 0
    for start, stop in whitespace:
        if not previous <= start < stop <= extent:
            raise ValueError("Invalid whitespace coordinate runs")
        for i in range(previous, start + 1):
            prefix[i] = i - removed
        prefix[start:stop + 1] = [start - removed] * (stop - start + 1)
        previous = stop
        removed += stop - start
    for i in range(previous, extent + 1):
        prefix[i] = i - removed
    return prefix


def _project(regions, prefix):
    return [[prefix[left], prefix[right]] for left, right in regions if prefix[left] < prefix[right]]


def score(receipt, response):
    """Count native-coordinate predictions using the study's non-whitespace metric."""
    if receipt.get("schema") != SCHEMA:
        raise ValueError("Use the corresponding economic writing evaluation export")
    manifest, truth = receipt["manifest"], receipt["truth"]
    # Validate actual delivered Unicode bounds before translating coordinates.
    count_events(manifest, truth, response, minimum_iou=.30, minimum_recall=.50)
    compact_manifest, compact_truth, compact_response = map(deepcopy, (manifest, truth, response))
    prefixes = {row["id"]: _compact_coordinates(row["extent"][0], receipt["whitespace"][row["id"]])
                for row in compact_truth["cases"]}
    compact_truth_rows = {}
    for row in compact_truth["cases"]:
        prefix = prefixes[row["id"]]
        row["regions"] = _project(row["regions"], prefix)
        row["extent"] = [prefix[-1]]
        compact_truth_rows[row["id"]] = row
    for case in compact_manifest["cases"]:
        truth_row = compact_truth_rows[case["id"]]
        case["localization_truth"].update(source_units=truth_row["extent"][0], target_units=_units(truth_row["regions"]))
    for row in compact_response["predictions"]:
        location = row["localization"]
        if location["status"] == "ok":
            location["regions"] = _project(location["regions"], prefixes[row["id"]])
    compact_manifest["coordinate_space"] = "non_whitespace_unicode_codepoints"
    key = independent.digest(compact_manifest)
    compact_truth["manifest_sha256"] = compact_response["manifest_sha256"] = key
    events = count_events(compact_manifest, compact_truth, compact_response, minimum_iou=.30, minimum_recall=.50)
    roles = {row["id"]: row["label"] for row in manifest["cases"]}
    totals = {
        "files": sum(row["planned"] for row in events["strata"]),
        "assessed": sum(row["assessed"] for row in events["strata"]),
        "known_edits": sum(row["target_events"] for row in events["strata"]),
        "located": sum(row["matched_events"] for row in events["strata"]),
        "misplaced": sum(row["unmatched_proposals"] for row in events["strata"] if row["label"]),
        "ordinary": sum(not label for label in roles.values()),
        "ordinary_flags": sum(not roles[row["id"]] and bool(row["localization"].get("regions"))
                              for row in compact_response["predictions"]),
    }
    return {"coordinate_space": "non_whitespace_unicode_codepoints", "totals": totals, "strata": events["strata"],
            "native_manifest_sha256": independent.digest(manifest), "native_response_sha256": independent.digest(response)}


def _source_cases(original, html, reply):
    from app.detectors.text.extract import from_string

    _, publisher, body = holdout.bodies(original["document"], html)
    if publisher != original["publisher"] or data.digest(html) != original["source_sha256"]:
        raise ValueError("Publisher HTML differs from the exact retained source")
    normalized = from_string(body).text
    lo, hi = original["source_codepoints"]
    excerpt = normalized[lo:hi]
    if data.digest(normalized.encode()) != original["normalized_body_sha256"] or excerpt != original["text"]:
        raise ValueError("The ordinary excerpt does not match the normalized publisher source")
    left, right = original["edit_target_codepoints"]
    roles = {"publisher_excerpt": data.native(excerpt, ()) | {"kind": "ordinary", "changed_words": []}}
    if donor := original["ordinary_edit"]:
        start, stop = donor["source_codepoints"]
        content = normalized[start:stop]
        if data.digest(content.encode()) != donor["replacement_sha256"] or not (stop <= lo or start >= hi):
            raise ValueError("An ordinary comparison did not use the committed same-article passage")
        ordinary = data.native(excerpt[:left] + content + excerpt[right:], ())
        if ordinary["text"] != donor["text"]:
            raise ValueError("The same-article substitution differs from the exact held version")
        roles["human_substitution"] = ordinary | {"kind": "ordinary", "changed_words": []}
    pair = transfer.edit_pair(original, reply)
    if pair is not None:
        _, revision = pair
        addition = " ".join(reply.split())
        if revision["text"] != excerpt[:left] + addition + excerpt[right:]:
            raise ValueError("The edited version differs outside the reserved substitution")
        copied = {data.compact(excerpt[a:b]) for a, b, *_ in roles["publisher_excerpt"]["sentences"]}
        expected = [[a, b] for a, b, *_ in revision["sentences"]
                    if left <= a < b <= left + len(addition) and data.compact(revision["text"][a:b]) not in copied]
        if [row["source"] for row in revision["truth"]] != expected:
            raise ValueError("Edit truth includes earlier copied wording or omits revised sentences")
        roles["generated_rewrite"] = revision
    return roles


def _record(work, role, source, plan_case, measured, path):
    identity = {"id": f"{work['group'][:24]}/{role}", "input_sha256": data.digest(path.read_bytes())}
    text = plan_case["text"]
    extent = len(text)
    positions = [index for index, char in enumerate(text) if not char.isspace()]
    whitespace = [list(match.span()) for match in re.finditer(r"\s+", text)]
    prefix = _compact_coordinates(extent, whitespace)
    regions = [row["source"] for row in plan_case["truth"]]
    if (identity["input_sha256"] != plan_case["native_text_sha256"] or identity["input_sha256"] != measured["native_sha256"]
            or len(positions) != plan_case["nonspace_characters"] or len(positions) != measured["extent"]
            or measured["sentences"] != plan_case["sentences"] or measured["truth"] != _project(regions, prefix)
            or any(not 0 <= a < b <= extent or (prefix[a], prefix[b]) != (c, d)
                   for a, b, c, d, *_ in measured["sentences"])):
        raise ValueError("Delivered UTF-8 bytes, Unicode offsets, or non-whitespace offsets differ")
    terms = "U.S. government publisher excerpt and retained local research revisions"
    case = identity | {"modality": "text", "split": "confirmation", "source_sha256": source["source_sha256"],
                       "group": work["group"], "lineage": [work["group"]], "label": int(bool(regions)),
                       "family": work["generator"] if regions else role, "condition": f"{work['genre']}/{role}",
                       "license": {"id": terms, "evaluation_permitted": True, "redistribution_permitted": False},
                       "localization_truth": {"unit": "unicode_codepoints", "source_units": extent,
                                              "target_units": _units(regions)}}
    truth = identity | {"extent": [extent], "regions": regions}
    return identity, case, truth, whitespace, positions


def export(plan, intake, freeze, fit, held, sources, directory, output):
    holdout.review(freeze, held, sources)
    roster, seal, fit_record, measured, source = (data.read(path) for path in (plan, freeze, fit, held, sources))
    reservation = data.read(intake / "reservation.json.gz")
    fit_bytes = json.dumps(fit_record["fit"], ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode() + b"\n"
    if (evaluation.bind(plan) != measured["plan_sha256"] or roster["reservation_sha256"] != measured["reservation_sha256"]
            or evaluation.bind(intake / "reservation.json.gz") != measured["reservation_sha256"]
            or roster["parents"] != measured["parents"] or roster["parents"].keys() != {"qwen", "smol2"}
            or data.digest(fit_bytes) != seal["fit_sha256"]):
        raise ValueError("Use the exact retained held plan, publisher reservation, editor files, and fit")
    replies = {}
    for editor, expected in roster["parents"].items():
        path = intake / f"{editor}-edits.json.gz"
        if evaluation.bind(path) != expected["reply_sha256"] or evaluation.bind(intake / f"{editor}.json.gz") != expected["seed_sha256"]:
            raise ValueError("Editor replies or their reserved inputs differ from the scored run")
        replies.update({row["document"]: row["response"] for row in data.read(path)["responses"]})
    runs = {(row["group"], row["native_sha256"]): row for row in measured["cases"]}
    original = {item["group"]: item for item in reservation["sources"]}
    readbacks = {row["group"]: row for row in source["assembled_works"]}
    directory.mkdir(mode=0o700)
    (directory / "inputs").mkdir()
    (directory / "publisher").mkdir()
    arm, view, mode = holdout.POINT.split("/")
    cutoff = seal["points"][f"{arm}/{view}"][mode]
    entries, truth, index, whitespace = [], [], [], {}
    population, failed = Counter(), Counter()
    predictions = {system: [] for system in SYSTEMS}
    for work in roster["works"]:
        parent = original.pop(work["group"])
        if not independent._hash(parent["source_sha256"]):
            raise ValueError("A publisher original needs its exact SHA-256")
        parent_name = f"publisher/{parent['source_sha256']}.html"
        html = (intake / parent_name).read_bytes()
        rebuilt = _source_cases(parent, html, replies.pop(work["document"]))
        (directory / parent_name).write_bytes(html)
        roles = {case["native_text_sha256"]: role for role, case in rebuilt.items()}
        if "generated_rewrite" not in rebuilt:
            failed[work["generator"]] += 1
        published = readbacks.pop(work["group"])
        if [{key: case[key] for key in expected}
                for case, expected in zip(work["cases"], published["deliveries"], strict=True)] != published["deliveries"]:
            raise ValueError("The private delivery plan differs from its public coordinate commitments")
        for plan_case in work["cases"]:
            sha = plan_case["native_text_sha256"]
            if not independent._hash(sha):
                raise ValueError("A held delivery needs its exact SHA-256")
            role = roles.pop(sha)
            if rebuilt[role] != plan_case:
                raise ValueError("The reconstructed delivery differs from the exact held input")
            current = runs.pop((work["group"], sha))
            input_name = f"inputs/{sha}.txt"
            path = directory / input_name
            path.write_bytes(plan_case["text"].encode())
            identity, case, target, spaces, positions = _record(work, role, parent, plan_case, current, path)
            entries.append(case)
            truth.append(target)
            index.append({"id": identity["id"], "input": input_name, "source": parent_name})
            whitespace[identity["id"]] = spaces
            population[role] += 1
            old = current["incumbent"]
            new = evaluation.regions(current, arm, view, cutoff)
            combined = sorted({tuple(span) for span in old + new})
            old_ready, new_ready = current["incumbent_assessed"], bool(current["readings"][arm])
            for system, (assessed, ranges) in zip(SYSTEMS, ((old_ready, old), (new_ready, new), (old_ready or new_ready, combined)), strict=True):
                native = [[positions[a], positions[b - 1] + 1] for a, b in ranges]
                predictions[system].append(prediction(identity, assessed, native))
        if roles:
            raise ValueError("A source reconstruction produced an extra delivery")
    if original or replies or readbacks or runs or population != POPULATION or failed != measured["unusable"]:
        raise ValueError("The export lost a delivery, publisher group, or failed editor attempt")
    manifest = {"schema": independent.SCHEMA, "cases": entries,
                "purpose": "Locate the staged economic-writing edits in their exact submitted text."}
    bind = {"manifest_sha256": independent.digest(manifest), "split": "confirmation"}
    truth = bind | {"schema": TRUTH_SCHEMA, "cases": truth}
    index = {"schema": INDEX_SCHEMA, "manifest_sha256": bind["manifest_sha256"], "cases": index}
    receipt = {"schema": SCHEMA, "manifest": manifest, "truth": truth, "file_index": index, "whitespace": whitespace,
               "file_verification": verify_inputs(manifest, index, directory),
               "verified_publisher_originals": len(reservation["sources"]), "population": dict(population),
               "unusable_rewrites": dict(failed), "predictions": {}, "scores": {},
               "frozen_inputs_sha256": {key: evaluation.bind(path) for key, path in
                                        (("plan", plan), ("freeze", freeze), ("fit", fit), ("held", held), ("sources", sources))}}
    base = evaluation.bind(ROOT / "backend/app/evidence/document_passage_intervals.py")
    tuned_code = independent.digest(source["public_freeze_replay"]["local_implementation"])
    old_weights = [BASELINE_ASSETS["model.safetensors"]]
    new_weights = [fit_record["fit"]["runs"][arm]["weights_sha256"]]
    for system, code, weights in zip(SYSTEMS, (base, tuned_code, independent.digest({"base": base, "tuned": tuned_code})),
                                     (old_weights, new_weights, old_weights + new_weights), strict=True):
        response = bind | {"system": system, "method_kind": "weighted", "score_kind": "raw",
                           "implementation_sha256": code, "weight_sha256": weights,
                           "response_units": "At least one selected range (0 or 1)", "predictions": predictions[system]}
        scored = score(receipt, response)
        expected = measured["results"][system]["all"]
        if scored["totals"] != {key: expected[key] for key in scored["totals"]}:
            raise ValueError("Independent event counts differ from the frozen comparison")
        receipt["predictions"][system] = response
        receipt["scores"][system] = scored
    data.save(output, receipt)
    print(receipt["file_verification"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, fields in {"export": ("plan", "intake", "freeze", "fit", "held", "sources", "directory", "output"),
                         "score": ("receipt", "predictions")}.items():
        sub = commands.add_parser(name)
        for field in fields:
            sub.add_argument(field, type=Path)
    args = vars(parser.parse_args())
    if args.pop("command") == "export":
        export(**args)
    else:
        print(json.dumps(score(data.read(args["receipt"]), data.read(args["predictions"])), indent=2, sort_keys=True))
