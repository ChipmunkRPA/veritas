# SPDX-License-Identifier: Apache-2.0
"""Verify the held writing bytes and package exact locations for an independent detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events, intervals
from scripts import writing_multi_passage_panel as data
from scripts import writing_token_supervision_eval as evaluation
from scripts.review_writing_masked_context import prediction
from scripts.writing_focus_v3 import BASELINE_ASSETS

ROOT = Path(__file__).resolve().parents[2]


def unicode_ranges(positions, regions):
    return [[positions[a], positions[b - 1] + 1] for a, b in regions]


def export(plan, weights, freeze, results, output):
    roster, fit, seal, measured = (data.read(path) for path in (plan, weights, freeze, results))
    if (fit["plans"]["held"] != evaluation.bind(plan) or measured["plan_sha256"] != evaluation.bind(plan)
            or measured["freeze_sha256"] != evaluation.bind(freeze) or seal["fit_sha256"] != evaluation.bind(weights)):
        raise ValueError("Independent inputs must be bound to the exact fit, decisions and held readings")
    evaluation.verify_positions(measured["readings"])
    cases = {(row["group"], row["kind"]): row for row in measured["readings"]}
    arm, view, mode = seal["selection"]["tuned"].split("/")
    cutoff = seal["points"][f"{arm}/{view}"][mode]
    output.mkdir(mode=0o700)
    (output / "inputs").mkdir()
    entries, truth, index, before, after = [], [], [], [], []
    for work in roster["works"]:
        ordinary = next(row["native_text_sha256"] for row in work["cases"] if row["kind"] == "ordinary")
        for item in work["cases"]:
            current = cases.pop((work["group"], item["kind"]))
            text = item["text"]
            body = text.encode()
            sha = data.digest(body)
            raw_labels = [row["compact"] for row in item["truth"]]
            if (current["native_sha256"] != sha or current["sentences"] != item["sentences"] or current["truth"] != raw_labels):
                raise ValueError("The independent source digest or positions differ from the measured delivery")
            positions = [i for i, char in enumerate(text) if not char.isspace()]
            if len(positions) != current["extent"]:
                raise ValueError("The displayed character extent was altered")

            exact_truth = unicode_ranges(positions, raw_labels)
            original = unicode_ranges(positions, current["incumbent"])
            selected = unicode_ranges(positions, evaluation.regions(current, arm, view, cutoff))
            input_name = f"inputs/{sha}.txt"
            (output / input_name).write_bytes(body)
            identity = {"id": f"{work['group'][:24]}/{item['kind']}", "input_sha256": sha}
            terms = ("GovReport CC BY 4.0 and OpAI-Bench Apache-2.0" if work["genre"] == "gao_report"
                     else "Frozen source terms recorded in the retained writing panel")
            entries.append(identity | {"modality": "text", "split": "confirmation", "source_sha256": ordinary,
                                       "group": work["group"], "lineage": [work["group"]], "label": int(bool(exact_truth)),
                                       "condition": f"{work['genre']} {item['kind']}",
                                       "family": str(work["generator"]) if exact_truth else "dated original",
                                       "license": {"id": terms, "evaluation_permitted": True, "redistribution_permitted": False},
                                       "localization_truth": {"unit": "unicode_codepoints", "source_units": len(text),
                                                              "target_units": sum(b-a for a, b in intervals(exact_truth))}})
            truth.append(identity | {"extent": [len(text)], "regions": exact_truth})
            index.append({"id": identity["id"], "input": input_name, "source": f"inputs/{ordinary}.txt"})
            before.append(prediction(identity, current["incumbent_assessed"], original))
            combined = [list(region) for region in sorted({tuple(span) for span in original + selected})]
            after.append(prediction(identity, current["incumbent_assessed"] or bool(current["readings"][arm]), combined))
    if cases:
        raise ValueError("The independent roster omitted measured input files")
    manifest = {"schema": independent.SCHEMA, "cases": entries}
    proof = {"manifest_sha256": independent.audit_manifest(manifest)["manifest_sha256"], "split": "confirmation"}
    truth = proof | {"schema": TRUTH_SCHEMA, "cases": truth}
    index = {"manifest_sha256": proof["manifest_sha256"], "schema": INDEX_SCHEMA, "cases": index}
    verified = verify_inputs(manifest, index, output)
    base_code = evaluation.bind(ROOT / "backend/app/evidence/document_passage_intervals.py")
    pins = [BASELINE_ASSETS["model.safetensors"]]
    output_counts = {}
    for name, measured, pin, code in (
        ("existing", before, pins, base_code),
        ("combined", after, pins + [fit["runs"][arm]["weights_sha256"]], independent.digest(seal["implementation"])),
    ):
        result = proof | {"system": name, "score_kind": "raw", "response_units": "At least one selected range (0 or 1)",
                          "method_kind": "weighted", "predictions": measured, "implementation_sha256": code, "weight_sha256": pin}
        counts = count_events(manifest, truth, result, minimum_iou=.30, minimum_recall=.50)
        output_counts[name] = {"events": sum(row["matched_events"] for row in counts["strata"]),
                               "details_sha256": independent.digest(counts)}
        data.save(output / f"{name}-predictions.json", result)
        data.save(output / f"{name}-events.json", counts)
    data.save(output / "manifest.json", manifest)
    data.save(output / "location-truth.json", truth)
    data.save(output / "input-index.json", index)
    receipt = {"input_verification": verified, "unit": "unicode_codepoints", "whitespace_included": True,
               "selection": seal["selection"]["tuned"], "counts": output_counts,
               "plans": fit["plans"], "prior_exposures": roster["prior_exposures"]}
    data.save(output / "receipt.json", receipt)
    print("Verified", len(entries), "independent inputs and their source commitments", output_counts, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "weights", "freeze", "results", "output"):
        cli.add_argument(name, type=Path)
    export(**vars(cli.parse_args()))
