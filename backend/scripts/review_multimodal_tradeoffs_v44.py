"""Post-result development review: image tradeoffs and actual writing-mask combinations.

No model is run or fitted. Previously observed sources cannot become fresh
validation because the comparison policy changed. All raw inputs stay private.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.evaluation_tradeoffs_v44 import observed_frontier, positive_hit_overlap
from scripts.writing_span_geometry_v7 import metrics, proposals_mask

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def combine_masks(masks, mode):
    need(mode in {"union", "intersection"} and len(masks) >= 2, "Explicit fixed combination required")
    need(
        all(isinstance(mask, np.ndarray) and mask.dtype == np.bool_ and mask.ndim == 1 for mask in masks)
        and all(mask.shape == masks[0].shape for mask in masks),
        "Complete source-aligned boolean masks required; unknown is not a clear mask",
    )
    return (np.logical_or if mode == "union" else np.logical_and).reduce(masks)


def complete_diagnostic_receipt(receipt, frozen_sha256):
    need(
        receipt["status"] == "complete" and receipt["role"] == "diagnostic"
        and receipt["documents"] == 288 and receipt["freeze_sha256"] == frozen_sha256,
        "Complete bound observed diagnostics required",
    )


def image_tradeoffs():
    path = ROOT / "docs/evaluation-panels/image-regional-calibration-v43-results.json"
    need(sha(path) == "eeee0ec58f300c015fd48df677b42cd5730f8abe5a1e86e0d34c6f0a53f7278a", "Completed image result changed")
    result = {}
    for fmt, part in read(path)["formats"].items():
        calibration = part["calibration"]
        summaries = {"current": calibration["baseline"]}
        summaries.update({row["id"]: row["summary"] for row in calibration["policy_results"]})
        models = {
            name: {
                "useful_component_mean": values["useful_photo_fraction_mean_over_groups"],
                **values["false_groups_by_kind"],
                "any_control_false_groups": values["any_negative_false_groups"],
            }
            for name, values in summaries.items()
        }
        result[fmt] = {
            "models": models,
            "comparison": observed_frontier(models, maximize=("useful_component_mean",), minimize=("original", "tone", "blur")),
            "historical_zero_increase_selection": calibration["selected_policy"],
            "historical_result_rewritten": False,
            "new_decision": "Retain all model settings as observed tradeoffs. Native candidate warrants comparison on unused sources; JPEG requires a separate decision.",
        }
    return {"input_sha256": sha(path), "formats": result, "fresh_validation": False}


def writing_combinations():
    trial = ROOT / "backend/data/independent-evaluation/writing-spans-v7"
    frozen, receipt = read(trial / "freeze.json"), read(trial / "diagnostic.json")
    need(sha(trial / "inputs.json") == frozen["input_sha256"], "Writing source/role inputs changed")
    geometry = "backend/scripts/writing_span_geometry_v7.py"
    need(sha(ROOT / geometry) == frozen["code_sha256"][geometry], "Frozen source-character geometry changed")
    complete_diagnostic_receipt(receipt, sha(trial / "freeze.json"))
    need(sha(trial / "diagnostic.jsonl") == receipt["journal_sha256"], "Observed writing journal changed")
    seal = read(trial / "control-seal.json")
    need(sha(trial / "control-seal.json") == receipt["control_seal_sha256"], "Original control seal changed")
    for filename, expected in seal["files"].items():
        need(sha(trial / filename) == expected, "Original head/point/calibration seal changed")
    cuts = read(trial / "points.json")
    cases = {row["id"]: row for row in read(trial / "inputs.json")["cases"] if row["role"] == "diagnostic"}
    rows = [json.loads(line) for line in (trial / "diagnostic.jsonl").read_text().splitlines()]
    need(len(rows) == len(cases) == 288 and len({r["id"] for r in rows}) == 288, "All observed documents required")
    need(Counter(row["recipe"] for row in rows) == {name: 32 for name in (
        "human", "ai", "mixed-small", "mixed-large", "mixed-human-gap", "mixed-many",
        "human-small-matched", "human-large-matched", "human-many-matched",
    )}, "Complete composition coverage required")
    methods = ("current", "token", "interval", "current_or_interval", "current_and_interval", "all_three_or")
    measured = {method: [] for method in methods}
    mixed_hits = {method: {} for method in methods}
    for row in rows:
        case = cases[row["id"]]
        need(row["status"] == "ok" and all(row[k] == case[k] for k in ("group", "recipe", "role", "label")), "Unchecked or unbound source cannot be dropped")
        masks = {arm: proposals_mask(case["text"], row["arms"][arm], cuts[arm]["cut"]) for arm in ("current", "token", "interval")}
        for arm, mask in masks.items():
            old = row["measures"][arm]
            actual = metrics(case["text"], case["labels"], mask)
            need(all(actual[k] == old[k] for k in actual), "Standalone source-character metrics did not reproduce")
        masks.update({
            "current_or_interval": combine_masks([masks["current"], masks["interval"]], "union"),
            "current_and_interval": combine_masks([masks["current"], masks["interval"]], "intersection"),
            "all_three_or": combine_masks([masks["current"], masks["interval"], masks["token"]], "union"),
        })
        for method, mask in masks.items():
            value = metrics(case["text"], case["labels"], mask)
            useful = bool(case["label"] and value["character_iou"] >= 0.3)
            recovered = 0
            for a, b in case["truth_spans"]:
                locations = [i for i in range(a, b) if not case["text"][i].isspace()]
                need(locations, "Non-whitespace truth interval required")
                recovered += sum(bool(mask[i]) for i in locations) / len(locations) >= 0.5
            measured[method].append({"case": case, "metrics": value, "useful": useful, "recovered": recovered})
            if method in mixed_hits and case["recipe"].startswith("mixed-"):
                mixed_hits[method][case["id"]] = useful

    summaries = {}
    for method, values in measured.items():
        mixed = [row for row in values if row["case"]["recipe"].startswith("mixed-")]
        genuine = [row for row in values if row["case"]["label"] == 0]
        summaries[method] = {
            "documents": len(values), "groups": len({row["case"]["group"] for row in values}),
            "mixed_documents": len(mixed), "useful_mixed_locations": sum(row["useful"] for row in mixed),
            "mean_mixed_character_iou": sum(row["metrics"]["character_iou"] for row in mixed) / len(mixed),
            "mixed_intervals_recovered": sum(row["recovered"] for row in mixed),
            "genuine_documents": len(genuine),
            "genuine_documents_highlighted": sum(row["metrics"]["highlighted_characters"] > 0 for row in genuine),
            "genuine_groups_highlighted": len({row["case"]["group"] for row in genuine if row["metrics"]["highlighted_characters"] > 0}),
            "genuine_human_characters_highlighted": sum(row["metrics"]["highlighted_human_characters"] for row in genuine),
            "by_recipe": {recipe: {"documents": sum(row["case"]["recipe"] == recipe for row in values), "useful_locations": sum(row["useful"] for row in values if row["case"]["recipe"] == recipe)} for recipe in sorted({row["case"]["recipe"] for row in values})},
        }
    return {
        "source_inputs_sha256": sha(trial / "inputs.json"), "observed_journal_sha256": sha(trial / "diagnostic.jsonl"),
        "control_seal_sha256": sha(trial / "control-seal.json"), "methods": summaries,
        "positive_complementarity": positive_hit_overlap(mixed_hits["current"], mixed_hits["interval"]),
        "combined_mask_gains_and_losses": {
            method: positive_hit_overlap(mixed_hits["current"], mixed_hits[method])
            for method in ("current_or_interval", "current_and_interval", "all_three_or")
        },
        "actual_combined_masks_measured": True,
        "combination_has_own_calibration": False,
        "fresh_validation": False,
        "scope": "Post-result, consumed controlled writing documents. Fixed boolean unions/intersections use complete original thresholded masks; no labels choose a method per document. Component cuts are unchanged. Standalone metrics reproduce exactly. This does not validate natural articles, email, word authorship or customer confidence.",
    }


def run(output):
    need(not output.exists(), "Never overwrite a prior review")
    result = {
        "schema": "chipmunk.multimodel-tradeoff-reconsideration/44",
        "image": image_tradeoffs(), "writing": writing_combinations(),
        "new_model_forwards": 0, "new_fits": 0, "fresh_source_validation": False,
        "serving_admitted": False, "automatic_rejection_from_one_worse_error_type": False,
        "scope": "Owner-directed re-evaluation of observed development evidence. Original frozen results remain unchanged. Models may be complementary; compare the actual combined rule before making a deployment decision.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"review": str(output), "image_formats": list(result["image"]["formats"]), "writing_methods": len(result["writing"]["methods"]), "new_model_forwards": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
