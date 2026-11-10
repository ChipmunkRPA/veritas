"""Corrected untouched held-out context: actor/sentence/ancestry components."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from scripts import evaluate_native_mouth_v127 as original
from scripts.video_appearance_corrected_v112 import scoped

f, calibration = original.f, original.calibration
ROOT, BASE, PANEL = f.ROOT, f.BASE, f.PANEL
TRIAL = BASE / "native-mouth-component-evaluation-v128"
PRIVATE = f.CACHE.parent / "native-mouth-component-evaluation-v128"
PUBLIC = PANEL / "native-mouth-component-evaluation-freeze-v128.json"
SOURCE_PLAN = BASE / "av-sentence-support-v104/preparation/plan.json"
CODE = (
    "backend/scripts/native_mouth_components_v128.py",
    "backend/tests/test_native_mouth_components_v128.py",
    "docs/native-mouth-component-evaluation-v128-method.md",
)
read, write, sha, require = f.read, f.write, f.sha, f.require


def connected(rows):
    """Reconstruct full components, not just per-actor source group codes."""
    require(
        rows
        and len({r["id"] for r in rows}) == len(rows)
        and len({r["asset_index"] for r in rows}) == len(rows),
        "Unique full source metadata required",
    )
    parent = {r["id"]: r["id"] for r in rows}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for row in rows:
        for field in ("actor", "sentence", "ancestry"):
            value = row[field]
            require(type(value) is str and bool(value), "Explicit actor/sentence/ancestry required")
            key = field, value
            if key in seen:
                a, b = root(row["id"]), root(seen[key])
                parent[max(a, b)] = min(a, b)
            seen[key] = row["id"]
    groups = defaultdict(set)
    for row in rows:
        component = root(row["id"])
        require(
            component == row["study_component"],
            "Declared full component contradicts independent reconstruction",
        )
        groups[component].add(row["study_role"])
    require(all(len(roles) == 1 for roles in groups.values()), "Connected context crosses study roles")
    return {r["asset_index"]: root(r["id"]) for r in rows}


def component_plan():
    rows = read(SOURCE_PLAN)["records"]
    mapping = connected(rows)
    counts = {
        role: {
            "sources": sum(r["study_role"] == role for r in rows),
            "connected_components": len({mapping[r["asset_index"]] for r in rows if r["study_role"] == role}),
        }
        for role in ("fusion_fit", "confidence_cal", "selection", "confirmation")
    }
    require(
        counts
        == {
            "fusion_fit": {"sources": 18, "connected_components": 4},
            "confidence_cal": {"sources": 10, "connected_components": 2},
            "selection": {"sources": 10, "connected_components": 3},
            "confirmation": {"sources": 10, "connected_components": 3},
        },
        "Exact independently reconstructed component inventory required",
    )
    declared = {r["asset_index"]: r["study_role"] for r in rows}
    require(
        all(r["study_role"] == declared[r["asset_index"]] for r in original.fitted.input_records()),
        "All feature roles must match full component assignment",
    )
    return mapping, counts


def paired_components(candidate, reference, mapping):
    def keys(rows):
        return [
            (r["asset_index"], r["source_group"], r["condition"], r["input_status"], r["label"]) for r in rows
        ]

    require(keys(candidate) == keys(reference), "Complete same-input paired outcomes required")
    components = sorted({mapping[r["asset_index"]] for r in candidate})
    require(2 <= len(components) < 5, "This sparse-component readout does not claim bootstrap intervals")

    def rate_differences(indices):
        positive = [
            i for i in indices if candidate[i]["input_status"] == "extracted" and candidate[i]["label"]
        ]
        negative = [
            i for i in indices if candidate[i]["input_status"] == "extracted" and not candidate[i]["label"]
        ]
        require(positive and negative, "Both ready classes required for paired component comparison")
        return {
            name: {
                "candidate_minus_reference": sum(
                    int(candidate[i][field]) - int(reference[i][field]) for i in selected
                )
                / len(selected),
                "group_bootstrap_ci95": None,
            }
            for name, field, selected in [
                ("ai_detection_rate_ready", "flagged", positive),
                ("ai_localization_rate_ready", "localized_iou_0_2", positive),
                ("false_alarm_rate_ready_genuine", "flagged", negative),
            ]
        }

    per_component = []
    for ordinal, component in enumerate(components):
        indices = [i for i, r in enumerate(candidate) if mapping[r["asset_index"]] == component]
        per_component.append(
            {
                "anonymous_component_ordinal": ordinal,
                "source_groups": len({candidate[i]["source_group"] for i in indices}),
                "planned_cases": len(indices),
                "rate_differences": rate_differences(indices),
            }
        )
    return {
        "source_groups": len({r["source_group"] for r in candidate}),
        "connected_components": len(components),
        "bootstrap_replicates": 0,
        "common_assessed_cases": sum(
            a["assessed"] and b["assessed"] for a, b in zip(candidate, reference, strict=True)
        ),
        "additional_assessed_cases": sum(
            a["assessed"] and not b["assessed"] for a, b in zip(candidate, reference, strict=True)
        ),
        "lost_assessed_cases": sum(
            not a["assessed"] and b["assessed"] for a, b in zip(candidate, reference, strict=True)
        ),
        "additional_localized_ai": sum(
            a["label"] and a["localized_iou_0_2"] and not b["localized_iou_0_2"]
            for a, b in zip(candidate, reference, strict=True)
        ),
        "lost_localized_ai": sum(
            a["label"] and not a["localized_iou_0_2"] and b["localized_iou_0_2"]
            for a, b in zip(candidate, reference, strict=True)
        ),
        "additional_genuine_flags": sum(
            not a["label"] and a["flagged"] and not b["flagged"]
            for a, b in zip(candidate, reference, strict=True)
        ),
        "removed_genuine_flags": sum(
            not a["label"] and not a["flagged"] and b["flagged"]
            for a, b in zip(candidate, reference, strict=True)
        ),
        "rate_differences": rate_differences(list(range(len(candidate)))),
        "component_aggregates": per_component,
        "scope": "Finite-panel paired differences. Only three independently reconstructed connected contexts in this held-out role; no bootstrap CI or population reliability bound is reported. All variants and sentence/actor/known-ancestry links stay together. Calibration weights/points remain source-group units.",
    }


def paired(candidate, reference):
    return paired_components(candidate, reference, component_plan()[0])


def corrected_write(path, body):
    if path.name == "result.json":
        _, inventory = component_plan()
        require(
            body["connected_groups"] == 10 and body["role"] in original.ROLES,
            "Exact original result schema required",
        )
        body = dict(body)
        body.pop("connected_groups")
        body.update(
            schema="chipmunk.native-mouth-component-evaluation/128",
            source_groups=10,
            connected_components=inventory[body["role"]]["connected_components"],
            component_inventory=inventory,
            weighting_and_research_point_unit="publisher source group; not an independent sentence/actor/ancestry component",
            uncertainty="Three connected held-out contexts; no bootstrap CI or population error-rate bound claimed.",
            scope=body["scope"]
            + " Component inventory corrected before any held-out scoring; original v127 had no attempted held-out worker. Current comparison retains source-group calibration weights/points and reports actual finite-panel errors without overstating independent sample size.",
        )
    write(path, body)


def identity():
    original.verify()
    require(
        all(not (original.TRIAL / f"bind-{i:02d}").exists() for i in (0, 1)),
        "Historical source-group bootstrap context must remain unused",
    )
    _, inventory = component_plan()
    return {
        "schema": "chipmunk.native-mouth-component-evaluation-freeze/128",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "original_unused_freeze_sha256": sha(original.PUBLIC),
        "source_component_plan_sha256": sha(SOURCE_PLAN),
        "private_calibrations_sha256": sha(calibration.PRIVATE / "calibrations-and-research-points.json"),
        "inventory": inventory,
        "minimum_component_count_for_interval_not_met": True,
        "bootstrap_replicates": 0,
        "scope": "Isolated corrected execution of unchanged v127 evaluator code object with full actor/sentence/ancestry component pairing, correct inventory and no misleading small-component CI. Original unused frozen v127 source/protocol remain untouched; classifiers, six calibrations and source-group research points are unchanged.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "held_out_cases_scored": 0, "bootstrap_replicates": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged full-component corrected context required",
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only locked corrected held-out stage permitted",
    )
    args = list(argv)
    args[3] = "scripts.native_mouth_components_v128"
    return subprocess.Popen(args, **kwargs)


def run(role):
    scoped(
        original.run,
        {"TRIAL": TRIAL, "verify": verify, "routed_popen": routed_popen, "__file__": str(Path(__file__))},
    )(role)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "bind"))
    p.add_argument("arguments", nargs="*")
    args = p.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One locked role required")
        run(args.arguments[0])
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in {"0", "1"},
            "Exact corrected output/role required",
        )
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Exclusive corrected context required")
        output.mkdir(exist_ok=False)
        scoped(
            original.evaluate,
            {"TRIAL": TRIAL, "PRIVATE": PRIVATE, "paired": paired, "write": corrected_write},
        )(original.ROLES[int(args.arguments[1])], output)
