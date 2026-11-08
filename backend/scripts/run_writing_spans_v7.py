"""Prospectively locked local span-learning comparison; no serving admission."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import run_writing_modernbert_v4 as old
from scripts import writing_modernbert_v4 as models
from scripts import writing_span_calibration_v7 as calibration
from scripts import writing_span_heads_v7 as heads
from scripts.writing_span_data_v7 import OLD, ROOT, SOURCE, prepare, sha
from scripts.writing_span_geometry_v7 import genuine_point, metrics, proposals_mask, require
from scripts.writing_span_native_v7 import Native

TRIAL = ROOT / "backend/data/independent-evaluation/writing-spans-v7"
ARMS = ("current", "token", "interval")
CODE = (
    tuple(
        "backend/scripts/" + name + ".py"
        for name in (
            "writing_span_data_v7",
            "writing_span_geometry_v7",
            "writing_span_heads_v7",
            "writing_span_calibration_v7",
            "writing_span_native_v7",
            "writing_span_runtime_v7",
            "run_writing_spans_v7",
            "review_writing_spans_v7",
            "specialist_runtime_v4",
            "run_writing_modernbert_v4",
            "writing_modernbert_v4",
            "writing_focus_v3",
            "prepare_writing_focus_v3",
            "writing_specialists_v5",
            "audit_writing_sources_v2",
        )
    )
    + tuple(
        "backend/app/detectors/text/" + name + ".py"
        for name in ("classifier", "coverage", "extract", "pipeline", "segment", "adapters/learned")
    )
    + ("backend/tests/test_writing_spans_v7.py", "docs/writing-spans-v7-protocol.md")
)
RULES = {
    "roles": {"fit": 288, "cal": 288, "point": 128, "diagnostic": 288},
    "groups_per_role": 32,
    "seed": 20261108,
    "useful_character_iou": 0.3,
    "interval_recovery_fraction": 0.5,
    "bootstrap_resamples": 2000,
    "genuine_only_point": "nextafter(second-highest of32 four-composition group maxima,+infinity); floor.001",
    "gates": "Interval must beat both current and matched token: more useful mixed documents, higher mean mixed characterIoU, no loss of previously useful mixed document, no fewer useful pureAI or shortAI documents or recovered source AI intervals; no more than1 extra genuine group overall or per genuine composition. Complete successful coverage and strict replay required.",
    "admission": "Consumed source development only. No serving/confidence admission, even if all development gates pass.",
    "features": "Complete native CPUfloat32 frozen ModernBERT features in RAM only;512-token stride0 windows;2threads; deterministic algorithms; no omitted non-whitespace character.",
    "fit": "Only288 fitting documents; token40epochs lr.003, interval60epochs lr.0005; AdamW decay.001,batch8,clip1,final epoch only.8queries,64latent,2decoder layers,4heads,FF128,dropout0;Hungarian class+5L1-2GIoU assignment,weightedCE+5L1+2(1-GIoU).",
    "calibration": "Separate nonnegative-logistic mapping for each arm, only288 calibration docs: proposed-interval retained publisher-AI non-whitespace fraction. Equal doc/group mass, raw clip1e-6,weighted standardization,L2slope.001,L-BFGS-B500,maxls50,gtol1e-9,ftol2.22e-16,projected gradient<=1e-6. Not customer authorship confidence.",
    "audit": "Reconstruct all992 documents from original CSV and connected source components;re-infer every native feature hash and all current responses;retrain both heads only on fitting features and require exact state equality;refit calibrators exactly;separately derive source masks/counts;recompute points,summary and gates. Local execution review, no human or independent-source validation.",
    "resource": "Sequential phases,900seconds,6GiB sampled treeRSS,min1536MiB available RAM/disk;failures retained,no implicit restart.",
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def code():
    return {p: sha(ROOT / p) for p in CODE}


def freeze(public):
    require(not (TRIAL / "freeze.json").exists() and not public.exists(), "New prospective paths required")
    inputs = read(TRIAL / "inputs.json")
    require(prepare() == inputs, "All original source slices and connected groups must reconstruct")
    smoke_path = ROOT / "output/evaluation/writing-span-v7-native-smoke-v2.json"
    smoke = read(smoke_path)
    require(
        smoke["implementation_sha256"] == sha(ROOT / "backend/scripts/writing_span_native_v7.py")
        and smoke["project_source_forwards"] == 0
        and smoke["fitted_heads"] == 0
        and smoke["records"][1]["status"] == "unsupported_coverage_refused"
        and all(r["native_feature_tap_exact"] for i, r in enumerate(smoke["records"]) if i != 1)
        and smoke["records"][-1]["windows"] > 1,
        "Supported native/overflow and explicit unsupported probe required",
    )
    preflight = read(TRIAL / "token-preflight.json")
    require(
        preflight["input_sha256"] == sha(TRIAL / "inputs.json")
        and preflight["documents"] == 992
        and preflight["native_nonwhitespace_coverage_failures"] == 0
        and preflight["project_model_forwards"] == 0,
        "Complete token-only source preflight required",
    )
    lock = {
        "schema": "chipmunk.writing-span-freeze/7",
        "rules": RULES,
        "code_sha256": code(),
        "input_sha256": sha(TRIAL / "inputs.json"),
        "source_sha256": sha(SOURCE),
        "original_plan_sha256": sha(OLD / "plan.json"),
        "original_freeze_sha256": sha(OLD / "freeze.json"),
        "runtime": old.runtime(),
        "installed_sha256": old.installed(),
        "model_assets": {"candidate": models.CANDIDATE_ASSETS, "current": models.BASELINE_ASSETS},
        "revisions": {"candidate": models.CANDIDATE_REVISION, "current": models.BASELINE_REVISION},
        "synthetic_smoke_sha256": sha(smoke_path),
        "synthetic_smoke": smoke,
        "token_preflight_sha256": sha(TRIAL / "token-preflight.json"),
        "token_preflight": preflight,
        "new_project_forwards": 0,
        "fitting_started": False,
        "serving_admitted": False,
    }
    paths = read(OLD / "freeze.json")["model_paths"]
    old.assets(Path(paths["candidate"]), models.CANDIDATE_ASSETS)
    old.assets(Path(paths["baseline"]), models.BASELINE_ASSETS)
    write(TRIAL / "freeze.json", lock)
    write(public, {**lock, "private_lock_sha256": sha(TRIAL / "freeze.json")})
    print(json.dumps({"frozen": True, "documents": len(inputs["cases"]), "project_forwards": 0}), flush=True)


def check():
    lock = read(TRIAL / "freeze.json")
    require(
        lock["rules"] == RULES
        and lock["code_sha256"] == code()
        and lock["runtime"] == old.runtime()
        and lock["installed_sha256"] == old.installed(),
        "Frozen implementation/method/runtime changed",
    )
    for path, expected in (
        (TRIAL / "inputs.json", lock["input_sha256"]),
        (SOURCE, lock["source_sha256"]),
        (OLD / "plan.json", lock["original_plan_sha256"]),
        (OLD / "freeze.json", lock["original_freeze_sha256"]),
    ):
        require(sha(path) == expected, "Frozen source/input changed")
    return read(TRIAL / "inputs.json")["cases"]


def phase_file(role):
    return TRIAL / (role + ".json")


def receipt(role):
    result = read(phase_file(role))
    require(
        result["status"] == "complete" and result["freeze_sha256"] == sha(TRIAL / "freeze.json"),
        "Complete exact predecessor required",
    )
    require(sha(TRIAL / (role + ".jsonl")) == result["journal_sha256"], "Predecessor journal changed")
    return result


def load_heads():
    r = receipt("fit")
    result = {}
    for kind in ("token", "interval"):
        path = TRIAL / (kind + "-head.json")
        require(sha(path) == r["head_sha256"][kind], "Sealed fitted head changed")
        result[kind] = heads.restore(kind, read(path)["state"])
    return result


def controls():
    seal = read(TRIAL / "control-seal.json")
    for filename, expected in seal["files"].items():
        require(sha(TRIAL / filename) == expected, "Pre-diagnostic control seal changed")
    for name in ("fit", "cal", "point"):
        receipt(name)
    return read(TRIAL / "calibrators.json"), read(TRIAL / "points.json")


def journal(role):
    return [json.loads(line) for line in (TRIAL / (role + ".jsonl")).read_text().splitlines()]


def scored(proposals, fitted):
    return [{**p, "score": calibration.predict(p["raw"], fitted)} for p in proposals]


def run(role):
    cases = [c for c in check() if c["role"] == role]
    require(role in RULES["roles"] and len(cases) == RULES["roles"][role], "Complete role required")
    path = phase_file(role)
    require(not path.exists() and not (TRIAL / (role + ".jsonl")).exists(), "No restart/overwrite permitted")
    learner = load_heads() if role != "fit" else None
    mapping = read(TRIAL / "calibrators.json") if role == "point" else None
    if role == "point":
        cal = receipt("cal")
        require(sha(TRIAL / "calibrators.json") == cal["calibrators_sha256"], "Calibration artifact changed")
    points = None
    if role == "diagnostic":
        mapping, points = controls()
    native, values, features = None, [], []
    started = time.monotonic()
    record = {"role": role, "freeze_sha256": sha(TRIAL / "freeze.json"), "status": "failed"}
    try:
        native = Native(with_current=role != "fit")
        with (TRIAL / (role + ".jsonl")).open("x") as stream:
            for i, case in enumerate(cases):
                f = native.features(case, audit=i == 0)
                row = {
                    "id": case["id"],
                    "group": case["group"],
                    "role": role,
                    "recipe": case["recipe"],
                    "label": case["label"],
                    "status": "ok",
                    "native": f["native"],
                }
                if role == "fit":
                    features.append(f)
                else:
                    current, row["current_native"] = native.current_proposals(case, audit=i == 0)
                    proposals = {
                        "current": current,
                        **{k: heads.propose(model, k, f) for k, model in learner.items()},
                    }
                    row["arms"] = {
                        k: scored(v, mapping[k]) if mapping is not None else v for k, v in proposals.items()
                    }
                    if points is not None:
                        row["measures"] = {
                            k: measure(case, v, points[k]["cut"]) for k, v in row["arms"].items()
                        }
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                values.append(row)
                if (i + 1) % 32 == 0:
                    print(
                        json.dumps(
                            {
                                "role": role,
                                "completed": i + 1,
                                "planned": len(cases),
                                "actual_pretrained_forwards": native.calls,
                            }
                        ),
                        flush=True,
                    )
        native.unchanged()
        if role == "fit":
            record["head_sha256"] = {}
            for kind in ("token", "interval"):
                model, history = heads.fit(features, kind)
                out = TRIAL / (kind + "-head.json")
                write(
                    out,
                    {
                        "state": heads.save_state(model),
                        "history": history,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "only_fitting_groups": 32,
                        "freeze_sha256": record["freeze_sha256"],
                    },
                )
                record["head_sha256"][kind] = sha(out)
        elif role == "cal":
            fitted = {
                k: calibration.fit([{**r, "proposals": r["arms"][k]} for r in values], cases) for k in ARMS
            }
            write(TRIAL / "calibrators.json", fitted)
            record["calibrators_sha256"] = sha(TRIAL / "calibrators.json")
            record["fit_receipt_sha256"] = sha(phase_file("fit"))
        elif role == "point":
            out = {k: genuine_point([{**r, "proposals": r["arms"][k]} for r in values]) for k in ARMS}
            write(TRIAL / "points.json", out)
            record["points_sha256"] = sha(TRIAL / "points.json")
            record["cal_receipt_sha256"] = sha(phase_file("cal"))
        elif role == "diagnostic":
            record["control_seal_sha256"] = sha(TRIAL / "control-seal.json")
        record.update(status="complete", documents=len(values), actual_pretrained_forwards=native.calls)
    except BaseException:
        record["failure"] = traceback.format_exc()
        raise
    finally:
        record["duration_s"] = time.monotonic() - started
        log = TRIAL / (role + ".jsonl")
        if log.exists():
            record["journal_sha256"] = sha(log)
        if native is not None:
            record["actual_pretrained_forwards"] = native.calls
        write(path, record)
    if role == "point":
        files = (
            "freeze.json",
            "fit.json",
            "cal.json",
            "point.json",
            "token-head.json",
            "interval-head.json",
            "calibrators.json",
            "points.json",
            "fit.jsonl",
            "cal.jsonl",
            "point.jsonl",
        )
        write(
            TRIAL / "control-seal.json",
            {"files": {p: sha(TRIAL / p) for p in files}, "diagnostic_forwards_started": False},
        )
    print(json.dumps(record), flush=True)


def measure(case, proposals, cut):
    mask = proposals_mask(case["text"], proposals, cut)
    value = metrics(case["text"], case["labels"], mask)
    recovered, total = 0, 0
    for a, b in case["truth_spans"]:
        locations = [i for i in range(a, b) if not case["text"][i].isspace()]
        require(locations, "No whitespace-only truth span")
        recovered += sum(bool(mask[i]) for i in locations) / len(locations) >= 0.5
        total += 1
    return {
        **value,
        "truth_intervals": total,
        "recovered_intervals": recovered,
        "useful_location": bool(case["label"] and value["character_iou"] >= 0.3),
    }


def summarize(rows):
    require(
        len(rows) == 288
        and len({r["id"] for r in rows}) == 288
        and Counter(r["recipe"] for r in rows)
        == {
            k: 32
            for k in (
                "human",
                "ai",
                "mixed-small",
                "mixed-large",
                "mixed-human-gap",
                "mixed-many",
                "human-small-matched",
                "human-large-matched",
                "human-many-matched",
            )
        }
        and all(r["status"] == "ok" and r["role"] == "diagnostic" for r in rows),
        "All diagnostic cases required",
    )
    arms = {}
    for arm in ARMS:
        recipes = {}
        for recipe in sorted({r["recipe"] for r in rows}):
            part = [r for r in rows if r["recipe"] == recipe]
            measures = [r["measures"][arm] for r in part]
            recipes[recipe] = {
                "documents": len(part),
                "highlighted_documents": sum(m["highlighted_characters"] > 0 for m in measures),
                "useful_locations": sum(m["useful_location"] for m in measures),
                "mean_character_iou": float(np.mean([m["character_iou"] for m in measures])),
                "recovered_intervals": sum(m["recovered_intervals"] for m in measures),
                "truth_intervals": sum(m["truth_intervals"] for m in measures),
                "highlighted_human_characters": sum(m["highlighted_human_characters"] for m in measures),
                "missed_ai_characters": sum(m["missed_ai_characters"] for m in measures),
            }
        genuine = [r for r in rows if r["label"] == 0]
        mixed = [r for r in rows if r["recipe"].startswith("mixed-")]
        arms[arm] = {
            "by_recipe": recipes,
            "genuine_groups_highlighted": len(
                {r["group"] for r in genuine if r["measures"][arm]["highlighted_characters"]}
            ),
            "mixed_documents": len(mixed),
            "useful_mixed_locations": sum(r["measures"][arm]["useful_location"] for r in mixed),
            "mean_mixed_character_iou": float(np.mean([r["measures"][arm]["character_iou"] for r in mixed])),
            "mixed_intervals_recovered": sum(r["measures"][arm]["recovered_intervals"] for r in mixed),
        }
    comparisons = {}
    for comparator in ("current", "token"):
        a, b = arms["interval"], arms[comparator]
        mixed = [r for r in rows if r["recipe"].startswith("mixed-")]
        lost = sum(
            r["measures"][comparator]["useful_location"] and not r["measures"]["interval"]["useful_location"]
            for r in mixed
        )
        checks = {
            "more_useful_mixed_locations": a["useful_mixed_locations"] > b["useful_mixed_locations"],
            "higher_mixed_character_iou": a["mean_mixed_character_iou"] > b["mean_mixed_character_iou"],
            "no_lost_useful_mixed_document": lost == 0,
            "preserves_pure_ai": a["by_recipe"]["ai"]["useful_locations"]
            >= b["by_recipe"]["ai"]["useful_locations"],
            "preserves_short_ai": a["by_recipe"]["mixed-small"]["useful_locations"]
            >= b["by_recipe"]["mixed-small"]["useful_locations"],
            "preserves_interval_recovery": a["mixed_intervals_recovered"] >= b["mixed_intervals_recovered"],
            "similar_genuine_group_flags": a["genuine_groups_highlighted"]
            <= b["genuine_groups_highlighted"] + 1,
            "similar_each_genuine_composition": all(
                a["by_recipe"][k]["highlighted_documents"] <= b["by_recipe"][k]["highlighted_documents"] + 1
                for k in a["by_recipe"]
                if k.startswith("human")
            ),
        }
        diffs = [
            np.mean(
                [
                    r["measures"]["interval"]["character_iou"] - r["measures"][comparator]["character_iou"]
                    for r in mixed
                    if r["group"] == group
                ]
            )
            for group in sorted({r["group"] for r in mixed})
        ]
        rng = np.random.default_rng(20261108 + 100)
        boots = np.asarray(diffs)[rng.integers(0, 32, (2000, 32))].mean(1)
        comparisons[comparator] = {
            "checks": checks,
            "passes_numerical_development_gates": all(checks.values()),
            "lost_previously_useful_mixed_documents": lost,
            "mean_iou_difference_source_group_bootstrap95": np.quantile(boots, [0.025, 0.975]).tolist(),
        }
    return {
        "schema": "chipmunk.writing-span-summary/7",
        "diagnostic_documents": 288,
        "diagnostic_groups": 32,
        "arms": arms,
        "interval_vs": comparisons,
        "passes_numerical_development_gates": all(
            v["passes_numerical_development_gates"] for v in comparisons.values()
        ),
        "serving_admitted": False,
        "customer_confidence_changed": False,
        "fresh_source_validation": False,
        "outside_human_reviewers": 0,
    }


def summary():
    check()
    controls()
    receipt("diagnostic")
    value = summarize(journal("diagnostic"))
    write(TRIAL / "summary.json", value)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "fit", "cal", "point", "diagnostic", "summary"))
    parser.add_argument("--public", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        require(args.public is not None, "Public prospective freeze path required")
        freeze(args.public)
    elif args.phase == "summary":
        summary()
    else:
        run(args.phase)
