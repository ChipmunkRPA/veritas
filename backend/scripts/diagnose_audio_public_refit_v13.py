"""Quantify cross-runtime refit differences without relaxing the exact check.

Original public arithmetic must pass first. Both unchanged fitters then run on
fitting sources only. This writes diagnostics, never replacement objects or new
calibration, never neural inference. Successful execution is not exact refit parity.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import review_audio_public_portable_v13 as portable  # noqa: E402

np = portable.np


def maximum_difference(a, b):
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Matching finite numerical shapes required")
    return float(np.max(np.abs(x - y), initial=0.))


def own_guards(rows, fitted):
    result = {}
    for label in (0, 1):
        pool = []
        for group in sorted({r["group"] for r in rows if r["label"] == label}):
            values = [portable.probability(r, fitted) for r in rows if r["label"] == label and r["group"] == group]
            maximum = None if None in values else max(p if label == 0 else 1 - p for p in values)
            pool.append({"group": group, "maximum": maximum, "rows": len(values)})
        result[str(label)] = pool
    return result


def diagnose(capsule, folder):
    original_review = portable.review(capsule, folder, False)
    body = portable.load(capsule, portable.CAPSULE_SHA)
    _, expected, _ = portable.original_inputs(folder)
    rows = portable.read_features(body, folder, expected)
    fitting = [r for r in rows if r["role"] == "fit"]
    calibration = [r for r in rows if r["role"] == "cal"]
    development = [r for r in rows if r["role"] == "development"]
    outcomes = portable.codec.unpack(body["outcomes"])
    freeze = portable.load(folder / "audio-v13-wavlm-freeze.json", portable.FREEZE_SHA)
    fitter = portable.frozen_fitter(freeze)
    arms = {}
    for arm in portable.ARMS:
        original = body["head"]["fitted"][arm]
        actual = fitter["fit"](fitting, arm)
        guards = own_guards(calibration, actual)
        original_reconstructed_guards = own_guards(calibration, original)
        changed, missing, max_probability = 0, 0, 0.
        for row, outcome in zip(development, outcomes, strict=True):
            p = portable.probability(row, actual)
            before = portable.probability(row, original)
            if p is not None:
                max_probability = max(max_probability, abs(p - before))
            probe = outcome["probe"]
            required = portable.initial(p, guards) == "original"
            missing += int(required and probe["status"] != "computed")
            band = portable.probability(probe, actual) if required and probe["status"] == "computed" else None
            band_raw = probe["raw"] if required and probe["status"] == "computed" else None
            decision = portable.final(p, band, row["raw"], band_raw, guards)["decision"]
            changed += int(decision != outcome["candidates"][arm]["decision"])
        excluded = {"transform", "coefficients"}
        metadata_equal = {k: v for k, v in actual.items() if k not in excluded} == {k: v for k, v in original.items() if k not in excluded}
        arms[arm] = {"complete_fitted_object_exact": actual == original,
            "non_numerical_metadata_equal": metadata_equal,
            "normalization_center_max_absolute_difference": maximum_difference(actual["transform"]["center"], original["transform"]["center"]),
            "normalization_scale_max_absolute_difference": maximum_difference(actual["transform"]["scale"], original["transform"]["scale"]),
            "coefficients_max_absolute_difference": maximum_difference(actual["coefficients"], original["coefficients"]),
            "own_guards_equal_to_same_arithmetic_original": guards == original_reconstructed_guards,
            "maximum_development_probability_difference": max_probability,
            "development_rows": len(development), "changed_directional_outcomes_using_available_probes": changed,
            "new_required_but_missing_public_probes": missing}
    from threadpoolctl import threadpool_info
    libraries = [{k: v for k, v in row.items() if k in {"user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer", "architecture"}}
                 for row in threadpool_info()]
    return {"schema": "chipmunk.public-audio-refit-diagnostics/13",
        "diagnostic_execution_completed": True, "original_arithmetic_review_passed": original_review["passed"],
        "all_refitted_objects_exact": all(a["complete_fitted_object_exact"] for a in arms.values()),
        "arms": arms, "runtime": original_review["runtime"], "numerical_libraries": libraries,
        "capsule_sha256": portable.CAPSULE_SHA,
        "reviewer_sha256": portable.digest(Path(__file__)), "portable_reviewer_sha256": portable.digest(Path(portable.__file__)),
        "original_exact_refit_check_relaxed": False, "replacement_objects_saved": False,
        "source_recordings_scored": 0, "serving_changed": False, "outside_reviewers": 0,
        "scope": "Diagnostic differences after complete original arithmetic replay. Refit predictions and own guards are measured using existing public probes; new required missing probes are explicit. No neural forwards, fresh predictive test, replacement fit/calibration or outside human review."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new diagnostic path")
    result = diagnose(args.capsule, args.capsule.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, allow_nan=False), flush=True)
