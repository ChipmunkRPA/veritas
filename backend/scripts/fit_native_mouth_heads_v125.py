"""Fit three independent small research heads on fitting groups only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import types
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import mouth_articulation_core_v122 as core
from scripts import review_native_mouth_features_v124 as reviewed
from scripts.video_appearance_corrected_v112 import scoped

f = reviewed.features
ROOT, BASE, PANEL = f.ROOT, f.BASE, f.PANEL
TRIAL = BASE / "native-mouth-heads-v125"
PRIVATE = f.CACHE.parent / "native-mouth-heads-v125"
PUBLIC = PANEL / "native-mouth-head-fit-freeze-v125.json"
CODE = (
    "backend/scripts/fit_native_mouth_heads_v125.py",
    "backend/tests/test_fit_native_mouth_heads_v125.py",
    "docs/native-mouth-head-fit-v125-method.md",
)
ARMS = ("visual_only", "concatenation", "interaction")
METHOD = "native-mouth-independent-linear/125"
read, write, sha, require = f.read, f.write, f.sha, f.require


def input_records():
    complete = read(PANEL / "native-mouth-features-complete-v124.json")
    require(
        complete["status"] == "complete"
        and complete["extracted_cases"] == 283
        and complete["unavailable_cases"] == 5,
        "Complete feature/audit bank required",
    )
    records = []
    for batch, commitment in enumerate(complete["batches"]):
        folder, _ = reviewed.completed_batch(batch)
        audit = reviewed.TRIAL / f"bind-{batch:02d}"
        for name, path in {
            "feature_result_sha256": folder / "output/result.json",
            "feature_profile_sha256": folder / "profile.json",
            "feature_journal_sha256": folder / "output/feature-cases.jsonl",
            "audit_result_sha256": audit / "output/result.json",
            "audit_profile_sha256": audit / "profile.json",
            "audit_journal_sha256": audit / "output/reviewed-cases.jsonl",
        }.items():
            require(sha(path) == commitment[name], "Whole qualified input artifact changed")
        r, p = read(audit / "output/result.json"), read(audit / "profile.json")
        require(
            commitment["batch"] == batch
            and r["status"] == "complete"
            and p["child_exit_code"] == 0
            and p["guard"] is None
            and p["sample_errors"] == 0
            and p["worker_receipt_sha256"] == sha(audit / "output/result.json"),
            "Successful exact complete feature replay required",
        )
        records.extend(
            json.loads(line) for line in (folder / "output/feature-cases.jsonl").read_text().splitlines()
        )
    require(
        len(records) == 288 and len(complete["batches"]) == 8, "Complete planned feature records required"
    )
    roles = defaultdict(set)
    for r in records:
        roles[r["source_group"]].add(r["study_role"])
    require(all(len(v) == 1 for v in roles.values()), "Connected source groups cannot cross roles")
    return records


def point_features(record):
    """Select every supported native centre; no condition/role/truth input."""
    from app.detectors.video.decode import FrameTable

    if record["status"] == "unavailable":
        return [], np.empty((0, 2304)), np.empty((0, 3840))
    require(record["status"] == "extracted", "Qualified extraction status required")
    table = FrameTable.from_json(record["table"])
    image = reviewed.array_file(record, np.float32)
    audio = reviewed.array_file(record["speech"], np.float64)
    images = {r["frame_index"]: r["feature_row"] for r in record["frames"] if r["status"] == "ok"}
    speech = {i: n for n, i in enumerate(record["speech"]["indices"])}
    indices = []
    visual = []
    sound = []
    for i in range(1, len(table) - 1):
        neighbours = (i - 1, i, i + 1)
        times = [Fraction(table.pts[j]) * table.tb for j in neighbours]
        if (
            i in speech
            and all(j in images for j in neighbours)
            and max(times[1] - times[0], times[2] - times[1]) <= Fraction(2, 25)
        ):
            indices.append(i)
            visual.append(core.temporal_visual(*(image[images[j]] for j in neighbours), times))
            sound.append(audio[speech[i]])
    return (
        indices,
        np.asarray(visual, np.float64).reshape(-1, 2304),
        np.asarray(sound, np.float64).reshape(-1, 3840),
    )


def balanced_weights(rows):
    require(
        rows and all(r["study_role"] == "fusion_fit" and r["label"] in (0, 1) for r in rows),
        "Only binary fitting labels may train heads",
    )
    cells = Counter((r["source_group"], r["label"], r["condition"]) for r in rows)
    groups = {label: {group for group, y, _ in cells if y == label} for label in (0, 1)}
    require(all(groups.values()), "Both classes required; do not replace unavailable inputs")
    conditions = defaultdict(set)
    for group, label, condition in cells:
        conditions[group, label].add(condition)
    # Each class gets half the mass; source groups and available conditions within
    # that class receive equal mass. Short original-only sources are retained.
    return np.asarray(
        [
            len(rows)
            / (
                2
                * len(groups[r["label"]])
                * len(conditions[r["source_group"], r["label"]])
                * cells[r["source_group"], r["label"], r["condition"]]
            )
            for r in rows
        ],
        np.float64,
    )


def fit_linear(rows, values):
    from sklearn.linear_model import LogisticRegression

    weights = balanced_weights(rows)
    x = np.asarray(values, np.float64)
    require(
        x.ndim == 2 and len(x) == len(rows) and x.shape[1] in (16, 32, 64) and np.isfinite(x).all(),
        "One complete finite declared representation per fitting point required",
    )
    mean = np.average(x, weights=weights, axis=0)
    scale = np.maximum(np.sqrt(np.average((x - mean) ** 2, weights=weights, axis=0)), 0.05)
    model = LogisticRegression(C=0.1, solver="lbfgs", max_iter=2000, tol=1e-8, random_state=0)
    model.fit((x - mean) / scale, [r["label"] for r in rows], sample_weight=weights)
    require(int(model.n_iter_.max()) < 2000, "Predeclared linear fit did not converge; no retry")
    return {
        "method": METHOD,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_.max()),
        "fit_points": len(rows),
        "response_kind": "raw_signed_margin_not_probability",
        "deployment_eligible": False,
    }


def margins(values, head):
    require(
        head["method"] == METHOD and head["response_kind"] == "raw_signed_margin_not_probability",
        "Exact raw head contract required",
    )
    x = np.asarray(values, np.float64)
    mean, scale, coef = (np.asarray(head[k], np.float64) for k in ("mean", "scale", "coef"))
    require(
        x.ndim == 2
        and mean.shape == scale.shape == coef.shape == (x.shape[1],)
        and all(np.isfinite(v).all() for v in (x, mean, scale, coef))
        and (scale > 0).all()
        and np.isfinite(head["intercept"]),
        "Finite matching head/input arrays required",
    )
    return ((x - mean) / scale) @ coef + head["intercept"]


def identity():
    reviewed.verify()
    records = input_records()
    return {
        "schema": "chipmunk.native-mouth-head-fit-freeze/125",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "complete_feature_audit_sha256": sha(PANEL / "native-mouth-features-complete-v124.json"),
        "packages": {
            p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "threadpoolctl")
        },
        "fitting_source_count": len({r["asset_index"] for r in records if r["study_role"] == "fusion_fit"}),
        "arms": list(ARMS),
        "linear_C": 0.1,
        "scale_floor": 0.05,
        "max_iterations": 2000,
        "tolerance": 1e-8,
        "scope": "Fitting-only genuine reference and three independent raw-margin linear heads. All supported native centres; central-frame known edit labels enter supervised fitting only. No calibration, operating point, selection/confirmation outcome or customer probability.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(json.dumps({"status": "frozen", "new_neural_calls": 0, "new_heads_fitted": 0}), flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(), "Unchanged fitting-only procedure required"
    )


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed fitting stage permitted",
    )
    routed = list(argv)
    routed[3] = "scripts.fit_native_mouth_heads_v125"
    return subprocess.Popen(routed, **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        f.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


def fit(output):
    from threadpoolctl import threadpool_limits

    require(
        all(os.environ.get(k) == "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        "One numerical thread required",
    )
    threadpool_limits(limits=1)
    records = [r for r in input_records() if r["study_role"] == "fusion_fit"]
    require(len({r["asset_index"] for r in records}) == 18, "All eighteen fixed fitting sources required")
    PRIVATE.mkdir(exist_ok=False)
    reference_rows = []
    for r in records:
        if r["condition"] != "original":
            continue
        _, visual, audio = point_features(r)
        reference_rows.extend(
            {
                "source_group": r["source_group"],
                "study_role": "fusion_fit",
                "condition": "original",
                "visual": v,
                "audio": a,
            }
            for v, a in zip(visual, audio, strict=True)
        )
    reference = core.fit_reference(reference_rows)
    reference_count = len(reference_rows)
    del reference_rows
    truth = {
        r["source"]["asset_index"]: r["source"]["edit_frames_half_open"]
        for r in read(f.clips.TRIAL / "plan.json")
    }
    rows = []
    parts = {arm: [] for arm in ARMS}
    coverage = []
    for r in records:
        indices, visual, audio = point_features(r)
        coverage.append(
            {
                "asset_index": r["asset_index"],
                "condition": r["condition"],
                "input_status": r["status"],
                "eligible_centres": len(indices),
            }
        )
        if not indices:
            continue
        representations = core.representations(visual, audio, reference)
        for arm in ARMS:
            parts[arm].append(representations[arm])
        bounds = truth[r["asset_index"]]
        # Truth is attached after the full blind eligible-index list is formed.
        rows.extend(
            {
                "source_group": r["source_group"],
                "study_role": "fusion_fit",
                "condition": r["condition"],
                "frame_index": i,
                "label": int(
                    r["condition"].startswith("ai_") and bounds is not None and bounds[0] <= i < bounds[1]
                ),
            }
            for i in indices
        )
    heads = {arm: fit_linear(rows, np.concatenate(parts[arm], axis=0)) for arm in ARMS}
    write(PRIVATE / "reference-and-heads.json", {"reference": reference, "heads": heads})
    write(PRIVATE / "fitting-labels-and-coverage.json", {"rows": rows, "coverage": coverage})
    result = {
        "schema": "chipmunk.native-mouth-head-fit/125",
        "status": "complete",
        "fitting_sources": 18,
        "reference_original_points": reference_count,
        "fitting_labelled_points": len(rows),
        "positive_central_frame_labels": sum(r["label"] for r in rows),
        "negative_central_frame_labels": sum(not r["label"] for r in rows),
        "raw_heads_fitted": len(heads),
        "dimensions": {k: len(v["coef"]) for k, v in heads.items()},
        "iterations": {k: v["iterations"] for k, v in heads.items()},
        "private_learned_artifact_sha256": sha(PRIVATE / "reference-and-heads.json"),
        "private_labels_coverage_sha256": sha(PRIVATE / "fitting-labels-and-coverage.json"),
        "freeze_sha256": sha(TRIAL / "freeze.json"),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "Fitting groups only, including every available native centre; three separately learned raw margins. No calibration/held-out accuracy, comparison winner, operating point, probability or serving contribution.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("freeze", "run", "bind"))
    p.add_argument("arguments", nargs="*")
    args = p.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "Freeze takes no arguments")
        freeze()
    elif args.command == "run":
        require(not args.arguments, "Fixed fitting stage takes no batch")
        run()
    else:
        verify()
        require(len(args.arguments) == 2 and args.arguments[1] == "0", "One fixed stage required")
        output = Path(args.arguments[0])
        require(output.resolve().is_relative_to(TRIAL.resolve()), "Exclusive private output required")
        output.mkdir(exist_ok=False)
        fit(output)
