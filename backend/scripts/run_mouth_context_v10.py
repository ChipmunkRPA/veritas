"""Freeze, score, calibrate and compare source-relative temporal mouth evidence."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch

from scripts import compare_mouth_probe_v6 as comparison
from scripts import compare_mouth_sequence_v4 as baseline
from scripts import mouth_context_v10 as method
from scripts import mouth_learner_v8 as feature_method
from scripts import run_mouth_learning_v8 as previous
from scripts import run_mouth_paired_v9 as paired
from scripts.liveportrait_mouth_benchmark import digest
from scripts.prepare_mouth_context_v10 import CONDITIONS, code_pins


def pins():
    root = Path(__file__).resolve().parents[2]
    names = [
        "backend/scripts/run_mouth_context_v10.py",
        "backend/scripts/mouth_context_v10.py",
        "backend/scripts/audit_mouth_context_v10.py",
        "backend/scripts/audit_mouth_video_v8.py",
        "backend/scripts/audit_mouth_probe_v6.py",
    ]
    return {**paired.implementation(), **code_pins(), **{p: digest(root / p) for p in names}}


def freeze(root, output):
    parent = root / "lipsync-v9"
    parent_lock = paired.check_lock(parent / "lock.json")
    calibration = root / "lipsync-v10/calibration-inputs-v2/manifest.json"
    plan = root / "lipsync-v10/calibration-plan.json"
    data = json.loads(calibration.read_text())
    if (
        data["plan_sha256"] != digest(plan)
        or len(data["items"]) != 24
        or any(r["status"] != "ready" for r in data["items"])
    ):
        raise ValueError("Every frozen ordinary calibration clip must be ready")
    parent_head = json.loads(Path(parent_lock["parent_head"]).read_text())
    point_groups = set(parent_head["point_groups"])
    if {r["group"] for r in data["items"]} != point_groups:
        raise ValueError("Calibration must retain the exact four separate point actors")
    for group in point_groups:
        if {r["condition"] for r in data["items"] if r["group"] == group} != set(CONDITIONS):
            raise ValueError("Missing calibration condition")
    for row in data["items"]:
        if (
            row["label"] != 0
            or row["role"] != "point"
            or digest(calibration.parent / row["path"]) != row["sha256"]
        ):
            raise ValueError("Ordinary source role/bytes changed")
    heads = {"v8": Path(parent_lock["parent_head"]), "v9": parent / "trained/head.json"}
    prior = {"v8": Path(parent_lock["parent_scores"]), "v9": parent / "development-scores.jsonl"}
    sources = {
        "parent_lock": parent / "lock.json",
        "calibration_manifest": calibration,
        "calibration_plan": plan,
        "prior_summary": Path(parent_lock["parent_summary"]),
        "calibration_source_audit": root / "lipsync-v10/calibration-source-audit.json",
    }
    audit = json.loads(sources["calibration_source_audit"].read_text())
    if audit["passed"] != 24 or audit["manifest_sha256"] != digest(calibration):
        raise ValueError("Every ordinary construction source audit must pass before scoring")
    output.mkdir(parents=True, exist_ok=False)
    body = {
        "schema": "chipmunk.mouth-context-lock/10",
        "method": method.METHOD,
        **{k: str(v.resolve()) for k, v in sources.items()},
        **{k + "_sha256": digest(v) for k, v in sources.items()},
        "heads": {
            k: {
                "path": str(p.resolve()),
                "sha256": digest(p),
                "weights_sha256": digest(p.parent / "learned.safetensors"),
            }
            for k, p in heads.items()
        },
        "prior_scores": {k: {"path": str(p.resolve()), "sha256": digest(p)} for k, p in prior.items()},
        "development": parent_lock["development"],
        "implementation_sha256": pins(),
        "candidate_priority": ["v9", "v8"],
        "deployment_eligible": False,
    }
    previous.write(output / "lock.json", body)
    print(
        json.dumps(
            {
                "lock_sha256": digest(output / "lock.json"),
                "ordinary_calibration_cases": 24,
                "development_cases": 88,
                "arms": body["candidate_priority"],
            }
        )
    )


def check_lock(path):
    lock = json.loads(path.read_text())
    if lock["implementation_sha256"] != pins():
        raise ValueError("Frozen V10 window/construction/scoring code changed")
    for key in (
        "parent_lock",
        "calibration_manifest",
        "calibration_plan",
        "prior_summary",
        "calibration_source_audit",
    ):
        if digest(Path(lock[key])) != lock[key + "_sha256"]:
            raise ValueError("Frozen experiment input changed: " + key)
    paired.check_lock(Path(lock["parent_lock"]))
    for head in lock["heads"].values():
        if (
            digest(Path(head["path"])) != head["sha256"]
            or digest(Path(head["path"]).parent / "learned.safetensors") != head["weights_sha256"]
        ):
            raise ValueError("Retained learned head/weights changed")
    for scores in lock["prior_scores"].values():
        if digest(Path(scores["path"])) != scores["sha256"]:
            raise ValueError("Immutable V8/V9 responses changed")
    return lock


def score_head(path, case, reference, models, head):
    captured = {}

    def contextual(centers, tracks, values, table):
        all_windows = method.windows(centers, tracks, values, table)
        captured.update(
            context_windows_all=all_windows,
            source_centers=sorted(centers),
            source_tracks={key: sorted({r.frame_index for r in samples}) for key, samples in tracks.items()},
        )
        return all_windows[:12]

    with (
        patch.object(comparison, "features", feature_method.features),
        patch.object(comparison, "margin_windows", side_effect=contextual),
    ):
        result = comparison.score_case(path, case, reference, models, head)
    return {**result, **captured}


def score(lock_path, stage, output):
    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video.adapters.loaders import load_dinov2_small

    lock = check_lock(lock_path)
    weights = previous.setup()
    models = {
        "yunet.2023mar": load_yunet("yunet.2023mar", "cpu", "float32"),
        "dinov2.small": load_dinov2_small("dinov2.small", "cpu", "float32"),
    }
    loaded, heads = {}, {}
    for arm, entry in lock["heads"].items():
        heads[arm] = json.loads(Path(entry["path"]).read_text())
        if heads[arm]["weights"] != weights:
            raise ValueError("Pretrained model declarations changed")
        loaded[arm] = previous.loaded_candidate(Path(entry["path"]))
    expected = []
    if stage == "calibration":
        manifest = Path(lock["calibration_manifest"])
        expected = [
            (case, None, manifest.parent, "ordinary-calibration")
            for case in json.loads(manifest.read_text())["items"]
        ]
    else:
        for panel in lock["development"]:
            manifest = Path(panel["manifest"])
            _, *refs = [json.loads(line) for line in Path(panel["receipt"]).read_text().splitlines()]
            expected.extend(
                (case, reference, manifest.parent, panel["family"])
                for case, reference in zip(json.loads(manifest.read_text())["items"], refs, strict=True)
            )
    prior = {}
    for arm, entry in lock["prior_scores"].items():
        _, *old = [json.loads(line) for line in Path(entry["path"]).read_text().splitlines()]
        prior[arm] = {r["id"]: r for r in old}
    header = {
        "schema": "chipmunk.mouth-context-scores/10",
        "lock_sha256": digest(lock_path),
        "stage": stage,
        "heads": {arm: entry["sha256"] for arm, entry in lock["heads"].items()},
        "planned": len(expected),
        "device": "mps",
        "baseline_device": "cpu",
        "weights": weights,
    }
    complete = []
    if output.exists():
        old_header, *complete = [json.loads(line) for line in output.read_text().splitlines()]
        if old_header != header or [r["id"] for r in complete] != [
            r[0]["id"] for r in expected[: len(complete)]
        ]:
            raise ValueError("Cannot resume changed source/context experiment")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    with output.open("a") as stream:
        for case, reference, root, family in expected[len(complete) :]:
            started = time.monotonic()
            row = {**case, "family": family, "mask_root": str(root), "heads": {}}
            try:
                path = root / case["path"]
                if case["status"] != "ready" or digest(path) != case["sha256"]:
                    raise ValueError("Prepared source bytes changed")
                if reference is None:
                    reference = baseline.score_case(path, case, models)
                row["current_reference"] = {
                    k: reference[k]
                    for k in (
                        "status",
                        "base_windows",
                        "sampling_centers",
                        "time_base",
                        "baseline_capture_parity",
                    )
                }
                for arm in ("v8", "v9"):
                    models["probe.dino"] = loaded[arm]
                    result = score_head(path, case, reference, models, heads[arm])
                    if stage == "development":
                        old_inputs = prior[arm][case["id"]]["model_inputs"]
                        if not paired.native_parity(result["model_inputs"], old_inputs) or any(
                            abs(a["margin"] - b["margin"]) > 1e-7
                            for a, b in zip(result["model_inputs"], old_inputs, strict=True)
                        ):
                            raise ValueError("Native inputs or frozen learned raw margins differ from " + arm)
                        result["prior_native_and_margin_parity"] = True
                    row["heads"][arm] = result
            except Exception as error:
                row["failure_reason"] = f"{type(error).__name__}: {error}"
            row["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "heads": {arm: value["status"] for arm, value in row["heads"].items()},
                        "failure": row.get("failure_reason"),
                        "seconds": row["runtime_s"],
                    }
                ),
                flush=True,
            )


def calibration_rows(lock_path, receipt):
    lock = check_lock(lock_path)
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    expected = json.loads(Path(lock["calibration_manifest"]).read_text())["items"]
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["stage"] != "calibration"
        or [r["id"] for r in rows] != [r["id"] for r in expected]
    ):
        raise ValueError("All exact ordinary calibration outcomes required")
    for row, source in zip(rows, expected, strict=True):
        if any(
            row[k] != source[k] for k in ("group", "role", "label", "condition", "sha256", "source_sha256")
        ):
            raise ValueError("Calibration source roles/labels/identities changed")
    return rows


def calibrate(lock_path, receipt, output):
    rows = calibration_rows(lock_path, receipt)
    body = {
        "schema": "chipmunk.mouth-context-points/10",
        "lock_sha256": digest(lock_path),
        "calibration_scores_sha256": digest(receipt),
        "calibration_scores": str(receipt.resolve()),
        "points": {arm: method.point(rows, arm) for arm in ("v8", "v9")},
        "ordinary_cases": 24,
        "point_groups": sorted({r["group"] for r in rows}),
        "response_kind": "raw source-relative margin; not AI probability",
        "deployment_eligible": False,
    }
    previous.write(output, body)
    print(json.dumps(body))


def review(lock_path, receipt, points_path, output):
    lock = check_lock(lock_path)
    points = json.loads(points_path.read_text())
    calibration_receipt = Path(points["calibration_scores"])
    if digest(calibration_receipt) != points["calibration_scores_sha256"]:
        raise ValueError("Calibration response receipt changed after point selection")
    cal_rows = calibration_rows(lock_path, calibration_receipt)
    if points["points"] != {arm: method.point(cal_rows, arm) for arm in ("v8", "v9")}:
        raise ValueError("The selected points differ from the fixed ordinary-window rule")
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    prior_summary = json.loads(Path(lock["prior_summary"]).read_text())
    expected = [r["id"] for r in prior_summary["cases"]["candidate"]]
    if (
        header["lock_sha256"] != digest(lock_path)
        or points["lock_sha256"] != digest(lock_path)
        or header["stage"] != "development"
        or [r["id"] for r in rows] != expected
    ):
        raise ValueError("Every exact frozen development outcome/point required")
    _, *absolute = [
        json.loads(line) for line in Path(lock["prior_scores"]["v8"]["path"]).read_text().splitlines()
    ]
    absolute_point = json.loads(Path(lock["heads"]["v8"]["path"]).read_text())["review_point"]
    families = [r["family"] for r in rows]
    results = {}
    for arm in ("v9", "v8"):
        relative, combined, selected = [], [], []
        for row, old in zip(rows, absolute, strict=True):
            root = Path(row["mask_root"])
            new = {**row, **row["heads"].get(arm, {"probe_windows": []})}
            relative += baseline.evaluate([new], "probe", points["points"][arm], root)
            signal = method.selected_signal(
                old["probe_windows"], absolute_point, new["probe_windows"], points["points"][arm]
            )
            if signal == "absolute":
                combined += baseline.evaluate([old], "probe", absolute_point, root)
            else:
                combined += baseline.evaluate([new], "probe", points["points"][arm], root)
            selected.append({"id": row["id"], "signal": signal})
        pseudo_rows = [
            {
                "status": "ok"
                if not r.get("failure_reason") and r["heads"].get(arm, {}).get("status") == "ok"
                else "failed",
                "baseline_replay_parity": r["heads"].get(arm, {}).get("baseline_replay_parity"),
                "prior_native_input_parity": r["heads"].get(arm, {}).get("prior_native_and_margin_parity"),
            }
            for r in rows
        ]
        gates = paired.gates(prior_summary["cases"]["candidate"], combined, families, pseudo_rows)
        results[arm] = {
            "relative_only": baseline.summary(relative),
            "combined": baseline.summary(combined),
            "gates": gates,
            "passes": all(gates.values()),
            "relative_cases": relative,
            "combined_cases": combined,
            "selected_signals": selected,
            "families": {
                family: baseline.summary([r for r, f in zip(combined, families, strict=True) if f == family])
                for family in sorted(set(families))
            },
        }
    selected_arm = next((arm for arm in lock["candidate_priority"] if results[arm]["passes"]), None)
    body = {
        "schema": "chipmunk.mouth-context-summary/10",
        "lock_sha256": digest(lock_path),
        "score_receipt_sha256": digest(receipt),
        "points_sha256": digest(points_path),
        "points": points["points"],
        "baseline": prior_summary["candidate"],
        "arms": results,
        "primary": "v9",
        "selected_for_confirmation_design": selected_arm,
        "deployment_eligible": False,
        "limitations": [
            "Four calibration actors and eight consumed video-development actors; no fresh or outside validation.",
            "Two predeclared combined arms with fixed priority; development selection still needs new confirmation.",
            "Absolute/relative margins are separate review signals, not calibrated probabilities or causal pixel proof.",
        ],
    }
    previous.write(output, body)
    print(
        json.dumps(
            {
                "baseline": body["baseline"],
                "points": body["points"],
                "arms": {
                    k: {name: value[name] for name in ("relative_only", "combined", "gates", "passes")}
                    for k, value in results.items()
                },
                "selected": selected_arm,
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "calibration", "development", "calibrate", "review"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--points", type=Path)
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.input, a.output)
    elif a.stage in ("calibration", "development"):
        score(a.input, a.stage, a.output)
    elif a.stage == "calibrate" and a.receipt:
        calibrate(a.input, a.receipt, a.output)
    elif a.stage == "review" and a.receipt and a.points:
        review(a.input, a.receipt, a.points, a.output)
    else:
        p.error("calibrate/review require --receipt; review also requires --points")
