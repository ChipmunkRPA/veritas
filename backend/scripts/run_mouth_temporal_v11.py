"""Freeze, extract, fit and evaluate complete-video temporal mouth evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image import gend
from app.detectors.video.decode import decode_frames, probe_frames
from scripts import compare_mouth_probe_v6 as comparison
from scripts import compare_mouth_sequence_v4 as baseline
from scripts import mouth_temporal_v11 as method
from scripts import run_mouth_context_v10 as context
from scripts import run_mouth_learning_v8 as previous
from scripts import run_mouth_paired_v9 as paired
from scripts.audit_mouth_probe_v6 import check_chip, detector
from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_probe_v6 import mouth_chip
from scripts.prepare_mouth_context_v10 import geometry_frames
from scripts.prepare_mouth_temporal_v11 import AI_CONDITIONS, CONDITIONS, code_pins


def implementation():
    root = Path(__file__).resolve().parents[2]
    names = ["backend/scripts/run_mouth_temporal_v11.py", "backend/scripts/mouth_temporal_v11.py"]
    return {**context.pins(), **code_pins(), **{p: digest(root / p) for p in names}}


def freeze(root, output):
    parent = root / "lipsync-v10/experiment/lock.json"
    old = context.check_lock(parent)
    source_plan = root / "lipsync-v11/training-plan.json"
    recipe = json.loads(source_plan.read_text())
    if recipe["implementation_sha256"] != code_pins():
        raise ValueError("Video generation plan changed")
    sources = {
        "parent_lock": parent,
        "fit_plan": source_plan,
        "calibration_receipt": parent.parent / "calibration-scores.jsonl",
        "development_receipt": parent.parent / "development-scores.jsonl",
    }
    output.mkdir(parents=True, exist_ok=False)
    body = {
        "schema": "chipmunk.mouth-temporal-lock/11",
        "method": method.METHOD,
        **{k: str(v.resolve()) for k, v in sources.items()},
        **{k + "_sha256": digest(v) for k, v in sources.items()},
        "implementation_sha256": implementation(),
        "heads": old["heads"],
        "fit_groups": sorted(s["group"] for s in recipe["sources"]),
        "feature_names": method.FEATURE_NAMES,
        "deployment_eligible": False,
    }
    previous.write(output / "lock.json", body)
    print(
        json.dumps(
            {
                "lock_sha256": digest(output / "lock.json"),
                "features": 49,
                "fit_clips": 160,
                "calibration_clips": 24,
                "development_clips": 88,
            }
        ),
        flush=True,
    )


def check_lock(path):
    lock = json.loads(path.read_text())
    if lock["implementation_sha256"] != implementation():
        raise ValueError("Frozen temporal method/learning implementation changed")
    for k in ("parent_lock", "fit_plan", "calibration_receipt", "development_receipt"):
        if digest(Path(lock[k])) != lock[k + "_sha256"]:
            raise ValueError("Frozen input changed: " + k)
    context.check_lock(Path(lock["parent_lock"]))
    return lock


def measurements(path, heads, task, face_detector):
    table = probe_frames(path)
    frames = [f.rgb for f in decode_frames(path, table, range(len(table)))]
    if len(frames) != len(table):
        raise ValueError("Missing native picture during geometry measurement")
    mesh = geometry_frames(frames, table, task)
    geometry, trace = {}, []
    for i in sorted(mesh):
        points = [[mesh[i][j].x, mesh[i][j].y, mesh[i][j].z] for j in method.MESH_POINTS]
        values = method.geometry_values(points, table.width, table.height)
        geometry[i] = values
        trace.append(
            {
                "frame_index": i,
                "points": points,
                "raw_roll": values[4],
                "decoded_rgb_sha256": hashlib.sha256(frames[i].tobytes()).hexdigest(),
            }
        )
    if trace:
        for row, roll in zip(trace, np.unwrap([r["raw_roll"] for r in trace]), strict=True):
            geometry[row["frame_index"]][4] = float(roll)
    a, b = heads["v8"], heads["v9"]
    if not paired.native_parity(a["model_inputs"], b["model_inputs"]):
        raise ValueError("Frozen heads measured different native source chips")
    items = []
    for binding in a["model_inputs"]:
        i = binding["frame_index"]
        if (
            i not in geometry
            or binding["decoded_rgb_sha256"] != hashlib.sha256(frames[i].tobytes()).hexdigest()
        ):
            raise ValueError("Missing unique measured mesh or changed native source")
        if (
            binding["pts"] != table.pts[i]
            or binding["time_base"] != table.time_base
            or binding["time_s"] != table.seconds(i)
        ):
            raise ValueError("Measurement is attached to a different source clock")
        check_chip(frames[i], binding, face_detector)
        matrix = np.array([[float.fromhex(x) for x in r] for r in binding["alignment_matrix_hex"]])
        chip, _ = mouth_chip(gend.warp(frames[i], matrix, binding["alignment_side"]))
        items.append(
            {
                "frame_index": i,
                "pts": table.pts[i],
                "time_s": table.seconds(i),
                "decoded_rgb_sha256": binding["decoded_rgb_sha256"],
                "mouth_chip_rgb_sha256": binding["mouth_chip_rgb_sha256"],
                "values": geometry[i] + method.quality_values(chip),
            }
        )
    return items, trace, table


def fit_cases(lock):
    plan = json.loads(Path(lock["fit_plan"]).read_text())
    manifest = Path(lock["fit_plan"]).parent / "training-inputs/manifest.json"
    data = json.loads(manifest.read_text())
    expected = [
        (s["id"] + ":v11:" + c, s["group"], c, int(c in AI_CONDITIONS))
        for s in plan["sources"]
        for c in CONDITIONS
    ]
    if (
        data["plan_sha256"] != digest(Path(lock["fit_plan"]))
        or [(r["id"], r["group"], r["condition"], r["label"]) for r in data["items"]] != expected
        or any(r["role"] != "fit" or r["status"] != "ready" for r in data["items"])
    ):
        raise ValueError("All160 exact planned fitting clips and conditions must be ready")
    return manifest, data["items"]


def load_rows(path):
    header, *rows = [json.loads(line) for line in path.read_text().splitlines()]
    return header, rows


def extract(lock_path, stage, output):
    lock = check_lock(lock_path)
    parent = context.check_lock(Path(lock["parent_lock"]))
    plan = json.loads(Path(lock["fit_plan"]).read_text())
    models, loaded, heads = {}, {}, {}
    if stage == "fit":
        from app.detectors.image.adapters.loaders import load_yunet
        from app.detectors.video.adapters.loaders import load_dinov2_small

        weights = previous.setup()
        models = {
            "yunet.2023mar": load_yunet("yunet.2023mar", "cpu", "float32"),
            "dinov2.small": load_dinov2_small("dinov2.small", "cpu", "float32"),
        }
        for arm, entry in lock["heads"].items():
            heads[arm] = json.loads(Path(entry["path"]).read_text())
            if heads[arm]["weights"] != weights:
                raise ValueError("Pretrained declarations changed")
            loaded[arm] = previous.loaded_candidate(Path(entry["path"]))
        manifest, cases = fit_cases(lock)
        expected = [(case, None, manifest.parent) for case in cases]
        input_sha = digest(manifest)
    else:
        receipt = Path(lock[stage + "_receipt"])
        header, rows = load_rows(receipt)
        if header["lock_sha256"] != digest(Path(lock["parent_lock"])) or header["stage"] != stage:
            raise ValueError("Wrong immutable prior response panel")
        if stage == "calibration":
            manifest = Path(parent["calibration_manifest"])
            cases = json.loads(manifest.read_text())["items"]
            roots = [manifest.parent] * len(cases)
        else:
            cases, roots = [], []
            for panel in parent["development"]:
                manifest = Path(panel["manifest"])
                local = json.loads(manifest.read_text())["items"]
                cases += local
                roots += [manifest.parent] * len(local)
        if [r["id"] for r in rows] != [r["id"] for r in cases]:
            raise ValueError("All exact prior cases required")
        expected = list(zip(cases, rows, roots, strict=True))
        input_sha = digest(receipt)
    face_detector = detector()
    header = {
        "schema": "chipmunk.mouth-temporal-features/11",
        "lock_sha256": digest(lock_path),
        "stage": stage,
        "input_sha256": input_sha,
        "planned": len(expected),
    }
    complete = []
    if output.exists():
        old, complete = load_rows(output)
        if old != header or [r["id"] for r in complete] != [r[0]["id"] for r in expected[: len(complete)]]:
            raise ValueError("Cannot resume changed temporal inputs")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    with output.open("a") as stream:
        for case, old, root in expected[len(complete) :]:
            started = time.monotonic()
            row = {**case, "mask_root": str(root), "status": "failed"}
            try:
                path = root / case["path"]
                if digest(path) != case["sha256"]:
                    raise ValueError("Prepared video changed")
                if old is None:
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
                    row["heads"] = {}
                    for arm in ("v8", "v9"):
                        models["probe.dino"] = loaded[arm]
                        row["heads"][arm] = context.score_head(path, case, reference, models, heads[arm])
                else:
                    if old.get("failure_reason") or any(
                        old[k] != case[k] for k in ("id", "group", "label", "condition", "sha256")
                    ):
                        raise ValueError("Prior case metadata or assessment changed")
                    row["heads"] = old["heads"]
                    row["current_reference"] = old["current_reference"]
                    row["family"] = old["family"]
                if any(h["status"] != "ok" or not h["baseline_replay_parity"] for h in row["heads"].values()):
                    raise ValueError("Incomplete frozen-head assessment")
                measured, trace, table = measurements(path, row["heads"], Path(plan["task"]), face_detector)
                windows = method.window_features(row["heads"], measured)
                if not windows:
                    raise ValueError("No eligible temporal windows")
                for window in windows:
                    window["region"] = comparison.region(window, row["heads"]["v8"]["model_inputs"], table)
                row.update(
                    status="ok",
                    measurements=measured,
                    geometry_trace=trace,
                    windows=windows,
                    native_input_replay=True,
                )
            except Exception as error:
                row["reason"] = f"{type(error).__name__}: {error}"
            row["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "windows": len(row.get("windows", [])),
                        "reason": row.get("reason"),
                        "seconds": row["runtime_s"],
                    }
                ),
                flush=True,
            )


def training_windows(rows):
    result = []
    for row in rows:
        masks = {}
        if row["label"]:
            for m in row["masks"]:
                path = Path(row["mask_root"]) / m["path"]
                if digest(path) != m["sha256"]:
                    raise ValueError("Fitting truth mask changed")
                masks[m["frame_index"]] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
        for window in row["windows"]:
            roi = window["region"]
            mask = masks.get(roi["frame_index"])
            changed = False
            if mask is not None:
                predicted = np.zeros(mask.shape, np.uint8)
                cv2.fillConvexPoly(predicted, np.rint(roi["polygon"]).astype(np.int32), 1)
                changed = bool(np.any(mask & (predicted > 0)))
            label = int(
                bool(
                    row["label"]
                    and changed
                    and baseline.temporal_iou([window["start_s"], window["end_s"]], row["edit_time_range_s"])
                    >= 0.2
                )
            )
            result.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "role": row["role"],
                    "condition": row["condition"],
                    "label": label,
                    "features": window["features"],
                }
            )
    return result


def checked_rows(lock_path, receipt, stage, count):
    header, rows = load_rows(receipt)
    if (
        header["lock_sha256"] != digest(lock_path)
        or header["stage"] != stage
        or len(rows) != count
        or any(r["status"] != "ok" or not r.get("native_input_replay") for r in rows)
    ):
        raise ValueError("All complete frozen feature cases required: " + stage)
    return rows


def train(lock_path, fit_receipt, calibration_receipt, output):
    lock = check_lock(lock_path)
    fit = checked_rows(lock_path, fit_receipt, "fit", 160)
    calibration = checked_rows(lock_path, calibration_receipt, "calibration", 24)
    if {r["group"] for r in fit} != set(lock["fit_groups"]) or set(lock["fit_groups"]) & {
        r["group"] for r in calibration
    }:
        raise ValueError("Wrong fitting roles or calibration leakage")
    # The inherited point validator enforces all four actors/six ordinary conditions.
    from scripts.mouth_context_v10 import point

    point([{**r, "status": "ready"} for r in calibration], "v8")
    audit_path = Path(lock["fit_plan"]).parent / "training-source-audit.json"
    audit = json.loads(audit_path.read_text())
    manifest, _ = fit_cases(lock)
    if audit["passed"] != 160 or audit["manifest_sha256"] != digest(manifest):
        raise ValueError("Every fitting construction must pass the source audit before learning")
    head = method.train(training_windows(fit))
    control_max = max(float(v) for r in calibration for v in method.margins(r["windows"], head))
    head.update(
        lock_sha256=digest(lock_path),
        fit_receipt_sha256=digest(fit_receipt),
        calibration_receipt_sha256=digest(calibration_receipt),
        calibration_receipt=str(calibration_receipt.resolve()),
        fit_receipt=str(fit_receipt.resolve()),
        point_groups=sorted({r["group"] for r in calibration}),
        review_point=math.nextafter(max(0.0, control_max), math.inf),
        ordinary_max_margin=control_max,
        source_audit_sha256=digest(audit_path),
    )
    previous.write(output, head)
    print(
        json.dumps(
            {k: head[k] for k in ("fit_windows", "iterations", "review_point", "ordinary_max_margin")}
        ),
        flush=True,
    )


def review(lock_path, head_path, receipt, output):
    lock = check_lock(lock_path)
    parent = context.check_lock(Path(lock["parent_lock"]))
    head = json.loads(head_path.read_text())
    if head["lock_sha256"] != digest(lock_path):
        raise ValueError("Wrong temporal head")
    for key in ("fit_receipt", "calibration_receipt"):
        if digest(Path(head[key])) != head[key + "_sha256"]:
            raise ValueError("Fitting/point receipt changed")
    calibration = checked_rows(lock_path, Path(head["calibration_receipt"]), "calibration", 24)
    expected_point = math.nextafter(
        max(0.0, max(float(v) for r in calibration for v in method.margins(r["windows"], head))), math.inf
    )
    if expected_point != head["review_point"]:
        raise ValueError("Review point was not selected by frozen ordinary rule")
    rows = checked_rows(lock_path, receipt, "development", 88)
    _, original = load_rows(Path(parent["prior_scores"]["v8"]["path"]))
    if [r["id"] for r in rows] != [r["id"] for r in original] or {r["group"] for r in rows} & (
        set(head["fit_groups"]) | set(head["point_groups"])
    ):
        raise ValueError("Development identities changed or roles overlap")
    old_summary = json.loads(Path(parent["prior_summary"]).read_text())
    absolute_point = json.loads(Path(lock["heads"]["v8"]["path"]).read_text())["review_point"]
    combined, temporal, signals, scored = [], [], [], []
    for row, old in zip(rows, original, strict=True):
        windows = [
            {**w, "response": float(v)}
            for w, v in zip(row["windows"], method.margins(row["windows"], head), strict=True)
        ]
        windows.sort(key=lambda w: (-w["response"], w["start_s"], w["track_id"]))
        new = {**row, "probe_windows": windows[:12], "roi_bindings": {"probe": windows[0]["region"]}}
        root = Path(row["mask_root"])
        temporal += baseline.evaluate([new], "probe", head["review_point"], root)
        signal = method.selected_signal(
            old["probe_windows"], absolute_point, new["probe_windows"], head["review_point"]
        )
        combined += baseline.evaluate(
            [old] if signal == "absolute" else [new],
            "probe",
            absolute_point if signal == "absolute" else head["review_point"],
            root,
        )
        signals.append({"id": row["id"], "signal": signal})
        scored.append({"id": row["id"], "windows": windows})
    pseudo = [
        {
            "status": r["status"],
            "baseline_replay_parity": all(h["baseline_replay_parity"] for h in r["heads"].values()),
            "prior_native_input_parity": r["native_input_replay"]
            and all(h["prior_native_and_margin_parity"] for h in r["heads"].values()),
        }
        for r in rows
    ]
    gates = paired.gates(old_summary["cases"]["candidate"], combined, [r["family"] for r in rows], pseudo)
    body = {
        "schema": "chipmunk.mouth-temporal-summary/11",
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "receipt_sha256": digest(receipt),
        "review_point": head["review_point"],
        "baseline": old_summary["candidate"],
        "temporal_only": baseline.summary(temporal),
        "combined": baseline.summary(combined),
        "gates": gates,
        "passes": all(gates.values()),
        "families": {
            f: baseline.summary([r for r, s in zip(combined, rows, strict=True) if s["family"] == f])
            for f in ("liveportrait", "musetalk")
        },
        "cases": {"combined": combined, "temporal": temporal},
        "selected_signals": signals,
        "scored_windows": scored,
        "deployment_eligible": False,
    }
    previous.write(output, body)
    print(
        json.dumps(
            {k: v for k, v in body.items() if k not in ("cases", "selected_signals", "scored_windows")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "fit", "calibration", "development", "train", "review"])
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--fit", type=Path)
    p.add_argument("--calibration", type=Path)
    p.add_argument("--head", type=Path)
    p.add_argument("--receipt", type=Path)
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.input, a.output)
    elif a.stage in ("fit", "calibration", "development"):
        extract(a.input, a.stage, a.output)
    elif a.stage == "train" and a.fit and a.calibration:
        train(a.input, a.fit, a.calibration, a.output)
    elif a.stage == "review" and a.head and a.receipt:
        review(a.input, a.head, a.receipt, a.output)
    else:
        p.error("train needs --fit/--calibration; review needs --head/--receipt")
