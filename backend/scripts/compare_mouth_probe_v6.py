"""Source-bound video development comparison for a frozen experimental mouth head."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.detectors.video.face_sequence import REVIEW_POINT
from scripts import compare_mouth_sequence_v4 as baseline
from scripts.compare_mouth_artifact_v5 import associate_native_crops
from scripts.fit_mouth_probe_v6 import check_lock
from scripts.liveportrait_mouth_benchmark import digest
from scripts.mouth_probe_v6 import BATCH, METHOD, features, margin_windows, margins, mouth_chip


def region(window, bindings, table):
    during = window["centers"][2:-2]
    i = during[(len(during) - 1) // 2]
    binding = next(r for r in bindings if r["frame_index"] == i)
    inverse = cv2.invertAffineTransform(
        np.array([[float.fromhex(v) for v in row] for row in binding["alignment_matrix_hex"]])
    )
    x0, y0, x1, y1 = binding["mouth_chip_bounds"]
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    polygon = np.c_[box, np.ones(4)] @ inverse.T
    return {
        "frame_index": i,
        "pts": table.pts[i],
        "time_s": table.seconds(i),
        "polygon": polygon.tolist(),
        "width": table.width,
        "height": table.height,
        "interpretation": "Actual mouth/lower-face measurement chip, not a causal pixel mask",
    }


def score_case(path, case, reference, models, head):
    from app.detectors.video.adapters import face_sequence as adapter

    captured, native = {}, []
    real_warp, real_features, real_windows = adapter.warp, adapter.patch_features, adapter.sequence_windows

    def capture_crop(rgb, matrix, side):
        crop = real_warp(rgb, matrix, side)
        native.append(
            (
                crop,
                {
                    "decoded_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                    "alignment_side": side,
                    "alignment_matrix_hex": [[float(x).hex() for x in row] for row in matrix],
                    "native_crop_rgb_sha256": hashlib.sha256(crop.tobytes()).hexdigest(),
                },
            )
        )
        return crop

    def capture_features(crops, *args, **kwargs):
        captured["native_by_index"] = associate_native_crops(crops, native)
        return real_features(crops, *args, **kwargs)

    def capture_windows(*args):
        captured["args"] = args
        return real_windows(*args)

    with (
        patch.object(adapter, "warp", side_effect=capture_crop),
        patch.object(adapter, "patch_features", side_effect=capture_features),
        patch.object(adapter, "sequence_windows", side_effect=capture_windows),
    ):
        replay = baseline.score_case(path, case, models)
    for key in ["status", "base_windows", "sampling_centers", "time_base", "baseline_capture_parity"]:
        if replay[key] != reference[key]:
            raise ValueError("Immutable production baseline mismatch: " + key)
    if "args" not in captured:
        return {
            "status": "not_applicable",
            "reason": "No eligible tracked baseline input",
            "probe_windows": [],
            "baseline_replay_parity": True,
        }
    centers, tracks, original_features, table = captured["args"]
    by_index = captured["native_by_index"]
    if set(by_index) != set(original_features):
        raise ValueError("Candidate and baseline native inputs differ")
    values, bindings = {}, []
    indices = sorted(by_index)
    for start in range(0, len(indices), BATCH):
        chosen = indices[start : start + BATCH]
        chips, bounds = zip(*(mouth_chip(by_index[i][0]) for i in chosen), strict=True)
        vectors, hashes = features(chips, models["probe.dino"].obj)
        responses = margins(vectors, head)
        for j, i in enumerate(chosen):
            binding = {
                **by_index[i][1],
                "frame_index": i,
                "pts": table.pts[i],
                "time_base": table.time_base,
                "time_s": table.seconds(i),
                "mouth_chip_bounds": bounds[j],
                "mouth_chip_rgb_sha256": hashlib.sha256(chips[j].tobytes()).hexdigest(),
                "model_input_float32_sha256": hashes[j],
                "features_float32_sha256": hashlib.sha256(vectors[j].tobytes()).hexdigest(),
                "margin": float(responses[j]),
            }
            bindings.append(binding)
            values[i] = float(responses[j])
    windows = margin_windows(centers, tracks, values, table)
    return {
        "status": "ok" if windows else "not_applicable",
        "reason": None if windows else "No eligible source-context windows",
        "probe_windows": windows,
        "model_inputs": bindings,
        "baseline_replay_parity": True,
        "roi_bindings": {
            **replay["roi_bindings"],
            **({"probe": region(windows[0], bindings, table)} if windows else {}),
        },
    }


def score(manifest, baseline_receipt, head_path, lock_path, output):
    import torch

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video.adapters.loaders import load_dinov2_small
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    lock = json.loads(lock_path.read_text())
    check_lock(lock)
    if (
        digest(manifest) != lock["baseline_manifest_sha256"]
        or digest(baseline_receipt) != lock["baseline_receipt_sha256"]
    ):
        raise ValueError("Comparison inputs changed after freeze")
    head = json.loads(head_path.read_text())
    if head["lock_sha256"] != digest(lock_path):
        raise ValueError("The fitted head uses a different experiment lock")
    m = json.loads(manifest.read_text())
    _, *references = [json.loads(l) for l in baseline_receipt.read_text().splitlines()]
    if {r["group"] for r in m["items"]} & (set(head["fit_groups"]) | set(head["point_groups"])):
        raise ValueError("Evaluation actors overlap fitting or point selection")
    if [r["id"] for r in m["items"]] != [r["id"] for r in references]:
        raise ValueError("Baseline is missing planned outcomes")
    if not torch.backends.mps.is_available():
        raise ValueError("This frozen comparison requires MPS")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models, weights = {}, {}
    for mid, loader, device, key in [
        ("yunet.2023mar", load_yunet, "cpu", "yunet.2023mar"),
        ("dinov2.small", load_dinov2_small, "cpu", "dinov2.small"),
        ("dinov2.small", load_dinov2_small, "mps", "probe.dino"),
    ]:
        spec = get_registry().manifest.model(mid)
        if spec.licence.class_ != "commercial":
            raise ValueError("Commercial checkpoint declaration required")
        weights[mid] = [{"name": p.name, "sha256": digest(p)} for p in ensure_verified(spec, force=True)]
        models[key] = loader(mid, device, "float32")
    header = {
        "schema": "chipmunk.mouth-trained-scores/6",
        "method": METHOD,
        "manifest_sha256": digest(manifest),
        "baseline_receipt_sha256": digest(baseline_receipt),
        "lock_sha256": digest(lock_path),
        "head_sha256": digest(head_path),
        "review_point": head["review_point"],
        "weights": weights,
        "candidate_device": "mps",
        "baseline_device": "cpu",
        "torch_version": torch.__version__,
    }
    complete = []
    if output.exists():
        old, *complete = [json.loads(l) for l in output.read_text().splitlines()]
        if old != header or [r["id"] for r in complete] != [r["id"] for r in m["items"][: len(complete)]]:
            raise ValueError("Cannot resume changed experiment")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header) + "\n")
    with output.open("a") as stream:
        for case, reference in zip(m["items"][len(complete) :], references[len(complete) :], strict=True):
            started = time.monotonic()
            row = {
                **case,
                "status": "not_applicable",
                "probe_windows": [],
                "base_windows": reference["base_windows"],
            }
            try:
                path = manifest.parent / case["path"]
                if case["status"] != "ready" or digest(path) != case["sha256"]:
                    raise ValueError("Prepared comparison source missing or changed")
                row.update(score_case(path, case, reference, models, head))
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            row["runtime_s"] = time.monotonic() - started
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": row["status"],
                        "seconds": row["runtime_s"],
                        "reason": row.get("reason"),
                    }
                ),
                flush=True,
            )


def review(manifest, receipt, head_path, output):
    header, *rows = [json.loads(l) for l in receipt.read_text().splitlines()]
    head = json.loads(head_path.read_text())
    m = json.loads(manifest.read_text())
    if (
        header["manifest_sha256"] != digest(manifest)
        or header["head_sha256"] != digest(head_path)
        or [r["id"] for r in rows] != [r["id"] for r in m["items"]]
    ):
        raise ValueError("Every planned source requires its exact frozen outcome")
    a = baseline.evaluate(rows, "base", REVIEW_POINT, manifest.parent)
    b = baseline.evaluate(rows, "probe", head["review_point"], manifest.parent)
    current, candidate = baseline.summary(a), baseline.summary(b)
    gains = sum(not x["localized"] and y["localized"] for x, y in zip(a, b, strict=True))
    losses = sum(x["localized"] and not y["localized"] for x, y in zip(a, b, strict=True))
    gates = {
        "additional_ai_localizations": gains - losses >= 3,
        "retained_localizations": losses <= 1,
        "control_flags": all(
            sum(r["flagged"] for r in b if r["condition"] == c)
            <= sum(r["flagged"] for r in a if r["condition"] == c) + 1
            for c in ("original", "local-light", "local-resample", "constant-audio-delay")
        ),
        "same_assessed_coverage": all(not x["assessed"] or y["assessed"] for x, y in zip(a, b, strict=True)),
        "spatial_pointing_gain": candidate["mean_ai_spatial_iou"] >= current["mean_ai_spatial_iou"] + 0.01,
        "all_cases_completed": all(r["status"] != "failed" and r.get("baseline_replay_parity") for r in rows),
    }
    result = {
        "schema": "chipmunk.mouth-trained-summary/6",
        "manifest_sha256": digest(manifest),
        "score_receipt_sha256": digest(receipt),
        "head_sha256": digest(head_path),
        "review_point": head["review_point"],
        "baseline": current,
        "candidate": candidate,
        "gains": gains,
        "losses": losses,
        "gates": gates,
        "advance_to_confirmation_design": all(gates.values()),
        "deployment_eligible": False,
        "conditions": {
            c: {
                "baseline": baseline.summary([r for r in a if r["condition"] == c]),
                "candidate": baseline.summary([r for r in b if r["condition"] == c]),
            }
            for c in sorted({r["condition"] for r in rows})
        },
        "cases": {"baseline": a, "candidate": b},
        "limitations": [
            "16 consumed fitting actors, 4 point-selection actors, 8 consumed development actors; no fresh confirmation.",
            "One editing family and four studio sentence types; pretraining overlap unknown.",
            "Raw mouth-chip margins are not calibrated authenticity confidence or causal pixel attribution.",
            "No independent outside reviewer has participated.",
        ],
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases", "conditions", "limitations")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["score", "review"])
    p.add_argument("manifest", type=Path)
    p.add_argument("input", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--lock", type=Path)
    a = p.parse_args()
    if a.stage == "score":
        if not a.lock:
            p.error("score requires --lock")
        score(a.manifest, a.input, a.head, a.lock, a.output)
    else:
        review(a.manifest, a.input, a.head, a.output)
