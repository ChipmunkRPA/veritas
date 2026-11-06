"""Source-matched learned forgery-response comparison; never a serving detector."""

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
from scripts import compare_mouth_sequence_v4 as previous
from scripts.liveportrait_mouth_benchmark import digest

METHOD = "learned-face-artifact-contrast/5.1"
CONTROLS = {"original", "local-light", "local-resample", "constant-audio-delay"}


def artifact_windows(centers, tracks, logits, table):
    windows = []
    for track, samples in tracks.items():
        indices = {r.frame_index for r in samples}
        blocks = [
            (i, float(np.mean([logits[k] for k in (i - 1, i, i + 1)])))
            for i in sorted(centers)
            if all(k in indices and k in logits for k in (i - 1, i, i + 1))
            and table.seconds(i + 1) - table.seconds(i - 1) <= 0.13
        ]
        if len(blocks) < 7:
            continue
        times = np.array([table.seconds(i) for i, _ in blocks])
        values = np.array([z for _, z in blocks])
        for start in range(2, len(blocks) - 4):
            for end in range(start + 3, len(blocks) - 1):
                span = times[end - 1] - times[start]
                if span > 2.6:
                    break
                if span < 0.52 or np.max(np.diff(times[start - 2 : end + 2])) > 0.55:
                    continue
                left, middle, right = [
                    float(p.mean())
                    for p in (values[start - 2 : start], values[start:end], values[end : end + 2])
                ]
                windows.append(
                    {
                        "track_id": track,
                        "start_s": float(times[start]),
                        "end_s": float(times[end - 1]),
                        "response": min(middle - left, middle - right) - 0.35 * abs(left - right),
                        "middle_logit": middle,
                        "before_logit": left,
                        "after_logit": right,
                        "centers": [i for i, _ in blocks[start - 2 : end + 2]],
                    }
                )
    return sorted(windows, key=lambda w: (-w["response"], w["start_s"], w["track_id"]))[:12]


def choose_point(rows):
    controls = [r for r in rows if not r["label"] and r.get("artifact_windows")]
    if {r["condition"] for r in controls} != CONTROLS:
        return None
    return float(np.nextafter(max(0.0, max(r["artifact_windows"][0]["response"] for r in controls)), np.inf))


def measured_region(window, inputs, table):
    during = window["centers"][2:-2]
    i = during[(len(during) - 1) // 2]
    binding = next(r for r in inputs if r["frame_index"] == i)
    matrix = np.array([[float.fromhex(x) for x in row] for row in binding["alignment_matrix_hex"]])
    inverse = cv2.invertAffineTransform(matrix)
    side = binding["alignment_side"]
    box = np.array([[0, 0], [side, 0], [side, side], [0, side]], np.float64)
    polygon = np.c_[box, np.ones(4)] @ inverse.T
    return {
        "frame_index": i,
        "pts": table.pts[i],
        "time_s": table.seconds(i),
        "polygon": polygon.tolist(),
        "width": table.width,
        "height": table.height,
        "interpretation": "Measured full-face crop, not causal mouth or pixel attribution",
    }


def associate_native_crops(crops, native):
    """Join at the adapter's insertion boundary, before patch_features sorts keys."""
    if len(crops) != len(native):
        raise ValueError("Native aligned crops do not correspond to baseline feature inputs")
    by_index = {}
    for (i, resized), (pixels, binding) in zip(crops.items(), native, strict=True):
        if not np.array_equal(cv2.resize(pixels, (112, 112), interpolation=cv2.INTER_AREA), resized):
            raise ValueError("Captured native pixels differ from the production feature crop")
        by_index[i] = (pixels, binding)
    return by_index


def score_case(path, case, reference, models, device):
    from app.detectors.image import gend
    from app.detectors.video.adapters import face_sequence as adapter

    captured, native = {}, []
    real_warp, real_windows, real_features = adapter.warp, adapter.sequence_windows, adapter.patch_features

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

    def capture_windows(*args):
        captured["args"] = args
        return real_windows(*args)

    def capture_features(crops, *args, **kwargs):
        captured["native_by_index"] = associate_native_crops(crops, native)
        return real_features(crops, *args, **kwargs)

    with (
        patch.object(adapter, "warp", side_effect=capture_crop),
        patch.object(adapter, "sequence_windows", side_effect=capture_windows),
        patch.object(adapter, "patch_features", side_effect=capture_features),
    ):
        replay = previous.score_case(path, case, models)
    for key in ("status", "base_windows", "sampling_centers", "time_base", "baseline_capture_parity"):
        if replay[key] != reference[key]:
            raise ValueError("Production replay differs from immutable v4 baseline: " + key)
    if "args" not in captured:
        return {"status": "not_applicable", "artifact_windows": [], "baseline_replay_parity": True}
    centers, tracks, features, table = captured["args"]
    native_by_index = captured["native_by_index"]
    if set(native_by_index) != set(features):
        raise ValueError("Native aligned crops do not correspond to baseline feature inputs")
    indices = list(features)
    if indices != sorted(indices):
        raise ValueError("Unexpected production crop ordering")
    logits, bindings = {}, []
    for start in range(0, len(indices), 4):
        batch = gend.to_batch([native_by_index[i][0] for i in indices[start : start + 4]])
        values = gend.logits(models["gend.clip_l14"].obj, batch, device)
        for offset, z in enumerate(values):
            k = start + offset
            i = indices[k]
            _, binding = native_by_index[i]
            binding.update(
                frame_index=i,
                pts=table.pts[i],
                time_base=table.time_base,
                time_s=table.seconds(i),
                model_input_float32_sha256=hashlib.sha256(batch[offset].numpy().tobytes()).hexdigest(),
                fake_minus_real_logit=z,
            )
            bindings.append(binding)
            logits[i] = z
    windows = artifact_windows(centers, tracks, logits, table)
    return {
        "status": "ok" if windows else "not_applicable",
        "artifact_windows": windows,
        "model_inputs": bindings,
        "baseline_replay_parity": True,
        "roi_bindings": {
            **replay["roi_bindings"],
            **({"artifact": measured_region(windows[0], bindings, table)} if windows else {}),
        },
    }


def frozen_implementation():
    root = Path(__file__).resolve().parents[2]
    paths = [
        "backend/scripts/compare_mouth_artifact_v5.py",
        "docs/mouth-artifact-v5-development.md",
        "backend/scripts/compare_mouth_sequence_v4.py",
        "backend/app/detectors/image/gend.py",
        "backend/app/detectors/image/faces.py",
        "backend/app/detectors/video/face_sequence.py",
        "backend/app/detectors/video/adapters/face_sequence.py",
        "backend/app/detectors/video/frame_store.py",
        "backend/app/detectors/video/decode.py",
        "backend/app/detectors/video/adapters/tracks.py",
        "backend/app/detectors/video/tracking.py",
        "backend/app/models.yaml",
    ]
    return {p: digest(root / p) for p in paths}


def freeze(manifest, baseline, output):
    header, *rows = [json.loads(line) for line in baseline.read_text().splitlines()]
    m = json.loads(manifest.read_text())
    if header["manifest_sha256"] != digest(manifest) or [r["id"] for r in rows] != [
        r["id"] for r in m["items"]
    ]:
        raise ValueError("Baseline must include every declared input in order")
    body = {
        "schema": "chipmunk.mouth-artifact-development-lock/5",
        "method": METHOD,
        "manifest_sha256": digest(manifest),
        "baseline_receipt_sha256": digest(baseline),
        "implementation_sha256": frozen_implementation(),
        "role": "Consumed development, not fresh confirmation",
        "deployment_eligible": False,
    }
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"lock_sha256": digest(output), "planned": len(rows)}))


def score(manifest, baseline, lock_path, output):
    import torch

    from app.detectors.image.adapters.loaders import load_gend, load_yunet
    from app.detectors.video.adapters.loaders import load_dinov2_small
    from app.engine.fetchers import ensure_verified
    from app.engine.registry import get_registry

    lock = json.loads(lock_path.read_text())
    if (
        lock["implementation_sha256"] != frozen_implementation()
        or lock["manifest_sha256"] != digest(manifest)
        or lock["baseline_receipt_sha256"] != digest(baseline)
    ):
        raise ValueError("Frozen implementation or inputs changed")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    if not torch.backends.mps.is_available():
        raise ValueError("This development lock requires MPS; no silent device substitution")
    models, weights = {}, {}
    for mid, loader, device in (
        ("yunet.2023mar", load_yunet, "cpu"),
        ("dinov2.small", load_dinov2_small, "cpu"),
        ("gend.clip_l14", load_gend, "mps"),
    ):
        spec = get_registry().manifest.model(mid)
        if spec.licence.class_ != "commercial":
            raise ValueError("Commercial checkpoint declaration is required")
        weights[mid] = [{"name": p.name, "sha256": digest(p)} for p in ensure_verified(spec, force=True)]
        models[mid] = loader(mid, device, "float32")
    header = {
        "schema": "chipmunk.mouth-artifact-development-scores/5",
        "method": METHOD,
        "lock_sha256": digest(lock_path),
        "manifest_sha256": digest(manifest),
        "baseline_receipt_sha256": digest(baseline),
        "weights": weights,
        "candidate_device": "mps",
        "baseline_device": "cpu",
        "torch_version": torch.__version__,
    }
    _, *references = [json.loads(line) for line in baseline.read_text().splitlines()]
    m = json.loads(manifest.read_text())
    complete = []
    if output.exists():
        old, *complete = [json.loads(line) for line in output.read_text().splitlines()]
        if old != header or [r["id"] for r in complete] != [r["id"] for r in m["items"][: len(complete)]]:
            raise ValueError("Cannot resume different inputs, weights or outcome order")
    else:
        with output.open("x") as stream:
            stream.write(json.dumps(header, allow_nan=False) + "\n")
    with output.open("a") as stream:
        for case, reference in zip(m["items"][len(complete) :], references[len(complete) :], strict=True):
            start = time.monotonic()
            row = {
                **case,
                "status": "not_applicable",
                "artifact_windows": [],
                "base_windows": reference["base_windows"],
            }
            try:
                if case["status"] != "ready":
                    raise ValueError("Prepared source is unavailable")
                path = manifest.parent / case["path"]
                if digest(path) != case["sha256"]:
                    raise ValueError("Source bytes changed")
                row.update(score_case(path, case, reference, models, "mps"))
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            row["runtime_s"] = time.monotonic() - start
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


def review(manifest, receipt, output):
    header, *rows = [json.loads(line) for line in receipt.read_text().splitlines()]
    m = json.loads(manifest.read_text())
    if header["manifest_sha256"] != digest(manifest) or [r["id"] for r in rows] != [
        r["id"] for r in m["items"]
    ]:
        raise ValueError("Every planned input requires one terminal outcome")
    point = choose_point(rows)
    a = previous.evaluate(rows, "base", REVIEW_POINT, manifest.parent)
    b = previous.evaluate(rows, "artifact", point, manifest.parent)
    base, candidate = previous.summary(a), previous.summary(b)
    gains = sum(not x["localized"] and y["localized"] for x, y in zip(a, b, strict=True))
    losses = sum(x["localized"] and not y["localized"] for x, y in zip(a, b, strict=True))
    gates = {
        "additional_ai_localizations": gains - losses >= 3,
        "control_flags": all(
            sum(r["flagged"] for r in b if r["condition"] == c)
            <= sum(r["flagged"] for r in a if r["condition"] == c) + 1
            for c in CONTROLS
        ),
        "retained_localizations": losses <= 1,
        "same_assessed_coverage": all(not x["assessed"] or y["assessed"] for x, y in zip(a, b, strict=True)),
        "spatial_pointing_gain": candidate["mean_ai_spatial_iou"] >= base["mean_ai_spatial_iou"] + 0.01,
        "all_cases_completed": all(r["status"] != "failed" and r.get("baseline_replay_parity") for r in rows),
        "point_defined": point is not None,
    }
    report = {
        "schema": "chipmunk.mouth-artifact-development-summary/5",
        "manifest_sha256": digest(manifest),
        "score_receipt_sha256": digest(receipt),
        "candidate_review_point": point,
        "baseline": base,
        "candidate": candidate,
        "gains": gains,
        "losses": losses,
        "gates": gates,
        "advance_to_confirmation_design": all(gates.values()),
        "deployment_eligible": False,
        "conditions": {
            c: {
                "baseline": previous.summary([r for r in a if r["condition"] == c]),
                "candidate": previous.summary([r for r in b if r["condition"] == c]),
            }
            for c in sorted({r["condition"] for r in rows})
        },
        "cases": {"baseline": a, "candidate": b},
        "limitations": [
            "Consumed development: eight actors, one renderer family, studio footage; no outside reviewer.",
            "Control-selected point is not independent false-alarm validation or AI confidence.",
            "Measured full-face region is not pixel-level attribution or proof of AI mouth editing.",
        ],
    }
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("conditions", "cases", "limitations")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "score", "review"])
    p.add_argument("manifest", type=Path)
    p.add_argument("input", type=Path, help="Baseline receipt for freeze/score; candidate receipt for review")
    p.add_argument("output", type=Path)
    p.add_argument("--lock", type=Path)
    a = p.parse_args()
    if a.stage == "freeze":
        freeze(a.manifest, a.input, a.output)
    elif a.stage == "score":
        if a.lock is None:
            p.error("score requires --lock")
        score(a.manifest, a.input, a.lock, a.output)
    else:
        review(a.manifest, a.input, a.output)
