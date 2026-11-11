"""Prospective local JPEG localization diagnostic on the reserved v26 role."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, features

from scripts import image_regional_inference_v28 as v28
from scripts import image_regional_readout_v27 as v27
from scripts.prepare_regional_calibration_v29 import ASSETS, runtime_versions

ROOT = v28.ROOT
DATA = ROOT / "backend/data/independent-evaluation"
PLAN = DATA / "image-regional-supervised-v26/plan.json"
PAIRS = DATA / "image-positive-pairs-v21-cache/qualification/pairs.jsonl"
CHAIN = DATA / "image-regional-calibration-v42/training-chain.json"
PANELS = ROOT / "docs/evaluation-panels"
V42 = PANELS / "image-regional-calibration-v42-execution-freeze.json"
V46 = PANELS / "image-native-confirmation-v46-freeze.json"
PLAN_SHA = "f38ec72f525edf0000f772c4bb75c36b031a796b1f0c2bebbe2260bde147a14a"
PAIRS_SHA = "c4ae1c532f4db1f26eb5165754c3591a87fc2c82a6ecb421c9cee9c523028489"
QUALITIES = (75, 90)
KINDS = (*v28.KINDS, "copy_move")
ARMS = ("baseline", "candidate")
METHODS = ("current", "sensitive", "shared")
POLICY_NAME = "p0.90-a0.01000"
POLICY = v27.policies()[POLICY_NAME]
PHOTOS, GROUPS, CHUNK = 59, 48, 25
PHASES = (PHOTOS * len(KINDS) + CHUNK - 1) // CHUNK
PREDICTOR_ASSETS = {k: v for k, v in ASSETS.items() if "/splice/" not in k}
CODE = (
    "backend/scripts/image_jpeg_diagnostic_v189.py",
    "backend/scripts/image_jpeg_diagnostic_runtime_v189.py",
    "backend/scripts/image_regional_training_v26.py",
    "backend/scripts/image_regional_inference_v28.py",
    "backend/scripts/image_regional_readout_v27.py",
    "backend/scripts/image_native_comparison_readout_v45.py",
    "backend/scripts/relayformer_native_v18.py",
    "backend/scripts/relayformer_learning_v23.py",
    "backend/scripts/inspect_relayformer_v18.py",
    "backend/scripts/prepare_relayformer_v18.py",
    "backend/scripts/replay_regional_calibration_v43.py",
    "backend/scripts/audit_fused_production_grid.py",
    "backend/scripts/score_splice_localizers.py",
    "backend/scripts/localization_benchmark.py",
    "backend/scripts/local_probe_limits_v24.py",
    "backend/scripts/data_resource_guard_v21.py",
    "backend/scripts/image_regional_continuation_runtime_v41.py",
    "backend/scripts/prepare_regional_calibration_v29.py",
    "backend/app/detectors/image/fused.py",
    "backend/app/detectors/image/jpeg_bitstream.py",
    "backend/app/detectors/image/localiser.py",
    "docs/evaluation-panels/relayformer-v18-acquisition-v2.json",
    "docs/image-jpeg-diagnostic-v189-protocol.md",
)
read, write, sha, require = v28.read, v28.write, v28.sha, v28.require


def jpeg_view(image, quality):
    require(quality in QUALITIES and image.mode == "RGB", "Fixed JPEG quality and upright RGB required")
    if quality == 75:
        return v28.encoded_view(image, "jpeg75")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    payload = buffer.getvalue()
    with Image.open(io.BytesIO(payload)) as decoded:
        result = decoded.convert("RGB").copy()
    require(result.size == image.size, "Encoding changed source coordinates")
    return payload, result


def changed_pixels(before, after):
    require(before.mode == after.mode == "RGB" and before.size == after.size, "Aligned RGB images required")
    return np.any(np.asarray(before) != np.asarray(after), axis=2)


def ordinary_copy(original, alpha):
    require(alpha.mode == "L" and alpha.size == original.size, "Source-aligned compositing alpha required")
    image = Image.composite(ImageOps.mirror(original), original, alpha)
    changed = changed_pixels(original, image)
    require(not np.any(changed & (np.asarray(alpha) == 0)), "Copy changed pixels outside the author alpha")
    return image


def sources(plan):
    rows = sorted(
        (row for row in plan["rows"] if row["role"] == "diagnostic"),
        key=lambda row: (row["component"], row["source_id"]),
    )
    ids = {row["source_id"] for row in rows}
    components = {row["component"] for row in rows}
    require(
        len(rows) == len(ids) == PHOTOS
        and len(components) == GROUPS
        and all(not row["old_linked"] and row["license_id"] in {4, 5} for row in rows)
        and not any(
            (row["source_id"] in ids or row["component"] in components) and row["role"] != "diagnostic"
            for row in plan["rows"]
        )
        and not any(row["source_id"] in ids for row in plan["schedule"]),
        "Complete untouched, grouped v26 diagnostic role required",
    )
    return rows


def load_inputs():
    require(sha(PLAN) == PLAN_SHA and sha(PAIRS) == PAIRS_SHA, "Frozen source plan or qualification changed")
    cases = sources(read(PLAN))
    index = {row["source_id"]: row for row in map(json.loads, PAIRS.read_text().splitlines())}
    return cases, index


def material(cases, index, position):
    from scripts.image_regional_training_v26 import source_views

    case = cases[position]
    qualified = index[case["source_id"]]
    edited, controls, ai = source_views(case, qualified)
    original = controls["original"]
    spec = qualified["compositing_mask"]
    with Image.open(ROOT / spec["root"] / spec["file"]) as alpha:
        copy = ordinary_copy(original, alpha.convert("L"))
    images = {"ai": edited, **controls, "copy_move": copy}
    masks = {"ai": ai > 0, **{kind: changed_pixels(original, images[kind]) for kind in KINDS[1:]}}
    return images, masks


def digest(value):
    return hashlib.sha256(value).hexdigest()


def view_binding(payload, image):
    return {
        "input_sha256": digest(payload),
        "decoded_rgb_sha256": digest(np.asarray(image).tobytes()),
        "source_size": list(image.size),
    }


def fixed_versions():
    return {
        **runtime_versions(),
        "jpeg": features.version_codec("jpg"),
        "libjpeg_turbo": features.version_feature("libjpeg_turbo"),
    }


def rotary_files():
    from scripts.relayformer_native_v18 import RUNTIME

    paths = sorted((RUNTIME / "rotary_embedding_torch").glob("*.py"))
    require(paths, "Pinned private rotary source unavailable")
    return {str(path.relative_to(RUNTIME)): sha(path) for path in paths}


def prior_chain():
    prior, confirmation = read(V42), read(V46)
    chain = read(CHAIN)
    require(
        sha(CHAIN) == prior["training_chain_sha256"]
        and chain["completed_updates"] == 1278
        and chain["final_checkpoint_sha256"]
        == prior["final_checkpoint_sha256"]
        == confirmation["final_checkpoint_sha256"]
        and confirmation["candidate_policies"] == [POLICY_NAME]
        and all(prior["asset_sha256"][key] == value for key, value in PREDICTOR_ASSETS.items()),
        "Exact v42/v46 fixed model and readout required",
    )
    return chain


def prepare(trial, raw_root):
    trial, raw_root = trial.resolve(), raw_root.resolve()
    require(
        not trial.exists()
        and raw_root.is_dir()
        and not any(raw_root.iterdir())
        and not trial.is_relative_to(raw_root)
        and not raw_root.is_relative_to(trial),
        "New study and a separate, existing empty private raw-map directory required",
    )
    cases, index = load_inputs()
    chain = prior_chain()
    require(
        all(sha(ROOT / path) == expected for path, expected in PREDICTOR_ASSETS.items())
        and sha(Path(chain["final_checkpoint"])) == chain["final_checkpoint_sha256"],
        "Previously pinned model bytes changed",
    )
    versions = fixed_versions()
    require(runtime_versions() == read(V42)["runtime_versions"], "Historical v42 numerical runtime required")
    trial.mkdir()
    (trial / "masks").mkdir()
    entries = []
    for number, case in enumerate(cases):
        images, masks = material(cases, index, number)
        target = trial / "masks" / f"{number:03d}.npz"
        with target.open("xb") as handle:
            np.savez_compressed(handle, **masks)
        entries.append(
            {
                "source_id": case["source_id"],
                "component": case["component"],
                "masks_sha256": sha(target),
                "changed_pixels": {kind: int(mask.sum()) for kind, mask in masks.items()},
                "inputs": {
                    str(quality): {kind: view_binding(*jpeg_view(images[kind], quality)) for kind in KINDS}
                    for quality in QUALITIES
                },
            }
        )
    write(trial / "inputs.json", {"raw_root": str(raw_root), "sources": entries})
    freeze = trial / "freeze.json"
    write(
        freeze,
        {
            "schema": "chipmunk.image-jpeg-diagnostic-freeze/189",
            "source_role": "diagnostic",
            "photos": PHOTOS,
            "groups": GROUPS,
            "qualities": list(QUALITIES),
            "kinds": list(KINDS),
            "arms": list(ARMS),
            "methods": list(METHODS),
            "sensitive_policy": POLICY_NAME,
            "useful_iou": 0.3,
            "useful_recall": 0.5,
            "bootstrap_draws": v27.BOOTSTRAPS,
            "bootstrap_seed": v27.SEED,
            "device": {"baseline": "mps", "candidate": "cpu"},
            "dtype": "float32",
            "cpu_threads": 2,
            "views_per_arm_quality": PHOTOS * len(KINDS),
            "views_per_batch": CHUNK,
            "batches": PHASES,
            "unique_model_calls": PHOTOS * len(KINDS) * len(QUALITIES) * len(ARMS),
            "exact_first_input_repeats": len(QUALITIES) * len(ARMS),
            "source_plan_sha256": sha(PLAN),
            "qualified_pair_index_sha256": sha(PAIRS),
            "private_input_bindings_sha256": sha(trial / "inputs.json"),
            "v42_freeze_sha256": sha(V42),
            "v46_freeze_sha256": sha(V46),
            "training_chain_sha256": sha(CHAIN),
            "candidate_checkpoint_sha256": chain["final_checkpoint_sha256"],
            "asset_sha256": PREDICTOR_ASSETS,
            "code_sha256": {name: sha(ROOT / name) for name in CODE},
            "versions": versions,
            "rotary_source_sha256": rotary_files(),
            "neural_calls_during_preparation": 0,
            "serving_admitted": False,
        },
    )
    print(sha(freeze), flush=True)


def lease(trial, expected, profile=None):
    trial = trial.resolve()
    require(sha(trial / "freeze.json") == expected, "Exact prospective freeze digest required")
    freeze = read(trial / "freeze.json")
    require(
        freeze["schema"] == "chipmunk.image-jpeg-diagnostic-freeze/189"
        and freeze["source_plan_sha256"] == PLAN_SHA
        and freeze["qualified_pair_index_sha256"] == PAIRS_SHA
        and freeze["v42_freeze_sha256"] == sha(V42)
        and freeze["v46_freeze_sha256"] == sha(V46)
        and freeze["training_chain_sha256"] == sha(CHAIN)
        and freeze["versions"] == fixed_versions()
        and freeze["rotary_source_sha256"] == rotary_files()
        and freeze["asset_sha256"] == PREDICTOR_ASSETS
        and freeze["private_input_bindings_sha256"] == sha(trial / "inputs.json")
        and freeze["code_sha256"] == {name: sha(ROOT / name) for name in CODE},
        "Frozen runtime, assets, methods or inputs changed",
    )
    if profile is not None:
        from scripts.image_jpeg_diagnostic_runtime_v189 import assert_profile

        assert_profile(profile, "prepare", ["prepare", str(trial), read(trial / "inputs.json")["raw_root"]])
    inputs = read(trial / "inputs.json")
    cases, index = load_inputs()
    require(
        [(r["source_id"], r["component"]) for r in cases]
        == [(r["source_id"], r["component"]) for r in inputs["sources"]],
        "Prepared source identities or order changed",
    )
    return inputs, cases, index


def tasks(phase):
    require(type(phase) is int and 0 <= phase < PHASES, "One planned diagnostic batch required")
    planned = [(n, kind) for n in range(PHOTOS) for kind in KINDS]
    return planned[phase * CHUNK : (phase + 1) * CHUNK]


def batch_views(trial, inputs, cases, index, quality, phase):
    previous = None
    for n, kind in tasks(phase):
        source = inputs["sources"][n]
        if previous != n:
            images, masks = material(cases, index, n)
            path = trial / "masks" / f"{n:03d}.npz"
            require(sha(path) == source["masks_sha256"], "Retained changed-pixel masks changed")
            with np.load(path, allow_pickle=False) as retained:
                require(
                    retained.files == list(KINDS)
                    and all(np.array_equal(retained[k], masks[k]) for k in KINDS),
                    "Retained masks differ from source-relative pixel changes",
                )
            previous = n
        payload, image = jpeg_view(images[kind], quality)
        binding = view_binding(payload, image)
        require(binding == source["inputs"][str(quality)][kind], "JPEG input or decoded pixels changed")
        yield (n, kind), payload, image, masks[kind], binding


def score(trial, expected, arm, quality, phase):
    inputs, cases, index = lease(trial, expected, prep_profile(trial))
    chain = prior_chain()
    needed = (name for name in PREDICTOR_ASSETS if ("/fused/" in name) == (arm == "baseline"))
    require(
        all(sha(ROOT / name) == PREDICTOR_ASSETS[name] for name in needed)
        and (arm == "baseline" or sha(Path(chain["final_checkpoint"])) == chain["final_checkpoint_sha256"]),
        "Fixed model bytes changed",
    )
    name = f"{arm}-q{quality}-{phase:02d}"
    out, raw_dir = trial / name, Path(inputs["raw_root"]) / name
    out.mkdir()
    raw_dir.mkdir()
    # The historical selector's JPEG label selects the same model path for both byte encodings.
    predict = v28.load_predictor(arm, "jpeg75", chain)
    journal = out / "rows.jsonl"
    with journal.open("x") as handle:
        for (n, kind), _payload, image, _truth, binding in batch_views(
            trial, inputs, cases, index, quality, phase
        ):
            try:
                raw, details = predict(image)
                wanted = (512, 512) if arm == "baseline" else (image.height, image.width)
                require(
                    raw.shape == wanted
                    and raw.dtype == np.float32
                    and np.isfinite(raw).all()
                    and raw.min() >= 0
                    and raw.max() <= 1,
                    "Finite source-bound raw response required",
                )
                if phase == n == 0 and kind == KINDS[0]:
                    repeat, repeated_details = predict(image)
                    require(
                        np.array_equal(raw, repeat) and details == repeated_details,
                        "First-input repeat differed",
                    )
                target = raw_dir / f"{n:03d}-{kind}.npy"
                with target.open("xb") as stream:
                    np.save(stream, raw, allow_pickle=False)
                require(np.array_equal(raw, np.load(target, allow_pickle=False)), "Raw map readback differs")
                row = {
                    "status": "ok",
                    "n": n,
                    "kind": kind,
                    "binding": binding,
                    "raw_map_file": str(target),
                    "raw_map_sha256": sha(target),
                    "raw_map_shape": list(raw.shape),
                    "details": details,
                }
            except Exception as error:
                handle.write(
                    json.dumps({"status": "failed", "n": n, "kind": kind, "error_type": type(error).__name__})
                    + "\n"
                )
                raise
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
    write(
        out / "completed.json",
        {
            "freeze_sha256": expected,
            "arm": arm,
            "quality": quality,
            "phase": phase,
            "unique_forwards": len(tasks(phase)),
            "repeat_forwards": int(phase == 0),
            "rows_sha256": sha(journal),
        },
    )


def prep_profile(trial):
    return trial.parent / f"{trial.name}-prepare-profile"


def phase_rows(trial, name, command, args, expected):
    from scripts.image_jpeg_diagnostic_runtime_v189 import assert_profile

    part = trial / name
    assert_profile(trial / f"{name}-profile", command, args)
    finished = read(part / "completed.json")
    journal = part / "rows.jsonl"
    require(
        finished["freeze_sha256"] == expected and sha(journal) == finished["rows_sha256"],
        "Completed phase or row journal changed",
    )
    return finished, [json.loads(line) for line in journal.read_text().splitlines()]


def producer_rows(trial, expected, arm, quality, phase):
    args = ["score", str(trial), expected, arm, str(quality), str(phase)]
    done, rows = phase_rows(trial, f"{arm}-q{quality}-{phase:02d}", "score", args, expected)
    planned = tasks(phase)
    require(
        done["arm"] == arm
        and done["quality"] == quality
        and done["phase"] == phase
        and done["unique_forwards"] == len(planned)
        and done["repeat_forwards"] == int(phase == 0)
        and [(row["n"], row["kind"]) for row in rows] == planned
        and all(row["status"] == "ok" for row in rows),
        "Incomplete or mismatched producer phase",
    )
    return rows


def measure_maps(current, adapted, image_score, payload, pixels, kind, footprint):
    from app.detectors.image.fused import candidate_eligibility
    from scripts.image_native_comparison_readout_v45 import source_mask
    from scripts.localization_benchmark import mask_metrics
    from scripts.score_splice_localizers import score_map

    require(
        kind in KINDS and footprint.shape == pixels.shape[:2] and footprint.dtype == bool,
        "Complete source-coordinate truth required",
    )
    ai_truth = footprint if kind == "ai" else np.zeros_like(footprint)
    require(bool(ai_truth.any()) == (kind == "ai"), "Only AI examples carry positive AI pixels")
    old_gate = candidate_eligibility(image_score)["eligible"]
    new_gate = v27.image_gate(adapted, POLICY)
    old_mask = source_mask(current, old_gate, 0.5, payload, pixels, ai_truth)
    new_mask = source_mask(adapted, new_gate, POLICY["pixel"], payload, pixels, ai_truth)
    for mask, raw, gate, threshold in (
        (old_mask, current, old_gate, 0.5),
        (new_mask, adapted, new_gate, POLICY["pixel"]),
    ):
        require(
            mask_metrics(mask, ai_truth)
            == score_map(raw, gate, threshold, payload, pixels, ai_truth)["metrics"],
            "Selected source pixels differ from production scoring",
        )
    selected = (old_mask, new_mask, old_mask & new_mask)
    # The observed footprint is a retouch reference only. It never changes the zero AI truth.
    return {
        name: {
            "ai": mask_metrics(mask, ai_truth),
            "ordinary": mask_metrics(mask, footprint) if kind in KINDS[2:] and footprint.any() else None,
            "mask_sha256": digest(mask.tobytes()),
        }
        for name, mask in zip(METHODS, selected, strict=True)
    }


def readout(trial, expected, quality, phase):
    import cv2

    from app.detectors.image.fused import candidate_eligibility
    from scripts.replay_regional_calibration_v43 import load_map

    cv2.setNumThreads(1)
    inputs, cases, index = lease(trial, expected, prep_profile(trial))
    current, adapted = (producer_rows(trial, expected, arm, quality, phase) for arm in ARMS)
    raw_root = Path(inputs["raw_root"])
    out = trial / f"readout-q{quality}-{phase:02d}"
    out.mkdir()
    journal = out / "rows.jsonl"
    with journal.open("x") as handle:
        for (key, payload, image, footprint, binding), old, new in zip(
            batch_views(trial, inputs, cases, index, quality, phase),
            current,
            adapted,
            strict=True,
        ):
            n, kind = key
            old_score = old["details"]["image_score"]
            require(
                old["binding"] == binding == new["binding"]
                and old["details"]["image_gate"] is candidate_eligibility(old_score)["eligible"]
                and old["details"]["pixel_threshold"] == 0.5
                and new["details"]["geometry"]["source_rgb_sha256"] == binding["decoded_rgb_sha256"]
                and new["details"]["geometry"]["source_size"] == binding["source_size"],
                "Producer input geometry or production gate differs",
            )
            metrics = measure_maps(
                load_map(old, raw_root / f"baseline-q{quality}-{phase:02d}", (512, 512)),
                load_map(new, raw_root / f"candidate-q{quality}-{phase:02d}", (image.height, image.width)),
                old_score,
                payload,
                np.asarray(image),
                kind,
                footprint,
            )
            row = {
                "n": n,
                "kind": kind,
                "component": cases[n]["component"],
                "quality": quality,
                "changed_pixels": int(footprint.sum()),
                "methods": metrics,
            }
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    write(
        out / "completed.json",
        {
            "freeze_sha256": expected,
            "phase": phase,
            "quality": quality,
            "rows_sha256": sha(journal),
            "views": len(tasks(phase)),
            "neural_calls": 0,
        },
    )


def summarize(rows, components, quality):
    planned = [(n, kind) for n in range(len(components)) for kind in KINDS]
    require(
        [(r["n"], r["kind"]) for r in rows] == planned
        and all(
            r["quality"] == quality
            and r["component"] == components[r["n"]]
            and set(r["methods"]) == set(METHODS)
            for r in rows
        ),
        "Exact full paired support required",
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["component"]].append(row)
    groups = sorted(grouped)
    draws = np.random.default_rng(v27.SEED).integers(0, len(groups), size=(v27.BOOTSTRAPS, len(groups)))

    def paired(values, reference):
        delta = np.asarray(values) - np.asarray(reference)
        return {
            "difference": float(delta.mean()),
            "pointwise_95pct_interval": np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
        }

    def useful(measure):
        return measure["iou"] >= 0.30 and measure["recall"] >= 0.50

    def flagged(measure):
        return measure["predicted_positive_pixels"] > 0

    def per_group(method):
        values = defaultdict(list)
        for group in groups:
            by_kind = defaultdict(list)
            for row in grouped[group]:
                by_kind[row["kind"]].append(row["methods"][method]["ai"])
            values["useful"].append(np.mean([useful(m) for m in by_kind["ai"]]))
            values["edited_iou"].append(np.mean([m["iou"] for m in by_kind["ai"]]))
            for kind in KINDS[1:]:
                values[f"{kind}_false_group"].append(float(any(flagged(m) for m in by_kind[kind])))
                values[f"{kind}_highlighted_area"].append(
                    np.mean([m["flagged_area_fraction"] for m in by_kind[kind]])
                )
        return values

    reference = per_group("current")
    methods = {}
    for method in METHODS:
        values = per_group(method)
        positives = [row["methods"][method]["ai"] for row in rows if row["kind"] == "ai"]
        original = [row["methods"]["current"]["ai"] for row in rows if row["kind"] == "ai"]
        controls = {}
        for kind in KINDS[1:]:
            views = [row["methods"][method] for row in rows if row["kind"] == kind]
            ordinary = [view["ordinary"] for view in views if view["ordinary"] is not None]
            controls[kind] = {
                "photos": len(views),
                "flagged_photos": sum(flagged(view["ai"]) for view in views),
                "flagged_groups": int(sum(values[f"{kind}_false_group"])),
                "group_mean_highlighted_area_fraction": float(np.mean(values[f"{kind}_highlighted_area"])),
                "ordinary_footprints_with_changed_pixels": len(ordinary),
                "ordinary_footprints_overlapped": sum(m["intersection"] > 0 for m in ordinary),
                "mean_iou_with_ordinary_edit_if_changed": float(np.mean([m["iou"] for m in ordinary]))
                if ordinary
                else None,
                "mean_ordinary_edit_recall_if_changed": float(np.mean([m["recall"] for m in ordinary]))
                if ordinary
                else None,
            }
        methods[method] = {
            "edited_photos": len(positives),
            "edited_photos_flagged": sum(flagged(m) for m in positives),
            "edited_photos_with_any_overlap": sum(m["intersection"] > 0 for m in positives),
            "edited_photos_usefully_localized": sum(useful(m) for m in positives),
            "new_useful_locations": sum(
                useful(b) and not useful(a) for a, b in zip(original, positives, strict=True)
            ),
            "lost_useful_locations": sum(
                useful(a) and not useful(b) for a, b in zip(original, positives, strict=True)
            ),
            "edited_photos_without_a_region": sum(not flagged(m) for m in positives),
            "group_mean_useful_fraction": float(np.mean(values["useful"])),
            "group_mean_edited_iou": float(np.mean(values["edited_iou"])),
            "controls": controls,
            "paired_against_current": {
                name: paired(metric, reference[name]) for name, metric in values.items()
            },
        }
    return {"quality": quality, "photos": len(components), "source_groups": len(groups), "methods": methods}


def seal(trial, expected):
    inputs, cases, _index = lease(trial, expected, prep_profile(trial))
    result = {}
    receipts = {"prepare": sha(prep_profile(trial) / "profile.json")}
    for quality in QUALITIES:
        rows = []
        for phase in range(PHASES):
            for arm in ARMS:
                producer_rows(trial, expected, arm, quality, phase)
                name = f"{arm}-q{quality}-{phase:02d}"
                receipts[name] = sha(trial / f"{name}-profile/profile.json")
            args = ["readout", str(trial), expected, str(quality), str(phase)]
            name = f"readout-q{quality}-{phase:02d}"
            done, batch = phase_rows(trial, name, "readout", args, expected)
            receipts[name] = sha(trial / f"{name}-profile/profile.json")
            require(
                done["phase"] == phase
                and done["quality"] == quality
                and done["neural_calls"] == 0
                and done["views"] == len(tasks(phase))
                and [(row["n"], row["kind"]) for row in batch] == tasks(phase)
                and all(
                    row["changed_pixels"] == inputs["sources"][row["n"]]["changed_pixels"][row["kind"]]
                    for row in batch
                ),
                "Incomplete readout or changed ordinary footprint",
            )
            rows.extend(batch)
        result[str(quality)] = summarize(rows, [case["component"] for case in cases], quality)
    freeze = read(trial / "freeze.json")
    write(
        trial / "summary.json",
        {
            "schema": "chipmunk.image-jpeg-diagnostic-result/189",
            "freeze_sha256": expected,
            "unique_model_calls": freeze["unique_model_calls"],
            "repeat_model_calls": freeze["exact_first_input_repeats"],
            "unassessed_calls": 0,
            "quality_results": result,
            "phase_profile_sha256": receipts,
            "serving_admitted": False,
            "calibration_performed": False,
            "ordinary_controls_have_zero_ai_truth": True,
            "scope": "Same SD2 editing family. Qualities share the same 48 source groups. Paired group bootstrap intervals "
            "are conditional and pointwise. Cue localization is separate from whole-file authorship confidence.",
        },
    )


def coverage(trial, expected):
    lease(trial, expected)
    totals = {}
    for quality in QUALITIES:
        for arm in ARMS:
            verified = 0
            unavailable = []
            for phase in range(PHASES):
                try:
                    verified += len(producer_rows(trial, expected, arm, quality, phase))
                except (OSError, ValueError, TypeError, KeyError):
                    unavailable.append(phase)
            totals[f"{arm}-q{quality}"] = {
                "planned": PHOTOS * len(KINDS),
                "verified": verified,
                "unverified": PHOTOS * len(KINDS) - verified,
                "unverified_phases": unavailable,
            }
    print(json.dumps(totals, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("trial", type=Path)
    build.add_argument("raw_root", type=Path)
    for name in ("score", "readout", "seal", "coverage"):
        command = commands.add_parser(name)
        command.add_argument("trial", type=Path)
        command.add_argument("freeze_sha256")
        if name == "score":
            command.add_argument("arm", choices=ARMS)
        if name in {"score", "readout"}:
            command.add_argument("quality", type=int, choices=QUALITIES)
            command.add_argument("phase", type=int, choices=range(PHASES))
    args = parser.parse_args()
    trial = args.trial.resolve()
    if args.command == "prepare":
        prepare(trial, args.raw_root)
    elif args.command == "score":
        score(trial, args.freeze_sha256, args.arm, args.quality, args.phase)
    elif args.command == "readout":
        readout(trial, args.freeze_sha256, args.quality, args.phase)
    else:
        (seal if args.command == "seal" else coverage)(trial, args.freeze_sha256)


if __name__ == "__main__":
    main()
