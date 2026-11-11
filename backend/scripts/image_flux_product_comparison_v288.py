"""Compare the actual PNG/JPEG regional routes on the six sealed v285 edits."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import image_flux_detector_comparison_v280 as previous
from scripts import image_flux_photo_expansion_v285 as photo
from scripts.video_appearance_corrected_v112 import scoped

ROOT, DATA, PANELS = previous.ROOT, previous.DATA, previous.PANELS
original, monitor = previous.original, previous.monitor
read, write, sha, require = previous.read, previous.write, previous.sha, previous.require
NAME = "image-flux-product-comparison-v288"
MODULE = "scripts.image_flux_product_comparison_v288"
TRIAL, RAW_ROOT = DATA / NAME, DATA / f"{NAME}-maps"
PUBLIC = PANELS / f"{NAME}-freeze.json"
FORMATS, QUALITIES, ARMS, KINDS = previous.FORMATS, previous.QUALITIES, previous.ARMS, previous.KINDS
PHOTOS, CHUNK, PHASES = 6, 10, 3
V285_RESULT = photo.TRIAL / "summary.json"
V285_PUBLIC_RESULT = PANELS / "image-flux-photo-expansion-complete-v285.json"
RECEIPTS = {
    photo.PUBLIC: "119e72d3d2672689560a3833f43237710fefbf616e3475352ddca4764c7f9827",
    V285_RESULT: "0118e1e7ec93249c0c31fb3776c332eb0052a7542d19ad1f82c42c82b7f8615f",
    V285_PUBLIC_RESULT: "8882d6f2ad434ee29c77846003ef16291a5aa7169a63038574cb502120eacaf2",
    **{
        path: digest
        for path, digest in previous.RECEIPTS.items()
        if path.parent == PANELS and "v264" in path.name
    },
    PANELS
    / "image-flux-detector-comparison-complete-v280.json": "23eeac4f9c85f5e6cac279caea9a0ea693c48044ca7802083f93ee1897266fa2",
}
SPLICE = "models/weights/splice/tidal-v30.1.pt"
SPLICE_SHA = "ca3aa144e6ca08f2bb2eae93ac45d76c27a2922770e028bf90ef76ce974ee061"
ROUTES = {
    0: {
        "detector": "loc.splice",
        "predictor_format": "native",
        "device": "cpu",
        "shape": [384, 384],
        "pixel_threshold": 0.70,
        "minimum_active_cells": 7993,
    },
    75: {
        "detector": "loc.fused",
        "predictor_format": "jpeg75",
        "device": "mps",
        "shape": [512, 512],
        "pixel_threshold": 0.50,
        "image_score_threshold": 0.50,
    },
}
CODE = (
    Path(__file__),
    ROOT / "backend/tests/test_image_flux_product_comparison_v288.py",
    ROOT / f"docs/{NAME}-method.md",
    Path(photo.__file__),
    Path(previous.__file__),
    ROOT / "backend/app/models.yaml",
    ROOT / "backend/app/detectors/image/adapters/localisers.py",
    ROOT / "backend/app/detectors/image/splice.py",
    ROOT / "backend/app/detectors/image/regional_support.py",
    ROOT / "backend/app/evidence/image_eligibility.py",
    ROOT / "backend/app/engine/registry.py",
    ROOT / "backend/app/core/config.py",
)


def source_cases():
    require(
        all(sha(path) == digest for path, digest in RECEIPTS.items()),
        "Exact v285 source and prior scoring receipts required",
    )
    sealed, aggregate = read(V285_RESULT), read(V285_PUBLIC_RESULT)
    require(
        sealed["status"] == "complete"
        and sealed["source_groups"] == sealed["generator_calls"] == PHOTOS
        and sealed["detector_calls"] == 0
        and sealed["case_outcomes"] == {"qualified_geometry": PHOTOS}
        and sealed["freeze_sha256"] == aggregate["freeze_sha256"] == RECEIPTS[photo.PUBLIC]
        and aggregate["sealed_result_sha256"] == RECEIPTS[V285_RESULT],
        "Six exact qualified v285 edits required",
    )
    cases, index = [], {}
    for batch in range(1, PHASES + 1):
        selected = photo.batch_inputs(batch)
        _, stage, output = photo.paths(batch)
        receipt = output / "result.json"
        require(
            sha(receipt) == sealed["batch_receipts"][str(batch)]["result_sha256"]
            and sha(stage / "profile.json") == sealed["batch_receipts"][str(batch)]["profile_sha256"],
            "A sealed v285 batch changed",
        )
        result = read(receipt)
        require(
            result["attempt_journal_sha256"] == sha(output / "attempts.jsonl")
            and result["freeze_sha256"] == RECEIPTS[photo.PUBLIC],
            "Exact v285 attempts journal required",
        )
        for before, after in zip(selected, result["cases"], strict=True):
            slot = before["slot"]
            require(
                slot == after["slot"]
                and after["status"] == "qualified_geometry"
                and after["outside_support_changes"] == 0,
                "Every frozen source needs its completed local edit",
            )
            cases.append({"source_id": slot, "component": f"photo-{len(cases) + 1}"})
            index[slot] = (before, after, output)
    require(
        len(cases) == len(index) == len({tuple(row[0]["group"]) for row in index.values()}) == PHOTOS,
        "Every one of the six frozen groups is required",
    )
    return cases, index


def material(cases, index, position):
    slot = cases[position]["source_id"]
    before, after, output = index[slot]
    return scoped(previous.material, {"photo": SimpleNamespace(TRIAL=photo.TRIAL, OUTPUT=output)})(
        cases, {slot: (before, after)}, position
    )


def installed_product_routes():
    from importlib.metadata import version
    from importlib.util import find_spec

    from app.core.config import settings
    from app.engine.fetchers import installed
    from app.engine.registry import get_registry

    registry = get_registry()
    current = [route["detector"] for route in ROUTES.values()]
    require(
        settings.image_region_mode == "legacy"
        and all(registry.decision(name).enabled for name in current)
        and not registry.decision("loc.native_regional").enabled
        and find_spec("segmentation_models_pytorch") is not None
        and version("segmentation-models-pytorch") == "0.5.0",
        "Exact configured product regional routes required",
    )
    for name in current:
        models = registry.required_models(registry.manifest.detector(name))
        require(
            all(installed(registry.manifest.model(model)) for model in models),
            "Product baseline assets are not installed",
        )
    splice = registry.manifest.model("tidal.splice_v30_1")
    require(
        splice.source.files[0].sha256 == SPLICE_SHA and splice.source.files[0].bytes == 97_958_262,
        "Installed PNG model manifest differs",
    )
    return {"image_region_mode": settings.image_region_mode, "segmentation_models_pytorch": "0.5.0"}


base_identity = scoped(previous.frozen_identity, {"PHOTOS": PHOTOS, "CODE": CODE, "RECEIPTS": RECEIPTS})


def frozen_identity():
    frozen, chain = base_identity()
    return {
        **frozen,
        "schema": "chipmunk.image-flux-product-comparison-freeze/288",
        "batches_per_format": PHASES,
        "views_per_batch": CHUNK,
        "current_routes": {FORMATS[quality]: route for quality, route in ROUTES.items()},
        "device": {"baseline_png": "cpu", "baseline_jpeg75": "mps", "candidate": "cpu"},
        "asset_sha256": frozen["asset_sha256"] | {SPLICE: SPLICE_SHA},
        "configured_product_route": installed_product_routes(),
        "candidate_jpeg75_use": "offline transfer probe only",
    }, chain


prepare = scoped(
    previous.prepare,
    {
        "TRIAL": TRIAL,
        "RAW_ROOT": RAW_ROOT,
        "source_cases": source_cases,
        "frozen_identity": frozen_identity,
        "material": material,
    },
)
MONITOR_IDENTITY = {"MODULE": MODULE, "__file__": __file__}
assert_profile = scoped(monitor.assert_profile, MONITOR_IDENTITY)
private_lease = scoped(
    previous.private_lease,
    {
        "TRIAL": TRIAL,
        "RAW_ROOT": RAW_ROOT,
        "source_cases": source_cases,
        "frozen_identity": frozen_identity,
        "assert_profile": assert_profile,
    },
)
lease = scoped(previous.lease, {"PUBLIC": PUBLIC, "private_lease": private_lease})
tasks = scoped(original.tasks, {"PHOTOS": PHOTOS, "PHASES": PHASES, "CHUNK": CHUNK})
batch_views = scoped(
    original.batch_views, {"tasks": tasks, "material": material, "jpeg_view": previous.encode}
)
phase_rows = scoped(previous.phase_rows, {"assert_profile": assert_profile})
producer_rows = scoped(original.producer_rows, {"phase_rows": phase_rows, "tasks": tasks})


def launch(argv, **kwargs):
    require(argv[1:3] == ["-m", MODULE], "Exact bounded v288 worker required")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.Popen([*argv[:3], "_worker", *argv[3:]], env=env, **kwargs)


run = scoped(
    monitor.run,
    {
        **MONITOR_IDENTITY,
        "assert_profile": assert_profile,
        "subprocess": SimpleNamespace(
            Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
        ),
    },
)


def response_shape(arm, quality, image):
    return tuple(ROUTES[quality]["shape"]) if arm == "baseline" else (image.height, image.width)


def score(trial, expected, arm, quality, phase):
    inputs, cases, index = lease(trial, expected, original.prep_profile(trial))
    chain = original.prior_chain()
    if arm == "baseline" and quality == 0:
        required = {SPLICE: SPLICE_SHA}
    else:
        required = {
            name: digest
            for name, digest in original.PREDICTOR_ASSETS.items()
            if ("/fused/" in name) == (arm == "baseline")
        }
    require(
        all(sha(ROOT / path) == digest for path, digest in required.items())
        and (arm == "baseline" or sha(Path(chain["final_checkpoint"])) == chain["final_checkpoint_sha256"]),
        "Frozen detector weights changed",
    )
    name = f"{arm}-q{quality}-{phase:02d}"
    out, raw_dir = trial / name, Path(inputs["raw_root"]) / name
    out.mkdir(mode=0o700)
    raw_dir.mkdir(mode=0o700)
    predict = original.v28.load_predictor(arm, ROUTES[quality]["predictor_format"], chain)
    journal = out / "rows.jsonl"
    with journal.open("x") as stream:
        for (n, kind), _payload, image, _truth, binding in batch_views(
            trial, inputs, cases, index, quality, phase
        ):
            try:
                raw, details = predict(image)
                require(
                    raw.shape == response_shape(arm, quality, image)
                    and raw.dtype == np.float32
                    and np.isfinite(raw).all()
                    and raw.min() >= 0
                    and raw.max() <= 1,
                    "Finite format-specific raw map required",
                )
                if phase == n == 0 and kind == KINDS[0]:
                    repeat, repeated_details = predict(image)
                    require(
                        np.array_equal(raw, repeat) and details == repeated_details,
                        "First-input repeat differs",
                    )
                target = raw_dir / f"{n:03d}-{kind}.npy"
                with target.open("xb") as destination:
                    np.save(destination, raw, allow_pickle=False)
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
                stream.write(
                    json.dumps({"status": "failed", "n": n, "kind": kind, "error_type": type(error).__name__})
                    + "\n"
                )
                raise
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
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


def current_gate(raw, details, quality):
    route = ROUTES[quality]
    threshold = route["pixel_threshold"]
    if quality == 0:
        require(details["image_score"] is None, "PNG recompression check has no image-level score")
        gate = np.count_nonzero(raw > threshold) >= route["minimum_active_cells"]
    else:
        from app.detectors.image.fused import candidate_eligibility

        gate = candidate_eligibility(details["image_score"])["eligible"]
    require(
        details["image_gate"] == gate and details["pixel_threshold"] == threshold,
        "Producer gate differs from the current product route",
    )
    return bool(gate), threshold


def measure(quality, old_map, old_details, new_map, payload, pixels, kind, footprint):
    from app.detectors.image.regional_support import readout as production_readout
    from scripts.image_native_comparison_readout_v45 import source_mask
    from scripts.localization_benchmark import mask_metrics

    ai_truth = footprint if kind == "ai" else np.zeros_like(footprint)
    require(
        kind in KINDS
        and footprint.dtype == bool
        and footprint.shape == pixels.shape[:2]
        and bool(ai_truth.any()) == (kind == "ai"),
        "Exact native truth or control support required",
    )
    old_gate, old_threshold = current_gate(old_map, old_details, quality)
    new_gate = original.v27.image_gate(new_map, original.POLICY)
    masks = []
    for raw, gate, threshold in (
        (old_map, old_gate, old_threshold),
        (new_map, new_gate, original.POLICY["pixel"]),
    ):
        height, width = footprint.shape
        chosen = production_readout(raw, gate, threshold, payload, pixels).source_mask(width, height)
        require(
            np.array_equal(chosen, source_mask(raw, gate, threshold, payload, pixels, ai_truth)),
            "Production and frozen source masks differ",
        )
        masks.append(chosen)
    masks.append(masks[0] & masks[1])
    return {
        name: {
            "ai": mask_metrics(mask, ai_truth),
            "ordinary": mask_metrics(mask, footprint) if kind in KINDS[2:] else None,
            "mask_sha256": original.digest(mask.tobytes()),
        }
        for name, mask in zip(original.METHODS, masks, strict=True)
    }


def readout(trial, expected, quality, phase):
    import cv2

    from scripts.replay_regional_calibration_v43 import load_map

    cv2.setNumThreads(1)
    inputs, cases, index = lease(trial, expected, original.prep_profile(trial))
    current, adapted = (producer_rows(trial, expected, arm, quality, phase) for arm in ARMS)
    raw_root = Path(inputs["raw_root"])
    out = trial / f"readout-q{quality}-{phase:02d}"
    out.mkdir(mode=0o700)
    journal = out / "rows.jsonl"
    with journal.open("x") as stream:
        for (key, payload, image, footprint, binding), old, new in zip(
            batch_views(trial, inputs, cases, index, quality, phase), current, adapted, strict=True
        ):
            n, kind = key
            require(
                old["binding"] == binding == new["binding"]
                and new["details"]["geometry"]["source_rgb_sha256"] == binding["decoded_rgb_sha256"]
                and new["details"]["geometry"]["source_size"] == binding["source_size"],
                "Exact decoded producer pixels required",
            )
            current_map = load_map(
                old, raw_root / f"baseline-q{quality}-{phase:02d}", response_shape("baseline", quality, image)
            )
            candidate_map = load_map(
                new,
                raw_root / f"candidate-q{quality}-{phase:02d}",
                response_shape("candidate", quality, image),
            )
            metrics = measure(
                quality,
                current_map,
                old["details"],
                candidate_map,
                payload,
                np.asarray(image),
                kind,
                footprint,
            )
            stream.write(
                json.dumps(
                    {
                        "n": n,
                        "kind": kind,
                        "component": cases[n]["component"],
                        "quality": quality,
                        "changed_pixels": int(footprint.sum()),
                        "methods": metrics,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
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


def summarize(rows, groups, quality):
    planned = [item for phase in range(PHASES) for item in tasks(phase)]
    result = scoped(previous.summarize, {"PHOTOS": PHOTOS, "tasks": lambda _: planned})(rows, groups, quality)
    result["current_product_detector"] = ROUTES[quality]["detector"]
    result["sensitive_scope"] = (
        "optional PNG product policy" if quality == 0 else "offline JPEG75 transfer probe"
    )
    for name in original.METHODS:
        for kind in KINDS[2:]:
            overlap = [row["methods"][name]["ordinary"] for row in rows if row["kind"] == kind]
            require(
                len(overlap) == PHOTOS and all(value is not None for value in overlap),
                "Every ordinary-edit footprint needs a paired measurement",
            )
            result["methods"][name]["controls"][kind]["groups_with_ordinary_footprint_overlap"] = sum(
                value["intersection"] > 0 for value in overlap
            )
    return result


def write_result(path, result):
    prepared = read(path.parent / "inputs.json")["sources"]
    write(
        path,
        {
            **{key: value for key, value in result.items() if key != "quality_results"},
            "schema": "chipmunk.image-flux-product-comparison-result/288",
            "format_results": {FORMATS[int(key)]: value for key, value in result["quality_results"].items()},
            "transport_changes_outside_native_footprint": {
                fmt: {
                    kind: [row["decoded_outside_native_footprint_changes"][fmt][kind] for row in prepared]
                    for kind in KINDS
                }
                for fmt in FORMATS.values()
            },
            "scope": "Six paired source groups. Format-matched current product routes and one frozen sensitive rule. JPEG75 sensitive is offline only. Exact native RGB truth in both formats. No fitting, population intervals or serving admission.",
        },
    )


seal = scoped(
    original.seal,
    {
        "lease": lease,
        "producer_rows": producer_rows,
        "phase_rows": phase_rows,
        "PHASES": PHASES,
        "tasks": tasks,
        "QUALITIES": QUALITIES,
        "summarize": summarize,
        "write": write_result,
    },
)
coverage = scoped(
    original.coverage,
    {
        "lease": lease,
        "producer_rows": producer_rows,
        "PHASES": PHASES,
        "PHOTOS": PHOTOS,
        "QUALITIES": QUALITIES,
    },
)
preflight = scoped(
    previous.preflight,
    {"source_cases": source_cases, "material": material, "frozen_identity": frozen_identity},
)


if __name__ == "__main__":
    scoped(
        previous.main,
        {
            "TRIAL": TRIAL,
            "RAW_ROOT": RAW_ROOT,
            "PUBLIC": PUBLIC,
            "PHASES": PHASES,
            "preflight": preflight,
            "prepare": prepare,
            "private_lease": private_lease,
            "run": run,
            "score": score,
            "readout": readout,
            "seal": seal,
            "coverage": coverage,
        },
    )()
