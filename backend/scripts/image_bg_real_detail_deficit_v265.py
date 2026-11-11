"""Measure cross-band detail deficits on the sealed BG-REAL image views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import scipy
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter

from scripts import image_bg_real_backup_jpeg_v264 as prior

NAME = "image-bg-real-detail-deficit-v265"
PUBLIC_FREEZE = prior.PANELS / f"{NAME}-freeze.json"
PRIOR_RESULT = prior.PANELS / "image-bg-real-backup-jpeg-v264-complete.json"
PRIOR_RESULT_SHA = "438f754a667e90e5b42b977085a36725021cdb8180a3d6e9fca0a9206e267fc4"
PRIOR_FREEZE_SHA = "4db71eddb75033e8ac41d9716d2eb5cf7356a1d544a51b91de08657fd7ed44f9"
CODE = (
    f"backend/scripts/{Path(__file__).name}",
    "backend/tests/test_image_bg_real_detail_deficit_v265.py",
    f"docs/{NAME}-method.md",
)
ENCODINGS = ("source", "jpeg75", "jpeg90")
ORDINARY_EDITS = ("tone", "blur", "copy_move")
TILE = 16
SIGMAS = (0.8, 2.0)
VARIANCE_FLOOR = 0.25
NEIGHBORHOOD = 5
LUMA = (0.299, 0.587, 0.114)


def tiles(array):
    rows, cols = (length // TILE for length in array.shape)
    return array[: rows * TILE, : cols * TILE].reshape(rows, TILE, cols, TILE)


def detail_map(pixels):
    """Compute one signed response per complete tile using only the image."""
    prior.require(
        pixels.ndim == 3
        and pixels.shape[2] == 3
        and pixels.dtype == np.uint8
        and min(pixels.shape[:2]) >= TILE * NEIGHBORHOOD,
        "Upright RGB source and sufficient tile context required",
    )
    luminance = pixels @ np.asarray(LUMA, dtype=np.float32)
    fine_blur, coarse_blur = (gaussian_filter(luminance, sigma, mode="reflect") for sigma in SIGMAS)
    fine, coarse = tiles(luminance - fine_blur), tiles(fine_blur - coarse_blur)
    tile_axes = (1, 3)
    fine = fine - fine.mean(axis=tile_axes, keepdims=True)
    coarse = coarse - coarse.mean(axis=tile_axes, keepdims=True)
    fine_power = (fine * fine).mean(axis=tile_axes)
    coarse_power = (coarse * coarse).mean(axis=tile_axes)
    covariance = (fine * coarse).mean(axis=tile_axes)
    unexplained = np.maximum(fine_power - covariance**2 / (coarse_power + VARIANCE_FLOOR), 0)
    ratio = np.log((coarse_power + VARIANCE_FLOOR) / (unexplained + VARIANCE_FLOOR))
    return ratio - median_filter(ratio, size=NEIGHBORHOOD, mode="reflect")


def compare(scores, footprint):
    """Compare affected tiles with remote tiles and weight by exact changed area."""
    weights = tiles(footprint).sum(axis=(1, 3))
    prior.require(scores.shape == weights.shape and np.isfinite(scores).all(), "Aligned tile map required")
    target = weights > 0
    controls = ~binary_dilation(target, structure=np.ones((3, 3), dtype=bool))
    if not target.any() or not controls.any():
        return None
    background = np.sort(scores[controls])
    above = background.searchsorted(scores[target], side="left")
    above_or_tied = background.searchsorted(scores[target], side="right")
    ranks = (above + above_or_tied) / (2 * len(background))
    return {
        "concordance": float(np.average(ranks, weights=weights[target])),
        "target_response": float(np.average(scores[target], weights=weights[target])),
        "footprint_retained": int(weights.sum()) / int(footprint.sum()),
    }


def identity():
    sealed = prior.read(PRIOR_RESULT)
    prior.require(
        prior.sha(PRIOR_RESULT) == PRIOR_RESULT_SHA
        and prior.sha(prior.PUBLIC) == PRIOR_FREEZE_SHA
        and sealed == prior.read(prior.TRIAL / "summary.json")
        and sealed["freeze_sha256"] == PRIOR_FREEZE_SHA
        and sealed["unique_model_calls"] == 240
        and sealed["unassessed_calls"] == 0,
        "Exact sealed v264 comparison required",
    )
    return {
        "schema": "chipmunk.image-bg-real-detail-deficit-freeze/265",
        "v264_public_result_sha256": PRIOR_RESULT_SHA,
        "v264_input_freeze_sha256": PRIOR_FREEZE_SHA,
        "code_sha256": {name: prior.sha(prior.ROOT / name) for name in CODE},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "source_groups": prior.PHOTOS,
        "views": list(prior.KINDS),
        "encodings": list(ENCODINGS),
        "tile_pixels": TILE,
        "gaussian_sigmas_pixels": list(SIGMAS),
        "minimum_variance_regularizer": VARIANCE_FLOOR,
        "context_tiles_per_side": NEIGHBORHOOD,
        "luma_weights": list(LUMA),
        "fitted": False,
        "new_neural_calls": 0,
    }


def views(cases, index, position, entry):
    images, footprints = prior.previous.material(cases, index, position)
    mask_path = prior.TRIAL / "masks" / f"{position:03d}.npz"
    prior.require(prior.sha(mask_path) == entry["masks_sha256"], "Exact v264 mask commitment required")
    with np.load(mask_path, allow_pickle=False) as masks:
        prior.require(
            set(masks.files) == set(footprints)
            and all(np.array_equal(masks[kind], mask) for kind, mask in footprints.items()),
            "Exact qualified footprints required",
        )
    for kind, image in images.items():
        yield "source", kind, image, footprints
        for quality in prior.QUALITIES:
            payload, decoded = prior.original.jpeg_view(image, quality)
            prior.require(
                prior.original.view_binding(payload, decoded) == entry["inputs"][str(quality)][kind],
                "Exact reconstructed v264 JPEG required",
            )
            yield f"jpeg{quality}", kind, decoded, footprints


def summarize(values):
    return {
        "groups": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
    }


def paired(values):
    return {
        **summarize(values),
        "positive": sum(value > 0 for value in values),
        "tie": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
    }


def readout(observations):
    by_encoding = {}
    for fmt in ENCODINGS:
        cohort = [observation[fmt] for observation in observations]
        report = {}
        for kind in prior.KINDS:
            fixed = [row[kind]["same"] for row in cohort if row[kind]["same"] is not None]
            own = [row[kind]["own"] for row in cohort if row[kind]["own"] is not None]
            report[kind] = {
                "same_site_concordance": summarize([item["concordance"] for item in fixed]),
                "own_footprint_concordance": summarize([item["concordance"] for item in own]),
                "same_site_exact_area_retained": summarize([item["footprint_retained"] for item in fixed]),
            }
        contrast = {
            kind: paired(
                [
                    row["ai"]["same"]["target_response"] - row[kind]["same"]["target_response"]
                    for row in cohort
                    if row["ai"]["same"] is not None and row[kind]["same"] is not None
                ]
            )
            for kind in prior.KINDS
            if kind != "ai"
        }
        hardest = paired(
            [
                row["ai"]["same"]["target_response"]
                - max(row[kind]["same"]["target_response"] for kind in ORDINARY_EDITS)
                for row in cohort
                if all(row[kind]["same"] is not None for kind in ("ai", *ORDINARY_EDITS))
            ]
        )
        by_encoding[fmt] = {
            "views": report,
            "ai_minus_control_same_site_response": contrast,
            "ai_minus_strongest_ordinary_edit_same_site_response": hardest,
        }
    return by_encoding


def run():
    locked = identity()
    prior.require(prior.read(PUBLIC_FREEZE) == locked, "Exact prospective v265 freeze required")
    inputs, cases, index = prior.lease(prior.TRIAL, PRIOR_FREEZE_SHA)
    observations = []
    for position, entry in enumerate(inputs["sources"]):
        group = {fmt: {} for fmt in ENCODINGS}
        for fmt, kind, image, footprints in views(cases, index, position, entry):
            scores = detail_map(np.asarray(image))
            same = compare(scores, footprints["ai"])
            own = same if kind == "ai" else compare(scores, footprints[kind])
            group[fmt][kind] = {"same": same, "own": own}
        prior.require(all(set(rows) == set(prior.KINDS) for rows in group.values()), "Incomplete group")
        observations.append(group)
    prior.require(len(observations) == len(cases) == prior.PHOTOS, "Incomplete qualified roster")
    return {
        "schema": "chipmunk.image-bg-real-detail-deficit-result/265",
        "freeze_sha256": prior.sha(PUBLIC_FREEZE),
        "source_groups": len(cases),
        "image_views": len(cases) * len(ENCODINGS) * len(prior.KINDS),
        "encodings": readout(observations),
        "new_neural_calls": 0,
        "serving_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    if parser.parse_args().command == "freeze":
        locked = identity()
        with PUBLIC_FREEZE.open("x") as output:
            json.dump(locked, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
        result = {"status": "frozen", "freeze_sha256": prior.sha(PUBLIC_FREEZE)}
    else:
        result = run()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
