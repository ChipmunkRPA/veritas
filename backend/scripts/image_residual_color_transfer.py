# SPDX-License-Identifier: Apache-2.0
"""Compare local residual-color relationships at three native spatial bands."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import __version__ as pillow_version

from app.detectors.image import jpeg_dq, sharp_boundary
from scripts import image_edge_spread_transfer as panel
from scripts import image_local_resampling_transfer as roster

SCHEMA = "chipmunk.image-residual-color-transfer/1"
ARMS = ("fine", "mid", "mid_agreement")
BASELINE = "format_reference"
CELL = panel.CELL
SIGMAS = (0.85, 1.8, 3.6)
MIN_RMS = 0.65
MAX_CLIPPED_FRACTION = .30
MIN_RELIABLE_SHARE = .50
MIN_DIRECTION_COSINE = .30
MIN_POINT = .015
MIN_EXTRA_FEATHERED = 2
MAX_EXTRA_CONTROLS = 4
MAX_EXTRA_CONTROL_GROUPS = 3


def _tiles(values):
    rows, cols = (size // CELL for size in values.shape[:2])
    return values[:rows * CELL, :cols * CELL].reshape(rows, CELL, cols, CELL, *values.shape[2:])


def describe(residual, clipped):
    """Describe same-location RGB correlation and the energy of color differences."""
    bound = 3 * cv2.blur(np.abs(residual), (9, 9)) + 1
    normalized = np.clip(residual, -bound, bound)
    tiles = _tiles(normalized)
    centered = tiles - tiles.mean(axis=(1, 3), keepdims=True)
    powers = (centered ** 2).mean(axis=(1, 3))
    agreement = []
    for left, right in ((0, 1), (0, 2), (1, 2)):
        covariance = (centered[..., left] * centered[..., right]).mean(axis=(1, 3))
        agreement.append(covariance / np.sqrt((powers[..., left] + .25) * (powers[..., right] + .25)))
    chroma = sum(((centered[..., left] - centered[..., right]) ** 2).mean(axis=(1, 3))
                 for left, right in ((0, 1), (0, 2), (1, 2)))
    chroma_share = chroma / (3 * powers.sum(axis=-1) + .75)
    valid = (np.sqrt(powers.mean(axis=-1)) >= MIN_RMS) & (_tiles(clipped).mean(axis=(1, 3)) <= MAX_CLIPPED_FRACTION)
    return roster.integral(np.stack((valid, *(np.where(valid, item, 0.) for item in (*agreement, chroma_share))), axis=-1)
                           .astype(np.float64))


def _regions(boxes):
    left, top, right, bottom = boxes.T
    gap, outer = panel.GAP, panel.RING
    return (boxes + np.array((1, 1, -1, -1)),
            np.stack((left - outer, top, left - gap, bottom), axis=1),
            np.stack((right + gap, top, right + outer, bottom), axis=1),
            np.stack((left, top - outer, right, top - gap), axis=1),
            np.stack((left, bottom + gap, right, bottom + outer), axis=1))


def contrast(integral, regions):
    totals = [panel.sums(integral, region) for region in regions]
    counts = np.stack([total[:, 0] for total in totals], axis=1)
    sizes = np.stack([(region[:, 2] - region[:, 0]) * (region[:, 3] - region[:, 1]) for region in regions], axis=1)
    eligible = (counts >= np.maximum(4, MIN_RELIABLE_SHARE * sizes)).all(axis=1)
    core, *sides = [total[:, 1:] / np.maximum(1., total[:, :1]) for total in totals]
    distance_to_core = np.stack([np.mean(np.abs(core - side), axis=-1) for side in sides], axis=1)
    # The core must differ from at least three of four sides, beyond ordinary side-to-side variation.
    between_sides = np.stack([np.mean(np.abs(a - b), axis=-1)
                              for i, a in enumerate(sides) for b in sides[i + 1:]], axis=1)
    signal = np.sort(distance_to_core, axis=1)[:, 1]
    noise = np.median(between_sides, axis=1)
    direction = core - np.mean(sides, axis=0)
    return np.where(eligible, np.maximum(signal - noise, 0.), -np.inf), direction, counts, signal, noise


def response(rgb):
    height, width = rgb.shape[:2]
    panel.require(rgb.dtype == np.uint8 and rgb.shape == (height, width, 3) and
                  min(height, width) >= 256 and height * width <= 1_048_576, "Require native RGB up to one megapixel")
    boxes = panel.candidate_boxes(height // CELL, width // CELL)
    if not len(boxes):
        return dict.fromkeys(ARMS)
    pixels = rgb.astype(np.float32)
    clipped = np.any((rgb <= 3) | (rgb >= 252), axis=2)
    bands = [pixels, *(cv2.GaussianBlur(pixels, (0, 0), scale, borderType=cv2.BORDER_REFLECT_101) for scale in SIGMAS)]
    regions = _regions(boxes)
    comparisons = [contrast(describe(a - b, clipped), regions) for a, b in zip(bands[:-1], bands[1:], strict=True)]
    fine, mid1, mid2 = (item[0] for item in comparisons)
    u, v = (item[1] for item in comparisons[1:])
    cosine = np.sum(u * v, axis=1) / np.maximum(1e-9, np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    coherent = np.where((cosine >= MIN_DIRECTION_COSINE) & np.isfinite(mid1) & np.isfinite(mid2),
                        np.minimum(mid1, mid2), -np.inf)
    proposals = {}
    for arm, scores in zip(ARMS, (fine, mid1, coherent), strict=True):
        position = int(np.argmax(scores))
        if not np.isfinite(scores[position]):
            proposals[arm] = None
            continue
        indices = {"fine": (0,), "mid": (1,), "mid_agreement": (1, 2)}[arm]
        proposals[arm] = {
            "box": (boxes[position] * CELL).tolist(), "score": float(scores[position]),
            "coherence": float(cosine[position]) if len(indices) == 2 else None,
            "bands": [{"supported_tiles": comparisons[i][2][position].astype(int).tolist(),
                       "core_distance": float(comparisons[i][3][position]),
                       "side_distance": float(comparisons[i][4][position])} for i in indices],
        }
    return proposals


def format_reference(fmt, body, rgb):
    height, width = rgb.shape[:2]
    if fmt == "png":
        if not sharp_boundary.supported_size(width, height):
            return None
        candidate = sharp_boundary.measure(rgb)
        return {"box": list(candidate.box), "score": candidate.score, "selected": candidate.score > sharp_boundary.REVIEW_POINT}
    measured = jpeg_dq.analyse(body)
    if measured is None:
        return None
    region = measured.localised
    return {"box": [8 * region.x, 8 * region.y, 8 * (region.x + region.w), 8 * (region.y + region.h)] if region else None,
            "selected": region is not None}


def profile(development, earlier, later, boundary):
    base = roster.profile(development, earlier, later, boundary)
    code = (Path(__file__), Path(jpeg_dq.__file__), Path(sharp_boundary.__file__))
    return {**base, "schema": SCHEMA, "implementation_sha256": {
                **base["implementation_sha256"], **{str(path.relative_to(panel.ROOT)): panel.sha(path) for path in code}},
            "runtime": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": pillow_version},
            "method": {"arms": list(ARMS), "formats": list(panel.FORMATS), "sigmas": list(SIGMAS), "cell": CELL,
                       "candidate_sizes": [list(size) for size in panel.BOX_SIZES],
                       "support": {"minimum_residual_rms": MIN_RMS, "maximum_clipped_tile_fraction": MAX_CLIPPED_FRACTION,
                                   "minimum_supported_tile_share_each_region": MIN_RELIABLE_SHARE, "minimum_tiles": 4},
                       "min_midband_direction_cosine": MIN_DIRECTION_COSINE,
                       "point_rule": "Strictly above the second-highest of eight development source maxima or 0.015, whichever is larger.",
                       "fit_roles": list(panel.DEV_CONTROLS), "held_roles": [*panel.POSITIVE, *panel.CONTROLS],
                       "baseline": "Serving sharp-boundary check on PNG and aligned JPEG double-quantization region on JPEG75.",
                       "useful": "At least 0.30 native changed-pixel IoU and 0.50 recall.",
                       "advance": {"arm": "mid_agreement", "scope": "each panel and delivery separately",
                                   "additional_feathered": MIN_EXTRA_FEATHERED, "additional_control_flags": MAX_EXTRA_CONTROLS,
                                   "additional_control_groups": MAX_EXTRA_CONTROL_GROUPS},
                       "confidence_contribution": 0}}


def fit(folder):
    maxima = {fmt: {arm: [] for arm in ARMS} for fmt in panel.FORMATS}
    _, plan, attempts = panel.scoring.checked_inputs(folder)
    for _, _, images, _ in panel.scoring.panels(folder, plan, attempts):
        controls = panel.development_controls(images["original"], images["processed"])
        for fmt in panel.FORMATS:
            proposals = [response(np.asarray(panel.paired.encoded(image, fmt)[1])) for image in controls.values()]
            for arm in ARMS:
                maxima[fmt][arm].append(max((row[arm]["score"] for row in proposals if row[arm] is not None), default=0.))
    panel.require(all(len(values) == 8 for arms in maxima.values() for values in arms.values()), "Incomplete development sources")
    return {fmt: {arm: {"source_maxima": values, "point": max(MIN_POINT, sorted(values)[-2])}
                  for arm, values in arms.items()} for fmt, arms in maxima.items()}


def confirm(folder, journal, points):
    expected = {panel._key(row): row for row in panel.read(journal)}
    panel.require(len(expected) == 192, "Require exact earlier encoded views")
    for case, images, known in panel.paired.inputs(folder):
        for role, image in panel.native_variants(images, case["box"]).items():
            truth = np.any(image != images["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
            truth_sha = panel.paired.shared.digest(truth.tobytes())
            for fmt in panel.FORMATS:
                body, decoded = panel.paired.encoded(image, fmt)
                digest = panel.paired.shared.digest(body)
                saved = expected.pop((case["slot"], case["generator"], role, fmt))
                panel.require(digest == saved["input_sha256"]
                              and saved.get("truth_sha256", truth_sha) == truth_sha
                              and int(truth.sum()) == next(iter(saved["measurements"].values()))["target"],
                              "Source delivery or native edit changed")
                rgb = np.asarray(decoded)
                proposals = response(rgb)
                masks = {arm: panel.selection(value, points[fmt][arm]["point"], truth.shape) for arm, value in proposals.items()}
                ref = format_reference(fmt, body, rgb)
                masks[BASELINE] = (panel.paired.boxed(ref["box"], truth.shape, True)
                                   if ref and ref["selected"] else np.zeros_like(truth))
                yield {"slot": case["slot"], "role": role, "format": fmt, "input_sha256": digest, "truth_sha256": truth_sha,
                       "proposals": {**proposals, BASELINE: ref}, "mask_sha256": {
                           arm: panel.paired.shared.digest(mask.tobytes()) for arm, mask in masks.items()},
                       "measurements": {arm: panel.scoring.measure(mask, truth) for arm, mask in masks.items()}}
    panel.require(not expected, "Missing encoded views")


def summarize(rows):
    panel.require(len(rows) == 192, "Incomplete confirmation rows")
    reports = {}
    for fmt in panel.FORMATS:
        cohort = [row for row in rows if row["format"] == fmt]
        report = {}
        for arm in (*ARMS, BASELINE):
            positives = [r for r in cohort if r["role"] in panel.POSITIVE]
            hits = [r for r in positives if panel.useful(r["measurements"][arm])]
            flags = [r for r in cohort if r["role"] in panel.CONTROLS and r["measurements"][arm]["reported"]]
            extra_hits = [r for r in hits if r["role"] == "feathered_edit" and not panel.useful(r["measurements"][BASELINE])]
            extra_flags = [r for r in flags if not r["measurements"][BASELINE]["reported"]]
            report[arm] = {"sharp": sum(r["role"] == "edited" for r in hits),
                           "feathered": sum(r["role"] == "feathered_edit" for r in hits), "ordinary_flags": len(flags),
                           "ordinary_flagged_groups": len({r["slot"] for r in flags}), "extra_feathered": len(extra_hits),
                           "extra_ordinary_flags": len(extra_flags), "extra_ordinary_flagged_groups": len({r["slot"] for r in extra_flags}),
                           "unassessed_positive": sum(r["proposals"][arm] is None for r in positives),
                           "unassessed_ordinary": sum(r["proposals"][arm] is None for r in cohort if r["role"] in panel.CONTROLS)}
        reports[fmt] = report
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for name in ("development", "earlier", "later", "boundary_rows", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    frozen = profile(args.development, args.earlier, args.later, args.boundary_rows)
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(freeze_path, frozen)
        return
    panel.require(frozen == panel.read(freeze_path), "Frozen profile changed")
    cv2.setNumThreads(1)
    points_path = args.output / "points.json"
    if args.stage == "fit":
        panel.write(points_path, {"freeze_sha256": panel.sha(freeze_path), "points": fit(args.development)})
        return
    points = panel.read(points_path)
    panel.require(points["freeze_sha256"] == panel.sha(freeze_path), "Points not tied to freeze")
    started = time.monotonic()
    results, journals = {}, {}
    for name, folder, previous in (("earlier", args.earlier, args.boundary_rows), ("later", args.later, args.later / "outcomes.json")):
        rows = list(confirm(folder, previous, points["points"]))
        journal = args.output / f"{name}-rows.json"
        panel.write(journal, rows)
        results[name], journals[name] = summarize(rows), panel.sha(journal)
        print(name, "verified", len(rows), "encoded views", flush=True)
    eligible = {}
    for fmt in panel.FORMATS:
        eligible[fmt] = all((result := by_format[fmt]["mid_agreement"])["extra_feathered"] >= MIN_EXTRA_FEATHERED
                            and result["extra_ordinary_flags"] <= MAX_EXTRA_CONTROLS
                            and result["extra_ordinary_flagged_groups"] <= MAX_EXTRA_CONTROL_GROUPS for by_format in results.values())
    panel.write(args.output / "aggregate.json", {"schema": SCHEMA, "freeze_sha256": panel.sha(freeze_path),
                "points_sha256": panel.sha(points_path), "journal_sha256": journals, "elapsed_s": round(time.monotonic() - started, 3),
                "results": results, "mid_agreement_advances": eligible, "views": 384})


if __name__ == "__main__":
    main()
