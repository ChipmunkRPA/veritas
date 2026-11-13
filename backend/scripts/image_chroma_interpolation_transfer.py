# SPDX-License-Identifier: Apache-2.0
"""Test localized loss of a prior JPEG chroma interpolation grid."""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, JpegImagePlugin
from PIL import __version__ as pillow_version

from scripts import image_bg_real_insertion_score as development
from scripts import image_edge_spread_transfer as panel
from scripts import image_residual_color_transfer as comparison

SCHEMA = "chipmunk.image-chroma-interpolation-transfer/1"
FORMATS = ("jpeg100_444", "jpeg95_444", "jpeg95_420")
CELL = 16
SIZES = ((4, 4), (4, 6), (6, 4), (5, 5), (6, 6), (6, 8), (8, 6), (8, 8), (8, 12), (12, 8), (10, 10), (12, 12))
POSITIVE = ("sharp_edit", "feathered_edit")
CONTROL = ("preserved", "no_prior_chroma_grid", "global_tone", "global_resize", "local_blur", "local_tone", "local_autoencoder")
MIN_FLANK = .08
POINT_FLOOR = .06
GATE = {"extra_feathered_per_panel": 2, "extra_ordinary_views_per_panel": 4, "extra_ordinary_groups_per_panel": 3}
sha = panel.sha
read, write, require = panel.read, panel.write, panel.require


def encode(rgb, quality, sampling):
    out = io.BytesIO()
    Image.fromarray(rgb).save(out, "JPEG", quality=quality, subsampling=sampling)
    return out.getvalue()


def decode(body, chroma=False):
    with Image.open(io.BytesIO(body)) as image:
        sampling = JpegImagePlugin.get_sampling(image)
        if chroma:
            image.draft("YCbCr", image.size)
            require(image.mode == "YCbCr", "Need native JPEG chroma samples")
        return np.asarray(image).copy(), sampling


def phase_energies(chroma, axis):
    """Return four-tap interpolation residual totals at both phases per native tile."""
    source = np.moveaxis(chroma.astype(np.float64), axis, 1)
    rows, cols = (side // CELL for side in source.shape[:2])
    # Both parity groups are inside each tile and share the same number of samples.
    complete = source[:rows * CELL, :cols * CELL].reshape(rows, CELL, cols, CELL, 2)
    residual = complete[:, :, :, 3:] - 3 * complete[:, :, :, 2:-1] + 3 * complete[:, :, :, 1:-2] - complete[:, :, :, :-3]
    magnitude = np.minimum(np.abs(residual[:, :, :, :12]), 24.)
    totals = magnitude.reshape(rows, CELL, cols, 6, 2, 2).sum(axis=(1, 3, 5))
    return totals if axis == 1 else totals.transpose(1, 0, 2)


def candidates(rows, cols):
    boxes = []
    for height, width in SIZES:
        x, y = np.meshgrid(np.arange(3, cols - width - 2), np.arange(3, rows - height - 2))
        boxes.append(np.stack((x, y, x + width, y + height), axis=-1).reshape(-1, 4))
    return np.concatenate(boxes)


def measure(chroma):
    if min(chroma.shape[:2]) < 256 or np.prod(chroma.shape[:2]) > 1_048_576:
        return None
    totals = np.stack([phase_energies(chroma, axis) for axis in (1, 0)], axis=2)
    boxes = candidates(*totals.shape[:2])
    integral = np.pad(totals.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0), (0, 0)))
    values = np.stack([panel.sums(integral, region) for region in comparison._regions(boxes)], axis=1)
    a, b = values[..., 0], values[..., 1]
    signed = (b - a) / np.maximum(a + b, 1.)
    inner, flanks = signed[:, 0], signed[:, 1:]
    phase = np.where(np.median(flanks, axis=1) >= 0, 1., -1.)
    agreement = np.sort(flanks * phase[:, None, :], axis=1)[:, 1]
    supported = (agreement >= MIN_FLANK).all(axis=1)
    deficit = agreement - inner * phase
    scores = np.where(supported, np.maximum(0., deficit.min(axis=1)), -np.inf)
    selected = int(scores.argmax())
    if not np.isfinite(scores[selected]):
        return None
    return {"box": (boxes[selected] * CELL).tolist(), "score": float(scores[selected]),
            "interior_signed": inner[selected].tolist(), "flank_signed": flanks[selected].tolist(),
            "phase": phase[selected].astype(int).tolist()}


def inputs(original, alternatives, box):
    prior = encode(original, 90, 2)
    base, sampling = decode(prior)
    require(sampling == 2, "Expected earlier 4:2:0 encoding")
    photo = Image.fromarray(base)
    width, height = photo.size
    resized = photo.resize((round(.8 * width), round(.8 * height)), Image.Resampling.LANCZOS).resize(
        photo.size, Image.Resampling.LANCZOS)
    local = panel.blend.local
    variants = {
        "preserved": base,
        "no_prior_chroma_grid": original,
        "global_tone": np.asarray(ImageEnhance.Contrast(photo).enhance(1.2)),
        "global_resize": np.asarray(resized),
        "local_blur": local(base, np.asarray(photo.filter(ImageFilter.GaussianBlur(1.25))), box, feather=True),
        "local_tone": local(base, np.asarray(ImageEnhance.Contrast(photo).enhance(1.2)), box, feather=True),
        "local_autoencoder": local(base, alternatives["processed"], box, feather=True),
    }
    if "edited" in alternatives:
        variants.update(sharp_edit=local(base, alternatives["edited"], box),
                        feathered_edit=local(base, alternatives["edited"], box, feather=True))
    return variants


def delivery(image, fmt):
    sampling = 0 if fmt.endswith("444") else 2
    body = encode(image, 100 if fmt.startswith("jpeg100") else 95, sampling)
    ycc, actual = decode(body, chroma=True)
    require(actual == sampling, "Unexpected final delivery sampling")
    return body, ycc[..., 1:]


def fit(folder):
    maxima = {fmt: [] for fmt in FORMATS}
    _, plan, attempts = development.checked_inputs(folder)
    for _, _, assets, _ in development.panels(folder, plan, attempts):
        height, width = assets["original"].shape[:2]
        box = (width // 3, height // 3, width * 7 // 12, height * 7 // 12)
        variants = inputs(assets["original"], {"processed": assets["processed"]}, box)
        for fmt in FORMATS:
            rows = [measure(delivery(image, fmt)[1]) for image in variants.values()]
            maxima[fmt].append(max((row["score"] for row in rows if row), default=0.))
    require(all(len(rows) == 8 for rows in maxima.values()), "Need eight separate calibration source groups")
    return {fmt: {"source_maxima": rows, "point": max(POINT_FLOOR, sorted(rows)[-2])} for fmt, rows in maxima.items()}


def verify_authored_truth(photo, image, role, box):
    if role not in POSITIVE:
        return np.zeros(photo.shape[:2], dtype=bool)
    x0, y0, x1, y1 = box
    truth = np.any(image != photo, axis=2)
    require(int(truth.sum()) == int(truth[y0:y1, x0:x1].sum()) and truth.any(), "Edit escaped its native truth geometry")
    return truth


def confirm(folder, points):
    for row, images, _ in panel.paired.inputs(folder):
        box = tuple(row["box"])
        alternatives = {"processed": images["local_processing"], "edited": images["edited"]}
        variants = inputs(images["original"], alternatives, box)
        for role, image in variants.items():
            truth = verify_authored_truth(variants["preserved"], image, role, box)
            for fmt in FORMATS:
                body, chroma = delivery(image, fmt)
                proposal = measure(chroma)
                point = points[fmt]["point"]
                rgb, _ = decode(body)
                incumbent = comparison.format_reference("jpeg75", body, rgb)
                box_a = panel.paired.boxed(proposal["box"] if proposal else (), truth.shape, bool(proposal and proposal["score"] > point))
                box_b = panel.paired.boxed(incumbent["box"] if incumbent else (), truth.shape, bool(incumbent and incumbent["selected"]))
                digest = panel.paired.shared.digest
                yield {"slot": row["slot"], "generator": row["generator"], "role": role, "format": fmt,
                       "input_sha256": digest(body), "truth_sha256": digest(truth.tobytes()),
                       "canvas": list(reversed(truth.shape)),
                       "candidate": proposal, "baseline": incumbent,
                       "candidate_overlap": development.measure(box_a, truth), "baseline_overlap": development.measure(box_b, truth),
                       "candidate_rank_overlap": development.measure(panel.paired.boxed(proposal["box"] if proposal else (),
                                                                                        truth.shape, bool(proposal)), truth)}


def summarize(rows):
    require(len(rows) == 12 * len(FORMATS) * (len(POSITIVE) + len(CONTROL)), "Incomplete held delivery roster")
    summary = {}
    for fmt in FORMATS:
        selected = [row for row in rows if row["format"] == fmt]
        arms = {}
        for arm in ("candidate", "baseline"):
            field = arm + "_overlap"
            arms[arm] = {"useful": {role: sum(panel.useful(row[field]) for row in selected if row["role"] == role)
                                    for role in POSITIVE},
                         "ordinary_flags": {role: sum(bool(row[field]["reported"]) for row in selected if row["role"] == role)
                                            for role in CONTROL},
                         "ordinary_groups": len({row["slot"] for row in selected
                                                 if row["role"] in CONTROL and row[field]["reported"]}),
                         "assessed": sum(row[arm] is not None for row in selected)}
        new = [row for row in selected if row["role"] in CONTROL and row["candidate_overlap"]["reported"]
               and not row["baseline_overlap"]["reported"]]
        extra = {"feathered": sum(row["role"] == "feathered_edit" and panel.useful(row["candidate_overlap"])
                                  and not panel.useful(row["baseline_overlap"]) for row in selected),
                 "ordinary_views": len(new), "ordinary_groups": len({row["slot"] for row in new})}
        summary[fmt] = {**arms, "extra": extra, "candidate_rank_useful": sum(panel.useful(row["candidate_rank_overlap"])
                                                                            for row in selected if row["role"] in POSITIVE),
                        "advance": (extra["feathered"] >= GATE["extra_feathered_per_panel"]
                                    and extra["ordinary_views"] <= GATE["extra_ordinary_views_per_panel"]
                                    and extra["ordinary_groups"] <= GATE["extra_ordinary_groups_per_panel"])}
    return summary


def profile(development_path, earlier, later):
    code = (Path(__file__), Path(panel.__file__), Path(comparison.__file__), Path(development.__file__),
            Path(panel.paired.__file__), Path(panel.blend.__file__), Path(comparison.jpeg_dq.__file__))
    sources = {role: {name: sha(path / name) for name in ("plan.json", "result.json", "attempts.jsonl")}
               for role, path in zip(("development", "earlier", "later"), (development_path, earlier, later), strict=True)}
    return {"schema": SCHEMA, "source_records": sources,
            "code": {path.name: sha(path) for path in code}, "runtime": {"opencv": cv2.__version__, "pillow": pillow_version, "numpy": np.__version__},
            "method": {"earlier_jpeg": "quality 90, 4:2:0", "delivery": list(FORMATS), "cell": CELL,
                       "candidate_shapes_in_cells": [list(size) for size in SIZES], "minimum_flank_phase": MIN_FLANK,
                       "point_floor": POINT_FLOOR, "point_rule": "second-highest development control-source maximum per delivery, strict exceedance",
                       "positive_roles": list(POSITIVE), "ordinary_roles": list(CONTROL), "gate": GATE,
                       "useful": {"iou": .30, "recall": .50}, "confidence_weight": 0}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for field in ("development", "earlier", "later", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    frozen = profile(args.development, args.earlier, args.later)
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(freeze_path, frozen)
        return
    require(frozen == read(freeze_path), "Frozen procedure or sources changed")
    cv2.setNumThreads(1)
    points_path = args.output / "points.json"
    if args.stage == "fit":
        write(points_path, {"freeze_sha256": sha(freeze_path), "points": fit(args.development)})
        return
    fitted = read(points_path)
    require(fitted["freeze_sha256"] == sha(freeze_path), "Points must match the freeze")
    started = time.monotonic()
    summary, measurements = {}, {}
    for role in ("earlier", "later"):
        rows = list(confirm(getattr(args, role), fitted["points"]))
        summary[role], measurements[role] = summarize(rows), rows
        print(role, summary[role], flush=True)
    write(args.output / "result.json", {"freeze": frozen, "freeze_sha256": sha(freeze_path), "points": fitted["points"],
                                        "points_sha256": sha(points_path), "summary": summary, "rows": measurements,
                                        "elapsed_s": round(time.monotonic() - started, 3)})


if __name__ == "__main__":
    main()
