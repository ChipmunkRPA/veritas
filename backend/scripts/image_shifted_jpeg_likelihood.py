# SPDX-License-Identifier: Apache-2.0
"""Measure local earlier-grid quantization evidence in a delivered JPEG."""
from __future__ import annotations

import argparse
import io
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, features
from PIL import __version__ as pillow_version

from app.provenance.metadata_rules import STD_LUMA, ijg_table
from scripts import image_bg_real_insertion_score as metrics
from scripts import image_local_grid_transfer as source

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "evaluation-panels"
SCHEMA = "chipmunk.image-shifted-jpeg-likelihood/1"
DELIVERIES = {"jpeg75": 75, "jpeg90": 90}
PRIOR_QUALITIES = (50, 59, 67, 75, 85)
FREQUENCIES = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2))
PHASES = tuple((x, y) for y in range(8) for x in range(8) if (x, y) != (0, 0))
CELL = 32
SIZES = ((3, 3), (4, 4), (4, 6), (6, 4), (5, 5), (6, 6))
RING = 2
MIN_OBSERVATIONS_PER_CELL = 8
MIN_COEFFICIENT_PERIODS = 0.7
MIN_PERIOD_TO_NOISE = 2.0
ROUNDING_SIGMA = 0.75
MIN_POINT = 0.0
DESIGNED_PHASE = (5, 3)
CONTROLS = tuple(role for role in source.ROLES if role not in source.EDITS)
METHODS = ("shifted_likelihood", "existing_aligned_grid", "dq_corroborated")
COMPARISONS = {"phase": "image-local-grid-transfer", "recompression": "image-multiscale-jpeg-transfer"}
PANELS = ("earlier", "later")
COUNTS = (8, 12, 12)


def _read(path):
    return json.loads(path.read_bytes())


def _write(path, data):
    with path.open("x", encoding="utf-8") as output:
        json.dump(data, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


@dataclass(frozen=True)
class Region:
    score: float
    box: tuple[int, int, int, int]
    phase: tuple[int, int]
    prior_quality: int
    inside_observations: int
    outside_observations: int


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dct_basis():
    pixels = np.arange(8) + 0.5
    basis = 0.5 * np.cos(np.pi / 8 * np.arange(8)[:, None] * pixels)
    basis[0] /= math.sqrt(2)
    return basis


def axis_projection(basis, frequency, phase):
    """Squared projections of one shifted 1-D basis onto the two final blocks."""
    lengths = (8 - phase, phase)
    return ((basis[:, phase:] @ basis[frequency, :lengths[0]]) ** 2
            + (basis[:, :phase] @ basis[frequency, lengths[0]:]) ** 2)


def noise_scales(final_table, basis, x, y):
    # Independent uniform errors in final quantized coefficients. Squared
    # projections sum across the final blocks and yield a variance in the
    # earlier DCT coefficient, including a conservative pixel-rounding term.
    variance = np.square(final_table.astype(np.float64)) / 12
    return np.array([math.sqrt(axis_projection(basis, v, y) @ variance @ axis_projection(basis, u, x)
                               + ROUNDING_SIGMA ** 2) for v, u in FREQUENCIES], dtype=np.float32)


def jpeg_luma(body):
    with Image.open(io.BytesIO(body)) as image:
        require(image.format == "JPEG" and image.mode in {"RGB", "L"} and image.bits == 8
                and min(image.size) >= 320 and math.prod(image.size) <= 1_048_576,
                "Only 8-bit luma JPEG geometry from 320 pixels up to one megapixel is assessed")
        require(image.mode == "L" or tuple(layer[0] for layer in image.layer) == (1, 2, 3),
                "The original JPEG must have an identifiable luma component")
        table = np.asarray(image.quantization[image.layer[0][3]], dtype=np.int32).reshape(8, 8)
        image.draft("L", image.size)
        return np.asarray(image.convert("L"), dtype=np.float32), table


def block_statistics(coefficients, period, sigma):
    """Wrapped-normal earlier-quantization likelihood against a uniform phase."""
    units = coefficients / period
    use = (period >= MIN_PERIOD_TO_NOISE * sigma) & (np.abs(units) >= MIN_COEFFICIENT_PERIODS)
    residual = units - np.floor(units + 0.5)
    ratio = period / sigma
    density = np.exp(-0.5 * np.square(residual * ratio))
    # At sigma <= period/2, the omitted terms each start at five standard deviations.
    for neighbor in (-2, -1, 1, 2):
        density += np.exp(-0.5 * np.square((residual + neighbor) * ratio))
    density *= ratio / np.sqrt(2 * np.pi)
    likelihood = np.where(use, np.log(np.maximum(density, 1e-5)), 0)
    return likelihood.sum(axis=0, dtype=np.float32), use.sum(axis=0, dtype=np.int32)


def _integral(blocks):
    rows, columns = (n // 4 for n in blocks.shape)
    cells = blocks[:4 * rows, :4 * columns].reshape(rows, 4, columns, 4).sum(axis=(1, 3), dtype=np.float64)
    return np.pad(cells.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))


def _window_totals(integral, h, w):
    return integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]


def best_window(block_scores, block_counts, x, y, quality):
    score, count = _integral(block_scores), _integral(block_counts)
    best = None
    for h, w in SIZES:
        if score.shape[0] <= h + 2 * RING or score.shape[1] <= w + 2 * RING:
            continue
        inside_s = _window_totals(score, h, w)[RING:-RING, RING:-RING]
        inside_n = _window_totals(count, h, w)[RING:-RING, RING:-RING]
        outside_s = _window_totals(score, h + 2 * RING, w + 2 * RING) - inside_s
        outside_n = _window_totals(count, h + 2 * RING, w + 2 * RING) - inside_n
        inside = inside_s / np.maximum(1, inside_n)
        outside = outside_s / np.maximum(1, outside_n)
        supported = ((inside_n >= h * w * MIN_OBSERVATIONS_PER_CELL)
                     & (outside_n >= ((h + 2 * RING) * (w + 2 * RING) - h * w) * MIN_OBSERVATIONS_PER_CELL))
        statistic = np.where(supported, np.minimum(inside, inside - outside) * np.sqrt(np.minimum(inside_n, outside_n)),
                             -np.inf)
        i, j = np.unravel_index(statistic.argmax(), statistic.shape)
        value = float(statistic[i, j])
        if math.isfinite(value) and (best is None or value > best.score):
            left, top = x + CELL * (j + RING), y + CELL * (i + RING)
            best = Region(value, (int(left), int(top), int(left + CELL * w), int(top + CELL * h)), (x, y), quality,
                          int(inside_n[i, j]), int(outside_n[i, j]))
    return best


def response(body):
    luma, table = jpeg_luma(body)
    basis = dct_basis()
    prior_tables = [ijg_table(STD_LUMA, q).reshape(8, 8) for q in PRIOR_QUALITIES]
    periods = [np.array([table[v, u] for v, u in FREQUENCIES], dtype=np.float32)[:, None, None]
               for table in prior_tables]
    height, width = ((size - 7) // 8 * 8 for size in luma.shape)
    filters = np.stack([cv2.sepFilter2D(luma, cv2.CV_32F, basis[u], basis[v], anchor=(0, 0))[:height + 7, :width + 7]
                        for v, u in FREQUENCIES])
    best = None
    for x, y in PHASES:
        coefficients = np.ascontiguousarray(filters[:, y:y + height:8, x:x + width:8])
        sigmas = noise_scales(table, basis, x, y)[:, None, None]
        for quality, period in zip(PRIOR_QUALITIES, periods, strict=True):
            scores, counts = block_statistics(coefficients, period, sigmas)
            candidate = best_window(scores, counts, x, y, quality)
            if candidate is not None and (best is None or candidate.score > best.score):
                best = candidate
    return best


def selection(region, point, shape):
    mask = np.zeros(shape, dtype=bool)
    if region is not None and region.score > point:
        left, top, right, bottom = region.box
        mask[top:bottom, left:right] = True
    return mask


def receipt_files(directories):
    return {label: {name: source.sha(folder / name) for name in ("plan.json", "attempts.jsonl", "result.json")}
            for label, folder in zip(("development", *PANELS), directories, strict=True)}


def reference_files(phase, recompression, receipts):
    bindings = {}
    for (label, study), directory in zip(COMPARISONS.items(), (phase, recompression), strict=True):
        report_path = DOCS / f"{study}-2026-11-11.json"
        report = _read(report_path)
        freeze_key = "shared_implementation_freeze_sha256" if label == "phase" else "freeze_sha256"
        journal_key = "confirmation_journal_sha256" if label == "phase" else "journal_sha256"
        freeze = directory / "freeze.json"
        journals = {name: source.sha(directory / f"{name}-rows.json") for name in PANELS}
        require(source.sha(freeze) == report[freeze_key] and journals == report[journal_key],
                "Historical journal differs from the published evaluation")
        source_key = "sources" if label == "phase" else "source_receipts_sha256"
        require(_read(freeze)[source_key] == [v["result.json"] for v in receipts.values()],
                "Historical evaluation used different source intake receipts")
        bindings[label] = {"public_report_sha256": source.sha(report_path), "freeze_sha256": source.sha(freeze),
                           "confirmation_journals_sha256": journals}
    return bindings


def lock(directories, phase, recompression):
    plans = [_read(folder / "plan.json")["cases"] for folder in directories]
    for key in ("flickr_photo_id", "flickr_owner"):
        groups = [{row[key] for row in cases} for cases in plans]
        require(tuple(map(len, groups)) == COUNTS and len(set.union(*groups)) == sum(COUNTS),
                "Expected exact source and creator disjoint development and confirmation roles")
    receipts = receipt_files(directories)
    dependency_files = (Path(__file__), Path(source.__file__), Path(metrics.__file__), Path(source.panel.__file__),
                        Path(source.blend.__file__), Path(source.shared.flux.previous.__file__))
    return {
        "schema": SCHEMA, "sources": receipts, "source_group_counts": list(COUNTS),
        "previous_comparisons": reference_files(phase, recompression, receipts),
        "implementation_sha256": {p.name: source.sha(p) for p in dependency_files},
        "runtime": {"pillow": pillow_version, "libjpeg": features.version("jpg"), "numpy": np.__version__,
                    "opencv": cv2.__version__},
        "method": {
            "deliveries": DELIVERIES, "prior_quality_grid": list(PRIOR_QUALITIES), "searched_nonzero_phases": len(PHASES),
            "frequencies_yx": [list(f) for f in FREQUENCIES], "rounding_sigma": ROUNDING_SIGMA,
            "minimum_period_to_noise_ratio": MIN_PERIOD_TO_NOISE, "minimum_magnitude_periods": MIN_COEFFICIENT_PERIODS,
            "cell_pixels": CELL, "candidate_sizes_cells_hw": [list(s) for s in SIZES], "ring_width_cells": RING,
            "minimum_observations_per_cell": MIN_OBSERVATIONS_PER_CELL, "minimum_point": MIN_POINT,
            "point_rule": "Second-highest source maximum among the eight development sources per final delivery. Strict greater-than flag.",
            "control_roles": list(CONTROLS), "edit_roles": list(source.EDITS), "known_nonzero_phase_xy": list(DESIGNED_PHASE),
            "useful": {"minimum_iou": 0.30, "minimum_recall": 0.50},
            "pilot_rule": "Per delivery, more useful shifted edit locations than the aligned comparison in total, no loss in either source panel, and no more control flags or flagged control groups in either panel.",
            "authorship_confidence_contribution": 0,
        },
    }


def develop(folder):
    _, plan, receipts = metrics.checked_inputs(folder)
    maxima = {fmt: [] for fmt in DELIVERIES}
    for _, _, native, _ in metrics.panels(folder, plan, receipts):
        original = native["original"]
        h, w = original.shape[:2]
        controls = source.materials(original, original, (w // 3, h // 3, w * 7 // 12, h * 7 // 12))
        for fmt, quality in DELIVERIES.items():
            values = []
            for role in CONTROLS:
                body, _ = source.delivery(controls[role], quality)
                region = response(body)
                values.append(region.score if region is not None else MIN_POINT)
            maxima[fmt].append(max(values))
    require(all(len(values) == COUNTS[0] for values in maxima.values()), "Incomplete development sources")
    return {fmt: {"point": max(MIN_POINT, sorted(values)[-2]), "source_maxima": values} for fmt, values in maxima.items()}


def _key(row):
    return row["slot"], row["role"], row["format"]


def reference_rows(name, phase, recompression):
    first, second = ({_key(row): row for row in _read(folder / f"{name}-rows.json") if row["format"] in DELIVERIES}
                     for folder in (phase, recompression))
    require(set(first) == set(second) and len(first) == 12 * len(source.ROLES) * len(DELIVERIES),
            "Historical JPEG view roster is incomplete")
    for key, row in first.items():
        other = second[key]
        require(all(row[field] == other[field] for field in ("input_sha256", "truth_sha256"))
                and row["measurements"]["existing_aligned_grid"] == other["measurements"]["existing_aligned_grid"],
                "Historical JPEG comparisons disagree on exact inputs or baseline outcomes")
    return second


def confirm(folder, points, previous):
    for case, native, _ in source.panel.inputs(folder):
        views = source.materials(native["original"], native["edited"], case["box"])
        for role, pixels in views.items():
            shape = pixels.shape[:2]
            truth = np.any(pixels != views["original"], axis=2) if role in source.EDITS else np.zeros(shape, dtype=bool)
            truth_sha = source.shared.digest(truth.tobytes())
            for fmt, quality in DELIVERIES.items():
                prior = previous[case["slot"], role, fmt]
                body, _ = source.delivery(pixels, quality)
                input_sha = source.shared.digest(body)
                require(input_sha == prior["input_sha256"] and truth_sha == prior["truth_sha256"],
                        "Delivered JPEG bytes or native truth mask differs from the sealed comparison")
                region = response(body)
                mask = selection(region, points[fmt]["point"], shape)
                yield {"slot": case["slot"], "role": role, "format": fmt, "input_sha256": input_sha,
                       "truth_sha256": truth_sha, "candidate_mask_sha256": source.shared.digest(mask.tobytes()),
                       "candidate_region": asdict(region) if region is not None else None,
                       "measurements": {"shifted_likelihood": metrics.measure(mask, truth),
                                        **{arm: prior["measurements"][arm] for arm in METHODS[1:]}}}


def useful(measurement):
    seen, truth, overlap = (measurement[k] for k in ("reported", "target", "overlap"))
    return truth > 0 and 10 * overlap >= 3 * (seen + truth - overlap) and 2 * overlap >= truth


def summarize(rows):
    roster = {_key(row) for row in rows}
    groups = {slot for slot, _, _ in roster}
    require(len(groups) == 12 and len(roster) == len(rows)
            and roster == set(product(groups, source.ROLES, DELIVERIES)), "Confirmation views are missing or repeated")
    result = {}
    for fmt in DELIVERIES:
        subset = [row for row in rows if row["format"] == fmt]
        result[fmt] = {}
        for arm in METHODS:
            counts = Counter()
            controls, edits = set(), set()
            for row in subset:
                role, value = row["role"], row["measurements"][arm]
                if role in CONTROLS and value["reported"]:
                    controls.add(row["slot"])
                    counts[f"{role}_flags"] += 1
                if role in source.EDITS and useful(value):
                    edits.add(row["slot"])
                    counts[f"{role}_useful"] += 1
            result[fmt][arm] = {"control_flags": sum(counts[f"{role}_flags"] for role in CONTROLS),
                                "control_groups_flagged": len(controls), "groups_with_useful_edits": len(edits),
                                "useful_locations": sum(counts[f"{role}_useful"] for role in source.EDITS),
                                "by_role": {role: counts[f"{role}_{'flags' if role in CONTROLS else 'useful'}"]
                                            for role in source.ROLES}}
        counts = Counter()
        for row in subset:
            arm = row["measurements"]
            if row["candidate_region"] is None:
                counts["unsupported_views"] += 1
            if row["role"] in source.EDITS:
                candidate, baseline = useful(arm[METHODS[0]]), useful(arm[METHODS[1]])
                counts["new_useful_vs_aligned"] += candidate and not baseline
                counts["lost_useful_vs_aligned"] += baseline and not candidate
                if candidate:
                    counts["correct_phase_in_useful_locations"] += tuple(row["candidate_region"]["phase"]) == DESIGNED_PHASE
        result[fmt]["paired"] = dict(counts)
    return result


def readout(output, runtime):
    points = _read(output / "points.json")
    panels = {name: summarize(_read(output / f"{name}-rows.json")) for name in PANELS}
    decision = {}
    for fmt in DELIVERIES:
        pairs = [(panels[name][fmt][METHODS[0]], panels[name][fmt][METHODS[1]]) for name in PANELS]
        decision[fmt] = (all(c["useful_locations"] >= b["useful_locations"] and c["control_flags"] <= b["control_flags"]
                             and c["control_groups_flagged"] <= b["control_groups_flagged"] for c, b in pairs)
                         and any(c["useful_locations"] > b["useful_locations"] for c, b in pairs))
    _write(output / "aggregate.json", {
        "schema": SCHEMA, "freeze_sha256": source.sha(output / "freeze.json"),
        "points_sha256": source.sha(output / "points.json"), "points": points,
        "confirmation_journal_sha256": {name: source.sha(output / f"{name}-rows.json") for name in PANELS},
        "development_control_views": 80, "confirmation_views": 336,
        "confirmation_source_groups": 24, "by_panel": panels, "pilot_advances": decision,
        "elapsed_s": round(runtime, 3), "authorship_confidence_contribution": 0,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fit", "confirm"))
    for name in ("development", "earlier", "later", "phase_comparison", "recompression_comparison", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    directories = (args.development, args.earlier, args.later)
    profile = lock(directories, args.phase_comparison, args.recompression_comparison)
    output = args.output
    if args.stage == "freeze":
        output.mkdir(mode=0o700)
        _write(output / "freeze.json", profile)
        return
    require(_read(output / "freeze.json") == profile, "Protocol, runtime or intake changed after freeze")
    cv2.setNumThreads(1)
    if args.stage == "fit":
        require(not any((output / name).exists() for name in ("points.json", "fit-receipt.json")),
                "Development points were already sealed")
        _write(output / "points.json", develop(args.development))
        _write(output / "fit-receipt.json", {"freeze_sha256": source.sha(output / "freeze.json"),
                                             "points_sha256": source.sha(output / "points.json")})
        print("Sealed JPEG review points", flush=True)
        return
    require(not any((output / name).exists() for name in ("earlier-rows.json", "later-rows.json", "aggregate.json")),
            "The confirmation was already started; use a fresh frozen run directory")
    seal = _read(output / "fit-receipt.json")
    require(seal == {"freeze_sha256": source.sha(output / "freeze.json"),
                     "points_sha256": source.sha(output / "points.json")}, "Development point receipt changed")
    points = _read(output / "points.json")
    start = time.monotonic()
    for name, directory in zip(PANELS, directories[1:], strict=True):
        prior = reference_rows(name, args.phase_comparison, args.recompression_comparison)
        rows = list(confirm(directory, points, prior))
        require({_key(row) for row in rows} == set(prior) and len(rows) == len(prior), "Exact view roster differs")
        _write(output / f"{name}-rows.json", rows)
        print(f"Verified and scored {len(rows)} {name} JPEG views", flush=True)
    readout(output, time.monotonic() - start)


if __name__ == "__main__":
    main()
