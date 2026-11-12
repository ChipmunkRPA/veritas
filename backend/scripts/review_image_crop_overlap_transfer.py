"""Rebuild crop inputs, rectangular proposals and held localization arithmetic."""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image

from scripts import image_crop_overlap_transfer as trial


@lru_cache(maxsize=32)
def layout(width, height):
    def coordinate(n, fraction):
        return -(-(n * fraction) // 24)

    boxes = [(coordinate(width, x), coordinate(height, y), coordinate(width, x + 8), coordinate(height, y + 8))
             for y, x in product(range(0, 17, 4), repeat=2)]
    recorded = {}
    for w, h in product((4, 6, 8), repeat=2):
        for y, x in product(range(0, 25 - h, 2), range(0, 25 - w, 2)):
            a, b, c, d = (coordinate(width, x), coordinate(height, y), coordinate(width, x + w), coordinate(height, y + h))
            total = (c - a) * (d - b)
            areas = [max(0, min(c, x1) - max(a, x0)) * max(0, min(d, y1) - max(b, y0)) for x0, y0, x1, y1 in boxes]
            if sum(value > 0 for value in areas) >= 3 and areas.count(0) >= 4 and 2 * max(areas) >= total:
                recorded[a, b, c, d] = np.array(areas, dtype=np.float64) / total
    designs = np.array([(1, (x0 + x1) / (2 * width), (y0 + y1) / (2 * height)) for x0, y0, x1, y1 in boxes])
    occupancy = np.stack(list(recorded.values())).T
    signatures = {}
    for arm, size in (("overlap_centered", 1), ("overlap_detrended", 3)):
        design = designs[:, :size]
        residual = occupancy - design @ np.linalg.lstsq(design, occupancy, rcond=None)[0]
        lengths = np.sqrt((residual * residual).sum(axis=0))
        signatures[arm] = np.divide(residual, lengths, out=np.zeros_like(residual), where=lengths > 1e-12)
    return boxes, list(recorded), signatures


def verify_row(row, image, truth, point):
    boxes, candidates, signatures = layout(image.width, image.height)
    hashes = [trial.source.digest(image.crop(box).resize((384, 384), Image.Resampling.BILINEAR).tobytes()) for box in boxes]
    assert hashes == row["crop_sha256"] and trial.digest_grid(truth) == row["truth_sha256"]
    logits = row["logits"]
    assert len(logits) == len(boxes) and np.isfinite(logits).all()
    peak = max(range(len(boxes)), key=logits.__getitem__)
    baseline = row["proposals"]["strongest_relative_crop"]
    assert baseline["box"] == list(boxes[peak]) and baseline["score"] == logits[peak] - sorted(logits)[12]
    for arm, signatures_by_box in signatures.items():
        responses = np.maximum(np.asarray(logits) @ signatures_by_box, 0)
        report = row["proposals"][arm]
        selected = candidates.index(tuple(report["box"]))
        assert abs(report["score"] - responses[selected]) < 1e-10
        assert abs(report["score"] - responses.max()) < 1e-10
    if point is None:
        return
    for arm, data in row["proposals"].items():
        x0, y0, x1, y1 = data["box"]
        report = row["measurements"][arm]
        selected = data["score"] > point[arm]
        overlap = sum(bool(pixel) for pixel in truth[y0:y1, x0:x1].flat) if selected else 0
        assert report == {"reported": (x1 - x0) * (y1 - y0) if selected else 0,
                          "target": int(truth.sum()), "overlap": overlap}


def audited_counts(rows):
    def detected(row, arm):
        result = row["measurements"][arm]
        overlap, target, reported = (result[key] for key in ("overlap", "target", "reported"))
        return target > 0 and overlap / (target + reported - overlap) >= .3 and overlap / target >= .5

    counts = {}
    for cohort, fmt in product(("earlier", "later"), ("png", "jpeg75")):
        cell = [row for row in rows if row["cohort"] == cohort and row["format"] == fmt]
        assert len(cell) == 96
        baseline = [row for row in cell if detected(row, "strongest_relative_crop")]
        table = {}
        for arm in trial.ARMS:
            hits = [row for row in cell if detected(row, arm)]
            flags = [row for row in cell if row["role"] not in ("edited", "feathered_edit") and row["measurements"][arm]["reported"]]
            table[arm] = {
                "sharp": sum(row["role"] == "edited" for row in hits),
                "feathered": sum(row["role"] == "feathered_edit" for row in hits),
                "ordinary_flags": len(flags),
                "untouched_flags": sum(row["role"] == "original" for row in flags),
                "added_feathered_groups": len({row["slot"] for row in hits if row["role"] == "feathered_edit" and row not in baseline}),
                "lost_baseline_locations": sum(row not in hits for row in baseline),
            }
        counts[f"{cohort}/{fmt}"] = table
    return counts


def audit(args):
    run = args.output
    result = trial.read(run / "result.json")
    assert trial.identity(args) == trial.read(run / "freeze.json")
    for path, key in (("held.jsonl", "held_sha256"), ("points.json", "points_sha256"), ("freeze.json", "freeze_sha256")):
        assert trial.source.file_digest(run / path) == result[key]
    points = trial.read(run / "points.json")
    development = trial.read(run / "development.json")
    held = [json.loads(line) for line in (run / "held.jsonl").read_text().splitlines()]
    for fmt, arm in product(trial.FORMATS, trial.ARMS):
        groups = {row["slot"] for row in development if row["format"] == fmt}
        maxima = sorted(max(row["proposals"][arm]["score"] for row in development
                            if row["slot"] == slot and row["format"] == fmt) for slot in groups)
        assert len(maxima) == 8 and maxima[-2] == points[fmt][arm]
    batches = {}
    prepared = 0
    for is_dev, rows in ((True, development), (False, held)):
        index = {(r["cohort"], r["slot"], r["role"], r["format"]): r for r in rows}
        for cohort, slot, editor, role, pixels, truth in trial.source_views(args, development=is_dev):
            for fmt in trial.FORMATS:
                row = index.pop((cohort, slot, role, fmt))
                encoded, image = trial.background.encoded(pixels, fmt)
                assert row["editor"] == editor and row["shape"] == list(truth.shape)
                assert row["input_sha256"] == trial.source.digest(encoded)
                verify_row(row, image, truth, None if is_dev else points[fmt])
                prepared += len(row["crop_sha256"])
                for start in range(0, len(row["crop_sha256"]), 5):
                    key = tuple(row["crop_sha256"][start:start + 5])
                    assert batches.setdefault(key, row["logits"][start:start + 5]) == row["logits"][start:start + 5]
        assert not index
    assert len(batches) == result["model_batches"] and prepared == 5 * (result["model_batches"] + result["cached_batches"])
    assert audited_counts(held) == result["counts"]
    advance = {}
    for fmt in trial.FORMATS:
        outcomes = [result["counts"][f"{panel}/{fmt}"] for panel in ("earlier", "later")]
        advance[fmt] = all(
            r["overlap_detrended"]["added_feathered_groups"] >= 2 and r["overlap_detrended"]["lost_baseline_locations"] <= 1
            and r["overlap_detrended"]["ordinary_flags"] <= r["strongest_relative_crop"]["ordinary_flags"] + 2
            and r["overlap_detrended"]["untouched_flags"] <= r["strongest_relative_crop"]["untouched_flags"] + 1
            for r in outcomes
        )
    return {"source_views": len(development) + len(held), "crop_inputs": prepared, "verified_unique_model_batches": len(batches),
            "native_region_measurements": len(held) * len(trial.ARMS), "advancement": advance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("development", "earlier", "later", "weight", "output"):
        parser.add_argument(key, type=Path)
    print(json.dumps(audit(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
