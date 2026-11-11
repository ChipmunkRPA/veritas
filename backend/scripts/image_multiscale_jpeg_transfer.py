"""Compare local JPEG recompression curves against known insertions and ordinary edits."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import jpeg_dq
from scripts import image_bg_real_insertion_score as metrics
from scripts import image_local_grid_transfer as source

SCHEMA = "chipmunk.image-multiscale-jpeg-transfer/1"
QUALITIES = (45, 55, 65, 75, 85, 90, 95)
DELIVERIES = {"jpeg75": 75, "jpeg90": 90}
METHODS = ("single_quality", "multi_quality", "dq_corroborated")
CELL = 32


def _reconstruction_curves(pixels: np.ndarray) -> np.ndarray:
    rows, cols = (size // CELL for size in pixels.shape[:2])
    centered = pixels[:rows * CELL, :cols * CELL].astype(np.float32)
    curves = []
    original = Image.fromarray(pixels)
    for quality in QUALITIES:
        with io.BytesIO() as buffer:
            original.save(buffer, "JPEG", quality=quality, subsampling=2)
            buffer.seek(0)
            with Image.open(buffer) as result:
                recompressed = np.asarray(result.convert("RGB"))[:rows * CELL, :cols * CELL]
                delta = np.abs(centered - recompressed).mean(axis=2)
                curves.append(delta.reshape(rows, CELL, cols, CELL).mean(axis=(1, 3)))
    return np.stack(curves, axis=-1)


def _local(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner = cv2.blur(array, (3, 3), borderType=cv2.BORDER_REFLECT)
    surrounding = (81 * cv2.blur(array, (9, 9), borderType=cv2.BORDER_REFLECT) - 9 * inner) / 72
    return inner, surrounding


def response(pixels: np.ndarray) -> dict[str, np.ndarray]:
    curves = _reconstruction_curves(pixels)
    logs = np.log(curves + 0.35)
    shape = logs - logs.mean(axis=-1, keepdims=True)
    inner, ring = _local(shape)
    single, single_ring = _local(logs[:, :, QUALITIES.index(90)])
    texture, surrounding_texture = _local(curves[:, :, 0])
    supported = (texture >= 1.0) & (surrounding_texture >= 1.0)
    return {"single_quality": np.where(supported, np.abs(single - single_ring), 0.0),
            "multi_quality": np.where(supported, np.mean(np.abs(inner - ring), axis=2), 0.0)}


def _dq_mask(data: bytes, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if (measured := jpeg_dq.analyse(data)) is not None and (r := measured.localised) is not None:
        mask[8 * r.y:8 * (r.y + r.h), 8 * r.x:8 * (r.x + r.w)] = True
    return mask


def _expand(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    pixels = np.repeat(np.repeat(grid, CELL, axis=0), CELL, axis=1)
    mask[:pixels.shape[0], :pixels.shape[1]] = pixels
    return mask


def _bounded_map(values: np.ndarray, point: float, shape: tuple[int, int]) -> np.ndarray:
    _, labels, stats, _ = cv2.connectedComponentsWithStats((values > point).astype(np.uint8), connectivity=4)
    choices = [(float(values[labels == i].sum()), i) for i, row in enumerate(stats[1:], 1)
               if 2 <= row[cv2.CC_STAT_AREA] <= 0.20 * values.size]
    return _expand(labels == max(choices)[1], shape) if choices else np.zeros(shape, dtype=bool)


def _dq_support(values: np.ndarray, mask: np.ndarray) -> float:
    rows, cols = values.shape
    selected = mask[:rows * CELL, :cols * CELL].reshape(rows, CELL, cols, CELL).mean(axis=(1, 3)) >= 0.5
    return float(np.median(values[selected])) if np.count_nonzero(selected) >= 2 else 0.0


def _encoded(pixels, quality):
    body, image = source.delivery(pixels, quality)
    return body, response(np.asarray(image)), _dq_mask(body, pixels.shape[:2])


def fit(folder):
    _, plan, receipts = metrics.checked_inputs(folder)
    maxima = {fmt: {arm: [] for arm in METHODS} for fmt in DELIVERIES}
    for _, _, native, _ in metrics.panels(folder, plan, receipts):
        image = native["original"]
        h, w = image.shape[:2]
        controls = source.materials(image, image, (w // 3, h // 3, w * 7 // 12, h * 7 // 12))
        for fmt, quality in DELIVERIES.items():
            peaks = {arm: 0.0 for arm in METHODS}
            for role, pixels in controls.items():
                if role in source.EDITS:
                    continue
                _, maps, baseline = _encoded(pixels, quality)
                scores = {arm: float(values.max()) for arm, values in maps.items()}
                scores["dq_corroborated"] = _dq_support(maps["multi_quality"], baseline)
                for arm, value in scores.items():
                    peaks[arm] = max(peaks[arm], value)
            for arm in METHODS:
                maxima[fmt][arm].append(peaks[arm])
    return {fmt: {arm: {"source_maxima": group, "point": max(0.015, sorted(group)[-2])} for arm, group in data.items()}
            for fmt, data in maxima.items()}


def compare(intake, points):
    for case, native, _ in source.panel.inputs(intake):
        views = source.materials(native["original"], native["edited"], case["box"])
        for role, pixels in views.items():
            shape = pixels.shape[:2]
            target = np.any(pixels != views["original"], axis=2) if role in source.EDITS else np.zeros(shape, bool)
            for fmt, quality in DELIVERIES.items():
                body, maps, baseline = _encoded(pixels, quality)
                masks = {arm: _bounded_map(values, points[fmt][arm]["point"], shape) for arm, values in maps.items()}
                support = _dq_support(maps["multi_quality"], baseline)
                masks["dq_corroborated"] = baseline if support > points[fmt]["dq_corroborated"]["point"] else np.zeros(shape, bool)
                masks["existing_aligned_grid"] = baseline
                yield {"slot": case["slot"], "role": role, "format": fmt, "input_sha256": source.shared.digest(body),
                       "truth_sha256": source.shared.digest(target.tobytes()), "support": support,
                       "measurements": {arm: metrics.measure(mask, target) for arm, mask in masks.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for name in ("development", "earlier", "later", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    panels = (args.development, args.earlier, args.later)
    groups = [{case["flickr_photo_id"] for case in json.loads((path / "plan.json").read_bytes())["cases"]} for path in panels]
    assert [len(g) for g in groups] == [8, 12, 12] and len(set.union(*groups)) == 32
    dependencies = (Path(__file__), Path(source.__file__), Path(jpeg_dq.__file__))
    freeze = {"schema": SCHEMA, "implementation_sha256": {p.name: source.sha(p) for p in dependencies},
              "source_receipts_sha256": [source.sha(p / "result.json") for p in panels],
              "qualities": list(QUALITIES), "deliveries": DELIVERIES, "roles": list(source.ROLES), "cell_px": CELL}
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        source.write(args.output / "freeze.json", freeze)
        return
    assert freeze == json.loads((args.output / "freeze.json").read_bytes())
    start = time.monotonic()
    cv2.setNumThreads(1)
    points = fit(args.development)
    assert all(len(item["source_maxima"]) == 8 for method in points.values() for item in method.values())
    source.write(args.output / "points.json", points)
    print("Sealed recompression review points", flush=True)
    journals = {}
    for name, path in (("earlier", args.earlier), ("later", args.later)):
        rows = list(compare(path, points))
        assert len(rows) == 168
        journal = args.output / f"{name}-rows.json"
        source.write(journal, rows)
        journals[name] = source.sha(journal)
        print(f"Completed {name} panel", flush=True)
    source.write(args.output / "result.json", {"schema": SCHEMA, "development_views": 80, "confirmation_views": 336,
                 "freeze_sha256": source.sha(args.output / "freeze.json"), "points_sha256": source.sha(args.output / "points.json"),
                 "journal_sha256": journals, "elapsed_s": round(time.monotonic() - start, 3)})


if __name__ == "__main__":
    main()
