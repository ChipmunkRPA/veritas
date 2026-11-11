"""Test a pinned inpainting segmenter on paired native edits and JPEG delivery."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from scripts import image_background_patch_score as paired
from scripts import image_bg_real_insertion_score as scoring
from scripts import image_boundary_blend_transfer as blend
from scripts import image_wavelet_region_transfer as wavelet

SCHEMA = "chipmunk.image-independent-segmenter-transfer/1"
REVISION = "f6cbed5c90a5b2a1f7919a1794ae50d1228ad0eb"
WEIGHTS_SHA256 = "574b34efa947905899f024cf5010a674ce45aa15e3fde63f58e56704ef97a13f"
SIZE = 512
PUBLISHER_POINT = 0.7
CONTRAST_FLOOR = 0.02
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
sha, write = wavelet.sha, wavelet.write


class Segmenter:
    def __init__(self, weights: Path):
        import segmentation_models_pytorch as smp
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        self.torch = torch
        self.model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
        # The released state contains both names of the same decoder block.
        self.model.decoder.block = torch.nn.Sequential(torch.nn.Dropout2d(0.3), self.model.decoder.block2)
        self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        self.model.eval()
        self.calls = 0

    def predict(self, image: Image.Image) -> np.ndarray:
        scaled = np.array(image.resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
        normalized = (scaled / np.float32(255) - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
        inputs = self.torch.from_numpy(normalized.transpose(2, 0, 1)).contiguous().unsqueeze(0)
        with self.torch.inference_mode():
            values = self.model(inputs).float().sigmoid()[0, 0].numpy()
        self.calls += 1
        if values.shape != (SIZE, SIZE) or not np.isfinite(values).all():
            raise ValueError("The checkpoint returned an invalid segmentation grid")
        return values


def local_contrast(values: np.ndarray) -> np.ndarray:
    inner = cv2.blur(values, (17, 17), borderType=cv2.BORDER_REFLECT)
    outer = cv2.blur(values, (97, 97), borderType=cv2.BORDER_REFLECT)
    area = 17 ** 2
    ring = (97 ** 2 * outer - area * inner) / (97 ** 2 - area)
    return np.maximum(0.0, inner - ring)


def _largest_supported(values: np.ndarray, point: float) -> np.ndarray:
    _, labels, stats, _ = cv2.connectedComponentsWithStats((values > point).astype(np.uint8), connectivity=8)
    candidates = [(float(values[labels == i].sum()), i) for i, item in enumerate(stats[1:], 1)
                  if 0.001 * values.size <= item[cv2.CC_STAT_AREA] <= 0.45 * values.size]
    return labels == max(candidates)[1] if candidates else np.zeros(values.shape, dtype=bool)


def native_masks(values: np.ndarray, response: np.ndarray, size: tuple[int, int], point: float) -> dict[str, np.ndarray]:
    quantized = Image.fromarray((255 * values).astype(np.uint8))
    raw = np.asarray(quantized.resize(size, Image.Resampling.BILINEAR)) / 255.0 > PUBLISHER_POINT
    component = Image.fromarray(_largest_supported(response, point).astype(np.uint8))
    local = np.asarray(component.resize(size, Image.Resampling.NEAREST), dtype=bool)
    return {"publisher_mask": raw, "local_contrast": local}


def _receipt(path: Path, filename: str, result_name: str, hash_key: str):
    file = path / filename
    expected = json.loads((path / result_name).read_bytes())[hash_key]
    if sha(file) != expected:
        raise ValueError(f"Prior journal changed: {filename}")
    return [json.loads(line) for line in file.read_text().splitlines()] if file.suffix == ".jsonl" else json.loads(file.read_bytes())


def _key(row):
    return row["slot"], row["role"], row["format"]


def _baseline(earlier: Path, boundary: Path):
    original = earlier.parent
    routes = (
        ("current_png", original / "scoring", "rows.jsonl", "results_sha256"),
        ("current_jpeg", original / "jpeg-reference", "rows.jsonl", "results_sha256"),
        ("current", boundary / "reference-png", "rows.json", "journal_sha256"),
        ("current", boundary / "reference-jpeg", "rows.json", "journal_sha256"),
    )
    baseline = {}
    for arm, directory, filename, key in routes:
        for row in _receipt(directory, filename, "result.json", key):
            if arm in row["measurements"]:
                assert _key(row) not in baseline
                baseline[_key(row)] = (row["input_sha256"], row["measurements"][arm])
    assert len(baseline) == 192
    return baseline


def _inputs(intake: Path, frozen_rows: list[dict]):
    locked = {_key(row): row for row in frozen_rows}
    assert len(locked) == 192
    for row, native, _ in paired.inputs(intake):
        original, box = native["original"], row["box"]
        tone = np.asarray(ImageEnhance.Contrast(Image.fromarray(original)).enhance(1.2))
        views = {**native, "feathered_edit": blend.local(original, native["edited"], box, feather=True),
                 "feathered_processing": blend.local(original, native["local_processing"], box, feather=True),
                 "local_contrast": blend.local(original, tone, box)}
        for role, pixels in views.items():
            truth = np.any(original != pixels, axis=2) if role in wavelet.ROLES else np.zeros(original.shape[:2], bool)
            truth_sha = wavelet.shared.digest(truth.tobytes())
            for fmt in paired.FORMATS:
                body, image = paired.encoded(pixels, fmt)
                item = locked.pop((row["slot"], role, fmt))
                assert item["input_sha256"] == wavelet.shared.digest(body) and item["generator"] == row["generator"]
                if "truth_sha256" in item:
                    assert item["truth_sha256"] == truth_sha
                else:
                    assert item["measurements"]["boundary_prominence"]["target"] == int(truth.sum())
                yield item, image, truth, truth_sha
    assert not locked


def fit(dev: Path, segmenter: Segmenter) -> dict:
    _, plan, attempts = scoring.checked_inputs(dev)
    maxima = {fmt: [] for fmt in paired.FORMATS}
    for _, _, views, _ in scoring.panels(dev, plan, attempts):
        pixels = views["original"]
        height, width = pixels.shape[:2]
        controls = wavelet.view_controls(pixels, views["processed"], (width // 3, height // 3, width * 7 // 12, height * 7 // 12))
        for fmt in paired.FORMATS:
            scores = [float(local_contrast(segmenter.predict(paired.encoded(control, fmt)[1])).max()) for control in controls]
            maxima[fmt].append(max(scores))
    assert all(len(values) == 8 for values in maxima.values())
    return {fmt: {"source_maxima": values, "point": max(CONTRAST_FLOOR, sorted(values)[-2])}
            for fmt, values in maxima.items()}


def confirmation(intake: Path, locked_rows: list[dict], segmenter: Segmenter, points: dict, baseline=None):
    for item, image, truth, truth_sha in _inputs(intake, locked_rows):
        values = segmenter.predict(image)
        contrast = local_contrast(values)
        maps = native_masks(values, contrast, image.size, points[item["format"]]["point"])
        measured = {arm: scoring.measure(mask, truth) for arm, mask in maps.items()}
        if baseline is not None:
            digest, measurement = baseline.pop(_key(item))
            assert digest == item["input_sha256"] and measurement["target"] == int(truth.sum())
            measured["current"] = measurement
        yield {**{key: item[key] for key in ("slot", "role", "format", "generator", "input_sha256")},
               "truth_sha256": truth_sha, "measurements": measured,
               "max_response": float(values.max()), "max_local_contrast": float(contrast.max())}
    assert not baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for name in ("weights", "development", "earlier", "earlier_boundary", "later", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if sha(args.weights) != WEIGHTS_SHA256:
        raise ValueError("The checkpoint SHA-256 differs from the pinned release")
    folders = (args.development, args.earlier, args.later)
    groups = [{case["flickr_photo_id"] for case in json.loads((p / "plan.json").read_bytes())["cases"]} for p in folders]
    assert [len(g) for g in groups] == [8, 12, 12] and len(set.union(*groups)) == 32
    receipts = {
        "development": args.development / "result.json", "earlier_intake": args.earlier / "result.json",
        "later_intake": args.later / "result.json", "later_inputs": args.later / "score-result.json",
        "earlier_inputs": args.earlier_boundary / "panel/result.json",
        "original_png": args.earlier.parent / "scoring/result.json", "original_jpeg": args.earlier.parent / "jpeg-reference/result.json",
        "added_png": args.earlier_boundary / "reference-png/result.json", "added_jpeg": args.earlier_boundary / "reference-jpeg/result.json",
    }
    dependencies = (Path(__file__), Path(paired.__file__), Path(scoring.__file__), Path(blend.__file__), Path(wavelet.__file__))
    freeze = {"schema": SCHEMA, "implementation_sha256": {p.name: sha(p) for p in dependencies},
              "receipts_sha256": {name: sha(p) for name, p in receipts.items()},
              "checkpoint_revision": REVISION, "checkpoint_sha256": WEIGHTS_SHA256, "map_size": SIZE,
              "publisher_point": PUBLISHER_POINT, "local_contrast_minimum": CONTRAST_FLOOR,
              "local_rings_px": [17, 97], "component_area_fraction": [0.001, 0.45],
              "localization_requirements": {"iou": 0.3, "recall": 0.5}}
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        write(args.output / "freeze.json", freeze)
        return
    assert freeze == json.loads((args.output / "freeze.json").read_bytes())
    cv2.setNumThreads(1)
    start = time.monotonic()
    segmenter = Segmenter(args.weights)
    points = fit(args.development, segmenter)
    assert segmenter.calls == 96
    write(args.output / "points.json", points)
    print("Both local contrast points sealed before confirmation", flush=True)
    panels = (
        ("earlier", args.earlier, _receipt(args.earlier_boundary / "panel", "rows.json", "result.json", "journal_sha256"),
         _baseline(args.earlier, args.earlier_boundary)),
        ("later", args.later, _receipt(args.later, "outcomes.json", "score-result.json", "outcomes_sha256"), None),
    )
    journals = {}
    for name, directory, locked, baseline in panels:
        rows = list(confirmation(directory, locked, segmenter, points, baseline))
        assert len(rows) == 192
        journal = args.output / f"{name}-rows.json"
        write(journal, rows)
        journals[name] = sha(journal)
        print(f"Completed {name} panel", flush=True)
    assert segmenter.calls == 480
    write(args.output / "result.json", {"schema": SCHEMA, "development_views": 96, "confirmation_views": 384,
          "actual_checkpoint_forwards": segmenter.calls, "elapsed_s": round(time.monotonic() - start, 3),
          "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "freeze_sha256": sha(args.output / "freeze.json"),
          "points_sha256": sha(args.output / "points.json"), "journal_sha256": journals})


if __name__ == "__main__":
    main()
