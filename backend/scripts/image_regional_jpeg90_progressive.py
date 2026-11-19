# SPDX-License-Identifier: Apache-2.0
"""Replay the quality-90 regional check across lossless progressive JPEG transport."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from scripts.image_jpeg90_agreement_holdout import comparison
from scripts.image_jpeg90_agreement_holdout import verify as verify_previous

SCHEMA = "chipmunk.jpeg90-progressive-transport/1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(command, data):
    return subprocess.run(command, input=data, capture_output=True, check=True, timeout=15).stdout


def rgb(payload):
    from PIL import Image

    with Image.open(io.BytesIO(payload)) as image:
        assert image.mode == "RGB" and image.getexif().get(274, 1) == 1
        return image.copy()


def measured_view(model, original, delivered):
    import numpy as np
    from PIL import Image

    from app.detectors.image import regional, regional_support
    from app.detectors.image.adapters.regional import repeat_components
    from app.evidence.image_eligibility import JPEG_REPEAT_CONSISTENCY
    from app.evidence.image_regional_jpeg import reencode

    pixels = np.asarray(original)
    repeated = rgb(reencode(original))
    native_prediction, repeat_prediction = [model.predict(image) for image in (original, repeated)]
    maps = [regional_support.readout(reading.raw_mask, reading.candidate_eligibility["eligible"],
                                     regional.PIXEL_THRESHOLD, delivered, pixels)
            for reading in (native_prediction, repeat_prediction)]
    native, repeat = maps
    retained, _, proof = repeat_components(native, repeat.selected_grid_mask, schema=JPEG_REPEAT_CONSISTENCY)
    supported_masks = [native.component_masks[index] for index, row in enumerate(proof["candidates"]) if row["retained"]]
    mask = np.logical_or.reduce(supported_masks) if supported_masks else np.zeros_like(native.selected_grid_mask)
    source = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(original.size, Image.Resampling.NEAREST)) > 0
    return {
        "input_tensor_sha256": [prediction.geometry["input_float32_sha256"]
                                for prediction in (native_prediction, repeat_prediction)],
        "response_sha256": [sha(prediction.raw_mask.tobytes()) for prediction in (native_prediction, repeat_prediction)],
        "regions": [row["region"]["bbox"] for row in retained],
        "mask_sha256": sha(source.tobytes()),
        "repeat_proof": proof,
    }


def witness_selection(rows):
    chosen = []
    for predicate in (
        lambda row: row["split"] == "fresh" and comparison.useful(row["agreement"]),
        lambda row: row["split"] == "fresh" and row["role"] == "ordinary" and row["agreement"] is not None,
        lambda row: row["split"] == "transfer" and row["role"] == "blur" and bool(row["agreement"] and row["agreement"]["selected"]),
    ):
        matches = [i for i, row in enumerate(rows) if predicate(row)]
        if not matches and len(chosen) == 2:
            matches = [i for i, row in enumerate(rows) if row["split"] == "transfer"
                       and row["role"] in {"ordinary", "tone", "copy"} and bool(row["agreement"] and row["agreement"]["selected"])]
        assert matches
        chosen.append(min(matches, key=lambda index: rows[index]["sha256"]))
    return chosen


def measure(args, baseline, baseline_sha256):
    import cv2
    import numpy as np
    import torch

    from app.detectors.image import jpeg_bitstream, localiser, regional
    from app.evidence.image_regional_jpeg import evaluated_jpeg, reencode

    jpegtran = shutil.which("jpegtran")
    djpeg = shutil.which("djpeg")
    assert jpegtran and djpeg, "Local libjpeg-turbo tools are required"
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    rows = baseline["rows"]
    witnesses = witness_selection(rows)
    inputs = {"fresh": args.fresh, "transfer": args.transfer}
    input_digests = {split: sha((path / "manifest.json").read_bytes()) for split, path in inputs.items()}
    assert input_digests == baseline["freeze"]["manifest_sha256"]
    freeze = {
        "baseline_receipt_sha256": baseline_sha256,
        "checkpoint_sha256": regional.CHECKPOINT_SHA256,
        "manifest_sha256": input_digests,
        "witness_ordinals": [i + 1 for i in witnesses],
    }

    checked = []
    model_inputs = {}
    for i, row in enumerate(rows):
        data = (inputs[row["split"]] / row["file"]).read_bytes()
        assert sha(data) == row["sha256"]
        old = jpeg_bitstream.read(data)
        assert old.frame == "baseline" and old.ijg_quality == 90
        assert evaluated_jpeg(data, quality=90) == (row["agreement"] is not None)
        progressive = run([jpegtran, "-copy", "all", "-progressive"], data)
        new, scans = jpeg_bitstream.progressive_frame(progressive)
        assert data != progressive and scans >= 2
        assert evaluated_jpeg(progressive, quality=90) == (row["agreement"] is not None)
        assert old.fingerprint == new.fingerprint and old.components == new.components
        assert old.quality_luma == new.quality_luma and (old.width, old.height) == (new.width, new.height)
        left, right = rgb(data), rgb(progressive)
        pixels_left, pixels_right = np.asarray(left), np.asarray(right)
        assert left.size == right.size and np.array_equal(pixels_left, pixels_right)
        standalone_left = run([djpeg, "-rgb", "-pnm"], data)
        standalone_right = run([djpeg, "-rgb", "-pnm"], progressive)
        assert standalone_left == standalone_right
        entry = {"ordinal": i + 1, "split": row["split"], "role": row["role"], "scans": scans,
                 "dimensions": list(left.size), "rgb_sha256": [sha(pixels_left.tobytes()), sha(pixels_right.tobytes())],
                 "independent_ppm_sha256": [sha(standalone_left), sha(standalone_right)],
                 "progressive_sha256": sha(progressive),
                 "old_assessed": row["agreement"] is not None}
        if not entry["old_assessed"]:
            assert min(left.size) < 512
        else:
            _, ga = regional.prepare(left)
            _, gb = regional.prepare(right)
            assert ga == gb
            quality_a = localiser.conditions_mask(pixels_left, old.quality_luma)
            quality_b = localiser.conditions_mask(pixels_right, new.quality_luma)
            assert quality_a[1] == quality_b[1] and np.array_equal(quality_a[0], quality_b[0])
            assert reencode(left) == reencode(right)
            assert sha(reencode(right)) == row["derived_sha256"]
            entry.update(input_tensor_sha256=[ga["input_float32_sha256"], gb["input_float32_sha256"]],
                         exclusion_mask_sha256=[sha(item[0].tobytes()) for item in (quality_a, quality_b)],
                         repeat_jpeg_sha256=[sha(reencode(image)) for image in (left, right)])
        if i in witnesses:
            model_inputs[i] = data, progressive
        checked.append(entry)
        if len(checked) % 72 == 0:
            print(f"Verified {len(checked)} coded pairs and their source conditions.", flush=True)

    model = regional.load_model(args.checkpoint)
    model_cases = []
    for index in witnesses:
        evaluations = [measured_view(model, rgb(data), data) for data in model_inputs[index]]
        assert evaluations[0] == evaluations[1]
        model_cases.append({"ordinal": index + 1, "evaluations": evaluations})
        print(f"Exact native effects and mask for witness {index + 1}.", flush=True)
    return {"schema": SCHEMA, "freeze": freeze, "rows": checked, "model_witnesses": model_cases}


def matching_digests(values):
    return (len(values) == 2 and values[0] == values[1]
            and len(values[0]) == 64 and all(char in "0123456789abcdef" for char in values[0]))


def review(record, baseline, baseline_sha256):
    freeze, rows = record["freeze"], record["rows"]
    previous = baseline["rows"]
    assert record["schema"] == SCHEMA
    assert freeze["baseline_receipt_sha256"] == baseline_sha256
    assert freeze["manifest_sha256"] == baseline["freeze"]["manifest_sha256"]
    assert freeze["checkpoint_sha256"] == baseline["freeze"]["weights_sha256"]
    assert len(rows) == len(previous) == 360
    tallies = {}
    for ordinal, (row, prior) in enumerate(zip(rows, previous, strict=True), 1):
        assert (row["ordinal"], row["split"], row["role"]) == (ordinal, prior["split"], prior["role"])
        assert 2 <= row["scans"] <= 16 and row["progressive_sha256"] != prior["sha256"]
        assert all(matching_digests(row[key]) for key in ("rgb_sha256", "independent_ppm_sha256"))
        admitted = prior["agreement"] is not None
        assert row["old_assessed"] is admitted
        assert (max(row["dimensions"]) == 768 and min(row["dimensions"]) >= 512) == admitted
        tally = tallies.setdefault(row["split"], Counter())
        positive = bool(prior["current"]["truth"])
        tally.update(deliveries=1, edits=int(positive), ordinary=int(not positive), assessed=int(admitted))
        if admitted:
            assert all(matching_digests(row[key]) for key in ("input_tensor_sha256", "exclusion_mask_sha256",
                                                            "repeat_jpeg_sha256"))
            assert row["repeat_jpeg_sha256"][0] == prior["derived_sha256"]
            tally.update(useful=int(comparison.useful(prior["agreement"])),
                         ordinary_flags=int(not positive and bool(prior["agreement"]["selected"])))
    expected = [index + 1 for index in witness_selection(previous)]
    assert freeze["witness_ordinals"] == expected
    assert [item["ordinal"] for item in record["model_witnesses"]] == expected
    for item in record["model_witnesses"]:
        ordinal = item["ordinal"]
        a, b = item["evaluations"]
        assert a == b and a["input_tensor_sha256"][0] == rows[ordinal - 1]["input_tensor_sha256"][0]
        prior = previous[ordinal - 1]
        assert b["mask_sha256"] == prior["mask_hashes"]["agreement"]
        assert [(row["source_pixels"], row["shared_pixels"]) for row in b["repeat_proof"]["candidates"]] == [
            tuple(pair) for pair in prior["component_overlap"]
        ]
    return {"two_decoders_exact": len(rows), "model_pairs_exact": len(expected),
            "previous_outcomes_preserved": {split: dict(tally) for split, tally in sorted(tallies.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path, help="Pinned quality-90 agreement receipt (.json.gz)")
    parser.add_argument("report", type=Path, help="New or existing progressive transport receipt (.json.gz)")
    parser.add_argument("--fresh", type=Path, help="Prepared fresh deliveries with their manifest")
    parser.add_argument("--transfer", type=Path, help="Prepared retained deliveries with their manifest")
    parser.add_argument("--checkpoint", type=Path, help="Pinned regional checkpoint")
    args = parser.parse_args()
    previous = args.previous.read_bytes()
    baseline = json.loads(gzip.decompress(previous))
    verify_previous(baseline)
    if args.fresh or args.transfer or args.checkpoint:
        if not (args.fresh and args.transfer and args.checkpoint) or args.report.exists():
            parser.error("Native replay needs both source roots, the checkpoint, and an unused output path")
        measured = measure(args, baseline, sha(previous))
        summary = review(measured, baseline, sha(previous))
        with args.report.open("xb") as output:
            output.write(gzip.compress(json.dumps(measured, sort_keys=True, allow_nan=False).encode(), mtime=0))
    else:
        summary = review(json.loads(gzip.decompress(args.report.read_bytes())), baseline, sha(previous))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
