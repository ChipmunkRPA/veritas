"""Compare native regional masks with a reversible horizontal pixel transform."""

from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import __version__ as pillow_version

from app.detectors.image import regional, regional_support, splice
from scripts import image_active_png_transfer as active
from scripts import image_edge_spread_transfer as panel

SCHEMA = "chipmunk.image-regional-flip-transfer/1"
BASE_ARMS = ("native", "mirrored", "recompression")
ARMS = (*BASE_ARMS, "shared_pixels", "supported_components")
RULE = {
    "minimum_saved_component_overlap": 0.5,
    "minimum_ordinary_relative_reduction": 0.2,
    "minimum_native_useful_retention": 0.75,
    "minimum_additional_useful_over_recompression_per_panel": 1,
}


def combine(native: np.ndarray, aligned: np.ndarray) -> dict[str, np.ndarray]:
    if native.shape != aligned.shape or native.dtype != bool or aligned.dtype != bool or native.ndim != 2:
        raise ValueError("Need two boolean masks on the same source pixel grid")
    count, labels = cv2.connectedComponents(native.astype(np.uint8), connectivity=8)
    areas = np.bincount(labels.ravel(), minlength=count)
    matched = np.bincount(labels[aligned], minlength=count)
    supported = matched * 2 >= areas
    supported[0] = False
    return {"shared_pixels": native & aligned, "supported_components": supported[labels]}


def profile(earlier: Path, later: Path):
    for field in ("flickr_photo_id", "flickr_owner"):
        sets = [{str(row[field]) for row in panel.read(path / "plan.json")["cases"]} for path in (earlier, later)]
        if [len(s) for s in sets] != [12, 12] or sets[0] & sets[1]:
            raise ValueError("Need 24 distinct source photos and credited photographers")
    current_path, current_hash = active.weights(regional.MODEL_ID)
    older_path, older_hash = active.weights("tidal.splice_v30_1")
    source = (Path(__file__), Path(active.__file__), Path(panel.__file__), Path(panel.paired.__file__),
              Path(regional.__file__), Path(regional_support.__file__), Path(splice.__file__))
    freeze = {
        "schema": SCHEMA,
        "intakes": {name: {file: panel.sha(path / file) for file in ("plan.json", "result.json", "attempts.jsonl")}
                    for name, path in zip(panel.PANELS, (earlier, later), strict=True)},
        "code_sha256": {path.name: panel.sha(path) for path in source},
        "weights_sha256": {"native": current_hash, "recompression": older_hash},
        "runtime": {"torch": torch.__version__, "opencv": cv2.__version__, "numpy": np.__version__,
                    "pillow": pillow_version, "device": "cpu", "dtype": "float32", "torch_threads": 2},
        "method": {
            "delivery": "PNG",
            "transform": "Horizontal pixel reversal before inference; exact inverse on the retained native-pixel mask.",
            "arms": list(ARMS),
            "primary": "supported_components",
            "scope": "Sharp and feathered inserts plus six matched ordinary conditions on two source-exposed panels.",
            "useful": "At least .30 IoU and .50 recall on the exact original-pixel grid.",
            "rule": RULE,
            "confidence_contribution": 0,
        },
    }
    return freeze, current_path, older_path


def observe(folder: Path, name: str, output: Path, native, recompression):
    for case, sources, known in panel.paired.inputs(folder):
        for role, pixels in panel.native_variants(sources, case["box"]).items():
            truth = np.any(pixels != sources["original"], axis=2) if role in panel.POSITIVE else np.zeros_like(known)
            body, image = panel.paired.encoded(pixels, "png")
            reflected_pixels = np.ascontiguousarray(pixels[:, ::-1])
            reflected_body, reflected_image = panel.paired.encoded(reflected_pixels, "png")

            def read_current(data, decoded):
                result = native.predict(decoded)
                selection = regional_support.readout(result.raw_mask, result.candidate_eligibility["eligible"],
                                                     regional.PIXEL_THRESHOLD, data, np.asarray(decoded))
                return selection.source_mask(*decoded.size), result.geometry["input_float32_sha256"]

            original, original_input = read_current(body, image)
            reflected, reflected_input = read_current(reflected_body, reflected_image)
            mirrored = np.ascontiguousarray(reflected[:, ::-1])
            older_response = splice.predict(recompression, image)
            older_gate = np.count_nonzero(older_response > splice.PIXEL_THRESHOLD) >= splice.MINIMUM_ACTIVE_CELLS
            older = regional_support.readout(older_response, older_gate, splice.PIXEL_THRESHOLD, body, pixels)
            masks = {"native": original, "mirrored": mirrored, "recompression": older.source_mask(*image.size)}
            masks.update(combine(original, mirrored))
            proof = output / "masks" / f"{name}-{case['slot']:02d}-{role}.bin"
            packed = b"".join(np.packbits(masks[arm], axis=None).tobytes() for arm in BASE_ARMS)
            with proof.open("xb") as stream:
                stream.write(packed)
            yield {
                "panel": name, "slot": case["slot"], "generator": case["generator"], "role": role,
                "shape": list(truth.shape), "input_sha256": panel.paired.shared.digest(body),
                "reflected_sha256": panel.paired.shared.digest(reflected_body),
                "truth_sha256": panel.paired.shared.digest(truth.tobytes()),
                "native_model_input_sha256": original_input, "reflected_model_input_sha256": reflected_input,
                "mask_bytes_sha256": panel.paired.shared.digest(packed),
                "measurements": {arm: panel.scoring.measure(mask, truth) for arm, mask in masks.items()},
            }
        print(f"{name} source {case['slot']:02d} completed", flush=True)


def summarize(rows):
    if len(rows) != 192 or len({(r["panel"], r["slot"], r["role"]) for r in rows}) != 192:
        raise ValueError("Expected 192 distinct held views")
    results = {}
    for name in panel.PANELS:
        selected = [row for row in rows if row["panel"] == name]
        positives = [r for r in selected if r["role"] in panel.POSITIVE]
        ordinary = [r for r in selected if r["role"] in panel.CONTROLS]
        counts = {}
        for arm in ARMS:
            useful = [r for r in positives if panel.useful(r["measurements"][arm])]
            flagged = [r for r in ordinary if r["measurements"][arm]["reported"]]
            counts[arm] = {
                "useful_sharp": sum(r["role"] == "edited" for r in useful),
                "useful_feathered": sum(r["role"] == "feathered_edit" for r in useful),
                "useful": len(useful),
                "useful_by_editor": {editor: sum(r["generator"] == editor for r in useful)
                                     for editor in sorted({r["generator"] for r in positives})},
                "ordinary_flags": len(flagged),
                "ordinary_groups": len({r["slot"] for r in flagged}),
                "untouched_flags": sum(r["role"] == "original" for r in flagged),
                "ordinary_by_treatment": {role: sum(r["role"] == role for r in flagged) for role in panel.CONTROLS},
                "removed_native_useful": sum(panel.useful(r["measurements"]["native"])
                                             and not panel.useful(r["measurements"][arm]) for r in positives),
            }
        chosen, native, older = (counts[key] for key in ("supported_components", "native", "recompression"))
        passes = (chosen["useful"] >= native["useful"] * RULE["minimum_native_useful_retention"]
                  and chosen["useful"] >= older["useful"] + RULE["minimum_additional_useful_over_recompression_per_panel"]
                  and chosen["ordinary_flags"] <= native["ordinary_flags"] * (1 - RULE["minimum_ordinary_relative_reduction"]))
        results[name] = {"arms": counts, "passes": passes}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    freeze, native_path, older_path = profile(args.earlier, args.later)
    freeze_path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(freeze_path, freeze)
        return
    if freeze != panel.read(freeze_path):
        raise ValueError("Runtime or inputs differ from the frozen study")
    (args.output / "masks").mkdir()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    started = time.monotonic()
    native = regional.load_model(native_path)
    older = splice.load_model(older_path)
    rows = [row for name, folder in zip(panel.PANELS, (args.earlier, args.later), strict=True)
            for row in observe(folder, name, args.output, native, older)]
    journal = args.output / "rows.json"
    counts = summarize(rows)
    panel.write(journal, rows)
    panel.write(args.output / "result.json", {
        "schema": SCHEMA, "freeze_sha256": panel.sha(freeze_path), "rows_sha256": panel.sha(journal),
        "views": len(rows), "inference_calls": len(rows) * 3, "counts": counts,
        "elapsed_s": round(time.monotonic() - started, 3), "peak_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    })


if __name__ == "__main__":
    main()
