"""Measure the active PNG localizers on matched edits and ordinary controls."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from app.detectors.image import regional, regional_support, splice
from app.engine.fetchers import file_path
from app.engine.registry import get_registry
from scripts import image_color_residual_transfer as color
from scripts import image_independent_segmenter_transfer as paired
from scripts import image_wavelet_region_transfer as panel

SCHEMA = "chipmunk.image-active-png-transfer/1"


def weights(model_id):
    model = get_registry().manifest.model(model_id)
    checkpoint, = model.source.files
    path = file_path(model, checkpoint)
    digest = panel.sha(path)
    if digest != checkpoint.sha256:
        raise ValueError(f"Unexpected weight digest for {model_id}")
    return path, digest


def measurements(path, journal, native, recompression):
    for source, image, truth, truth_sha in paired._inputs(path, journal):
        if source["format"] != "png":
            continue
        data, _ = panel.paired.encoded(np.asarray(image), "png")
        assert panel.shared.digest(data) == source["input_sha256"]
        pixels = np.asarray(image)
        native_map = native.predict(image)
        fine = regional_support.readout(native_map.raw_mask, native_map.candidate_eligibility["eligible"],
                                       regional.PIXEL_THRESHOLD, data, pixels).source_mask(*image.size)
        older_map = splice.predict(recompression, image)
        older_gate = np.count_nonzero(older_map > splice.PIXEL_THRESHOLD) >= splice.MINIMUM_ACTIVE_CELLS
        older = regional_support.readout(older_map, older_gate, splice.PIXEL_THRESHOLD,
                                        data, pixels).source_mask(*image.size)
        arms = {"regional": fine, "recompression": older, "wider": fine | older, "shared": fine & older}
        yield {**{key: source[key] for key in ("slot", "role", "format", "generator", "input_sha256")},
               "truth_sha256": truth_sha, "regional_response_sha256": panel.shared.digest(native_map.raw_mask.tobytes()),
               "recompression_response_sha256": panel.shared.digest(older_map.tobytes()),
               "measurements": {arm: panel.scoring.measure(mask, truth) for arm, mask in arms.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("holdout", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    native_path, native_sha = weights(regional.MODEL_ID)
    prior_path, prior_sha = weights("tidal.splice_v30_1")
    roster = paired._receipt(args.holdout, "outcomes.json", "score-result.json", "outcomes_sha256")
    source_code = (Path(__file__), Path(regional.__file__), Path(splice.__file__), Path(regional_support.__file__),
                   Path(color.__file__), Path(paired.__file__), Path(panel.__file__))
    freeze = {"schema": SCHEMA, "intake_sha256": panel.sha(args.holdout / "result.json"),
              "journal_sha256": panel.sha(args.holdout / "outcomes.json"), "views": len(roster) // 2,
              "code_sha256": {p.name: panel.sha(p) for p in source_code}, "torch": torch.__version__,
              "device": "cpu", "dtype": "float32", "threads": 2,
              "weight_sha256": {"regional": native_sha, "recompression": prior_sha},
              "required_iou": 0.30, "required_recall": 0.50}
    path = args.output / "freeze.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        panel.write(path, freeze)
        return
    assert freeze == json.loads(path.read_bytes())
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    started = time.monotonic()
    rows = list(measurements(args.holdout, roster, regional.load_model(native_path), splice.load_model(prior_path)))
    assert len(rows) == 96 and len({row["slot"] for row in rows}) == 12
    journal = args.output / "rows.json"
    panel.write(journal, rows)
    panel.write(args.output / "result.json", {"schema": SCHEMA, "views": len(rows), "neural_forwards": len(rows) * 2,
          "freeze_sha256": panel.sha(path), "rows_sha256": panel.sha(journal), "counts": color.counts(rows),
          "elapsed_s": round(time.monotonic() - started, 3),
          "peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__ == "__main__":
    main()
