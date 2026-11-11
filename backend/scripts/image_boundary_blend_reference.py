"""Run the current regional routes on the frozen feathered-edit comparison."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from scripts import image_background_patch_score as paired
from scripts import image_boundary_blend_transfer as study
from scripts import image_flux_product_comparison_v288 as product
from scripts import image_relative_patch_transfer as shared

ROLES = ("feathered_edit", "feathered_processing", "local_contrast")


def bound_inputs(intake, candidate, fmt):
    receipt = json.loads((candidate / "result.json").read_bytes())
    journal = candidate / "rows.json"
    assert study.sha(journal) == receipt["journal_sha256"]
    locked = {(r["slot"], r["role"]): r for r in json.loads(journal.read_bytes()) if r["format"] == fmt and r["role"] in ROLES}
    for row, views, hard_truth in paired.inputs(intake):
        base, box = views["original"], row["box"]
        variants = (study.local(base, views["edited"], box, feather=True),
                    study.local(base, views["local_processing"], box, feather=True),
                    study.local(base, np.asarray(ImageEnhance.Contrast(Image.fromarray(base)).enhance(1.2)), box))
        for role, pixels in zip(ROLES, variants, strict=True):
            truth = np.any(base != pixels, axis=2) if role == "feathered_edit" else np.zeros_like(hard_truth)
            data, image = paired.encoded(pixels, fmt)
            recorded = locked.pop((row["slot"], role))
            assert shared.digest(data) == recorded["input_sha256"] and row["generator"] == recorded["generator"]
            assert int(truth.sum()) == recorded["measurements"]["boundary_prominence"]["target"]
            yield recorded, data, image, truth
    assert not locked


def main():
    import torch

    from app.detectors.image import fused, regional_support, splice
    from scripts.image_native_comparison_readout_v45 import source_mask

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("format", choices=paired.FORMATS)
    parser.add_argument("intake", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = shared.flux.ROOT
    png = args.format == "png"
    route = product.ROUTES[0 if png else 75]
    paths = [root / product.SPLICE] if png else [root / "models/weights/fused" / name for name in (fused.CHECKPOINT_FILE, fused.BACKBONE_FILE)]
    checkpoints = [study.sha(path) for path in paths]
    assert checkpoints == ([product.SPLICE_SHA] if png else [fused.CHECKPOINT_SHA256, fused.BACKBONE_SHA256])
    identity = {"schema": "chipmunk.image-boundary-blend-reference/1", "code_sha256": study.sha(Path(__file__)),
                "candidate_code_sha256": study.sha(Path(study.__file__)), "intake_sha256": study.sha(args.intake / "result.json"),
                "candidate_sha256": study.sha(args.candidate / "result.json"), "format": args.format,
                "route": route, "weights_sha256": checkpoints, "roles": list(ROLES)}
    if args.stage == "freeze":
        args.destination.mkdir(mode=0o700)
        study.write(args.destination / "freeze.json", identity)
        return
    assert identity == json.loads((args.destination / "freeze.json").read_bytes())
    assert png or torch.backends.mps.is_available()
    torch.set_num_threads(2)
    start = time.monotonic()
    model = splice.load_model(paths[0]) if png else fused.load_model(*paths, "mps")
    rows = []
    for record, data, image, truth in bound_inputs(args.intake, args.candidate, args.format):
        threshold = route["pixel_threshold"]
        if png:
            response = splice.predict(model, image)
            gate = np.count_nonzero(response > threshold) >= splice.MINIMUM_ACTIVE_CELLS
        else:
            prediction = model.predict(image)
            response, gate = prediction.raw_mask, prediction.candidate_eligibility["eligible"]
        pixels = np.asarray(image)
        mask = regional_support.readout(response, bool(gate), threshold, data, pixels).source_mask(image.width, image.height)
        assert np.array_equal(mask, source_mask(response, bool(gate), threshold, data, pixels, truth))
        rows.append({**{key: record[key] for key in ("slot", "role", "format", "generator", "input_sha256")},
                     "measurements": {"current": study.earlier.measure(mask, truth)}})
    assert len(rows) == 36
    study.write(args.destination / "rows.json", rows)
    summary = study.earlier.summarize([{**r, "delivery": r["format"], "role": "edited" if r["role"] == "feathered_edit" else r["role"]}
                                      for r in rows])
    study.write(args.destination / "result.json", {"schema": identity["schema"], "views": len(rows), "forwards": len(rows),
                "freeze_sha256": study.sha(args.destination / "freeze.json"), "journal_sha256": study.sha(args.destination / "rows.json"),
                "elapsed_s": round(time.monotonic() - start, 2), "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "results": summary})


if __name__ == "__main__":
    main()
