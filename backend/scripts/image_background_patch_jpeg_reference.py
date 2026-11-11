"""Run the current JPEG regional route on the frozen two-editor panel."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from scripts import image_background_patch_score as candidate
from scripts import image_bg_real_insertion_score as scoring
from scripts import image_flux_product_comparison_v288 as product
from scripts import image_relative_patch_transfer as shared


def run(args):
    import torch

    from app.detectors.image import fused, regional_support
    from scripts.image_native_comparison_readout_v45 import source_mask

    weights = shared.flux.ROOT / "models/weights/fused"
    identity = {"schema": "chipmunk.background-patch-jpeg-reference/1",
                "code_sha256": shared.file_digest(Path(__file__)),
                "intake_sha256": shared.file_digest(args.intake / "result.json"),
                "candidate_implementation_sha256": shared.file_digest(Path(candidate.__file__)),
                "checkpoint_sha256": shared.file_digest(weights / fused.CHECKPOINT_FILE),
                "backbone_sha256": shared.file_digest(weights / fused.BACKBONE_FILE),
                "route": product.ROUTES[75]}
    assert identity["checkpoint_sha256"] == fused.CHECKPOINT_SHA256
    assert identity["backbone_sha256"] == fused.BACKBONE_SHA256
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        shared.write(args.output / "freeze.json", identity)
        return
    assert identity == json.loads((args.output / "freeze.json").read_bytes())
    assert torch.backends.mps.is_available(), "Use the frozen local MPS reference route"
    torch.set_num_threads(2)
    begin = time.monotonic()
    model = fused.load_model(weights / fused.CHECKPOINT_FILE, weights / fused.BACKBONE_FILE, "mps")
    rows = []
    with (args.output / "rows.jsonl").open("x") as journal:
        for group, views, truth in candidate.inputs(args.intake):
            for role, pixels in views.items():
                body, image = candidate.encoded(pixels, "jpeg75")
                rgb = np.asarray(image)
                expected = truth if role == "edited" else np.zeros_like(truth)
                prediction = model.predict(image)
                gate = prediction.candidate_eligibility["eligible"]
                threshold = product.ROUTES[75]["pixel_threshold"]
                mask = regional_support.readout(prediction.raw_mask, gate, threshold, body, rgb).source_mask(
                    image.width, image.height)
                assert np.array_equal(mask, source_mask(prediction.raw_mask, gate, threshold, body, rgb, expected))
                row = {"slot": group["slot"], "generator": group["generator"], "role": role, "format": "jpeg75",
                       "input_sha256": shared.digest(body), "measurements": {"current_jpeg": scoring.measure(mask, expected)}}
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
                rows.append(row)
            print(f"Group {group['slot']} complete", flush=True)
    planned = sum(candidate.panel.roster.read(args.intake / "result.json")["qualified"].values()) * 5
    assert len(rows) == planned
    shared.write(args.output / "result.json", {
        "schema": identity["schema"], "views": len(rows), "model_forwards": len(rows),
        "freeze_sha256": shared.file_digest(args.output / "freeze.json"),
        "results_sha256": shared.file_digest(args.output / "rows.jsonl"),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "mps_driver_bytes_at_end": torch.mps.driver_allocated_memory(),
        "elapsed_s": round(time.monotonic() - begin, 3), "results": candidate.summarise(rows),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("intake", type=Path)
    parser.add_argument("output", type=Path)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
