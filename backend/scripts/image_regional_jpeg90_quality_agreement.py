# SPDX-License-Identifier: Apache-2.0
"""Check whether direct JPEG90 regions survive a geometry-preserving JPEG75 save."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.detectors.image import jpeg_bitstream, regional, regional_support
from scripts import image_regional_jpeg90_carrier as carrier
from scripts import image_regional_jpeg_transfer as prior
from scripts import image_regional_tile_rescue_transfer as tile
from scripts.localization_benchmark import mask_metrics
from scripts.review_image_regional_jpeg90_quality_agreement import DESIGN, SCHEMA, summary

ROOT = Path(__file__).resolve().parents[2]
DIRECT = ROOT / "docs/evaluation-panels/image-regional-jpeg90-direct-2026-11-13.json"


def delivered_view(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=75, subsampling="4:2:0", optimize=False, progressive=False)
    body = buffer.getvalue()
    with Image.open(io.BytesIO(body)) as decoded:
        return body, decoded.convert("RGB")


def agreement(native, repeated):
    """Only keep exact native components that retain half their cells."""
    selected = np.zeros_like(repeated.selected_grid_mask)
    trace = []
    for component in native.component_masks:
        count = int(np.count_nonzero(component))
        shared = int(np.count_nonzero(component & repeated.selected_grid_mask))
        trace.append([count, shared])
        if 2 * shared >= count:
            selected |= component
    return selected, trace


def frozen(args):
    source = prior.read(DIRECT)
    journal = args.tile_run / "rows.jsonl"
    require = tile.require
    require(prior.file_sha(prior.CHECKPOINT) == regional.CHECKPOINT_SHA256 == source["freeze"]["checkpoint_sha256"],
            "Pinned weights differ")
    require(source["freeze"]["source_journal_sha256"] == prior.file_sha(journal), "Original JPEG75 journal differs")
    modules = [Path(__file__), Path(carrier.__file__), Path(prior.__file__), Path(tile.__file__),
               Path(regional.__file__), Path(regional_support.__file__),
               Path(__file__).with_name("review_image_regional_jpeg90_quality_agreement.py")]
    return {"schema": SCHEMA, "design": DESIGN, "direct_receipt_sha256": prior.file_sha(DIRECT),
            "source_journal_sha256": prior.file_sha(journal), "checkpoint_sha256": regional.CHECKPOINT_SHA256,
            "code_sha256": {str(path.relative_to(ROOT)): prior.file_sha(path) for path in modules},
            "runtime": {"pytorch": str(torch.__version__), "numpy": np.__version__, "opencv": cv2.__version__}}


def rows(args):
    model = regional.load_model(prior.CHECKPOINT)
    history = prior.read(DIRECT)["rows"]
    recorded = [json.loads(line) for line in (args.tile_run / "rows.jsonl").read_text().splitlines()]
    groups = {}
    for index, (delivery, previous, journal) in enumerate(zip(carrier.deliveries(args), history, recorded, strict=True)):
        key, jpeg75, truth, pixels = delivery
        witness = tile.witness(key, jpeg75, truth)
        tile.require(all(previous[field] == value == journal[field] for field, value in witness.items()),
                     "Original delivery or source truth differs")
        jpeg90, native_image = prior.derived_view(Image.fromarray(pixels))
        tile.require(prior.sha(jpeg90) == previous["q90_sha256"], "Direct JPEG90 bytes differ")
        native_rgb = np.asarray(native_image)
        native_prediction = model.predict(native_image)
        native = prior.native_readout(native_prediction, jpeg90, native_rgb)
        native_mask = native.source_mask(*native_image.size)
        tile.require(prior.sha(native_mask.tobytes()) == previous["q90_mask_sha256"], "Native JPEG90 mask differs")

        jpeg75_view, view_image = delivered_view(native_image)
        structure = jpeg_bitstream.read(jpeg75_view)
        tile.require(structure and structure.ijg_quality == 75 and view_image.size == native_image.size,
                     "Unexpected view quality or geometry")
        view_prediction = model.predict(view_image)
        repeated = prior.native_readout(view_prediction, jpeg90, native_rgb)
        selected, support = agreement(native, repeated)
        masks = {"projected_q75": repeated.source_mask(*native_image.size),
                 "agreement": prior.project(selected, native_image)}
        group = key["group"]
        group_index = groups.setdefault(group, len(groups) + 1)
        yield {"source_index": index, "group_index": group_index, "native_components": support,
               "derived_sha256": prior.sha(jpeg75_view),
               "mask_sha256": {arm: prior.sha(mask.tobytes()) for arm, mask in masks.items()},
               "measures": {arm: carrier.short_metrics(mask_metrics(mask, truth)) for arm, mask in masks.items()}}
        if (index + 1) % 25 == 0:
            print(f"measured {index + 1} direct and repeated views", flush=True)
    tile.require(len(groups) == 22, "Incomplete two-bank source roster")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint",
                 "output-fresh", "external", "tile-run", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    args.direct = True
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    freeze = frozen(args)
    lock = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        prior.write(lock, freeze)
        return
    tile.require(prior.read(lock) == freeze, "Frozen method or prior rows changed")
    measured = list(rows(args))
    result = {"freeze": freeze, "rows": measured, "summary": summary(measured, prior.read(DIRECT)["rows"])}
    receipt = args.output / "receipt.json"
    if args.phase == "score":
        prior.write(receipt, result)
    else:
        tile.require(result == prior.read(receipt), "Exact re-encoding, mask or overlap replay differs")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
