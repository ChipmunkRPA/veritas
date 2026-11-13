"""Test JPEG90 delivery for the existing native regional image check."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from app.detectors.image import jpeg_bitstream, regional
from scripts import image_regional_jpeg_transfer as prior
from scripts import image_regional_tile_rescue_transfer as tile
from scripts.localization_benchmark import mask_metrics

SCHEMA = "chipmunk.image-regional-jpeg90-carrier/1"
DESIGN = {
    "sources": "Previously studied licensed camera images, exact known edited regions, two disjoint banks. Use every delivery in the earlier native JPEG75 trial.",
    "carrier": "Decode the pinned original JPEG75 as RGB and re-encode baseline IJG JPEG90 4:2:0 without resizing or progressive mode. This is one explicit tandem delivery workflow.",
    "comparison": "Use the existing regional checkpoint and fixed component, pixel and area points. Recompute the original quality-dependent reliability mask from the delivered bytes. Read the paired native JPEG75 masks from the earlier frozen panel.",
    "gate": "Keep at least eight of the 11 useful JPEG75 edit locations and one from each previously detected family, at most five additional ordinary files from at most two new source groups, and at most four additional wrong edited regions. Candidate adds zero AI confidence.",
}
DIRECT_DESIGN = DESIGN | {
    "carrier": "Encode each exact pre-delivery RGB view directly as baseline IJG JPEG90 4:2:0. Independently recreate its prior JPEG75 file and exact mask before inference.",
}


def plans(args):
    journal = args.tile_run / "rows.jsonl"
    freeze = args.tile_run / "freeze.json"
    tile.require(prior.file_sha(prior.CHECKPOINT) == regional.CHECKPOINT_SHA256, "Regional checkpoint differs")
    return {"schema": SCHEMA, "design": DIRECT_DESIGN if args.direct else DESIGN, "runner_sha256": prior.file_sha(Path(__file__)),
            "checkpoint_sha256": regional.CHECKPOINT_SHA256, "source_freeze_sha256": prior.file_sha(freeze),
            "source_journal_sha256": prior.file_sha(journal), "torch": str(torch.__version__)}


def short_metrics(full):
    return {key: full[key] for key in ("reference_positive_pixels", "predicted_positive_pixels", "intersection", "pixels")}


def useful(m):
    target, predicted, overlap = (m[key] for key in ("reference_positive_pixels", "predicted_positive_pixels", "intersection"))
    return bool(target and 2 * overlap >= target and 10 * overlap >= 3 * (target + predicted - overlap))


def summary(rows):
    counts = Counter()
    known_families = set()
    kept_families = set()
    old_ordinary = set()
    new_ordinary = set()
    for row in rows:
        old, new = row["q75"], row["q90"]
        if row["role"] in prior.POSITIVE:
            found_old, found_new = useful(old), useful(new)
            counts.update(edited=1, old_useful=found_old, new_useful=found_new, kept=found_old and found_new,
                          gained=found_new and not found_old, old_wrong=bool(old["predicted_positive_pixels"]) and not found_old,
                          new_wrong=bool(new["predicted_positive_pixels"]) and not found_new)
            if found_old:
                known_families.add(row["family"])
            if found_old and found_new:
                kept_families.add(row["family"])
        else:
            a, b = bool(old["predicted_positive_pixels"]), bool(new["predicted_positive_pixels"])
            counts.update(ordinary=1, old_flags=a, new_flags=b, added_flags=b and not a)
            if a:
                old_ordinary.add(row["group"])
            if b:
                new_ordinary.add(row["group"])
    counts["ordinary_groups_added"] = len(new_ordinary - old_ordinary)
    passed = (counts["old_useful"] == 11 and counts["kept"] >= 8 and kept_families == known_families
              and counts["added_flags"] <= 5 and counts["ordinary_groups_added"] <= 2
              and counts["new_wrong"] <= counts["old_wrong"] + 4)
    return dict(counts), sorted(kept_families), passed


def deliveries(args):
    if not args.direct:
        for key, jpeg75, truth, _ in tile.material(args):
            yield key, jpeg75, truth, tile.source_panel.common.decode(jpeg75)
        return
    from types import SimpleNamespace
    from unittest.mock import patch

    common = tile.source_panel.common
    originals = {}
    encode = common.encode

    def track(rgb, quality):
        body = encode(rgb, quality)
        if quality == 75:
            originals[prior.sha(body)] = rgb
        return body

    with patch.object(common, "encode", track):
        for key, jpeg75, truth, _ in tile.material(SimpleNamespace(**vars(args))):
            pixels = originals.pop(prior.sha(jpeg75), None)
            tile.require(pixels is not None, "Direct RGB delivery was not captured from the prior JPEG encoder")
            originals.clear()
            yield key, jpeg75, truth, pixels


def measure(args):
    model = regional.load_model(prior.CHECKPOINT)
    previous = [json.loads(line) for line in (args.tile_run / "rows.jsonl").read_text().splitlines()]
    rows = []
    for (key, jpeg75, truth, pixels), saved in zip(deliveries(args), previous, strict=True):
        witness = tile.witness(key, jpeg75, truth)
        tile.require(all(saved[field] == value for field, value in witness.items()), "Prior source or exact mask differs")
        jpeg90, image = prior.derived_view(Image.fromarray(pixels))
        if args.witness is not None and key["role"] == "feathered" and key["family"] == "sd15" and saved["measures"]["native"]["iou"] >= .30:
            if not args.witness.exists():
                args.witness.write_bytes(jpeg90)
        profile = jpeg_bitstream.read(jpeg90)
        tile.require(profile and profile.ijg_quality == 90 and profile.frame == "baseline" and
                     [(c.id, c.h, c.v) for c in profile.components] == [(1, 2, 2), (2, 1, 1), (3, 1, 1)], "Unexpected JPEG90 profile")
        prediction = model.predict(image)
        native = prior.native_readout(prediction, jpeg90, np.asarray(image)).source_mask(*image.size)
        rows.append(witness | {"q90_sha256": prior.sha(jpeg90), "q90_mask_sha256": prior.sha(native.tobytes()),
                               "q75": short_metrics(saved["measures"]["native"]), "q90": short_metrics(mask_metrics(native, truth))})
        if len(rows) % 25 == 0:
            print(f"measured {len(rows)}", flush=True)
    tile.require(len(rows) == 200 and len({row["group"] for row in rows}) == 22, "Missing paired JPEG sources")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay"))
    for name in ("archive", "prior-probe", "residual", "early-f5", "later-f5", "checkpoint", "output-fresh", "external", "tile-run", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--witness", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    freeze = plans(args)
    lock = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        prior.write(lock, freeze)
        return
    tile.require(prior.read(lock) == freeze, "Frozen method or delivery inputs differ")
    rows = measure(args)
    counts, families, passed = summary(rows)
    result = {"freeze": freeze, "rows": rows, "counts": counts, "kept_families": families, "passes": passed}
    if args.phase == "score":
        prior.write(args.output / "result.json", result)
    elif result != prior.read(args.output / "result.json"):
        raise ValueError("JPEG90 delivery, inference or exact mask replay differs")
    print(json.dumps({"counts": counts, "kept_families": families, "passes": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
