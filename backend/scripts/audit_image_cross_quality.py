# SPDX-License-Identifier: Apache-2.0
"""Read back the native image bytes, mask geometry and evaluation independence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import audit_image_region_rank as independent
from scripts import image_cross_quality_fit as study
from scripts import image_region_rank_inputs as panel


def audit_inputs(retained, fresh):
    from PIL import Image

    signatures, valid_images, valid_masks = [], 0, 0
    for root in (retained, fresh):
        manifest = panel.read(root / "manifest.json")
        groups = {row["tag"]: row for row in manifest["groups"]}
        seen_masks = set()
        ordinary = {}
        for row in manifest["images"]:
            body, mask = (root / row[name] for name in ("file", "mask"))
            assert panel.sha(body) == row["sha256"] and panel.sha(mask) == row["mask_sha256"]
            with Image.open(body, formats=["JPEG"]) as encoded, Image.open(mask, formats=["PNG"]) as truth:
                shape = groups[row["group"]]["shape"]
                assert list(encoded.size[::-1]) == shape and encoded.size == truth.size and encoded.mode == "RGB"
                bins = truth.convert("L").histogram()
                assert bins[255] == row["changed_pixels"] and not sum(bins[1:255])
            seen_masks.add(row["mask"])
            if row["quality"] == 90 and row["role"] == "ordinary":
                ordinary[row["group"]] = independent.signature(body)
            valid_images += 1
        assert ordinary.keys() == groups.keys()
        signatures.append(ordinary)
        valid_masks += len(seen_masks)
    distances = [(a ^ b).bit_count() for a in signatures[0].values() for b in signatures[1].values()]
    if min(distances) <= 16:
        raise ValueError("A fresh source could match a retained photo")
    return {"verified_deliveries": valid_images, "verified_truth_maps": valid_masks,
            "cross_archive_photo_pairs": len(distances), "closest_thumbnail_distance": min(distances)}


def audit_regions(packet, retained, fresh):
    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.image import jpeg_cross_quality as jpeg

    study.verify(packet)
    bank = {"transfer": retained, "fresh": fresh}
    cv2.setNumThreads(1)
    models = packet["freeze"]["heads"]
    checks, model_outputs = 0, 0
    for root in bank.values():
        manifest = panel.read(root / "manifest.json")
        contents = {row["file"]: row for row in manifest["images"]}
        for row in packet["rows"]:
            if bank[row["split"]] != root:
                continue
            expected = contents[row["file"]]
            assert expected["sha256"] == row["sha256"] and expected["mask_sha256"] == row["mask_sha256"]
            feature = study.read_features(root, expected)
            with Image.open(root / expected["mask"], formats=["PNG"]) as bitmap:
                truth = bitmap.convert("L")
                for arm, result in row["checks"].items():
                    model = models[arm][str(row["quality"])] if arm == "separate" else models[arm]
                    score, cells = jpeg.region(feature, model)
                    assert abs(score - result["score"]) <= 1e-6
                    coordinates = np.argwhere(cells).tolist()
                    area = sum(truth.crop((x * jpeg.CELL, y * jpeg.CELL, (x + 1) * jpeg.CELL, (y + 1) * jpeg.CELL)).histogram()[255]
                               for y, x in coordinates)
                    assert area == result["intersection"] and len(coordinates) * jpeg.CELL**2 == result["selected"]
                    height, width = cells.shape
                    native = np.broadcast_to(cells[:, None, :, None], (height, jpeg.CELL, width, jpeg.CELL))
                    assert hashlib.sha256(native.tobytes()).hexdigest() == result["pixel_sha256"]
                    model_outputs += 1
            checks += 1
    assert checks == len(packet["rows"])
    return {"holdout_deliveries_read_back": checks, "native_crop_intersections": model_outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("retained", "fresh", "receipt", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    result = audit_inputs(args.retained, args.fresh)
    result |= audit_regions(panel.read(args.receipt), args.retained, args.fresh)
    panel.save(args.output, result)
    print(json.dumps(result, sort_keys=True))
