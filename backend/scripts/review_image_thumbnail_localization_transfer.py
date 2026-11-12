# SPDX-License-Identifier: Apache-2.0
"""Replay embedded thumbnails and independently count native mask overlap."""

from __future__ import annotations

import numpy as np

from scripts import image_thumbnail_localization_transfer as study


def area_by_rectangles(mask, truth):
    if mask is None or truth is None:
        return {"selected_thumb": 0 if mask is None else int(mask.sum()), "selected_native": None,
                "native_overlap": 0, "useful": False}
    rows, cols = mask.shape
    height, width = truth.shape
    y, x = np.nonzero(mask)
    top, bottom = (y * height + rows - 1) // rows, ((y + 1) * height + rows - 1) // rows
    left, right = (x * width + cols - 1) // cols, ((x + 1) * width + cols - 1) // cols
    native = int(((bottom - top) * (right - left)).sum())
    exact = np.pad(truth.astype(np.int64), ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    overlap = int((exact[bottom, right] - exact[top, right] - exact[bottom, left] + exact[top, left]).sum())
    target = int(truth.sum())
    useful = bool(target and overlap / target >= .5 and overlap / (native + target - overlap) >= .3)
    return {"selected_thumb": len(y), "selected_native": native, "native_overlap": overlap, "useful": useful}


def verify(args, rows, receipt):
    readback = 0
    mapped = 0
    study.REQUIRE(study.file_sha(args.run / "held.json") == receipt["held_sha256"]
                  and study.file_sha(args.run / "freeze.json") == receipt["freeze_sha256"]
                  and study.summarize(rows, args.confirmation) == receipt["summary"], "Frozen receipt differs from the measurement ledger")
    for case, observed in zip(study.cases(args), rows, strict=True):
        key, body, truth, _ = case
        status, masks = study.candidate.measure(body, minimum_component_fraction=study.component_floor(args))
        study.REQUIRE(list(key) == observed["case"] and status["status"] == observed["result"]["status"], "Status replay differs")
        measures = (None, None) if masks is None else masks
        for arm, digest_field, mask in zip(("combined", "luma"), ("mask_sha256", "luma_mask_sha256"), measures, strict=True):
            digest = None if mask is None else study.sha(mask.tobytes())
            study.REQUIRE(digest == observed[digest_field] and area_by_rectangles(mask, truth) == observed[arm], "Native overlap replay differs")
            readback += 1
        if masks is None:
            continue
        tw, th = status["thumbnail_size"]
        nw, nh = status["native_size"]
        ox, oy = status["trim"][:2]
        for field, mask in zip(("regions", "luma_regions"), masks, strict=True):
            study.REQUIRE(status[field] == observed["result"][field], "Mapped boxes differ on replay")
            for region in status[field]:
                thumb, native = region["embedded_thumbnail"], region["submitted"]
                x, y = thumb["x"] - ox, thumb["y"] - oy
                w, h = thumb["w"], thumb["h"]
                left, top = x * nw // tw, y * nh // th
                right, bottom = ((x + w) * nw + tw - 1) // tw, ((y + h) * nh + th - 1) // th
                study.REQUIRE(0 <= x < x + w <= tw and 0 <= y < y + h <= th
                              and native == {"x": left, "y": top, "w": right - left, "h": bottom - top}
                              and region["selected_thumbnail_pixels"] <= int(mask[y:y+h, x:x+w].sum()), "Region exceeds the thumbnail geometry")
                mapped += 1
    return {"cases": len(rows), "mask_and_area_checks": readback, "mapped_regions": mapped}
