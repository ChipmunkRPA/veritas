# SPDX-License-Identifier: Apache-2.0
"""Repeat the JPEG90 quality-agreement rule on source-disjoint photographs."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

from app.detectors.image import jpeg_bitstream, regional
from scripts import image_jpeg_contour_gap_transfer as source
from scripts import image_regional_jpeg90_carrier as carrier
from scripts import image_regional_jpeg90_quality_agreement as agreement
from scripts import image_regional_jpeg_transfer as prior
from scripts.localization_benchmark import mask_metrics
from scripts.review_image_regional_jpeg90_quality_confirmation import DESIGN, SCHEMA, summary


def cases(args):
    exposed = [prior.read(path)["cases"] for path in (args.main_roster, args.external_roster)]
    blocked = {str(key).casefold() for rows in exposed for row in rows for key in (
        row["flickr_photo_id"], row["flickr_owner"], row["photo_attribution"]["Author"], row["original"]["sha256"],
    )}
    group = 0
    for item, native, exact in source.prior_groups(SimpleNamespace(earlier=args.confirmation_roster)):
        identity = source.identity(item, native, exact)
        keys = {identity["photo"], identity["owner"], identity["author_profile"], identity["credit"]["Author"],
                identity["original_sha256"]}
        keys = {str(value).casefold() for value in keys}
        prior.require(not blocked & keys, "Confirmation source or credited photographer overlaps")
        if max(exact.shape) != 768 or min(exact.shape) < 512:
            continue
        blocked.update(keys)
        group += 1
        controls, edits = source.common.authored(native, exact, item["recipient"])
        for role, pixels in (controls | edits).items():
            truth = np.any(pixels != native["original"], axis=2) if role in edits else np.zeros_like(exact)
            jpeg75 = source.common.encode(pixels, 75)
            jpeg90, native_image = prior.derived_view(Image.fromarray(pixels))
            _, baseline_image = decode(jpeg75)
            geometry = list(truth.shape)
            yield {"group": group, "family": item["generator"], "role": role, "geometry": geometry,
                   "q75_sha256": prior.sha(jpeg75), "q90_sha256": prior.sha(jpeg90),
                   "truth_sha256": prior.sha(truth.tobytes())}, jpeg75, baseline_image, jpeg90, native_image, truth
    prior.require(group == 11, "Eleven source-disjoint photographs required")


def decode(body):
    structure = jpeg_bitstream.read(body)
    prior.require(structure and structure.frame == "baseline" and structure.ijg_quality == 75,
                  "Expected a baseline quality-75 comparison image")
    with Image.open(io.BytesIO(body)) as decoded:
        return structure, decoded.convert("RGB")


def profile(args):
    roles = {"main": args.main_roster, "external": args.external_roster,
             "confirmation_plan": args.confirmation_roster / "plan.json",
             "confirmation_acquisition": args.confirmation_roster / "acquired.json",
             "earlier_quality_agreement_freeze": args.earlier_freeze}
    paths = (Path(__file__), Path(source.__file__), Path(agreement.__file__), Path(regional.__file__),
             Path(agreement.regional_support.__file__), Path(prior.__file__),
             Path(__file__).with_name("review_image_regional_jpeg90_quality_confirmation.py"))
    prior.require(prior.file_sha(prior.CHECKPOINT) == regional.CHECKPOINT_SHA256,
                  "Pinned regional weights differ")
    return {"schema": SCHEMA, "design": DESIGN, "checkpoint_sha256": regional.CHECKPOINT_SHA256,
            "code_sha256": {path.name: prior.file_sha(path) for path in paths},
            "source_sha256": {key: prior.file_sha(path) for key, path in roles.items()},
            "rows": [key for key, *_ in cases(args)]}


def evaluate(model, baseline, native, earlier_body, later_body, truth):
    first = prior.native_readout(model.predict(baseline), earlier_body, np.asarray(baseline))
    original = prior.native_readout(model.predict(native), later_body, np.asarray(native))
    view_body, view = agreement.delivered_view(native)
    repeat = prior.native_readout(model.predict(view), later_body, np.asarray(native))
    mask, trace = agreement.agreement(original, repeat)
    size = native.size
    arms = {"native_q75": first.source_mask(*size), "direct_q90": original.source_mask(*size),
            "projected_q75": repeat.source_mask(*size), "agreement": prior.project(mask, native)}
    return {"native_components": trace, "derived_sha256": prior.sha(view_body),
            "mask_sha256": {arm: prior.sha(result.tobytes()) for arm, result in arms.items()},
            "measures": {arm: carrier.short_metrics(mask_metrics(result, truth)) for arm, result in arms.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "score", "replay"))
    for name in ("main-roster", "external-roster", "confirmation-roster", "earlier-freeze", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    frozen = profile(args)
    lock = args.output / "freeze.json"
    if args.phase == "freeze":
        args.output.mkdir(mode=0o700)
        prior.write(lock, frozen)
        return
    prior.require(frozen == prior.read(lock), "Frozen selection, code or inputs differ")
    model = regional.load_model(prior.CHECKPOINT)
    rows = []
    for planned, case in zip(frozen["rows"], cases(args), strict=True):
        key, q75, first, q90, native, truth = case
        prior.require(key == planned, "Prepared JPEG or edit truth differs")
        rows.append(evaluate(model, first, native, q75, q90, truth))
        if len(rows) % 16 == 0:
            print(f"measured {len(rows)} confirmation deliveries", flush=True)
    value = {"freeze": frozen, "rows": rows, "summary": summary(frozen["rows"], rows)}
    receipt = args.output / "receipt.json"
    if args.phase == "score":
        prior.write(receipt, value)
    else:
        prior.require(value == prior.read(receipt), "Exact source, prediction or selected-mask replay differs")
    print(json.dumps(value["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
