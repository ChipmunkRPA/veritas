# SPDX-License-Identifier: Apache-2.0
"""Replay the published JPEG90 agreement rule against the serving JPEG90 path."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

from scripts import image_region_rank_compare as comparison
from scripts import image_region_rank_inputs as panel

SCHEMA = "chipmunk.image-jpeg90-agreement-holdout/1"
DESIGN = {
    "input": "Upright RGB baseline JPEG with exact IJG quality-90 tables and 8-bit 4:2:0 geometry, at 512–768 by 768 pixels.",
    "primary": "Preserve native components with at least half the grid pixels covered in a same-geometry JPEG75 re-encode.",
    "variants": "Direct native regions and re-encode regions alone. Both use the original JPEG90 conditions mask.",
    "outcomes": "A correct region has at least 30% intersection over union and 50% changed-pixel recall. Any ordinary highlight counts.",
    "weights": "Keep existing weights and operating point. No fit or change of the selection rule.",
}
ARMS = ("native", "reencoded", "agreement")


def summarize(rows):
    totals = {}
    for row in rows:
        positive = row["role"] in panel.POSITIVE
        for arm in ("current", *ARMS):
            metric = row[arm]
            tally = totals.setdefault(f"{row['split']}/{arm}", Counter())
            flagged = bool(metric and metric["selected"])
            useful = comparison.useful(metric)
            tally.update(edited=int(positive), ordinary=int(not positive), assessed=int(metric is not None),
                         correct=int(useful), misplaced=int(positive and flagged and not useful),
                         ordinary_flags=int(not positive and flagged))
    return {key: dict(tally) for key, tally in sorted(totals.items())}


def evaluate(retained, fresh, reference, output):
    import cv2
    import numpy as np
    import torch
    from PIL import Image

    from app.detectors.image import jpeg_bitstream, regional
    from app.detectors.image.adapters.regional import evaluated_jpeg
    from scripts import image_cross_quality_fit as inputs
    from scripts import image_regional_jpeg90_quality_agreement as rule
    from scripts import image_regional_jpeg_transfer as prior
    from scripts.image_region_rank_reference import intersections

    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    plan = {"schema": SCHEMA, "design": DESIGN,
            "code": {module.__name__: panel.sha(Path(module.__file__)) for module in (rule, prior, regional)},
            "weights_sha256": regional.CHECKPOINT_SHA256,
            "manifest_sha256": {"transfer": panel.sha(retained / "deliveries/manifest.json"),
                                "fresh": panel.sha(fresh / "manifest.json")}}
    lock = output.with_suffix(".freeze.json")
    if lock.exists() and panel.read(lock) != plan:
        raise ValueError("Comparison differs from the frozen inputs or rule")
    panel.save(lock, plan)
    model = regional.load_model(prior.CHECKPOINT)
    rows = []
    for split, media, journal in (("fresh", fresh, reference),
                                    ("transfer", retained / "deliveries", retained / "reference.jsonl")):
        _, current = comparison.incumbent(journal)
        for group, cases in inputs.images(panel.read(media / "manifest.json"), (split,)):
            for (role, quality), row in cases.items():
                if quality != 90:
                    continue
                data = (media / row["file"]).read_bytes()
                supplied = current[row["file"]]
                if prior.sha(data) != row["sha256"] or supplied["sha256"] != row["sha256"] or supplied["regional"]:
                    raise ValueError("JPEG90 source or serving comparison differs")
                entry = {key: row[key] for key in ("file", "sha256", "mask_sha256")}
                entry.update(split=split, group=group["rgb_sha256"], family=group["family"], role=role,
                             current=supplied["jpeg"], native=None, reencoded=None, agreement=None)
                with Image.open(io.BytesIO(data)) as source:
                    structure = jpeg_bitstream.read(data)
                    if (structure and structure.frame == "baseline" and evaluated_jpeg(data, quality=90)
                            and source.mode == "RGB" and source.getexif().get(274, 1) == 1):
                        image = source.copy()
                        truth = inputs.load_mask(media, row)
                        rgb = np.asarray(image)
                        native = prior.native_readout(model.predict(image), data, rgb)
                        derived, reencode = rule.delivered_view(image)
                        repeated = prior.native_readout(model.predict(reencode), data, rgb)
                        agreed, trace = rule.agreement(native, repeated)
                        masks = (native.source_mask(*image.size), repeated.source_mask(*image.size), prior.project(agreed, image))
                        entry.update(zip(ARMS, (intersections(mask, truth) for mask in masks), strict=True))
                        entry.update(derived_sha256=prior.sha(derived), component_overlap=trace,
                                     mask_hashes=dict(zip(ARMS, (prior.sha(mask.tobytes()) for mask in masks), strict=True)))
                rows.append(entry)
            if len(rows) % 36 == 0:
                print(f"Compared {len(rows)} delivered JPEG90 images.", flush=True)
    result = {"freeze": plan, "rows": rows, "summary": summarize(rows)}
    print(json.dumps(verify(result), sort_keys=True), flush=True)
    panel.save(output, result)


def verify(result):
    assert result["freeze"]["schema"] == SCHEMA and result["freeze"]["design"] == DESIGN
    rows = result["rows"]
    assert len({(row["split"], row["file"]) for row in rows}) == len(rows) == 360
    for row in rows:
        assert all(row[arm] is None for arm in ARMS) or all(row[arm] is not None for arm in ARMS)
        for arm in ("current", *ARMS):
            if metric := row[arm]:
                assert 0 <= metric["intersection"] <= min(metric["selected"], metric["truth"])
                assert bool(metric["truth"]) == (row["role"] in panel.POSITIVE)
        if row["agreement"] is not None:
            assert row["agreement"]["selected"] <= row["native"]["selected"]
            assert bool(row["agreement"]["selected"]) == any(2 * shared >= area for area, shared in row["component_overlap"])
    assert summarize(rows) == result["summary"]
    return result["summary"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    score = modes.add_parser("score")
    for item in ("retained", "fresh", "reference", "output"):
        score.add_argument(item, type=Path)
    check = modes.add_parser("verify")
    check.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.mode == "score":
        evaluate(args.retained, args.fresh, args.reference, args.output)
    else:
        from scripts.image_cross_quality_fit import open_receipt

        print(json.dumps(verify(open_receipt(args.receipt)), sort_keys=True))
