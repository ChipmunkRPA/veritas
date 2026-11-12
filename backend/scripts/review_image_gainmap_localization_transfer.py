# SPDX-License-Identifier: Apache-2.0
"""Check gain-map pilot inputs, native rectangle overlaps and paired decisions."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from PIL import Image

from scripts import image_gainmap_localization_transfer as study


def overlap(mask, box):
    if mask is None or box is None:
        return False
    x, y, right, bottom = box
    height, width = mask.shape
    assert 0 <= x < right <= width and 0 <= y < bottom <= height
    area = int(mask.sum())
    hit = int(mask[y:bottom, x:right].sum())
    return bool(area and hit / area >= .50 and hit / ((right-x)*(bottom-y)+area-hit) >= .30)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive", type=Path)
    cli.add_argument("run", type=Path)
    args = cli.parse_args()
    frozen = (args.run / "freeze.json").read_bytes()
    committed = json.loads(frozen)
    assert committed == study.binding(args.archive)
    measured = json.loads((args.run / "held.json").read_bytes())
    cal = json.loads((args.run / "points.json").read_bytes())
    assert cal["freeze"] == study.sha(frozen) and len(cal["calibration"]) == 24 and len(measured["held"]) == 54
    for arm, point in cal["points"].items():
        assert point == max(row["scores"][arm]["score"] for row in cal["calibration"])
    records = {(row["slot"], row["role"]): row["scores"] for row in measured["held"]}
    reviewed, coordinates, without_thumbnail = 0, 0, 0
    union = {role: 0 for role in study.POS}
    ordinary, ordinary_groups = 0, set()
    for slot, role, data, mask in study.cases(args.archive):
        if (slot, role) not in records:
            continue
        scores = records[slot, role]
        decoded = study.split(data)
        assert (decoded is None) == (scores is None) == (role == "no_gainmap")
        with Image.open(io.BytesIO(data)) as main:
            thumb = study._pillow_thumbnail(main, main.getexif())
            assert (thumb is None) == (role != "untouched_camera")
            without_thumbnail += thumb is None
        reviewed += 1
        if scores is None:
            continue
        for score in scores.values():
            assert overlap(mask, score["box"]) == score["useful"]
            coordinates += score["box"] is not None
        above = [arm for arm, point in cal["points"].items() if scores[arm]["score"] > point]
        if role in study.POS:
            union[role] += any(scores[arm]["useful"] for arm in above)
        elif above:
            ordinary += 1
            ordinary_groups.add(slot)
    proof = {"reviewed_views": reviewed, "reviewed_native_rectangles": coordinates,
             "views_without_ifd1_thumbnail": without_thumbnail,
             "retrospective_union": {"useful": union, "ordinary_views_flagged": ordinary, "ordinary_groups_flagged": len(ordinary_groups)}}
    study.write(args.run / "audit.json", proof)
    print(proof, flush=True)


if __name__ == "__main__":
    main()
