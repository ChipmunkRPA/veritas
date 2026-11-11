"""Replay the sealed sharp-boundary point on a second matched PNG source panel."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from scripts import image_background_patch_score as paired
from scripts import image_boundary_blend_transfer as method
from scripts.image_bg_real_insertion_score import measure

SCHEMA = "chipmunk.image-boundary-prominence-holdout/1"
ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "docs/evaluation-panels"
POSITIVES = {"edited", "feathered_edit"}
CONVENTIONAL = {"hard_same_image_paste", "hard_other_photo_paste"}


def read(path):
    return json.loads(path.read_bytes())


def frozen(args):
    previous = read(PUBLIC / "image-boundary-blend-transfer-2026-11-11.json")
    active = read(PUBLIC / "image-active-png-transfer-2026-11-11.json")
    assert active["freeze_sha256"] == method.sha(args.active / "freeze.json")
    assert active["rows_sha256"] == method.sha(args.active / "rows.json")
    assert active["frozen_configuration"]["intake_sha256"] == method.sha(args.intake / "result.json")
    assert method.sha(args.points) == previous["candidate"]["points_sha256"]
    return {
        "schema": SCHEMA,
        "implementation_sha256": method.sha(Path(__file__)),
        "fixed_method_sha256": method.sha(Path(method.__file__)),
        "points_sha256": method.sha(args.points),
        "png_point": read(args.points)["png"]["point"],
        "intake_sha256": method.sha(args.intake / "result.json"),
        "baseline_sha256": active["rows_sha256"],
        "format": "png",
        "frozen_criteria": {
            "minimum_iou": 0.30,
            "minimum_recall": 0.50,
            "minimum_sharp_locations": 4,
            "maximum_highlighted_ordinary_controls": 13,
        },
        "scope": "Bounded candidate for a spatial seam observation only. No AI-authorship direction or score.",
        "conventional_control_rule": (
            "Paste an exact unresized RGB patch into the same target rectangle from the first nonoverlapping corner "
            "of that original and from the center of the next original in qualified slot order. "
            "No synthetic foreground or blending is used in these two controls."
        ),
    }


def conventional_views(source, other, box):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    rows, cols = source.shape[:2]
    candidate = next(
        (x, y)
        for y in (4, rows - height - 4)
        for x in (4, cols - width - 4)
        if x + width <= left or right <= x or y + height <= top or bottom <= y
    )
    x, y = candidate
    ox, oy = (other.shape[1] - width) // 2, (other.shape[0] - height) // 2
    native = {}
    for name, donor in (
        ("hard_same_image_paste", source[y : y + height, x : x + width]),
        ("hard_other_photo_paste", other[oy : oy + height, ox : ox + width]),
    ):
        assert donor.shape == (height, width, 3)
        result = source.copy()
        result[top:bottom, left:right] = donor
        native[name] = result
    return native


def inputs(folder):
    cases = list(paired.inputs(folder))
    assert len(cases) == 12
    assert Counter(row["generator"] for row, _, _ in cases) == {"sdxl": 6, "kandinsky": 6}
    for index, (row, native, sharp_truth) in enumerate(cases):
        source, box = native["original"], row["box"]
        other = cases[(index + 1) % len(cases)][1]["original"]
        tone = np.asarray(ImageEnhance.Contrast(Image.fromarray(source)).enhance(1.2))
        variants = (
            native
            | {
                "feathered_edit": method.local(source, native["edited"], box, feather=True),
                "feathered_processing": method.local(source, native["local_processing"], box, feather=True),
                "local_contrast": method.local(source, tone, box),
            }
            | conventional_views(source, other, box)
        )
        for role, pixels in variants.items():
            truth = (
                np.any(pixels != source, axis=2)
                if role in POSITIVES | CONVENTIONAL
                else np.zeros_like(sharp_truth)
            )
            assert role not in POSITIVES | CONVENTIONAL or truth.any()
            assert role != "edited" or np.array_equal(truth, sharp_truth)
            payload, image = paired.encoded(pixels, "png")
            yield row["slot"], row["generator"], role, np.asarray(image), truth, method.shared.digest(payload)


def baseline_map(path):
    data = read(path)
    results = {(row["slot"], row["role"]): row for row in data}
    assert len(data) == len(results) == 96
    return results


def score(args, plan):
    baseline = baseline_map(args.active / "rows.json")
    for slot, generator, role, pixels, truth, checksum in inputs(args.intake):
        known = baseline.get((slot, role))
        truth_hash = method.shared.digest(truth.tobytes())
        if role not in CONVENTIONAL:
            assert (
                known is not None
                and checksum == known["input_sha256"]
                and truth_hash == known["truth_sha256"]
            )
        response, box = method.rank(pixels, "png")
        selected = response > plan["png_point"]
        measured = measure(paired.boxed(box, truth.shape, selected), truth)
        yield {
            "slot": slot,
            "role": role,
            "generator": generator,
            "input_sha256": checksum,
            "truth_sha256": truth_hash,
            "score": response,
            "box": box,
            "measurements": {"boundary": measured, **(known or {}).get("measurements", {})},
        }


def audit(args, plan):
    path = args.output / "rows.json"
    scored = read(path)
    baseline = baseline_map(args.active / "rows.json")
    counts = {}
    for captured, (slot, generator, role, _, truth, checksum) in zip(
        scored, inputs(args.intake), strict=True
    ):
        assert (captured["slot"], captured["generator"], captured["role"], captured["input_sha256"]) == (
            slot,
            generator,
            role,
            checksum,
        )
        assert captured["truth_sha256"] == method.shared.digest(truth.tobytes())
        left, top, right, bottom = captured["box"]
        target = int(truth.sum())
        selected = captured["score"] > plan["png_point"]
        reported = (right - left) * (bottom - top) if selected else 0
        overlap = int(truth[top:bottom, left:right].sum()) if selected else 0
        expected = {
            "reported": reported,
            "target": target,
            "overlap": overlap,
            "iou": overlap / (reported + target - overlap) if target else None,
            "recall": overlap / target if target else None,
        }
        assert expected == captured["measurements"]["boundary"]
        assert {k: v for k, v in captured["measurements"].items() if k != "boundary"} == baseline.get(
            (slot, role), {}
        ).get("measurements", {})
        for arm, measured in captured["measurements"].items():
            group = counts.setdefault(arm, {}).setdefault(
                role,
                {
                    "views": 0,
                    "highlighted": 0,
                    "useful_locations": 0,
                    "by_generator": {},
                },
            )
            for out in (
                group,
                group["by_generator"].setdefault(
                    generator, {"views": 0, "highlighted": 0, "useful_locations": 0}
                ),
            ):
                out["views"] += 1
                out["highlighted"] += measured["reported"] > 0
                out["useful_locations"] += bool(
                    measured["target"] and measured["iou"] >= 0.30 and measured["recall"] >= 0.50
                )
    assert len(scored) == 120
    ordinary_flags = sum(
        m["highlighted"] for role, m in counts["boundary"].items() if role not in POSITIVES | CONVENTIONAL
    )
    useful = counts["boundary"]["edited"]["useful_locations"]
    criteria = plan["frozen_criteria"]
    return {
        "schema": SCHEMA,
        "views": len(scored),
        "source_groups": 12,
        "actual_classifier_forwards": 0,
        "counts": counts,
        "ordinary_control_highlights": ordinary_flags,
        "independently_checked_inputs": 120,
        "meets_fixed_transfer_criteria": useful >= criteria["minimum_sharp_locations"]
        and ordinary_flags <= criteria["maximum_highlighted_ordinary_controls"],
        "frozen_configuration": plan,
        "freeze_sha256": method.sha(args.output / "freeze.json"),
        "rows_sha256": method.sha(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "score", "audit"))
    for name in ("intake", "points", "active", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    plan = frozen(args)
    if args.stage == "freeze":
        method.write(args.output / "freeze.json", plan)
        return
    assert plan == read(args.output / "freeze.json")
    if args.stage == "score":
        method.write(args.output / "rows.json", list(score(args, plan)))
    else:
        print(json.dumps(audit(args, plan), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
