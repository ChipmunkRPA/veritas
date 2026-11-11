"""Qualify new image groups and score the sealed detail-contrast points."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from scripts import image_background_patch_transfer as intake
from scripts import image_wavelet_region_transfer as method

SCHEMA = "chipmunk.image-wavelet-region-holdout/1"
POINTS_SHA256 = "5c93b3f4f4952cf63a93c0e6d51caf5b8254a9e9cef2fc967db54874dc73b31c"


def selection(args):
    keys, known, exposure = intake.match_exposure(args.calibration, args.fitting)
    receipt = json.loads((args.previous / "result.json").read_bytes())
    plan_path, journal_path = args.previous / "plan.json", args.previous / "attempts.jsonl"
    assert method.sha(plan_path) == receipt["plan_sha256"] and method.sha(journal_path) == receipt["attempts_sha256"]
    for case, line in zip(json.loads(plan_path.read_bytes())["cases"], journal_path.read_text().splitlines(), strict=True):
        keys.update(intake.roster.source_keys(case))
        row = json.loads(line)
        assert row["slot"] == case["slot"]
        if row["status"] == "qualified":
            fingerprint = row["fingerprint"]
            known[0].append(fingerprint)
            known[1].add(fingerprint["encoded_sha256"])
            known[2].add(fingerprint["rgb_sha256"])
    plan = intake.bindings(args.metadata, keys.copy()) | {"heldout_schema": SCHEMA, "exposure_sha256": exposure,
            "earlier_paired_qualification_sha256": method.sha(args.previous / "result.json"),
            "scoring_implementation_sha256": method.sha(Path(method.__file__)),
            "holdout_implementation_sha256": method.sha(Path(__file__)),
            "points_sha256": method.sha(args.points)}
    assert plan["points_sha256"] == POINTS_SHA256
    return plan, keys, known


def observations(folder, points):
    for row, native, sharp_truth in method.paired.inputs(folder):
        source, box = native["original"], row["box"]
        tone = np.asarray(ImageEnhance.Contrast(Image.fromarray(source)).enhance(1.2))
        views = native | {"feathered_edit": method.blend.local(source, native["edited"], box, feather=True),
                          "feathered_processing": method.blend.local(source, native["local_processing"], box, feather=True),
                          "local_contrast": method.blend.local(source, tone, box)}
        for role, pixels in views.items():
            if role in method.ROLES:
                truth = np.any(pixels != source, axis=2)
                assert truth.any() and not (truth & ~sharp_truth).any()
            else:
                truth = np.zeros_like(sharp_truth)
            for fmt in method.paired.FORMATS:
                body, image = method.paired.encoded(pixels, fmt)
                proposal = method.rank(image, np.asarray(points[fmt]["fit"]["weights"]))
                measures = {arm: method.scoring.measure(method.paired.boxed(found["box"], truth.shape,
                            found["score"] > points[fmt]["controls"][arm]["point"]), truth)
                            for arm, found in proposal.items()}
                yield {"slot": row["slot"], "generator": row["generator"], "role": role, "format": fmt,
                       "input_sha256": method.shared.digest(body), "truth_sha256": method.shared.digest(truth.tobytes()),
                       "measurements": measures, "proposals": proposal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "acquire", "run"))
    for key in ("metadata", "fitting", "calibration", "previous", "points", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    plan, keys, known = selection(args)
    plan_file = args.output / "plan.json"
    if args.stage == "freeze":
        args.output.mkdir(mode=0o700)
        method.write(plan_file, plan)
        print(plan["source_counts"], flush=True)
        return
    assert plan == json.loads(plan_file.read_bytes())
    if args.stage == "acquire":
        intake.acquire(plan, args.output, keys, known)
        return
    method.cv2.setNumThreads(1)
    rows = list(observations(args.output, json.loads(args.points.read_bytes())))
    assert rows and len(rows) == len({(row["slot"], row["role"], row["format"]) for row in rows})
    path = args.output / "outcomes.json"
    with path.open("x") as stream:
        json.dump(rows, stream, sort_keys=True, allow_nan=False)
    counts = Counter(row["generator"] for row in rows)
    assert all(count % 16 == 0 for count in counts.values())
    feathered = [{**row, "delivery": row["format"], "role": "edited"} for row in rows if row["role"] == "feathered_edit"]
    method.write(args.output / "score-result.json", {"schema": SCHEMA, "points_sha256": POINTS_SHA256,
                 "plan_sha256": method.sha(plan_file), "intake_sha256": method.sha(args.output / "result.json"),
                 "outcomes_sha256": method.sha(path), "views": len(rows), "views_by_generator": counts,
                 "sharp_and_controls": method.paired.summarise([r for r in rows if r["role"] != "feathered_edit"]),
                 "feathered": method.scoring.summarize(feathered)})


if __name__ == "__main__":
    main()
