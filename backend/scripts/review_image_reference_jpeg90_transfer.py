"""Independently check JPEG delivery bytes, source coordinates and the held decision counts."""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from fractions import Fraction
from math import ceil, floor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.detectors.image import reference_crop_resize
from scripts import image_reference_jpeg90_transfer as trial


def mapped(region, crop, size):
    sx = Fraction(crop["w"], size[0])
    sy = Fraction(crop["h"], size[1])
    left, top = floor(region["x"] * sx), floor(region["y"] * sy)
    right = ceil((region["x"] + region["w"]) * sx)
    bottom = ceil((region["y"] + region["h"]) * sy)
    return {"x": crop["x"] + left, "y": crop["y"] + top, "w": right-left, "h": bottom-top}


def audit(run):
    lock = trial.read(run / "freeze.json")
    rows = trial.read(run / "sources.json")
    cases = trial.read(run / "cases.json")
    scored = trial.read(run / "result.json")
    assert all(trial.digest((run / path).read_bytes()) == expected for path, expected in lock["files"].items())
    assert len(rows) == len({r["author_key"] for r in rows}) == 24
    latents = [trial.master(row) for row in rows]
    tally, nuisance, checks = {}, {}, Counter()
    cases_and_results = iter(zip(cases, scored["outcomes"], strict=True))
    for slot, latent in enumerate(latents):
        for role, quality, before, after, truth, rect in trial.deliveries(slot, latent, latents[(slot+1) % len(rows)]):
            case, row = next(cases_and_results)
            assert (case["slot"], case["quality"], case["role"], case["crop"], case["size"]) == (
                slot, quality, role, rect, list(truth.shape[1::-1]))
            assert trial.digest(before) == case["earlier_sha256"] and trial.digest(after) == case["submitted_sha256"]
            assert before == (run / "cases" / case["earlier"]).read_bytes()
            assert after == (run / "cases" / case["submitted"]).read_bytes()
            assert trial.digest(truth.tobytes()) == case["truth_sha256"]
            assert np.array_equal(truth, np.load(run / "cases" / f'{case["submitted"]}.npy', allow_pickle=False))
            candidate, regions = row["candidate"], row["regions"]
            chosen = overlap = 0
            if candidate is not None:
                with Image.open(io.BytesIO(after)) as carrier:
                    expected, mask = reference_crop_resize.compare(trial.pixels(before), trial.pixels(after), carrier)
                    assert expected["status"] == "comparable"
                assert candidate == {key: expected["reference_crop_bounds"][key] for key in ("x", "y", "w", "h")}
                assert trial.digest(mask.tobytes()) == row["mask_sha256"] and expected["regions"] == regions
                chosen, overlap = int(np.count_nonzero(mask)), int(np.count_nonzero(mask & truth))
                assert sum(region["selected_pixels"] for region in regions) == chosen
                for region in regions:
                    assert region["reference_region"] == mapped(region, candidate, case["size"])
                    checks["mapped_native_regions"] += 1
                checks["verified_masks"] += 1
            else:
                assert not regions and row["mask_sha256"] is None
            denominator = int(np.count_nonzero(truth))
            useful = bool(denominator and overlap / denominator >= .5 and overlap / (chosen + denominator - overlap) >= .3)
            if candidate:
                assert row["metrics"] == {"pixels": chosen, "truth": denominator, "overlap": overlap, "useful": useful}
            exact = candidate is not None and candidate == rect
            cells = tally.setdefault(str(quality), {})
            cells.setdefault(role, Counter()).update(cases=1, old_comparable=row["baseline_status"] == "comparable",
                new_comparable=candidate is not None, exact=exact, wrong=candidate is not None and not exact,
                highlighted=bool(regions), useful=exact and useful)
            if role in {"unchanged", "independent_earlier_delivery", "prior_recompression", "global_contrast"} and regions:
                nuisance.setdefault(str(quality), set()).add(slot)
            checks["verified_deliveries"] += 1
    assert next(cases_and_results, None) is None
    for quality, cells in tally.items():
        current = scored["summary"][quality]
        assert cells == current["by_role"] and len(nuisance.get(quality, ())) == current["nuisance_groups"]
        passed = (cells["unchanged"]["exact"] >= 8 and not cells["unchanged"]["highlighted"]
                  and cells["feathered"]["useful"] >= 6 and sum(v["wrong"] for v in cells.values()) == 0
                  and sum(cells[role]["highlighted"] for role in ("unchanged", "independent_earlier_delivery",
                                                                "prior_recompression", "global_contrast")) <= 4
                  and current["nuisance_groups"] <= 3 and not cells["q75_regression"]["new_comparable"]
                  and not cells["q87_unchanged"]["new_comparable"])
        assert passed == current["passed"]
    return {"schema": lock["schema"], "design": lock["design"], "results": scored["summary"], "independent": checks,
            "freeze_sha256": trial.digest((run / "freeze.json").read_bytes()),
            "outcomes_sha256": trial.digest((run / "result.json").read_bytes()),
            "credits": [{"index": i, **row["credit"]} for i, row in enumerate(rows)]}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run", type=Path)
    cli.add_argument("receipt", type=Path)
    args = cli.parse_args()
    cv2.setNumThreads(1)
    summary = audit(args.run)
    trial.write(args.receipt, summary)
    print(json.dumps(summary["independent"]), flush=True)


if __name__ == "__main__":
    main()
