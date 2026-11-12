"""Verify all recorded PNG deliveries and independently measure their exact evidence."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from scripts import image_edge_spread_transfer as panel
from scripts import image_png_exact_crop_resize as source

READBACK_KERNELS = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR,
                    "lanczos": Image.Resampling.LANCZOS, "opencv_area": None}


def render(reference, crop, size, label):
    x, y, width, height = (crop[k] for k in ("x", "y", "w", "h"))
    pixels = reference[y:y+height, x:x+width]
    kernel = READBACK_KERNELS[label]
    return (np.asarray(Image.fromarray(pixels).resize(size, kernel)) if kernel is not None
            else cv2.resize(pixels, size, interpolation=cv2.INTER_AREA))


def native_regions(mask, crop):
    labels, total = ndimage.label(mask, structure=np.ones((3, 3), bool))
    count = np.bincount(labels.ravel(), minlength=total + 1)
    boxes = ndimage.find_objects(labels)
    floor = max(48, round(mask.size * .0004))
    all_kept = sorted((i for i in range(1, total + 1) if count[i] >= floor), key=lambda i: (-count[i], i))
    keep = all_kept[:source.reference_compare.MAX_REGIONS]
    selection = np.isin(labels, keep)
    expected = []
    sh, sw = mask.shape
    for i in keep:
        y, x = boxes[i-1]
        rx = crop["x"] + x.start * crop["w"] // sw
        ry = crop["y"] + y.start * crop["h"] // sh
        end_x = crop["x"] + math.ceil(x.stop * crop["w"] / sw)
        end_y = crop["y"] + math.ceil(y.stop * crop["h"] / sh)
        expected.append({"x": x.start, "y": y.start, "w": x.stop-x.start, "h": y.stop-y.start,
                         "selected_pixels": int(count[i]), "reference_region": {"x": rx, "y": ry,
                                                                                "w": end_x-rx, "h": end_y-ry}})
    return selection, expected, len(all_kept) - len(keep)


def verify_proof(proof, reference, submitted):
    crop = proof["crop"]
    x, y, width, height = (crop[key] for key in ("x", "y", "w", "h"))
    sh, sw = submitted.shape[:2]
    rh, rw = reference.shape[:2]
    panel.require(all(type(i) is int for i in (x, y, width, height))
                  and 0 <= x < x + width <= rw and 0 <= y < y + height <= rh
                  and (x, y, width, height) != (0, 0, rw, rh) and min(width, height, sw, sh) >= 160
                  and width * height >= .55 * rw * rh and max(rh, rw) <= 768 and max(sh, sw) <= 845
                  and (width, height) != (sw, sh) and abs(sw*height/(sh*width) - 1) <= .008
                  and round(.72*width) <= sw <= round(1.17*width) and round(.72*height) <= sh <= round(1.17*height),
                  "Unsupported or misbound exact geometry")
    kernels = proof["kernels"]
    panel.require(0 < len(kernels) == len(set(kernels)) and set(kernels) <= READBACK_KERNELS.keys()
                  and 1 <= proof["proposed_rank"] <= 8, "Unknown resize kernel or proposal")
    projections = [render(reference, crop, (sw, sh), label) for label in kernels]
    projected = projections[0]
    panel.require(all((item == projected).all() for item in projections)
                  and panel.paired.shared.digest(projected.tobytes()) == proof["projected_sha256"], "Projection differs")
    mismatched = np.any(projected != submitted, axis=-1)
    equal = int(mismatched.size - np.count_nonzero(mismatched))
    panel.require(equal == proof["exact_pixels"] and 100*equal >= 65*mismatched.size, "Insufficient global exact support")
    supported = []
    for row, column in product(range(5), repeat=2):
        cy, cx = round((row+.5)/5*sh), round((column+.5)/5*sw)
        if mismatched[cy-10:cy+10, cx-10:cx+10].any():
            continue
        t = submitted[cy-10:cy+10, cx-10:cx+10].astype(np.int16)
        horizontal = (np.max(abs(t[:, :-1] - t[:, 1:]), axis=-1) >= 6).sum()
        vertical = (np.max(abs(t[:-1] - t[1:]), axis=-1) >= 6).sum()
        if horizontal + vertical >= .12 * 760:
            supported.append([row, column])
    panel.require(supported == proof["exact_tiles"] and len(supported) >= 12
                  and len({r for r, _ in supported}) >= 4 and len({c for _, c in supported}) >= 4
                  and all(any(r in a and c in b for r, c in supported) for a, b in product((range(2), range(3, 5)), repeat=2)),
                  "Exact, distributed texture proof differs")
    positions = np.where(mismatched)
    if len(positions[0]):
        area = math.prod(int(a.max()-a.min()+1) for a in positions)
        panel.require(5*area <= 2*mismatched.size, "Widespread differences cannot serve as local evidence")
    mask, regions, omitted = native_regions(mismatched, crop)
    panel.require(regions == proof["regions"] and omitted == proof["omitted"], "Native regions or their earlier bounds differ")
    return mask


def useful(target, reported, overlap):
    return bool(target and 10*overlap >= 3*(target+reported-overlap) and 2*overlap >= target)


def replay(views, saved):
    cells = {}
    nuisance = set()
    input_count, proof_count = 0, 0
    for data, row in zip(views, saved, strict=True):
        slot, family, reference, role, rgb, truth, expected, old, kernel = data
        body, carrier = panel.paired.encoded(rgb, "png")
        panel.require((slot, family, role, expected, kernel) == (row["slot"], row["family"], row["role"],
                      row["authored_crop"], row["authored_kernel"]), "Exact source order or authorship changed")
        panel.require(panel.paired.shared.digest(body) == row["exact_input"]
                      and panel.paired.shared.digest(truth.tobytes()) == row["truth_sha256"], "Input or truth hash changed")
        prior = {key: old[key] for key in ("crop", "candidate", "reference")} if old else None
        panel.require(prior == row["prior"], "Earlier comparison differs")
        proof = row["proof"]
        mask = verify_proof(proof, reference, np.asarray(carrier)) if proof is not None else np.zeros_like(truth)
        area, target, overlap = int(mask.sum()), int(truth.sum()), int(np.count_nonzero(mask & truth))
        panel.require((target, area, overlap) == tuple(row["measurement"][key] for key in ("target", "reported", "overlap"))
                      and panel.paired.shared.digest(mask.tobytes()) == row["mask_sha256"], "Exact native overlap differs")
        hit = useful(target, area, overlap)
        if prior is not None:
            prior_hit = useful(*(prior["reference"][key] for key in ("target", "reported", "overlap")))
        else:
            prior_hit = False
        cell = cells.setdefault(role, Counter())
        cell.update(planned=1, exact_crop=int(proof is not None and proof["crop"] == expected),
                    wrong_crop=int(proof is not None and proof["crop"] != expected),
                    localized=int(hit), added=int(hit and not prior_hit), highlighted=int(area > 0),
                    exact_kernel=int(proof is not None and kernel in proof["kernels"]))
        if role in ("untouched_crop", "prior_jpeg", "contrast") and area > 0:
            nuisance.add(slot)
        input_count += 1
        proof_count += int(proof is not None)
    outcome = {"roles": cells, "wrong_or_incompatible": sum(c["wrong_crop"] for c in cells.values()),
               "nuisance": sum(cells[r]["highlighted"] for r in ("untouched_crop", "prior_jpeg", "contrast")),
               "nuisance_groups": len(nuisance)}
    return outcome, input_count, proof_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("development", "earlier", "later", "prior", "commons", "excluded", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    run = args.output
    frozen = panel.read(run / "freeze.json")
    result = panel.read(run / "result.json")
    panel.require(frozen == source.profile(args) and result["freeze_sha256"] == panel.sha(run / "freeze.json"),
                  "Protocol or input commitments differ")
    expected = {"development": 80, "earlier": 120, "later": 120, "commons": 120}
    verified = {}
    for name, count in expected.items():
        views = (source.fresh(args.commons, frozen["commons"]) if name == "commons"
                 else source.exposed(getattr(args, name), args.prior, name))
        journal = run / f"{name}.json"
        panel.require(panel.sha(journal) == result["journals"][name], "A recorded journal differs")
        outcome, actual, proofs = replay(views, panel.read(journal))
        panel.require(actual == count and outcome == result["outcomes"][name], "Panel counts differ")
        verified[name] = {"inputs": actual, "exact_crop_proofs": proofs}
    held = [result["outcomes"][key] for key in ("earlier", "later")]
    exposed_pass = all(group["roles"]["feathered"]["added"] >= 4
                       and group["roles"]["untouched_crop"]["exact_crop"] >= 6
                       and group["wrong_or_incompatible"] == 0 and group["nuisance"] <= 4
                       and group["nuisance_groups"] <= 3 for group in held)
    fresh = result["outcomes"]["commons"]
    fresh_pass = (fresh["wrong_or_incompatible"] == fresh["nuisance"] == 0
                  and min(fresh["roles"][kind]["localized"] for kind in ("local_blur", "local_contrast")) >= 8
                  and fresh["roles"]["untouched_crop"]["exact_crop"] >= 8)
    checks = {"exposed_regression_passed": exposed_pass, "fresh_ordinary_passed": fresh_pass,
              "scope_qualifies": exposed_pass and fresh_pass}
    panel.require(all(result[key] == decision for key, decision in checks.items()), "Advancement decision differs")
    panel.write(run / "audit.json", {"panels": verified, "result_sha256": panel.sha(run / "result.json"), "checks": checks})
    print(verified, flush=True)


if __name__ == "__main__":
    main()
