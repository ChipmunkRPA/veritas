# SPDX-License-Identifier: Apache-2.0
"""Reconstruct screenshot measurements using frozen native coordinates and masks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import find_objects, label

from scripts import image_reference_embedded_screenshot_transfer as study


def pixels(bounds):
    x, y, w, h = (bounds[key] for key in ("x", "y", "w", "h"))
    return np.s_[y:y+h, x:x+w]


def independent_mask(difference):
    components, count = label(difference, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(components.ravel(), minlength=count + 1)
    minimum = max(48, round(difference.size * .0004))
    retained = sorted((i for i in range(1, count + 1) if sizes[i] >= minimum), key=lambda i: (-sizes[i], i))[:5]
    objects = find_objects(components)
    return np.isin(components, retained), [(objects[i - 1], int(sizes[i])) for i in retained]


def native_case(case, row):
    key, _, earlier, _, screenshot, truth, (reference_rect, screen_rect) = case
    study.require(list(key) == row["key"] and int(truth.sum()) == row["target"], "Case order or exact truth changed")
    proof = row["proof"]
    if proof is None:
        study.require(not (row["selected"] or row["overlap"] or row["useful"] or row["mask_sha256"] or row["regions"]),
                      "An unassessed pair exported evidence")
        return 0
    study.require(key[-1] in study.COMPARABLE and row["exact_geometry"], "An accepted pair has no exact known alignment")
    study.require(proof["reference_bounds"] == reference_rect and proof["submitted_bounds"] == screen_rect,
                  "Reference and screenshot placement differ from the known capture")
    before, after = earlier[pixels(reference_rect)], screenshot[pixels(screen_rect)]
    difference = np.any(before != after, axis=2)
    height, width = difference.shape
    study.require(np.array_equal(difference, truth[pixels(screen_rect)]) and int((~difference).sum()) == proof["exact_pixels"]
                  and study.sha(difference.tobytes()) == proof["native_difference_sha256"], "Native difference proof differs")
    for r, c in proof["exact_tiles"]:
        x, y = round((c + .5) * width / 5), round((r + .5) * height / 5)
        study.require(not difference[y-10:y+10, x-10:x+10].any(), "An exact RGB witness contains different pixels")
    patch, selected = independent_mask(difference)
    mask = np.zeros(truth.shape, bool)
    mask[pixels(screen_rect)] = patch
    study.require(study.sha(mask.tobytes()) == row["mask_sha256"] and int(mask.sum()) == row["selected"]
                  and int((mask & truth).sum()) == row["overlap"], "Selected native pixels differ")
    measured = []
    for (y, x), count in selected:
        width, height = x.stop - x.start, y.stop - y.start
        measured.append((study.bounds(x.start + screen_rect["x"], y.start + screen_rect["y"], width, height),
                         study.bounds(x.start + reference_rect["x"], y.start + reference_rect["y"], width, height), count))
    observed = [({k: region[k] for k in ("x", "y", "w", "h")}, region["reference_region"], region["selected_pixels"])
                for region in row["regions"]]
    def canonical(region):
        before, after, area = region
        return tuple(box[key] for box in (before, after) for key in ("x", "y", "w", "h")) + (area,)

    study.require(sorted(map(canonical, measured)) == sorted(map(canonical, observed)), "Reported before/after regions differ")
    return len(proof["exact_tiles"])


def main():
    parser = study.parser()
    args = parser.parse_args()
    study.require(args.stage == "run", "Review takes the run stage and a retained result directory")
    frozen, held, result = (args.run / f"{name}.json" for name in ("freeze", "held", "result"))
    committed, rows, outcome = (study.read(path) for path in (frozen, held, result))
    study.require(outcome["freeze_sha256"] == study.file_sha(frozen) and outcome["held_sha256"] == study.file_sha(held)
                  and committed == study.freeze(args) and outcome["report"] == study.summarize(rows), "Study inputs or results differ")
    witnesses = sum(native_case(case, row) for case, row in zip(study.cases(args), rows, strict=True))
    receipt = {"reviewed_pairs": len(rows), "reviewed_matched_rectangles": sum(row["proof"] is not None for row in rows),
               "exact_tile_witnesses": witnesses, "result_sha256": study.file_sha(result)}
    study.write(args.run / "native-review.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
