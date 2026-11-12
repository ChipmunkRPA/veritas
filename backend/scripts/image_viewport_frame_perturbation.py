# SPDX-License-Identifier: Apache-2.0
"""Change only the frame around identical clean/edited native photo pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.image-viewport-frame-perturbation/1"
POINT = .8292734622955322
ARMS = ("base", "dark_rim", "light_rim", "texture_apron")
RIM = 16
DARK = (19, 25, 32)
LIGHT = (245, 249, 252)
SHAPE = (768, 768)
PANELS = {"discovery": 12, "transfer": 6}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def need(value, message):
    if not value:
        raise ValueError(message)


def area(box):
    x, y, r, b = box
    return (r-x)*(b-y)


def intersection(a, b):
    shared = math.prod(max(0, min(a[i+2], b[i+2])-max(a[i], b[i])) for i in (0, 1))
    return shared/(area(a)+area(b)-shared)


def score_sides(sides):
    return max(0., math.prod(sides.values()))**.25


def capture_pairs(args):
    import numpy as np

    from scripts import image_crop_layout_transfer as prior

    previous = read(args.previous/"result.json")
    need(previous["freeze_sha256"] == file_sha(args.previous/"freeze.json")
         and previous["rows_sha256"] == file_sha(args.previous/"rows.json"), "Previous native study receipts changed")
    rows = {tuple(row["key"]): row for row in read(args.previous/"rows.json")}
    for slot in range(12):
        receipt = rows["screen", slot, "sharp"]
        viewport = receipt["viewport"]
        x, y, right, bottom = viewport
        native = []
        for tag in ("photo", "sharp"):
            key = "screen", slot, tag
            path = args.previous/"captures"/f"{slot}-{tag}.png"
            need(file_sha(path) == rows[key]["input_sha256"], "Previous screenshot changed")
            rgb = prior.pixels(path)
            crop = prior.pixels(args.previous/"photo-fixtures"/f"{slot}-{tag}.png")
            need(rgb.shape == (*SHAPE, 3) and np.array_equal(rgb[y:bottom, x:right], crop), "Old native viewport differs")
            native.append(rgb)
        truth = (native[0] != native[1]).any(axis=2)
        need(sha(truth.tobytes()) == receipt["mask_sha256"], "Previous screenshot truth differs")
        yield "discovery", slot, "archived-screenshot", viewport, *native, receipt["whole"]


def transfer_pairs(args):
    import numpy as np

    from scripts import image_crop_layout_transfer as prior

    prior_seal = read(args.previous/"freeze.json")
    need(prior_seal["source_sha256"]["earlier"] == file_sha(args.earlier/"result.json")
         and prior_seal["source_sha256"]["later"] == file_sha(args.later/"result.json"), "Qualified source archive changed")
    archive = read(args.earlier/"plan.json")["cases"]
    alternate = read(args.later/"plan.json")["cases"]
    selection, former = [], []
    for name in ("sdxl", "kandinsky"):
        candidates = sorted((row for row in archive if row["generator"] == name),
                            key=lambda row: sha(f"viewport/1/{row['flickr_photo_id']}".encode()))
        need(len(candidates) == 6, "Expected each sealed generator family")
        former.extend(candidates[:3])
        selection.extend(candidates[3:])
    for key in ("flickr_photo_id", "flickr_owner"):
        held = {str(row[key]).casefold() for row in selection}
        known = {str(row[key]).casefold() for row in former+alternate}
        need(len(held) == 6 and not held&known, "Additional framing sources overlap the discovery roster")
    decoded = {case["slot"]: (images, mask) for case, images, mask in prior.REGIONAL.paired.inputs(args.earlier)}
    for slot, row in enumerate(selection):
        images, mask = decoded[row["slot"]]
        before, after = images["original"], images["edited"]
        truth = (before != after).any(axis=2)
        need(truth.any() and not (truth & ~mask).any(), "New photo differs from qualified edit mask")
        h, w = truth.shape
        a, b, right, bottom = prior.source_box(truth)
        pw, ph = min(w, 544), min(h, 528)
        x0 = min(max(0, (a+right-pw)//2), w-pw)
        y0 = min(max(0, (b+bottom-ph)//2), h-ph)
        need(x0 <= a < right <= x0+pw and y0 <= b < bottom <= y0+ph, "New viewport crop loses edited pixels")
        x, y = (86+11*(slot%3), 100+13*(slot//3))
        native = []
        for rgb in (before, after):
            canvas = np.full((*SHAPE, 3), (221, 231, 240), dtype=np.uint8)
            canvas[:44] = (53, 75, 92)
            canvas[y:y+ph,x:x+pw] = rgb[y0:y0+ph,x0:x0+pw]
            native.append(canvas)
        yield "transfer", slot, row["generator"], [x, y, x+pw, y+ph], *native, None


def versions(viewport, clean, edited):
    import numpy as np

    x, y, right, bottom = viewport
    core = np.s_[y:bottom, x:right]
    outer = np.s_[y-RIM:bottom+RIM,x-RIM:right+RIM]
    clean_photo = clean[core]
    apron = np.pad(clean_photo, ((RIM, RIM), (RIM, RIM), (0, 0)), mode="reflect")
    need(x >= RIM and y >= RIM and right+RIM <= SHAPE[1] and bottom+RIM <= SHAPE[0], "Frame perturbation exceeds input")
    truth = (clean != edited).any(axis=2)
    need(truth.any() and truth.sum() == truth[core].sum(), "Edits leave the preserved photo")
    yield "base", clean, edited, truth
    for name, fill in (("dark_rim", DARK), ("light_rim", LIGHT), ("texture_apron", apron)):
        twins = []
        for original in (clean, edited):
            altered = original.copy()
            altered[outer] = fill
            altered[core] = original[core]
            need(np.array_equal(altered[core], original[core]), "Framing changed a photo pixel")
            twins.append(altered)
        need(np.array_equal((twins[0]!=twins[1]).any(axis=2), truth), "Framing changed paired truth")
        yield name, *twins, truth


def sources(args):
    yield from capture_pairs(args)
    yield from transfer_pairs(args)


def bindings(pair):
    panel, slot, family, viewport, clean, edited, old = pair
    source_rgb = []
    for arm, a, b, truth in versions(viewport, clean, edited):
        source_rgb.append({"arm": arm, "clean_sha256": sha(a.tobytes()), "edited_sha256": sha(b.tobytes()),
                           "mask_sha256": sha(truth.tobytes()), "target": int(truth.sum())})
    return {"panel": panel, "slot": slot, "family": family, "viewport": viewport, "variants": source_rgb,
            "prior_box": old["box"] if old else None, "prior_score_hex": old["score"].hex() if old else None}


def freeze(args):
    from app.detectors.image import sharp_boundary

    need(sharp_boundary.REVIEW_POINT == POINT, "Existing point changed")
    cases = [bindings(pair) for pair in sources(args)]
    need(Counter(row["panel"] for row in cases) == PANELS, "Study panel denominator changed")
    return {"schema": SCHEMA, "point": POINT, "arms": list(ARMS), "rim": RIM, "dark": list(DARK), "light": list(LIGHT),
            "cases": cases, "source_sha256": {"previous": file_sha(args.previous/"result.json"),
                  "earlier": file_sha(args.earlier/"result.json"), "later": file_sha(args.later/"result.json")},
            "implementation_sha256": file_sha(Path(__file__)), "existing_check_sha256": file_sha(Path(sharp_boundary.__file__))}


def side_support(rgb, box):
    import numpy as np

    def side(source, line, low, high):
        segment = source[low:high]
        positions = (line-4,line-3,line-1,line,line+2,line+3)
        raw = np.stack([segment[:, i] if i >= 0 else np.zeros_like(segment[:,0]) for i in positions], axis=1)
        array = raw.astype(np.float64)
        before, current, after = (np.abs(array[:, i]-array[:, i+1]).mean(axis=1) for i in (0,2,4))
        if line == 3:
            before[:] = 0
        response = float(np.clip(np.log((current+3)/((before+after)/2+3)), 0, 3).mean())
        return response, sha(raw.tobytes())

    a, b, c, d = box
    flipped = rgb.transpose(1,0,2)
    data = {"left": side(rgb,a,b,d), "right": side(rgb,c,b,d),
            "top": side(flipped,b,a,c), "bottom": side(flipped,d,a,c)}
    values = {key: result[0] for key, result in data.items()}
    score = score_sides(values)
    return {"score": score, "flag": score > POINT, "sides": values, "pixels": {key: r[1] for key,r in data.items()}}


def winner(rgb, truth):
    from app.detectors.image import sharp_boundary

    measured = sharp_boundary.measure(rgb)
    a,b,c,d = box = list(measured.box)
    target = int(truth.sum())
    overlap = int(truth[b:d,a:c].sum())
    useful = bool(target and 2*overlap >= target and 10*overlap >= 3*(area(box)+target-overlap))
    return {"box": box, "score": measured.score, "flag": measured.score > POINT, "useful": useful,
            "overlap": overlap, "sides": measured.sides}


def score(pair):
    import numpy as np

    panel, slot, family, viewport, before, after, previous = pair
    rows = {}
    reference_box = None
    for name, clean, edited, truth in versions(viewport, before, after):
        ordinary = winner(clean, np.zeros(SHAPE, dtype=bool))
        edited_box = winner(edited, truth)
        if name == "base":
            reference_box = edited_box["box"]
            if previous:
                need(previous["box"] == reference_box and previous["score"] == edited_box["score"], "Prior selected result failed exact replay")
        check = side_support(edited, reference_box)
        if name == "base":
            need(abs(check["score"]-edited_box["score"]) <= 2e-5, "Fixed-box direct arithmetic differs")
        rows[name] = {"clean": ordinary, "edited": edited_box, "fixed_clean": side_support(clean, reference_box),
                      "fixed_edited": check, "target": int(truth.sum()), "selected_iou_with_base": intersection(reference_box, edited_box["box"])}
    return {"panel": panel, "slot": slot, "family": family, "viewport": viewport, "reference_box": reference_box, "arms": rows}


def aggregate(rows):
    output = {}
    for panel, denominator in PANELS.items():
        subset = [row for row in rows if row["panel"] == panel]
        need(len(subset) == denominator and {row["slot"] for row in subset} == set(range(denominator)), "Study groups changed")
        baseline = [row for row in subset if row["arms"]["base"]["edited"]["flag"] and row["arms"]["base"]["edited"]["useful"]]
        data = {}
        for arm in ARMS:
            c = Counter(groups=denominator, base_useful=len(baseline))
            for row in subset:
                current = row["arms"][arm]
                c.update(clean_flags=current["clean"]["flag"], edited_flags=current["edited"]["flag"],
                         edited_useful=current["edited"]["flag"] and current["edited"]["useful"],
                         fixed_clean_flags=current["fixed_clean"]["flag"], fixed_edited_flags=current["fixed_edited"]["flag"])
            for row in baseline:
                earlier, current = (row["arms"][name] for name in ("base",arm))
                c.update(retained_useful=current["edited"]["flag"] and current["edited"]["useful"],
                         same_selected_region=current["selected_iou_with_base"] >= .5,
                         retained_fixed_edited=current["fixed_edited"]["flag"], fixed_base_useful_clean_flags=current["fixed_clean"]["flag"],
                         changed_fixed_side_inputs=sum(current["fixed_edited"]["pixels"][key] != value
                               for key, value in earlier["fixed_edited"]["pixels"].items()))
            data[arm] = dict(c)
        output[panel] = data
    return output


def verify(record):
    need(record["schema"] == SCHEMA and record["point"] == POINT, "Portable study identity changed")
    for row in record["rows"]:
        need(set(row["arms"]) == set(ARMS), "Framing variants missing")
        for arm in row["arms"].values():
            need(arm["target"] > 0 and abs(arm["selected_iou_with_base"]-intersection(row["reference_box"],arm["edited"]["box"])) < 1e-12,
                 "Selected geometry changed")
            for label in ("clean", "edited", "fixed_clean", "fixed_edited"):
                item = arm[label]
                need(abs(score_sides(item["sides"])-item["score"]) <= 2e-5 and item["flag"] == (item["score"] > POINT),
                     "Boundary arithmetic or selection changed")
                if label in {"clean", "edited"}:
                    target = arm["target"] if label == "edited" else 0
                    count = item["overlap"]
                    need(0 <= count <= min(target, area(item["box"])) and item["useful"] ==
                         bool(target and count/target >= .5 and count/(target+area(item["box"])-count) >= .3), "Native overlap changed")
    need(aggregate(record["rows"]) == record["result"], "Aggregate changed")
    return record["result"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "review"))
    parser.add_argument("paths", nargs="+", type=Path)
    cli = parser.parse_args()
    if cli.stage == "review":
        path, = cli.paths
        print(json.dumps(verify(read(path)),sort_keys=True))
        return
    cli.previous, cli.earlier, cli.later, cli.run = cli.paths
    lock = freeze(cli)
    if cli.stage == "freeze":
        cli.run.mkdir(mode=0o700)
        write(cli.run/"freeze.json", lock)
        print(file_sha(cli.run/"freeze.json"), len(lock["cases"]), flush=True)
        return
    need(lock == read(cli.run/"freeze.json"), "Frozen inputs or code changed")
    started = time.monotonic()
    rows = []
    for pair, expected in zip(sources(cli), lock["cases"], strict=True):
        need(bindings(pair) == expected, "A committed framing input changed")
        rows.append(score(pair))
    record = {"schema": SCHEMA, "point": POINT, "freeze_sha256": file_sha(cli.run/"freeze.json"),
              "source_sha256": lock["source_sha256"], "implementation_sha256": lock["implementation_sha256"],
              "existing_check_sha256": lock["existing_check_sha256"], "arms": list(ARMS), "rim": RIM,
              "rows": rows, "result": aggregate(rows), "elapsed_s": time.monotonic()-started}
    verify(record)
    write(cli.run/"result.json", record)
    print(json.dumps(record["result"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
