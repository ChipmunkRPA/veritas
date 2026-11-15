# SPDX-License-Identifier: Apache-2.0
"""Measure source-linked differences in lossy WebP on a pinned photo bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.image_small_region_repeat import read, require, save, sha, useful

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/evaluation-panels/image-edge-context-new-sources-2026-11-14.json"
SCHEMA = "chipmunk.image-reference-lossy-webp-transfer/1"
FILTERED_SCHEMA = "chipmunk.image-reference-lossy-webp-transfer/2"
DELIVERIES = {"q60": [60, 4], "q70": [70, 4], "q75": [75, 4], "q85": [85, 4], "q90": [90, 4], "fast_q75": [75, 0]}
REGULAR = ("original", "sharp", "feathered", "local_tone", "local_blur", "local_copy")
CONFOUNDS = ("translated", "unrelated", "prior_webp", "prior_jpeg")
EDITS = {"sharp", "feathered"}
DESIGN = {
    "sources": "Use the pinned 26 photographed sources from the earlier padding transfer. No detector training or point selection.",
    "inputs": "Pair the native lossless PNG with native lossless source edits delivered as simple opaque lossy WebP. "
              "Use six encoding settings with known quality and effort. Keep both genuinely changed ordinary images and unchanged re-encodes.",
    "confounds": "Use matched dimensions for a wholly unrelated image, horizontal translation by three pixels, "
                 "an earlier WebP delivery at quality 62, and an earlier JPEG at quality 83. Encode these at WebP75.",
    "comparison": "Compare the existing unassessed route, a direct native RGB residual, and the preset WebP reconstructions "
                  "with coordinate-locked texture checks. Score the measured mask against exact pre-delivery changed pixels.",
    "useful": "At least 0.30 intersection over union and 0.50 recall on known edits. Count highlights on untouched and "
              "re-encoded sources separately from genuinely changed conventional controls.",
}
FILTERED_DESIGN = DESIGN | {
    "sources": "Replay the same 26 previously measured photographs as development for the filtered brightness method.",
    "comparison": "Compare direct RGB residuals with filtered luminance at the same native coordinates. "
                  "This recount applies the selected rule to all ordinary controls. Separate sources provide confirmation.",
}


def inputs(directory):
    import io

    import numpy as np
    from PIL import Image

    from scripts.image_edge_context_new_sources import deliveries
    from scripts.image_residual_encoder_transfer import decode, encode

    def webp(rgb, quality, effort):
        with io.BytesIO() as buffer:
            Image.fromarray(rgb).save(buffer, format="WEBP", quality=quality, method=effort)
            return buffer.getvalue()

    def reload_webp(body):
        with Image.open(io.BytesIO(body), formats=["WEBP"]) as image:
            return np.asarray(image.convert("RGB"))

    bank = defaultdict(dict)
    for family, source, _, role, body, pixels, truth, _ in deliveries(directory, read(SOURCE)["sources"]):
        bank[family, source][role] = (body, pixels, truth)
    original_rows = list(bank.values())
    for ix, ((family, source), views) in enumerate(bank.items()):
        reference, original, empty = views["original"]
        for carrier, (quality, effort) in DELIVERIES.items():
            for role in REGULAR:
                _, pixels, truth = views[role]
                yield family, source, carrier, role, reference, webp(pixels, quality, effort), truth
        foreign = original_rows[(ix + len(bank) // 2) % len(bank)]["original"][1]
        confounds = {
            "translated": np.roll(original, 3, axis=1),
            "unrelated": np.array(Image.fromarray(foreign).resize(original.shape[1::-1], Image.Resampling.BILINEAR)),
            "prior_webp": reload_webp(webp(original, 62, 4)),
            "prior_jpeg": decode(encode(original, 83)),
        }
        for role in CONFOUNDS:
            yield family, source, "q75", role, reference, webp(confounds[role], 75, 4), empty


def reconstruction_trial(earlier, submitted):
    """The first trial ranked three re-encodings and applied RGB comparison."""
    import cv2
    import numpy as np

    from app.detectors.image import reference_compare, reference_lossy_webp

    reference, questioned = np.asarray(earlier), np.asarray(submitted)
    blocks = reference_lossy_webp._blocks
    textured = np.ptp(blocks(cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)), axis=(-2, -1)) >= 24

    def smooth(rgb):
        return cv2.GaussianBlur(rgb.astype(np.float32), (0, 0), 1.25)

    submitted_smooth = smooth(questioned)
    qualified = []
    for quality in (60, 75, 90):
        expected = image_reencode(reference, quality)
        errors = blocks(np.max(np.abs(smooth(expected) - submitted_smooth), axis=2)).mean(axis=(-2, -1))
        agreed = textured & (errors <= 2.)
        rows, columns = agreed.shape
        zones = Counter((int(y) * 3 // rows, int(x) * 3 // columns) for y, x in np.argwhere(agreed))
        supported = [list(zone) for zone, count in sorted(zones.items()) if count >= 2]
        matched, total = int(agreed.sum()), int(textured.sum())
        if (matched < 24 or 20 * matched < 13 * total or len(supported) < 6
                or any(len({p[axis] for p in supported}) != 3 for axis in (0, 1))):
            continue
        median = round(float(np.median(errors[textured])), 6)
        proof = {"reference_reconstruction_quality": quality, "textured_tiles": total, "matching_tiles": matched,
                 "tile_side_px": 32, "supported_zones": supported, "median_tile_residual": median,
                 "reconstructed_rgb_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}
        rank = (median, -matched, quality)
        qualified.append((rank, proof, expected))
        if median == 0 and np.array_equal(expected, questioned):
            qualified = [(rank, proof, expected)]
            break
    if qualified:
        _, proof, expected = min(qualified, key=lambda item: item[0])
        result, mask = reference_compare.compare(expected, questioned)
        if result["status"] == "comparable":
            return result | {"lossy_webp": proof}, mask
    return {"status": reference_lossy_webp.UNASSESSED}, None


def image_reencode(rgb, quality):
    import io

    import numpy as np
    from PIL import Image

    with io.BytesIO() as output:
        Image.fromarray(rgb).save(output, format="WEBP", quality=quality, method=4)
        output.seek(0)
        with Image.open(output, formats=["WEBP"]) as image:
            return np.asarray(image.convert("RGB"))


def measure(directory, *, filtered=False):
    import io

    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.image import reference_compare, reference_lossy_webp

    cv2.setNumThreads(1)
    records = []
    for family, source, carrier, role, reference, submitted, truth in inputs(directory):
        with Image.open(io.BytesIO(reference)) as before, Image.open(io.BytesIO(submitted)) as after:
            require(reference_lossy_webp.supported(after, submitted), "The staged lossy carrier is not a qualified image")
            native = reference_compare.compare(np.asarray(before), np.asarray(after))
            compared = (reference_lossy_webp.compare(before, after, reference, submitted) if filtered
                        else reconstruction_trial(before, after))
        readings = {}
        for method, (result, mask) in (("native", native), ("webp", compared)):
            seen = int(mask.sum()) if mask is not None else 0
            overlap = int((mask & truth).sum()) if mask is not None else 0
            readings[method] = {"status": result["status"], "selected": seen, "overlap": overlap,
                                "mask_sha256": sha(mask.tobytes()) if mask is not None else None}
            if method == "webp" and "lossy_webp" in result:
                readings[method]["framing"] = result["lossy_webp"]
        records.append({"source": source, "generator": family, "carrier": carrier, "role": role,
                        "reference_sha256": sha(reference), "webp_sha256": sha(submitted), "truth_sha256": sha(truth.tobytes()),
                        "changed": int(truth.sum()), "readings": readings})
        if len(records) % 80 == 0:
            print("Measured", len(records), flush=True)
    return records


def summary(records):
    tallies = defaultdict(Counter)
    groups = defaultdict(set)
    for row in records:
        source, carrier, role, target = (row[key] for key in ("source", "carrier", "role", "changed"))
        require(carrier in DELIVERIES and (role in REGULAR or carrier == "q75" and role in CONFOUNDS)
                and (carrier, role) not in groups[source] and bool(target) == (role in EDITS), "Wrong carrier or role")
        groups[source].add((carrier, role))
        require(set(row["readings"]) == {"native", "webp"}, "Missing paired readings")
        for method, item in row["readings"].items():
            selected, overlap = item["selected"], item["overlap"]
            compared = item["status"] == "comparable"
            require(0 <= overlap <= min(selected, target) and (compared or selected == 0)
                    and compared == bool(item["mask_sha256"]), "Invalid region measurement")
            if method == "webp":
                require(compared == bool(item.get("framing")), "Missing lossy framing proof")
            valid = useful((selected, overlap), target)
            for name in ("all", row["generator"]):
                tallies[f"{carrier}/{name}/{role}/{method}"].update(files=1, compared=int(compared), flagged=int(bool(selected)), useful=int(valid))
    expected = {(carrier, role) for carrier in DELIVERIES for role in REGULAR} | {("q75", role) for role in CONFOUNDS}
    require(len(groups) == 26 and all(group == expected for group in groups.values()), "Comparison bank is incomplete")
    return dict(tallies)


def review(document):
    designs = {SCHEMA: DESIGN, FILTERED_SCHEMA: FILTERED_DESIGN}
    require(document["schema"] in designs and document["design"] == designs[document["schema"]]
            and document["source_sha256"] == sha(SOURCE.read_bytes()),
            "The fixed transfer protocol changed")
    outcomes = summary(document["rows"])
    require(outcomes == document["outcomes"], "Outcome arithmetic changed")
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--filtered", action="store_true")
    args = parser.parse_args()
    if args.phase == "measure":
        from PIL import features

        rows = measure(args.sources, filtered=args.filtered)
        results = {"schema": FILTERED_SCHEMA if args.filtered else SCHEMA,
                   "design": FILTERED_DESIGN if args.filtered else DESIGN, "source_sha256": sha(SOURCE.read_bytes()),
                   "libwebp": features.version("webp"), "rows": rows, "outcomes": summary(rows)}
        save(args.path, results)
    else:
        results = read(args.path)
    print(json.dumps(review(results), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
