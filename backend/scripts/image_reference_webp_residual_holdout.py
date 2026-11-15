# SPDX-License-Identifier: Apache-2.0
"""Replay lossy WebP comparison on a second set of source photographs."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_reference_lossy_webp_transfer as development
from scripts.image_small_region_repeat import read, require, save, sha, useful

SCHEMA = "chipmunk.image-reference-webp-residual-holdout/1"
PROFILES = {"q65": [65, 4], "fast_q82": [82, 0], "slow_q92": [92, 6]}
COUNTS = {"native": 16, "lama": 8}
ROLES = ("original", "sharp", "feathered", "local_tone", "local_blur", "prior_webp", "prior_jpeg", "translated", "unrelated")


def image_array(data, fmt):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data), formats=[fmt]) as image:
        return np.asarray(image.convert("RGB"))


def encode(rgb, fmt, **options):
    from PIL import Image

    with io.BytesIO() as buffer:
        Image.fromarray(rgb).save(buffer, format=fmt, **options)
        return buffer.getvalue()


def admit(args):
    import cv2
    from PIL import Image

    from scripts.image_bg_real_acquisition_v221 import sketch
    from scripts.image_edge_context_transfer import native_deliveries
    from scripts.image_reference_delivery_envelope_lama import source_pixels

    earlier = read(development.SOURCE)["sources"]
    excluded = {case["original"]["sha256"] for case in earlier}
    seen = []
    for row in earlier:
        path = args.development_sources / str(row["slot"]) / "original.png"
        require(sha(path.read_bytes()) == row["proof"]["assets"]["original"], "Changed development original")
        with Image.open(path) as image:
            seen.append(sketch(image.convert("RGB")))

    def unique(rgb):
        signature = sketch(Image.fromarray(rgb))
        if any((signature ^ other).bit_count() <= 8 for other in seen):
            return False
        seen.append(signature)
        return True

    bank = []
    covers = set()
    latest = None
    for cohort, group, source, role, _, rgb, _, _ in native_deliveries(args):
        if cohort != "native_png" or group in covers or source in excluded:
            continue
        h, w = rgb.shape[:2]
        if min(w, h) * 3 < max(w, h) * 2:
            continue
        scaled = cv2.resize(rgb, (w * 768 // max(w, h), h * 768 // max(w, h)), interpolation=cv2.INTER_AREA)
        if role == "ai":
            latest = (source, scaled)
        if role == "original" and latest is not None and latest[0] == source and unique(scaled):
            bank.append(("native", group, source, scaled, latest[1]))
            covers.add(group)
            excluded.add(source)
        if len(bank) == COUNTS["native"]:
            break
    require(len(bank) == COUNTS["native"], "The native source bank is incomplete")
    qualified = args.background / "qualified-private.json"
    parent = read(development.ROOT / "docs/evaluation-panels/image-bg-real-pixel-qualification-complete-v225.json")
    require(sha(qualified.read_bytes()) == parent["qualified_private_sha256"], "LaMa source ledger differs")
    roster = read(qualified)
    authors = set()
    for row in sorted(roster["qualified"], key=lambda case: case["source"]["original"]["sha256"]):
        source = row["source"]
        digest = source["original"]["sha256"]
        author = source["flickr_owner"].casefold()
        if digest in excluded or author in authors or min(source["size"]) < 512:
            continue
        require(source["photo_attribution"]["License"] == "https://creativecommons.org/licenses/by/2.0/", "Original terms differ")
        original, edited, _ = source_pixels(Path(roster["cache"]), row)
        if not unique(original):
            continue
        bank.append(("lama", digest, digest, original, edited))
        authors.add(author)
        if len(authors) == COUNTS["lama"]:
            break
    require(len(authors) == COUNTS["lama"], "The LaMa source bank is incomplete")
    return bank


def cases(args):
    import numpy as np
    from PIL import Image

    from scripts.image_residual_encoder_transfer import authored

    bank = admit(args)
    for index, (cohort, group, source, before, after) in enumerate(bank):
        changed = np.any(before != after, axis=-1)
        ordinary, edited = authored({"original": before, "edited": after}, changed, source)
        ref = encode(before, "PNG")
        foreign = bank[(index + len(bank) // 2) % len(bank)][3]
        views = {**ordinary, **edited, "translated": np.roll(before, 3, axis=1),
                 "unrelated": np.asarray(Image.fromarray(foreign).resize(before.shape[1::-1], Image.Resampling.BILINEAR))}
        views["prior_webp"] = image_array(encode(before, "WEBP", quality=56, method=4), "WEBP")
        views["prior_jpeg"] = image_array(encode(before, "JPEG", quality=81, subsampling=2), "JPEG")
        for profile, (quality, method) in PROFILES.items():
            for role in ROLES:
                source_view = views[role]
                delivery = encode(source_view, "WEBP", quality=quality, method=method)
                truth = np.any(before != source_view, axis=2) if role in development.EDITS else np.zeros_like(changed)
                row = {"cohort": cohort, "group": group, "source": source, "profile": profile, "role": role,
                       "ref_sha256": sha(ref), "webp_sha256": sha(delivery), "mask_sha256": sha(truth.tobytes()),
                       "changed": int(truth.sum()), "size": list(before.shape[1::-1])}
                yield row, ref, delivery, truth


def readout(rows):
    results = defaultdict(Counter)
    groups = defaultdict(set)
    deliveries_by_source = defaultdict(set)
    identities = set()
    for row in rows:
        case = row["case"]
        cohort, group, source, profile, role, changed = (case[key] for key in ("cohort", "group", "source", "profile", "role", "changed"))
        require(cohort in COUNTS and profile in PROFILES and role in ROLES and bool(changed) == (role in development.EDITS),
                "Unknown group or delivery type")
        require((source, profile, role) not in identities, "Duplicate source delivery")
        identities.add((source, profile, role))
        groups[cohort].add(group)
        deliveries_by_source[cohort, group, source].add((profile, role))
        status, size, overlap = (row[key] for key in ("status", "selected", "overlap"))
        compared = status == "comparable"
        require((compared or not size) and compared == bool(row["selected_sha256"])
                and 0 <= overlap <= min(changed, size), "Invalid selected image mask")
        for name in ("all", cohort):
            results[profile, name, role].update(files=1, compared=int(compared), flags=int(bool(size)),
                                                useful=int(useful((size, overlap), changed)))
    expected = {(profile, role) for profile in PROFILES for role in ROLES}
    require({cohort: len(groups_for_cohort) for cohort, groups_for_cohort in groups.items()} == COUNTS
            and len(deliveries_by_source) == sum(COUNTS.values())
            and all(deliveries == expected for deliveries in deliveries_by_source.values()), "Holdout incomplete")
    return {"/".join(keys): dict(value) for keys, value in sorted(results.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-sources", type=Path)
    parser.add_argument("--prior-webp-results", type=Path)
    data = development.ROOT / "backend/data/independent-evaluation"
    parser.add_argument("--native", type=Path, default=data / "native-regional-serving-v47")
    parser.add_argument("--pairs", type=Path, default=data / "image-positive-pairs-v21-cache")
    parser.add_argument("--background", type=Path, default=data / "image-bg-real-acquisition-v225")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.path)
        require(receipt["freeze"]["schema"] == SCHEMA and receipt["freeze"]["profiles"] == PROFILES
                and [row["case"] for row in receipt["rows"]] == receipt["freeze"]["cases"]
                and readout(receipt["rows"]) == receipt["outcomes"], "Holdout replay changed")
        print(json.dumps(receipt["outcomes"], sort_keys=True))
        return

    import cv2
    from PIL import Image, features

    from app.detectors.image import reference_lossy_webp as detector

    cv2.setNumThreads(1)
    configurations = {"method": detector.METHOD, "size": detector.TILE}
    if args.phase == "freeze":
        save(args.path, {"schema": SCHEMA, "profiles": PROFILES, "configurations": configurations,
                        "prior_sha256": sha(args.prior_webp_results.read_bytes()), "libwebp": features.version("webp"),
                        "cases": [row for row, *_ in cases(args)]})
        return
    frozen = read(args.path)
    require(frozen["configurations"] == configurations and frozen["libwebp"] == features.version("webp"), "Held method or runtime changed")
    results = []
    for (row, ref, body, truth), identity in zip(cases(args), frozen["cases"], strict=True):
        require(row == identity, "Frozen input bytes changed")
        with Image.open(io.BytesIO(ref)) as earlier, Image.open(io.BytesIO(body)) as submitted:
            result, mask = detector.compare(earlier, submitted, ref, body)
        results.append({"case": row, "status": result["status"], "selected": int(mask.sum()) if mask is not None else 0,
                        "overlap": int((mask & truth).sum()) if mask is not None else 0,
                        "selected_sha256": sha(mask.tobytes()) if mask is not None else None})
    save(args.output, {"freeze": frozen, "rows": results, "outcomes": readout(results)})


if __name__ == "__main__":
    main()
