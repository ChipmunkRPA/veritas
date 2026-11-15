# SPDX-License-Identifier: MIT
"""Evaluate native earlier-copy evidence after proportional WebP downsizing."""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts import image_reference_webp_residual_holdout as parent

ROOT = parent.development.ROOT
SCHEMA = "chipmunk.image-reference-resized-webp-transfer/1"
SOURCE = ROOT / "docs/evaluation-panels/image-reference-lossy-webp-holdout-2026-11-14.json"
PROFILES = {
    "development": {"bilinear_webp77": [87, "bilinear", 77, 2]},
    "held": {"bicubic_webp70": [75, "bicubic", 70, 4], "lanczos_webp82": [84, "lanczos", 82, 0],
             "area_webp88": [93, "area", 88, 6]},
}
EDITS = {"sharp", "feathered", "small_sharp", "small_feathered"}
REENCODES = {"original", "prior_webp", "prior_jpeg"}
OTHER = {"translated", "unrelated", "cropped", "stretched"}
ROLES = tuple(sorted(EDITS | REENCODES | OTHER | {"local_tone", "local_blur"}))
POLICY = {
    "truth": "Project the earlier and edited native arrays at the same fixed downsampling settings, then compare exact RGB before WebP. "
             "The small edit family constrains the authored change to a 52-pixel square selected by exact change density.",
    "variants": "Use pinned earlier photos with ancestry separate between development and held. Include an area projection "
                "not used by the detector. Use distinct edit types, prior loss, ordinary appearance changes, and wrong framing.",
    "readout": "Required IoU .30, changed-pixel recall .50. Native earlier crop bounds round the corresponding edges outward.",
}


def source_bank(args, split):
    if split == "held":
        yield from parent.admit(args)
        return
    from scripts.image_edge_context_new_sources import deliveries

    views = defaultdict(dict)
    for cohort, group, _, role, _, pixels, _, _ in deliveries(args.development_sources, parent.read(parent.development.SOURCE)["sources"]):
        if role in ("original", "sharp"):
            views[cohort, group][role] = pixels
    for (cohort, group), records in views.items():
        yield cohort, group, group, records["original"], records["sharp"]


def resize(rgb, size, method):
    import cv2
    import numpy as np
    from PIL import Image

    if method == "area":
        return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    with Image.fromarray(rgb).resize(size, getattr(Image.Resampling, method.upper())) as projected:
        return np.array(projected)


def small_edit(original, edited, *, feathered):
    import cv2
    import numpy as np

    strength = np.mean(np.abs(edited.astype(np.int16) - original), axis=2)
    density = cv2.boxFilter(strength, -1, (52, 52), normalize=False, borderType=cv2.BORDER_CONSTANT)
    limit = 26
    h, w = strength.shape
    y, x = np.unravel_index(int(density[limit:h-limit, limit:w-limit].argmax()), (h - 2 * limit, w - 2 * limit))
    patch = (slice(y, y + 52), slice(x, x + 52))
    out = original.copy()
    if feathered:
        edge = np.minimum(np.arange(52), np.arange(51, -1, -1))
        weight = np.minimum(np.minimum(edge[:, None], edge[None, :]) / 6, 1)[:, :, None]
        out[patch] = np.rint(weight * edited[patch] + (1 - weight) * original[patch]).astype(np.uint8)
    else:
        out[patch] = edited[patch]
    return out


def deliveries(args, split, *, identity=None):
    import numpy as np

    from scripts.image_residual_encoder_transfer import authored

    bank = list(source_bank(args, split))
    for index, (family, group, source, original, edited) in enumerate(bank):
        if identity is not None and source != identity:
            continue
        changed = np.any(original != edited, axis=-1)
        ordinary, alterations = authored({"original": original, "edited": edited}, changed, source)
        ref = parent.encode(original, "PNG")
        w, h = original.shape[1::-1]
        foreign = bank[(index + len(bank) // 2) % len(bank)][3]
        sources = {"original": original, "sharp": alterations["sharp"], "feathered": alterations["feathered"],
                   "small_sharp": small_edit(original, alterations["sharp"], feathered=False),
                   "small_feathered": small_edit(original, alterations["sharp"], feathered=True),
                   "local_tone": ordinary["local_tone"], "local_blur": ordinary["local_blur"],
                   "unrelated": foreign, "cropped": original[h // 25:h - h // 25, w // 25:w - w // 25],
                   "prior_webp": parent.image_array(parent.encode(original, "WEBP", quality=62, method=4), "WEBP"),
                   "prior_jpeg": parent.image_array(parent.encode(original, "JPEG", quality=82, subsampling=2), "JPEG")}
        for profile, (percent, kernel, quality, effort) in PROFILES[split].items():
            size = round(w * percent / 100), round(h * percent / 100)
            base = resize(original, size, kernel)
            for role in ROLES:
                if role == "translated":
                    pixels = np.roll(base, 4, axis=1)
                elif role == "stretched":
                    pixels = resize(original, (size[0] + 8, size[1]), kernel)
                else:
                    pixels = resize(sources[role], size, kernel)
                body = parent.encode(pixels, "WEBP", quality=quality, method=effort)
                truth = np.any(base != pixels, axis=2) if role in EDITS else np.zeros(pixels.shape[:2], dtype=bool)
                case = {"family": family, "group": group, "source": source, "profile": profile, "role": role,
                        "reference_sha256": parent.sha(ref), "submitted_sha256": parent.sha(body), "changed": int(truth.sum()),
                        "truth_sha256": parent.sha(truth.tobytes()), "submitted_size": list(pixels.shape[1::-1])}
                yield case, ref, body, truth


def calculate(rows, split):
    profiles = PROFILES[split]
    tallies = defaultdict(Counter)
    groups = defaultdict(set)
    for row in rows:
        case = row["case"]
        source, profile, role = (case[key] for key in ("source", "profile", "role"))
        changed, selected, overlap = case["changed"], row["selected"], row["overlap"]
        parent.require(role in ROLES and profile in profiles and (profile, role) not in groups[source]
                       and bool(changed) == (role in EDITS) and 0 <= overlap <= min(changed, selected)
                       and bool(row["selection_sha256"]) == (row["status"] == "comparable"), "Invalid panel row")
        parent.require(row["status"] == "comparable" or selected == overlap == 0, "An unassessed route highlights pixels")
        groups[source].add((profile, role))
        tallies[profile, role].update(files=1, baseline_compared=int(row["baseline"] == "comparable"),
                                     compared=int(row["status"] == "comparable"), flags=int(selected > 0),
                                     useful=int(parent.useful((selected, overlap), changed)))
    parent.require(len(groups) == (26 if split == "development" else sum(parent.COUNTS.values()))
                   and all(key == {(profile, role) for profile in profiles for role in ROLES} for key in groups.values()),
                   "Incomplete independent-source panel")
    return {"/".join(keys): dict(counts) for keys, counts in sorted(tallies.items())}


def verify(receipt):
    split = receipt["split"]
    parent.require(receipt["schema"] == SCHEMA and receipt["method"] == POLICY and receipt["profiles"] == PROFILES[split]
                   and receipt["parents"] == [parent.sha(p.read_bytes()) for p in (parent.development.SOURCE, SOURCE)],
                   "Earlier source record differs")
    outcomes = calculate(receipt["rows"], split)
    parent.require(outcomes == receipt["outcomes"], "Reported counts differ from measured masks")
    return outcomes


def run(args):
    import cv2
    from PIL import Image, features

    from app.detectors.image import reference_lossy_webp, reference_lossy_webp_resize

    cv2.setNumThreads(1)
    rows = []
    for case, ref, body, truth in deliveries(args, args.phase):
        with Image.open(io.BytesIO(ref)) as earlier, Image.open(io.BytesIO(body)) as submitted:
            baseline, _ = reference_lossy_webp.compare(earlier, submitted, ref, body)
            result, mask = reference_lossy_webp_resize.compare(earlier, submitted, ref, body)
        rows.append({"case": case, "baseline": baseline["status"], "status": result["status"],
                     "selected": int(mask.sum()) if mask is not None else 0,
                     "overlap": int((mask & truth).sum()) if mask is not None else 0,
                     "selection_sha256": parent.sha(mask.tobytes()) if mask is not None else None})
        if len(rows) % 130 == 0:
            print(args.phase, len(rows), flush=True)
    receipt = {"schema": SCHEMA, "method": POLICY, "parents": [parent.sha(p.read_bytes()) for p in (parent.development.SOURCE, SOURCE)],
               "split": args.phase, "profiles": PROFILES[args.phase], "libwebp": features.version("webp"),
               "outcomes": calculate(rows, args.phase), "rows": rows}
    verify(receipt)
    parent.save(args.path, receipt)
    return receipt["outcomes"]


def arguments(parser):
    data = ROOT / "backend/data/independent-evaluation"
    parser.add_argument("--development-sources", type=Path)
    parser.add_argument("--native", type=Path, default=data / "native-regional-serving-v47")
    parser.add_argument("--pairs", type=Path, default=data / "image-positive-pairs-v21-cache")
    parser.add_argument("--background", type=Path, default=data / "image-bg-real-acquisition-v225")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "held", "review"))
    parser.add_argument("path", type=Path)
    arguments(parser)
    args = parser.parse_args()
    data = verify(parent.read(args.path)) if args.phase == "review" else run(args)
    print(json.dumps(data, sort_keys=True), flush=True)
