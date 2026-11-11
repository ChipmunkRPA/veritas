"""Authored-only, model-free same-image donor corroboration for a proposed region."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, features
from PIL import __version__ as pillow_version

from app.detectors.image import copy_move

SIDE_LIMIT = 512
PATCH = 16
BORDER = 4
MIN_TEXTURE_STD = 8.0
MIN_ANCHORS = 4
MIN_CORRELATION = 0.96
MIN_SUPPORT_FRACTION = 2 / 3
MIN_DISTINCT_DONOR_MARGIN = 0.03
JPEG_QUALITY = 90
TRANSFORMS = ("translation", "horizontal_reflection")
SCOPE = "Same-image copy compatibility only. The existing candidate remains an editing cue. No authenticity or authorship verdict."


def bounds(mask):
    x = np.flatnonzero(mask.any(axis=0))
    y = np.flatnonzero(mask.any(axis=1))
    return int(x[0]), int(y[0]), int(x[-1] - x[0] + 1), int(y[-1] - y[0] + 1)


def positions(start, extent):
    available = extent - PATCH - 2 * BORDER
    if available < 0:
        return ()
    count = min(3, 1 + available // PATCH)
    return np.linspace(start + BORDER, start + BORDER + available, count, dtype=int).tolist()


def anchors(luma, selection, box):
    left, top, width, height = box
    result = []
    for y in positions(top, height):
        for x in positions(left, width):
            window = np.s_[y : y + PATCH, x : x + PATCH]
            patch = luma[window]
            if selection[window].all() and patch.std() >= MIN_TEXTURE_STD:
                result.append((x - left, y - top, patch))
    return result


def candidate_donors(luma, patches, box, reflected):
    left, top, width, height = box
    rows, columns = luma.shape[0] - height + 1, luma.shape[1] - width + 1
    response = []
    for x, y, patch in patches:
        if reflected:
            x = width - PATCH - x
            patch = np.ascontiguousarray(patch[:, ::-1])
        matches = cv2.matchTemplate(luma, patch, cv2.TM_CCOEFF_NORMED)
        response.append(matches[y : y + rows, x : x + columns])
    scores = np.stack(response)
    support = np.count_nonzero(scores >= MIN_CORRELATION, axis=0)
    median = np.median(scores, axis=0)
    yy, xx = np.ogrid[:rows, :columns]
    target_overlap = (xx < left + width) & (xx + width > left) & (yy < top + height) & (yy + height > top)
    eligible = (support >= math.ceil(len(patches) * MIN_SUPPORT_FRACTION)) & ~target_overlap
    return np.where(eligible, median, -np.inf), support


def strongest(maps):
    best = None
    for transform, (scores, support) in zip(TRANSFORMS, maps, strict=True):
        y, x = np.unravel_index(np.argmax(scores), scores.shape)
        score = float(scores[y, x])
        if np.isfinite(score) and (best is None or score > best["correlation"]):
            best = {
                "origin": (int(x), int(y)),
                "transform": transform,
                "correlation": score,
                "supporting_anchors": int(support[y, x]),
            }
    return best


def competing_donor(maps, origin, width, height):
    best = None
    for scores, _ in maps:
        yy, xx = np.ogrid[: scores.shape[0], : scores.shape[1]]
        intersection = np.maximum(0, width - abs(xx - origin[0])) * np.maximum(
            0, height - abs(yy - origin[1])
        )
        # Same spatial donor, including a small alignment shift, is one alternative.
        distinct = intersection * 3 < 2 * width * height
        second = float(np.max(np.where(distinct, scores, -np.inf)))
        if np.isfinite(second) and (best is None or second > best):
            best = second
    return best


def corroborate(rgb, selection):
    """Evaluate one existing source-coordinate Boolean mask without modifying it."""
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or selection.dtype != bool
        or selection.shape != rgb.shape[:2]
        or not selection.any()
    ):
        raise ValueError("Upright RGB and one nonempty source-coordinate Boolean selection required")
    box = bounds(selection)
    record = {
        "proposal_box_xywh": list(box),
        "proposal_pixels": int(selection.sum()),
        "proposal_mask_sha256": hashlib.sha256(selection.tobytes()).hexdigest(),
        "anchor_count": 0,
        "status": "unsupported_geometry",
        "donor": None,
        "scope": SCOPE,
    }
    if min(rgb.shape[:2]) < copy_move.MIN_INPUT_SIDE or max(rgb.shape[:2]) > SIDE_LIMIT:
        return record
    luma, xscale, yscale = copy_move._luma(rgb)
    if xscale != 1 or yscale != 1:
        raise ValueError("Native coordinates must be retained")
    patches = anchors(luma, selection, box)
    record.update(anchor_count=len(patches), status="insufficient_texture")
    if len(patches) < MIN_ANCHORS:
        return record
    maps = [candidate_donors(luma, patches, box, reflected) for reflected in (False, True)]
    best = strongest(maps)
    record["status"] = "no_donor"
    if best is None:
        return record
    width, height = box[2:]
    second = competing_donor(maps, best["origin"], width, height)
    margin = best["correlation"] - second if second is not None else None
    record["competing_correlation"] = second
    record["donor_margin"] = margin
    if margin is not None and margin < MIN_DISTINCT_DONOR_MARGIN:
        record["status"] = "ambiguous_repetition"
        return record
    record.update(
        status="copy_consistent",
        donor={
            "box_xywh": [*best["origin"], width, height],
            "transform": best["transform"],
            "correlation": best["correlation"],
            "supporting_anchors": best["supporting_anchors"],
        },
    )
    return record


def mask_for(shape, box):
    x, y, width, height = box
    selection = np.zeros(shape, dtype=bool)
    selection[y : y + height, x : x + width] = True
    return selection


def fixture_texture(seed, shape=(240, 256)):
    generator = np.random.default_rng(seed)
    base = generator.integers(0, 256, size=(*shape, 3), dtype=np.uint8)
    softened = cv2.GaussianBlur(base, (5, 5), 0.85)
    y, x = np.indices(shape)
    pattern = (20 * np.sin(x / 2.5) + 24 * np.cos(y / 4.0))[..., None]
    return np.clip(softened.astype(np.float32) + pattern, 0, 255).astype(np.uint8)


def authored_inputs():
    source = fixture_texture(295)
    target, donor = (31, 33, 64, 64), (151, 138, 64, 64)
    y, x = donor[1], donor[0]
    selection = mask_for(source.shape[:2], target)
    moved = source.copy()
    moved[selection] = source[y : y + 64, x : x + 64].reshape(-1, 3)
    reflected = source.copy()
    reflected[selection] = source[y : y + 64, x : x + 64][:, ::-1].reshape(-1, 3)
    mixed = reflected.copy()
    independent = mask_for(source.shape[:2], (145, 25, 64, 64))
    mixed[independent] = fixture_texture(902, (64, 64)).reshape(-1, 3)
    periodic = np.tile(fixture_texture(78, (48, 48)), (5, 6, 1))[:, :256]
    flat = np.full_like(source, 119)
    return (
        ("translated_copy", moved, selection, selection, None, donor),
        ("reflected_copy", reflected, selection, selection, None, donor),
        ("repeated_texture", periodic, selection, None, None, None),
        ("independent_replacement", mixed, independent, selection, independent, None),
        ("mixed_copy_region", mixed, selection, selection, independent, donor),
        ("flat_selection", flat, selection, None, None, None),
        ("untouched_candidate", source, selection, None, None, None),
    )


def jpeg(rgb):
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=JPEG_QUALITY)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"))


def authored():
    cv2.setNumThreads(1)
    fixtures = authored_inputs()
    rows = []
    for name, image, proposal, ordinary, noncopy, known_donor in fixtures:
        expected_mask = proposal.copy()
        ordinary_pixels = int(np.count_nonzero(proposal & ordinary)) if ordinary is not None else 0
        noncopy_pixels = int(np.count_nonzero(proposal & noncopy)) if noncopy is not None else 0
        for encoding, submitted in (("native", image), ("jpeg90", jpeg(image))):
            result = corroborate(submitted, proposal)
            consistent = result["status"] == "copy_consistent"
            result.update(
                fixture=name,
                encoding=encoding,
                candidate_retained=np.array_equal(proposal, expected_mask),
                ordinary_pixels_under_proposal=ordinary_pixels,
                authored_noncopy_pixels_under_proposal=noncopy_pixels,
                donor_matches_fixture=bool(
                    result["donor"] and known_donor and result["donor"]["box_xywh"] == list(known_donor)
                ),
                copy_consistent_overlap_pixels={
                    "ordinary_only": ordinary_pixels if consistent else 0,
                    "authored_noncopy": noncopy_pixels if consistent else 0,
                },
            )
            rows.append(result)
    outcomes = Counter(row["status"] for row in rows)
    return {
        "planned_views": 2 * len(fixtures),
        "completed_views": len(rows),
        "evaluated_views": len(rows) - outcomes["unsupported_geometry"] - outcomes["insufficient_texture"],
        "candidate_retained_views": sum(row["candidate_retained"] for row in rows),
        "outcomes": dict(outcomes),
        "ordinary_copy_corroborated": sum(row["donor_matches_fixture"] for row in rows),
        "authored_noncopy_overlap_views": sum(
            row["copy_consistent_overlap_pixels"]["authored_noncopy"] > 0 for row in rows
        ),
        "untouched_copy_supported": sum(
            row["ordinary_pixels_under_proposal"] == row["authored_noncopy_pixels_under_proposal"] == 0
            and row["status"] == "copy_consistent"
            for row in rows
        ),
        "generator_calls": 0,
        "rows": rows,
    }


def protocol():
    implementation = Path(__file__)
    existing = Path(copy_move.__file__)
    return {
        "schema": "chipmunk.image-copy-corroboration-protocol/295",
        "implementation_sha256": hashlib.sha256(implementation.read_bytes()).hexdigest(),
        "existing_copy_move_sha256": hashlib.sha256(existing.read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "pillow": pillow_version,
        "jpeg_codec": features.version_codec("jpg"),
        "side_limit": SIDE_LIMIT,
        "patch_side": PATCH,
        "border_pixels": BORDER,
        "minimum_anchors": MIN_ANCHORS,
        "minimum_luma_std": MIN_TEXTURE_STD,
        "minimum_correlation": MIN_CORRELATION,
        "minimum_anchor_support_fraction": MIN_SUPPORT_FRACTION,
        "minimum_donor_margin": MIN_DISTINCT_DONOR_MARGIN,
        "transforms": list(TRANSFORMS),
        "jpeg_quality": JPEG_QUALITY,
        "scope": SCOPE,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("authored", "protocol"))
    args = parser.parse_args()
    print(json.dumps({"authored": authored, "protocol": protocol}[args.command](), indent=2, allow_nan=False))
