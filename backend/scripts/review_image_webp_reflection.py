# SPDX-License-Identifier: Apache-2.0
"""Replay the frozen exact-reflection WebP comparison without media libraries."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.image-exact-webp-reflection/1"
EDITED = frozenset({"horizontal", "vertical", "feathered_vertical"})
ABSTENTION = frozenset({"ambiguous_pairs", "unassessed_match_budget", "unsupported_profile", "unsupported_geometry",
                        "unavailable"})
SIZES = ((512, 512), (576, 520), (628, 532), (704, 512), (530, 748))
STAGED_SIZES = {"horizontal": (30, 34), "vertical": (32, 28), "feathered_vertical": (50, 48),
                "translation": (30, 36), "quarter_turn": (32, 34)}


def _iou(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    y = max(0, min(ay + ah, by + bh) - max(ay, by))
    common = x * y
    return common / (aw * ah + bw * bh - common)


def _validate_pair(pair, size):
    if len(pair) != 2:
        raise ValueError("Image geometry requires two boxes")
    for x, y, w, h in pair:
        if not (0 <= x < x + w <= size[0] and 0 <= y < y + h <= size[1]):
            raise ValueError("Box is outside the exact source raster")


def _useful(pairs, truth, counts, size):
    for pair in pairs:
        _validate_pair(pair, size)
    if truth is None:
        if counts is not None:
            raise ValueError("Ordinary file has recorded edit coverage")
        return False
    _validate_pair(truth, size)
    possible = False
    for (a, b), record in zip(pairs, counts, strict=True):
        fit = max(min(_iou(a, left), _iou(b, right)) for left, right in (truth, truth[::-1]))
        numerator, denominator = record["mask_overlap"], record["mask_total"]
        if abs(fit - record["min_box_iou"]) > 1e-11 or not 0 <= numerator <= denominator or denominator <= 0:
            raise ValueError("Stored overlap differs from the candidate geometry")
        possible |= fit >= .5 and numerator * 2 >= denominator
    return possible


def review(receipt):
    if receipt["schema"] != SCHEMA or receipt["format_controls"] != dict.fromkeys(
        ("alpha", "animated", "extended", "lossy", "smaller"), "unsupported_profile"
    ):
        raise ValueError("Wrong report or format controls")
    plan, rows = receipt["plan"], receipt["rows"]
    expected = {(r["bank"], r["id"], role) for r in plan["sources"] for role in plan["roles"][r["bank"]]}
    if len(rows) != len(expected) or {(r["bank"], r["id"], r["role"]) for r in rows} != expected:
        raise ValueError("Deliveries differ from the frozen evaluation roster")

    outcomes = {}
    extra_groups = {"useful_edits": set(), "ordinary": set()}
    for row in rows:
        role = row["role"]
        counts = outcomes.setdefault(role, Counter())
        is_edit = role in EDITED
        truth = row["truth"]
        if bool(truth) != is_edit:
            raise ValueError("Role and independent edit labels disagree")
        earlier = [pair for values in row["existing"].values() for pair in values]
        trial = row["candidate"]
        proof, status = trial["proof"], trial["status"]
        if bool(proof) != (status == "supported_pair"):
            raise ValueError("Candidate coverage and reported pair disagree")
        candidate = [proof["pair"]] if proof else []
        if proof:
            first, second = proof["pair"]
            measure = proof["measurement"]
            if (proof["axis"] not in {"horizontal", "vertical"} or first[2:] != second[2:]
                    or measure["matching_pixels"] != first[2] * first[3]
                    or not 64 <= measure["distinct_rgb_count"] <= measure["matching_pixels"]
                    or len(measure["perimeter_mad"]) != 4
                    or sum(value >= 3 for value in measure["perimeter_mad"]) < 3):
                raise ValueError("Exact reflected pair failed its geometric witness")
        prior = _useful(earlier, truth, row["existing_matches"], row["size"])
        correct = _useful(candidate, truth, row["candidate_matches"], row["size"])
        combined = bool(candidate and not earlier and not (set(row["existing_status"].values()) & ABSTENTION))
        if combined != row["combined_adds"]:
            raise ValueError("Stored fallback does not preserve the incumbent")

        counts["files"] += 1
        counts["existing_highlight"] += bool(earlier)
        counts["candidate_highlight"] += bool(candidate)
        counts["combined_highlight"] += bool(earlier or combined)
        counts["existing_useful"] += prior
        counts["candidate_useful"] += correct
        counts["combined_useful"] += prior or (combined and correct)
        counts["candidate_misplaced"] += is_edit and bool(candidate) and not correct
        counts["combined_added_misplaced"] += is_edit and combined and not correct
        counts["candidate_abstained"] += status in ABSTENTION
        counts["candidate_clear"] += status == "no_pair"
        if combined and correct and not prior:
            extra_groups["useful_edits"].add(row["id"])
        if combined and not is_edit:
            extra_groups["ordinary"].add((row["bank"], row["id"]))
    return {"roles": {role: dict(counts) for role, counts in outcomes.items()},
            "groups": {k: len(v) for k, v in extra_groups.items()}}


def _delivery(rgb, role, index):
    import cv2
    import numpy as np

    if role == "original":
        return rgb
    if role == "global_tone":
        return np.clip(np.rint(rgb * .89 + 8), 0, 255).astype(np.uint8)
    view = rgb.copy()
    height, width = rgb.shape[:2]
    if role == "local_blur":
        y, x = height // 2 - 36, width // 2 - 42
        view[y:y+72, x:x+84] = cv2.GaussianBlur(rgb[y:y+72, x:x+84], (15, 15), 3)
        return view
    if role == "symmetrical_layout":
        half = width // 2
        view[:, half:] = view[:, :half][:, ::-1]
        return view
    x, y = 48 + (index * 29) % 88, 58 + (index * 19) % 112
    right_x, right_y = width - 140 - (index * 17) % 68, height - 138 - (index * 23) % 80
    w, h = STAGED_SIZES[role]
    donor = rgb[y:y+h, x:x+w]
    if role == "horizontal":
        donor = donor[:, ::-1]
    elif role in {"vertical", "feathered_vertical"}:
        donor = donor[::-1]
    elif role == "quarter_turn":
        donor = np.rot90(donor)
    h, w = donor.shape[:2]
    target = view[right_y:right_y+h, right_x:right_x+w]
    if role == "feathered_vertical":
        rows, cols = np.mgrid[:h, :w]
        step = np.minimum.reduce([cols, w - 1 - cols, rows, h - 1 - rows])
        alpha = np.minimum(1.0, step / 5.0)[..., None]
        donor = np.rint(target * (1 - alpha) + donor * alpha).astype(np.uint8)
    target[:] = donor
    return view


def native_replay(receipt, photographs, patterns):
    """Regenerate the encoded files and compare the shipping candidate to the freeze."""
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    from app.evidence.image_exact_small_copy_source import review_reflection

    cv2.setNumThreads(1)
    indexed = {(row["bank"], row["id"], row["role"]): row for row in receipt["rows"]}
    for index, record in enumerate(receipt["plan"]["sources"]):
        bank, name = record["bank"], record["id"]
        path = photographs / f"{name}.png" if bank == "photograph" else patterns / f"{name}-repetition-90.jpg"
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError(f"Archived source differs from its commitment for {name}")
        with Image.open(io.BytesIO(content)) as source:
            rgb = np.array(ImageOps.fit(source.convert("RGB"), SIZES[index % len(SIZES)], Image.Resampling.LANCZOS))
        for role in receipt["plan"]["roles"][bank]:
            row = indexed[bank, name, role]
            view = _delivery(rgb, role, index)
            buffer = io.BytesIO()
            Image.fromarray(view).save(buffer, format="WEBP", lossless=True, method=1)
            data = buffer.getvalue()
            if hashlib.sha256(data).hexdigest() != row["sha256"] or hashlib.sha256(view.tobytes()).hexdigest() != row["rgb_sha256"]:
                raise ValueError(f"Staged WebP or decoded pixels differ for {name} {role}")
            if role in EDITED:
                mask = np.any(view != rgb, axis=2)
                if hashlib.sha256(mask.tobytes()).hexdigest() != row["mask_sha256"]:
                    raise ValueError(f"Staged change mask differs for {name} {role}")
            got, pairs, transform = review_reflection(data)
            expected = row["candidate"]
            if got["status"] != expected["status"] or bool(pairs) != bool(expected["proof"]):
                raise ValueError(f"Serving coverage differs for {name} {role}")
            if pairs:
                proof = expected["proof"]
                native = got["measurement"]
                if ([list(box) for box in pairs[0]] != proof["pair"] or transform != f"{proof['axis']}_reflection"
                        or {"matching_pixels": native["matching_pixels"], "distinct_rgb_count": native["different_color_values"],
                            "perimeter_mad": native["perimeter_mean_absolute_difference"]} != proof["measurement"]):
                    raise ValueError(f"Serving reflected measurement differs for {name} {role}")
    return len(indexed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--photographs", type=Path)
    parser.add_argument("--patterns", type=Path)
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as file:
        record = json.load(file)
    computed = review(record)
    if "summary" in record and computed != record["summary"]:
        raise ValueError("Publication summary differs from the row-level review")
    if bool(args.photographs) != bool(args.patterns):
        parser.error("Provide both source folders for native replay")
    if args.photographs:
        computed["native_candidate_readback"] = native_replay(record, args.photographs, args.patterns)
    print(json.dumps(computed, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
