"""Test whether boundary prominence survives feathered local edits and JPEG encoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from scripts import image_background_patch_score as paired
from scripts import image_bg_real_insertion_score as earlier
from scripts import image_relative_patch_transfer as shared

SCHEMA = "chipmunk.image-boundary-blend/1"
SCALES = (1 / 6, 1 / 4, 1 / 3)
PHASE_CONTEXT = {"png": 3, "jpeg75": 8}
FEATHER_PX = 12
sha, write = shared.file_digest, shared.write


def step_prominence(pixels: np.ndarray, offset: int) -> np.ndarray:
    """Compare an inter-pixel step with parallel steps at fixed displacement."""
    steps = np.zeros(pixels.shape[:2], dtype=np.float32)
    steps[:, 1:] = np.abs(pixels[:, 1:] - pixels[:, :-1]).mean(axis=2)
    near = (steps[:, :-2 * offset] + steps[:, 2 * offset:]) / 2
    center = steps[:, offset:-offset]
    prominence = np.zeros_like(steps)
    prominence[:, offset:-offset] = np.clip(np.log((center + 3) / (near + 3)), 0, 3)
    return prominence


def rank(pixels: np.ndarray, fmt: str):
    """Search bounded rectangles for consistently prominent boundaries on all sides."""
    y, x = pixels.shape[:2]
    offset = PHASE_CONTEXT[fmt]
    rgb = pixels.astype(np.float32)
    vertical = np.pad(step_prominence(rgb, offset).cumsum(0), ((1, 0), (0, 0)))
    horizontal = np.pad(step_prominence(rgb.transpose(1, 0, 2), offset).T.cumsum(1), ((0, 0), (1, 0)))
    best = (-1., ())
    for width in (round(x * ratio) for ratio in SCALES):
        for height in (round(y * ratio) for ratio in SCALES):
            left, top = np.arange(offset, x - offset - width), np.arange(offset, y - offset - height)
            right, bottom = left + width, top + height
            sides = (vertical[np.ix_(bottom, left)] - vertical[np.ix_(top, left)]) / height
            sides *= (vertical[np.ix_(bottom, right)] - vertical[np.ix_(top, right)]) / height
            sides *= (horizontal[np.ix_(top, right)] - horizontal[np.ix_(top, left)]) / width
            sides *= (horizontal[np.ix_(bottom, right)] - horizontal[np.ix_(bottom, left)]) / width
            row, col = np.unravel_index(int(sides.argmax()), sides.shape)
            score = float(max(0, sides[row, col]) ** .25)
            if score > best[0]:
                best = score, (int(left[col]), int(top[row]), int(right[col]), int(bottom[row]))
    return best


def local(source, alternative, box, *, feather=False):
    left, top, right, bottom = box
    region = np.s_[top:bottom, left:right]
    result = source.copy()
    if feather:
        y, x = np.ogrid[:bottom - top, :right - left]
        distance = np.minimum(np.minimum(x, right - left - 1 - x), np.minimum(y, bottom - top - 1 - y))
        alpha = .5 - .5 * np.cos(np.pi * np.minimum(distance / FEATHER_PX, 1.0))
        result[region] = np.rint(source[region] * (1 - alpha[:, :, None]) + alternative[region] * alpha[:, :, None]).astype(np.uint8)
    else:
        result[region] = alternative[region]
    return result


def control_views(source, processed):
    h, w = source.shape[:2]
    box = (w // 3, h // 3, w // 3 + w // 4, h // 3 + h // 4)
    photo = Image.fromarray(source)
    blur = local(source, np.asarray(photo.filter(ImageFilter.GaussianBlur(1.25))), box)
    tone = local(source, np.asarray(ImageEnhance.Contrast(photo).enhance(1.2)), box)
    return source, processed, blur, tone


def fitted_points(development):
    _, plan, attempts = earlier.checked_inputs(development)
    maxima = {fmt: [] for fmt in paired.FORMATS}
    for _, _, views, _ in earlier.panels(development, plan, attempts):
        controls = control_views(views["original"], views["processed"])
        for fmt in paired.FORMATS:
            maxima[fmt].append(max(rank(np.asarray(paired.encoded(rgb, fmt)[1]), fmt)[0] for rgb in controls))
    return {fmt: {"group_maxima": values, "point": sorted(values)[-2]} for fmt, values in maxima.items()}


def confirm(intake, points):
    for row, original_views, hard_truth in paired.inputs(intake):
        source, box = original_views["original"], row["box"]
        photo = Image.fromarray(source)
        views = original_views | {
            "feathered_edit": local(source, original_views["edited"], box, feather=True),
            "feathered_processing": local(source, original_views["local_processing"], box, feather=True),
            "local_contrast": local(source, np.asarray(ImageEnhance.Contrast(photo).enhance(1.2)), box),
        }
        soft_truth = np.any(views["feathered_edit"] != source, axis=2)
        assert 0 < int(soft_truth.sum()) < int(hard_truth.sum()) and not (soft_truth & ~hard_truth).any()
        truth = {"edited": hard_truth, "feathered_edit": soft_truth}
        for role, pixels in views.items():
            expected = truth.get(role, np.zeros_like(hard_truth))
            for fmt in paired.FORMATS:
                body, image = paired.encoded(pixels, fmt)
                score, found = rank(np.asarray(image), fmt)
                measured = earlier.measure(paired.boxed(found, hard_truth.shape, score > points[fmt]["point"]), expected)
                yield {"slot": row["slot"], "format": fmt, "role": role, "generator": row["generator"],
                       "input_sha256": shared.digest(body), "score": score, "region": found,
                       "measurements": {"boundary_prominence": measured}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("development", type=Path)
    parser.add_argument("intake", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    identity = {"schema": SCHEMA, "implementation_sha256": sha(Path(__file__)),
                "panel_implementation_sha256": sha(Path(paired.__file__)), "development_sha256": sha(args.development / "result.json"),
                "intake_sha256": sha(args.intake / "result.json"), "rect_scales": list(SCALES), "feather_px": FEATHER_PX,
                "fit_rule": "strictly above the second-largest source maximum on separate controls per format", "phase_context": PHASE_CONTEXT}
    if args.stage == "freeze":
        args.destination.mkdir(mode=0o700)
        write(args.destination / "freeze.json", identity)
        return
    assert identity == json.loads((args.destination / "freeze.json").read_bytes())
    points = fitted_points(args.development)
    write(args.destination / "points.json", points)
    rows = list(confirm(args.intake, points))
    assert len(rows) == 192 and len({(r["slot"], r["format"], r["role"]) for r in rows}) == len(rows)
    write(args.destination / "rows.json", rows)
    aggregate = earlier.summarize([{**row, "delivery": row["format"], "role": "edited"} for row in rows if row["role"] == "feathered_edit"])
    write(args.destination / "result.json", {"schema": SCHEMA, "views": len(rows), "freeze_sha256": sha(args.destination / "freeze.json"),
          "points_sha256": sha(args.destination / "points.json"), "journal_sha256": sha(args.destination / "rows.json"),
          "hard_and_controls": paired.summarise([row for row in rows if row["role"] != "feathered_edit"]),
          "feathered": aggregate})


if __name__ == "__main__":
    main()
