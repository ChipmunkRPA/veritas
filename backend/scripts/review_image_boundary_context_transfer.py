# SPDX-License-Identifier: Apache-2.0
"""Independently read and replay boundary context measurements and source crops."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_image_small_boundary_control_stress import audit_images, encoded, geometry, require, sha  # noqa: E402
from review_image_small_boundary_screenshot_admission import check_box, constant_blocks  # noqa: E402

ARMS = ("fixed", "broad", "flat", "alpha", "periodic", "context", "combined")
POSITIVES = {"sharp_paste", "repeat_paste", "tint_edit", "feathered_paste"}
CONTROLS = {"original", "global_tone", "feathered_blur", "resize_roundtrip"}


def decisions(frozen, row):
    fixed, broad = row["fixed"], row["combined"]
    for arm in (fixed, broad):
        check_box(arm["proposal"], row["size"], arm["score"])
    old_box = fixed["proposal"] if fixed["score"] > frozen["fixed_point"] else None
    x, y, right, bottom = broad["proposal"]
    w, h = right - x, bottom - y
    width, height = row["size"]
    can_use = (broad["score"] > frozen["fallback_point"] and min(w, h) >= frozen["minimum_side"]
               and w <= width * frozen["maximum_axis_fraction"] and h <= height * frozen["maximum_axis_fraction"])
    broad_box = old_box or (broad["proposal"] if can_use else None)
    fixed_useful = geometry(row, fixed["selected"], old_box)
    broad_useful = geometry(row, broad["selected"], broad_box)
    require(old_box is None or fixed["selected"] == broad["selected"], "The existing source crop changed")
    context = row["context"]
    require((context is not None) == (broad_box is not None and old_box is None), "Unexpected contextual score")
    flat = alpha = period = False
    if context:
        fraction = context["flat_share"]
        require(type(fraction) is float and 0 <= fraction <= 1, "Invalid flat-block share")
        flat = fraction >= frozen["criteria"]["previous_interior_flat_floor"]
        config = frozen["alpha"]
        sides = context["alpha_sides"]
        require(not sides or [side["side"] for side in sides] == [0, 1, 2, 3], "Alpha side order differs")
        usable = [side["beta"] for side in sides if (
            side["outer_std"] >= config["outer_std_min"] and config["beta_min"] <= side["beta"] <= config["beta_max"]
            and all(-config["matte_slack"] <= v <= 255 + config["matte_slack"] for v in side["matte"])
            and side["gain"] >= config["held_gain_min"] and side["median_error"] <= config["held_median_error_max"])]
        alpha = len(usable) >= config["minimum_held_sides"] and max(usable) - min(usable) <= config["max_beta_spread"]
        record = context["periodic"]
        point = frozen["periodic"]
        for candidate in record["proposals"]:
            dy, dx = candidate["lag"]
            require(0 <= dy <= point["search_radius"] and abs(dx) <= point["search_radius"]
                    and max(dy, abs(dx)) >= point["minimum_lag"] and (dy > 0 or dx >= 0)
                    and point["min_correlation"] <= candidate["one"] <= 1 and point["min_correlation"] <= candidate["two"] <= 1
                    and -1 <= candidate["outer"] <= 1 and candidate["outer_pairs"] >= point["minimum_outer_pairs"], "Invalid period lag record")
        period = record["interior_rms"] >= point["min_interior_rms"] and any(
            lag["outer"] <= point["max_outer_correlation"] for lag in record["proposals"])
    withheld = (False, False, flat, alpha, period, alpha or period, flat or alpha or period)
    flags = dict(zip(ARMS, (old_box is not None, *(broad_box is not None and not gate for gate in withheld[1:])), strict=True))
    useful = dict(zip(ARMS, (fixed_useful, *(broad_useful and not gate for gate in withheld[1:])), strict=True))
    return flags, useful


def summarize(frozen, seal, candidate):
    require(seal["freeze_sha256"] == sha(encoded(frozen)) and candidate["seal_sha256"] == sha(encoded(seal)), "Phase commitments differ")
    observed = {(row["slot"], row["type"]): row for row in frozen["captures"]["rows"]}
    eligible = {}
    panels, captions = defaultdict(Counter), defaultdict(Counter)
    for row in seal["all_views"]:
        case = row["slot"], row["type"]
        captured = observed.pop(case)
        require(all(row[key] == captured[key] for key in ("caption", "filename", "size")), "Screenshot binding differs")
        check_box(row["proposal"], row["size"], row["score"])
        cell = panels[row["type"]]
        cell["all_views"] += 1
        if row["score"] <= frozen["fixed_point"]:
            eligible[case] = row
            cell["incumbent_negative"] += 1
        else:
            cell["incumbent_flags"] += 1
    require(not observed and seal["eligible"] == [list(key) for key in eligible], "Incumbent-only stratum differs")
    screen_groups = {slot for slot, _ in eligible}
    target_count = len(frozen["photo_roster"]) // 3
    expected = {("photo", slot, role) for slot in range(target_count) for role in POSITIVES | CONTROLS}
    expected.update(("sign", slot, role) for slot in range(len(frozen["printed_signs"])) for role in ("printed_original", "printed_resize"))
    by_role = defaultdict(Counter)
    removed, screen_residue, ordinary_groups = set(), set(), set()
    for row in candidate["rows"]:
        role, slot, cohort = row["role"], row["slot"], row["cohort"]
        flags, useful = decisions(frozen, row)
        if cohort == "screenshot":
            old = eligible.pop((slot, role))
            require(row["caption"] == old["caption"] and row["target"] is None
                    and row["fixed"]["proposal"] == old["proposal"] and row["fixed"]["score"] == old["score"], "Sealed control differs")
            for cell in (panels[role], captions["captioned" if row["caption"] else "caption_free"]):
                cell.update({f"{arm}_flags": flagged for arm, flagged in flags.items()})
                cell["scored_views"] += 1
            if flags["flat"] and not flags["combined"]:
                removed.add(slot)
            if flags["combined"]:
                screen_residue.add(slot)
            continue
        expected.remove((cohort, slot, role))
        require((role in POSITIVES) == (row["target"] is not None and row["target_pixels"] > 0), "Unexpected target")
        totals = by_role[role]
        totals["views"] += 1
        totals.update({f"{arm}_flags": value for arm, value in flags.items()})
        totals.update({f"{arm}_useful": value for arm, value in useful.items()})
        totals["lost_by_combined"] += useful["broad"] and not useful["combined"]
        if role not in POSITIVES and flags["combined"] and not flags["fixed"]:
            ordinary_groups.add((cohort, slot))
    require(not expected and not eligible, "A planned measured view is missing")
    config = frozen["criteria"]
    gate = {
        "sealed_screens": len(screen_groups) >= config["minimum_incumbent_negative_screen_groups"],
        "challenged_screen_types": all(panels[k]["broad_flags"] >= config["minimum_broad_flags_per_challenge_type"] for k in ("alpha", "periodic")),
        "context_helps_beyond_flatness": len(removed) >= config["minimum_extra_context_resolved_screen_groups"],
        "residual_screen_groups": len(screen_residue) <= config["maximum_residual_screen_groups"],
        "preserved_standard_pastes": by_role["sharp_paste"]["lost_by_combined"] <= config["maximum_lost_standard_pastes"],
        "challenged_true_repeat_and_tint": all(by_role[k]["broad_useful"] >= config["minimum_broad_repeat_and_tint_locations"]
                                              for k in ("repeat_paste", "tint_edit")),
        "preserved_repeat_and_tint": sum(by_role[k]["lost_by_combined"] for k in ("repeat_paste", "tint_edit"))
                                     <= config["maximum_lost_repeat_or_tint_edits"],
        "ordinary_photo_and_signs": len(ordinary_groups) <= config["maximum_new_ordinary_groups"],
    }
    return {"panels": panels, "captions": captions, "photos": by_role, "incumbent_negative_screen_groups": len(screen_groups),
            "extra_screen_groups_resolved": len(removed), "residual_screen_groups": len(screen_residue),
            "new_ordinary_photo_or_sign_groups": len(ordinary_groups), "criteria": gate, "qualifies": all(gate.values())}


def check_alpha(image, box, recorded):
    import numpy as np

    x, y, right, bottom = box
    vertical, horizontal = slice(y + 6, bottom - 6), slice(x + 6, right - 6)
    pairs = ((image[vertical, x - 2], image[vertical, x + 1]),
             (image[y - 2, horizontal], image[y + 1, horizontal]),
             (image[vertical, right + 1], image[vertical, right - 2]),
             (image[bottom + 1, horizontal], image[bottom - 2, horizontal]))
    for held, item in enumerate(recorded):
        outer, inner = pairs[held]
        train_x = np.concatenate([pair[0] for j, pair in enumerate(pairs) if j != held])
        train_y = np.concatenate([pair[1] for j, pair in enumerate(pairs) if j != held])
        centered_x, centered_y = train_x - train_x.mean(0), train_y - train_y.mean(0)
        beta = float(np.linalg.lstsq(centered_x.reshape((-1, 1)), centered_y.ravel(), rcond=None)[0][0])
        offset = np.percentile(train_y - beta * train_x, 50, axis=0)
        matte = offset / (1 - beta) if beta < 1 else np.full(3, -1_000.)
        residual = inner - beta * outer - offset
        variance = np.linalg.norm(inner - inner.mean(0))
        gain = 1 - (np.linalg.norm(residual) / variance) ** 2 if variance > 0 else -1.
        observed = [beta, gain, np.median(np.abs(residual)), np.std(outer, axis=0).mean(), *matte]
        expected = [item["beta"], item["gain"], item["median_error"], item["outer_std"], *item["matte"]]
        require(np.allclose(observed, expected, atol=1.e-8), "Independently fitted alpha side differs")


def check_period(image, box, observed, config):
    import cv2
    import numpy as np

    left, top, right, bottom = box
    x, y = max(0, left - 24), max(0, top - 24)
    roi = image[y:bottom + 24, x:right + 24].astype(np.uint8)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY).astype(np.float64)
    indexes = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(indexes / 1.2) ** 2 / 2)
    kernel /= kernel.sum()
    residual = gray - cv2.sepFilter2D(gray, cv2.CV_64F, kernel, kernel)
    inset = config["inset"]
    inside = residual[top - y + inset:bottom - y - inset, left - x + inset:right - x - inset]
    rms = np.linalg.norm(inside) / math.sqrt(inside.size)
    require(math.isclose(rms, observed["interior_rms"], abs_tol=1.e-8), "Interior RMS differs")
    if rms < config["min_interior_rms"]:
        require(not observed["proposals"], "A flat interior has periodic proposals")
        return

    def shift(array, dy, dx):
        rolled = np.roll(array, (-dy, -dx), axis=(0, 1))
        rows, cols = array.shape
        valid = np.s_[max(0, -dy):min(rows, rows - dy), max(0, -dx):min(cols, cols - dx)]
        return array[valid], rolled[valid]

    def correlation(a, b):
        a, b = a - a.mean(), b - b.mean()
        scale = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a.ravel(), b.ravel()) / scale) if scale > 1.e-6 else 0.

    mask = np.ones(residual.shape, dtype=bool)
    mask[max(0, top - y - inset):bottom - y + inset, max(0, left - x - inset):right - x + inset] = False
    period = {tuple(row["lag"]): row for row in observed["proposals"]}
    radius = config["search_radius"]
    for dy in range(radius + 1):
        for dx in range(-radius, radius + 1):
            if max(abs(dy), abs(dx)) < config["minimum_lag"] or (dy == 0 and dx < 0):
                continue
            one = correlation(*shift(inside, dy, dx))
            if one < config["min_correlation"]:
                continue
            two = correlation(*shift(inside, 2 * dy, 2 * dx))
            if two < config["min_correlation"]:
                continue
            a, b = shift(mask, dy, dx)
            paired = a & b
            if int(paired.sum()) < config["minimum_outer_pairs"]:
                continue
            l, r = shift(residual, dy, dx)
            row = period.pop((dy, dx))
            require(row["outer_pairs"] == int(paired.sum())
                    and np.allclose([one, two, correlation(l[paired], r[paired])], [row["one"], row["two"], row["outer"]], atol=1.e-8),
                    "Independently recomputed periodic lag differs")
    require(not period, "Unexpected periodic lag")


def native_audit(run, frozen, candidate):
    import numpy as np
    from PIL import Image

    stats = audit_images(run, candidate["rows"])
    for row in frozen["captures"]["rows"]:
        path = run / "captures" / row["filename"]
        with Image.open(path) as image:
            require(sha(path.read_bytes()) == row["png_sha256"] and sha(image.convert("RGB").tobytes()) ==
                    frozen["capture_rgb_sha256"][row["filename"]], "Original capture differs")
    measured = 0
    for row in candidate["rows"]:
        path = run / "inputs" / f"{row['cohort']}-{row['slot']:02d}-{row['role']}.png"
        with Image.open(path) as source:
            if row["cohort"] == "screenshot":
                name = f"{row['slot']}-{row['role']}.png"
                require(sha(source.tobytes()) == frozen["capture_rgb_sha256"][name], "Scored capture pixels differ")
            if row["context"] is None:
                continue
            box = row["combined"]["selected"]["box"]
            x, y, right, bottom = box
            with source.crop((x + 4, y + 4, right - 4, bottom - 4)) as inner:
                require(constant_blocks(inner.tobytes(), *inner.size) == row["context"]["flat_share"], "Flat-block replay differs")
            rgb = np.asarray(source, dtype=np.float64)
            check_alpha(rgb, box, row["context"]["alpha_sides"])
            check_period(rgb, box, row["context"]["periodic"], frozen["periodic"])
            measured += 1
    return stats | {"native_capture_hashes": len(frozen["captures"]["rows"]), "independent_context_rows": measured}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--publish", type=Path)
    args = parser.parse_args()
    if args.source.is_file():
        package = json.loads(args.source.read_bytes())
        require(sha(encoded(package["candidate"])) == package["audit"]["candidate_sha256"], "Published context commitment differs")
        summary = summarize(package["freeze"], package["incumbent"], package["candidate"])
        require(summary == package["summary"], "Published counts differ")
    else:
        frozen, seal, candidate = (json.loads((args.source / f"{name}.json").read_bytes()) for name in ("freeze", "incumbent", "candidate"))
        summary = summarize(frozen, seal, candidate)
        audit = native_audit(args.source, frozen, candidate) | {"candidate_sha256": sha(encoded(candidate))}
        if args.publish:
            with args.publish.open("xb") as stream:
                stream.write(encoded({"schema": frozen["schema"], "freeze": frozen, "incumbent": seal, "candidate": candidate,
                                      "summary": summary, "audit": audit}))
        print(json.dumps(audit, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
