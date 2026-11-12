# SPDX-License-Identifier: Apache-2.0
"""Recompute the small boundary stress decisions and audit local native crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

PASTES = {"self_paste", "other_paste"}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(document) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()


def require(valid, message: str) -> None:
    if not valid:
        raise ValueError(message)


def select(row, policy):
    width, height = row["size"]
    fixed, extra = row["fixed"], row["combined"]
    for arm in (fixed, extra):
        x, y, right, bottom = arm["proposal"]
        require(0 <= x < right <= width and 0 <= y < bottom <= height and math.isfinite(arm["score"]) and arm["score"] >= 0,
                "Invalid recorded proposal")
    old = fixed["proposal"] if fixed["score"] > policy["fixed_point"] else None
    x, y, right, bottom = extra["proposal"]
    method = policy["method"]
    w, h = right - x, bottom - y
    admitted = (extra["score"] > method["point"] and method["minimum_side"] <= min(w, h)
                and w <= width * method["maximum_axis_fraction"] and h <= height * method["maximum_axis_fraction"])
    return old, old or (extra["proposal"] if admitted else None)


def geometry(row, entry, expected):
    require((entry is None) == (expected is None), "Different selection state")
    if entry is None:
        return False
    x, y, right, bottom = expected
    width, height = row["size"]
    area = (right - x) * (bottom - y)
    pad = max(24, (12 * max(right - x, bottom - y) + 50) // 100)
    crop = [max(0, x - pad), max(0, y - pad), min(width, right + pad), min(height, bottom + pad)]
    changed = row["target_pixels"]
    overlap = entry["overlap"]
    require(entry["box"] == expected and entry["area"] == area and entry["crop"] == crop
            and type(overlap) is int and 0 <= overlap <= min(changed, area), "Different source geometry")
    return changed > 0 and 10 * overlap >= 3 * (area + changed - overlap) and 2 * overlap >= changed


def audit_rows(frozen, rows):
    cells = defaultdict(Counter)
    gained, control_groups, original_groups, graphic_groups = set(), set(), set(), set()
    wanted = {(cohort, slot, role) for cohort, count, roles in (
        ("photo", len(frozen["roster"]["sources"]) // 2, frozen["photo_roles"]),
        ("graphic", frozen["graphics_groups"], frozen["graphic_roles"])) for slot in range(count) for role in roles}
    for row in rows:
        group = (row["cohort"], row["slot"])
        role = row["role"]
        case = (*group, role)
        require(case in wanted, "Unknown or duplicate case")
        wanted.remove(case)
        old, new = select(row, frozen)
        fixed, combined = row["fixed"]["selected"], row["combined"]["selected"]
        useful_old, useful_new = geometry(row, fixed, old), geometry(row, combined, new)
        require(fixed is None or fixed == combined, "An incumbent evidence crop changed")
        require((role in PASTES) == (row["target"] is not None and row["target_pixels"] > 0), "Invalid truth scope")
        added = old is None and new is not None
        cells[role].update({"views": 1, "fixed_flags": old is not None, "combined_flags": new is not None,
                            "fixed_useful": useful_old, "combined_useful": useful_new, "added_flags": added,
                            "added_useful": added and useful_new, "added_wrong_pastes": added and role in PASTES and not useful_new})
        if added and useful_new:
            gained.add(group)
        if added and role not in PASTES:
            control_groups.add(group)
            if role == "original":
                original_groups.add(group)
            if group[0] == "graphic":
                graphic_groups.add(group)
    require(not wanted, "Missing cases")
    method = frozen["method"]
    added_useful = sum(cells[role]["added_useful"] for role in PASTES)
    added_controls = sum(row["added_flags"] for role, row in cells.items() if role not in PASTES)
    wrong_pastes = sum(cells[role]["added_wrong_pastes"] for role in PASTES)
    passed = {
        "additional_useful": added_useful >= method["minimum_additional_useful"],
        "additional_useful_groups": len(gained) >= method["minimum_additional_groups"],
        "additional_per_paste_type": all(cells[role]["added_useful"] >= method["minimum_additional_per_paste_type"] for role in PASTES),
        "control_views": added_controls <= method["maximum_added_control_views"],
        "control_groups": len(control_groups) <= method["maximum_added_control_groups"],
        "untouched_photo_groups": len(original_groups) <= method["maximum_added_untouched_photo_groups"],
        "graphic_groups": len(graphic_groups) <= method["maximum_added_graphic_groups"],
        "wrong_paste_locations": wrong_pastes <= method["maximum_wrong_added_paste_views"],
        "all_incumbent_crops_preserved": True,
    }
    return {"by_role": cells, "added_useful": added_useful, "added_useful_groups": len(gained),
            "added_control_views": added_controls, "added_control_groups": len(control_groups),
            "added_untouched_photo_groups": len(original_groups), "added_graphic_groups": len(graphic_groups),
            "added_wrong_paste_views": wrong_pastes, "criteria": passed, "qualifies": all(passed.values())}


def audit_images(run, rows):
    from PIL import Image

    verified_crops = 0
    for row in rows:
        cohort, slot, role = row["cohort"], row["slot"], row["role"]
        folder = run / "inputs"
        path = folder / f"{cohort}-{slot:02d}-{role}.png"
        require(sha(path.read_bytes()) == row["input_sha256"], "Delivery has different bytes")
        with Image.open(path) as image:
            width, height = image.size
            require(image.mode == "RGB" and [width, height] == row["size"], "Different delivery geometry")
            raw = image.tobytes()
            if row["target"]:
                with Image.open(folder / f"photo-{slot:02d}-original.png") as source:
                    earlier = source.tobytes()
                mask = bytes(raw[i:i + 3] != earlier[i:i + 3] for i in range(0, len(raw), 3))
                x, y, right, bottom = row["target"]
                require(sum(mask) == sum(sum(mask[j * width + x:j * width + right]) for j in range(y, bottom)),
                        "Native changes escaped the target")
            else:
                mask = bytes(width * height)
            require(sha(mask) == row["truth_sha256"] and sum(mask) == row["target_pixels"], "Native truth differs")
            for arm in ("fixed", "combined"):
                if (entry := row[arm]["selected"]) is None:
                    continue
                require(sha(image.crop(entry["crop"]).tobytes()) == entry["crop_rgb_sha256"], "Native crop differs")
                x, y, right, bottom = entry["box"]
                overlap = sum(sum(mask[j * width + x:j * width + right]) for j in range(y, bottom))
                require(overlap == entry["overlap"], "Native changed-pixel overlap differs")
                verified_crops += 1
    return {"native_deliveries": len(rows), "native_crops": verified_crops}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local run directory or published JSON ledger")
    parser.add_argument("--publish", type=Path, help="save audited aggregate and ledger from a local run")
    args = parser.parse_args()
    if args.source.is_file():
        package = json.loads(args.source.read_bytes())
        require(sha(encoded(package["freeze"])) == package["audit"]["freeze_sha256"]
                and sha(encoded(package["rows"])) == package["audit"]["rows_sha256"], "Published commitments differ from the ledger")
        summary = audit_rows(package["freeze"], package["rows"])
        require(summary == package["summary"], "Published counts differ from the ledger")
    else:
        freeze = (args.source / "freeze.json").read_bytes()
        raw_rows = (args.source / "rows.json").read_bytes()
        frozen, rows = json.loads(freeze), json.loads(raw_rows)
        summary = audit_rows(frozen, rows)
        native = audit_images(args.source, rows)
        if args.publish:
            package = {"schema": frozen["schema"], "freeze": frozen, "rows": rows, "summary": summary,
                       "audit": native | {"freeze_sha256": sha(freeze), "rows_sha256": sha(raw_rows)}}
            args.publish.write_bytes(encoded(package))
        print(json.dumps(native, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
