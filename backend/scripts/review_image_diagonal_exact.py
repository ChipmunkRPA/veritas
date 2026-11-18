"""Recount the two exact-diagonal image trials without the photographs."""

import argparse
import gzip
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

SCHEMA = "chipmunk.image-diagonal-reflection-review/1"
TRANSFORMS = {"top_left_to_bottom_right", "top_right_to_bottom_left"}
CLEAR = {"no_pair", "unsupported_geometry"}
DISCLOSURE = "This shows one verified pair. Other copies may be present."
EDIT_ROLES = {"diagonal", "anti_diagonal", "feathered_diagonal"}
CONTROL_ROLES = {
    "unchanged",
    "tone",
    "blur",
    "ordinary_axis",
    "ordinary_turn",
    "whole_diagonal",
    "competing",
    "similar_nonexact",
}
NATIVE_COUNTS = {
    "initial": (176, 48, 42),
    "confirmation": (148, 36, 36),
}
PROFILES = {
    "initial": {
        "groups": 12,
        "edits": 24,
        "ordinary": 56,
        "competing": 8,
        "required_useful": 12,
        "required_groups": 5,
        "allowed_ordinary": 1,
    },
    "confirmation": {
        "groups": 10,
        "edits": 18,
        "ordinary": 50,
        "competing": 6,
        "required_useful": 10,
        "required_groups": 4,
        "allowed_ordinary": 0,
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def area(box):
    return box[2] * box[3]


def iou(a, b):
    w = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    h = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return Fraction(w * h, area(a) + area(b) - w * h)


def useful(pairs, truth, counts, total):
    if not truth:
        require(not counts and total is None, "Control has positive labels")
        return False
    require(len(counts) == len(pairs) and 0 < total <= area(truth[1]), "Invalid overlap ledger")
    matches = []
    for (a, b), (masked,) in zip(pairs, counts, strict=True):
        best = max(min(iou(a, x), iou(b, y)) for x, y in (truth, truth[::-1]))
        require(0 <= masked <= min(total, max(area(a), area(b))), "Invalid covered pixel count")
        matches.append(best >= Fraction(1, 2) and 2 * masked >= total)
    return any(matches)


def row_result(row, profile):
    size = row["size"]
    require(len(size) == 2 and all(512 <= k <= 672 for k in size), "Invalid native image geometry")
    previous = [pair for check in row["old"] for pair in check["pairs"]]
    trial = row["candidate"]
    proposed = [item["pair"] for item in trial["pairs"]]
    require(
        len(proposed) <= 1 and bool(proposed) == (trial["status"] == "supported_pair"),
        "Invalid candidate result",
    )
    for item in trial["pairs"]:
        first, second = item["pair"]
        for x, y, w, h in (first, second):
            require(
                16 <= min(w, h) <= max(w, h) <= 56
                and x >= 3
                and y >= 3
                and x + w + 3 <= size[0]
                and y + h + 3 <= size[1],
                "Invalid candidate crop",
            )
        require(
            item["transform"] in TRANSFORMS and first[-2:] == second[-2:][::-1] and iou(first, second) == 0,
            "Invalid diagonal geometry",
        )
        witnesses = item["measurement"]
        rim = witnesses["perimeter_mean_absolute_difference"]
        require(
            witnesses["matching_pixels"] == area(first)
            and 64 <= witnesses["different_color_values"] <= area(first)
            and len(rim) == 4
            and sum(value >= 3 for value in rim) >= 3,
            "Invalid pixel witness",
        )
    admitted = (
        bool(proposed)
        and not previous
        and row["axis_status"] == "no_pair"
        and all(check["status"] in CLEAR for check in row["old"])
    )
    require(admitted == row["admitted"], "Candidate routing differs")
    if profile == "confirmation":
        require(row["disclosure"] == (DISCLOSURE if admitted else None), "Disclosure differs")
    before = useful(previous, row["truth"], row["old_overlap"], row["mask_pixels"])
    after = admitted and useful(proposed, row["truth"], row["proposed_overlap"], row["mask_pixels"])
    correct_guard = True
    if row["role"] == "competing" and admitted:
        (entry,) = trial["pairs"]
        correct_guard = any(
            entry["transform"] == token and sorted(entry["pair"]) == sorted(pair)
            for token, pair in row["planted_competition"]
        )
    return previous, admitted, bool(before), bool(after), correct_guard


def profile_result(rows, profile):
    fixed = PROFILES[profile]
    seen = set()
    paired = {}
    by_carrier = {}
    for carrier in ("PNG", "WEBP"):
        data = Counter()
        groups = set()
        useful_groups = set()
        for row in rows:
            if row["carrier"] != carrier:
                continue
            group, role = row["group"], row["role"]
            require(
                type(group) is int and 1 <= group <= fixed["groups"] and role in EDIT_ROLES | CONTROL_ROLES,
                "Unknown source group or delivery role",
            )
            require(bool(row["truth"]) == (role in EDIT_ROLES), "Edit labels disagree with the frozen role")
            matched = tuple(
                row[name] for name in ("size", "truth", "mask_pixels", "candidate", "proposed_overlap")
            )
            if (group, role) in paired:
                require(paired[group, role] == matched, "Paired lossless measurements differ")
            paired[group, role] = matched
            key = group, role, carrier
            require(key not in seen, "Repeated delivery")
            seen.add(key)
            groups.add(row["group"])
            previous, admitted, before, after, correct_guard = row_result(row, profile)
            data["rows"] += 1
            if row["truth"]:
                data["edits"] += 1
                data["baseline_useful"] += before
                data["new_useful"] += after
                data["new_wrong"] += admitted and not after
                if after:
                    useful_groups.add(row["group"])
            else:
                guarding = row["role"] == "competing"
                data["prior_existing_controls"] += bool(previous)
                data["competing" if guarding else "ordinary"] += 1
                data["added_competing" if guarding else "added_ordinary"] += admitted
                data["incorrect_competing"] += not correct_guard
                data["added_whole_symmetry"] += admitted and row["role"] == "whole_diagonal"
                data["withheld_competing"] += guarding and row["candidate"]["status"] == "ambiguous_pairs"
        require(
            len(groups) == fixed["groups"]
            and all(data[key] == fixed[key] for key in ("edits", "ordinary", "competing")),
            "Receipt population differs from freeze",
        )
        data["new_groups"] = len(useful_groups)
        ordinary = data["added_ordinary"]
        if profile == "initial":
            ordinary += data["added_competing"]
        gates = {
            "useful": data["new_useful"] >= fixed["required_useful"],
            "groups": data["new_groups"] >= fixed["required_groups"],
            "wrong": data["new_wrong"] <= (1 if profile == "initial" else 0),
            "ordinary": ordinary <= fixed["allowed_ordinary"],
            "symmetry": data["added_whole_symmetry"] == 0,
            "competition": data["added_competing" if profile == "initial" else "incorrect_competing"] == 0,
        }
        by_carrier[carrier] = {"counts": dict(data), "gates": gates, "passed": all(gates.values())}
    require(len(seen) == len(rows) == 2 * len(paired), "Unknown or missing carrier")
    return by_carrier


def review(receipt):
    require(receipt["schema"] == SCHEMA, "Unsupported receipt")
    for profile, expected in NATIVE_COUNTS.items():
        audit = receipt["independent_native_audit_counts"][profile]
        require(
            tuple(
                audit[field]
                for field in (
                    "deliveries_verified",
                    "edited_masks_verified",
                    "source_pixel_pairs_verified",
                )
            )
            == expected,
            "Native audit counts differ",
        )
    return {profile: profile_result(receipt["measurements"][profile], profile) for profile in PROFILES}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    args = p.parse_args()
    try:
        content = args.receipt.read_bytes()
        if args.receipt.suffix == ".gz":
            content = gzip.decompress(content)
        receipt = json.loads(content)
        result = review(receipt)
        require(result == receipt["recount"], "Receipt summary does not match row-level measurements")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
