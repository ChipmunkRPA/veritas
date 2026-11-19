"""Recount the preregistered clean and edited HEVC ordinary-control panel."""

import json
import sys
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.video-short-face-hevc-ordinary/1"
POINT = (0.20).hex()
SCOPE = {
    "rate": "25/1",
    "resolution": [1280, 720],
    "frames": 120,
    "frame_ticks": 3600,
    "time_base": "1/90000",
}
GATE = {
    "minimum_measured_ordinary_groups": 6,
    "maximum_ordinary_flags": 1,
    "maximum_cut_flags": 0,
}
ROLES = ("untouched", "gaussian", "disc", "cut")


def require(test, message):
    if not test:
        raise ValueError(message)


def review(doc):
    require(
        doc["schema"] == SCHEMA
        and doc["point"] == POINT
        and doc["scope"] == SCOPE
        and doc["gate"] == GATE,
        "Expected frozen comparison",
    )
    require(
        doc["groups"] == 9
        and doc["eligible_pool"] == 19
        and doc["decoded_rgb_pts"] == 3840
        and doc["raw_scene_screen_hits"] == 0,
        "Input proof counts differ",
    )
    roles = {k: Counter() for k in ROLES}
    seen = set()
    measured_groups = set()
    native = 0
    cut_count = 0
    positions = {}
    for row in doc["observations"]:
        role, group = row["role"], row["group"]
        require(
            type(group) is int
            and group in range(1, 10)
            and role in ROLES
            and (group, role) not in seen,
            "Duplicate or unknown ordinary case",
        )
        seen.add((group, role))
        stages = row["authored_boundaries"]
        if role == "untouched":
            require(stages is None, "An original-only control has a staged edit")
        else:
            require(
                isinstance(stages, list)
                and len(stages) == 2
                and 61 <= stages[0] < stages[1] <= 120
                and (
                    stages[1] == 120
                    if role == "cut"
                    else stages[1] - stages[0] in (3, 4)
                ),
                "Changed native treatment clock",
            )
            if group not in positions:
                positions[group] = stages[0]
            require(
                stages[0] == positions[group],
                "Paired treatments do not share a native start",
            )
        proposed, measured = row["proposals"], row["measured"]
        require(
            type(proposed) is type(measured) is int and 0 <= measured <= proposed <= 8,
            "Invalid comparable-face coverage",
        )
        require(
            type(row["baseline_flag"]) is bool
            and type(row["cut_seen"]) is bool
            and (role == "cut" or not row["cut_seen"]),
            "Invalid incumbent or cut count",
        )
        c = roles[role]
        c["files"] += 1
        c["incumbent"] += row["baseline_flag"]
        c["has_face_views"] += bool(measured)
        c["shortlisted"] += bool(proposed)
        native_range = row["supplemental"]
        c["added"] += native_range is not None
        cut_count += row["cut_seen"]
        if role != "cut" and measured:
            measured_groups.add(group)
        if native_range:
            a, b = native_range["range"]
            require(
                measured
                and 6 <= a < b < 115
                and b - a in (3, 4)
                and native_range["clock"] == [a * 3600, b * 3600]
                and native_range["anchors"] == [a - 1, a + (b - a) // 2, b]
                and 0.20 < float.fromhex(native_range["reading"]) <= 2.0,
                "A reported range or its above-point native evidence differs",
            )
            native += 3
    require(
        seen == {(g, r) for g in range(1, 10) for r in ROLES if r != "cut" or g % 2},
        "An authored condition is missing",
    )
    flags = sum(c["added"] for role, c in roles.items() if role != "cut")
    passed = (
        len(measured_groups) >= GATE["minimum_measured_ordinary_groups"]
        and flags <= GATE["maximum_ordinary_flags"]
        and roles["cut"]["added"] <= GATE["maximum_cut_flags"]
    )
    return {
        "by_role": {r: dict(v) for r, v in roles.items()},
        "ordinary_with_measured_groups": len(measured_groups),
        "ordinary_added_flags": flags,
        "correct_preexisting_cut_views": cut_count,
        "native_frame_witnesses": native,
        "passes_ordinary_confirmation": passed,
    }


if __name__ == "__main__":
    d = json.loads(Path(sys.argv[1]).read_bytes())
    result = review(d)
    require(result == d["summary"], "Recount differs from the stored result")
    print(json.dumps(result, sort_keys=True))
