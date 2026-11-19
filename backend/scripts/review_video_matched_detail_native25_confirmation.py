"""Recount the anonymous 25 fps matched-detail comparison using only Python."""

import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

SCOPE = {
    "fps": 25,
    "frame_count": 120,
    "width": 1280,
    "height": 720,
    "time_base": "1/90000",
    "ticks_per_picture": 3600,
}
SCHEMA = "chipmunk.video-matched-detail-native25-confirmation/1"
POINT = 0.10
GATE = {
    "min_new_exact_compounds": 2,
    "min_added_groups": 2,
    "max_new_ordinary": 1,
    "max_wrong": 1,
    "min_measured_ordinary": 8,
    "min_measured_ordinary_groups": 6,
    "min_each_measured_blur_family": 3,
    "require_prior_ranges_preserved": True,
}
ROLES = {
    "swap",
    "swap_plus_blur",
    "ordinary_blur",
    "ordinary_alternate_blur",
    "ordinary_same_take_jump",
    "untouched",
    "hard_cut",
}
POSITIVE = {"swap", "swap_plus_blur"}
STATUSES = {"localized", "no_match", "unassessed"}
EARLIER = {"face_sequence", "face_boundary", "face_return", "face_return_secondary"}
DIGEST = re.compile(r"[0-9a-f]{64}")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def interval(r, length=None):
    require(
        r is None
        or isinstance(r, list)
        and len(r) == 2
        and all(type(i) is int for i in r)
        and 0 <= r[0] < r[1] <= 120
        and (length is None or r[1] - r[0] in length),
        "Invalid native bounds",
    )
    return r


def useful(r, t):
    return bool(
        r
        and t
        and Fraction(
            max(0, min(r[1], t[1]) - max(r[0], t[0])), max(r[1], t[1]) - min(r[0], t[0])
        )
        >= Fraction(1, 2)
    )


def verify(panel):
    require(
        set(panel)
        == {
            "schema",
            "scope",
            "fixed_point_hex",
            "requirements",
            "source_groups",
            "commitments",
            "evidence_totals",
            "observations",
            "result",
        },
        "Unexpected transfer-record fields",
    )
    require(
        panel["schema"] == SCHEMA
        and panel["scope"] == SCOPE
        and panel["fixed_point_hex"] == POINT.hex()
        and panel["requirements"] == GATE,
        "Frozen comparison does not match",
    )
    keys = {
        "design",
        "author",
        "media",
        "full_face_stage_and_candidate",
        "independent_media",
        "independent_crop_replay",
    }
    require(
        set(panel["commitments"]) == keys
        and all(DIGEST.fullmatch(v) for v in panel["commitments"].values()),
        "Missing comparison commitments",
    )
    groups = panel["source_groups"]
    require(
        len(groups) == 8 and [g["group"] for g in groups] == list(range(1, 9)),
        "Require eight distinct anonymous groups",
    )
    by_group = {}
    blur_family = {}
    alternate = {}
    for g in groups:
        require(
            set(g) == {"group", "placement", "alternate_blur", "blur", "renderer"}
            and g["blur"] in {"gaussian", "disc"}
            and g["renderer"] in {"first", "second"},
            "Unexpected group fields",
        )
        by_group[g["group"]] = interval(g["placement"], {3, 4})
        alternate[g["group"]] = interval(g["alternate_blur"], {3, 4})
        blur_family[g["group"]] = g["blur"]
        require(
            alternate[g["group"]]
            and max(g["placement"][0], g["alternate_blur"][0])
            >= min(g["placement"][1], g["alternate_blur"][1]),
            "Ordinary comparison overlaps its matched placement",
        )
    require(
        Counter(g["blur"] for g in groups) == {"gaussian": 4, "disc": 4}
        and Counter(g["renderer"] for g in groups) == {"first": 4, "second": 4}
        and Counter(b - a for a, b in by_group.values()) == {3: 4, 4: 4},
        "Frozen treatment balance differs",
    )
    roles = defaultdict(Counter)
    seen = set()
    gains = set()
    totals = Counter()
    ordinary_group = set()
    by_blur = Counter()
    for r in panel["observations"]:
        require(
            set(r)
            == {
                "group",
                "role",
                "expected",
                "prior",
                "addition",
                "response_hex",
                "coverage_status",
                "candidate_pairs",
                "measured_pairs",
            },
            "Unexpected measurement fields",
        )
        k, role = r["group"], r["role"]
        require(
            k in by_group and role in ROLES and (k, role) not in seen,
            "Duplicate or unknown source role",
        )
        seen.add((k, role))
        t = interval(r["expected"], {3, 4})
        require(
            t == (by_group[k] if role in POSITIVE else None),
            "Expected treatment differs from its locked source",
        )
        old = r["prior"]
        require(
            isinstance(old, dict)
            and set(old) <= EARLIER
            and all(interval(v) for v in old.values()),
            "Invalid comparator range list",
        )
        n = interval(r["addition"], {3, 4})
        status = r["coverage_status"]
        cand, views = r["candidate_pairs"], r["measured_pairs"]
        require(
            status in STATUSES
            and type(cand) is type(views) is int
            and 0 <= views <= cand <= 8
            and (status == "localized") == bool(n)
            and (status != "unassessed") == bool(views),
            "Inconsistent comparison coverage",
        )
        if n:
            response = float.fromhex(r["response_hex"])
            require(
                r["response_hex"] == response.hex() and POINT < response <= 2,
                "Invalid fixed response",
            )
        else:
            require(r["response_hex"] is None, "Unlocalized row includes a response")
        require(
            not n or not {"face_return", "face_return_secondary"} & old.keys(),
            "Earlier brief review must take precedence",
        )
        x = roles[role]
        x["files"] += 1
        x["before_exact"] += bool(t and t in old.values())
        x["before_useful"] += any(useful(o, t) for o in old.values())
        x["with_candidate_exact"] += bool(t and (t == n or t in old.values()))
        x["with_candidate_useful"] += useful(n, t) or any(
            useful(o, t) for o in old.values()
        )
        x["before_any"] += bool(old)
        x["with_candidate_any"] += bool(old or n)
        x["new_ranges"] += bool(n)
        x["unassessed"] += status == "unassessed"
        if role in POSITIVE:
            totals["wrong_location"] += bool(n and not useful(n, t))
            if role == "swap_plus_blur" and t == n and t not in old.values():
                gains.add(k)
                totals["exact_compound_gains"] += 1
        else:
            totals["ordinary_added_ranges"] += bool(n)
            measured = status != "unassessed"
            totals["ordinary_measured"] += measured
            if measured:
                ordinary_group.add(k)
                family = blur_family[k]
                if role == "ordinary_blur":
                    by_blur[family] += 1
                elif role == "ordinary_alternate_blur":
                    by_blur["gaussian" if family == "disc" else "disc"] += 1
    require(
        seen == {(k, role) for k in by_group for role in ROLES},
        "Incomplete measured matrix",
    )
    totals["groups_with_exact_compound_gain"] = len(gains)
    totals["measured_ordinary_groups"] = len(ordinary_group)
    totals["measured_ordinary_gaussian"] = by_blur["gaussian"]
    totals["measured_ordinary_disc"] = by_blur["disc"]
    totals["gate_pass"] = (
        totals["exact_compound_gains"] >= GATE["min_new_exact_compounds"]
        and len(gains) >= GATE["min_added_groups"]
        and (
            totals["wrong_location"] <= GATE["max_wrong"]
            and totals["ordinary_added_ranges"] <= GATE["max_new_ordinary"]
            and totals["ordinary_measured"] >= GATE["min_measured_ordinary"]
            and len(ordinary_group) >= GATE["min_measured_ordinary_groups"]
            and all(
                by_blur[m] >= GATE["min_each_measured_blur_family"]
                for m in ("gaussian", "disc")
            )
        )
    )
    summary = {
        "totals": dict(totals),
        "by_condition": {role: dict(n) for role, n in roles.items()},
    }
    e = panel["evidence_totals"]
    require(
        e
        == {
            "delivered_native_rgb_timestamps": len(seen) * 120,
            "authored_edit_frames": sum(b - a for a, b in by_group.values()),
            "display_anchors": sum(n["new_ranges"] for n in roles.values()) * 3,
            "all_replayed_matched_detail_crops": sum(
                sum(
                    r["addition"][1] - r["addition"][0] + 6
                    for r in panel["observations"]
                    if r["role"] == role and r["addition"]
                )
                for role in ROLES
            ),
        },
        "Source evidence totals do not agree with the displayed ranges",
    )
    return summary


if __name__ == "__main__":
    panel = json.loads(Path(sys.argv[1]).read_bytes())
    result = verify(panel)
    require(result == panel["result"], "Published totals differ from the fixed recount")
    print(json.dumps(result, indent=2, sort_keys=True))
