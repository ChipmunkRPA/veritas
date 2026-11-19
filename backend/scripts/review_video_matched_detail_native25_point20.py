"""Recount the two frozen appearance points on this anonymous native 25 fps panel."""

import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

POINTS = (0.10, 0.20)
SCHEMA = "chipmunk.video-matched-detail-native25-point20/1"
SCOPE = {
    "width": 1280,
    "height": 720,
    "frames": 120,
    "fps": 25,
    "ticks": 3600,
    "time_base": "1/90000",
}
PRIMARY = {
    "swap_plus_blur",
    "ordinary_blur",
    "ordinary_alternate_blur",
    "ordinary_gradual_defocus",
}
SUPPLEMENT = {"untouched", "hard_cut"}
GATE = {
    "min_new_exact_compounds": 3,
    "min_added_groups": 3,
    "max_new_ordinary": 1,
    "max_wrong": 0,
    "min_measured_ordinary": 8,
    "min_measured_ordinary_groups": 5,
    "min_measured_gaussian": 2,
    "min_measured_circular": 4,
    "require_all_low_point_exact_compounds": True,
    "require_prior_ranges_preserved": True,
}
KEYS = {"face_sequence", "face_boundary", "face_return", "face_return_secondary"}
SHA = re.compile(r"[a-f0-9]{64}")


def require(check, detail):
    if not check:
        raise ValueError(detail)


def interval(value, widths=None):
    require(
        value is None
        or isinstance(value, list)
        and len(value) == 2
        and all(type(i) is int for i in value)
        and 0 <= value[0] < value[1] <= 120
        and (widths is None or value[1] - value[0] in widths),
        "Invalid half-open bounds",
    )
    return value


def useful(window, truth):
    return bool(
        window
        and truth
        and Fraction(
            max(0, min(window[1], truth[1]) - max(window[0], truth[0])),
            max(window[1], truth[1]) - min(window[0], truth[0]),
        )
        >= Fraction(1, 2)
    )


def verify(data):
    require(
        set(data)
        == {
            "schema",
            "scope",
            "points_hex",
            "requirements",
            "commitments",
            "source_groups",
            "observations",
            "evidence",
            "totals",
        },
        "Unexpected public receipt fields",
    )
    require(
        data["schema"] == SCHEMA
        and data["scope"] == SCOPE
        and data["points_hex"] == [p.hex() for p in POINTS]
        and data["requirements"] == GATE,
        "Frozen panel profile differs",
    )
    commits = data["commitments"]
    require(
        set(commits)
        == {
            "development_first",
            "development_second",
            "design",
            "authored",
            "media",
            "scores",
            "independent_media",
            "crop_replay",
        }
        and all(type(x) is str and SHA.fullmatch(x) for x in commits.values()),
        "Missing frozen proof commitments",
    )
    groups = data["source_groups"]
    require(
        len(groups) == 6
        and all(type(row["group"]) is int for row in groups)
        and [r["group"] for r in groups] == list(range(1, 7)),
        "Invalid native source group roster",
    )
    locked = {}
    for g in groups:
        require(
            set(g)
            == {
                "group",
                "compound",
                "alternate_blur",
                "ordinary_ramp",
                "matched_kernel",
                "control",
                "renderer",
            },
            "Unexpected source group fields",
        )
        require(
            g["matched_kernel"] in {"gaussian", "disc"}
            and g["control"] in SUPPLEMENT
            and g["renderer"] in {"first", "second"},
            "Invalid role medley",
        )
        require(
            interval(g["compound"], {3, 4})
            and interval(g["alternate_blur"], {3, 4})
            and interval(g["ordinary_ramp"], {9}),
            "Missing source treatment",
        )
        require(
            not useful(g["compound"], g["alternate_blur"]),
            "The alternative defocus overlaps the compound interval",
        )
        locked[g["group"]] = g
    require(
        Counter(g["matched_kernel"] for g in groups) == {"gaussian": 3, "disc": 3}
        and Counter(g["control"] for g in groups) == {"untouched": 3, "hard_cut": 3}
        and Counter(g["renderer"] for g in groups) == {"first": 3, "second": 3}
        and Counter(g["compound"][1] - g["compound"][0] for g in groups)
        == {3: 3, 4: 3},
        "Treatment balances changed",
    )
    gains = [set(), set()]
    alerts = [set(), set()]
    wrong = [0, 0]
    roles = defaultdict(Counter)
    ordinary_groups = set()
    ordinary_kernels = Counter()
    seen = set()
    witness_count = [0, 0]
    crop_count = 0
    for o in data["observations"]:
        require(
            set(o)
            == {
                "group",
                "role",
                "truth",
                "ordinary_treatment",
                "full_ranges",
                "point10_range",
                "point20_range",
                "response_hex",
                "coverage",
            },
            "Unexpected case fields",
        )
        n, role = o["group"], o["role"]
        require(
            type(n) is int
            and n in locked
            and role in PRIMARY | {locked[n]["control"]}
            and (n, role) not in seen,
            "Unexpected or duplicate evaluated role",
        )
        seen.add((n, role))
        g = locked[n]
        expected = g["compound"] if role == "swap_plus_blur" else None
        treatment = {
            "ordinary_blur": g["compound"],
            "ordinary_alternate_blur": g["alternate_blur"],
            "ordinary_gradual_defocus": g["ordinary_ramp"],
            "hard_cut": [g["compound"][0], 120],
        }.get(role)
        require(
            o["truth"] == expected and o["ordinary_treatment"] == treatment,
            "Authored interval changed",
        )
        before = o["full_ranges"]
        require(
            isinstance(before, dict)
            and set(before) <= KEYS
            and all(interval(x) for x in before.values()),
            "Invalid Full ranges",
        )
        low, high = (
            interval(o[name], {3, 4}) for name in ("point10_range", "point20_range")
        )
        require(
            not low or not (before.keys() & {"face_return", "face_return_secondary"}),
            "An older brief range must take precedence",
        )
        status, reason, proposed, measured = (
            o["coverage"][name]
            for name in ("status", "reason", "candidate_pairs", "measured_pairs")
        )
        require(
            set(o["coverage"])
            == {
                "status",
                "reason",
                "candidate_pairs",
                "measured_pairs",
                "affects_overall_confidence",
            }
            and o["coverage"]["affects_overall_confidence"] is False
            and status in {"localized", "no_match", "unassessed"}
            and (status == "localized") == bool(low)
            and type(proposed) is type(measured) is int
            and 0 <= measured <= proposed <= 8
            and bool(measured) == (status != "unassessed")
            and (
                (
                    reason
                    in {"face_context", "shot_context", "prior_review", "time_budget"}
                )
                if status == "unassessed"
                else reason is None
            ),
            "Invalid measurement coverage",
        )
        if low:
            response = float.fromhex(o["response_hex"])
            require(
                o["response_hex"] == response.hex() and POINTS[0] < response <= 2,
                "Invalid appearance response",
            )
            witness_count[0] += 3
            crop_count += low[1] - low[0] + 6
            if high:
                witness_count[1] += 3
        else:
            require(
                o["response_hex"] is None,
                "A missing interval cannot possess a response",
            )
            response = None
        require(
            high == (low if response is not None and response > POINTS[1] else None),
            "The stricter point changed the interval",
        )
        r = roles[role]
        r["files"] += 1
        r["full_exact"] += bool(expected and expected in before.values())
        r["full_useful"] += any(useful(x, expected) for x in before.values())
        r["full_any"] += bool(before)
        r["measured"] += bool(measured)
        for arm, b in enumerate((low, high), 1):
            availability = bool(b)
            r[f"point{arm}0_any"] += bool(before or b)
            r[f"point{arm}0_exact"] += bool(
                expected and (expected == b or expected in before.values())
            )
            r[f"point{arm}0_useful"] += any(
                useful(x, expected) for x in before.values()
            ) or useful(b, expected)
            if expected:
                if expected == b and expected not in before.values():
                    gains[arm - 1].add(n)
                wrong[arm - 1] += availability and not useful(b, expected)
            elif b:
                alerts[arm - 1].add((n, role))
        if measured and not expected:
            ordinary_groups.add(n)
            if role == "ordinary_blur":
                ordinary_kernels[g["matched_kernel"]] += 1
            elif role in {"ordinary_alternate_blur", "ordinary_gradual_defocus"}:
                ordinary_kernels["disc"] += 1
    require(
        seen == {(k, r) for k, g in locked.items() for r in PRIMARY | {g["control"]}},
        "Incomplete case matrix",
    )
    evidence = {
        "native_rgb_and_pts": len(seen) * 120,
        "planted_frames": sum(g["compound"][1] - g["compound"][0] for g in groups),
        "point10_frames": witness_count[0],
        "point20_frames": witness_count[1],
        "replayed_crops": crop_count,
    }
    require(data["evidence"] == evidence, "Native evidence counts changed")
    measured_controls = sum(
        r["measured"] for role, r in roles.items() if role != "swap_plus_blur"
    )
    gate_pass = (
        len(gains[1]) >= GATE["min_new_exact_compounds"]
        and len(gains[1]) >= GATE["min_added_groups"]
        and gains[0] == gains[1]
        and len(alerts[1]) <= GATE["max_new_ordinary"]
        and wrong[1] <= GATE["max_wrong"]
        and measured_controls >= GATE["min_measured_ordinary"]
        and len(ordinary_groups) >= GATE["min_measured_ordinary_groups"]
        and ordinary_kernels["gaussian"] >= GATE["min_measured_gaussian"]
        and ordinary_kernels["disc"] >= GATE["min_measured_circular"]
    )
    return {
        "gate_pass": gate_pass,
        "cases": len(seen),
        "point10_exact_gains": len(gains[0]),
        "point20_exact_gains": len(gains[1]),
        "point10_added_ordinary": len(alerts[0]),
        "point20_added_ordinary": len(alerts[1]),
        "point10_wrong": wrong[0],
        "point20_wrong": wrong[1],
        "measured_controls": measured_controls,
        "measured_groups": len(ordinary_groups),
        "measured_gaussian": ordinary_kernels["gaussian"],
        "measured_disc": ordinary_kernels["disc"],
        "by_role": {role: dict(row) for role, row in roles.items()},
    }


if __name__ == "__main__":
    packet = json.loads(Path(sys.argv[1]).read_bytes())
    out = verify(packet)
    require(out == packet["totals"], "Recount differs from the published totals")
    print(json.dumps(out, indent=2, sort_keys=True))
