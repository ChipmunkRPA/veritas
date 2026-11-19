"""Recount the anonymous, frozen two-shot timing study with standard Python."""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROLES = (
    "ordinary_cut",
    "ordinary_pause",
    "constant_delay",
    "after_early",
    "after_late",
)
EDIT = {"after_early", "after_late"}
ORDINARY = {"ordinary_cut", "ordinary_pause"}
POINT = 0.10
GAP = 0.025
FRAME = 60
TICKS = 3600
STEP = 640
CLOCK = "1/90000"
SCHEMA = "chipmunk.video-shot-lag-local-confirmation/1"
REQUIREMENTS = {
    "minimum_correct_sign_and_exact_cut": 4,
    "minimum_groups": 3,
    "maximum_wrong_positive": 1,
    "maximum_ordinary": 1,
    "maximum_constant": 0,
    "minimum_assessed_ordinary": 6,
}
SHA = re.compile(r"[a-f0-9]{64}")


def require(test, message):
    if not test:
        raise ValueError(message)


def verify(obj):
    require(
        set(obj)
        == {"schema", "clock", "method", "evidence", "commitments", "rows", "results"}
        and obj["schema"] == SCHEMA,
        "Invalid frozen study schema",
    )
    require(
        obj["clock"]
        == {
            "time_base": CLOCK,
            "ticks_per_picture": TICKS,
            "samples_per_picture": STEP,
            "pictures": 120,
            "sample_count": 76800,
        },
        "The frozen carrier clock changed",
    )
    require(
        obj["method"]
        == {
            "readable_per_half": 6,
            "point_hex": POINT.hex(),
            "ambiguity_margin_hex": GAP.hex(),
            "minimum_shift": 2560,
            "signed_tolerance": 960,
            "cut_frame": FRAME,
            "requirements": REQUIREMENTS,
        },
        "The frozen decision rule changed",
    )
    require(
        set(obj["commitments"])
        == {"design", "media", "measured", "independent_frames", "independent_crops"}
        and all(
            type(v) is str and SHA.fullmatch(v) for v in obj["commitments"].values()
        ),
        "Independent verification commitments are incomplete",
    )
    cases = obj["rows"]
    require(len(cases) == 30, "The held matrix has an unexpected size")
    seen = set()
    groups = set()
    details = defaultdict(Counter)
    totals = Counter()
    for row in cases:
        require(
            set(row)
            == {
                "group",
                "role",
                "authored_shift",
                "cut",
                "status",
                "before_alert",
                "before_assessed",
                "reading",
                "anchors",
            },
            "Case contains an unexpected field",
        )
        group, role = row["group"], row["role"]
        require(
            type(group) is int
            and 1 <= group <= 6
            and role in ROLES
            and (group, role) not in seen,
            "Unknown or repeated case",
        )
        seen.add((group, role))
        wanted = (
            (3200 if group % 2 else 4480) * (-1 if role == "after_early" else 1)
            if role in EDIT
            else None
        )
        require(
            row["authored_shift"] == wanted and row["cut"] == FRAME,
            "An authored label or exact cut differs from its frozen plan",
        )
        before = row["before_alert"]
        require(
            type(before) is type(row["before_assessed"]) is bool,
            "The comparison status is incomplete",
        )
        status = row["status"]
        reading = row["reading"]
        active = status == "localized"
        require(
            status in {"localized", "ambiguous", "no_match"}
            and isinstance(reading, dict),
            "Missing measurable lag hypotheses",
        )
        require(
            set(reading)
            == {
                "strength_hex",
                "delta_samples",
                "lags_samples",
                "paired_support",
                "paired_difference_hex",
                "opposite_strength_hex",
            },
            "Invalid paired lag readings",
        )
        delta = reading["delta_samples"]
        lags = reading["lags_samples"]
        require(
            type(delta) is int
            and len(lags) == 2
            and lags[1] - lags[0] == delta
            and abs(delta) >= 2560
            and all(
                type(v) is int and -6400 <= v <= 6400 and v % 320 == 0 for v in lags
            ),
            "The signed lag differs from its source-relative hypotheses",
        )
        support = reading["paired_support"]
        differences = reading["paired_difference_hex"]
        require(
            len(support) == len(differences) == 4
            and all(type(v) is int and 6 <= v <= 28 for v in support),
            "Insufficient common comparison rows",
        )
        n = float.fromhex(reading["strength_hex"])
        opposition = reading["opposite_strength_hex"]
        opposing = None if opposition is None else float.fromhex(opposition)
        require(
            n.hex() == reading["strength_hex"]
            and -2 <= n <= 2
            and n == min(float.fromhex(x) for x in differences)
            and (opposing is None or -2 <= opposing <= n),
            "Strength differs from the four paired readings",
        )
        decided = (
            "no_match"
            if n <= POINT
            else "ambiguous"
            if opposing is not None and n - opposing < GAP
            else "localized"
        )
        require(
            status == decided, "Recorded disposition differs from the frozen points"
        )
        anchors = row["anchors"]
        require(
            isinstance(anchors, list)
            and (len(anchors) == 3 if active else not anchors),
            "Incorrect source anchor coverage",
        )
        if active:
            require(
                all(type(v) is int for v in anchors)
                and 0 <= anchors[0] < FRAME <= anchors[1] < anchors[2] < 120,
                "Source anchors do not bracket the measured cut",
            )
        case = details[role]
        case["files"] += 1
        case["incumbent_alerts"] += before
        case["incumbent_assessed"] += row["before_assessed"]
        case["candidate_alerts"] += active
        case["ambiguous"] += status == "ambiguous"
        case["no_match"] += status == "no_match"
        if role in EDIT:
            useful = abs(delta - wanted) <= 960
            case["correct_signed_and_exact_cut"] += active and useful
            case["correct_before_point"] += useful
            case["wrong_signed"] += active and not useful
            if active and useful:
                groups.add(group)
            totals["correct_signed_and_exact_cut"] += active and useful
            totals["wrong_signed"] += active and not useful
        elif role in ORDINARY:
            totals["ordinary_measured"] += 1
            totals["ordinary_alerts"] += active
        else:
            totals["constant_alerts"] += active
        totals["original_crop_anchors"] += len(anchors)
    require(
        seen == {(k, r) for k in range(1, 7) for r in ROLES},
        "A required matched case is missing",
    )
    totals["correct_source_groups"] = len(groups)
    require(
        obj["evidence"]
        == {
            "native_picture_frames_and_clocks": 3600,
            "decoded_audio_samples": 2304000,
            "original_crop_anchors": totals["original_crop_anchors"],
        },
        "Exact source evidence counts changed",
    )
    accept = (
        totals["correct_signed_and_exact_cut"]
        >= REQUIREMENTS["minimum_correct_sign_and_exact_cut"]
        and len(groups) >= REQUIREMENTS["minimum_groups"]
        and totals["wrong_signed"] <= REQUIREMENTS["maximum_wrong_positive"]
        and totals["ordinary_alerts"] <= REQUIREMENTS["maximum_ordinary"]
        and totals["constant_alerts"] <= REQUIREMENTS["maximum_constant"]
        and totals["ordinary_measured"] >= REQUIREMENTS["minimum_assessed_ordinary"]
    )
    return {
        "totals": dict(totals),
        "by_role": {key: dict(value) for key, value in details.items()},
        "gate_pass": accept,
    }


if __name__ == "__main__":
    proof = json.loads(Path(sys.argv[1]).read_bytes())
    result = verify(proof)
    require(
        result == proof["results"], "Published result differs from the frozen rules"
    )
    print(json.dumps(result, sort_keys=True, indent=2))
