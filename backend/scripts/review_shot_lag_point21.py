"""Recount a held two-shot shift transfer without models or source media."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.video-shot-relative-lag-point21-held/1"
SCOPE = {
    "fps": "30000/1001",
    "time_base": "1/90000",
    "frame_ticks": 3003,
    "frames": 120,
    "samples": 64064,
    "cut_samples": 32032,
}
ROLES = {
    "after_early",
    "after_late",
    "constant_delay",
    "ordinary_pause",
    "ordinary_cut",
}
EDITED = {"after_early", "after_late"}
LIMITS = {
    "correct": 4,
    "groups": 3,
    "wrong": 1,
    "ordinary": 1,
    "constant": 0,
    "ordinary_measured": 6,
}


def require(test, message):
    if not test:
        raise ValueError(message)


def verify(doc):
    require(
        set(doc)
        == {
            "schema",
            "scope",
            "points",
            "rules",
            "media_replay",
            "commitments",
            "observations",
            "summary",
        },
        "Unexpected record fields",
    )
    require(
        doc["schema"] == SCHEMA and doc["scope"] == SCOPE and doc["rules"] == LIMITS,
        "Frozen scope changed",
    )
    require(doc["points"] == [p.hex() for p in (0.10, 0.21)], "Frozen points changed")
    commitments = doc["commitments"]
    require(
        set(commitments)
        == {
            "freeze.json",
            "media.json",
            "measurements.jsonl",
            "independent-media.json",
            "independent-anchors.json",
            "independent-peaks-freeze.json",
            "independent-peaks-measurements.jsonl",
        }
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in commitments.values()
        ),
        "Malformed aggregate commitments",
    )
    summary = {arm: Counter() for arm in ("point10", "point21")}
    groups = {arm: set() for arm in summary}
    conditions = defaultdict(Counter)
    stable = Counter()
    keys = set()
    cut_groups = set()
    total_anchors = strict_anchors = 0
    for r in doc["observations"]:
        require(
            set(r)
            == {
                "group",
                "role",
                "wanted_lag",
                "cut",
                "state",
                "stricter_state",
                "reading",
                "original_anchors",
                "incumbent_alert",
                "independent_state",
            },
            "Unexpected scalar fields",
        )
        group, role = r["group"], r["role"]
        require(
            type(group) is int
            and 1 <= group <= 6
            and role in ROLES
            and (group, role) not in keys,
            "Duplicate or unknown case",
        )
        keys.add((group, role))
        truth = (
            (3200 if group % 2 else 4480) * (-1 if role == "after_early" else 1)
            if role in EDITED
            else None
        )
        require(
            r["wanted_lag"] == truth and r["cut"] in (None, 60),
            "A source label differs",
        )
        states = (r["state"], r["stricter_state"])
        rec = r["reading"]
        if r["cut"] is None:
            require(
                states == ("shot_coverage", "shot_coverage")
                and r["independent_state"] == "shot_coverage"
                and rec is None,
                "Unobserved cut was assessed",
            )
            estimated = None
        else:
            cut_groups.add(group)
            require(
                isinstance(rec, dict) and r["independent_state"] == "unstable_peak",
                "Invalid comparison coverage",
            )
            require(
                set(rec)
                == {
                    "strength_hex",
                    "delta_samples",
                    "lags_samples",
                    "paired_support",
                    "paired_difference_hex",
                    "opposite_strength_hex",
                },
                "Invalid hypothesis fields",
            )
            level = float.fromhex(rec["strength_hex"])
            opp = rec["opposite_strength_hex"]
            opposition = float.fromhex(opp) if opp is not None else None
            estimated = rec["lags_samples"][1] - rec["lags_samples"][0]
            require(
                estimated == rec["delta_samples"]
                and abs(estimated) >= 2560
                and len(rec["paired_support"]) == len(rec["paired_difference_hex"]) == 4
                and all(type(n) is int and 6 <= n <= 28 for n in rec["paired_support"])
                and -2 <= level <= 2
                and level
                == min(float.fromhex(v) for v in rec["paired_difference_hex"]),
                "Altered paired native evidence",
            )
            state = (
                "no_match"
                if level <= 0.10
                else "ambiguous"
                if opposition is not None and level - opposition < 0.025
                else "localized"
            )
            require(
                states
                == (
                    state,
                    "no_match" if state == "localized" and level <= 0.21 else state,
                ),
                "Stored decision differs from the two points",
            )
        anchors = r["original_anchors"]
        if states[0] == "localized":
            require(
                anchors == [59, 60, 90] and all(type(i) is int for i in anchors),
                "Original mouth anchors do not bracket the cut",
            )
            total_anchors += 3
            strict_anchors += 3 * (states[1] == "localized")
        else:
            require(anchors == [], "Unlocalized row has an anchor")
        role_summary = conditions[role]
        role_summary["planned"] += 1
        role_summary["measurable"] += r["cut"] == 60
        role_summary["incumbent_alerts"] += r["incumbent_alert"]
        stable[r["independent_state"]] += 1
        for arm, state in zip(summary, states, strict=True):
            x = summary[arm]
            active = state == "localized"
            if truth is not None:
                useful = bool(active and abs(estimated - truth) <= 960)
                x["correct"] += useful
                x["wrong"] += active and not useful
                role_summary[arm + "_correct"] += useful
                role_summary[arm + "_wrong"] += active and not useful
                if useful:
                    groups[arm].add(group)
            elif role == "constant_delay":
                x["constant"] += active
                role_summary[arm + "_alerts"] += active
            else:
                x["ordinary"] += active
                x["ordinary_measured"] += r["cut"] == 60
                role_summary[arm + "_alerts"] += active
    require(
        keys == {(i, r) for i in range(1, 7) for r in ROLES}, "Incomplete frozen matrix"
    )
    for arm, values in summary.items():
        values["groups"] = len(groups[arm])
        values["gate_pass"] = (
            values["correct"] >= LIMITS["correct"]
            and values["groups"] >= LIMITS["groups"]
            and values["wrong"] <= LIMITS["wrong"]
            and values["ordinary"] <= LIMITS["ordinary"]
            and values["constant"] <= LIMITS["constant"]
            and values["ordinary_measured"] >= LIMITS["ordinary_measured"]
        )
    require(
        doc["media_replay"]
        == {
            "picture_pts_rgb": 3600,
            "audio_samples": 1921920,
            "source_picture_ordinals": 720,
            "source_origins": 12,
            "original_crop_anchors": total_anchors,
            "strict_original_crop_anchors": strict_anchors,
        },
        "Native media and position counts differ",
    )
    return {
        "by_condition": {k: dict(v) for k, v in conditions.items()},
        "arms": {k: dict(v) for k, v in summary.items()},
        "independent_peak_status": dict(stable),
        "cut_groups": len(cut_groups),
    }


if __name__ == "__main__":
    d = json.loads(Path(sys.argv[1]).read_bytes())
    result = verify(d)
    require(
        result == d["summary"], "Independent recount does not agree with publication"
    )
    print(json.dumps(result, sort_keys=True, indent=2))
