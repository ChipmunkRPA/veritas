"""Recount the no-media short face phase experiment with the standard library."""

import argparse
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path

SCHEMA = "chipmunk.video-face-phase-windows/1"
POINTS = (0.002, 0.004, 0.006, 0.009, 0.012, 0.018, 0.024, 0.035, 0.05, 0.075, 0.12)
ARMS = ("collar", "face")
DELIVERIES = {"direct", "tandem"}
EDIT_ROLES = {"swap_a", "swap_b"}
ORDINARY_ROLES = {"original", "gain", "picture_jump", "different_shot", "face_blur"}
GATE = {
    "added_useful": 6,
    "added_groups": 3,
    "ordinary_files": 2,
    "ordinary_groups": 1,
    "additional_wrong": 2,
    "same_world_cuts": 0,
}
BUDGETS = (0, 1, 2, 4, 8, 20)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def span_valid(span):
    return span is None or (
        isinstance(span, list)
        and len(span) == 2
        and all(type(frame) is int for frame in span)
        and 0 <= span[0] < span[1] <= 100
    )


def validate_rows(rows, phase, groups_count, expected, arms):
    keys = {
        "group",
        "delivery",
        "role",
        "truth",
        "brief_assessed",
        "brief",
        "phase_readings",
        "native_face_frames",
        "native_picture_frames",
        "full_app",
    }
    seen = set()
    groups = set()
    roles = Counter()
    for row in rows:
        require(set(row) == keys, f"Unexpected {phase} row fields")
        group, delivery, role = row["group"], row["delivery"], row["role"]
        require(type(group) is int and 1 <= group <= groups_count, f"Invalid {phase} group")
        require(delivery in DELIVERIES and role in EDIT_ROLES | ORDINARY_ROLES, f"Invalid {phase} role")
        require((group, delivery, role) not in seen, f"Duplicate {phase} case")
        seen.add((group, delivery, role))
        groups.add(group)
        roles[role] += 1
        for field in ("truth", "brief"):
            require(span_valid(row[field]), f"Invalid {phase} {field} span")
        require(bool(row["truth"]) == (role in EDIT_ROLES), f"Invalid {phase} edit label")
        require(
            type(row["brief_assessed"]) is bool and (row["brief_assessed"] or row["brief"] is None),
            f"Invalid {phase} brief coverage",
        )
        require(set(row["phase_readings"]) == set(arms), f"Invalid {phase} phase arms")
        for reading in row["phase_readings"].values():
            if reading is not None:
                require(set(reading) == {"score", "span"}, "Unexpected reading fields")
                require(
                    type(reading["score"]) in (int, float) and math.isfinite(reading["score"]),
                    "Invalid phase score",
                )
                require(reading["span"] is not None and span_valid(reading["span"]), "Invalid phase span")
        require(
            type(row["native_face_frames"]) is int and 0 <= row["native_face_frames"] <= 50,
            "Invalid native face-frame count",
        )
        require(row["native_picture_frames"] == 100, "Invalid native picture count")
        full_app = row["full_app"]
        if full_app is not None:
            require(
                phase == "held"
                and isinstance(full_app, list)
                and all(span is not None and span_valid(span) for span in full_app),
                "Invalid worker ranges",
            )
    require(len(groups) == groups_count and len(rows) == expected, f"Wrong {phase} population")
    max_role = groups_count * len(DELIVERIES)
    require(
        all(roles[role] == max_role for role in EDIT_ROLES | (ORDINARY_ROLES - {"face_blur"})),
        f"Wrong {phase} role counts",
    )
    require(roles["face_blur"] == max_role - (2 if phase == "held" else 0), f"Wrong {phase} blur count")
    for group in groups:
        for role in EDIT_ROLES | ORDINARY_ROLES:
            require(
                len({delivery for delivery in DELIVERIES if (group, delivery, role) in seen}) in (0, 2),
                f"Unpaired {phase} encoding",
            )


def matches(truth, guess):
    if truth is None or guess is None:
        return False
    overlap = max(0, min(truth[1], guess[1]) - max(truth[0], guess[0]))
    union = truth[1] - truth[0] + guess[1] - guess[0] - overlap
    return 2 * overlap >= union


def picked(row, arm, point):
    reading = row["phase_readings"][arm]
    return reading["span"] if reading is not None and reading["score"] > point else None


def counts(rows, arm, point):
    totals = Counter(delivered=len(rows))
    useful_groups = set()
    ordinary_groups = set()
    for row in rows:
        truth, brief = row["truth"], row["brief"]
        guess = picked(row, arm, point)
        assessed = row["phase_readings"][arm] is not None
        totals["native_picture_frames"] += row["native_picture_frames"]
        totals["native_face_frames"] += row["native_face_frames"]
        if truth is not None:
            totals["edit"] += 1
            totals["brief_assessed_edit"] += row["brief_assessed"]
            totals["phase_assessed_edit"] += assessed
            old, new = matches(truth, brief), matches(truth, guess)
            totals["brief_useful"] += old
            totals["phase_useful"] += new
            totals["combined_useful"] += old or new
            totals["exact_phase"] += guess == truth
            totals["phase_wrong"] += guess is not None and not new
            totals["added_wrong"] += guess is not None and not new and not matches(brief, guess)
            totals["new_useful"] += new and not old
            if new and not old:
                useful_groups.add(row["group"])
                totals["new_within_one_frame_of_both_edges"] += all(
                    abs(a - b) <= 1 for a, b in zip(truth, guess, strict=True)
                )
        else:
            totals["ordinary"] += 1
            totals["brief_assessed_ordinary"] += row["brief_assessed"]
            totals["phase_assessed_ordinary"] += assessed
            totals["brief_ordinary_flags"] += brief is not None
            totals["phase_ordinary_flags"] += guess is not None
            totals["combined_ordinary_flags"] += brief is not None or guess is not None
            added = guess is not None and brief is None
            totals["added_ordinary"] += added
            if added:
                ordinary_groups.add(row["group"])
            totals["added_cuts"] += added and row["role"] == "different_shot"
    totals["new_groups"] = len(useful_groups)
    totals["added_ordinary_groups"] = len(ordinary_groups)
    return dict(totals)


def selection(rows):
    options = []
    for arm, point in product(ARMS, POINTS):
        result = counts(rows, arm, point)
        admissible = (
            result["added_ordinary"] <= 2
            and result["added_ordinary_groups"] <= 1
            and result["added_cuts"] == 0
        )
        utility = 4 * result["new_useful"] - 3 * result["added_ordinary"] - 2 * result["added_wrong"]
        rank = (admissible, utility, result["new_useful"], -result["added_wrong"], -point, arm == "collar")
        options.append((rank, arm, point, result))
    _, arm, point, result = max(options)
    return {"arm": arm, "point": point, "counts": result}


def full_app_comparison(rows, arm, point):
    totals = Counter()
    for row in rows:
        guess = picked(row, arm, point)
        truth, previous = row["truth"], row["full_app"]
        if previous is None:
            totals["phase_flag_without_retained_app_result"] += guess is not None
        elif truth is not None:
            already = any(matches(truth, span) for span in previous)
            new = matches(truth, guess)
            totals.update(
                app_compared_edited=1,
                app_useful_once=already,
                phase_correct_on_app_compared=new,
                new_useful_against_full_app=new and not already,
            )
        else:
            totals.update(
                app_compared_ordinary=1,
                app_any_ordinary=bool(previous),
                new_phase_flags_on_app_compared_ordinary=guess is not None and not previous,
            )
    return dict(totals)


def frontier(rows, arm):
    """Maximize new correct ranges within each held ordinary-file budget."""
    scores = sorted({row["phase_readings"][arm]["score"] for row in rows if row["phase_readings"][arm]})
    candidates = [(point, counts(rows, arm, point)) for point in [scores[0] - 1, *scores]]
    fields = (
        "new_useful",
        "new_groups",
        "added_ordinary",
        "added_ordinary_groups",
        "phase_wrong",
        "added_cuts",
    )
    result = {}
    for budget in BUDGETS:
        eligible = [(point, values) for point, values in candidates if values["added_ordinary"] <= budget]
        point, values = max(
            eligible,
            key=lambda run: (
                run[1]["new_useful"],
                -run[1]["added_ordinary"],
                -run[1]["phase_wrong"],
                run[0],
            ),
        )
        result[str(budget)] = {field: values[field] for field in fields}
        result[str(budget)]["point"] = point
    return result


def gate_decision(result):
    return {
        "new": result["new_useful"] >= GATE["added_useful"],
        "groups": result["new_groups"] >= GATE["added_groups"],
        "ordinary_cases": result["added_ordinary"] <= GATE["ordinary_files"],
        "ordinary_groups": result["added_ordinary_groups"] <= GATE["ordinary_groups"],
        "misplaced": result["added_wrong"] <= GATE["additional_wrong"],
        "cuts": result["added_cuts"] == GATE["same_world_cuts"],
    }


def recount(receipt):
    require(receipt["schema"] == SCHEMA and receipt["gate"] == GATE, "Wrong schema or frozen gate")
    require(receipt["development_points"] == list(POINTS), "Wrong formation points")
    development, held = receipt["development"], receipt["held"]
    validate_rows(development, "development", 4, 56, ARMS)
    validate_rows(held, "held", 8, 110, ("collar",))
    chosen = selection(development)
    arm, point = chosen["arm"], chosen["point"]
    require(arm == "collar" and point == 0.075, "The recomputed choice differs from the frozen point")
    result = counts(held, arm, point)
    ordinary_flags = Counter(
        f"{row['delivery']}/{row['role']}"
        for row in held
        if row["truth"] is None and picked(row, arm, point) and row["brief"] is None
    )
    gate = gate_decision(result)
    return {
        "selection": chosen,
        "held": result,
        "delivery": {
            delivery: counts([row for row in held if row["delivery"] == delivery], arm, point)
            for delivery in sorted(DELIVERIES)
        },
        "added_ordinary_roles": dict(ordinary_flags),
        "gate": gate,
        "gate_passed": all(gate.values()),
        "previous_selected_worker": full_app_comparison(held, arm, point),
        "post_hoc_ordinary_budget_frontier": frontier(held, arm),
        "post_hoc_permissive_point": counts(held, arm, 0.002),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
        result = recount(receipt)
        require(result == receipt["published_recount"], "Receipt summary does not match rows")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
