"""Recount a saved face-transition confirmation and its app replay without models or media."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

BASIS = ("incumbent", "fine_tiles", "off_face_motion")
PRECEDENCE = ("incumbent", "off_face_motion", "fine_tiles")
COMBINED = "off_face_motion_then_fine"
PROFILES = {"crop135", "crop160"}
CASE_FIELDS = (
    "role",
    "reference_frame",
    "readings_by_basis_each_frame_and_score_hex",
    "serving_frame",
    "serving_basis",
    "before_transition_after_frames",
    "serving_seconds",
)
CASE_ROLES = {
    "authentic": 1,
    "exposure_step": 1,
    "focus_step": 1,
    "same_source_jump": 1,
    "similar_background_cut": 1,
    "prefix": 2,
    "suffix": 2,
}


def valid_frame(frame) -> bool:
    return type(frame) is int and 0 <= frame < 100


def selected_frame(reading, point: float) -> int | None:
    if reading is None:
        return None
    frame, score_hex = reading
    score = float.fromhex(score_hex)
    if not valid_frame(frame) or not math.isfinite(score):
        raise ValueError("Invalid saved reading")
    return frame if score > point else None


def recount(rows: list, arm: str) -> dict:
    counts = Counter()
    localized, alerted = set(), set()
    for group, role, truth, frames in rows:
        frame = frames[arm]
        if truth is None:
            if frame is not None:
                alerted.add(group)
                counts["ordinary_alerts"] += 1
                counts["ordinary_cut_alerts"] += role == "similar_background_cut"
        elif frame is None:
            counts["missed_edits"] += 1
        elif abs(frame - truth) <= 1:
            localized.add(group)
            counts["localized_within_one_frame"] += 1
            counts["localized_exact_frame"] += frame == truth
        else:
            counts["mislocalized_edits"] += 1
    fields = (
        "localized_within_one_frame",
        "localized_exact_frame",
        "mislocalized_edits",
        "missed_edits",
        "ordinary_alerts",
        "ordinary_cut_alerts",
    )
    return {key: counts[key] for key in fields} | {
        "localized_groups": len(localized),
        "ordinary_alert_groups": len(alerted),
    }


def replay(path: Path) -> dict:
    receipt = json.loads(path.read_bytes())
    if (
        receipt["schema"] != "chipmunk.video-face-sparse-transition-confirmation/1"
        or tuple(receipt["row_fields"]) != CASE_FIELDS
        or tuple(receipt["basis_order"]) != BASIS
        or tuple(receipt["combined_selection"]) != PRECEDENCE
        or receipt["tolerance_native_frames"] != 1
    ):
        raise ValueError("Unexpected confirmation procedure")
    point = float.fromhex(receipt["locked_operating_point_hex"])
    if not math.isfinite(point):
        raise ValueError("Invalid operating point")

    by_profile = defaultdict(list)
    durations, serving_bases = [], Counter()
    witnesses = 0
    for group_number, group in enumerate(receipt["recording_groups"], 1):
        profile = group["profile"]
        if profile not in PROFILES or Counter(row[0] for row in group["cases"]) != CASE_ROLES:
            raise ValueError(f"Group {group_number} has unexpected framing or cases")
        for role, truth, readings, frame, basis, anchors, seconds in group["cases"]:
            reference_matches_role = valid_frame(truth) if role in {"prefix", "suffix"} else truth is None
            if not reference_matches_role:
                raise ValueError(f"Unexpected reference for {role} in group {group_number}")
            estimates = {
                arm: selected_frame(value, point) for arm, value in zip(BASIS, readings, strict=True)
            }
            chosen = next((arm for arm in PRECEDENCE if estimates[arm] is not None), None)
            if (
                basis != chosen
                or frame != estimates.get(chosen)
                or (frame is not None and not valid_frame(frame))
            ):
                raise ValueError(f"App replay differs for {role} in group {group_number}")
            if frame is not None:
                if (
                    anchors is None
                    or len(anchors) != 3
                    or not all(map(valid_frame, anchors))
                    or anchors[:2] != [frame - 1, frame]
                    or anchors[2] <= frame
                ):
                    raise ValueError(f"Invalid evidence frames for {role} in group {group_number}")
                witnesses += len(anchors)
                serving_bases[basis] += 1
            elif anchors is not None:
                raise ValueError(f"Unexpected evidence frames for {role} in group {group_number}")
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError(f"Invalid measurement duration for {role} in group {group_number}")
            durations.append(seconds)
            estimates[COMBINED] = frame
            by_profile[profile].append((group_number, role, truth, estimates))
    all_rows = [row for rows in by_profile.values() for row in rows]
    if set(by_profile) != PROFILES or len(all_rows) != receipt["native_delivery_clocks_verified"]:
        raise ValueError("Saved row count does not match the native clock verification")
    results = {
        profile: {arm: recount(rows, arm) for arm in (*BASIS, COMBINED)}
        for profile, rows in {"all": all_rows, **by_profile}.items()
    }
    if results != receipt["results"]:
        raise ValueError("Reported counts do not match the saved readings")

    published = receipt["serving_replay"]
    combined = results["all"][COMBINED]
    parity = {
        "rows": len(all_rows),
        "same_predictions": len(all_rows),
        "exact_localizations": combined["localized_exact_frame"],
        "ordinary_alerts": combined["ordinary_alerts"],
        "wrong_edit_locations": sum(
            truth is not None and estimates[COMBINED] not in (None, truth)
            for _, _, truth, estimates in all_rows
        ),
        "localized_arms": dict(serving_bases),
        "native_source_witnesses_checked": witnesses,
        "within_12_seconds": sum(seconds < 12.0 for seconds in durations),
        "measurement_seconds": {
            "min": min(durations),
            "median": round(median(durations), 6),
            "max": max(durations),
        },
    }
    if published["budget_seconds"] != 12.0 or any(published[key] != value for key, value in parity.items()):
        raise ValueError("Reported app parity or timing does not match the saved rows")
    return {"results": results, "serving_replay": parity}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        result = replay(args.receipt)
    except (IndexError, KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
