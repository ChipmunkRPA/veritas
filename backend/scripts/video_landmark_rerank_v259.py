"""Replay one exploratory landmark-quality rule on saved appearance windows."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from statistics import mean
from types import SimpleNamespace

from app.evidence.face_interval_binding import APPEARANCE_REVIEW_POINT
from scripts import video_landmark_transfer_v255 as transfer
from scripts import video_saved_landmark_support_v254 as saved
from scripts.video_appearance_corrected_v112 import scoped

ROOT, PANEL = saved.webcam.ROOT, saved.webcam.PANEL
NAME = "video-landmark-rerank-v259"
PUBLIC = PANEL / f"{NAME}-freeze.json"
SAVED = PANEL / "video-saved-landmark-support-v254-complete.json"
TRANSFER = PANEL / "video-landmark-transfer-v255-complete.json"
RECEIPTS = {
    SAVED: "7402d598c251158a6d4ec4b42b5e6b059f739e62acf8c46d015011aceab66021",
    TRANSFER: "6b85af0a22f2a78073064d5458281e9f912d9ed26f0bff9e88c961b78f983c32",
    transfer.PUBLIC: "0bc2ad8114de2345e491c0c9eceea2a4f6f781bad63b13761f38a11a91c8da1e",
}
CODE = (
    "backend/scripts/video_landmark_rerank_v259.py",
    "backend/tests/test_video_landmark_rerank_v259.py",
    "backend/app/evidence/face_interval_binding.py",
    f"docs/{NAME}-method.md",
)
MINIMUM_FLANK_FRACTION = 0.60
read, sha, require = saved.read, saved.sha, saved.require


def protocol():
    require(all(sha(path) == digest for path, digest in RECEIPTS.items()), "Exact saved audits required")
    require(
        all(
            sha(ROOT / "backend/scripts" / path) == digest
            for path, digest in read(SAVED)["code_sha256"].items()
        )
        and read(transfer.PUBLIC) == transfer.protocol(),
        "Exact saved audit code required",
    )
    return {
        "schema": "chipmunk.video-landmark-rerank-freeze/259",
        "source_receipts": {str(path.relative_to(ROOT)): digest for path, digest in RECEIPTS.items()},
        "code_sha256": {path: sha(ROOT / path) for path in CODE},
        "minimum_during_over_mean_flank_eye_box_ratio": MINIMUM_FLANK_FRACTION,
        "cutoff_basis": "exploratory_after_observing_v254_webcam_results",
        "appearance_review_point": APPEARANCE_REVIEW_POINT,
        "retained_windows_per_upload_cap": 12,
        "new_neural_calls_permitted": False,
        "serving_change_permitted": False,
    }


def quality(before, during, after):
    if before is None or during is None or after is None:
        return "geometry_unavailable"
    return "supported" if during >= MINIMUM_FLANK_FRACTION * mean((before, after)) else "below_flank_fraction"


def ranked_window(record):
    measurement = saved.support.measure(record)
    eye = measurement.get("eye_box_width", {})
    during = record["centers"]["during"]
    return {
        "measurement": measurement,
        "quality": quality(*(eye.get(role, {}).get("median") for role in saved.support.ROLES)),
        "response": float.fromhex(record["measurement_hex"]["response"]),
        "start_pts": record["start_pts"],
        "track_id": record["track_id"],
        "frames": (during[0], during[-1]),
    }


def rerank(retained, point=APPEARANCE_REVIEW_POINT):
    rows = [row for _, row in retained]
    keys = [(-row["response"], row["start_pts"], row["track_id"]) for row in rows]
    require(0 < len(rows) <= 12 and keys == sorted(keys), "Exact saved appearance rank required")
    above = [(rank, row) for rank, row in enumerate(rows, 1) if row["response"] > point]
    require(
        [rank for rank, (flag, _) in enumerate(retained, 1) if flag] == ([1] if above else []),
        "Existing flag differs from saved rank",
    )
    choice = next(((rank, row) for rank, row in above if row["quality"] == "supported"), None)
    outcomes = Counter(row["quality"] for _, row in above)
    if choice is not None:
        state = "same_rank" if choice[0] == 1 else "lower_rank"
    elif outcomes["geometry_unavailable"]:
        state = "undetermined_geometry"
    else:
        state = "no_supported_flag" if above else "no_baseline_flag"
    return {
        "state": state,
        "rank": choice[0] if choice else None,
        "before_frames": rows[0]["frames"] if above else None,
        "after_frames": choice[1]["frames"] if choice else None,
        "geometry_unavailable_ahead_of_choice": sum(
            row["quality"] == "geometry_unavailable" for rank, row in above if choice and rank < choice[0]
        ),
        "all_qualities": Counter(row["quality"] for row in rows),
        "above_point_qualities": outcomes,
    }


def overlap(interval, target):
    return len(target.intersection(range(interval[0], interval[1] + 1))) if interval else 0


def summarize(replays, target=None):
    count = Counter(replay["state"] for replay in replays)
    total = {
        "uploads": len(replays),
        "baseline_flags": sum(row["before_frames"] is not None for row in replays),
        "reranked_flags": count["same_rank"] + count["lower_rank"],
        "outcomes": dict(count),
        "reranked_retained_rank_counts": dict(Counter(str(row["rank"]) for row in replays if row["rank"])),
        "reranked_replacement_rank_drops": dict(
            Counter(str(row["rank"] - 1) for row in replays if row["state"] == "lower_rank")
        ),
        "reranked_flags_with_unchanged_range": sum(
            row["after_frames"] is not None and row["before_frames"] == row["after_frames"] for row in replays
        ),
        "reranked_flags_with_unavailable_geometry_at_higher_rank": sum(
            bool(row["geometry_unavailable_ahead_of_choice"]) for row in replays
        ),
        **{
            key: dict(sum((row[key] for row in replays), Counter()))
            for key in ("all_qualities", "above_point_qualities")
        },
    }
    if target is not None:
        prior, current = (
            [overlap(row[key], target) for row in replays] for key in ("before_frames", "after_frames")
        )
        total["known_edit_overlap"] = {
            "baseline_flags_overlapping_edit": sum(n > 0 for n in prior),
            "reranked_flags_overlapping_edit": sum(n > 0 for n in current),
            "baseline_edited_frames_covered_across_renders": sum(prior),
            "reranked_edited_frames_covered_across_renders": sum(current),
            "renders_with_same_edit_overlap": sum(a == b for a, b in zip(prior, current, strict=True)),
        }
    return total


def archived_replay():
    windows = scoped(saved.windows, {"support": SimpleNamespace(measure=ranked_window)})
    cases = [
        case
        for reader in (saved.swap_cases, saved.webcam_cases)
        for case in scoped(reader, {"windows": windows})()
    ]
    require(
        all(
            row["actual_changed_frames"] == list(saved.swap.EDIT_FRAMES)
            for row in read(saved.swap.MANIFEST)["cases"]
            if row["condition"].startswith("swap_")
        ),
        "Exact edited frames required",
    )
    result = {}
    for cohort, previous in read(SAVED)["cohorts"].items():
        retained = [rows for name, rows in cases if name == cohort]
        require(
            len(retained) == previous["renders"]
            and all(
                saved.aggregate(
                    [row["measurement"] for rows in retained for selected, row in rows if selected == flag]
                )
                == previous[label]
                for label, flag in (("selected", True), ("other_retained", False))
            ),
            "Retained geometry must reproduce v254 coverage and measurements",
        )
        target = set(saved.swap.EDIT_FRAMES) if cohort == "known_swap_renders" else None
        result[cohort] = {
            "source_groups": previous["source_groups"],
            **summarize([rerank(rows) for rows in retained], target),
        }
    return result


def transfer_replay():
    cases, point = transfer.load_bound_packets()
    windows = transfer.observations(cases, point)
    flags = {key: row for key, row in windows.items() if row["current_flag"]}
    require(
        len(windows) == 690
        and len(flags) == 46
        and {key[1] for key in flags} == set(transfer.SWAPS)
        and all(
            transfer.summarize({key: row for key, row in flags.items() if key[1] == condition})
            == read(TRANSFER)["by_condition"][condition]["existing_flags"]
            for condition in transfer.CONDITIONS
        ),
        "Existing transfer flags and measurements must reproduce v255",
    )
    counts = Counter(
        quality(*((row["metrics"] or {}).get(f"{role}_median_eye_box_width") for role in transfer.PHASES))
        for row in flags.values()
    )
    return {
        "source_groups_with_existing_swap_flags": len({key[0] for key in flags}),
        "assessed_existing_swap_flags": len(flags),
        "existing_control_flags": 0,
        "existing_swap_flag_quality": dict(counts),
        "historical_appearance_review_point": point,
    }


def readout():
    require(read(PUBLIC) == protocol(), "Exact exploratory rule freeze required")
    return {
        "schema": "chipmunk.video-landmark-rerank-result/259",
        "status": "complete",
        "freeze_sha256": sha(PUBLIC),
        "archived_full_app_rerank": archived_replay(),
        "constructed_swap_transfer": transfer_replay(),
        "independent_known_swap_interventions_in_full_app": 1,
        "decoded_frames": 0,
        "model_calls": 0,
        "serving_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "readout"))
    if parser.parse_args().command == "freeze":
        frozen = protocol()
        with PUBLIC.open("x") as stream:
            stream.write(json.dumps(frozen, indent=2, sort_keys=True, allow_nan=False) + "\n")
        value = {"status": "frozen", "freeze_sha256": sha(PUBLIC), "model_calls": 0}
    else:
        saved.edge.enforce_read_only()
        value = readout()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
