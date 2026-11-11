"""Replay the full guarded GRID app pilot and emit only public aggregates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from app.detectors.video import lipsync
from scripts import grid_serving_baseline_v156 as serving
from scripts import grid_serving_readout_v159 as readout
from scripts.review_grid_timing_opportunity import audio_search_indices

ROUTES = {
    "aud.sound_events": "background_sound",
    "vid.frames.dfdc_b7": "frame_authenticity",
    readout.APPEARANCE: "face_appearance",
    readout.TIMING: "audio_picture_timing",
}


def exact_route_counts(rows):
    counts = {label: {"statuses": Counter(), "memory_budget_skips": 0} for label in ROUTES.values()}
    for row in rows:
        for detector, label in ROUTES.items():
            matching = [run for run in row["detector_runs"] if run["detector_id"] == detector]
            serving.require(len(matching) <= 1, "Repeated route in a fixed serving case")
            if not matching:
                counts[label]["statuses"]["unobserved"] += 1
                continue
            run = matching[0]
            serving.require(
                run["status"] in readout.DETECTOR_STATES, "Unknown route status in serving record"
            )
            counts[label]["statuses"][run["status"]] += 1
            if run["status"] == "skipped" and run["reason"] == "memory budget":
                counts[label]["memory_budget_skips"] += 1
    return counts


def paired_soundtrack(construction, observed):
    counts = Counter()
    original_directions, delayed_directions = Counter(), Counter()

    def audio_binding(case, row):
        sampled = row["sampled_input"] or {}
        if (
            case["status"] != "ready"
            or row["status"] != "done"
            or not all(
                sampled.get(key) is True
                for key in ("audio_samples_equal", "audio_origin_equal", "audio_pcm_equal")
            )
        ):
            return None
        expected = case["expected"]
        return tuple(expected[key] for key in ("audio_samples", "audio_start_seconds", "pcm_s16_sha256"))

    def soundtrack(row):
        return (row.get("customer_assessment") or {}).get("soundtrack")

    def direction(value):
        known = {"ai", "original", "inconclusive", "unavailable"}
        observed_direction = (value or {}).get("direction")
        return observed_direction if observed_direction in known else "unavailable"

    for ordinal, rows in observed.items():
        cases = {row["condition"]: row for row in construction[ordinal]}
        original = next(row for row in rows if row["condition"] == "original")
        original_audio = audio_binding(cases["original"], original)
        before = soundtrack(original)
        before_direction = direction(before)
        original_directions[before_direction if original_audio else "unavailable"] += 1
        for row in rows:
            condition = row["condition"]
            if condition == "original":
                continue
            after = soundtrack(row)
            after_direction = direction(after)
            bound = audio_binding(cases[condition], row)
            delay = condition == "audio_delay_320ms"
            matched = original_audio and bound and original_audio[:2] == bound[:2]
            if delay:
                verified = bool(matched and original_audio[2] != bound[2])
                counts["verified_ordinary_delay_pairs"] += verified
                delayed_directions[after_direction if verified else "unavailable"] += 1
                counts["ordinary_delay_direction_changed"] += bool(
                    verified and before_direction != after_direction
                )
            else:
                verified = bool(matched and original_audio == bound)
                counts["verified_same_pcm_pairs"] += verified
                counts["same_pcm_full_assessment_equal"] += bool(verified and before == after)
                counts["same_pcm_both_assessments_present"] += bool(
                    verified and before is not None and after is not None
                )
    return {
        "original_directions": original_directions,
        "ordinary_delay_directions": delayed_directions,
        "matched_counts": counts,
        "scope": "Separate soundtrack outcomes on genuine audio and a non-neural 320 ms audio copy control.",
    }


def complete():
    plan = serving.baseline.planning.load_completed()
    ordinals = [row["ordinal"] for row in plan["sources"] if row["role"] == "pilot"]
    serving.require(ordinals == list(range(16)), "Expected the fixed sixteen-source pilot")
    construction, observed = {}, {}
    bindings, profiles = [], []
    for ordinal in ordinals:
        stage = serving.stage_folder(ordinal)
        construction[ordinal] = serving.baseline.clips.load_completed(ordinal)
        observed[ordinal] = serving.load_completed(ordinal)
        profile = serving.read(stage / "profile.json")
        journal = serving.WORK / f"clip-{ordinal:02d}" / "cases.jsonl"
        bindings.append(
            {
                "profile_sha256": serving.sha(stage / "profile.json"),
                "worker_receipt_sha256": profile["worker_receipt_sha256"],
                "case_journal_sha256": serving.sha(journal),
            }
        )
        profiles.append(profile)
    aggregate = readout.aggregate(plan, construction, observed)
    overall = aggregate["overall"]
    serving.require(
        overall["planned"] == overall["constructed"] == overall["attempted"] == 176,
        "Every planned case must have a validated serving record",
    )
    rows = [row for batch in observed.values() for row in batch]
    ceilings = Counter(
        len(audio_search_indices(case["expected"]["audio_samples"]))
        for batch in construction.values()
        for case in batch
        if case["status"] == "ready"
    )
    minimum_windows = min(2 * lipsync.MIN_RUN_WINDOWS, lipsync.COLLAPSE_MIN_TRACK_WINDOWS)
    commitments = json.dumps(bindings, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema": "chipmunk.grid-serving-completion/156",
        "status": "complete_guarded_pilot",
        "serving_freeze_sha256": serving.sha(serving.PUBLIC),
        "ordered_private_artifact_commitments_sha256": hashlib.sha256(commitments).hexdigest(),
        "complete_guarded_batches": len(profiles),
        "max_sampled_descendant_rss_bytes": max(p["sampled_tree_rss_lower_bound_bytes"] for p in profiles),
        "sum_worker_duration_seconds": round(sum(p["duration_s"] for p in profiles), 3),
        "outside_validated_envelope": sum(
            ((row.get("condition_profile") or {}).get("profile") or {}).get("within_validated_envelope")
            is False
            for row in rows
        ),
        "selected_detector_route_counts": exact_route_counts(rows),
        "paired_soundtrack": paired_soundtrack(construction, observed),
        "timing_change_opportunity": {
            "full_search_windows_possible_per_track_from_audio_extent": ceilings,
            "minimum_windows_for_either_change_rule": minimum_windows,
            "cases_meeting_minimum_before_picture_speech_and_attribution_gates": sum(
                count for ceiling, count in ceilings.items() if ceiling >= minimum_windows
            ),
            "scope": (
                "The raw timing readout counts a bound lag measurement as assessed. "
                "This ceiling measures opportunity to detect a timing change, before further coverage gates."
            ),
        },
        "overall": overall,
        "families": aggregate["families"],
    }


if __name__ == "__main__":
    print(json.dumps(complete(), indent=2, sort_keys=True, allow_nan=False))
