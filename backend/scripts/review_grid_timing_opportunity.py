"""Replay frozen GRID timing change-rule support from saved metadata only."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from fractions import Fraction

import numpy as np

from app.detectors.video import lipsync, xcorr
from scripts import grid_native_mouth_scores_v155 as continuation

original = continuation.original
plan_context = original.planning
ROOT, BASE, PANEL, ARCHIVE = continuation.ROOT, continuation.BASE, continuation.PANEL, continuation.ARCHIVE
read, require, sha = continuation.read, continuation.require, continuation.sha
lines, stage = plan_context.reviewed.lines, plan_context.reviewed.stage
COMPLETE = PANEL / "grid-native-mouth-scores-complete-v155.json"
XCORR_FREEZE = PANEL / "native-mouth-calibration-freeze-v126.json"
XCORR_CODE = (
    "backend/app/detectors/video/adapters/lipsync_xcorr.py",
    "backend/app/detectors/video/av_evidence.py",
    "backend/app/detectors/video/lipsync.py",
    "backend/app/detectors/video/xcorr.py",
    "backend/scripts/native_mouth_readout_v126.py",
)
CONSTRUCTOR = "grid-native-clips-v149"
DELAY = "audio_delay_320ms"
UNATTRIBUTED = (
    "no face track whose mouth movement follows the speech "
    "(occluded, turned or closed mouth, voice-over or an off-screen speaker)"
)


def saved_inputs():
    """Verify the published execution collection before opening its bound JSON rows."""
    public = read(COMPLETE)
    previous, extended, calibration = map(read, (original.PUBLIC, continuation.PUBLIC, XCORR_FREEZE))
    prev_sha, ext_sha, cal_sha = map(sha, (original.PUBLIC, continuation.PUBLIC, XCORR_FREEZE))
    require(
        public["schema"] == "chipmunk.grid-native-mouth-scores-complete/155"
        and public["status"] == public["comparison"]["status"] == "complete"
        and public["complete_guarded_stages"] == 16
        and public["comparison"]["planned_cases"] == 176
        and public["comparison"]["v151_freeze_sha256"] == extended["v151_freeze_sha256"] == prev_sha
        and public["comparison"]["v155_freeze_sha256"] == ext_sha
        and previous == read(original.TRIAL / "freeze.json")
        and extended == read(continuation.TRIAL / "freeze.json")
        and extended["tail_clip_ordinals"] == list(continuation.TAIL)
        and extended["retained_v151_clip1_monitor_failure"] == continuation.failure_history(),
        "Exact completed v155 continuation required",
    )
    require(
        calibration["schema"] == "chipmunk.native-mouth-calibration-freeze/126"
        and read(PANEL / "native-mouth-calibration-result-v126.json")["freeze_sha256"] == cal_sha
        and all(calibration["code_sha256"][path] == sha(ROOT / path) for path in XCORR_CODE),
        "Frozen v126 timing code required",
    )
    plan_folder = plan_context.TRIAL / "bind-00"
    plan_file = plan_folder / "output/private-plan.json"
    plan_hash = sha(plan_file)
    plan_freeze = sha(plan_context.PUBLIC)
    plan_result = stage(plan_folder, plan_freeze, "grid_transfer_plan_v143.py", 0, {})
    require(
        previous["transfer_plan_freeze_sha256"] == plan_freeze
        and previous["private_transfer_plan_sha256"] == plan_result["private_plan_sha256"] == plan_hash,
        "Exact pilot plan required",
    )
    pilots = original.pilots(read(plan_file))
    constructor_sha = sha(PANEL / "grid-native-clips-freeze-v149.json")
    require(previous["construction_freeze_sha256"] == constructor_sha, "Frozen v149 constructor required")
    collection, bundles = [], []
    for source, ordinal in zip(pilots, (0, *continuation.TAIL), strict=True):
        require(source["ordinal"] == ordinal, "Only the original sixteen pilot ordinals are permitted")
        trial, cache, freeze_sha, caller = (
            (original.TRIAL, original.CACHE, prev_sha, continuation.ORIGINAL_CALLER)
            if ordinal == 0
            else (continuation.TRIAL, continuation.CACHE, ext_sha, continuation.CALLER)
        )
        score_stage = continuation.stage_path(trial, ordinal)
        record = stage(score_stage, freeze_sha, caller, 0, {})
        commitment = continuation.commitments(trial, cache, ordinal)
        require(
            record["schema"] == "chipmunk.grid-native-mouth-score-result/151"
            and record["clip_ordinal"] == ordinal
            and record["planned_cases"] == 11
            and record["new_head_or_calibration_fits"] == 0
            and all(
                record[key] == digest for key, digest in commitment.items() if key.startswith("private_")
            ),
            "Complete score-stage metadata and private journal commitments required",
        )
        clip = f"clip-{ordinal:02d}"
        folder = cache / clip
        ctor_folder = BASE / CONSTRUCTOR / clip / "bind-00"
        ctor = stage(ctor_folder, constructor_sha, "grid_native_clips_v149.py", 0, {})
        ctor_journal = ARCHIVE / CONSTRUCTOR / clip / "cases.jsonl"
        input_binding = read(folder / "input-binding.json")
        require(
            ctor["schema"] == "chipmunk.grid-native-clips-result/149"
            and ctor["clip_ordinal"] == ordinal
            and ctor["planned_cases"] == 11
            and input_binding["constructor_result_sha256"] == sha(ctor_folder / "output/result.json")
            and input_binding["constructor_profile_sha256"] == sha(ctor_folder / "profile.json")
            and input_binding["constructor_case_journal_sha256"]
            == ctor["private_case_journal_sha256"]
            == sha(ctor_journal),
            "Exact constructed inputs must be bound by each successful timing stage",
        )
        collection.append(commitment)
        bundles.append((source, ctor_journal, folder / "scored-cases.jsonl"))
    receipt_sha = hashlib.sha256(
        json.dumps(collection, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(
        collection[0] == extended["retained_v151_clip0"]
        and receipt_sha == public["comparison"]["receipt_collection_sha256"],
        "Actual ordered pilot completion collection differs from the public receipt",
    )
    return (
        bundles,
        public["comparison"]["common_input_timing"],
        {
            "v155_published_result_sha256": sha(COMPLETE),
            "v155_receipt_collection_sha256": receipt_sha,
            "v126_timing_freeze_sha256": cal_sha,
        },
    )


def audio_search_indices(samples):
    """Audio-only ceiling before picture, speech and speaker gates."""
    require(type(samples) is int and samples >= 0, "Exact saved PCM sample count required")
    audio_ticks = samples // lipsync.HOP
    windows = (
        lipsync.Window("track:0", index, 1.0, 0.0, xcorr.HOP_TICKS)
        for index in range(math.ceil(audio_ticks / xcorr.HOP_TICKS))
    )
    return {window.index for window in windows if xcorr.full_search(window, audio_ticks)}


def replay_window(track_id, saved, eligible):
    index, peak, offset = saved["window_index"], saved["peak_correlation"], saved["offset_ticks"]
    require(
        type(index) is int
        and index in eligible
        and type(offset) is int
        and saved["unit_ref"] == f"{track_id}@window={index}"
        and saved["hop_ticks"] == xcorr.HOP_TICKS,
        "Measured window lies outside the exact search grid",
    )
    window = lipsync.Window(
        track_id, index, saved["speech_fraction"], saved["median_face_px"], xcorr.HOP_TICKS
    )
    require(
        window.start_tick == saved["start_tick"]
        and window.ticks.stop == saved["end_tick"]
        and lipsync.MIN_SPEECH_FRACTION <= window.speech_fraction <= 1
        and math.isfinite(window.face_px)
        and window.face_px > 0,
        "Measured window has inconsistent tick or face support",
    )
    profile = saved["lag_profile"]
    expected_lags = list(range(-lipsync.VSHIFT, lipsync.VSHIFT + 1))
    correlations = [row["correlation"] for row in profile]
    require(
        [row["offset_ticks"] for row in profile] == expected_lags
        and all(
            type(value) in (int, float) and math.isfinite(value) and abs(value) <= 1 + 1e-9
            for value in correlations
        ),
        "All finite saved lag coefficients required",
    )
    winner = max(range(len(correlations)), key=correlations.__getitem__)
    require(
        offset == expected_lags[winner] and math.isclose(peak, correlations[winner], rel_tol=0, abs_tol=1e-9),
        "Saved winning lag or peak contradicts the complete lag profile",
    )
    return lipsync.WindowSync(
        window,
        (window.start_tick,),
        (1 - np.asarray(correlations, dtype=np.float64))[None, :],
        offset,
        1 - peak,
        peak,
    )


def track_opportunity(track, eligible):
    track_id = track["track_id"]
    windows = [replay_window(track_id, window, eligible) for window in track["windows"]]
    indices = [window.window.index for window in windows]
    require(windows and indices == sorted(set(indices)), "Ordered unique measured track windows required")
    result = lipsync.track_sync(windows, xcorr.SETTINGS)
    require(
        track["minimum_trusted_peak"] == xcorr.MIN_PEAK
        and track["reading"] == result.reading
        and track["baseline_offset_ticks"] == result.baseline
        and track["collapse_threshold"] == result.collapse_threshold
        and [row["low_correlation"] for row in track["windows"]] == list(result.collapsed)
        and track["segments"] == [asdict(segment) for segment in (*result.steps, *result.collapses)],
        "Frozen change-rule replay differs from the saved timing outcome",
    )
    trusted = sum(
        not down and window.confidence >= xcorr.MIN_PEAK
        for window, down in zip(windows, result.collapsed, strict=True)
    )
    adjacent_pairs = sum(b == a + 1 for a, b in zip(indices, indices[1:], strict=False))
    return {
        "windows": windows,
        "trusted_windows": trusted,
        "minimum_step_support": trusted >= 2 * lipsync.MIN_RUN_WINDOWS,
        "minimum_collapse_support": result.collapse_threshold is not None and adjacent_pairs > 0,
        "steps": len(result.steps),
        "collapses": len(result.collapses),
        "stable_offset_runs": len(result.runs),
    }


def case_opportunity(row, case):
    timing = row["timing"]
    samples = case["expected"]["audio_samples"]
    eligible = audio_search_indices(samples)
    proof = timing["details"].get("av_input", {})
    require(
        type(timing["assessed"]) is bool and type(timing["flagged"]) is bool,
        "Explicit timing status required",
    )
    if timing["assessed"]:
        require(
            timing["status"] == "ok"
            and proof["version"] == "av-correlation-input/1"
            and proof["status"] == "ready"
            and proof["method"] == "mouth_rms_correlation"
            and proof["source_sha256"] == row["case_file_sha256"]
            and proof["pcm_sha256"] == case["audio_output_clock_mapping"]["pcm_sha256"]
            and proof["samples"] == samples
            and proof["sample_rate"] == lipsync.SR
            and proof["tick_rate"] == lipsync.FPS
            and proof["samples_per_tick"] == lipsync.HOP
            and proof["lag_search_ticks"] == [-lipsync.VSHIFT, lipsync.VSHIFT]
            and Fraction(proof["audio_origin_pts"]) * Fraction(proof["audio_time_base"])
            == Fraction(case["expected"]["audio_start_seconds"])
            and len(proof["tracks"]) == 1
            and proof["tracks"][0]["track_id"] == "track:0",
            "Actual measured common-input timing proof required",
        )
        track = track_opportunity(proof["tracks"][0], eligible)
        windows = track["windows"]
        require(
            timing["metrics"]["windows"] == len(windows)
            and timing["metrics"]["offset_steps"] == track["steps"]
            and timing["metrics"]["correlation_collapses"] == track["collapses"],
            "Saved diagnostic metrics differ from the replay",
        )
    else:
        require(
            timing["status"] == "not_applicable"
            and timing["reason"] == UNATTRIBUTED
            and not proof
            and not timing["unit_scores"],
            "Expected explicit speaker-attribution failure for an unassessed GRID case",
        )
        track = {
            "windows": [],
            "trusted_windows": 0,
            "minimum_step_support": False,
            "minimum_collapse_support": False,
            "steps": 0,
            "collapses": 0,
            "stable_offset_runs": 0,
        }
        windows = track["windows"]
    require(
        row["timing_lag_windows_replayed"] == len(windows)
        and row["timing_lag_coefficients_replayed"] == len(windows) * (2 * lipsync.VSHIFT + 1)
        and timing["flagged"] is bool(track["steps"] or track["collapses"])
        and timing["flagged"] is bool(timing["segments"]) is (timing["aggregate_raw"] is not None),
        "Saved lag/window and flag totals disagree",
    )
    return track | {"audio_search_indices": sorted(eligible), "flagged": timing["flagged"]}


def delay_contract(plan, base, delay):
    """Bind exact waveform value routing using the completed constructor's saved proof."""
    support = plan["audio_delay"]
    a, b = support["output_samples_half_open"]
    c, d = support["source_samples_half_open"]
    shift = support["shift_samples"]
    samples = delay["expected"]["audio_samples"]
    origin = Fraction(delay["expected"]["audio_start_seconds"])
    origin_sample = origin * lipsync.SR
    original_pcm, delayed_pcm = (case["expected"]["pcm_s16_sha256"] for case in (base, delay))
    require(
        support["status"] == "ready"
        and shift == 5120
        and shift % lipsync.HOP == 0
        and support["preserve_length_origins_and_outside_samples"] is True
        and support["padding_or_wraparound"] is False
        and origin_sample.denominator == 1
        and 0 <= c < d <= samples
        and 0 < a < b < samples
        and (a - c, b - d) == (shift, shift)
        and base["waveform_id"] == "original_pcm"
        and delay["waveform_id"] == "delayed_audio_control"
        and all(
            base["expected"][key] == delay["expected"][key]
            for key in ("audio_start_seconds", "audio_samples", "rgb_sha256", "picture_pts_ms")
        )
        and original_pcm != delayed_pcm
        and base["changed_audio_samples"] == 0
        and 0 < delay["changed_audio_samples"] <= b - a
        and delay["changed_native_pixels"] == 0,
        "Exact local delayed waveform on an unchanged picture and sample clock required",
    )

    def mapping(out_start, out_end, in_start, in_end):
        return {
            "output_samples_half_open": [out_start, out_end],
            "input_samples_half_open": [in_start, in_end],
            "input_source_pts_half_open": [in_start + int(origin_sample), in_end + int(origin_sample)],
            "time_base": f"1/{lipsync.SR}",
        }

    bounds = [str(origin + Fraction(index, lipsync.SR)) for index in (a, b)]
    require(
        delay["source_time_bounds_seconds"] == bounds
        and base["audio_value_mapping"] == [mapping(0, samples, 0, samples)]
        and delay["audio_value_mapping"]
        == [mapping(0, a, 0, a), mapping(a, b, c, d), mapping(b, samples, b, samples)],
        "Saved constructor value routing differs from the exact delay control",
    )
    return {
        "shift_samples": shift,
        "shift_ticks": shift // lipsync.HOP,
        "shift_seconds": str(Fraction(shift, lipsync.SR)),
        "output_samples_half_open": [a, b],
        "source_samples_half_open": [c, d],
        "output_seconds_half_open": bounds,
        "source_seconds_half_open": [str(origin + Fraction(index, lipsync.SR)) for index in (c, d)],
        "audio_origin_seconds": str(origin),
        "pcm_samples": samples,
    }


def replay():
    bundles, saved_totals, commitments = saved_inputs()
    rows, delay_pairs, controls = [], [], []
    for source, constructor_journal, score_journal in bundles:
        constructor_rows, score_rows = lines(constructor_journal), lines(score_journal)
        condition_results = {}
        for spec, case, row in zip(source["cases"], constructor_rows, score_rows, strict=True):
            condition = spec["id"]
            require(
                case["condition"] == row["condition"] == condition
                and case["asset_index"] == row["asset_index"] == source["ordinal"]
                and case["source_slot"] == row["source_slot"] == source["source_slot"]
                and case["component_index"] == row["component_index"] == source["component_index"]
                and case["study_role"] == row["study_role"] == "pilot"
                and case["status"] == row["case_status"] == "ready"
                and case["waveform_id"] == row["waveform_id"] == spec["waveform_id"]
                and case["file_sha256"] == row["case_file_sha256"]
                and case["source_time_bounds_seconds"] == row["truth"]["requested_edit_interval_seconds"],
                "Saved scoring row must match the exact constructed pilot case",
            )
            track = case_opportunity(row, case)
            rows.append((condition, track))
            condition_results[condition] = case, track
        base, original_track = condition_results["original"]
        delay, delay_track = condition_results[DELAY]
        controls.append(delay_contract(source, base, delay))
        delay_pairs.append((original_track["windows"], delay_track["windows"]))

    measured = sum(bool(row["windows"]) for _, row in rows)
    flagged = sum(row["flagged"] for _, row in rows)
    require(
        len(rows) == 176
        and len(controls) == 16
        and all(control == controls[0] for control in controls)
        and measured == saved_totals["assessed_cases"]
        and flagged == saved_totals["flagged_cases"],
        "Full matched GRID pilot and original timing aggregate required",
    )
    matched = [
        (a, b)
        for originals, delays in delay_pairs
        for a in originals
        for b in delays
        if a.start_tick == b.start_tick and a.end_tick == b.end_tick
    ]
    trusted = [(a, b) for a, b in matched if min(a.confidence, b.confidence) >= xcorr.MIN_PEAK]
    step_count = sum(row["minimum_step_support"] for _, row in rows)
    collapse_count = sum(row["minimum_collapse_support"] for _, row in rows)
    return {
        "schema": "chipmunk.grid-timing-opportunity/1",
        "scope": "Saved dense common-input timing, distinct from the serving tracker and from AI attribution.",
        "commitments": commitments,
        "cases": len(rows),
        "measured_cases": measured,
        "unassessed_speaker_attribution_cases": len(rows) - measured,
        "measured_but_insufficient_for_either_rule": sum(
            bool(row["windows"]) and not (row["minimum_step_support"] or row["minimum_collapse_support"])
            for _, row in rows
        ),
        "measured_windows": sum(len(row["windows"]) for _, row in rows),
        "trusted_peak_windows": sum(row["trusted_windows"] for _, row in rows),
        "saved_lag_coefficients_checked": sum(len(row["windows"]) for _, row in rows)
        * (2 * lipsync.VSHIFT + 1),
        "stable_offset_runs": sum(row["stable_offset_runs"] for _, row in rows),
        "audio_only_full_search_window_count_histogram": dict(
            Counter(len(row["audio_search_indices"]) for _, row in rows)
        ),
        "audio_only_full_search_window_indices": sorted(
            {index for _, row in rows for index in row["audio_search_indices"]}
        ),
        "minimum_step_support_cases": step_count,
        "minimum_collapse_support_cases": collapse_count,
        "flagged_cases": flagged,
        "zero_flags_have_structural_opportunity": bool(step_count or collapse_count)
        if flagged == 0
        else None,
        "measured_cases_by_condition": dict(Counter(condition for condition, row in rows if row["windows"])),
        "known_delay_controls": {
            **controls[0],
            "pairs": len(controls),
            "measured_delayed_cases": sum(bool(b) for _, b in delay_pairs),
            "pairs_with_both_windows_at_same_time": len(matched),
            "pairs_with_both_peaks_at_least_0_4": len(trusted),
            "measured_lag_delta_ticks_histogram": dict(
                sorted(Counter(b.offset - a.offset for a, b in matched).items())
            ),
            "exact_positive_8_tick_deltas": sum(b.offset - a.offset == 8 for a, b in matched),
            "exact_positive_8_tick_deltas_both_trusted": sum(b.offset - a.offset == 8 for a, b in trusted),
        },
    }


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2, allow_nan=False))
