"""Score fixed GRID pilot cases with the existing native mouth heads."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import subprocess
import types
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from scripts import grid_transfer_plan_v143 as planning

ROOT, BASE, PANEL, ARCHIVE = planning.ROOT, planning.BASE, planning.PANEL, planning.ARCHIVE
TRIAL = BASE / "grid-native-mouth-scores-v151"
CACHE = ARCHIVE / "grid-native-mouth-scores-v151"
PUBLIC = PANEL / "grid-native-mouth-scores-freeze-v151.json"
HEADS = ARCHIVE / "native-mouth-heads-v125/reference-and-heads.json"
CALIBRATIONS = ARCHIVE / "native-mouth-calibration-v126/calibrations-and-research-points.json"
ARMS = planning.ARMS
SETTINGS = ("0", "1", "2")
PRIMARY = "2"
CODE = (
    "backend/scripts/grid_native_mouth_scores_v151.py",
    "backend/tests/test_grid_native_mouth_scores_v151.py",
    "docs/grid-native-mouth-scores-v151-method.md",
    "backend/scripts/grid_transfer_plan_v143.py",
    "backend/scripts/grid_native_clips_v149.py",
    "backend/scripts/grid_native_mouth_features_v150.py",
    "backend/scripts/mouth_articulation_core_v122.py",
    "backend/scripts/native_mouth_features_v123.py",
    "backend/scripts/review_native_mouth_features_v124.py",
    "backend/scripts/fit_native_mouth_heads_v125.py",
    "backend/scripts/native_mouth_readout_v126.py",
    "backend/scripts/evaluate_native_mouth_v127.py",
    "backend/scripts/review_native_mouth_v129.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/local_probe_limits_v24.py",
)
read, write, sha, require = planning.read, planning.write, planning.sha, planning.require


def producers():
    from scripts import grid_native_clips_v149 as construction
    from scripts import grid_native_mouth_features_v150 as features

    return construction, features


def pilots(plan):
    selected = [source for source in plan["sources"] if source["role"] == "pilot"]
    require(len(selected) == 16, "Exactly sixteen original pilot clips required")
    require(
        [source["ordinal"] for source in selected] == sorted({source["ordinal"] for source in selected}),
        "Unique ordered original pilot ordinals required",
    )
    require(
        all([case["id"] for case in source["cases"]] == list(planning.CASES) for source in selected),
        "Every original pilot needs its eleven fixed cases",
    )
    return selected


def pilot(plan, ordinal):
    require(type(ordinal) is int and 0 <= ordinal < 32, "Original integer pilot ordinal required")
    source = next((source for source in pilots(plan) if source["ordinal"] == ordinal), None)
    require(source is not None, "Reserve or unknown clip cannot be scored")
    return source


def learned_bindings(plan):
    refs = plan["dependencies"]["detectors"]
    require(
        refs["arms"] == list(ARMS)
        and refs["settings"] == [0, 1, 2]
        and refs["primary_setting"] == 2
        and refs["refitting_permitted"] is False
        and plan["new_fitting_or_calibration_permitted"] is False
        and sha(HEADS) == refs["heads_sha256"]
        and sha(CALIBRATIONS) == refs["calibrations_and_settings_sha256"],
        "Actual original v125 heads and v126 calibrations/settings required",
    )
    return {
        "reference_and_heads_sha256": refs["heads_sha256"],
        "calibrations_and_research_points_sha256": refs["calibrations_and_settings_sha256"],
    }


def identity():
    plan = planning.load_completed()
    construction, features = producers()
    frozen = planning.frozen_context(construction.TRIAL, construction.PUBLIC, {})
    planning.frozen_context(features.TRIAL, features.PUBLIC, {})
    return {
        "schema": "chipmunk.grid-native-mouth-score-freeze/151",
        "code_sha256": {name: sha(ROOT / name) for name in CODE},
        "transfer_plan_freeze_sha256": sha(planning.PUBLIC),
        "private_transfer_plan_sha256": sha(planning.TRIAL / "bind-00/output/private-plan.json"),
        "construction_freeze_sha256": sha(construction.PUBLIC),
        "feature_freeze_sha256": sha(features.PUBLIC),
        **learned_bindings(plan),
        "binaries_sha256": frozen["binaries_sha256"],
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "torch",
                "Pillow",
                "opencv-python-headless",
                "psutil",
                "threadpoolctl",
            )
        },
        "pilot_clips": len(pilots(plan)),
        "planned_cases": 176,
        "arms": list(ARMS),
        "settings": list(SETTINGS),
        "primary_setting": PRIMARY,
        "comparison": "strict_greater_than",
        "terminal_cell_policy": "exclude_unverified_terminal_cell",
        "scope": "Frozen mouth-head transfer on the fixed pilot and separate dense common-input timing diagnostic. No complete serving baseline, reserve execution, refitting or confidence change.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","pilot_clips":16,"planned_cases":176,"new_neural_calls":0}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged published pilot scoring contract required",
    )


def owned(path, clip_root, cache):
    root, resolved = clip_root.resolve(), Path(path).resolve()
    require(
        root == cache.resolve() / clip_root.name and resolved.is_relative_to(root),
        "Case or array path escaped its owned pilot cache",
    )
    return resolved


def validate_pair(source, spec, case, feature, construction_cache, feature_cache, journal_sha):
    ordinal, name = source["ordinal"], spec["id"]
    require(
        case["asset_index"] == feature["asset_index"] == ordinal
        and case["condition"] == feature["condition"] == name
        and case["source_slot"] == feature["source_group"] == source["source_slot"]
        and case["component_index"] == feature["component_index"] == source["component_index"]
        and case["study_role"] == feature["study_role"] == source["role"] == "pilot"
        and case["waveform_id"] == feature["waveform_id"] == spec["waveform_id"]
        and feature["constructor_case_journal_sha256"] == journal_sha
        and case["status"] in {"ready", "unavailable"}
        and feature["input_status"] == case["status"],
        "Exact same source/component/case/waveform and actual constructor journal required",
    )
    if case["status"] == "unavailable":
        require(
            feature["status"] == "unavailable"
            and feature["reason"] == case["reason"]
            and not {"file", "speech", "frames"}.intersection(feature),
            "Unavailable constructor case cannot acquire features or scope",
        )
        return
    speech, table, expected = feature["speech"], feature["table"], case["expected"]
    require(
        feature["status"] == "extracted"
        and feature["source_clip_sha256"] == case["file_sha256"]
        and feature["dtype"] == "float32"
        and feature["shape"][1:] == [768]
        and speech["dtype"] == "float64"
        and speech["shape"] == [len(speech["indices"]), 3840]
        and speech["waveform_id"] == case["waveform_id"]
        and speech["pcm_sha256"] == case["audio_output_clock_mapping"]["pcm_sha256"]
        and table["time_base"] == "1/1000"
        and table["pts"] == expected["picture_pts_ms"]
        and len(table["pts"]) == 75
        and [Fraction(p, 1000) for p in table["pts"]]
        == [Fraction(p) * Fraction(case["source_video_time_base"]) for p in case["source_video_pts"]]
        and [f["frame_index"] for f in feature["frames"]] == list(range(75))
        and [f["pts"] for f in feature["frames"]] == table["pts"]
        and [f["rgb_sha256"] for f in feature["frames"]] == expected["rgb_sha256"]
        and feature["constructed_last_picture_duration_ms"]
        == expected["constructed_last_picture_duration_ms"],
        "Exact observed native feature table, decoded RGB and matching complete soundtrack required",
    )
    clip = f"clip-{ordinal:02d}"
    owned(case["file"], construction_cache / clip, construction_cache)
    owned(feature["file"], feature_cache / clip, feature_cache)
    owned(speech["file"], feature_cache / clip, feature_cache)


def input_binding(plan, ordinal):
    source = pilot(plan, ordinal)
    construction, features = producers()
    cases, rows = construction.load_completed(ordinal), features.load_completed(ordinal)
    clip = f"clip-{ordinal:02d}"
    construction_stage = construction.TRIAL / clip / "bind-00"
    feature_stage = features.TRIAL / clip / "authored"
    paths = {
        "constructor_case_journal_sha256": construction.CACHE / clip / "cases.jsonl",
        "feature_case_journal_sha256": features.CACHE / clip / "feature-cases.jsonl",
        "constructor_result_sha256": construction_stage / "output/result.json",
        "constructor_profile_sha256": construction_stage / "profile.json",
        "feature_result_sha256": feature_stage / "output/result.json",
        "feature_profile_sha256": feature_stage / "profile.json",
    }
    bindings = {name: sha(path) for name, path in paths.items()}
    require(
        len(cases) == len(rows) == len(source["cases"]) == 11,
        "All eleven actual construction/feature records required",
    )
    for spec, case, row in zip(source["cases"], cases, rows, strict=True):
        validate_pair(
            source,
            spec,
            case,
            row,
            construction.CACHE,
            features.CACHE,
            bindings["constructor_case_journal_sha256"],
        )
    return source, cases, rows, bindings


def score_windows(indices, margins, table):
    from scripts.native_mouth_readout_v126 import windows

    # The last eligible point is retained in the point readout. Excluding it
    # here removes just the window whose right edge requires the final cell.
    keep = len(indices)
    if indices and indices[-1] == len(table) - 2:
        keep -= 1
    return windows(indices[:keep], margins[:keep], table, 0)


def blind_score(row, learned, calibrated):
    from app.detectors.video.decode import FrameTable
    from scripts import fit_native_mouth_heads_v125 as fitted
    from scripts import native_mouth_readout_v126 as method
    from scripts import review_native_mouth_v129 as replay

    indices, visual, audio = fitted.point_features(row)
    replay_indices, replay_margins = replay.reconstruct(row, learned)
    require(indices == replay_indices, "Independent blind native centre schedule differs")
    table = FrameTable.from_json(row["table"]) if indices else None
    representations = fitted.core.representations(visual, audio, learned["reference"]) if indices else None
    arms = {}
    for arm in ARMS:
        raw = fitted.margins(representations[arm], learned["heads"][arm]) if indices else []
        replay.close(raw, replay_margins[arm])
        windows = score_windows(indices, raw, table) if indices else []
        cal = calibrated[arm]
        point_probabilities = method.probability(raw, cal["point"])
        replay.close(point_probabilities, replay.probability(raw, cal["point"]))
        best = windows[0]["response"] if windows else None
        clip_probability = None if best is None else float(method.probability(best, cal["clip"]))
        if best is not None:
            replay.close(clip_probability, replay.probability(best, cal["clip"]))
        arms[arm] = {
            "point_margins": list(map(float, raw)),
            "windows": windows,
            "window_score": best,
            "study_point_probabilities": point_probabilities.tolist(),
            "study_clip_probability": clip_probability,
        }
    return {"eligible_centres": indices, "arms": arms}


def truth_for(spec, case):
    ready, planned = case["status"] == "ready", spec["planned_neural_provenance"]
    verified = case["neural_provenance_verified"]
    require(
        type(planned) is bool
        and case["planned_neural_provenance"] is planned
        and verified is (ready and planned),
        "Actual construction provenance differs from the fixed case specification",
    )
    interval = None
    if ready and spec["edit_frames_half_open"] is not None:
        require(
            spec["edit_frames_half_open"] == case["edit_frames_half_open"] == list(planning.EDIT),
            "Exact originally requested native edit support required",
        )
        interval = [
            str(Fraction(case["source_video_pts"][i]) * Fraction(case["source_video_time_base"]))
            for i in planning.EDIT
        ]
        require(interval == case["source_time_bounds_seconds"], "Actual native edit PTS differ")
    changed = case.get("observed_neural_pixel_change", False)
    require(
        type(changed) is bool and (not changed or verified), "Observed neural pixels need verified provenance"
    )
    if ready:
        count = case["changed_native_pixels"]
        require(
            type(count) is int and count >= 0 and changed is (verified and count > 0),
            "Observed neural-pixel status differs from construction counts",
        )
    category = "unavailable"
    if ready:
        category = "observable_neural" if changed else ("zero_effect_neural" if verified else "non_neural")
    return {
        "category": category,
        "planned_neural_provenance": planned,
        "neural_provenance_verified": verified,
        "observed_neural_pixel_change": changed,
        "requested_edit_interval_seconds": interval,
        "observable_neural_interval_seconds": interval if changed else None,
    }


def roi_metrics(mask, bounds, changed_pixels):
    import numpy as np

    require(
        mask.ndim == 2 and mask.dtype == np.uint8 and np.isin(mask, (0, 255)).all(),
        "Exact binary native changed-pixel mask required",
    )
    x0, y0, x1, y1 = bounds
    require(
        all(type(v) is int for v in bounds)
        and 0 <= x0 < x1 <= mask.shape[1]
        and 0 <= y0 < y1 <= mask.shape[0],
        "Inspected native ROI outside its picture",
    )
    total = int(np.count_nonzero(mask))
    require(total == changed_pixels, "Observed native changed-pixel count differs")
    intersection = int(np.count_nonzero(mask[y0:y1, x0:x1]))
    area = (x1 - x0) * (y1 - y0)
    return {
        "pixel_iou": intersection / (total + area - intersection),
        "changed_pixel_coverage": None if not total else intersection / total,
    }


def selected_evidence(row, case, window, construction_cache):
    import hashlib

    import numpy as np
    from PIL import Image

    anchors = {}
    for role in ("before", "during", "after"):
        frame = row["frames"][window[f"{role}_frame_index"]]
        anchors[role] = {
            "frame_index": frame["frame_index"],
            "pts": frame["pts"],
            "time_base": row["table"]["time_base"],
            "rgb_sha256": frame["rgb_sha256"],
            "source_pts": case["source_video_pts"][frame["frame_index"]],
            "source_time_base": case["source_video_time_base"],
        }
    index = window["centre_frame_index"]
    bounds = row["frames"][index]["region_bbox"]
    proof = next((item for item in case["edit_proofs"] if item["frame_index"] == index), None)
    region = {
        "inspected_region_bbox": bounds,
        "frame_index": index,
        "pixel_iou": None,
        "changed_pixel_coverage": None,
        "observed_mask": proof is not None,
    }
    if proof is not None:
        path = owned(
            proof["mask_file"], construction_cache / f"clip-{case['asset_index']:02d}", construction_cache
        )
        require(
            sha(path) == proof["mask_sha256"] and proof["pts"] == case["source_video_pts"][index],
            "Actual bound mask/native PTS required",
        )
        with Image.open(path) as image:
            require(image.format == "PNG", "Exact lossless native mask required")
            mask = np.asarray(image)
        require(
            mask.shape == (row["table"]["height"], row["table"]["width"])
            and hashlib.sha256(mask.tobytes()).hexdigest() == proof["mask_raw_sha256"],
            "Native changed-pixel mask bytes or dimensions differ",
        )
        region.update(roi_metrics(mask, bounds, proof["changed_pixels"]))
    return {"frame_references": anchors, "inspected_roi": region}


def selected_windows(windows, point):
    require(
        point["comparison"] == "strict_greater_than" and math.isfinite(point["raw_margin"]),
        "Exact finite frozen raw research point required",
    )
    return [window for window in windows if window["response"] > point["raw_margin"]]


def case_outcomes(score, truth, row, case, calibrated, construction_cache):
    from app.detectors.video.decode import FrameTable
    from scripts import native_mouth_readout_v126 as method
    from scripts import review_native_mouth_v129 as replay

    table = FrameTable.from_json(row["table"]) if row["status"] == "extracted" else None
    bounds = list(planning.EDIT) if truth["category"] == "observable_neural" else None
    outcomes, evidence_by_centre = {}, {}
    for arm, part in score["arms"].items():
        outcomes[arm] = {}
        for setting in SETTINGS:
            selected = selected_windows(part["windows"], calibrated[arm]["research_points"][setting])
            spans = [[w["start_seconds"], w["end_seconds"]] for w in selected]
            metrics = method.interval_metrics(spans, truth["observable_neural_interval_seconds"])
            if table is not None:
                require(
                    all(w["support_frames_half_open"][1] < len(table) for w in selected),
                    "Unverified terminal cell cannot enter localization",
                )
                checked = replay.frame_union(table, 0, selected, bounds)
                for key, value in checked.items():
                    replay.close(metrics[key], value)
            evidence = None
            if selected:
                centre = selected[0]["centre_frame_index"]
                if centre not in evidence_by_centre:
                    evidence_by_centre[centre] = selected_evidence(row, case, selected[0], construction_cache)
                evidence = evidence_by_centre[centre]
            outcomes[arm][setting] = {
                "assessed": bool(part["windows"]),
                "flagged": bool(selected),
                "study_probability": part["study_clip_probability"],
                "selected_intervals_seconds": spans,
                **metrics,
                "requested_edit_overlap_seconds": method.interval_metrics(
                    spans, truth["requested_edit_interval_seconds"]
                )["edit_overlap_seconds"],
                "evidence": evidence,
            }
    return outcomes


def visual_invariance(cases, scores):
    import numpy as np

    by_case = {case["condition"]: case for case in cases}
    by_score = {score["condition"]: score for score in scores}
    original, delayed = (by_case[name] for name in ("original", "audio_delay_320ms"))
    a, b = (by_score[name] for name in ("original", "audio_delay_320ms"))
    if (
        original["status"] != "ready"
        or delayed["status"] != "ready"
        or original["expected"]["rgb_sha256"] != delayed["expected"]["rgb_sha256"]
        or original["expected"]["picture_pts_ms"] != delayed["expected"]["picture_pts_ms"]
        or not a["eligible_centres"]
        or a["eligible_centres"] != b["eligible_centres"]
    ):
        return {
            "status": "unassessed",
            "bitwise_equal": None,
            "common_centres": 0,
            "reason": "Ready cases with identical RGB, clocks and nonempty centre scope required",
        }
    left, right = (score["arms"]["visual_only"] for score in (a, b))

    def numeric_bytes(part):
        return np.asarray(
            part["point_margins"] + [w["response"] for w in part["windows"]], dtype="<f8"
        ).tobytes()

    same = left["windows"] == right["windows"] and numeric_bytes(left) == numeric_bytes(right)
    return {
        "status": "verified" if same else "failed",
        "bitwise_equal": same,
        "common_centres": len(a["eligible_centres"]),
        "reason": None,
    }


def counts(rows, arm, setting):
    categories = Counter(row["truth"]["category"] for row in rows)
    outcomes = [(row["truth"]["category"], row["outcomes"][arm][setting]) for row in rows]
    positive = [item for kind, item in outcomes if kind == "observable_neural"]
    negative = [item for kind, item in outcomes if kind == "non_neural"]
    zero = [item for kind, item in outcomes if kind == "zero_effect_neural"]
    observed_flags = sum(item["flagged"] for item in positive)
    false_ai = sum(item["flagged"] for item in negative)
    return {
        "planned_cases": len(rows),
        "unavailable_cases": categories["unavailable"],
        "observable_neural_cases": len(positive),
        "non_neural_cases": len(negative),
        "zero_effect_neural_cases": len(zero),
        "assessed_cases": sum(item["assessed"] for _, item in outcomes),
        "unassessed_ready_cases": sum(
            not item["assessed"] for kind, item in outcomes if kind != "unavailable"
        ),
        "observable_neural_flags": observed_flags,
        "observable_neural_flag_rate": observed_flags / len(positive) if positive else None,
        "observable_neural_localizations_iou_0_2": sum(item["localized_iou_0_2"] for item in positive),
        "observable_neural_mean_temporal_iou": (
            sum(item["temporal_union_iou"] for item in positive) / len(positive)
        )
        if positive
        else None,
        "observable_neural_edit_overlap_seconds": sum(item["edit_overlap_seconds"] for item in positive),
        "observable_neural_outside_edit_flagged_seconds": sum(
            item["outside_edit_flagged_seconds"] for item in positive
        ),
        "observable_neural_assessed_misses": sum(
            item["assessed"] and not item["flagged"] for item in positive
        ),
        "observable_neural_unassessed": sum(not item["assessed"] for item in positive),
        "non_neural_ai_flags": false_ai,
        "non_neural_ai_flag_rate": false_ai / len(negative) if negative else None,
        "non_neural_assessed": sum(item["assessed"] for item in negative),
        "non_neural_unassessed": sum(not item["assessed"] for item in negative),
        "zero_effect_neural_flags": sum(item["flagged"] for item in zero),
        "zero_effect_neural_assessed": sum(item["assessed"] for item in zero),
    }


def paired(rows, arm):
    pairs = [
        (row["truth"]["category"], row["outcomes"][arm][PRIMARY], row["outcomes"]["visual_only"][PRIMARY])
        for row in rows
        if row["truth"]["category"] in {"observable_neural", "non_neural"}
    ]
    return {
        "paired_binary_cases": len(pairs),
        "common_assessed_cases": sum(a["assessed"] and b["assessed"] for _, a, b in pairs),
        "additional_assessed_cases": sum(a["assessed"] and not b["assessed"] for _, a, b in pairs),
        "lost_assessed_cases": sum(b["assessed"] and not a["assessed"] for _, a, b in pairs),
        "additional_localizations": sum(
            kind == "observable_neural" and a["localized_iou_0_2"] and not b["localized_iou_0_2"]
            for kind, a, b in pairs
        ),
        "lost_localizations": sum(
            kind == "observable_neural" and b["localized_iou_0_2"] and not a["localized_iou_0_2"]
            for kind, a, b in pairs
        ),
        "additional_non_neural_ai_flags": sum(
            kind == "non_neural" and a["flagged"] and not b["flagged"] for kind, a, b in pairs
        ),
        "removed_non_neural_ai_flags": sum(
            kind == "non_neural" and b["flagged"] and not a["flagged"] for kind, a, b in pairs
        ),
    }


def summary(rows):
    return {
        "arms": {arm: {setting: counts(rows, arm, setting) for setting in SETTINGS} for arm in ARMS},
        "primary_paired_vs_visual": {arm: paired(rows, arm) for arm in ARMS if arm != "visual_only"},
        "common_input_timing": {
            "assessed_cases": sum(row["timing"]["assessed"] for row in rows),
            "flagged_cases": sum(row["timing"]["flagged"] for row in rows),
            "ordinary_edit_timing_flags": sum(
                row["truth"]["category"] == "non_neural"
                and row["condition"] != "original"
                and row["timing"]["flagged"]
                for row in rows
            ),
            "untouched_original_timing_flags": sum(
                row["condition"] == "original" and row["timing"]["flagged"] for row in rows
            ),
        },
    }


def aggregate_records(plan, rows):
    expected = [
        (s["ordinal"], c["id"], s["source_slot"], s["component_index"])
        for s in pilots(plan)
        for c in s["cases"]
    ]
    observed = [(r["asset_index"], r["condition"], r["source_slot"], r["component_index"]) for r in rows]
    require(
        len(rows) == 176 and observed == expected and all(r["study_role"] == "pilot" for r in rows),
        "All 176 original pilot cases required in order, without reserve or substitution",
    )

    def grouped(field):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        return [
            {
                "source_slots": len({r["source_slot"] for r in group}),
                "planned_cases": len(group),
                "paired_vs_visual": {arm: paired(group, arm) for arm in ARMS if arm != "visual_only"},
            }
            for group in groups.values()
        ]

    components = grouped("component_index")
    return {
        "planned_cases": 176,
        "source_slots": len({r["source_slot"] for r in rows}),
        "known_components": len(components),
        **summary(rows),
        "conditions_at_primary": {
            name: {arm: counts([r for r in rows if r["condition"] == name], arm, PRIMARY) for arm in ARMS}
            for name in planning.CASES
        },
        "source_slot_paired_aggregates": grouped("source_slot"),
        "component_paired_aggregates": components,
    }


def score(ordinal, output):
    import torch
    from threadpoolctl import threadpool_limits

    from scripts import native_mouth_readout_v126 as method
    from scripts import review_native_mouth_v129 as replay

    plan = planning.load_completed()
    source, cases, rows, bindings = input_binding(plan, ordinal)
    learned_bindings(plan)
    for name, digest in read(PUBLIC)["binaries_sha256"].items():
        path = shutil.which(name)
        require(
            path is not None and sha(path) == digest, "Qualified timing diagnostic decoder binary changed"
        )
    for case in cases:
        if case["status"] == "ready":
            require(sha(case["file"]) == case["file_sha256"], "Actual constructed clip bytes changed")
    learned, calibrated = read(HEADS), read(CALIBRATIONS)["arms"]
    construction, _ = producers()
    folder = CACHE / f"clip-{ordinal:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "input-binding.json", bindings)
    scores, forwards = [], []

    def reject_forward(*_args):
        forwards.append(True)
        raise RuntimeError("Pretrained forward forbidden in data-only scoring")

    hook = torch.nn.modules.module.register_module_forward_pre_hook(reject_forward)
    try:
        with threadpool_limits(limits=1), (folder / "scored-cases.jsonl").open("x") as journal:
            for spec, case, row in zip(source["cases"], cases, rows, strict=True):
                blind = blind_score(row, learned, calibrated)
                truth = truth_for(spec, case)
                outcomes = case_outcomes(blind, truth, row, case, calibrated, construction.CACHE)
                timing = method.current_check(row, case)
                require(
                    timing["flagged"] == bool(timing["segments"])
                    and (timing["assessed"] or not timing["flagged"]),
                    "Actual common-input timing flags need observed windows and segments",
                )
                timing_spans = [
                    [str(segment["start_s"]), str(segment["end_s"])] for segment in timing["segments"]
                ]
                timing["requested_edit_interval_metrics"] = method.interval_metrics(
                    timing_spans, truth["requested_edit_interval_seconds"]
                )
                lag_windows, lag_coefficients = replay.lag_replay(timing)
                record = {
                    "asset_index": ordinal,
                    "condition": case["condition"],
                    "study_role": "pilot",
                    "source_slot": source["source_slot"],
                    "component_index": source["component_index"],
                    "case_status": case["status"],
                    "feature_status": row["status"],
                    "reason": case["reason"],
                    "case_file_sha256": case.get("file_sha256"),
                    "waveform_id": case["waveform_id"],
                    "truth": truth,
                    **blind,
                    "outcomes": outcomes,
                    "timing": timing,
                    "timing_lag_windows_replayed": lag_windows,
                    "timing_lag_coefficients_replayed": lag_coefficients,
                }
                scores.append(record)
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
    finally:
        hook.remove()
    invariant = visual_invariance(cases, scores)
    write(folder / "visual-invariance.json", invariant)
    require(
        invariant["bitwise_equal"] is not False and not forwards,
        "Audio-only visual invariance or zero-forward requirement failed",
    )
    result = {
        "schema": "chipmunk.grid-native-mouth-score-result/151",
        "status": "complete",
        "clip_ordinal": ordinal,
        "planned_cases": len(scores),
        "primary_setting": PRIMARY,
        "terminal_cell_windows_eligible": False,
        "eligible_centres": sum(len(row["eligible_centres"]) for row in scores),
        "visual_invariance": invariant,
        **summary(scores),
        "private_score_journal_sha256": sha(folder / "scored-cases.jsonl"),
        "private_input_binding_sha256": sha(folder / "input-binding.json"),
        "private_visual_invariance_sha256": sha(folder / "visual-invariance.json"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": len(forwards),
        "new_head_or_calibration_fits": 0,
        "defaults_changed": False,
        "scope": "Fixed heads and study settings, with actual common dense YuNet timing diagnostic. This diagnostic does not reproduce the serving tracker or complete serving pipeline. Requested timing does not establish achieved synchronization.",
    }
    write(output / "result.json", result)
    print(
        json.dumps({"status": "complete", "planned_cases": len(scores), "new_neural_calls": len(forwards)}),
        flush=True,
    )


def load_completed(ordinal):
    verify()
    plan = planning.load_completed()
    source, cases, features, bindings = input_binding(plan, ordinal)
    clip = f"clip-{ordinal:02d}"
    result = planning.reviewed.stage(
        TRIAL / clip / "bind-00", sha(PUBLIC), "grid_native_mouth_scores_v151.py", 0, {}
    )
    folder = CACHE / clip
    journal, binding, invariant = (
        folder / name for name in ("scored-cases.jsonl", "input-binding.json", "visual-invariance.json")
    )
    require(
        result["schema"] == "chipmunk.grid-native-mouth-score-result/151"
        and result["clip_ordinal"] == ordinal
        and result["planned_cases"] == 11
        and result["primary_setting"] == PRIMARY
        and result["terminal_cell_windows_eligible"] is False
        and result["new_head_or_calibration_fits"] == 0
        and result["private_score_journal_sha256"] == sha(journal)
        and result["private_input_binding_sha256"] == sha(binding)
        and read(binding) == bindings
        and result["private_visual_invariance_sha256"] == sha(invariant),
        "Actual bound complete zero-forward score stage required",
    )
    rows = planning.reviewed.lines(journal)
    require(len(rows) == 11, "All eleven completed score rows required")
    for spec, case, feature, row in zip(source["cases"], cases, features, rows, strict=True):
        require(
            row["asset_index"] == ordinal
            and row["study_role"] == "pilot"
            and row["condition"] == spec["id"]
            and row["source_slot"] == source["source_slot"]
            and row["component_index"] == source["component_index"]
            and row["case_status"] == case["status"]
            and row["feature_status"] == feature["status"]
            and row["case_file_sha256"] == case.get("file_sha256")
            and row["waveform_id"] == case["waveform_id"]
            and row["truth"] == truth_for(spec, case),
            "Actual score row lost exact case/provenance correspondence",
        )
    replayed = visual_invariance(cases, rows)
    require(
        read(invariant) == result["visual_invariance"] == replayed
        and replayed["bitwise_equal"] is not False
        and result["eligible_centres"] == sum(len(r["eligible_centres"]) for r in rows)
        and all(result[key] == value for key, value in summary(rows).items()),
        "Actual score aggregates or visual invariance differ",
    )
    return rows


def aggregate_completed():
    plan = planning.load_completed()
    rows = [row for source in pilots(plan) for row in load_completed(source["ordinal"])]
    return aggregate_records(plan, rows)


def routed_popen(ordinal, argv, **kwargs):
    expected = TRIAL / f"clip-{ordinal:02d}/bind-00/output"
    require(
        type(ordinal) is int
        and 0 <= ordinal < 32
        and argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"]
        and argv[5:] == [str(expected), "0"],
        "Only exact exclusive data-only pilot scoring permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_native_mouth_scores_v151", "score", str(expected), str(ordinal)], **kwargs
    )


def run(ordinal):
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    input_binding(planning.load_completed(), ordinal)
    stage = TRIAL / f"clip-{ordinal:02d}"
    stage.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kw: routed_popen(ordinal, argv, **kw),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        runtime.run,
        {
            "TRIAL": stage,
            "verify": verify,
            "prerequisites": lambda _c, _b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "score"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "No freeze arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One explicit pilot ordinal required")
        run(int(args.arguments[0]))
    else:
        require(len(args.arguments) == 2, "Exclusive output and pilot ordinal required")
        verify()
        output, ordinal = Path(args.arguments[0]), int(args.arguments[1])
        require(
            output.resolve() == (TRIAL / f"clip-{ordinal:02d}/bind-00/output").resolve(),
            "Exact exclusive worker output required",
        )
        output.mkdir(exist_ok=False)
        score(ordinal, output)
