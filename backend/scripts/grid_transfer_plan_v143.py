"""Metadata-only fixed middle-placement, two-generator transfer planning."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts import av_conditioning_clock_v109 as conditioning
from scripts import review_grid_native_inputs_v142 as reviewed

ROOT, BASE, PANEL, ARCHIVE = reviewed.ROOT, reviewed.BASE, reviewed.PANEL, reviewed.ARCHIVE
TRIAL = BASE / "grid-transfer-plan-v143"
PUBLIC = PANEL / "grid-transfer-plan-freeze-v143.json"
EDIT = (27, 47)
EARLIER_FRAMES = 8
OFFSET = Fraction(8, 25)
ARMS = ("visual_only", "concatenation", "interaction")
CASES = (
    "original",
    "musetalk_source_time",
    "musetalk_earlier_320ms",
    "liveportrait_source_time",
    "liveportrait_earlier_320ms",
    "resample_musetalk",
    "resample_liveportrait",
    "brightness_musetalk",
    "brightness_liveportrait",
    "genuine_frame_retime",
    "audio_delay_320ms",
)
CODE = (
    "backend/scripts/grid_transfer_plan_v143.py",
    "backend/tests/test_grid_transfer_plan_v143.py",
    "docs/grid-transfer-plan-v143-method.md",
    "backend/scripts/av_conditioning_clock_v109.py",
    "backend/scripts/review_grid_native_inputs_v142.py",
    "backend/scripts/video_appearance_runtime_v108.py",
    "backend/scripts/video_appearance_corrected_v112.py",
    "backend/scripts/audio_tree_sampler_v64.py",
    "backend/scripts/local_probe_limits_v24.py",
)
read, write, sha, require = reviewed.read, reviewed.write, reviewed.sha, reviewed.require


def exact(value):
    require(type(value) in (int, str, Fraction), "Exact observed rational time required")
    return Fraction(value)


def partition(records, links):
    """Keep every source/exact-content component on one side; no quality selection."""
    require(
        len(records) == 32 and [r["ordinal"] for r in records] == list(range(32)),
        "All original32 audit observations required",
    )
    expected = reviewed.components(records)
    require(links == expected, "Actual independently reviewed component membership changed")
    roles, component_ids = {}, {}
    for number, component in enumerate(links["components"]):
        pilot = any(records[i]["source_slot"] < 8 for i in component)
        for i in component:
            require(i not in roles, "A source may appear in only one component")
            roles[i], component_ids[i] = ("pilot" if pilot else "reserve"), number
    require(sorted(roles) == list(range(32)), "Components must cover all inputs, including unavailable")
    return roles, component_ids


def frame_support(row, index):
    """Pure v109 routing from whole-PCM metadata; no pictures/arrays are opened."""
    result = {
        "frame_index": index,
        "status": "unavailable",
        "unavailable_reason": row["status"],
        "source_picture": None,
        "earlier_picture": None,
        "source_time_seconds": None,
        "earlier_time_seconds": None,
        "source_time_conditioning": None,
        "earlier_conditioning": None,
    }
    if row["status"] != reviewed.AVAILABLE:
        return result
    table, mapping = row["native_video_table"], row["pcm_mapping"]
    pictures = {p["frame_index"]: p for p in row["pictures"]}
    if index not in pictures or index - EARLIER_FRAMES not in pictures:
        result["unavailable_reason"] = "Missing actual source/earlier native picture; no replacement"
        return result
    current, earlier = pictures[index], pictures[index - EARLIER_FRAMES]
    require(
        current["pts"] == table["pts"][index] and earlier["pts"] == table["pts"][index - EARLIER_FRAMES],
        "Picture commitments differ from native clock table",
    )
    source_time, driver_time = (exact(p["pts"]) * exact(table["time_base"]) for p in (current, earlier))
    result.update(
        source_picture=copy.deepcopy(current),
        earlier_picture=copy.deepcopy(earlier),
        source_time_seconds=str(source_time),
        earlier_time_seconds=str(driver_time),
    )
    if source_time - driver_time != OFFSET:
        result["unavailable_reason"] = (
            "Observed earlier picture is not exactly320ms earlier; no index/fps inference"
        )
        return result
    samples = mapping["samples"]
    if type(samples) is not int or not 640 <= samples <= 480000:
        result["unavailable_reason"] = "Whole observed mono16k extent unsupported by fixed v109 routing"
        return result
    # v104/v109 native timestamp quantization allowance, not a local encoder dependency bound.
    native = row["native_audio_clock"]
    allowance = exact(native["time_base"]) + Fraction(1, native["sample_rate"])
    audio_origin, video_origin = exact(row["audio_origin_seconds"]), exact(row["video_origin_seconds"])
    require(
        audio_origin == Fraction(mapping["timeline"][0]["source_pts"], 16000)
        and video_origin == exact(table["pts"][0]) * exact(table["time_base"]),
        "Preserve observed separate source origins",
    )
    aligned = conditioning.frame_plan(
        index, str(source_time), str(audio_origin), str(video_origin), samples, str(allowance)
    )
    shifted = conditioning.frame_plan(
        index, str(source_time - OFFSET), str(audio_origin), str(video_origin), samples, str(allowance)
    )
    result.update(source_time_conditioning=aligned, earlier_conditioning=shifted)
    if not all(
        p["source_sample_status"] == "supported" and all(p["retained_tick_slots"]) for p in (aligned, shifted)
    ):
        result["unavailable_reason"] = (
            "Missing observed source sample or full ten-slot conditioning; no padding/substitution"
        )
        return result
    result.update(status="ready", unavailable_reason=None)
    return result


def geometry_requests(row):
    """Stable v144 API:28 stateless picture inputs plus exact repeat of picture27."""
    indices = list(range(EDIT[0] - EARLIER_FRAMES, EDIT[1]))
    result = {
        "status": "unavailable",
        "unavailable_reason": "Original video picture context unavailable",
        "frame_indices": indices,
        "repeat_frame_index": EDIT[0],
        "pictures": [],
        "planned_unique_face_task_inputs": 28,
        "planned_repeat_face_task_inputs": 1,
        "source_output_frames_half_open": list(EDIT),
        "earlier_driver_frames_half_open": [19, 39],
    }
    table = row.get("native_video_table")
    pictures = {p["frame_index"]: p for p in row.get("pictures", [])}
    if table is None or not reviewed.supported_video(table) or any(i not in pictures for i in indices):
        return result
    for i in indices:
        require(pictures[i]["pts"] == table["pts"][i], "Geometry picture clock differs")
    if any(
        (exact(table["pts"][i]) - exact(table["pts"][i - EARLIER_FRAMES])) * exact(table["time_base"])
        != OFFSET
        for i in range(*EDIT)
    ):
        result["unavailable_reason"] = (
            "Fixed eight-picture driver correspondence lacks exact320ms observed support"
        )
        return result
    result.update(
        status="ready", unavailable_reason=None, pictures=[copy.deepcopy(pictures[i]) for i in indices]
    )
    return result


def audio_delay_support(row):
    """The audio control copies observed samples; re-encode the complete changed waveform."""
    result = {
        "status": "unavailable",
        "unavailable_reason": row["status"],
        "source_pcm": "original_pcm",
        "output_pcm": "delayed_audio_control",
        "whole_input_reencoding_required": True,
        "shift_samples": 5120,
        "output_samples_half_open": None,
        "source_samples_half_open": None,
        "preserve_length_origins_and_outside_samples": True,
        "padding_or_wraparound": False,
    }
    if row["status"] != reviewed.AVAILABLE:
        return result
    table, mapping = row["native_video_table"], row["pcm_mapping"]
    if len(table["pts"]) <= EDIT[1]:
        result["unavailable_reason"] = "No observed first-after-edit timestamp"
        return result
    origin = exact(row["audio_origin_seconds"])
    bounds = [(exact(table["pts"][i]) * exact(table["time_base"]) - origin) * 16000 for i in EDIT]
    if any(p.denominator != 1 for p in bounds):
        result["unavailable_reason"] = "Edit endpoints lack exact observed canonical sample coordinates"
        return result
    a, b = map(int, bounds)
    if not (0 <= a - 5120 < b - 5120 < b <= mapping["samples"]):
        result["unavailable_reason"] = "Earlier audio samples or complete output interval unavailable"
        return result
    result.update(
        status="ready",
        unavailable_reason=None,
        output_samples_half_open=[a, b],
        source_samples_half_open=[a - 5120, b - 5120],
    )
    return result


def case_specs(row, frames, geometry, audio):
    """All11 variants retained even if individual construction support is unavailable."""
    result = []
    for name in CASES:
        neural = name.startswith(("musetalk_", "liveportrait_"))
        branch = next((b for b in ("musetalk", "liveportrait") if b in name), None)
        reason = None if row["status"] == reviewed.AVAILABLE else row["status"]
        if branch is not None and geometry["status"] != "ready":
            reason = geometry["unavailable_reason"]
        if neural and any(f["status"] != "ready" for f in frames):
            reason = "A required source/driver/conditioning context is unassessable"
        if name == "audio_delay_320ms" and audio["status"] != "ready":
            reason = audio["unavailable_reason"]
        if name == "genuine_frame_retime" and not all(
            f["source_picture"] is not None
            and f["earlier_picture"] is not None
            and exact(f["source_time_seconds"]) - exact(f["earlier_time_seconds"]) == OFFSET
            for f in frames
        ):
            reason = "Missing actual copied source/earlier native pictures or exact320ms correspondence"
        operation = (
            "identity"
            if name == "original"
            else "neural_redraw"
            if neural
            else "branch_resample"
            if name.startswith("resample_")
            else "branch_resample_then_brightness"
            if name.startswith("brightness_")
            else "copy_genuine_earlier_pictures"
            if name == "genuine_frame_retime"
            else "copy_earlier_audio_samples"
        )
        result.append(
            {
                "id": name,
                "status": "planned" if reason is None else "unavailable",
                "unavailable_reason": reason,
                "format": "native_lossless_rgb_pcm",
                "operation": operation,
                "generator_branch": branch,
                "driver": "earlier_320ms"
                if name.endswith("earlier_320ms")
                else "source_time"
                if neural
                else None,
                "requested_offset_seconds": "-8/25"
                if name.endswith("earlier_320ms") or name in {"genuine_frame_retime", "audio_delay_320ms"}
                else "0",
                "planned_neural_provenance": neural,
                "neural_provenance_verified": False,
                "achieved_synchronization_qualified": False,
                "achieved_motion_assessed": False,
                "modified_channels": []
                if name == "original"
                else ["audio"]
                if name == "audio_delay_320ms"
                else ["video"],
                "edit_frames_half_open": None if name == "original" else list(EDIT),
                "output_picture_indices": []
                if name in {"original", "audio_delay_320ms"}
                else list(range(*EDIT)),
                "driver_picture_indices": list(range(19, 39))
                if name in {"liveportrait_earlier_320ms", "genuine_frame_retime"}
                else list(range(*EDIT))
                if name == "liveportrait_source_time"
                else None,
                "waveform_id": "delayed_audio_control" if name == "audio_delay_320ms" else "original_pcm",
                "branch_geometry_and_alpha_required": branch is not None,
                "brightness_channel_multiplier": "11/10" if name.startswith("brightness_") else None,
                "opacity": None
                if name in {"original", "audio_delay_320ms"}
                else {"first_and_last": "1" if name == "genuine_frame_retime" else "1/2", "interior": "1"},
                "ground_truth_scope": "Actual generated pixels/changed support must be verified later; requested offset is not achieved lip synchronization."
                if neural
                else "Ordinary non-neural control. A timing/edit flag may be correct; calling it AI generation is false attribution.",
            }
        )
    return result


def source_plan(row, role, component_index, ordinal):
    require(
        role in {"pilot", "reserve"} and type(ordinal) is int and 0 <= ordinal < 32,
        "Exact planned role/ordinal required",
    )
    require(row["planned"]["ordinal"] == ordinal, "Original fixed source row differs")
    frames = [frame_support(row, i) for i in range(*EDIT)]
    geometry, audio = geometry_requests(row), audio_delay_support(row)
    cases = case_specs(row, frames, geometry, audio)
    available = all(c["status"] == "planned" for c in cases)
    return {
        "ordinal": ordinal,
        "source_slot": row["planned"]["source_ordinal"],
        "within_source_ordinal": row["planned"]["within_source_ordinal"],
        "component_index": component_index,
        "role": role,
        "input": copy.deepcopy(row),
        "availability": row["status"],
        "plan_status": "ready" if available else "unavailable",
        "unavailable_reason": None
        if available
        else "At least one planned condition has unavailable input support; no substitution",
        "geometry": geometry,
        "frames": frames,
        "audio_delay": audio,
        "encoder_inputs": [
            {
                "id": "original_pcm",
                "operation": "identity",
                "input_reference": "input.pcm_file",
                "status": "planned" if row["status"] == reviewed.AVAILABLE else "unavailable",
                "unavailable_reason": None if row["status"] == reviewed.AVAILABLE else row["status"],
                "whole_input_required": True,
                "mapping_reference": "input.pcm_mapping",
            },
            {
                "id": "delayed_audio_control",
                "operation": "copy_earlier_samples_inside_edit",
                "status": "planned" if audio["status"] == "ready" else "unavailable",
                "unavailable_reason": audio["unavailable_reason"],
                "whole_input_required": True,
                "support_reference": "audio_delay",
                "reuse_original_hidden_states_for_changed_audio": False,
            },
        ],
        "cases": cases,
    }


def build_plan(rows, observations, links, dependencies):
    require(
        len(rows) == 32 and [r["planned"]["ordinal"] for r in rows] == list(range(32)),
        "All original32 native rows required",
    )
    require(
        all(
            o["source_slot"] == r["planned"]["source_ordinal"]
            and o["status"] == r["status"]
            and o["source_sha256"] == r["source_sha256"]
            for o, r in zip(observations, rows, strict=True)
        ),
        "Native rows differ from actual independent input review",
    )
    roles, components = partition(observations, links)
    sources = [source_plan(r, roles[i], components[i], i) for i, r in enumerate(rows)]
    return {
        "schema": "chipmunk.grid-native-transfer-plan/143",
        "sources": sources,
        "case_order": list(CASES),
        "edit_frames_half_open": list(EDIT),
        "driver_offset_seconds": "-8/25",
        "earlier_native_frames": 8,
        "nominal_seed_source_slots": list(range(8)),
        "nominal_pilot_cases": 176,
        "component_policy": "Every known exact-content/source-slot component touching slots0..7 is pilot; all others reserve. No quality-based replacements.",
        "dependencies": dependencies,
        "detector_heads": list(ARMS),
        "research_settings": [0, 1, 2],
        "primary_research_setting": 2,
        "new_fitting_or_calibration_permitted": False,
        "baseline_required": [
            "current_complete_serving_video_pipeline",
            "current_appearance_and_fallback",
            "current_common_input_correlation",
        ],
        "interpretation": "Source-time or earlier-conditioned redraw requests, not verified achieved synchronization. Study probabilities are not SaaS authenticity guarantees.",
    }


def bind_metadata(path, bindings):
    path = Path(path)
    if path.is_relative_to(ROOT):
        name = str(path.relative_to(ROOT))
    else:
        require(
            path.is_relative_to(ARCHIVE) and path.suffix in {".json", ".jsonl"},
            "Only private evaluation metadata may be bound",
        )
        name = "archive-metadata/" + str(path.relative_to(ARCHIVE))
    bindings[name] = sha(path)
    return read(path) if path.suffix == ".json" else None


def frozen_context(folder, public, bindings):
    frozen = bind_metadata(folder / "freeze.json", bindings)
    require(frozen == bind_metadata(public, bindings), "Actual public/private frozen context differs")
    for name, digest in frozen["code_sha256"].items():
        require(sha(ROOT / name) == digest, "Qualified executed code changed")
    return frozen


def neural_receipt(folder, freeze, caller, public, expected_calls, bindings):
    result = bind_metadata(folder / "output/result.json", bindings)
    profile = bind_metadata(folder / "profile.json", bindings)
    bind_metadata(folder / "resources.jsonl", bindings)
    wrapper = bind_metadata(public, bindings)
    require(
        result["status"] == "complete"
        and result["freeze_sha256"] == sha(freeze)
        and result["new_neural_calls"] == profile["new_neural_calls"] == expected_calls
        and result["defaults_changed"] is False
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["error_types"] == []
        and profile["resource_limit_observations"] == []
        and profile["command"] == "authored"
        and profile["batch"] is None
        and profile["worker_receipt_sha256"] == sha(folder / "output/result.json")
        and profile["resource_journal_sha256"] == sha(folder / "resources.jsonl")
        and profile["runtime_sha256"] == sha(ROOT / "backend/scripts" / caller)
        and wrapper["measurement"] == result
        and wrapper["profile"] == profile
        and wrapper["result_sha256"] == sha(folder / "output/result.json")
        and wrapper["profile_sha256"] == sha(folder / "profile.json"),
        "Actual successful published renderer qualification required",
    )
    return result


def renderer_bindings(bindings):
    manifest_path = PANEL / "mouth-v7-generator-assets.json"
    manifest = bind_metadata(manifest_path, bindings)
    mt_path = BASE / "native-mouth-renderer-v118"
    mt_freeze = frozen_context(mt_path, PANEL / "native-mouth-renderer-freeze-v118.json", bindings)
    acquired = bind_metadata(BASE / "lipsync-v7/assets/verified-assets.json", bindings)
    require(
        len(manifest["assets"]) == 7
        and all(a["license"] in {"mit", "apache-2.0"} for a in manifest["assets"])
        and mt_freeze["generator_manifest_sha256"] == acquired["manifest_sha256"] == sha(manifest_path)
        and mt_freeze["generator_assets"] == acquired["assets"]
        and acquired["restricted_components_loaded"] == 0,
        "Bound permitted MuseTalk manifests/actual acquisition receipt required",
    )
    mt = neural_receipt(
        mt_path / "authored",
        mt_path / "freeze.json",
        "native_mouth_runtime_v118.py",
        PANEL / "native-mouth-renderer-result-v118.json",
        17,
        bindings,
    )
    require(
        mt["renderer_response_qualified"] is True
        and mt["seeded_repeat_bitwise"] is True
        and mt["achieved_mouth_synchronization_qualified"] is False
        and mt["render_calls"] == 4
        and mt["native_module_forward_counts"]
        == {"vae_encode": 8, "vae_decode": 4, "unet": 4, "speech_encoder": 0},
        "Exact qualified generator response scope required",
    )
    bind_metadata(mt_path / "authored/output/private-render-records.json", bindings)
    require(
        mt["private_render_records_sha256"] == sha(mt_path / "authored/output/private-render-records.json"),
        "MuseTalk qualification journal differs",
    )
    lp_path = BASE / "portrait-expression-qualifier-v140"
    lp_freeze = frozen_context(lp_path, PANEL / "portrait-expression-qualifier-freeze-v140.json", bindings)
    catalogue_path = BASE / "lipsync-v4/permissions/liveportrait-verified-assets.json"
    catalogue = bind_metadata(catalogue_path, bindings)
    license_path = BASE / "lipsync-v4/permissions/liveportrait-source-LICENSE.txt"
    require(
        catalogue["publisher_checkpoint_license"].lower() == "mit"
        and catalogue["restricted_face_models_downloaded"] is False
        and len(catalogue["weights"]) == 5
        and sha(catalogue_path) == lp_freeze["asset_catalogue_sha256"]
        and sha(license_path) == lp_freeze["source_license_sha256"],
        "Retained permitted human animation catalogue/license required",
    )
    bindings[str(license_path.relative_to(ROOT))] = sha(license_path)
    lp = neural_receipt(
        lp_path / "authored",
        lp_path / "freeze.json",
        "portrait_expression_qualifier_v140.py",
        PANEL / "portrait-expression-qualifier-result-v140.json",
        15,
        bindings,
    )
    expected = {
        "appearance_feature_extractor": 3,
        "motion_extractor": 6,
        "warping_module": 3,
        "spade_generator": 3,
        "lip": 0,
    }
    require(
        lp["renderer_response_qualified"] is True
        and lp["renderer_repeat_bitwise"] is True
        and lp["native_coordinate_containment_qualified"] is True
        and lp["strict_exact_checkpoint_loads"] == 5
        and lp["new_face_task_inputs"] == 0
        and lp["renderer_inputs"] == 3
        and lp["top_level_module_forwards_started"] == lp["top_level_module_forwards_completed"] == expected
        and lp["achieved_motion_assessed"] is False
        and lp["achieved_phonetic_alignment_qualified"] is False,
        "Actual full-expression response/containment, not achieved sync, required",
    )
    for key, name in (
        ("private_render_records_sha256", "private-render-records.json"),
        ("private_input_files_sha256", "private-input-files.json"),
        ("private_render_journal_sha256", "render-records.jsonl"),
        ("private_forward_progress_sha256", "forward-progress.jsonl"),
    ):
        path = lp_path / "authored/output" / name
        bind_metadata(path, bindings)
        require(lp[key] == sha(path), "Full-expression qualification journal changed")
    return {
        "musetalk_manifest_sha256": sha(manifest_path),
        "liveportrait_catalogue_sha256": sha(catalogue_path),
        "musetalk_qualification_sha256": sha(mt_path / "authored/output/result.json"),
        "liveportrait_qualification_sha256": sha(lp_path / "authored/output/result.json"),
        "scope": "Metadata/actual qualification binding only. Future guarded geometry/generator workers must verify actual assets before loading. Old qualification sources do not qualify fresh GRID geometry.",
    }


def detector_bindings(bindings):
    fitted, calibrated = BASE / "native-mouth-heads-v125", BASE / "native-mouth-calibration-v126"
    fitted_freeze = frozen_context(fitted, PANEL / "native-mouth-head-fit-freeze-v125.json", bindings)
    cal_freeze = frozen_context(calibrated, PANEL / "native-mouth-calibration-freeze-v126.json", bindings)
    fit = reviewed.stage(
        fitted / "bind-00", sha(fitted / "freeze.json"), "fit_native_mouth_heads_v125.py", 0, bindings
    )
    cal = reviewed.stage(
        calibrated / "bind-00", sha(calibrated / "freeze.json"), "calibrate_native_mouth_v126.py", 0, bindings
    )
    public_fit = bind_metadata(PANEL / "native-mouth-head-fit-result-v125.json", bindings)
    public_cal = bind_metadata(PANEL / "native-mouth-calibration-result-v126.json", bindings)
    heads_path = ARCHIVE / "native-mouth-heads-v125/reference-and-heads.json"
    cal_path = ARCHIVE / "native-mouth-calibration-v126/calibrations-and-research-points.json"
    heads, maps = bind_metadata(heads_path, bindings), bind_metadata(cal_path, bindings)
    require(
        set(heads) == {"reference", "heads"}
        and set(heads["heads"]) == set(ARMS)
        and fit["raw_heads_fitted"] == 3
        and fitted_freeze["arms"] == cal_freeze["arms"] == list(ARMS)
        and fit["private_learned_artifact_sha256"]
        == public_fit["private_learned_artifact_sha256"]
        == sha(heads_path)
        and cal_freeze["learned_reference_heads_sha256"] == sha(heads_path)
        and cal["private_calibrations_sha256"] == public_cal["private_calibrations_sha256"] == sha(cal_path)
        and cal["calibrations_fitted"] == 6
        and set(maps["arms"]) == set(ARMS)
        and cal_freeze["research_group_budgets"] == [0, 1, 2]
        and cal_freeze["primary_research_group_budget"] == 2,
        "Actual fixed three-head/six-calibration artifacts and settings required",
    )
    for arm in maps["arms"].values():
        require(
            set(arm) == {"point", "clip", "research_points"}
            and set(arm["research_points"]) == {"0", "1", "2"},
            "Exact own point/clip calibrations and settings required",
        )
        for key in ("point", "clip"):
            require(
                arm[key]["method"] == "positive-slope-balanced-study-logit/126"
                and arm[key]["deployment_eligible"] is False,
                "Study probability must remain outside customer authenticity confidence",
            )
        for setting, point in arm["research_points"].items():
            require(
                type(point["raw_margin"]) in (int, float)
                and math.isfinite(point["raw_margin"])
                and point["comparison"] == "strict_greater_than"
                and point["calibration_group_budget"] == int(setting),
                "Unchanged finite frozen research point required",
            )
    return {
        "heads_sha256": sha(heads_path),
        "calibrations_and_settings_sha256": sha(cal_path),
        "arms": list(ARMS),
        "settings": [0, 1, 2],
        "primary_setting": 2,
        "refitting_permitted": False,
    }


def dependencies():
    # v142 verify is metadata-only: it never re-opens saved media.
    reviewed.verify()
    bindings = {}
    frozen = frozen_context(reviewed.TRIAL, reviewed.PUBLIC, bindings)
    result = reviewed.stage(
        reviewed.TRIAL / "bind-00", sha(reviewed.PUBLIC), "review_grid_native_inputs_v142.py", 0, bindings
    )
    wrapper = bind_metadata(PANEL / "grid-native-input-review-result-v142.json", bindings)
    require(
        result["schema"] == "chipmunk.grid-native-input-review-result/142"
        and result["planned_clips"] == result["reviewed_clips"] == 32
        and result["mpg_decoder_calls"] == 0,
        "Actual complete independent saved-bank review required",
    )
    require(
        wrapper["measurement"] == result
        and wrapper["profile"] == read(reviewed.TRIAL / "bind-00/profile.json")
        and wrapper["result_sha256"] == sha(reviewed.TRIAL / "bind-00/output/result.json")
        and wrapper["profile_sha256"] == sha(reviewed.TRIAL / "bind-00/profile.json"),
        "Actual input review must match its published execution wrapper",
    )
    review_path = reviewed.TRIAL / "bind-00/output/reviewed-inputs.jsonl"
    links_path = reviewed.TRIAL / "bind-00/output/private-components.json"
    bind_metadata(review_path, bindings)
    links = bind_metadata(links_path, bindings)
    require(
        result["private_reviewed_inputs_sha256"] == sha(review_path)
        and result["private_components_sha256"] == sha(links_path),
        "Actual review/component journals required",
    )
    observations = reviewed.lines(review_path)
    _upstream, rows = reviewed.metadata()
    require(
        frozen["completed_metadata_sha256"] == _upstream
        and result["availability"] == dict(Counter(r["status"] for r in rows)),
        "Unchanged whole32 native metadata/availability commitment required",
    )
    partition(observations, links)
    refs = {
        "renderers": renderer_bindings(bindings),
        "detectors": detector_bindings(bindings),
        "input_review_result_sha256": sha(reviewed.TRIAL / "bind-00/output/result.json"),
    }
    return bindings, rows, observations, links, refs


def counts(plan):
    pilot = [s for s in plan["sources"] if s["role"] == "pilot"]
    return {
        "source_slots": 16,
        "original_clips": len(plan["sources"]),
        "pilot_clips": len(pilot),
        "reserve_clips": len(plan["sources"]) - len(pilot),
        "pilot_source_slots": len({s["source_slot"] for s in pilot}),
        "pilot_cases": len(pilot) * 11,
        "planned_cases_all_roles": len(plan["sources"]) * 11,
        "pilot_ready_cases": sum(c["status"] == "planned" for s in pilot for c in s["cases"]),
        "pilot_unavailable_cases": sum(c["status"] == "unavailable" for s in pilot for c in s["cases"]),
        "pilot_neural_cases": len(pilot) * 4,
        "pilot_original_and_ordinary_cases": len(pilot) * 7,
        "pilot_unique_geometry_pictures": sum(len(s["geometry"]["pictures"]) for s in pilot),
        "pilot_availability": dict(Counter(s["availability"] for s in pilot)),
    }


def identity():
    bindings, rows, observations, links, refs = dependencies()
    roles, _components = partition(observations, links)
    return {
        "schema": "chipmunk.grid-transfer-plan-freeze/143",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "completed_metadata_sha256": bindings,
        "input_clips": len(rows),
        "case_order": list(CASES),
        "nominal_pilot_clips": 16,
        "nominal_pilot_cases": 176,
        "actual_pilot_clips": sum(v == "pilot" for v in roles.values()),
        "actual_pilot_cases": 11 * sum(v == "pilot" for v in roles.values()),
        "dependencies": refs,
        "scope": "Metadata-only fixed native middle-placement plan with source/exact-content component closure. No media reads, model, construction, achieved sync, accuracy or serving promotion.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print(
        json.dumps(
            {"status": "frozen", "actual_pilot_cases": body["actual_pilot_cases"], "new_neural_calls": 0}
        ),
        flush=True,
    )


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged published transfer planning context required",
    )


def plan_worker(output):
    _bindings, rows, observations, links, refs = dependencies()
    plan = build_plan(rows, observations, links, refs)
    write(output / "private-plan.json", plan)
    result = {
        "schema": "chipmunk.grid-transfer-plan-result/143",
        "status": "complete",
        **counts(plan),
        "private_plan_sha256": sha(output / "private-plan.json"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": 0,
        "new_media_reads": 0,
        "defaults_changed": False,
        "scope": "Only metadata/source-clock planning. Reserve remains unscored; unavailable retained. No changed picture, achieved motion/sync, detector response or customer confidence result.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def load_completed():
    """Metadata-only API for future geometry/encoder/generator stages."""
    verify()
    result = reviewed.stage(TRIAL / "bind-00", sha(PUBLIC), "grid_transfer_plan_v143.py", 0, {})
    require(
        result["schema"] == "chipmunk.grid-transfer-plan-result/143"
        and result["new_media_reads"] == 0
        and result["private_plan_sha256"] == sha(TRIAL / "bind-00/output/private-plan.json"),
        "Actual completed transfer plan required",
    )
    plan = read(TRIAL / "bind-00/output/private-plan.json")
    _bindings, rows, observations, links, refs = dependencies()
    require(
        plan == build_plan(rows, observations, links, refs)
        and all(result[k] == v for k, v in counts(plan).items()),
        "Actual private plan/source/component/spec correspondence differs",
    )
    return plan


def routed_popen(argv, **kwargs):
    expected = TRIAL / "bind-00/output"
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"]
        and argv[5:] == [str(expected), "0"],
        "Only exclusive data-only transfer planner may run",
    )
    return subprocess.Popen([*argv[:3], "scripts.grid_transfer_plan_v143", "plan", str(expected)], **kwargs)


def run():
    from scripts import video_appearance_runtime_v108 as runtime
    from scripts.video_appearance_corrected_v112 import scoped

    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda _c, _b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "plan"))
    parser.add_argument("output", nargs="?")
    args = parser.parse_args()
    if args.command == "freeze":
        require(args.output is None, "No extra freeze arguments")
        freeze()
    elif args.command == "run":
        require(args.output is None, "No extra stage arguments")
        run()
    else:
        verify()
        require(
            args.output is not None and Path(args.output).resolve() == (TRIAL / "bind-00/output").resolve(),
            "Exact exclusive plan output required",
        )
        target = Path(args.output)
        target.mkdir(exist_ok=False)
        plan_worker(target)
