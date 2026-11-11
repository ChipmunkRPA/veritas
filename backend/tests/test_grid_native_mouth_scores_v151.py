import copy
from fractions import Fraction

import numpy as np
import pytest

from scripts import grid_native_mouth_scores_v151 as score


def authored_case(name, *, ready=True, pixels=5):
    neural = name.startswith(("musetalk_", "liveportrait_"))
    bounds = None if name == "original" else [27, 47]
    spec = {"id": name, "planned_neural_provenance": neural, "edit_frames_half_open": bounds}
    case = {
        "condition": name,
        "status": "ready" if ready else "unavailable",
        "edit_frames_half_open": bounds,
        "planned_neural_provenance": neural,
        "neural_provenance_verified": ready and neural,
        "observed_neural_pixel_change": ready and neural and pixels > 0,
        "changed_native_pixels": pixels,
        "source_video_pts": [9000 + i * 3600 for i in range(75)],
        "source_video_time_base": "1/90000",
        "source_time_bounds_seconds": ["59/50", "99/50"],
    }
    return spec, case


@pytest.mark.parametrize(
    "name,ready,pixels,category",
    [
        ("liveportrait_earlier_320ms", True, 5, "observable_neural"),
        ("musetalk_source_time", True, 0, "zero_effect_neural"),
        ("audio_delay_320ms", True, 0, "non_neural"),
        ("musetalk_earlier_320ms", False, 0, "unavailable"),
    ],
)
def test_actual_provenance_labels_and_native_times_do_not_depend_on_ai_prefix(name, ready, pixels, category):
    truth = score.truth_for(*authored_case(name, ready=ready, pixels=pixels))
    assert truth["category"] == category
    assert truth["observable_neural_interval_seconds"] == (
        ["59/50", "99/50"] if category == "observable_neural" else None
    )
    assert truth["requested_edit_interval_seconds"] == (["59/50", "99/50"] if ready else None)


def test_neural_pixel_presence_cannot_disagree_with_actual_construction_counts():
    spec, case = authored_case("musetalk_source_time")
    case["observed_neural_pixel_change"] = False
    with pytest.raises(ValueError, match="construction counts"):
        score.truth_for(spec, case)


def test_threshold_equality_does_not_flag():
    point = {"comparison": "strict_greater_than", "raw_margin": 0.5}
    windows = [{"response": x} for x in (0.5, np.nextafter(0.5, 1.0), 0.49)]
    assert score.selected_windows(windows, point) == [windows[1]]


def test_empty_change_mask_keeps_null_coverage_and_native_roi_iou():
    mask = np.zeros((5, 5), np.uint8)
    assert score.roi_metrics(mask, [0, 0, 2, 2], 0) == {"pixel_iou": 0, "changed_pixel_coverage": None}
    mask[0, 0] = mask[4, 4] = 255
    assert score.roi_metrics(mask, [0, 0, 2, 2], 2) == {"pixel_iou": 1 / 5, "changed_pixel_coverage": 1 / 2}
    mask[4, 4] = 1
    with pytest.raises(ValueError, match="binary"):
        score.roi_metrics(mask, [0, 0, 2, 2], 2)


def authored_learned():
    def projection(dim):
        return {"center": np.zeros(dim).tolist(), "components": np.eye(16, dim).tolist(), "scale": [1.0] * 16}

    reference = {
        "method": "native-mouth-reference-interaction/122",
        "rank": 16,
        "ridge": 0.1,
        "response_kind": "features_only_not_probability",
        "visual_projection": projection(2304),
        "audio_projection": projection(3840),
        "audio_to_visual": np.eye(16).tolist(),
    }
    heads = {}
    for arm, dim in zip(score.ARMS, (16, 32, 64), strict=True):
        heads[arm] = {
            "method": "native-mouth-independent-linear/125",
            "response_kind": "raw_signed_margin_not_probability",
            "mean": [0.0] * dim,
            "scale": [1.0] * dim,
            "coef": [1.0] * dim,
            "intercept": 0.0,
        }
    cal = {
        "method": "positive-slope-balanced-study-logit/126",
        "centre": 0.0,
        "scale": 1.0,
        "slope": 1.0,
        "intercept": 0.0,
    }
    calibrated = {
        arm: {
            "point": cal,
            "clip": cal,
            "research_points": {
                setting: {"comparison": "strict_greater_than", "raw_margin": 1.0}
                for setting in score.SETTINGS
            },
        }
        for arm in score.ARMS
    }
    return {"reference": reference, "heads": heads}, calibrated


def authored_feature(tmp_path, name, speech_value):
    vision = np.zeros((8, 768), np.float32)
    vision[:, 0] = np.arange(8, dtype=np.float32)
    audio = np.zeros((8, 3840), np.float64)
    audio[:, 0] = speech_value
    picture_file, speech_file = tmp_path / f"{name}-visual.npy", tmp_path / f"{name}-speech.npy"
    np.save(picture_file, vision, allow_pickle=False)
    np.save(speech_file, audio, allow_pickle=False)
    table = {
        "pts": [100 + i * 40 for i in range(8)],
        "time_base": "1/1000",
        "width": 100,
        "height": 80,
        "coded_width": 100,
        "coded_height": 80,
        "key_frames": [0],
        "pict_types": "IPPPPPPP",
        "interpolated": [],
        "truncated": False,
        "rotation": 0,
    }
    return {
        "status": "extracted",
        "file": str(picture_file),
        "file_sha256": score.sha(picture_file),
        "shape": [8, 768],
        "table": table,
        "frames": [{"frame_index": i, "feature_row": i, "status": "ok"} for i in range(8)],
        "speech": {
            "file": str(speech_file),
            "sha256": score.sha(speech_file),
            "shape": [8, 3840],
            "indices": list(range(8)),
        },
    }


def test_frozen_blind_arithmetic_retains_final_point_and_excludes_final_cell(tmp_path):
    learned, calibrated = authored_learned()
    feature = authored_feature(tmp_path, "authored", 1.5)
    result = score.blind_score(feature, learned, calibrated)
    assert result["eligible_centres"] == [1, 2, 3, 4, 5, 6]
    visual = result["arms"]["visual_only"]
    assert len(visual["point_margins"]) == 6
    assert {w["centre_frame_index"] for w in visual["windows"]} == {2, 3, 4}
    assert all(w["support_frames_half_open"][1] < 8 for w in visual["windows"])
    last = next(w for w in visual["windows"] if w["centre_frame_index"] == 4)
    assert Fraction(last["end_seconds"]) == Fraction(19, 50)


def test_audio_only_case_preserves_visual_bytes_and_allows_audio_head_to_change(tmp_path):
    learned, calibrated = authored_learned()
    first = score.blind_score(authored_feature(tmp_path, "original", 0.0), learned, calibrated)
    second = score.blind_score(authored_feature(tmp_path, "delayed", 3.0), learned, calibrated)
    first["condition"], second["condition"] = "original", "audio_delay_320ms"
    expected = {"rgb_sha256": ["same"] * 8, "picture_pts_ms": [100 + i * 40 for i in range(8)]}
    cases = [
        {"condition": name, "status": "ready", "expected": expected}
        for name in ("original", "audio_delay_320ms")
    ]
    assert score.visual_invariance(cases, [first, second])["bitwise_equal"] is True
    assert first["arms"]["interaction"]["point_margins"] != second["arms"]["interaction"]["point_margins"]
    second["arms"]["visual_only"]["point_margins"][0] += 1
    assert score.visual_invariance(cases, [first, second])["bitwise_equal"] is False
    second["eligible_centres"].pop()
    assert score.visual_invariance(cases, [first, second])["bitwise_equal"] is None


def test_unavailable_case_has_no_margin_window_or_clip_probability():
    learned, calibrated = authored_learned()
    result = score.blind_score({"status": "unavailable"}, learned, calibrated)
    assert result["eligible_centres"] == []
    assert all(
        part["point_margins"] == part["windows"] == [] and part["study_clip_probability"] is None
        for part in result["arms"].values()
    )


def pair_fixture(tmp_path):
    construction, features = tmp_path / "construction", tmp_path / "features"
    source = {"ordinal": 0, "source_slot": 0, "component_index": 0, "role": "pilot"}
    spec = {"id": "audio_delay_320ms", "waveform_id": "delayed_audio_control"}
    pts = [100 + 40 * i for i in range(75)]
    rgb = [f"authored{i}" for i in range(75)]
    base = {
        "asset_index": 0,
        "component_index": 0,
        "study_role": "pilot",
        "condition": spec["id"],
        "waveform_id": spec["waveform_id"],
    }
    case = {
        **base,
        "source_slot": 0,
        "status": "ready",
        "file": str(construction / "clip-00/00.mkv"),
        "file_sha256": "file",
        "audio_output_clock_mapping": {"pcm_sha256": "delayed"},
        "source_video_time_base": "1/90000",
        "source_video_pts": [p * 90 for p in pts],
        "expected": {"picture_pts_ms": pts, "rgb_sha256": rgb, "constructed_last_picture_duration_ms": 40},
    }
    feature = {
        **base,
        "source_group": 0,
        "input_status": "ready",
        "status": "extracted",
        "constructor_case_journal_sha256": "journal",
        "source_clip_sha256": "file",
        "dtype": "float32",
        "shape": [0, 768],
        "file": str(features / "clip-00/features.npy"),
        "table": {"pts": pts, "time_base": "1/1000"},
        "constructed_last_picture_duration_ms": 40,
        "frames": [{"frame_index": i, "pts": t, "rgb_sha256": rgb[i]} for i, t in enumerate(pts)],
        "speech": {
            "waveform_id": spec["waveform_id"],
            "pcm_sha256": "delayed",
            "dtype": "float64",
            "indices": [],
            "shape": [0, 3840],
            "file": str(features / "clip-00/speech.npy"),
        },
    }
    return source, spec, case, feature, construction, features, "journal"


@pytest.mark.parametrize(
    "damage", [None, "original_state", "wrong_clip", "foreign_array", "clock", "frame_hash", "reserve"]
)
def test_metadata_pair_binds_actual_clip_waveform_clock_and_ownership_without_opening_payload(
    tmp_path, damage
):
    args = pair_fixture(tmp_path)
    source, _spec, case, feature, *_ = args
    if damage is None:
        assert score.validate_pair(*args) is None  # Paths intentionally contain no payload files.
        return
    if damage == "original_state":
        feature["speech"]["pcm_sha256"] = "original"
    elif damage == "wrong_clip":
        feature["source_clip_sha256"] = "foreign"
    elif damage == "foreign_array":
        feature["file"] = str(tmp_path / "features/clip-01/features.npy")
    elif damage == "clock":
        case["source_video_time_base"] = "1/1000"
    elif damage == "frame_hash":
        feature["frames"][1]["rgb_sha256"] = "changed"
    else:
        source["role"] = "reserve"
    with pytest.raises(ValueError):
        score.validate_pair(*args)


def aggregate_fixture():
    sources, rows = [], []
    for ordinal in range(32):
        role = "pilot" if ordinal < 16 else "reserve"
        group = ordinal // 2
        sources.append(
            {
                "ordinal": ordinal,
                "role": role,
                "source_slot": group,
                "component_index": group,
                "cases": [{"id": name} for name in score.planning.CASES],
            }
        )
        if role == "reserve":
            continue
        for name in score.planning.CASES:
            neural = name.startswith(("liveportrait_", "musetalk_"))
            category = "observable_neural" if neural else "non_neural"
            if ordinal == 0 and name == "musetalk_source_time":
                category = "zero_effect_neural"
            elif ordinal == 0 and name == "musetalk_earlier_320ms":
                category = "unavailable"
            outcome = {
                "assessed": category != "unavailable",
                "flagged": category == "zero_effect_neural",
                "localized_iou_0_2": False,
                "temporal_union_iou": 0.0,
                "edit_overlap_seconds": 0.0,
                "outside_edit_flagged_seconds": 0.0,
            }
            outcomes = {
                arm: {setting: copy.deepcopy(outcome) for setting in score.SETTINGS} for arm in score.ARMS
            }
            rows.append(
                {
                    "asset_index": ordinal,
                    "study_role": role,
                    "source_slot": group,
                    "component_index": group,
                    "condition": name,
                    "truth": {"category": category},
                    "outcomes": outcomes,
                    "timing": {"assessed": True, "flagged": name == "audio_delay_320ms"},
                }
            )
    return {"sources": sources}, rows


def test_full_denominator_and_component_pairs_keep_zero_effect_unavailable_and_timing_controls():
    plan, rows = aggregate_fixture()
    body = score.aggregate_records(plan, rows)
    total = body["arms"]["interaction"]["2"]
    assert body["planned_cases"] == 176 and body["known_components"] == 8
    assert total["observable_neural_cases"] == 62 and total["non_neural_cases"] == 112
    assert total["zero_effect_neural_cases"] == total["zero_effect_neural_flags"] == 1
    assert total["unavailable_cases"] == 1 and total["non_neural_ai_flags"] == 0
    assert body["common_input_timing"]["ordinary_edit_timing_flags"] == 16
    assert body["primary_paired_vs_visual"]["interaction"]["paired_binary_cases"] == 174
    assert sum(r["planned_cases"] for r in body["component_paired_aggregates"]) == 176


@pytest.mark.parametrize("damage", ["drop", "order", "reserve", "wrong_component"])
def test_aggregate_rejects_cohort_substitution_or_incomplete_pairs(damage):
    plan, rows = aggregate_fixture()
    if damage == "drop":
        rows.pop()
    elif damage == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif damage == "reserve":
        rows[0]["study_role"] = "reserve"
    else:
        rows[0]["component_index"] = 1
    with pytest.raises(ValueError, match="176 original pilot cases"):
        score.aggregate_records(plan, rows)


def test_exact_data_route_cannot_dispatch_neural_or_wrong_clip_worker(monkeypatch):
    observed = []
    monkeypatch.setattr(score.subprocess, "Popen", lambda argv, **kw: observed.append(argv))
    output = str(score.TRIAL / "clip-02/bind-00/output")
    args = ["python", "-B", "-m", "scripts.video_appearance_native_v108", "bind", output, "0"]
    score.routed_popen(2, args)
    assert observed == [["python", "-B", "-m", "scripts.grid_native_mouth_scores_v151", "score", output, "2"]]
    with pytest.raises(ValueError):
        score.routed_popen(1, args)
    with pytest.raises(ValueError):
        score.routed_popen(2, [*args[:4], "authored", *args[5:]])
