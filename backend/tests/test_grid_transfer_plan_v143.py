import copy
from fractions import Fraction

import pytest

from scripts import grid_transfer_plan_v143 as plan


def native_row(ordinal=0, origin=Fraction(0), audio_offset=Fraction(0)):
    start = origin * 90000
    assert start.denominator == 1
    pts = [int(start) + 3600 * i for i in range(75)]
    audio_origin = origin + audio_offset
    assert (audio_origin * 16000).denominator == 1
    return {
        "planned": {"ordinal": ordinal, "source_ordinal": ordinal // 2, "within_source_ordinal": ordinal % 2},
        "status": plan.reviewed.AVAILABLE,
        "source_sha256": f"authored-source-{ordinal}",
        "source_license": "CC-BY-4.0",
        "study_role": "external_transfer",
        "native_video_table": {
            "pts": pts,
            "time_base": "1/90000",
            "width": 360,
            "height": 288,
            "truncated": False,
            "interpolated": [],
        },
        "pictures": [
            {
                "frame_index": i,
                "pts": p,
                "file": f"/authored/{ordinal}/frame-{i:04d}.png",
                "file_sha256": f"authored-png-{ordinal}-{i}",
                "rgb_sha256": f"authored-rgb-{ordinal}-{i}",
            }
            for i, p in enumerate(pts)
        ],
        "pcm_file": f"/authored/{ordinal}/pcm.npy",
        "wav_file": f"/authored/{ordinal}/pcm.wav",
        "pcm_mapping": {
            "samples": 47648,
            "sample_rate": 16000,
            "timeline_time_base": "1/16000",
            "timeline": [{"start_sample": 0, "end_sample": 47648, "source_pts": int(audio_origin * 16000)}],
        },
        "native_audio_clock": {"time_base": "1/90000", "sample_rate": 44100},
        "audio_origin_seconds": str(audio_origin),
        "video_origin_seconds": str(origin),
    }


def cohort():
    rows = [native_row(i) for i in range(32)]
    observations = [
        {
            "ordinal": i,
            "source_slot": i // 2,
            "within_source_ordinal": i % 2,
            "status": r["status"],
            "source_sha256": r["source_sha256"],
            "video_content_sha256": f"video-{i}",
            "pcm_content_sha256": f"pcm-{i}",
        }
        for i, r in enumerate(rows)
    ]
    return rows, observations, plan.reviewed.components(observations)


def test_complete_middle_plan_preserves_all32_and_has176_pilot_cases_without_media_io():
    rows, observations, links = cohort()
    actual = plan.build_plan(rows, observations, links, {})
    counts = plan.counts(actual)
    assert counts["pilot_clips"] == counts["reserve_clips"] == 16
    assert counts["pilot_cases"] == counts["pilot_ready_cases"] == 176
    assert counts["pilot_neural_cases"] == 64
    assert counts["pilot_original_and_ordinary_cases"] == 112
    assert counts["planned_cases_all_roles"] == 352
    assert counts["pilot_unique_geometry_pictures"] == 448
    assert [s["input"] for s in actual["sources"]] == rows
    assert all([c["id"] for c in source["cases"]] == list(plan.CASES) for source in actual["sources"])
    assert actual["new_fitting_or_calibration_permitted"] is False


def test_exact_alias_component_closure_expands_pilot_without_quality_substitution():
    rows, observations, _links = cohort()
    observations[16]["pcm_content_sha256"] = observations[0]["pcm_content_sha256"]
    links = plan.reviewed.components(observations)
    actual = plan.build_plan(rows, observations, links, {})
    assert plan.counts(actual)["pilot_clips"] == 18
    assert plan.counts(actual)["pilot_cases"] == 198
    assert plan.counts(actual)["reserve_clips"] == 14
    assert actual["sources"][16]["role"] == actual["sources"][17]["role"] == "pilot"
    assert len({actual["sources"][i]["role"] for i in links["components"][0]}) == 1


def test_component_partition_rejects_stale_links_and_missing_inputs():
    rows, observations, links = cohort()
    with pytest.raises(ValueError):
        plan.partition(observations[:-1], links)
    changed = copy.deepcopy(links)
    changed["components"][0].pop()
    with pytest.raises(ValueError):
        plan.partition(observations, changed)
    with pytest.raises(ValueError):
        plan.build_plan(rows[:-1], observations, links, {})


def test_v109_full_waveform_routes_keep_output_index_and_actual_nonzero_origins():
    row = native_row(origin=Fraction(27, 50))
    current = plan.frame_support(row, 27)
    assert current["status"] == "ready"
    a, b = current["source_time_conditioning"], current["earlier_conditioning"]
    assert a["video_frame_index"] == b["video_frame_index"] == 27
    assert a["source_audio_origin_seconds"] == a["source_video_origin_seconds"] == "27/50"
    assert a["encoder_anchor"] == 54 and b["encoder_anchor"] == 38
    assert a["tick_indices"] == list(range(50, 60))
    assert b["tick_indices"] == list(range(34, 44))
    assert a["observed_mono_samples"] == b["observed_mono_samples"] == 47648
    assert current["earlier_picture"]["frame_index"] == 19
    assert Fraction(current["source_time_seconds"]) - Fraction(current["earlier_time_seconds"]) == Fraction(
        8, 25
    )
    assert "whole padded-input" in a["hidden_state_dependency"]


def test_actual_audio_video_offset_changes_conditioning_and_control_sample_positions():
    row = native_row(audio_offset=Fraction(1, 20))
    support = plan.frame_support(row, 27)
    assert support["source_time_conditioning"]["encoder_anchor"] == 51
    assert support["earlier_conditioning"]["encoder_anchor"] == 35
    assert plan.audio_delay_support(row)["output_samples_half_open"] == [16480, 29280]


def test_24hz_or_missing_driver_is_unassessable_not_replaced_or_padded():
    row = native_row()
    row["native_video_table"]["pts"] = [i * 3750 for i in range(75)]
    for i, picture in enumerate(row["pictures"]):
        picture["pts"] = i * 3750
    assert plan.frame_support(row, 27)["status"] == "unavailable"
    assert plan.geometry_requests(row)["status"] == "unavailable"
    row = native_row()
    row["pictures"] = [p for p in row["pictures"] if p["frame_index"] != 19]
    assert plan.frame_support(row, 27)["source_picture"] is None
    assert plan.geometry_requests(row)["status"] == "unavailable"


def test_conditioning_window_requires_observed_ten_slots_instead_of_zero_substitution():
    row = native_row()
    row["pcm_mapping"]["samples"] = 12000
    frame = plan.frame_support(row, 27)
    assert frame["status"] == "unavailable"
    assert frame["source_time_conditioning"] is not None
    assert frame["source_time_conditioning"]["source_sample_status"] == "unavailable"


def test_geometry_api_has28_unique_pictures_and_stateless_repeat27():
    row = native_row()
    geometry = plan.geometry_requests(row)
    assert geometry["status"] == "ready"
    assert (
        geometry["frame_indices"] == [p["frame_index"] for p in geometry["pictures"]] == list(range(19, 47))
    )
    assert geometry["repeat_frame_index"] == 27
    assert geometry["planned_unique_face_task_inputs"] + geometry["planned_repeat_face_task_inputs"] == 29


def test_audio_delay_requires_full_changed_waveform_and_preserves_all_other_samples():
    source = plan.source_plan(native_row(), "pilot", 0, 0)
    delay = source["audio_delay"]
    assert delay["output_samples_half_open"] == [17280, 30080]
    assert delay["source_samples_half_open"] == [12160, 24960]
    assert delay["whole_input_reencoding_required"] is True
    assert delay["preserve_length_origins_and_outside_samples"] is True
    assert delay["padding_or_wraparound"] is False
    assert source["encoder_inputs"][1]["reuse_original_hidden_states_for_changed_audio"] is False
    cases = {c["id"]: c for c in source["cases"]}
    assert cases["audio_delay_320ms"]["modified_channels"] == ["audio"]
    assert cases["audio_delay_320ms"]["output_picture_indices"] == []
    assert all(c["waveform_id"] == "original_pcm" for name, c in cases.items() if name != "audio_delay_320ms")


def test_unavailable_clock_and_short_audio_do_not_drop_or_clear_source():
    rows, observations, _links = cohort()
    rows[0] = {**rows[0], "status": "unavailable_video_probe"}
    observations[0]["status"] = rows[0]["status"]
    actual = plan.build_plan(rows, observations, plan.reviewed.components(observations), {})
    assert len(actual["sources"]) == 32 and len(actual["sources"][0]["cases"]) == 11
    assert actual["sources"][0]["availability"] == "unavailable_video_probe"
    assert actual["sources"][0]["plan_status"] == "unavailable"
    assert all(c["status"] == "unavailable" for c in actual["sources"][0]["cases"])
    assert plan.counts(actual)["pilot_unavailable_cases"] == 11


def test_only_branch_cases_require_facegeometry_cuts_and_audio_use_their_own_support():
    row = native_row()
    frames = [plan.frame_support(row, i) for i in range(*plan.EDIT)]
    geometry = {"status": "unavailable", "unavailable_reason": "Geometry has not qualified"}
    cases = {c["id"]: c for c in plan.case_specs(row, frames, geometry, plan.audio_delay_support(row))}
    assert (
        cases["original"]["status"]
        == cases["genuine_frame_retime"]["status"]
        == cases["audio_delay_320ms"]["status"]
        == "planned"
    )
    assert all(c["status"] == "unavailable" for c in cases.values() if c["generator_branch"])
    frames[0]["earlier_picture"] = None
    cases = {c["id"]: c for c in plan.case_specs(row, frames, geometry, plan.audio_delay_support(row))}
    assert cases["genuine_frame_retime"]["status"] == "unavailable"
    assert cases["audio_delay_320ms"]["status"] == "planned"


def test_controls_match_both_generator_branches_and_keep_provenance_separate_from_sync():
    cases = {c["id"]: c for c in plan.source_plan(native_row(), "pilot", 0, 0)["cases"]}
    for branch in ("musetalk", "liveportrait"):
        assert (
            cases[f"resample_{branch}"]["generator_branch"]
            == cases[f"brightness_{branch}"]["generator_branch"]
            == branch
        )
        assert cases[f"brightness_{branch}"]["brightness_channel_multiplier"] == "11/10"
        for driver in ("source_time", "earlier_320ms"):
            row = cases[f"{branch}_{driver}"]
            assert row["planned_neural_provenance"] is True
            assert (
                row["neural_provenance_verified"]
                is row["achieved_synchronization_qualified"]
                is row["achieved_motion_assessed"]
                is False
            )
    assert cases["genuine_frame_retime"]["opacity"] == {"first_and_last": "1", "interior": "1"}
    assert cases["genuine_frame_retime"]["driver_picture_indices"] == list(range(19, 39))


@pytest.mark.parametrize("value", [0.0, True])
def test_clock_helpers_reject_floating_and_boolean_clock_values(value):
    with pytest.raises(ValueError):
        plan.exact(value)


def test_only_exact_new_data_planning_worker_is_dispatched(monkeypatch):
    calls = []
    monkeypatch.setattr(plan.subprocess, "Popen", lambda argv, **kw: calls.append(argv))
    argv = [
        "python",
        "-B",
        "-m",
        "scripts.video_appearance_native_v108",
        "bind",
        str(plan.TRIAL / "bind-00/output"),
        "0",
    ]
    plan.routed_popen(argv)
    assert calls[0][3:] == ["scripts.grid_transfer_plan_v143", "plan", argv[5]]
    with pytest.raises(ValueError):
        plan.routed_popen([*argv[:-1], "1"])
    with pytest.raises(ValueError):
        plan.routed_popen([*argv[:4], "extract", *argv[5:]])
