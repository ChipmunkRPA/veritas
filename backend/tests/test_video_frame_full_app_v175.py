from copy import deepcopy
from pathlib import Path

import pytest

from scripts import video_frame_full_app_v175 as study


@pytest.fixture
def paired():
    common = {
        "pipeline_status": "done",
        "analysis_status": "done",
        "result_hash": "result",
        "packet_set_ready": True,
        "input": {
            "clock_exact": True,
            "rgb_exact": True,
            "pcm_exact": True,
            "sampled_frames": 24,
            "faces": 11,
            "face_inputs_sha256": "shared-face-support",
        },
        "non_target_runs": [("vid.faces.yunet", "ok"), ("unchanged", "ok")],
        "condition_profile": {"height": 1080},
        "fusion_mode": "artefact",
        "fuser_sha256": "fuser",
        "target_rows_bound": True,
        "target_raw_verified": True,
    }
    reference = {
        **deepcopy(common),
        "target_status": "skipped",
        "target_reason": "memory budget",
        "target_scored_frames": 0,
        "target_signal_published": False,
        "target_series_published": False,
    }
    candidate = {
        **deepcopy(common),
        "target_status": "ok",
        "target_reason": None,
        "target_scored_frames": 11,
        "target_signal_published": True,
        "target_series_published": True,
        "target_device": "mps",
        "target_dtype": "float32",
        "target_revisions": dict.fromkeys(study.original.MODELS, "pinned"),
    }
    return reference, candidate


def test_detected_faces_require_a_bound_score_for_each_available_crop(paired):
    reference, candidate = paired
    assert all(study.compare(reference, candidate).values())
    assert not study.compare(reference, {**candidate, "target_scored_frames": 10})[
        "candidate_disposition_verified"
    ]
    assert not study.compare(reference, {**candidate, "target_series_published": False})[
        "candidate_disposition_verified"
    ]
    assert not study.compare({**reference, "target_status": "failed"}, candidate)[
        "reference_disposition_verified"
    ]


def test_both_arms_can_correctly_report_no_detected_face(paired):
    reference, candidate = paired
    for arm in (reference, candidate):
        arm["input"]["faces"] = 0
        arm.update(
            target_status="not_applicable",
            target_reason=study.NO_FACE,
            target_scored_frames=0,
            target_signal_published=False,
            target_series_published=False,
        )
    assert all(study.compare(reference, candidate).values())
    assert not study.compare(reference, {**candidate, "target_scored_frames": 1})[
        "candidate_disposition_verified"
    ]
    assert not study.compare(reference, {**candidate, "target_signal_published": True})[
        "candidate_disposition_verified"
    ]
    assert not study.compare(
        reference, {**candidate, "target_status": "skipped", "target_reason": "memory budget"}
    )["candidate_disposition_verified"]
    candidate["non_target_runs"][0] = ("vid.faces.yunet", "failed")
    assert not study.compare(reference, candidate)["same_detected_face_coverage"]


def test_pairing_requires_same_other_routes_fuser_and_face_coverage(paired):
    reference, candidate = paired
    candidate["non_target_runs"][1] = ("unchanged", "timeout")
    candidate["input"]["faces"] = 10
    candidate["fuser_sha256"] = "other"
    candidate["condition_profile"]["height"] = 720
    comparison = study.compare(reference, candidate)
    for key in (
        "same_non_target_route_statuses",
        "same_detected_face_coverage",
        "same_fuser",
        "same_condition_profile",
    ):
        assert comparison[key] is False


def test_matching_face_counts_cannot_hide_different_sampled_face_support(paired):
    reference, candidate = paired
    candidate["input"]["face_inputs_sha256"] = "different-support"
    assert not study.compare(reference, candidate)["same_detected_face_coverage"]


def test_face_support_hash_binds_native_anchor_and_boxes(monkeypatch):
    monkeypatch.setattr(study.original, "verify_input", lambda *_: {"faces": 1})
    outputs = {
        "decode": {"frames": [{"frame_index": 5, "pts": 5005}]},
        "faces": {"faces": [[{"x": 7}]]},
    }
    original = study.verify_input(outputs, {}, "input")
    outputs["decode"]["frames"][0]["pts"] += 1
    assert original != study.verify_input(outputs, {}, "input")
    outputs["decode"]["frames"][0]["pts"] -= 1
    outputs["faces"]["faces"][0][0]["x"] += 1
    assert original != study.verify_input(outputs, {}, "input")
    outputs["faces"]["faces"] = []
    assert study.verify_input(outputs, {}, "input")["face_inputs_sha256"] is None


def test_successor_preparation_binds_the_exact_native_clocks(monkeypatch):
    clock = {
        "picture_pts": list(range(0, 180180, 1001)),
        "video_time_base": "1/30000",
        "video_end_pts": 180180,
        "verified_h264_packets": 180,
        "last_packet_duration": 1001,
        "audio_time_base": "1/48000",
        "audio_origin_pts": 0,
        "audio_samples": 288288,
    }
    expected = (
        Path("group-03/02.mp4"),
        {"file_sha256": study.CASE_SHA256, "clock": clock},
        study.CASE_RECEIPT_SHA256,
    )

    def scoped(function, overrides):
        assert function is study.original.preparation
        assert overrides == {"GROUP": "03"}
        return lambda: expected

    monkeypatch.setattr(study, "scoped", scoped)
    assert study.preparation() == expected
    clock["picture_pts"][42] += 1
    with pytest.raises(ValueError, match="picture/audio clocks"):
        study.preparation()


def test_verified_no_face_coverage_does_not_claim_candidate_admission(monkeypatch):
    def summary():
        return {
            "engineering_passed": True,
            "comparison": {
                "same_detected_face_coverage": True,
                "both_exact_input_bindings": True,
                "reference_disposition_verified": True,
                "candidate_disposition_verified": True,
            },
            "arms": {
                arm: {"target_status": "not_applicable", "customer_direction": "unavailable"}
                for arm in study.original.ARMS
            },
        }

    monkeypatch.setattr(study, "scoped", lambda *_: summary)
    observed = study.readout()
    assert observed["engineering_passed"] is True
    assert observed["candidate_admitted_and_scored"] is False
    assert all(row["target_coverage"] == "no_detected_face" for row in observed["arms"].values())
    assert all(row["customer_direction"] == "unavailable" for row in observed["arms"].values())
