"""Verify v202 binding and false-alarm accounting without running models."""

import subprocess
import sys

import pytest

from scripts import video_vcd2_full_app_v202 as study


def test_fresh_worker_import():
    subprocess.run(
        [sys.executable, "-B", "-c", "from scripts import video_vcd2_full_app_v202 as v; v.require_fresh()"],
        check=True,
    )


def test_preparation_distinguishes_changed_pixels_from_designated_interval():
    hashes = ["a" * 64] * 6
    cases = {
        role: {
            "file": f"{role}.mp4",
            "sha256": "f" * 64,
            "rgb_sha256": hashes.copy(),
            "verified_frames": 6,
            "changed_frames": [],
        }
        for role in study.ROLES
    }
    cases["recompression_only"]["rgb_sha256"][3] = "b" * 64
    cases["recompression_only"]["changed_frames"] = [{"index": 3, "pts": 11, "rgb_mae": 0.1}]
    row = {"slot": 1, "selected": {"source_sha256": "e" * 64}}
    doc = {
        "slot": 1,
        "source_sha256": "e" * 64,
        "frame_count": 6,
        "relative_pts": [0, 3, 9, 11, 15, 18],
        "time_base": "1/3",
        "frame_interval": [2, 5],
        "width": 100,
        "height": 50,
        "model_calls": 0,
        "cases": cases,
    }
    study.qualify_slot(row, doc)
    cases["recompression_only"]["changed_frames"][0]["pts"] = 9
    with pytest.raises(ValueError, match="picture hashes"):
        study.qualify_slot(row, doc)


def test_worker_binds_legacy_and_extra_native_inputs(monkeypatch):
    rgb = ["a", "b", "c"]
    expected = {"time_base": "1/1000", "pts": [0, 33, 77], "size": (1280, 720), "rgb": rgb}
    monkeypatch.setattr(study, "expected_case", lambda _: expected)
    outputs = {
        "ingest": {"sha256": "source"},
        "probe": {
            "plan": {"legacy": {"frames": [0, 2]}},
            "info": {"has_audio": False},
            "frame_table": {
                "pts": expected["pts"],
                "time_base": expected["time_base"],
                "width": 1280,
                "height": 720,
                "truncated": False,
                "interpolated": [],
                "rotation": 0,
            },
        },
        "decode": {
            "audio": {"status": "absent", "samples": 0, "source_mapping": None},
            "frames": [
                {"index": 0, "frame_index": 0, "pts": 0, "sha256": "a"},
                {"index": 1, "frame_index": 2, "pts": 77, "sha256": "c"},
            ],
        },
        "faces": {"faces": [[], [True]]},
        "render": {
            "frame_anchors": [
                {"frame_index": i, "pts": pts, "decoded_rgb_sha256": rgb[i]}
                for i, pts in enumerate(expected["pts"])
            ]
        },
    }
    assert study.verify_input(outputs, {}, "source") == {
        "sampled_frames": 2,
        "face_frame_indices": [2],
        "sampled_indices": [0, 2],
        "legacy_samples": outputs["decode"]["frames"],
        "native_render_anchors": 3,
    }
    outputs["render"]["frame_anchors"][1]["decoded_rgb_sha256"] = "a"
    with pytest.raises(ValueError, match="differs from v200"):
        study.verify_input(outputs, {}, "source")


def test_flag_and_customer_directions_remain_separate():
    metrics = study.overlap(2, 6, {"ordinary": [3, 6], "changed": [3, 5]})
    assert metrics["overlap_designated_frames"] == 3
    assert metrics["overlap_changed_frames"] == 2
    assert metrics["outside_changed_frames"] == 3
    method = "face-appearance-change/1"
    measured = {
        "status": "scored",
        "role": "unchanged",
        "customer_direction": "inconclusive",
        "face_review_intervals": {method: [metrics]},
        "edge_refinement": {"status": "no_stable_return_bracket", "crop_forwards": 7},
        "appearance_check_status": "ok",
        "appearance_route_status": "ok",
        "appearance_measured_windows": 1,
        "appearance_selected_window": True,
        "legacy_frame_status": "ok",
        "verified_sampled_frames": 4,
        "verified_native_anchors": 5,
        "verified_evidence_frames": 3,
        "verified_evidence_assets": 7,
    }
    skipped = {"status": "unsupported_upload_size", "role": "chroma_conversion_only"}
    available = measured | {
        "role": "recompression_only",
        "face_review_intervals": {},
        "customer_direction": "original",
    }
    result = study.describe([measured, skipped, available])
    assert result["negative_cases_with_face_review_interval"] == 1
    assert result["customer_direction_counts"]["ai"] == 0
    assert result["customer_direction_unobserved"] == 1
    assert result["case_outcomes"] == {"scored": 2, "unsupported_upload_size": 1}
    assert result["face_review_methods"][method]["negative_cases_flagged"] == 1
    assert study.paired_flag_changes({1: [measured, skipped, available]}) == {
        "chroma_conversion_only": {"comparison_unavailable": 1},
        "recompression_only": {"flagged_to_clear": 1},
    }
