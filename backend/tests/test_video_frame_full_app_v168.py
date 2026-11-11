from copy import deepcopy

import pytest
import yaml

from scripts import video_frame_full_app_v168 as study


def test_private_manifest_changes_only_the_one_adapter_and_version():
    source = study.baseline.MANIFEST.read_bytes()
    candidate = study.candidate_manifest(source)
    original, changed = (yaml.safe_load(value) for value in (source, candidate))
    target = next(row for row in original["detectors"] if row["id"] == study.DETECTOR)
    version, adapter, _ = study.ARMS["batch-one"]
    target.update(version=version, adapter=adapter)
    assert changed == original
    with pytest.raises(ValueError, match="roster"):
        study.candidate_manifest(
            source.replace(
                b"models: [dfdc.effnet_b7_555, dfdc.effnet_b7_999]", b"models: [dfdc.effnet_b7_555]"
            )
        )


@pytest.fixture
def inputs():
    expected = {
        "picture_pts": [0, 40, 80],
        "video_time_base": "1/1000",
        "rgb_sha256": ["first", "second", "third"],
        "audio_samples": 1920,
        "pcm_sha256": "audio",
    }
    outputs = {
        "probe": {
            "frame_table": {
                "time_base": "1/1000",
                "pts": [0, 40, 80],
                "width": 1920,
                "height": 1080,
                "truncated": False,
                "interpolated": [],
            },
            "plan": {"legacy": {"frames": [0, 2]}},
        },
        "decode": {
            "frames": [
                {"index": 0, "frame_index": 0, "pts": 0, "time_s": 0, "sha256": "first"},
                {"index": 1, "frame_index": 2, "pts": 80, "time_s": 0.08, "sha256": "third"},
            ],
            "audio": {
                "samples": 1920,
                "source_mapping": {
                    "samples": 1920,
                    "pcm_sha256": "audio",
                    "source_sha256": "video",
                    "timeline_time_base": "1/16000",
                    "timeline": [{"start_sample": 0, "end_sample": 1920, "source_pts": 0}],
                },
            },
            "missing_frames": [],
        },
        "faces": {"faces": [[{"w": 100}], []]},
    }
    return outputs, expected


def test_verified_picture_plan_rgb_and_audio_are_bound_independently(inputs):
    outputs, expected = inputs
    assert study.verify_input(outputs, expected, "video") == {
        "clock_exact": True,
        "rgb_exact": True,
        "pcm_exact": True,
        "sampled_frames": 2,
        "faces": 1,
    }
    tampered = deepcopy(outputs)
    tampered["decode"]["frames"][1]["sha256"] = "first"
    assert not study.verify_input(tampered, expected, "video")["rgb_exact"]
    tampered = deepcopy(outputs)
    tampered["probe"]["frame_table"]["pts"][1] = 39
    assert not study.verify_input(tampered, expected, "video")["clock_exact"]
    outputs["decode"]["audio"]["source_mapping"]["timeline"][0]["source_pts"] = 1
    assert not study.verify_input(outputs, expected, "video")["pcm_exact"]


def test_target_series_requires_ordered_scored_faces_and_exact_published_points(inputs):
    outputs, _ = inputs
    units = [{"unit_ref": "frame:0", "raw": 0.2, "quality": {"time_s": 0}}]
    assert study.frame_series_exact(units, [[0, 0.2]], outputs)
    assert not study.frame_series_exact(units, [[0.08, 0.2]], outputs)
    assert not study.frame_series_exact(units, [[0, 0.21]], outputs)
    units[0]["raw"] = 0.23456789
    assert study.frame_series_exact(units, [[0, 0.234568]], outputs)
    units[0]["unit_ref"] = "frame:1"
    assert not study.frame_series_exact(units, [[0, 0.2]], outputs)


def test_candidate_receipt_needs_more_than_a_success_status():
    base = {
        "pipeline_status": "done",
        "analysis_status": "done",
        "result_hash": "result",
        "packet_set_ready": True,
        "input": {"clock_exact": True, "rgb_exact": True, "pcm_exact": True},
        "non_target_runs": [("unmodified", "ok")],
        "fusion_mode": "artefact",
        "fuser_sha256": "fuser",
    }
    reference = {**base, "target_status": "skipped", "target_reason": "memory budget"}
    candidate = {
        **base,
        "target_status": "ok",
        "target_scored_frames": 4,
        "target_rows_bound": True,
        "target_raw_verified": True,
        "target_signal_published": True,
        "target_series_published": True,
        "target_device": "mps",
        "target_dtype": "float32",
        "target_revisions": dict.fromkeys(study.MODELS, "pinned"),
    }
    assert all(study.compare(reference, candidate).values())
    assert not study.compare(reference, {**candidate, "target_series_published": False})[
        "candidate_scored_and_published"
    ]
    assert not study.compare(reference, {**candidate, "fuser_sha256": "other"})["same_fuser"]
