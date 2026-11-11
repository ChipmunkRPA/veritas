import pytest
import yaml

from scripts import video_vcd2_batch_one_v302 as study


def test_candidate_manifest_changes_only_the_intended_video_adapter():
    reference = yaml.safe_load(study.current.MANIFEST.read_bytes())
    candidate = yaml.safe_load(study.current.APP.candidate_manifest(study.current.MANIFEST.read_bytes()))
    target = next(row for row in reference["detectors"] if row["id"] == study.current.APP.DETECTOR)
    target["adapter"] = study.current.APP.ARMS["batch-one"][1]
    target["version"] = study.current.APP.ARMS["batch-one"][0]
    assert candidate == reference


def test_frame_coverage_requires_the_complete_evidence_chain():
    ready = {
        "input": {"face_frame_indices": [3, 5, 7]},
        "target_scored_frames": 3,
        "target_status": "ok",
        "target_reason": None,
        "target_rows_bound": True,
        "target_raw_verified": True,
        "target_signal_published": True,
        "target_series_published": True,
    }
    assert study.coverage(ready)["scored_frames"] == 3
    with pytest.raises(ValueError, match="fully linked"):
        study.coverage({**ready, "target_series_published": False})
    with pytest.raises(ValueError, match="fully linked"):
        study.coverage({**ready, "target_scored_frames": 2})
    skipped = {
        **ready,
        "target_status": "skipped",
        "target_reason": "memory budget",
        "target_scored_frames": 0,
    }
    assert study.coverage(skipped) == {
        "face_frames": 3,
        "scored_frames": 0,
        "status": "skipped",
        "budget_skip": True,
    }
    with pytest.raises(ValueError, match="must not supply scores"):
        study.coverage({**skipped, "target_scored_frames": 1})
