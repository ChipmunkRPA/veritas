import pytest

from scripts import video_faceswapai_full_app_v187 as trial


def sample_input():
    indices = [0, 72, 144]
    table = {"pts": trial.PICTURE_PTS}
    hashes = [str(index) for index in range(len(trial.PICTURE_PTS))]
    expected = {"frame_table": table, "rgb_sha256": hashes}
    outputs = {
        "ingest": {"sha256": "source"},
        "probe": {"frame_table": table, "plan": {"legacy": {"frames": indices}}, "info": {"has_audio": False}},
        "decode": {
            "frames": [
                {"index": i, "frame_index": index, "pts": trial.PICTURE_PTS[index], "sha256": hashes[index]}
                for i, index in enumerate(indices)
            ],
            "audio": {"status": "absent", "samples": 0, "source_mapping": None},
        },
        "faces": {"faces": [["face"], [], ["face"]]},
    }
    return outputs, expected


def test_silent_input_maps_sample_rows_to_the_exact_native_frames():
    outputs, expected = sample_input()
    assert trial.verify_input(outputs, expected, "source") == {
        "sampled_frames": 3,
        "face_frame_indices": [0, 144],
        "sampled_indices": [0, 72, 144],
    }


@pytest.mark.parametrize("key,value", [("pts", 0), ("sha256", "wrong"), ("index", 5)])
def test_changed_input_rows_cannot_be_counted(key, value):
    outputs, expected = sample_input()
    outputs["decode"]["frames"][1][key] = value
    with pytest.raises(ValueError, match="exact sampled RGB"):
        trial.verify_input(outputs, expected, "source")


def test_present_audio_cannot_be_recorded_as_absent():
    outputs, expected = sample_input()
    outputs["probe"]["info"]["has_audio"] = True
    with pytest.raises(ValueError, match="absent soundtrack"):
        trial.verify_input(outputs, expected, "source")


def test_score_summary_respects_the_half_open_edit_and_preserves_missing_scores():
    assert trial.score_summary([(71, 0.2), (72, 0.6), (143, 0.8)]) == {
        "before": {"frames": 1, "mean": 0.2, "maximum": 0.2},
        "inside": {"frames": 2, "mean": 0.7, "maximum": 0.8},
        "after": {"frames": 0, "mean": None, "maximum": None},
    }
    assert trial.partition(144) == "after"
    assert trial.overlap({71, 72, 73}, range(72, 76)) == {"intersection_frames": 2, "union_frames": 5, "iou": 0.4}
