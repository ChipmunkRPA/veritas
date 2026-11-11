import struct

import pytest

from scripts import audio_noise_speech_gate_transfer_v267 as transfer


def pcm_windows(*values):
    return b"".join(struct.pack("<h", value) * transfer.SAMPLES for value in values)


def test_control_grid_excludes_partial_interval_overlaps(monkeypatch):
    monkeypatch.setattr(transfer, "FIRST", transfer.SAMPLES)
    data = {"kitchen": pcm_windows(1, 2, 3, 4, 5)}
    start = transfer.SAMPLES + 17
    rows = transfer.control_rows(data, [start])
    assert [row["start_sample"] for row in rows] == [3 * transfer.SAMPLES, 4 * transfer.SAMPLES]


def test_control_grid_rejects_waveform_duplicate_of_mix(monkeypatch):
    monkeypatch.setattr(transfer, "FIRST", 0)
    with pytest.raises(ValueError, match="duplicates a mixture excerpt"):
        transfer.control_rows({"kitchen": pcm_windows(1, 2, 1)}, [0])


def test_summary_separates_unavailable_and_counts_rescued_groups():
    cases = [
        {"case": "a", "arm": "genuine", "condition": "kitchen_0db", "triplet": "first", "payload": b"a"},
        {"case": "b", "arm": "genuine", "condition": "kitchen_0db", "triplet": "second", "payload": b"b"},
        {"case": "c", "arm": "genuine", "condition": "kitchen_0db", "triplet": "third", "payload": None},
    ]
    controls = [{"slot": "kitchen", "environment": "kitchen", "start_sample": 0, "sha256": "a"}]
    groups = {("genuine", "first"): "one", ("genuine", "second"): "one", ("genuine", "third"): "three"}
    observed = {
        "matched": [
            {"case": "a", "spans": [[0, 16000]]},
            {"case": "b", "spans": [[0, 17000]]},
            {"case": "c", "spans": None},
        ],
        "unscreened_background": [{"control": controls[0], "spans": []}],
    }
    result = transfer.summarize(cases, groups, controls, observed)
    outcome = result["by_arm_and_condition"]["genuine"]["kitchen_0db"]
    assert outcome == {
        "planned": 3,
        "prepared": 2,
        "baseline_admitted": 0,
        "candidate_admitted": 2,
        "additional_admissions": 2,
        "source_groups_with_additional_admissions": 1,
    }
    assert result["background_controls_by_environment"]["kitchen"]["candidate_admitted"] == 0
    assert result["speech_free_background_verified"] is False
    observed["matched"][2]["spans"] = []
    with pytest.raises(ValueError, match="differs from the exact transfer input"):
        transfer.summarize(cases, groups, controls, observed)
