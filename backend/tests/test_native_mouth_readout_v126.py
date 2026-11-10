import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.video.decode import FrameTable
from scripts import native_mouth_readout_v126 as method


def calibration_rows():
    return [
        {
            "source_group": f"group-{g}",
            "condition": "ai_aligned" if y else "original",
            "study_role": "confidence_cal",
            "label": y,
        }
        for g in range(10)
        for y in (0, 1)
    ]


def test_separate_study_calibration_is_monotone_and_never_reverses_bad_head():
    rows = calibration_rows()
    raw = np.asarray([2 * r["label"] - 1 for r in rows], np.float64)
    calibration = method.calibrate(raw, rows)
    p = method.probability(raw, calibration)
    assert p[1] > 0.7 and p[0] < 0.3 and calibration["deployment_eligible"] is False
    assert np.all(np.diff(method.probability(np.arange(-4, 5), calibration)) >= 0)
    reversed_calibration = method.calibrate(-raw, rows)
    assert reversed_calibration["slope"] == pytest.approx(0, abs=1e-7)
    assert np.allclose(method.probability(raw, reversed_calibration), 0.5, atol=1e-6)


@pytest.mark.parametrize("role", ["fusion_fit", "selection", "confirmation"])
def test_fit_or_test_groups_cannot_calibrate(role):
    rows = calibration_rows()
    rows[0]["study_role"] = role
    with pytest.raises(ValueError, match="calibration"):
        method.calibrate(np.zeros(len(rows)), rows)


def test_calibration_rejects_nonfinite_or_missing_margin():
    rows = calibration_rows()
    with pytest.raises(ValueError, match="finite"):
        method.calibrate([np.nan] * len(rows), rows)
    with pytest.raises(ValueError, match="margin"):
        method.calibrate([0.0], rows)


def test_class_source_condition_mass_is_balanced_without_discarding_short_source():
    rows = calibration_rows() + [
        {"source_group": "short", "condition": "original", "study_role": "confidence_cal", "label": 0}
    ]
    rows += [copy.deepcopy(rows[0]) for _ in range(9)]
    weights = method.balanced_weights(rows, "confidence_cal")
    assert weights.sum() == pytest.approx(1)
    assert weights[[i for i, r in enumerate(rows) if r["label"] == 0]].sum() == pytest.approx(0.5)
    assert weights[[i for i, r in enumerate(rows) if r["source_group"] == "short"]].sum() == pytest.approx(
        0.5 / 11
    )


def test_research_point_budgets_count_worst_control_per_group_with_strict_comparison():
    rows = []
    for group in range(10):
        for shift in (0.0, -0.2, -0.5):
            rows.append(
                {
                    "source_group": str(group),
                    "condition": "original",
                    "study_role": "confidence_cal",
                    "label": 0,
                    "window_score": group + shift,
                }
            )
    points = method.operating_points(rows)
    for budget in method.BUDGETS:
        assert sum(g > points[str(budget)]["raw_margin"] for g in range(10)) == budget
    rows[0]["study_role"] = "selection"
    with pytest.raises(ValueError, match="calibration"):
        method.operating_points(rows)


def test_all_consecutive_native_windows_have_exact_five_picture_support():
    table = FrameTable("1/1000", tuple(5000 + 40 * i for i in range(9)), (), "?" * 9, 96, 64, 96, 64)
    windows = method.windows(list(range(1, 8)), np.arange(7, dtype=np.float64), table, 40)
    assert len(windows) == 5
    highest = windows[0]
    assert highest["centre_frame_index"] == 6 and highest["support_frames_half_open"] == [4, 9]
    assert highest["start_seconds"] == "129/25" and highest["end_seconds"] == "134/25"
    assert highest["response"] == pytest.approx(5)
    assert (
        highest["before_frame_index"] == 3
        and highest["during_frame_index"] == 6
        and highest["after_frame_index"] == 8
    )


def test_missing_native_centres_and_timestamp_gaps_never_form_bridged_windows():
    table = FrameTable("1/1000", (0, 40, 80, 120, 160, 1000, 1040, 1080, 1120), (), "?" * 9, 96, 64, 96, 64)
    assert method.windows([1, 2, 4, 5, 6, 7], np.ones(6), table, 40) == []
    assert method.windows([], [], table, 40) == []


def test_exact_interval_union_does_not_double_count_overlap():
    result = method.interval_metrics([["1", "2"], ["3/2", "3"]], ["2", "4"])
    assert result["flagged_duration_seconds"] == 2
    assert result["edit_overlap_seconds"] == 1
    assert result["temporal_union_iou"] == pytest.approx(1 / 3)
    assert result["outside_edit_flagged_seconds"] == 1
    assert method.interval_metrics([], ["2", "4"])["localized_iou_0_2"] is False


def test_reliability_has_known_brier_and_ece_and_handles_missing_scope():
    result = method.reliability([0.2, 0.8], [0, 1])
    assert result["brier"] == pytest.approx(0.04) and result["ece_10"] == pytest.approx(0.2)
    assert method.reliability([], [])["brier"] is None
    with pytest.raises(ValueError):
        method.reliability([1.5], [0])


@pytest.mark.parametrize("speech_present", [False, True])
def test_current_actual_adapter_routes_native_faces_pcm_and_evidence_without_model_access(
    tmp_path, monkeypatch, speech_present
):
    from app.detectors.audio import decode as audio_decode
    from app.detectors.video import lipsync

    n = 100
    table = FrameTable("1/1000", tuple(5000 + 40 * i for i in range(n)), (), "?" * n, 96, 64, 96, 64)
    path = tmp_path / "authored.bin"
    path.write_bytes(b"authored route fixture, not actual media")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pcm = np.sin(2 * np.pi * 400 * np.arange(n * 640) / 16000).astype(np.float32) * 0.3
    mapping = {
        "samples": len(pcm),
        "pcm_sha256": hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest(),
        "source_sha256": digest,
        "timeline_time_base": "1/16000",
        "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 80000}],
    }
    monkeypatch.setattr(
        audio_decode, "decode_audio", lambda *a, **k: SimpleNamespace(pcm=pcm, mapping=mapping)
    )
    levels = np.asarray([0.1 + ((i * 17) % 73) / 100 for i in range(n)])
    monkeypatch.setattr(
        lipsync,
        "vad",
        lambda pcm: lipsync.Speech(
            20 * np.log10(levels), levels, np.zeros(n), np.full(n, speech_present), -50
        ),
    )
    monkeypatch.setattr(
        lipsync, "speaker_score", lambda *a: 0.9
    )  # Exercise input/result routing, not attribution accuracy.
    landmarks = [[30, 22], [66, 22], [48, 34], [35, 42], [61, 42]]
    row = {
        "status": "extracted",
        "table": table.to_json(),
        "frames": [
            {
                "status": "ok",
                "frame_index": i,
                "pts": table.pts[i],
                "face": {"box": [12, 8, 72, 52], "landmarks": landmarks},
            }
            for i in range(n)
        ],
    }

    class Store:
        def __init__(self, *args):
            self.indices = []

        def iter_many(self, indices):
            for i in indices:
                self.indices.append(i)
                rgb = np.full((64, 96, 3), 160, np.uint8)
                rgb[33:51, 35:61] = 20 + (i * 17) % 73
                yield i, rgb

        def native_anchors(self):
            return [
                {
                    "frame_index": i,
                    "pts": table.pts[i],
                    "time_base": table.time_base,
                    "decoded_rgb_sha256": "a" * 64,
                }
                for i in self.indices
            ]

    monkeypatch.setattr(method, "VerifiedFrameStore", Store)
    result = method.current_check(row, {"file": str(path), "file_sha256": digest})
    assert result["status"] == "ok"
    assert result["assessed"] is speech_present
    if speech_present:
        assert result["details"]["av_input"]["status"] == "ready"
        assert result["details"]["av_input"]["audio_origin_pts"] == 80000
        assert all(
            p["frame_pts"] == table.pts[p["frame_index"]]
            for t in result["details"]["av_input"]["tracks"]
            for p in t["frame_inputs"]
        )
    else:
        assert result["flagged"] is False
