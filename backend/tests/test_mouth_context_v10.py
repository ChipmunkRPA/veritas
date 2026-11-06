from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.video.decode import FrameTable
from scripts.mouth_context_v10 import point, selected_signal, windows
from scripts.prepare_mouth_context_v10 import clock_span


def example():
    centers = set(range(1, 96, 3))
    tracks = {"t0": [SimpleNamespace(frame_index=i) for i in range(100)]}
    values = {i: 4.0 if 29 <= i <= 53 else -1.0 for i in range(100)}
    table = SimpleNamespace(seconds=lambda i: 7 + i / 25)
    return centers, tracks, values, table


def test_within_source_comparison_removes_constant_identity_offset():
    centers, tracks, values, table = example()
    a = windows(centers, tracks, values, table)
    b = windows(centers, tracks, {k: v + 31 for k, v in values.items()}, table)
    assert len(a) > 12
    assert [(w["start_s"], w["end_s"], w["centers"]) for w in a] == [
        (w["start_s"], w["end_s"], w["centers"]) for w in b
    ]
    assert [w["response"] for w in a] == pytest.approx([w["response"] for w in b])
    assert a[0]["response"] > 4
    assert a[0]["start_s"] >= 8 and a[0]["end_s"] < 9.3


def test_single_scene_step_is_not_a_middle_only_manipulation():
    centers, tracks, _, table = example()
    values = {i: -2.0 if i < 45 else 3.0 for i in range(100)}
    assert max(w["response"] for w in windows(centers, tracks, values, table)) <= 0


def test_missing_triplet_and_source_gap_cannot_create_a_context_window():
    centers, tracks, values, table = example()
    assert windows(centers, tracks, {i: values[i] for i in centers}, table) == []
    spaced = SimpleNamespace(seconds=lambda i: i / 4)
    assert windows(centers, tracks, values, spaced) == []


def calibration():
    conditions = (
        "original",
        "local-light",
        "local-resample",
        "muse-resample",
        "local-blur-2",
        "local-blur-4",
    )
    return [
        {
            "id": f"{g}:{c}",
            "group": g,
            "condition": c,
            "role": "point",
            "label": 0,
            "status": "ready",
            "heads": {"v8": {"status": "ok", "probe_windows": [{"response": 0.5}]}},
        }
        for g in ("a", "b", "c", "d")
        for c in conditions
    ]


def test_window_point_requires_all_ordinary_reserved_cases_and_is_above_maximum():
    rows = calibration()
    assert 0.5 < point(rows, "v8") < 0.500001
    rows[0]["label"] = 1
    with pytest.raises(ValueError, match="ordinary"):
        point(rows, "v8")
    rows = calibration()
    rows[-1]["heads"]["v8"]["status"] = "not_applicable"
    with pytest.raises(ValueError, match="assessed"):
        point(rows, "v8")
    with pytest.raises(ValueError, match="all24"):
        point(calibration()[:-1], "v8")


def test_absolute_evidence_is_preserved_even_if_relative_response_is_larger():
    assert selected_signal([{"response": 2}], 2, [{"response": 100}], 1) == "absolute"
    assert selected_signal([{"response": 1}], 2, [{"response": 0.4}], 0.4) == "relative"
    assert selected_signal([{"response": 1}], 2, [{"response": 0.3}], 0.4) is None


def test_common_clock_trims_incomplete_picture_tail_and_never_pads():
    table = FrameTable("1/1000", tuple(40 + i * 40 for i in range(56)), (), "P" * 56, 480, 360, 480, 360)
    audio = SimpleNamespace(
        pcm=np.zeros(35840, np.float32),
        mapping={"sample_rate": 16000, "timeline": [{"start_sample": 0, "source_pts": 0}]},
    )
    span = clock_span(table, audio)
    assert span["frames"] == 55
    assert span["source_sample_range"] == [640, 35840]
    assert span["excluded_end_picture_frames"] == 1
    assert table.tb == Fraction(1, 1000)
    audio.mapping["timeline"].append({"start_sample": 100, "source_pts": 200})
    with pytest.raises(ValueError, match="Contiguous"):
        clock_span(table, audio)
