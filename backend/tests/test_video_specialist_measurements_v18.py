"""Exact fixed-window arithmetic and unavailable context; no pretrained model."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.detectors.video import face_appearance, face_sequence
from app.detectors.video.decode import FrameTable
from scripts.measure_video_specialists_v18 import fixed_change, overlap, supported_window
from scripts.prepare_video_scoring_units_v18 import plan_case
from scripts.prepare_video_specialists_v18 import frame_clock


def fixture():
    table = FrameTable.from_json({'pts': [i * 512 for i in range(144)], 'time_base': '1/12288',
        'truncated': False, 'interpolated': [], 'width': 640, 'height': 480, 'rotation': 0,
        'coded_width': 640, 'coded_height': 480,
        'key_frames': [0], 'pict_types': 'I' + 'P' * 143})
    row = {'source_geometry': frame_clock(table.to_json(), 144), 'known_edit': None, 'label': 0}
    unit = next(u for u in plan_case(row)['units'] if u['view'] == 'face-appearance')
    centers = set(unit['before_centers'] + unit['middle_centers'] + unit['after_centers'])
    inputs = {i for c in centers for i in (c - 1, c, c + 1)}
    return table, unit, centers, inputs


@pytest.mark.parametrize('appearance', [False, True])
def test_fixed_three_middle_arithmetic_matches_existing_cue_on_exact_same_support(appearance):
    table, unit, centers, inputs = fixture()
    rng = np.random.default_rng(71)
    features = {i: rng.normal(size=(128,) if appearance else (64, 384)).astype(np.float32) for i in inputs}
    if not appearance:
        features = {i: value / np.linalg.norm(value, axis=-1, keepdims=True) for i, value in features.items()}
    tracks = {'authored-single-sequence': [SimpleNamespace(frame_index=i) for i in sorted(inputs)]}
    reference = (face_appearance.windows if appearance else face_sequence.sequence_windows)(centers, tracks, features, table)
    assert len(reference) == 1
    assert fixed_change(unit, features, appearance=appearance) == pytest.approx(reference[0]['response'], abs=1e-12)


@pytest.mark.parametrize('appearance', [False, True])
def test_missing_planned_frame_returns_no_numeric_negative(appearance):
    _, unit, _, inputs = fixture()
    features = {i: np.ones((128,) if appearance else (64, 384), np.float32) for i in inputs}
    features.pop(min(inputs))
    assert fixed_change(unit, features, appearance=appearance) is None


def test_geometric_discontinuity_is_unavailable_without_an_identity_claim():
    _, unit, _, inputs = fixture()
    boxes = {i: [100, 100, 100, 100] for i in inputs}
    assert supported_window(unit, boxes)
    boxes[max(inputs)] = [400, 300, 100, 100]
    assert not supported_window(unit, boxes)
    assert overlap([0, 0, 10, 10], [20, 20, 10, 10]) == 0


def test_invalid_appearance_feature_is_a_failure_not_an_unavailable_negative():
    _, unit, _, inputs = fixture()
    features = {i: np.ones(128, np.float32) for i in inputs}
    features[min(inputs)][:] = float('nan')
    with pytest.raises(ValueError, match='finite nonzero'):
        fixed_change(unit, features, appearance=True)
