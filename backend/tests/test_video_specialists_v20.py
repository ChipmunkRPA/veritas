"""Selection ties, minimum support and missing combined observations."""
from types import SimpleNamespace

import pytest

from scripts.score_video_specialists_v20 import (
    MODELS,
    best_iou,
    intervals,
    matched_point,
    native_objects,
    predict,
    summarize,
)


def test_real_loaded_model_wrapper_contract_is_unwrapped_before_native_use():
    from app.detectors.image.adapters.loaders import LoadedModel

    payloads = {mid: object() for mid in MODELS}
    wrappers = {mid: LoadedModel(obj=obj, revision='authored-test') for mid, obj in payloads.items()}
    lease = SimpleNamespace(get=wrappers.__getitem__)
    assert native_objects(lease, lease) == tuple(payloads[MODELS[i]] for i in (0, 1, 3, 2))
    raw = SimpleNamespace(get=payloads.__getitem__)
    with pytest.raises(ValueError, match='wrappers'):
        native_objects(raw, raw)


def test_matched_point_caps_budget_and_never_flags_a_tie():
    values = {str(i): i / 40 for i in range(40)}
    point = matched_point(values, {g: True for g in values})
    assert point['budget'] == 4
    assert sum(v > point['threshold'] for v in values.values()) == 4
    tied = matched_point(dict.fromkeys(values, .5), dict.fromkeys(values, True))
    assert tied['calibration_groups_flagged'] == 0


def test_insufficient_common_support_has_no_threshold():
    assert matched_point({'x': .2}, {'x': False})['threshold'] is None
    assert matched_point({str(i): .1 for i in range(30)}, {'absent': True})['status'] == 'insufficient_support'


@pytest.mark.parametrize('status', ['skipped', 'failed', 'unavailable', 'not_applicable'])
def test_missing_local_observation_cannot_create_combined_original_probability(status):
    file = {'modality': 'video', 'required_families': ['video-face'], 'units': [
        {'specialist': 'crop', 'context': 'single', 'raw': None, 'status': status}]}
    artifact = {'policy': {'specialists': [{'id': 'crop', 'family': 'video-face'}]},
        'specialists': {'crop|single': {'status': 'fitted'}}, 'combiners': {}}
    assert predict(file, artifact) is None


def test_suspicious_support_spans_merge_only_inside_declared_gap():
    locations = [{'start': a, 'end': b} for a, b in
        [(0, 40_000_000), (500_000_000, 540_000_000), (1_500_000_000, 1_540_000_000)]]
    assert intervals(locations) == [[0, 540_000_000], [1_500_000_000, 1_540_000_000]]
    assert best_iou([[20, 60]], [30, 70]) == pytest.approx(.6)
    assert best_iou([], [30, 70]) == 0
    assert best_iou([[20, 60]], None) is None


def test_inconclusive_positives_stay_in_missed_denominator():
    rows = [{'id': 'ai', 'group': 'a', 'label': 1, 'probability': None},
        {'id': 'human', 'group': 'b', 'label': 0, 'probability': None}]
    result = summarize(rows, {'status': 'insufficient_support', 'threshold': None})
    assert result['missed_ai_including_inconclusive'] == 1
    assert result['inconclusive'] == 2
    assert result['source_class_balanced_brier_on_scored_cases'] is None
