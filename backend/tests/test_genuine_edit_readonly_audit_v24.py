"""Adverse source/count inputs must not pass the read-only review."""
from copy import deepcopy

import pytest

from scripts.audit_genuine_edits_v24 import counter_summary, source_proof
from scripts.run_genuine_edits_v24 import ARMS
from scripts.score_video_specialists_v20 import VIEWS


def records():
    return [{'id': f'{kind}-{i}', 'component': f'pair-{i // 2}', 'kind': kind,
        'raw_inspection_alerts_not_ai_proof': {v: None for v in VIEWS},
        'reference_ai_proxy_flags': {a: None for a in (*ARMS, 'appearance-mapped')}}
        for kind in ('original-baseline', 'ordinary-edit') for i in range(40)]


def test_unassessed_cases_are_counted_without_becoming_clean_decisions():
    result = counter_summary(records())
    for kind in result.values():
        assert all(r == {'inspection_alerts': 0, 'inconclusive': 40} for r in kind['raw_cues'].values())
        assert all(r == {'unsupported_ai_proxy_flags': 0, 'inconclusive': 40, 'components_with_proxy_flags': 0}
            for r in kind['reference_methods'].values())


@pytest.mark.parametrize('mutation', ['missing-case', 'missing-component', 'integer-alert', 'integer-proxy'])
def test_missing_cases_or_ambiguous_booleans_refuse_aggregate_release(mutation):
    rows = records()
    if mutation == 'missing-case':
        rows.pop()
    elif mutation == 'missing-component':
        rows[0]['component'] = rows[2]['component']
        rows[1]['component'] = rows[2]['component']
    elif mutation == 'integer-alert':
        rows[0]['raw_inspection_alerts_not_ai_proof'][VIEWS[0]] = 1
    else:
        rows[0]['reference_ai_proxy_flags'][ARMS[0]] = 0
    with pytest.raises(ValueError):
        counter_summary(rows)


def proof():
    counts = {'face_detection_inputs': 1, 'crop_classifier_inputs': 0,
        'appearance_inputs': 0, 'patch_feature_inputs': 0, 'native_repeat_inputs': 0}
    plan = {'id': 'clip', 'sha256': 'a' * 64, 'source_geometry': {'pts_table_sha256': 'b' * 64},
        'units': [{'location': {'sampled_frames': [0], 'context_frames': []}}]}
    planned = {'plan': plan, 'rgb_sha256': ['c' * 64]}
    native = {'id': 'clip', 'input_sha256': 'a' * 64, 'native_table_sha256': 'b' * 64,
        'frame_inputs': [{'frame_index': 0, 'pts': 0, 'time_base': '1/24',
            'source_rgb_sha256': 'c' * 64, 'status': 'not_applicable'}],
        'native_input_counts': counts, 'native_repeats': []}
    return planned, native


def test_native_source_proof_preserves_unassessed_frame_without_invented_model_inputs():
    planned, native = proof()
    assert source_proof(planned, native) == native['native_input_counts']


@pytest.mark.parametrize('mutation', ['rgb', 'clock', 'pts', 'duplicate', 'counter', 'invented-feature'])
def test_source_frame_and_input_counts_must_correspond(mutation):
    planned, native = proof()
    if mutation == 'rgb':
        native['frame_inputs'][0]['source_rgb_sha256'] = 'd' * 64
    elif mutation == 'clock':
        native['native_table_sha256'] = 'e' * 64
    elif mutation == 'pts':
        native['frame_inputs'][0]['pts'] = 1
    elif mutation == 'duplicate':
        native['frame_inputs'].append(deepcopy(native['frame_inputs'][0]))
    elif mutation == 'counter':
        native['native_input_counts']['appearance_inputs'] = 1
    else:
        native['frame_inputs'][0]['patch_feature_sha256'] = 'f' * 64
    with pytest.raises(ValueError):
        source_proof(planned, native)
