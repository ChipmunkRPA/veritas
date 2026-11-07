"""Boundary truth, deterministic mixtures and source-only routing contracts."""
import json

import numpy as np
import pytest

from scripts import prepare_audio_specialists_v15 as preparation


@pytest.mark.parametrize(('labels', 'expected'), [([0, 0], [0, 0, 0]), ([1, 1], [1, 1, 1]),
    ([0, 1], [0, 1, 1]), ([1, 0], [1, 1, 0])])
def test_local_labels_follow_actual_source_boundaries_not_the_file_label(labels, expected):
    assert [preparation.local_truth(labels, start, start + 64000) for start in (0, 32000, 64000)] == expected
    assert preparation.local_truth(labels, 63999, 64000) == labels[0]
    assert preparation.local_truth(labels, 64000, 64001) == labels[1]


def test_method_roundtrips_exactly_before_freezing():
    assert json.loads(json.dumps(preparation.METHOD)) == preparation.METHOD
    assert preparation.METHOD['minimum_groups_per_class'] == 25
    assert preparation.METHOD['serving_admitted'] is False


def sources():
    cache = {'human': (.2 * np.sin(np.arange(64000) * .15)).astype(np.float32),
        'ai': (.3 * np.sin(np.arange(64000) * .2)).astype(np.float32)}
    backgrounds = []
    for index, kind in enumerate(('noise', 'music', 'speech')):
        key = 'background-' + kind
        cache[key] = np.random.default_rng(index).normal(0, .1, 80000).astype(np.float32)
        backgrounds.append({'id': key, 'sha256': key, 'kind': kind, 'role': 'specialist_cal'})
    pair = {'id': 'authored-pair', 'role': 'specialist_cal', 'generator': 'authored-only',
        'foregrounds': [{'sha256': 'human'}, {'sha256': 'ai'}]}
    return pair, {'backgrounds': backgrounds}, cache


def test_noisy_composites_preserve_first_half_common_gain_and_local_truth():
    pair, plan, cache = sources()
    rows = list(preparation.pair_signals(pair, plan, cache))
    repeated = list(preparation.pair_signals(pair, plan, cache))
    assert len(rows) == 28
    assert len({row['common_peak_gain'] for row, _ in rows}) == 1
    for (row, pcm), (again, copy) in zip(rows, repeated, strict=True):
        assert row == again and np.array_equal(pcm, copy)
        assert pcm.dtype == np.float32 and pcm.shape == (128000,)
        assert np.abs(pcm).max() <= .980001
    for condition in {r['condition'] for r, _ in rows}:
        human = next(pcm for r, pcm in rows if r['condition'] == condition and r['pattern'] == 'human')
        mixed = next(pcm for r, pcm in rows if r['condition'] == condition and r['pattern'] == 'human-ai')
        assert np.array_equal(human[:64000], mixed[:64000])
        assert not np.array_equal(human[64000:], mixed[64000:])


def test_silent_fixed_background_is_refused_without_choosing_another():
    pair, plan, cache = sources()
    cache['background-noise'][:] = 0
    with pytest.raises(ValueError, match='do not select another offset'):
        list(preparation.pair_signals(pair, plan, cache))


def test_operating_point_uses_only_assessed_genuine_fusion_cal_groups(monkeypatch):
    from scripts import run_audio_specialists_v15 as scoring

    rows = [{'id': str(i), 'group': str(i), 'role': 'fusion_cal', 'label': 0, 'authored_score': i / 40}
        for i in range(30)]
    rows += [{'id': 'missing', 'group': 'missing', 'role': 'fusion_cal', 'label': 0, 'authored_score': None},
        {'id': 'ai', 'group': 'ai', 'role': 'fusion_cal', 'label': 1, 'authored_score': 1.},
        {'id': 'reserved', 'group': 'reserved', 'role': 'confirmation', 'label': 0, 'authored_score': 1.}]
    monkeypatch.setattr(scoring, 'predict_file', lambda file, _: file['authored_score'])
    point = scoring.operating_point({'files': rows}, {})
    assert point['genuine_groups_total'] == 31
    assert point['genuine_groups_with_assessed_files'] == 30
    assert point['unassessed_genuine_files'] == 1
    assert point['threshold'] == 27 / 40
    assert point['calibration_genuine_groups_flagged'] == 2
    rows[-1]['authored_score'] = 0
    rows[-2]['authored_score'] = 0
    assert scoring.operating_point({'files': rows}, {}) == point


def test_sparse_operating_support_never_fills_a_missing_score_with_zero(monkeypatch):
    from scripts import run_audio_specialists_v15 as scoring

    rows = [{'id': str(i), 'group': str(i), 'role': 'fusion_cal', 'label': 0,
        'authored_score': .8 if i < 24 else None} for i in range(30)]
    monkeypatch.setattr(scoring, 'predict_file', lambda file, _: file['authored_score'])
    point = scoring.operating_point({'files': rows}, {})
    assert point['status'] == 'insufficient_support' and point['threshold'] is None
    assert point['genuine_groups_with_assessed_files'] == 24


def test_inconclusive_ai_counts_as_not_found_and_does_not_clear_the_file():
    from scripts import run_audio_specialists_v15 as scoring

    rows = [{'id': 'mixed', 'group': 'pair', 'label': 1, 'probability': None},
        {'id': 'human', 'group': 'pair', 'label': 0, 'probability': .5}]
    result = scoring.summarize(rows, {'status': 'fitted', 'threshold': .9})
    assert result['inconclusive'] == 1 and result['ai_not_flagged_including_inconclusive'] == 1
    assert result['ai_flagged'] == 0 and result['false_ai'] == 0
