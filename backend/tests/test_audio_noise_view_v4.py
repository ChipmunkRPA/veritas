from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from scripts.audio_noise_view_v4 import enhance
from scripts.prepare_audio_noise_v4 import connected_folds
from scripts.review_audio_noise_v4 import cv, decision, metrics, weights


class IdentitySpectrum(torch.nn.Module):
    def forward(self, spectrum):
        return spectrum


@pytest.mark.parametrize('samples', [16000, 40000, 48000, 52000, 60000, 64000])
def test_signal_view_keeps_every_sample_and_does_not_shift_source_clock(samples):
    values = np.random.default_rng(samples).normal(0, .1, samples).astype(np.float32)
    before = values.copy()
    result = enhance(values, IdentitySpectrum())
    assert result.shape == values.shape and result.dtype == np.float32
    np.testing.assert_allclose(result, values, atol=1e-7)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize('values', [np.zeros(15999, np.float32), np.zeros(64001, np.float32),
    np.zeros((16000, 2), np.float32), np.full(16000, np.nan, np.float32), np.zeros(16000, np.float64)])
def test_unsupported_signal_is_refused_before_enhancement(values):
    with pytest.raises(ValueError, match='finite float32 mono'):
        enhance(values, IdentitySpectrum())


def test_source_repetitions_do_not_gain_probability_fit_weight():
    rows = [{'base_id':'real-a','label':0,'group':'a'}, {'base_id':'real-b','label':0,'group':'b'},
            {'base_id':'ai-a','label':1,'group':'c'}]
    expected = weights(rows)
    extended = rows+[dict(rows[0]) for _ in range(19)]
    actual = weights(extended)
    assert actual[[0, *range(3, 22)]].sum() == pytest.approx(expected[0])
    assert actual[1] == pytest.approx(expected[1]) and actual[2] == pytest.approx(expected[2])
    assert actual.sum() == pytest.approx(1.)


def test_connected_voice_and_known_prompt_indices_never_cross_folds():
    bases = []
    for i in range(20):
        bases.extend([{'id':f'auth-{i}','cluster_id':f'voice-{i}','dataset':'isjwdu/DFADD','label':0},
                      {'id':f'fake:p{i+200}_{i//2}_engine','cluster_id':f'voice-{i}',
                       'dataset':'isjwdu/DFADD','label':1}])
    assignment, groups, counts = connected_folds(bases)
    assert len(groups) == 10 and sum(map(sum, counts)) == 40
    for i in range(20):
        assert assignment[f'auth-{i}'] == assignment[f'fake:p{i+200}_{i//2}_engine']
    for i in range(0, 20, 2):
        assert assignment[f'auth-{i}'] == assignment[f'auth-{i+1}']


def development_rows():
    return [{'base_id':f'{fold}-{label}-{n}','group':f'voice-{fold}-{label}-{n}',
             'fold':fold,'label':label,'raw':.15+.70*label+.01*n,'enhanced_raw':.2+.60*label+.01*n}
            for fold in range(5) for label in (0, 1) for n in range(3)]


def test_heldout_labels_cannot_change_their_own_cv_probability():
    rows = development_rows()
    before, _ = cv(rows)
    changed = deepcopy(rows)
    for row in changed:
        if row['fold'] == 0:
            row['label'] = 1-row['label']
    after, _ = cv(changed)
    for left, right in zip(before, after, strict=True):
        if left['fold'] == 0:
            assert {k:v for k,v in left.items() if k.endswith('_p')} == {k:v for k,v in right.items() if k.endswith('_p')}


def test_connected_group_leakage_is_rejected():
    rows = development_rows()
    rows[0]['group'] = rows[-1]['group']
    with pytest.raises(ValueError, match='crossed development folds'):
        cv(rows)


def test_unassessed_inputs_stay_in_operational_denominators():
    rows = [{'base_id':'a','label':0,'group':'a','p':None}, {'base_id':'b','label':1,'group':'b','p':.9}]
    result = metrics(rows, 'p')
    assert result['planned_inputs'] == 2 and result['assessed_inputs'] == 1
    assert result['unassessed_inputs'] == 1 and result['decisions']['inconclusive'] == 1
    assert result['correct_original'] == 0 and result['brier'] is None
    assert decision(None) == decision(float('nan')) == decision(True) == 'inconclusive'


def test_operational_reference_respects_saved_guard_instead_of_score_only():
    rows = [{'base_id':'a','label':0,'group':'a','p':.05,'saved':'inconclusive'},
            {'base_id':'b','label':1,'group':'b','p':.95,'saved':'ai'}]
    result = metrics(rows, 'p', decision_key='saved')
    assert result['correct_original'] == 0 and result['correct_ai'] == 1
    assert result['decisions']['inconclusive'] == 1
