"""Local source binding and balanced errors; authored, no model performance."""
from copy import deepcopy

import pytest

from scripts.review_audio_specialists_v15 import bind_local, local_metrics


def packet():
    windows = [{'start': start, 'quality': {'flatness': .7}, 'speech_eligible': start != 64000}
        for start in (0, 32000, 64000)]
    source = {'id': 'mixed', 'group': 'pair', 'role': 'confirmation', 'input_sha256': 'input',
        'labels': [0, 1], 'pattern': 'human-ai', 'condition': 'speech_-5', 'generator': 'generator', 'windows': windows}
    units, outcomes = [], []
    for start, truth in ((0, 0), (32000, 1), (64000, 1)):
        span = {'kind': 'samples', 'unit': 'samples', 'start': start, 'end': start + 64000,
            'extent': 128000, 'sample_rate': 16000}
        units.append({'id': f'full.{start}', 'specialist': 'audio.full', 'label': truth,
            'context': 'diffuse', 'location': span, 'status': 'ok' if start != 64000 else 'not_applicable',
            'raw': .3 if start != 64000 else None})
        outcomes.append({'file_id': 'mixed', 'unit_id': f'full.{start}', 'specialist': 'audio.full',
            'context': 'diffuse', 'location': deepcopy(span), 'probability': .4 if start != 64000 else None})
    document = {'files': [{'id': 'mixed', 'group': 'pair', 'role': 'confirmation', 'input_sha256': 'input',
        'label': 1, 'units': units}]}
    return document, {'unit_outcomes': outcomes}, {'mixed': source}


def test_mixed_local_truth_and_unassessed_windows_are_not_file_truth():
    document, result, planned = packet()
    rows = bind_local(document, result, planned, contextual=True)
    assert [r['label'] for r in rows] == [0, 1, 1]
    metrics = local_metrics(rows)
    assert metrics['planned'] == 3 and metrics['inconclusive'] == 1
    assert metrics['ai_not_leaning_including_ties_and_inconclusive'] == 2


@pytest.mark.parametrize('change', ['label', 'result_clock', 'missing_result', 'ineligible_score', 'source_input'])
def test_corrupted_local_evidence_cannot_pass(change):
    document, result, planned = packet()
    if change == 'label':
        document['files'][0]['units'][0]['label'] = 1
    elif change == 'result_clock':
        result['unit_outcomes'][0]['location']['start'] = 1
    elif change == 'missing_result':
        result['unit_outcomes'].pop()
    elif change == 'ineligible_score':
        document['files'][0]['units'][-1].update(status='ok', raw=.99)
    else:
        document['files'][0]['input_sha256'] = 'different-file'
    with pytest.raises(ValueError):
        bind_local(document, result, planned, contextual=True)


def test_balancing_uses_source_groups_not_the_number_of_repeated_windows():
    rows = [{'group': 'a', 'label': 0, 'probability': .9}] * 100
    rows += [{'group': 'b', 'label': 0, 'probability': .1}, {'group': 'c', 'label': 1, 'probability': .8}]
    result = local_metrics(rows)
    assert result['source_class_balanced_brier_on_scored_units'] == pytest.approx(((.81 + .01) / 2 + .04) / 2)
    assert result['false_ai'] == 100


def test_tied_and_missing_ai_are_in_the_ai_not_leaning_denominator():
    result = local_metrics([{'group': 'a', 'label': 1, 'probability': .5},
        {'group': 'b', 'label': 1, 'probability': None}])
    assert result['ai'] == 2 and result['ties'] == 1 and result['inconclusive'] == 1
    assert result['ai_not_leaning_including_ties_and_inconclusive'] == 2
