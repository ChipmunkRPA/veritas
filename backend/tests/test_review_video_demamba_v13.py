"""Separate arithmetic must retain denominators and refuse altered receipts."""
import json

import numpy as np
import pytest
from test_video_demamba_execution_v13 import matrix

from scripts import review_video_demamba_v13 as review
from scripts import run_video_demamba_v13 as runner


def fixture(tmp_path, monkeypatch):
    lock, body, paths, _, _ = matrix(tmp_path)
    body.update(recordings_scored=0, calibration_fitted=False, deployment_eligible=False,
                native_cases=runner.native_plan(body['rows']))
    monkeypatch.setattr(runner, 'check', lambda _: body)
    summary = tmp_path / 'summary.json'
    runner.compare(lock, paths['baseline'], paths['candidate'], summary)
    freeze = tmp_path / 'public-freeze.json'
    freeze.write_text(json.dumps({'private_lock_sha256': runner.digest(lock), 'lock': body}))
    monkeypatch.setattr(review, 'FREEZE_SHA', review.sha(freeze))
    compared = json.loads(summary.read_text())
    gates = {**compared['gates_before_native_audit'], 'native_correspondence': True}
    native = tmp_path / 'native.json'
    native.write_text(json.dumps({'lock_sha256': runner.digest(lock), 'summary_sha256': runner.digest(summary),
                                 'cases': [{'identity': k, 'passed': True} for k in body['native_cases']],
                                 'gates': gates, 'advance_to_fresh_confirmation': all(gates.values()), 'deployment_eligible': False}))
    return freeze, paths['baseline'], paths['candidate'], summary, native, tmp_path / 'review.json'


def test_complete_standalone_arithmetic_agrees_with_original_synthetic_comparison(tmp_path, monkeypatch):
    args = fixture(tmp_path, monkeypatch)
    review.review(*args)
    receipt = json.loads(args[-1].read_text())
    assert receipt['passed'] and receipt['development_outcomes_per_method'] == 192
    assert receipt['totals']['candidate']['inconclusive'] == 4
    assert not receipt['serving_admitted']


@pytest.mark.parametrize('fault', ['freeze', 'source', 'omitted', 'missing_vote', 'failure', 'check_failure', 'counts', 'probability', 'gate', 'native_order', 'native_status'])
def test_changed_complete_packets_are_rejected(tmp_path, monkeypatch, fault):
    args = fixture(tmp_path, monkeypatch)
    if fault == 'freeze':
        args[0].write_text(args[0].read_text() + '\n')
    elif fault in {'source', 'omitted', 'missing_vote', 'failure', 'check_failure'}:
        index = 1 if fault == 'check_failure' else 2
        rows = [json.loads(line) for line in args[index].read_text().splitlines()]
        if fault == 'source':
            rows[1]['label'] = 1
        elif fault == 'omitted':
            rows.pop()
        elif fault == 'missing_vote':
            rows[1]['status'] = 'not_applicable'
        elif fault == 'failure':
            rows[1]['status'] = 'failed'
        else:
            rows[1]['checks'][runner.CHECKS[0]]['status'] = 'failed'
        args[index].write_text(''.join(json.dumps(r) + '\n' for r in rows))
    elif fault in {'counts', 'probability', 'gate'}:
        value = json.loads(args[3].read_text())
        if fault == 'counts':
            value['totals']['candidate']['false_ai'] += 1
        elif fault == 'probability':
            value['calibrated_development']['candidate'][0]['p'] = .01
        else:
            value['gates_before_native_audit']['brier_gain'] = not value['gates_before_native_audit']['brier_gain']
        args[3].write_text(json.dumps(value))
    else:
        value = json.loads(args[4].read_text())
        if fault == 'native_order':
            value['cases'].reverse()
        else:
            value['cases'][0]['passed'] = 1
        args[4].write_text(json.dumps(value))
    with pytest.raises(ValueError):
        review.review(*args)
    assert not args[-1].exists()


def test_tied_ranks_have_hand_checked_auc_and_average_precision():
    result = review.metrics(np.asarray([1, 0, 1, 0]), np.asarray([.8, .8, .4, .1]))
    assert result['auc'] == .625
    assert result['average_precision'] == pytest.approx((.5 + 2 / 3) / 2)


def test_unassessed_probability_does_not_become_original():
    assert review.decision(None) == 'inconclusive'
    assert review.raw_margin({'status': 'not_applicable'}, 'candidate') is None
