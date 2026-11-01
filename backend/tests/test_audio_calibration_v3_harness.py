"""Boundary and measurement tests for the isolated prospective validation harness."""
import importlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
h = importlib.import_module('calibrate_audio_v3')


def row(label, *, identity=None, base=None, p=.1, decision='inconclusive', condition='clean'):
    identity = identity or ('vctk:one' if label else 'libri:one')
    return dict(label=label, speaker=identity, cluster_id=identity, base_id=base or identity + str(label),
        old_p=p, new_p=p, old_guarded=decision, new_guarded=decision, new_unguarded=decision,
        old_threshold_ai=False, new_threshold_ai=False, old_threshold_established=True,
        new_threshold_established=True, background_id=None, condition=condition)


def test_uncertain_predictions_remain_in_error_and_coverage_denominators():
    values = [row(1, p=.1, decision='original'), row(1, base='another', p=None),
              row(0, p=.1, decision='original')]
    result = h.numeric_metrics(values, 'new_guarded')
    assert result['false_original_rate'] == .5
    assert result['ai_coverage'] == .5
    assert result['real_coverage'] == 1
    assert result['coverage'] == .75


def test_veto_cannot_improve_proper_scores_by_deleting_hard_cases():
    values = [row(0, p=.1, decision='original'), row(1, p=.1, decision='original')]
    before = h.numeric_metrics(values, 'new_unguarded')
    values[1]['new_guarded'] = 'inconclusive'
    after = h.numeric_metrics(values, 'new_guarded')
    assert after['brier'] == before['brier']
    assert after['log_loss'] == before['log_loss']
    assert after['false_original_rate'] < before['false_original_rate']
    assert after['coverage'] < before['coverage']


def test_applied_veto_is_not_reported_as_applied_to_unguarded_system():
    value = row(0, p=.1, decision='original')
    value.update(old_raw=.1, new_raw=.1, old_status='ok', new_status='ok',
        new_probability_only='original', new_guarded='inconclusive',
        new_decision_detail={'original_stability_veto': True}, new_views={'probe_status': 'computed'})
    assert h.point_metrics([value], 'new_guarded')['stability_veto_count'] == 1
    assert h.point_metrics([value], 'new_unguarded')['stability_veto_count'] == 0
    assert h.point_metrics([value], 'new_probability_only')['stability_veto_count'] == 0


def test_common_assessed_population_is_identical_for_both_systems():
    values = [row(0, p=.1), row(1, p=.9), row(1, base='unavailable-old', p=.1)]
    values[-1]['old_p'] = None
    before = h.numeric_metrics(values, 'old_guarded', common=True)
    after = h.numeric_metrics(values, 'new_guarded', common=True)
    assert before['brier'] == after['brier'] == pytest.approx(.01)
    assert h.numeric_metrics(values, 'new_guarded')['brier'] > after['brier']


def test_repeated_variants_do_not_inflate_original_weight():
    values = [row(0, p=.1), row(1, p=.9)]
    expected = h.numeric_metrics(values, 'new_guarded')
    repeated = [*values, dict(values[1]), dict(values[1])]
    assert h.numeric_metrics(repeated, 'new_guarded')['brier'] == pytest.approx(expected['brier'])


def test_missing_threshold_is_unknown_not_zero_false_positive_rate():
    values = [row(0), row(1)]
    for item in values:
        item['new_threshold_established'] = False
    result = h.numeric_metrics(values, 'new_guarded')
    assert result['threshold_fpr'] is None
    assert result['threshold_recall'] is None
    assert h.event_bound(values, 'new_guarded', 0, threshold=True)['one_sided_95_upper'] is None


def test_threshold_uses_independent_identity_maxima_and_conservative_tie():
    values = [dict(split='cal', label=0, speaker=str(i), p=.01 * i) for i in range(40)]
    values.append(dict(split='cal', label=0, speaker='0', p=.99))
    result = h.nominal_threshold(values, 'p')
    assert result['identity_count'] == 40
    assert result['rank'] == 39
    assert result['order_statistic'] == .39
    assert result['threshold'] > .39
    assert result['maxima']['0'] == .99


def test_threshold_missing_identity_score_is_infinity_not_dropped():
    values = [dict(split='cal', label=0, speaker=str(i), p=.01 * i) for i in range(40)]
    values[0]['p'] = values[1]['p'] = None
    result = h.nominal_threshold(values, 'p')
    assert result['identity_count'] == 40
    assert result['status'] == 'not_established'
    assert result['threshold'] is None


def test_threshold_rejects_holdout_data():
    with pytest.raises(ValueError, match='cal only'):
        h.nominal_threshold([dict(split='test', label=0, speaker='one', p=.1)], 'p')


def test_zero_observed_errors_have_nonzero_finite_sample_upper_bound():
    values = [row(0, identity=str(i)) for i in range(20)]
    result = h.event_bound(values, 'new_guarded', 0)
    assert result['events'] == 0
    assert result['identities'] == 20
    assert result['one_sided_95_upper'] == pytest.approx(1 - .05 ** (1 / 20))


def test_zero_bootstrap_difference_does_not_pass_underpowered_identity_precision():
    values = [row(1, identity='vctk:' + str(i)) for i in range(57)]
    result = h.sparse_discordance_precision(values, 1)
    assert result['zero_discordance']
    assert result['new_harm_event_one_sided_95_upper'] > .02
    assert not result['two_percentage_point_precision_established']
    larger = [row(0, identity='libri:' + str(i)) for i in range(150)]
    assert h.sparse_discordance_precision(larger, 0)['two_percentage_point_precision_established']


def test_shared_real_and_generated_voice_stays_in_joint_bootstrap_stratum(monkeypatch):
    monkeypatch.setattr(h, 'BOOTSTRAP_REPLICATES', 4)
    monkeypatch.setattr(h, 'CONDITIONS', ['clean'])
    values = [row(0, identity='vctk:shared', p=.1), row(1, identity='vctk:shared', p=.9),
              row(0, identity='libri:other', p=.2)]
    result = h.bootstrap(values)
    assert result['identity_strata'] == {'librispeech': 1, 'vctk_joint': 1}
    assert result['intervals']['voice']['all']['new_minus_old:brier']['percentile_95'] == [0., 0.]


def test_holdout_authorization_rejects_test_scores_without_prior_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'OUT', tmp_path)
    monkeypatch.setattr(h, 'verify_lock', lambda: ({}, {}))
    monkeypatch.setattr(h, 'read_scores', lambda splits: [])
    (tmp_path / 'run-lock.json').write_text('{}')
    (tmp_path / 'calibration.json').write_text(json.dumps({'run_lock_sha256': h.file_sha(tmp_path / 'run-lock.json'),
        'development_rows_sha256': h.sha(h.util.canonical([]))}))
    (tmp_path / 'scores.jsonl').write_text(json.dumps({'split': 'test'}) + '\n')
    with pytest.raises(ValueError, match='without a test-start lock'):
        h.authorize_test()


def test_holdout_authorization_binds_calibration_and_development_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'OUT', tmp_path)
    monkeypatch.setattr(h, 'verify_lock', lambda: ({}, {}))
    monkeypatch.setattr(h, 'read_scores', lambda splits: [])
    (tmp_path / 'run-lock.json').write_text('{}')
    (tmp_path / 'calibration.json').write_text(json.dumps({'run_lock_sha256': h.file_sha(tmp_path / 'run-lock.json'),
        'development_rows_sha256': h.sha(h.util.canonical([]))}))
    path = tmp_path / 'scores.jsonl'
    path.write_text(json.dumps({'split': 'fit'}) + '\n')
    first = h.authorize_test()
    path.write_text(path.read_text() + json.dumps({'split': 'test'}) + '\n')
    assert h.authorize_test() == first
    path.write_text(path.read_text().replace('fit', 'cal'))
    with pytest.raises(ValueError, match='prefix changed'):
        h.authorize_test()


def test_holdout_cannot_seal_changed_development_scores_after_fitting(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'OUT', tmp_path)
    monkeypatch.setattr(h, 'verify_lock', lambda: ({}, {}))
    monkeypatch.setattr(h, 'read_scores', lambda splits: [{'changed': True}])
    (tmp_path / 'run-lock.json').write_text('{}')
    (tmp_path / 'calibration.json').write_text(json.dumps({'run_lock_sha256': h.file_sha(tmp_path / 'run-lock.json'),
        'development_rows_sha256': h.sha(h.util.canonical([]))}))
    (tmp_path / 'scores.jsonl').write_text(json.dumps({'split': 'fit'}) + '\n')
    with pytest.raises(ValueError, match='Development scores changed'):
        h.authorize_test()
    assert not (tmp_path / 'test-start.json').exists()


def test_probability_fit_is_permanently_closed_after_test_start(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'OUT', tmp_path)
    (tmp_path / 'test-start.json').write_text('{}')
    with pytest.raises(ValueError, match='after test authorization'):
        h.fit()
