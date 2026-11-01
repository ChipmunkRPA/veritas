"""Protection against source leakage, pseudo-replication, and optimistic missing/tie handling."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('audio_evaluation', Path(__file__).parents[1] / 'scripts/evaluate_audio_candidates.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def rows(n=24, split='cal'):
    return [dict(base_id=f'{label}-{i}', speaker=f'{label}-{i}', label=label, condition=condition,
                 split=split, old=.1+.8*label, new=.2+.6*label)
            for label in (0, 1) for i in range(n) for condition in ('clean','noise','music')]


def test_mix_uses_common_gain_and_measured_snr():
    rng = np.random.default_rng(12)
    signal = rng.normal(size=e.N).astype('float32')
    signal[0] = 100
    mixed, gain = e.mix_conditions(signal, rng.normal(size=e.N), rng.normal(size=e.N))
    assert gain < 1
    assert max(np.abs(x).max() for x in mixed.values()) <= .980001
    for condition in ('noise', 'music'):
        snr = 20*np.log10(np.sqrt(np.mean(mixed['clean'].astype(float)**2))/np.sqrt(np.mean((mixed[condition].astype(float)-mixed['clean'])**2)))
        assert snr == pytest.approx(10, abs=1e-5)


def test_source_weights_and_platt_smoothing_do_not_triple_n():
    original = rows(4)
    assert e.source_weights(original).sum() == pytest.approx(1)
    repeated = original + [dict(r) for r in original if r['base_id']=='0-0']
    params = e.fit_monotonic(e.logit([r['old'] for r in original]), original, calibration=True)
    again = e.fit_monotonic(e.logit([r['old'] for r in repeated]), repeated, calibration=True)
    assert params['source_n_by_class']==[4,4]
    assert params['slopes']==pytest.approx(again['slopes'], abs=1e-6)
    assert params['intercept']==pytest.approx(again['intercept'], abs=1e-6)


def test_holdout_is_rejected_by_fitting_even_if_unlabelled():
    with pytest.raises(ValueError, match='Holdout'):
        e.fit_candidates(rows(3,'fit')+rows(4,'cal')+rows(2,'test'))


def test_threshold_uses_complete_bases_and_strict_ties():
    data=rows()
    result=e.threshold_from_cal(data, np.full(len(data), .8))
    assert result['n_complete_real_sources']==24
    assert result['rank']==24
    assert result['threshold']>.8
    probs=np.full(len(data), .8)
    probs[0]=np.nan
    assert e.threshold_from_cal(data, probs)['n_complete_real_sources']==23
    assert e.threshold_from_cal(rows(10), np.full(60,.8))['threshold'] is None


def test_missing_remains_abstention_not_negative():
    result=e.metric_values([0,0,1,1],[.1,np.nan,.9,np.nan], [.25]*4)
    assert result['assessed_fraction']==.5
    assert result['recall_original_80']==.5
    assert result['recall_ai_80']==.5
    assert result['false_original_80']==0
    assert result['decided_risk_80']==0
    assert result['brier']==pytest.approx(.01)


def test_bootstrap_carries_all_source_conditions():
    data=rows(4)
    for r in data:
        if r['label']==1:
            r['speaker']='shared-voice-'+r['base_id'][-1]
    sampled=e.grouped_indices(data,np.random.default_rng(1))
    for i in range(0,len(data),3):
        assert len([j for j in sampled if j==i])==len([j for j in sampled if j==i+1])==len([j for j in sampled if j==i+2])


def test_sha_leakage_is_rejected_before_model_scoring():
    bases=[dict(id=str(i),speaker=str(i),source_group=str(i),transcript_sha256=str(i),sha256=str(i),decoded_pcm_sha256='same',
                split=s,label=1,generator=g) for i,(s,g) in enumerate([('fit','one'),('cal','one'),('test','two')])]
    with pytest.raises(ValueError,match='Cross-split'):
        e.audit_groups(bases)


def test_rounding_ties_cannot_inflate_discrimination():
    result=e.metric_values([0,1],[round(.99999951,6),round(.99999961,6)],[.5,.5])
    assert result['auc']==.5
    assert result['eer']==.5


def test_no_errors_is_not_zero_upper_risk():
    interval=e.wilson(0,40)
    assert interval['high']==pytest.approx(.08762,abs=.0001)
    assert interval['low']==0


def test_s3_uses_fit_and_cal_with_missing_pairs_excluded():
    fitting=rows(4,'fit')
    calibration=rows(5,'cal')
    calibration[0]['new']=np.nan
    parameters=e.fit_candidates(fitting+calibration)
    assert parameters[e.SYSTEMS[3]]['stacker']['source_n_by_class']==[4,4]
    assert parameters[e.SYSTEMS[3]]['platt']['source_n_by_class']==[5,5]
    assert parameters[e.SYSTEMS[3]]['platt']['condition_rows']==29
    assert all(slope>=0 for slope in parameters[e.SYSTEMS[3]]['stacker']['slopes'])


def test_legacy_float_experiment_cannot_be_relocked_as_production_pcm(monkeypatch):
    monkeypatch.setattr(e,'read_manifest',lambda:{'schema':'chipmunk.audio-evaluation/1'})
    with pytest.raises(ValueError,match='Legacy FLOAT-input experiment is frozen'):
        e.lock()


def test_completed_artifact_cannot_be_overwritten(tmp_path):
    target=tmp_path/'evaluation.json'
    e.save_json(target,{'score':.3})
    original=target.read_bytes()
    with pytest.raises(ValueError,match='Refusing to replace'):
        e.save_json(target,{'score':.8})
    assert target.read_bytes()==original
