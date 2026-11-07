"""New architecture must preserve coverage, model-only design and strict loading."""
import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch

from scripts import run_writing_modernbert_v4 as run
from scripts import writing_focus_v3 as original
from scripts import writing_modernbert_v4 as method


def test_original_rules_and_data_remain_identical_and_independent():
    assert method.INPUT_SHA == original.INPUT_SHA
    assert method.BASELINE_ASSETS == original.BASELINE_ASSETS
    for key in ('documents', 'author_groups', 'foci', 'views', 'device', 'dtype', 'threads',
                'batch', 'max_len', 'reference_points', 'author_bootstrap', 'development_continuation'):
        assert method.METHOD[key] == original.METHOD[key]
    snapshot = copy.deepcopy(original.METHOD)
    method.METHOD['reference_points'].append(0.999)
    try:
        assert snapshot == original.METHOD
    finally:
        method.METHOD['reference_points'].pop()
    assert method.development_gates is original.development_gates
    assert method.validate_inputs is original.validate_inputs


def model_and_tokenizer():
    parameter = SimpleNamespace(numel=lambda: 149606402, is_meta=False,
                                device=SimpleNamespace(type='cpu'), dtype=torch.float32)
    model = SimpleNamespace(config=SimpleNamespace(model_type='modernbert', num_labels=2,
        classifier_pooling='mean', problem_type='multi_label_classification',
        id2label={0: 'LABEL_0', 1: 'LABEL_1'}), parameters=lambda: iter([parameter]))
    return model, SimpleNamespace(is_fast=True), parameter


def test_exact_architecture_contract_accepts_generic_labels_without_rewriting_saved_problem_type():
    model, tokenizer, _ = model_and_tokenizer()
    run.validate_candidate(model, tokenizer, {})
    assert model.config.problem_type == 'multi_label_classification'


@pytest.mark.parametrize('fault', ['missing', 'unexpected', 'mismatched', 'error', 'dtype', 'device',
                                 'meta', 'count', 'pooling', 'labels', 'width', 'slow_tokenizer'])
def test_incomplete_or_different_model_refused(fault):
    model, tokenizer, parameter = model_and_tokenizer()
    info = {}
    if fault in ('missing', 'unexpected', 'mismatched', 'error'):
        key = {'missing': 'missing_keys', 'unexpected': 'unexpected_keys',
               'mismatched': 'mismatched_keys', 'error': 'error_msgs'}[fault]
        info[key] = ['failed tensor']
    elif fault == 'dtype':
        parameter.dtype = torch.float16
    elif fault == 'device':
        parameter.device.type = 'mps'
    elif fault == 'meta':
        parameter.is_meta = True
    elif fault == 'count':
        parameter.numel = lambda: 1
    elif fault == 'pooling':
        model.config.classifier_pooling = 'cls'
    elif fault == 'labels':
        model.config.id2label = {0: 'ai', 1: 'human'}
    elif fault == 'width':
        model.config.num_labels = 1
    else:
        tokenizer.is_fast = False
    with pytest.raises(ValueError):
        run.validate_candidate(model, tokenizer, info)


@pytest.mark.parametrize('command', [
    ['python', '-m', 'scripts.repeat_audio_separation_v14', 'worker'],
    ['python', '/example/backend/scripts/repeat_audio_separation_v14.py', 'worker'],
])
def test_audio_evaluation_must_finish_before_either_writing_model_loads(monkeypatch, command):
    import psutil

    other = SimpleNamespace(pid=os.getpid() + 100000,
        info={'cmdline': command})
    monkeypatch.setattr(psutil, 'process_iter', lambda _: iter([other]))
    with pytest.raises(ValueError, match='sequentially'):
        run.load_model('candidate', {})
    monkeypatch.setattr(psutil, 'process_iter', lambda _: iter([]))
    run.require_sequential_evaluation()


def test_candidate_direct_audit_uses_positive_index1_and_token_weights_without_temperature():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.tensor([[0, 4, 5, 2], [0, 6, 2, 1]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
                'special_tokens_mask': torch.tensor([[1, 0, 0, 1], [1, 0, 1, 1]]),
                'overflow_to_sample_mapping': torch.tensor([0, 0])}

    class Model:
        def __call__(self, **encoded):
            values = [0., 0.] if encoded['input_ids'][0, 1] == 4 else [-100., 100.]
            return SimpleNamespace(logits=torch.tensor([values]))

    classifier = SimpleNamespace(tok=Tokenizer(), model=Model())
    assert run.direct_response(classifier, 'candidate', 'synthetic') == (0.5 * 2 + 1.) / 3


def synthetic_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'implementation', lambda: {'code': 'fixed'})
    monkeypatch.setattr(run, 'installed', lambda: {'installed': 'fixed'})
    monkeypatch.setattr(run, 'runtime', lambda: {'runtime': 'fixed'})
    body = {'schema': 'chipmunk.writing-modernbert-synthetic-runtime/4',
        'candidate_assets': copy.deepcopy(method.CANDIDATE_ASSETS), 'implementation_sha256': run.implementation(),
        'installed_sha256': run.installed(), 'runtime': run.runtime(),
        'synthetic_input_sha256': [method.text_sha(t) for t in run.SYNTHETIC_TEXTS],
        'strict_state_load': True, 'repeat_exact': True, 'project_text_forwards': 0, 'serving_admitted': False,
        'responses': [0.3, 0.4, 0.5], 'max_single_batch_difference': 0., 'max_direct_difference': 0.,
        'coverage': [{'tokens': n, 'token_windows': w, 'token_ids_sha256': 'a' * 64} for n, w in [(10, 1), (12, 1), (900, 2)]]}
    path = tmp_path / 'smoke.json'
    path.write_text(json.dumps(body))
    return body, path


def test_complete_runtime_receipt_is_required_before_freezing_project_inputs(tmp_path, monkeypatch):
    body, path = synthetic_receipt(tmp_path, monkeypatch)
    assert run.validate_smoke(path) == body
    body['strict_state_load'] = False
    path.write_text(json.dumps(body))
    monkeypatch.setattr(run, 'read_inputs', lambda _: ['private input read, no classifier'])
    monkeypatch.setattr(run, 'assets', lambda *_: pytest.fail('Model assets reached before required runtime evidence'))
    private, public = tmp_path / 'lock.json', tmp_path / 'public.json'
    with pytest.raises(ValueError, match='synthetic runtime'):
        run.freeze(tmp_path, tmp_path, tmp_path, tmp_path, private, public, path)
    assert not private.exists() and not public.exists()


@pytest.mark.parametrize('fault', ['code', 'runtime', 'installed', 'assets', 'input', 'repeat', 'project',
                                 'admit', 'missing_coverage', 'no_overflow', 'invalid_response', 'large_difference', 'nan'])
def test_invalid_synthetic_runtime_cannot_unlock_real_text(tmp_path, monkeypatch, fault):
    body, path = synthetic_receipt(tmp_path, monkeypatch)
    if fault in ('code', 'runtime', 'installed', 'assets'):
        key = {'code': 'implementation_sha256', 'runtime': 'runtime', 'installed': 'installed_sha256',
               'assets': 'candidate_assets'}[fault]
        body[key] = {}
    elif fault == 'input':
        body['synthetic_input_sha256'].reverse()
    elif fault == 'repeat':
        body['repeat_exact'] = False
    elif fault == 'project':
        body['project_text_forwards'] = 1
    elif fault == 'admit':
        body['serving_admitted'] = True
    elif fault == 'missing_coverage':
        body['coverage'].pop()
    elif fault == 'no_overflow':
        body['coverage'][-1]['token_windows'] = 1
    elif fault == 'invalid_response':
        body['responses'][0] = True
    elif fault == 'large_difference':
        body['max_direct_difference'] = 1e-4
    else:
        body['max_single_batch_difference'] = float('nan')
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        run.validate_smoke(path)
