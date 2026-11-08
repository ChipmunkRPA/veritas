"""Only verified documentary facts may prevent model acquisition."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.engine import provenance_gate as gate
from app.engine.detector import Modality, Profile
from app.engine.stages import common


def evidence(kind='image'):
    return {'detectors': [{'detector_id': gate.CREDENTIALS[kind], 'status': 'ok',
        'provenance_override': 'decisive_ai', 'metrics': {'trusted': 1., 'valid': 1., 'ai_declared': 1.,
            'covers_bytes': 1., 'invalid': 0., 'remote_reference': 0., 'failure_codes': 0.}}]}


@pytest.mark.parametrize('kind', ['image', 'video', 'audio'])
def test_asset_bound_trusted_ai_declaration_can_stop_statistical_detection(kind):
    got = gate.decide(evidence(kind), kind)
    assert got['action'] == 'stop_statistical_detection'
    assert got['basis'] == 'trusted_asset_bound_ai_declaration'
    assert got['external_provider_verification'] == 'not_run'


@pytest.mark.parametrize('damage', ['untrusted', 'invalid', 'unbound', 'remote', 'capture', 'missing-metric',
    'string', 'boolean', 'nan', 'failed', 'wrong-detector', 'duplicate', 'unsigned'])
def test_ambiguous_or_forgeable_markers_never_skip_models(damage):
    doc = evidence()
    row = doc['detectors'][0]
    if damage == 'untrusted':
        row['metrics']['trusted'] = 0.
    elif damage == 'invalid':
        row['metrics']['failure_codes'] = 1.
    elif damage == 'unbound':
        row['metrics']['covers_bytes'] = 0.
    elif damage == 'remote':
        row['metrics']['remote_reference'] = 1.
    elif damage == 'capture':
        row['provenance_override'] = 'trusted_capture'
    elif damage == 'missing-metric':
        row['metrics'].pop('covers_bytes')
    elif damage == 'string':
        row['metrics']['trusted'] = '1'
    elif damage == 'boolean':
        row['metrics']['trusted'] = True
    elif damage == 'nan':
        row['metrics']['trusted'] = float('nan')
    elif damage == 'failed':
        row['status'] = 'failed'
    elif damage == 'wrong-detector':
        row['detector_id'] = 'img.meta.generator'
    elif damage == 'duplicate':
        doc['detectors'].append(deepcopy(row))
    else:
        row['provenance_override'] = 'self_declared_ai'
    assert gate.decide(doc, 'image')['action'] == 'continue_detection'


@pytest.mark.parametrize('doc', [None, {}, {'detectors': []}, {'detectors': [{'status': 'not_applicable'}]}])
def test_absent_unsupported_checks_never_become_original_verdicts(doc):
    got = gate.decide(doc, 'image')
    assert got['action'] == 'continue_detection'
    assert got['basis'] is None


def test_whitespace_and_provider_names_cannot_authenticate_text():
    assert gate.decide({'text': 'Some\u200b text\u00a0  OpenAI Anthropic Grok Gemini Muse'}, 'document')['action'] == 'continue_detection'


def context():
    outputs = {'ingest': {'sha256': 'a' * 64}, 'provenance': evidence()}
    ctx = SimpleNamespace(outputs=outputs, state={'kind': 'image'})
    outputs['provenance_gate'] = gate.run(ctx)
    return ctx


def test_cached_gate_is_bound_to_current_source_and_reverified_evidence():
    ctx = context()
    assert gate.stop(ctx)
    ctx.outputs['ingest']['sha256'] = 'b' * 64
    assert not gate.stop(ctx)
    ctx = context()
    ctx.outputs['provenance']['detectors'][0]['metrics']['trusted'] = 0.
    assert not gate.stop(ctx)


def test_refusion_does_not_erase_historical_checks():
    ctx = context()
    ctx.state['refusion'] = True
    ctx.outputs['provenance_gate'] = gate.run(ctx)
    assert not gate.stop(ctx)
    assert 'Historical' in ctx.outputs['provenance_gate']['reason']


def test_validated_gate_records_skip_without_running_or_acquiring_a_model(monkeypatch):
    from app.detectors.image.adapters.provenance import C2PADetector

    ctx = context()
    spec = replace(C2PADetector.spec, id='content.example', family='learned-whole-image',
        requirements=replace(C2PADetector.spec.requirements, model_ids=('must-never-load',)))
    detector = SimpleNamespace(spec=spec)
    registry = SimpleNamespace(enabled=lambda *_: ['content.example'], instance=lambda _: detector)
    run_store = SimpleNamespace(run_id=lambda *_: 'saved-skip')
    ctx.state.update(registry=registry, run_store=run_store, media=object(), models=object())
    ctx.analysis_id, ctx.stage_id, ctx.profile = 'analysis', 'detect', Profile.FULL
    ctx.check, ctx.emit = Mock(), Mock()
    ctx.deadline = None
    ctx.log = SimpleNamespace(bind=lambda **_: None)
    ctx.exhibits = SimpleNamespace(bind=lambda _: None, exhibits=[], artefacts=[])
    monkeypatch.setattr(common, 'run_device', lambda _: 'cpu')
    calls = []
    monkeypatch.setattr(common, 'store_run', lambda _, record, *__: calls.append(record) or record)
    monkeypatch.setattr(common, 'run_detector', Mock(side_effect=AssertionError('Model boundary must not be reached')))
    out = common.run_detectors(ctx, (Modality.IMAGE,))
    common.run_detector.assert_not_called()
    assert len(calls) == 1 and calls[0].result.status == 'skipped'
    assert out['detectors'][0]['reason'] == gate.SKIP_REASON
    assert out['detectors'][0]['aggregate'] is None


def test_every_model_stage_waits_for_the_provenance_gate():
    from app.engine.pipelines import PIPELINES
    from app.engine.runner import toposort

    for stages in PIPELINES.values():
        assert toposort(stages)
        by_id = {s.id: s for s in stages}
        for name in ('faces', 'tracks', 'detect'):
            if name in by_id:
                assert 'provenance_gate' in by_id[name].after
        assert 'provenance_gate' in by_id


def test_audio_credentials_adapter_has_no_model_and_rejects_unsupported_container():
    from app.detectors.audio.provenance_adapter import AudioCredentialsDetector
    from app.engine.registry import get_registry

    adapter = get_registry().instance('prov.c2pa.audio')
    assert isinstance(adapter, AudioCredentialsDetector)
    assert adapter.spec.requirements.model_ids == () and adapter.spec.modality == Modality.AUDIO
    got = adapter.run(SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: b'ID3unsupported-test')))
    assert got.status == 'not_applicable' and 'not supported' in got.reason
