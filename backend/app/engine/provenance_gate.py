"""Stop statistical checks only for a trusted, asset-bound AI declaration.

Unsigned metadata, public watermark messages, text spacing and absent signals
cannot activate this gate. Provider-private watermark APIs are not simulated.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VERSION = 'provenance-first/1'
CREDENTIALS = {'image': 'prov.c2pa', 'video': 'prov.c2pa.video', 'audio': 'prov.c2pa.audio', 'document': 'prov.c2pa.pdf'}
SKIP_REASON = ('Further detection was not run: trusted Content Credentials bound to this file declare AI involvement. '
    'The record does not locate edited regions or establish the complete history of the file.')


def decide(provenance: Any, kind: str) -> dict:
    expected = CREDENTIALS.get(kind)
    result = {'version': VERSION, 'action': 'continue_detection', 'basis': None,
        'scope': 'AI involvement declared by signed provenance; not pixel, speaker or passage authorship.',
        'external_provider_verification': 'not_run', 'provider_private_watermarks_checked': False}
    if expected is None:
        return {**result, 'reason': 'No applicable local Content Credentials check; continue statistical detection.'}
    entries = provenance.get('detectors', []) if isinstance(provenance, Mapping) else []
    if not isinstance(entries, (list, tuple)):
        return {**result, 'reason': 'Content Credentials output is unavailable; continue statistical detection.'}
    matching = [e for e in entries if isinstance(e, Mapping) and e.get('detector_id') == expected]
    if len(matching) != 1:
        return {**result, 'reason': 'No single usable Content Credentials check; continue statistical detection.'}
    entry = matching[0]
    metrics = entry.get('metrics')
    if not isinstance(metrics, Mapping):
        return {**result, 'reason': 'Content Credentials verification is incomplete; continue statistical detection.'}
    # Compare actual numeric verification facts, never truthy strings or a name
    # copied out of the uploaded file. Every required fact must be present.
    def one(name):
        value = metrics.get(name)
        return type(value) in (int, float) and value == 1.

    def zero(name):
        value = metrics.get(name)
        return type(value) in (int, float) and value == 0.

    verified = (entry.get('status') == 'ok' and entry.get('provenance_override') == 'decisive_ai'
        and all(one(k) for k in ('trusted', 'valid', 'ai_declared', 'covers_bytes'))
        and all(zero(k) for k in ('invalid', 'remote_reference', 'failure_codes'))
        and (kind != 'document' or one('pdf_document_binding')))
    if not verified:
        return {**result, 'reason': 'No trusted, valid, asset-bound AI declaration; continue statistical detection. Absence is not proof of originality.'}
    return {**result, 'action': 'stop_statistical_detection', 'basis': 'trusted_asset_bound_ai_declaration',
        'record_detector_id': expected, 'reason': SKIP_REASON,
        'limitation': 'Offline trust-list validation; revocation and remote manifests are not checked.'}


def run(ctx):
    result = decide(ctx.outputs.get('provenance'), ctx.state.get('kind', ''))
    if ctx.state.get('refusion'):
        result = {**result, 'action': 'continue_detection',
            'reason': 'Historical detector evidence is reused; this gate does not erase earlier checks.'}
    return {**result, 'source_sha256': (ctx.outputs.get('ingest') or {}).get('sha256')}


def stop(ctx) -> bool:
    gate = ctx.outputs.get('provenance_gate') or {}
    if not isinstance(gate, Mapping):
        return False
    # Recompute from verified stage evidence as well, rather than accepting an
    # arbitrary cached gate flag without its original documentary basis.
    fresh = decide(ctx.outputs.get('provenance'), ctx.state.get('kind', ''))
    return (gate.get('version') == VERSION and gate.get('action') == fresh['action'] == 'stop_statistical_detection'
        and gate.get('source_sha256') == (ctx.outputs.get('ingest') or {}).get('sha256')
        and bool(gate.get('source_sha256')))


def documentary_fields(entry: dict, gate: dict, *, modality: str, reason: str):
    """Build the decision and its source-bound signal without assigning an AI probability."""
    from app.engine.fusion import decisive
    from app.schemas.evidence import Decision, SignalV2

    sid = entry['detector_id'] + ':global'
    signal = SignalV2(id=sid, display_id='S01', detector_id=entry['detector_id'], detector_version=entry['version'],
        name=entry['name'], family='provenance', modality=modality, category='provenance', hypothesis='H3_provenance_integrity',
        status='ok', scope='global', raw_score=entry['aggregate']['raw'], raw_kind=entry['aggregate']['raw_kind'],
        effective_weight=0., observed=entry['observed'], reference=entry['reference'], reasoning=entry['reasoning'],
        metrics=entry['metrics'], details={'provenance_gate': gate},
        exhibit_ids=entry['exhibit_ids'], model_revisions={}, provenance_override='decisive_ai').model_dump(mode='json')
    decision = Decision(code='ai_supported', verbal='Signed Content Credentials declare AI involvement', band='likely_ai',
        confidence='high', driving_hypothesis='H3_provenance_integrity', fusion_mode='fallback', reason=reason,
        provenance_record=decisive.record('decisive_ai', [entry], [sid])).model_dump(mode='json')
    return decision, signal


def audio_documentary_result(ctx):
    """A signed audio declaration decides without loading a speech calibration.

    This path also covers the legacy speech mode, whose statistical fuser did
    not previously support standalone audio Content Credentials.
    """
    if not stop(ctx) or ctx.state.get('kind') != 'audio':
        raise ValueError('A verified asset-bound audio declaration is required')
    entry = next(e for e in ctx.outputs['provenance']['detectors'] if e['detector_id'] == CREDENTIALS['audio'])
    reason = ('The trusted signed record bound to this recording declares AI generation or AI editing. '
        'Statistical voice and background checks were not run. This does not locate every edit or establish the complete recording history.')
    decision, signal = documentary_fields(entry, dict(ctx.outputs['provenance_gate']), modality='audio', reason=reason)
    caveats = [SKIP_REASON, ctx.outputs['provenance_gate']['limitation'],
        'This is documentary evidence of the signer\'s declaration, not a statistical probability or proof of every sound\'s origin.']
    verdict = {'probability_ai': None, 'band': 'likely_ai', 'confidence': 'high', 'label': decision['verbal'], 'summary': reason,
        'reasoning': [reason], 'caveats': caveats, 'signals_for_ai': 1, 'signals_for_authentic': 0, 'signals_total': 1}
    return {'decision': decision, 'signals': [signal], 'assessments': [
        {'hypothesis': 'H1_whole_content_synthesis', 'applicable': False, 'note': 'Statistical checks were not run.'},
        {'hypothesis': 'H2_localised_manipulation', 'applicable': False, 'note': 'No voice or background authenticity estimate was formed.'},
        {'hypothesis': 'H3_provenance_integrity', 'applicable': True, 'note': reason}],
        'condition_profile': {**((ctx.outputs.get('condition') or {}).get('profile') or {}),
            'within_validated_envelope': False, 'validation_support': 'unsupported'},
        'caveats': caveats, 'calibration': [],
        'fusion': {'mode': 'fallback', 'reason': 'Signed documentary record; statistical fusion bypassed.'},
        'v1': {'verdict': verdict, 'signals': [], 'audio': ctx.outputs['probe']['facts'], 'artifacts': [], 'timings': {}},
        'artefacts': []}
