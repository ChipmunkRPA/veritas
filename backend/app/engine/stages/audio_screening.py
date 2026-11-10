"""Local candidate speech decision; never uses the incumbent fusion coefficients."""
from __future__ import annotations

import math

from app.detectors.audio import calibration as current_calibration
from app.engine.fusion import decisive
from app.engine.stages import common, video
from app.schemas.evidence import Decision, SignalV2
from app.storage.artefacts import register_artefact
from app.storage.db import FuserRow


def fuse(ctx, *, configuration=None):
    calibration = current_calibration if configuration is None else configuration
    calibration.load()  # fail closed if the pinned calibration is missing or changed
    signals = video.synthesis_signals(ctx, 1)
    primary_id = getattr(calibration, 'DETECTOR_ID', 'aud.pella_v2')
    # Retain comparator runs in the detection record. Publish only the selected
    # speech calculation so one grouped finding cannot borrow another map.
    signals = [s for s in signals if s.get('family') != 'voice' or s['detector_id'] == primary_id]
    video.attach_temporal(ctx, signals)
    primary = next((s for s in signals if s['detector_id'] == primary_id), None)
    screen = (primary or {}).get('details', {}).get('audio_screening', {})
    if primary and not screen:
        raise ValueError('The saved speech response has no candidate calibration; run a fresh analysis in candidate mode.')
    if screen and screen.get('calibration_sha256') != calibration.SHA256:
        raise ValueError('The saved voice response uses a different confidence calibration.')
    p = screen.get('probability_ai')
    direction = screen.get('decision', 'inconclusive')
    reason = screen.get('reason', 'No usable speech response was available; no authenticity conclusion can be made.')
    from app.detectors.audio.reference import COMBINED_SHA256, combined_tendency

    if calibration.SHA256 == COMBINED_SHA256:
        p, direction = combined_tendency(screen)
        reason = {'ai': 'Leaning towards AI-generated speech.', 'original': 'Leaning towards original speech.'}.get(
            direction, 'The combined speech evidence does not support a usable directional conclusion in its source window.')
        if primary:
            primary['details']['audio_screening'] = {**screen, 'probability_ai': p, 'decision': direction,
                                                       'reason': reason, 'limitation': calibration.NOTE}
            primary.update(calibrated_probability=p, effective_weight=0.0, contribution_logit=None,
                           contribution_pp=None, log_lr=None)
    code = {'ai': 'ai_supported', 'original': 'no_ai_indicators'}.get(direction, 'inconclusive' if primary else 'not_assessable')
    verbal = {'ai': 'Leaning towards AI-generated speech', 'original': 'Leaning towards original speech'}.get(direction, 'Inconclusive audio assessment')
    band = {'ai': 'likely_ai', 'original': 'likely_authentic'}.get(direction, 'uncertain')
    contributions = []
    if primary and p is not None:
        eta = math.log(p) - math.log1p(-p)
        primary.update(calibrated_probability=p, calibration_ref=calibration.SHA256,
                       contribution_logit=eta, contribution_pp=100 * (p - .5), effective_weight=1.0,
                       outside_estimate=None, log_lr=eta)
        contributions = [{'kind': 'neutral_start', 'label': 'Balanced reference prior', 'logit': 0., 'pp': 50.},
                         {'kind': 'detector', 'label': 'Calibrated speech check', 'signal_id': primary['id'],
                          'logit': eta, 'pp': 100 * (p - .5), 'baseline': 'p50', 'counted': True}]
    decision = Decision(code=code, verbal=verbal, probability_ai_calibrated=p, fusion_mode='artefact',
                        fuser_sha256=calibration.SHA256, band=band, confidence='medium' if direction != 'inconclusive' else 'low',
                        driving_hypothesis='H2_localised_manipulation', reason=reason,
                        contributions=contributions).model_dump(mode='json')
    caveats = [calibration.NOTE, getattr(calibration, 'EXTRA_CAVEAT', 'This is a local experimental release. The stricter research admission criteria were not all met.')]
    entries = common.entries(ctx, ('provenance',))
    overrides = [e for e in entries if e['status'] == 'ok' and e.get('provenance_override')]
    override = decisive.resolve(e['provenance_override'] for e in overrides)
    if override:
        source_ids = []
        for entry in overrides:
            if entry['provenance_override'] != override:
                continue
            sid = common.signal_id(entry['detector_id'], entry.get('aggregate'))
            source_ids.append(sid)
            signals.append(SignalV2(id=sid, display_id=f'S{len(signals)+1:02d}', detector_id=entry['detector_id'],
                detector_version=entry['version'], name=entry['name'], family='provenance', modality='audio', category='provenance',
                hypothesis='H3_provenance_integrity', provenance_override=override, observed=entry['observed'],
                reference=entry['reference'], reasoning=entry['reasoning'], model_revisions=entry['model_revisions'],
                exhibit_ids=entry['exhibit_ids']).model_dump(mode='json'))
        marker = decisive.marker_kind(overrides)
        label = decisive.marker_verbal(marker, 'audio') if override == 'self_declared_ai' else decisive.OVERRIDE_LABELS[override]
        decision.update(code='no_ai_indicators' if override == 'trusted_capture' else 'ai_supported', verbal=label,
                        reason=label, driving_hypothesis='H3_provenance_integrity',
                        band='likely_authentic' if override == 'trusted_capture' else 'likely_ai',
                        confidence='low' if override == 'self_declared_ai' else 'high',
                        statistical_estimate_before_provenance=p,
                        statistical_alone={'code': code} if p is not None else None,
                        provenance_record=decisive.record(override, overrides, source_ids))
        if override == 'self_declared_ai':
            caveats.append(decisive.marker_caveat(marker, 'audio'))
    # No new confidence interval or general-population accuracy claim is fabricated.
    profile = {'slice_id': getattr(calibration, 'SLICE_ID', 'audio/local-speech-v3'), 'duration_s': ctx.outputs['probe']['facts'].get('duration_s'),
               'within_validated_envelope': False, 'validation_support': 'unsupported',
               'limiting_conditions': [calibration.NOTE]}
    with ctx.sessions() as db:
        payload = calibration.report_payload() if hasattr(calibration, 'report_payload') else calibration.PATH.read_bytes()
        artifact = register_artefact(db, ctx.store, payload, media_type='application/json',
                                    role='stage_output', created_by='audio-confidence', analysis_id=ctx.analysis_id)
        if db.get(FuserRow, calibration.SHA256) is None:
            db.add(FuserRow(sha256=calibration.SHA256, modality='audio', version=calibration.VERSION,
                            bench_version=getattr(calibration, 'BENCH_VERSION', 'audio-v3')))
        db.commit()
        artifact_ref = {'id': artifact.id, 'sha256': artifact.sha256, 'size_bytes': artifact.bytes,
                        'media_type': artifact.media_type, 'role': artifact.role, 'created_by': artifact.created_by}
    verdict = {'probability_ai': p, 'band': decision['band'], 'confidence': decision['confidence'], 'label': decision['verbal'],
               'summary': decision['verbal'], 'reasoning': [decision['reason']], 'caveats': caveats,
               'signals_for_ai': int(direction == 'ai'), 'signals_for_authentic': int(direction == 'original'),
               'signals_total': int(primary is not None)}
    assessments = [{'hypothesis': 'H1_whole_content_synthesis', 'applicable': False,
                    'note': 'The speech assessment does not establish whether every sound or edit is original.'},
                   {'hypothesis': 'H2_localised_manipulation', 'applicable': p is not None,
                    'probability': p, 'note': reason + ' ' + calibration.NOTE}]
    return {'decision': decision, 'signals': signals, 'assessments': assessments,
            'condition_profile': profile, 'caveats': caveats,
            'calibration': [{'subject': 'audio-local-screening', 'method': getattr(calibration, 'METHOD', 'platt'), 'artefact_sha256': calibration.SHA256,
                             'bench_version': getattr(calibration, 'BENCH_VERSION', 'audio-v3')}],
            'fusion': {'mode': 'artefact', 'version': calibration.VERSION, 'sha256': calibration.SHA256},
            'v1': {'verdict': verdict, 'signals': [], 'audio': ctx.outputs['probe']['facts'], 'artifacts': [], 'timings': {}},
            'artefacts': [artifact_ref]}
