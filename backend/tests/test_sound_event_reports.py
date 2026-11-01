from copy import deepcopy

import pytest
from test_audio_evidence import audio_fixture as audio_fixture

from app.core.canonical import canonical_hash, canonical_json
from app.evidence import audio_packets, service
from app.reports import multimedia_evidence
from app.reports.customer_view import customer_view
from app.reports.specimens import specimen
from app.storage.objects import ObjectIntegrityError


@pytest.fixture
def sound_report(audio_fixture):
    db, store, analysis, result, pcm, _ = audio_fixture
    record = result['manifest']['sampling']['audio_evidence']
    entry = {'detector_id': 'aud.sound_events', 'status': 'ok', 'details': {
        'audio_input': record['input'], 'sound_windows': [{'start_sample': 0, 'end_sample': len(pcm),
            'labels': [{'label': 'Speech', 'score': .83}, {'label': 'Music', 'score': .12}]}]}}
    raw = store.put_bytes(canonical_json(entry))
    record['sound_checks'] = [{'detector_id': 'aud.sound_events', 'raw_output_sha256': raw.sha256}]
    analysis.result_hash = store.put_bytes(canonical_json(result)).sha256
    document = audio_packets.build_packet_set(db, analysis, store, result)
    s = specimen('audio', 'standard')
    full = deepcopy(s.result)
    full['input']['sha256'] = analysis.sha256
    full['manifest']['sampling'] = deepcopy(result['manifest']['sampling'])
    document.update(analysis_id=s.analysis.id, revision=s.analysis.revision, original_result_hash=canonical_hash(full))
    document['sha256'] = service.packet_hash(document)
    s = s.model_copy(update={'result': full, 'evidence': document,
                            'media': s.media.model_copy(update={'sha256': analysis.sha256})})
    return s, store


def test_sound_category_report_uses_exact_frozen_windows_and_commits_all_assets(sound_report):
    s, store = sound_report
    before = canonical_hash(s.model_dump(mode='python'))
    shown = multimedia_evidence.view(s, store=store)
    c = next(p for p in shown['packets'] if p['id'] == 'C01')
    assert c['kind'] == 'sound' and c['json_pointer'] == '/sound_events/windows/0'
    assert c['locator'] == 'PCM 0.000-8.000 s; samples [0, 128000)'
    assert c['observation'] == 'Speech: 0.830; Music: 0.120'
    assert len(c['images']) == 1 and len(c['assets']) == 2
    assert {a['sha256'] for a in c['assets']} <= set(service.asset_refs(s.evidence))
    assert {a['sha256'] for a in c['assets']} <= {a['sha256'] for a in multimedia_evidence.manifest(s)['artifacts']}
    public = customer_view(s, {'header_left': 'test', 'evidence': shown}, lambda x: x)
    assert 'No AI/original conclusion' in next(p for p in public['evidence']['packets'] if p['id'] == 'C01')['customer_inference']
    assert canonical_hash(s.model_dump(mode='python')) == before


@pytest.mark.parametrize('change', ['score', 'source_time', 'clip'])
def test_rehashed_sound_category_evidence_must_still_match_frozen_raw_output(sound_report, change):
    s, store = sound_report
    evidence = deepcopy(s.evidence)
    window = evidence['sound_events']['windows'][0]
    if change == 'score':
        window['labels'][0]['score'] = .99
    elif change == 'source_time':
        window['source_timestamp_ranges'][0]['source_start_pts'] += 1
    else:
        window['artifacts']['selected_clip']['sha256'] = evidence['source']['wav_sha256']
        # The full decoded WAV happens to be the same 8-second selection here;
        # force a different committed role rather than relying on byte inequality.
        window['artifacts']['selected_clip']['channels'] = 2
    evidence['sha256'] = service.packet_hash(evidence)
    with pytest.raises((ObjectIntegrityError, ValueError)):
        multimedia_evidence.verify_assets(evidence, store, result=s.result)
