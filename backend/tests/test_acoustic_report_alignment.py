from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import subprocess
import sys

import numpy as np
import pytest
import test_audio_evidence
from test_audio_evidence import audio_fixture as audio_fixture

from app.core.canonical import canonical_json
from app.detectors.audio.decode import wav_bytes
from app.evidence import audio_packets, service
from app.evidence.audio_background_binding import validate_acoustic_context
from app.reports import engine, multimedia_evidence
from app.reports.specimens import specimen
from app.storage.objects import ObjectIntegrityError


@pytest.fixture
def acoustic_report_case(request, monkeypatch):
    def source_file(path):
        if getattr(request, 'param', None) == 'spectral':
            time = np.arange(4 * 16000) / 16000
            pcm = np.concatenate([.15 * np.sin(2 * np.pi * 300 * time), .15 * np.sin(2 * np.pi * 3000 * time)])
        else:
            pcm = np.random.default_rng(209).normal(size=8 * 16000) * .002
            pcm[4 * 16000:] *= 12
        path.write_bytes(wav_bytes(pcm.astype(np.float32)))

    monkeypatch.setattr(test_audio_evidence, '_write_source', source_file)
    db, store, analysis, result, *_ = request.getfixturevalue('audio_fixture')
    document = audio_packets.build_packet_set(db, analysis, store, result)
    old = specimen('audio', 'standard')
    frozen = old.model_copy(update={
        'result': result, 'evidence': document,
        'analysis': old.analysis.model_copy(update={'id': analysis.id, 'revision': analysis.revision}),
        'media': old.media.model_copy(update={'sha256': analysis.sha256})})
    return frozen, store


@pytest.mark.parametrize('acoustic_report_case', ['spectral'], indirect=True)
def test_continuity_report_preserves_source_locations_and_public_title(acoustic_report_case):
    frozen, store = acoustic_report_case
    event, = [e for e in frozen.evidence['acoustic_context']['events'] if e['category'] == 'spectral_transition']
    view = multimedia_evidence.view(frozen, store=store)
    row = next(p for p in view['packets'] if p['id'] == event['id'])
    assert row['title'] == 'Possible sound-content transition'
    assert row['audio']['interval'] == event['interval'] == {'start_sample': 56000, 'end_sample': 72000}
    assert row['audio']['source_ranges'] == event['source_timestamp_ranges']
    assert row['measurements'] == event['observations']
    html = engine._env.get_template('sections/multimedia_evidence.html.j2').render(v={'evidence': view}, s=frozen)
    assert 'Possible sound-content transition' in html and 'Ordinary music' in html
    assert row['json_pointer'] in html and event['artifacts']['selected_clip']['sha256'] in html
    assert 'audioldm' not in html.lower() and 'aasist' not in html.lower()


def test_report_events_link_to_exact_json_samples_source_clock_and_pngs(acoustic_report_case):
    frozen, store = acoustic_report_case
    doc = frozen.evidence
    context = doc['acoustic_context']
    assert context['events'] and context['events'][0]['category'] == 'energy_change'
    view = multimedia_evidence.view(frozen, store=store)
    html = engine._env.get_template('sections/multimedia_evidence.html.j2').render(v={'evidence': view}, s=frozen)
    assert 'Background and acoustic context' in html and 'Evidence locator' in html
    assert 'aud.gustking_xlsr' not in html and 'gustking' not in json.dumps(doc)
    for index, event in enumerate(context['events']):
        row = next(p for p in view['packets'] if p['id'] == event['id'])
        assert row['kind'] == 'acoustic'
        assert row['json_pointer'] == f'/acoustic_context/events/{index}'
        assert row['json_pointer'] in html
        assert row['audio']['interval'] == event['interval']
        assert row['audio']['source_ranges'] == event['source_timestamp_ranges']
        assert row['related_voice_packet_ids'] == event['related_voice_packet_ids']
        assert f"samples [{event['interval']['start_sample']}, {event['interval']['end_sample']})" in row['locator']
        assert all(image['label'].startswith(f"{event['id']} - {row['locator']} - ") for image in row['images'])
        assert all(image['label'] in html for image in row['images'])
        assert event['artifacts']['selected_clip']['sha256'] in html
    embedded = {hashlib.sha256(base64.b64decode(x)).hexdigest()
                for x in re.findall(r'data:image/png;base64,([A-Za-z0-9+/=]+)', html)}
    declared = {a['sha256'] for a in service.asset_refs(doc).values() if a['media_type'] == 'image/png'}
    assert embedded == declared
    manifest = multimedia_evidence.manifest(frozen)
    assert {a['sha256'] for a in manifest['artifacts']} == set(service.asset_refs(doc))


def test_redacted_report_keeps_time_links_but_withholds_acoustic_pixels(acoustic_report_case):
    frozen, store = acoustic_report_case
    frozen = frozen.model_copy(update={'report': frozen.report.model_copy(update={'redacted': True})})
    view = multimedia_evidence.view(frozen, store=store)
    html = engine._env.get_template('sections/multimedia_evidence.html.j2').render(v={'evidence': view}, s=frozen)
    assert '/acoustic_context/events/0' in html and 'Measured acoustic interval' in html
    assert 'data:image/png;base64' not in html


def test_report_capture_recomputes_descriptive_measurements_after_rehash(acoustic_report_case):
    frozen, store = acoustic_report_case
    altered = copy.deepcopy(frozen.evidence)
    # This remains structurally plausible and correctly rehashed, but it is not
    # the measured frequency content of the frozen source samples.
    altered['acoustic_context']['bins'][0]['spectral_flatness'] = .123456
    altered['sha256'] = service.packet_hash(altered)
    with pytest.raises(ObjectIntegrityError, match='acoustic context'):
        multimedia_evidence.view(frozen.model_copy(update={'evidence': altered}), store=store)


@pytest.mark.parametrize('change', ['interval', 'clock', 'reference', 'missing_plot', 'nonfinite', 'source'])
def test_structural_validator_rejects_misleading_acoustic_locations(acoustic_report_case, change):
    frozen, _ = acoustic_report_case
    source = frozen.evidence['source']
    context = copy.deepcopy(frozen.evidence['acoustic_context'])
    event = context['events'][0]
    if change == 'interval':
        event['interval']['start_sample'] += 1
    elif change == 'clock':
        event['source_timestamp_ranges'][0]['source_start_pts'] += 1
    elif change == 'reference':
        event['related_voice_packet_ids'] = ['A99']
    elif change == 'missing_plot':
        context['overview'].pop('spectrogram')
    elif change == 'nonfinite':
        context['bins'][0]['rms_dbfs'] = float('nan')
    else:
        context['source']['pcm_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        validate_acoustic_context(context, source=source, voice_packets=frozen.evidence['packets'])


def test_acoustic_validation_is_offline_without_application_or_numeric_runtime(acoustic_report_case, tmp_path):
    frozen, _ = acoustic_report_case
    path = tmp_path / 'frozen-context.json'
    path.write_bytes(canonical_json(frozen.evidence))
    script = '''
import importlib.abc,json,socket,sys
class RuntimeUnavailable(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('app.evidence','app.evidence.audio_background_binding','app.evidence.audio_bandwidth_binding'):
            return None
        if fullname.startswith(('app.storage','app.core.config','app.detectors','app.evidence','app.api','numpy','PIL')):
            raise RuntimeError('Offline validation imported runtime: '+fullname)
sys.meta_path.insert(0,RuntimeUnavailable())
def no_network(*a,**k): raise RuntimeError('network forbidden')
socket.socket=no_network
from app.evidence.audio_background_binding import validate_acoustic_context
with open(sys.argv[1]) as stream: doc=json.load(stream)
validate_acoustic_context(doc['acoustic_context'],source=doc['source'],voice_packets=doc['packets'])
'''
    result = subprocess.run([sys.executable, '-c', script, str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
