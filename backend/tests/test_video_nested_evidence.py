"""Video v3 joins source-bound voice/AV evidence under one immutable parent."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import zipfile
from datetime import UTC, datetime

import pymupdf
import pytest
from test_video_av_evidence import av_fixture as av_fixture
from test_video_av_evidence import av_source as av_source

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.evidence import service, video_packets
from app.reports import engine, multimedia_evidence, package, signing, snapshot, verify
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import AnalysisStage, ReportRow
from app.storage.objects import ObjectIntegrityError


@pytest.fixture(params=['video-evidence-3', 'video-evidence-4'])
def nested_case(av_fixture, monkeypatch, request):
    f = av_fixture
    anchors = list(f.anchors.values())
    sampling = f.result['manifest']['sampling']
    sampling.update(frame_anchor_version=video_packets.ANCHOR_VERSION,
                    frame_anchor_sha256=canonical_hash(anchors), frame_table_sha256=canonical_hash(f.table.to_json()),
                    timeline_origin_s=5.)
    for stage, content in [('probe', {'frame_table': f.table.to_json(), 'info': {'has_audio': True}}),
                           ('render', {'frame_anchor_version': video_packets.ANCHOR_VERSION, 'frame_anchors': anchors})]:
        art = register_artefact(f.db, f.store, canonical_json(content), media_type='application/json', role='stage_output',
                               created_by='av-fixture', analysis_id=f.analysis.id)
        f.db.add(AnalysisStage(analysis_id=f.analysis.id, stage=stage, status='done', output_artefact_id=art.id))
    f.result['exhibits'] = []
    f.analysis.result_hash = canonical_hash(f.result)
    f.store.put_bytes(canonical_json(f.result))
    f.db.flush()
    # Pin this fixture's writer independently of future versions; older sets
    # remain readable and are never retroactively enriched.
    monkeypatch.setattr(video_packets, 'VERSION', request.param)
    monkeypatch.setattr(video_packets, 'CREATOR', 'video-evidence-packets/' + request.param.rsplit('-', 1)[1])
    if request.param == 'video-evidence-3':
        from app.evidence import audio_packets

        current_builder = audio_packets.build_content
        def historical_content(*args, **kwargs):
            content = current_builder(*args, **kwargs)
            content.pop('acoustic_context')
            return content
        monkeypatch.setattr(audio_packets, 'build_content', historical_content)
    f.document = video_packets.build_packet_set(f.db, f.analysis, f.store, f.result)
    frozen = specimen('video', 'standard')
    f.frozen = frozen.model_copy(update={
        'result': copy.deepcopy(f.result), 'evidence': copy.deepcopy(f.document),
        'analysis': frozen.analysis.model_copy(update={'id': f.analysis.id, 'revision': f.analysis.revision}),
        'media': frozen.media.model_copy(update={'sha256': f.analysis.sha256, 'size_bytes': f.analysis.size_bytes,
                                                'filename': f.analysis.filename, 'content_type': f.analysis.content_type}),
        'report': frozen.report.model_copy(update={'template_set': engine.TEMPLATE_SET})})
    return f


def test_parent_builder_capture_assets_and_manifest_are_one_set(nested_case, monkeypatch):
    f = nested_case
    assert f.document['version'] in {'video-evidence-3', 'video-evidence-4'}
    assert ('acoustic_context' in f.document['audio']) == (f.document['version'] == 'video-evidence-4')
    assert f.document['audio']['status'] == f.document['av']['status'] == 'ready'
    assert service.load_packet_set(f.db, f.analysis, f.store) == f.document
    assert multimedia_evidence.capture(f.db, f.analysis, result=f.result, store=f.store) == f.document
    assets = service.asset_refs(f.document)
    assert {a['media_type'] for a in assets.values()} == {'image/png', 'audio/wav', 'video/mp4'}
    manifest = multimedia_evidence.manifest(f.frozen)
    assert {a['sha256'] for a in manifest['artifacts']} == set(assets)
    assert all(f'/{f.document["id"]}/assets/' in a['url'] for a in assets.values())
    assert len(manifest['artifacts']) > len(assets)  # intentional repeated patch/clip bytes
    assert snapshot.dump(snapshot.load(snapshot.dump(f.frozen))) == snapshot.dump(f.frozen)
    monkeypatch.setattr(service, 'load_packet_set', lambda *a, **k: pytest.fail('render read mutable analysis'))
    multimedia_evidence.view(f.frozen, store=f.store)


@pytest.mark.parametrize('redacted', [False, True])
def test_nested_report_section_has_input_proof_and_withholds_source(nested_case, redacted):
    f = nested_case
    frozen = f.frozen.model_copy(update={'report': f.frozen.report.model_copy(update={'redacted': redacted})})
    view = multimedia_evidence.view(frozen, store=f.store)
    assert [p['kind'] for p in view['packets']] == ['audio', 'av', 'av', 'av']
    html = engine._env.get_template('sections/multimedia_evidence.html.j2').render(v={'evidence': view}, s=frozen)
    for required in ('Synthetic-voice inspection window', 'All 31 lag candidates', 'Source audio PTS', 'actual 32 by 24 inputs'):
        assert required in html
    assert ('<img src="data:image/png;base64,' in html) is not redacted
    assert 'aud.gustking_xlsr' not in html
    assert len(view['packets'][1]['measured_inputs']) == 25
    for row in view['packets']:
        for image in row['images']:
            assert image['label'].startswith(f"{row['id']} - {row['locator']} - ")
        if row['kind'] == 'av':
            for tick in row['measured_inputs']:
                assert tick['image']['label'].startswith(f"{row['id']} - tick {tick['tick']}; frame {tick['frame_index']}")


def _report_scaffold(f):
    """Labelled report specimen around real frozen source inputs, with compatible published check refs."""
    original = specimen('video','standard')
    result = copy.deepcopy(original.result)
    result['input']['sha256'] = f.analysis.sha256
    result['manifest']['sampling'] = copy.deepcopy(f.result['manifest']['sampling'])
    template = result['signals'][0]
    result['signals'] = [{**copy.deepcopy(template), **row, 'name':'Controlled engineering check'} for row in f.result['signals']]
    result['findings'] = []
    result['exhibits'] = []
    evidence = copy.deepcopy(f.document)
    evidence['original_result_hash'] = canonical_hash(result)
    evidence['sha256'] = service.packet_hash(evidence)
    return f.frozen.model_copy(update={'result':result,'evidence':evidence})


@pytest.mark.parametrize('profile', ['standard','forensic','summary','newsroom'])
def test_full_report_profiles_include_nested_evidence_without_internal_sampling(nested_case,monkeypatch,profile):
    f=nested_case
    frozen=_report_scaffold(f)
    frozen=frozen.model_copy(update={'report':frozen.report.model_copy(update={'profile':profile})})
    monkeypatch.setattr(multimedia_evidence,'default_store',lambda:f.store)
    html=engine.render_html(frozen,embed_fonts=False)
    assert 'All 31 lag candidates' in html
    assert 'video_audio_evidence' not in html and 'voice_checks' not in html and 'av_checks' not in html
    assert 'aud.gustking_xlsr' not in html
    assert ('<img src="data:image/png;base64,' in html) is (profile!='newsroom')


@pytest.mark.parametrize('part,change', [('audio','score'), ('audio','availability'), ('av','profile'), ('av','raw_run')])
def test_nested_capture_rejects_semantic_tampering_even_after_rehash(nested_case, part, change):
    f = nested_case
    altered = copy.deepcopy(f.document)
    if change == 'score':
        altered['audio']['packets'][0]['observations'][0]['value'] = .9
        altered['audio']['coverage']['windows'][0]['score'] = .9
    elif change == 'availability':
        from app.evidence.video_audio import unavailable
        altered['audio'] = unavailable(f.analysis.sha256, 'Invented missing input')
    elif change == 'profile':
        altered['av']['packets'][0]['measurement']['audio_curve'][0]['rms'] += .01
    else:
        altered['av']['packets'][0]['source_record']['raw_output_sha256'] = 'b'*64
    altered['sha256'] = service.packet_hash(altered)
    with pytest.raises((ObjectIntegrityError, ValueError)):
        multimedia_evidence.verify_assets(altered, f.store, result=f.result)


def test_old_video_sets_keep_their_bytes_and_remain_readable(nested_case):
    f = nested_case
    for version in ('video-evidence-1', 'video-evidence-2'):
        old = copy.deepcopy(f.document)
        old.pop('audio')
        old.pop('av')
        old['version'] = version
        old['sha256'] = service.packet_hash(old)
        before = canonical_json(old)
        video_packets.validate_packet_set(old, analysis_id=f.analysis.id, revision=f.analysis.revision,
            original_result_hash=f.analysis.result_hash, source_sha256=f.analysis.sha256)
        assert canonical_json(old) == before


def test_nested_public_zip_offline_verifier_without_runtime(nested_case, tmp_path, monkeypatch):
    f = nested_case
    monkeypatch.setattr(settings, 'data_dir', tmp_path/'data')
    key = signing.instance_key(tmp_path/'key.pem')
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((40,60), 'Controlled AV public export integrity fixture')
        pdf_bytes=pdf.tobytes()
    pdf_art=register_artefact(f.db,f.store,pdf_bytes,media_type='application/pdf',role='report',created_by='fixture',analysis_id=f.analysis.id)
    snap_art=register_artefact(f.db,f.store,snapshot.dump(f.frozen),media_type='application/json',role='stage_output',created_by='fixture',analysis_id=f.analysis.id)
    report=ReportRow(id=f.frozen.report.id,display_id=f.frozen.report.display_id,revision=1,
        analysis_id=f.analysis.id,analysis_revision=f.analysis.revision,profile='standard',redacted=False,
        review_state='machine',issuable=False,template_set=engine.TEMPLATE_SET,engine='fixture',status='done',
        snapshot_hash=snapshot.snapshot_hash(f.frozen),snapshot_artefact_id=snap_art.id,artefact_id=pdf_art.id,
        sha256=pdf_art.sha256,signature=key.sign(pdf_bytes),signing_key_id=key.key_id,
        requested_by='fixture',requested_at=datetime.now(UTC),page_count=1)
    f.db.add(report)
    f.db.flush()
    row=package.prepare(f.db,f.analysis,report,kind='draft',include_original=False,created_by='fixture',schema_id=package.PUBLIC_SCHEMA)
    built=package.build(f.db,row,store=f.store,key=key,exported_at=datetime.now(UTC))
    checked=verify.verify_package(built.path,public_key_pem=key.public_pem())
    assert checked.ok, checked.as_dict()
    with zipfile.ZipFile(built.path) as archive:
        assert json.loads(archive.read('evidence.json')) == f.document
        bundle=json.loads(archive.read('bundle.json'))
        assert {a['sha256'] for a in bundle['artifacts']} == set(service.asset_refs(f.document))
        assert all(archive.read(a['file']) == f.store.read_bytes(a['sha256']) for a in bundle['artifacts'])
    trusted=tmp_path/'trusted.pem'
    trusted.write_bytes(key.public_pem())
    offline=subprocess.run([sys.executable,'-c', '''
import importlib.abc,json,socket,sys
from pathlib import Path
class RuntimeUnavailable(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname in ('app.evidence','app.evidence.mp4','app.evidence.audio_binding','app.evidence.audio_background_binding','app.evidence.av_binding','app.evidence.image_eligibility'):
            return None
        if fullname.startswith(('app.storage','app.core.config','app.detectors','app.evidence','app.api')):
            raise RuntimeError('Offline verifier imported runtime: '+fullname)
sys.meta_path.insert(0,RuntimeUnavailable())
def no_network(*a,**k): raise RuntimeError('network forbidden')
socket.socket=no_network
from app.reports.verify import verify_package
r=verify_package(Path(sys.argv[1]),public_key_pem=Path(sys.argv[2]).read_bytes())
print(json.dumps(r.as_dict()));sys.exit(0 if r.ok else 1)
''',str(built.path),str(trusted)],capture_output=True,text=True,timeout=20)
    assert offline.returncode==0, offline.stdout+offline.stderr
