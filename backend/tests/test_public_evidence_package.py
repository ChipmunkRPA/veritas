from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import UTC, datetime

import pymupdf
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from test_image_evidence_reports import packet_snapshot

from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.reports import engine, package, signing, snapshot, verify
from app.reports.public_package import public_result
from app.reports.reproduce import ReproduceRefused, reproduce
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Base, PackageRow, ReportRow


@pytest.fixture
def public_fixture(tmp_path, monkeypatch):
    frozen, store = packet_snapshot(tmp_path)
    db_engine = create_engine('sqlite://')
    Base.metadata.create_all(db_engine)
    monkeypatch.setattr(settings, 'data_dir', tmp_path / 'data')
    key = signing.instance_key(tmp_path / 'test-key.pem')
    with Session(db_engine) as db:
        source = store.read_bytes(frozen.media.sha256)
        analysis = Analysis(id=frozen.analysis.id, kind='image', filename='original.png', content_type='image/png',
                            size_bytes=len(source), sha256=frozen.media.sha256, status='done', revision=frozen.analysis.revision,
                            result_hash=canonical_hash(frozen.result))
        db.add(analysis)
        db.flush()
        for packet in frozen.image_evidence['packets']:
            for asset in packet['artifacts'].values():
                register_artefact(db, store, store.read_bytes(asset['sha256']), media_type='image/png', role='preview',
                                 created_by='evidence-test', analysis_id=analysis.id)
        pdf = pymupdf.open()
        pdf.new_page().insert_text((50, 70), 'Chipmunk Verify - public test report')
        pdf_bytes = pdf.tobytes()
        pdf_art = register_artefact(db, store, pdf_bytes, media_type='application/pdf', role='report',
                                    created_by='report', analysis_id=analysis.id)
        frozen = frozen.model_copy(update={'report': frozen.report.model_copy(update={'template_set': engine.TEMPLATE_SET}),
                                           'media': frozen.media.model_copy(update={'size_bytes': len(source)})})
        snap_art = register_artefact(db, store, snapshot.dump(frozen), media_type='application/json', role='stage_output',
                                     created_by='report-snapshot', analysis_id=analysis.id)
        report = ReportRow(id=frozen.report.id, display_id=frozen.report.display_id, revision=1, analysis_id=analysis.id,
                           analysis_revision=analysis.revision, profile='standard', redacted=False, review_state='machine',
                           issuable=False, template_set=engine.TEMPLATE_SET, engine='test', status='done',
                           snapshot_hash=snapshot.snapshot_hash(frozen), snapshot_artefact_id=snap_art.id,
                           artefact_id=pdf_art.id, sha256=pdf_art.sha256, signature=key.sign(pdf_bytes), signing_key_id=key.key_id,
                           requested_by='test', requested_at=datetime.now(UTC), page_count=1)
        db.add(report)
        request = package.prepare(db, analysis, report, kind='draft', include_original=False, created_by='test',
                                  schema_id=package.PUBLIC_SCHEMA)
        monkeypatch.setattr(package, '_open_original', lambda *_: store.open(analysis.sha256))
        yield db, store, key, frozen, analysis, report, request


def build(fixture):
    db, store, key, *_, request = fixture
    return package.build(db, request, store=store, key=key, exported_at=datetime.now(UTC))


def rewrite(path, target, mutate, key=None):
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    mutate(files)
    if key is not None:
        names = sorted(set(files) - {'manifest.sha256', 'manifest.sha256.sig'})
        manifest = ''.join(f'{hashlib.sha256(files[n]).hexdigest()}  {n}\n' for n in names).encode()
        files['manifest.sha256'], files['manifest.sha256.sig'] = manifest, key.sign(manifest).encode()
    with zipfile.ZipFile(target, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return target


def test_public_package_matches_frozen_evidence_and_authenticates_only_with_pin(public_fixture):
    built = build(public_fixture)
    _, _, key, frozen, _, report, request = public_fixture
    verified = verify.verify_package(built.path, public_key_pem=key.public_pem())
    assert verified.ok, verified.as_dict()
    assert not verify.verify_package(built.path).ok
    assert verify.verify_package(built.path).outcome == 'self_consistent'
    assert verify.verify_package(built.path, expect_key_id=key.key_id).ok
    with zipfile.ZipFile(built.path) as archive:
        evidence = json.loads(archive.read('evidence.json'))
        assert evidence == frozen.image_evidence
        bundle = json.loads(archive.read('bundle.json'))
        assert bundle['analysis']['original_result_sha256'] == canonical_hash(frozen.result)
        assert bundle['result']['sha256'] != bundle['analysis']['original_result_sha256']
        assert bundle['custody']['complete_event_chain_included'] is False
        assert len(bundle['reports']) == 1 and bundle['reports'][0]['sha256'] == report.sha256
        assert all(not n.startswith('source/') for n in archive.namelist())
        assert all('/img.' not in n and '/loc.' not in n for n in archive.namelist())
        assert 'manifest' not in json.loads(archive.read('result.json'))
    assert request.schema_id == package.PUBLIC_SCHEMA
    assert any('NOT verified' in check.detail for check in verified.checks)


def test_optional_original_exact_and_reproduction_explicitly_refused(public_fixture):
    _, store, key, _, analysis, _, request = public_fixture
    request.include_original = True
    built = build(public_fixture)
    result = verify.verify_package(built.path, public_key_pem=key.public_pem())
    assert result.ok, result.as_dict()
    with zipfile.ZipFile(built.path) as archive:
        assert archive.read(result.bundle['source']['file']) == store.read_bytes(analysis.sha256)
    with pytest.raises(ReproduceRefused, match='cannot reproduce inference'):
        reproduce(built.path, public_key_pem=key.public_pem())


@pytest.mark.parametrize('failure', ['tampered', 'missing', 'wrong_version', 'unknown_version', 'wrong_source', 'wrong_asset', 'custody_claim', 'extra_internal'])
def test_tampering_and_misleading_resigned_content_refused(public_fixture, tmp_path, failure):
    built = build(public_fixture)
    key = public_fixture[2]

    def mutate(files):
        bundle = json.loads(files['bundle.json'])
        asset = bundle['artifacts'][0]['file']
        if failure == 'tampered':
            files[asset] += b'alteration'
        elif failure == 'missing':
            files.pop(asset)
        elif failure == 'wrong_version':
            bundle['evidence']['version'] = 'invented-version'
        elif failure == 'unknown_version':
            evidence = json.loads(files['evidence.json'])
            evidence['version'] = 'invented-version'
            evidence['sha256'] = canonical_hash({k: v for k, v in evidence.items() if k != 'sha256'})
            files['evidence.json'] = canonical_json(evidence)
            bundle['evidence'].update(version=evidence['version'], set_sha256=evidence['sha256'],
                                      sha256=hashlib.sha256(files['evidence.json']).hexdigest())
            bundle['reports'][0]['evidence_set_sha256'] = evidence['sha256']
        elif failure == 'wrong_source':
            bundle['source_commitment']['sha256'] = '0' * 64
        elif failure == 'wrong_asset':
            files[asset] += b'validly-signed-but-wrong-frozen-asset'
            bundle['artifacts'][0]['bytes'] = len(files[asset])
        elif failure == 'custody_claim':
            bundle['custody']['complete_event_chain_included'] = True
        elif failure == 'extra_internal':
            files['internal-model-registry.json'] = b'{}'
        files['bundle.json'] = canonical_json(bundle)

    altered = rewrite(built.path, tmp_path / 'altered.zip', mutate, None if failure == 'tampered' else key)
    result = verify.verify_package(altered, public_key_pem=key.public_pem())
    assert result.outcome == 'not_verified', result.as_dict()


def test_wrong_key_and_old_report_or_mismatched_snapshot_refused(public_fixture, tmp_path):
    built = build(public_fixture)
    other = signing.instance_key(tmp_path / 'other.pem')
    assert not verify.verify_package(built.path, public_key_pem=other.public_pem()).consistent
    report = public_fixture[-2]
    report.template_set = 'veritas-report-2.0.r47'
    with pytest.raises(package.PackageError, match='current'):
        build(public_fixture)
    report.template_set = engine.TEMPLATE_SET
    report.snapshot_hash = '0' * 64
    with pytest.raises(package.PackageError, match='snapshot'):
        build(public_fixture)


def test_public_projection_preserves_source_quotes_but_omits_internal_registry():
    result = {'manifest': {'models': ['Mesorch']}, 'signals': [{'id': 'loc.mesorch:global', 'name': 'Mesorch',
              'observed': 'Mesorch private generated text', 'details': {'model': 'Mesorch'}, 'raw_score': 0.7}],
              'findings': [{'id': 'F1', 'family': 'F1_provenance', 'statement': 'The user source contains Mesorch'}]}
    projected = public_result(result)
    assert projected['findings'][0]['statement'] == result['findings'][0]['statement']
    assert 'Mesorch' not in json.dumps(projected['checks']) and 'manifest' not in projected
    assert projected['original_result_sha256'] == canonical_hash(result)


def test_existing_package_rows_default_to_legacy_format():
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(Analysis(id='a', kind='image', filename='original.png', content_type='image/png',
                        size_bytes=1, sha256='a' * 64, status='done'))
        db.flush()
        row = PackageRow(id='legacy', analysis_id='a', report_id='r', kind='draft', status='done',
                         created_by='test', created_at=datetime.now(UTC))
        db.add(row)
        db.flush()
        assert row.schema_id == package.SCHEMA


def test_legacy_offline_verification_keeps_original_custody_semantics(public_fixture, tmp_path):
    built = build(public_fixture)
    key = public_fixture[2]

    def legacy_bundle(files):
        public = json.loads(files['bundle.json'])
        report = public['reports'][0]
        event = {field: None for field in verify.BODY_FIELDS}
        event.update(seq=1, ts='2026-10-31T00:00:00Z', actor_id='test', action='report_generated',
                     analysis_id=public['analysis']['id'], report_id=report['id'], after_hash=report['sha256'],
                     object_ids=[], details={}, prev_hash='0' * 64)
        event['event_hash'] = verify.event_hash(event['prev_hash'], event)
        files['bundle.json'] = canonical_json({'schema': package.SCHEMA, 'bundle_id': 'legacy', 'kind': 'draft',
            'analysis': public['analysis'], 'reports': [report], 'custody': [event],
            'custody_anchor': {'head_seq': 1, 'head_hash': event['event_hash']}})

    legacy = rewrite(built.path, tmp_path / 'legacy.zip', legacy_bundle, key)
    original = legacy.read_bytes()
    result = verify.verify_package(legacy, public_key_pem=key.public_pem())
    assert result.ok, result.as_dict()
    assert any(c.name == 'custody' and c.ok for c in result.checks)
    assert legacy.read_bytes() == original

    def altered_custody(files):
        bundle = json.loads(files['bundle.json'])
        bundle['custody'][0]['actor_id'] = 'changed-without-updating-original-event-hash'
        files['bundle.json'] = canonical_json(bundle)

    altered = rewrite(legacy, tmp_path / 'legacy-altered.zip', altered_custody, key)
    assert not verify.verify_package(altered, public_key_pem=key.public_pem()).consistent
