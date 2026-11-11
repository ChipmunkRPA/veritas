"""Actual C2PA signing/validation on authored pixels and a local test trust root.

This tests documentary verification, not a provider's proprietary watermark.
The deliberately declared test claim is not AI-authorship ground truth.
"""
import hashlib
import io
import json
import subprocess
import uuid
import wave
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import c2pa
import numpy as np
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from PIL import Image

from app.engine.provenance_gate import decide
from app.provenance import c2pa as verifier


@pytest.fixture(scope='module')
def signing_material():
    now = datetime.now(UTC)
    root_key, leaf_key = (ec.generate_private_key(ec.SECP256R1()) for _ in range(2))
    # c2pa-rs currently requires Organization in this signing profile. Without
    # it, a valid local signature is reported as claimSignature.mismatch:
    # https://github.com/contentauth/c2pa-rs/issues/2262
    organization = x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'Chipmunk authored test only')
    root_name = x509.Name([organization, x509.NameAttribute(NameOID.COMMON_NAME, 'Chipmunk authored test CA')])
    leaf_name = x509.Name([organization, x509.NameAttribute(NameOID.COMMON_NAME, 'Chipmunk authored test signer')])
    root = (x509.CertificateBuilder().subject_name(root_name).issuer_name(root_name)
        .public_key(root_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=1), critical=True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), critical=False)
        .sign(root_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder().subject_name(leaf_name).issuer_name(root_name)
        .public_key(leaf_key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1)).not_valid_after(now + timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.KeyUsage(True, True, False, False, False, False, False, False, False), critical=True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()), critical=False)
        .sign(root_key, hashes.SHA256()))
    pem = serialization.Encoding.PEM
    certificates = leaf.public_bytes(pem) + root.public_bytes(pem)
    private_key = leaf_key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    signer = c2pa.Signer.from_info(c2pa.C2paSignerInfo(b'es256', certificates, private_key, None))
    anchor = root.public_bytes(pem).decode()
    bundle = verifier.TrustBundle(name='local-test', label='Local authored test only', revision='test', retrieved='test',
        anchors_pem=anchor, anchors_sha256=hashlib.sha256(anchor.encode()).hexdigest())
    return signer, bundle


def sign(data, mime, signer):
    manifest = {'claim_generator': 'Chipmunk Verify authored test', 'title': 'Authored test pixels',
        'format': mime, 'assertions': [{'label': 'c2pa.actions', 'data': {'actions': [{
            'action': 'c2pa.created', 'digitalSourceType': 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'}]}}]}
    output = io.BytesIO()
    with c2pa.Builder(manifest) as builder:
        builder.sign(signer, mime, io.BytesIO(data), output)
    return output.getvalue()


@pytest.fixture(scope='module')
def signed_asset(signing_material):
    signer, bundle = signing_material
    pixels = np.random.default_rng(71).integers(0, 256, (512, 768, 3), dtype=np.uint8)
    original = io.BytesIO()
    Image.fromarray(pixels).save(original, format='PNG')
    return sign(original.getvalue(), 'image/png', signer), pixels, bundle


@pytest.fixture(scope='module')
def signed_audio(signing_material):
    signer, bundle = signing_material
    output = io.BytesIO()
    with wave.open(output, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        samples = (np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000) * 12000).astype('<i2')
        audio.writeframes(samples.tobytes())
    return sign(output.getvalue(), 'audio/wav', signer), bundle


@pytest.fixture(scope='module')
def compressed_audio_assets(signing_material, signed_audio, tmp_path_factory):
    signer, bundle = signing_material
    assets = {}
    for suffix, mime in [('mp3', 'audio/mpeg'), ('flac', 'audio/flac'), ('m4a', 'audio/mp4')]:
        directory = tmp_path_factory.mktemp('authored-audio-' + suffix)
        source, destination = directory / 'source.wav', directory / ('source.' + suffix)
        source.write_bytes(signed_audio[0])
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-i', str(source), '-map_metadata', '-1',
            '-threads', '1', str(destination)], check=True, timeout=30)
        assets[suffix] = sign(destination.read_bytes(), mime, signer), bundle, mime
    return assets


@pytest.mark.parametrize('encoding', ['mp3', 'flac', 'm4a'])
def test_actual_compressed_audio_credentials_are_validated_before_gate(encoding, compressed_audio_assets, monkeypatch):
    from app.detectors.audio.provenance_adapter import AudioCredentialsDetector
    from app.engine.stages.common import detector_entry

    data, bundle, mime = compressed_audio_assets[encoding]
    assert verifier.sniff_mime(memoryview(data)) == mime
    report = verifier.read(data, mime, bundle=bundle, legacy=bundle)
    assert report.state == 'trusted' and report.ai_declared and report.covers_bytes, report
    monkeypatch.setattr(verifier, 'load_bundle', lambda _: bundle)
    ctx = SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: data),
        exhibits=SimpleNamespace(table=Mock(return_value='table'), image=Mock(return_value='thumb')))
    adapter = AudioCredentialsDetector()
    entry = detector_entry(adapter.spec, adapter.run(ctx), 'authored-compressed-audio')
    assert decide({'detectors': [entry]}, 'audio')['action'] == 'stop_statistical_detection'


@pytest.mark.parametrize('encoding', ['mp3', 'flac'])
def test_changed_audio_under_retained_credentials_does_not_activate_gate(encoding, compressed_audio_assets, monkeypatch):
    from app.detectors.audio.provenance_adapter import AudioCredentialsDetector
    from app.engine.stages.common import detector_entry

    data, bundle, mime = compressed_audio_assets[encoding]
    changed = bytearray(data)
    # The credentials are in the leading ID3 tag; change a trailing audio byte.
    assert changed[:3] == b'ID3'
    tag_end = 10 + sum(b << shift for b, shift in zip(changed[6:10], (21, 14, 7, 0), strict=True))
    assert tag_end < len(changed) - 32
    changed[-32] ^= 1
    report = verifier.read(changed, mime, bundle=bundle, legacy=bundle)
    assert report.state == 'invalid' and not report.covers_bytes, report
    monkeypatch.setattr(verifier, 'load_bundle', lambda _: bundle)
    ctx = SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: changed),
        exhibits=SimpleNamespace(table=Mock(return_value='table'), image=Mock(return_value='thumb')))
    adapter = AudioCredentialsDetector()
    entry = detector_entry(adapter.spec, adapter.run(ctx), 'changed-audio-under-credentials')
    assert decide({'detectors': [entry]}, 'audio')['action'] == 'continue_detection'


@pytest.mark.parametrize('data,expected', [
    (b'fLaC' + bytes(12), 'audio/flac'),
    (b'ID3\x04\x00\x00\x00\x00\x00\x04TESTfLaC', 'audio/flac'),
    (b'ID3\x04\x00\x00\x00\x00\x00\x04TEST\xff\xfb\x90\x64', 'audio/mpeg'),
    (b'ID3\x04\x00\x00\x00\x00\x00\x00', 'audio/mpeg'),
    (b'\xff\xfb\x90\x64', 'audio/mpeg'),
    (b'ID3unsupported-test', None), (b'ID3\x04\x00\x00\x80\x00\x00\x00', None),
    (b'\xff\xe0\x00\x00', None), (b'\xff\xfb\x9c\x64', None),
    (b'\xff\xfb\xf0\x64', None), (b'\xff\xeb\x90\x64', None),
])
def test_audio_container_dispatch_is_bounded_and_not_verification(data, expected):
    assert verifier.sniff_mime(memoryview(data)) == expected


@pytest.fixture(scope='module')
def signed_video(signing_material, tmp_path_factory):
    signer, bundle = signing_material
    path = tmp_path_factory.mktemp('authored-video') / 'source.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i',
        'testsrc2=size=128x96:rate=24:duration=1', '-c:v', 'libx264', '-threads', '1',
        '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(path)], check=True, timeout=30)
    return sign(path.read_bytes(), 'video/mp4', signer), bundle


def actual_entry(data, bundle, monkeypatch):
    from app.detectors.image.adapters.provenance import C2PADetector, c2pa_result
    from app.engine.stages.common import detector_entry

    monkeypatch.setattr(verifier, 'load_bundle', lambda _: bundle)
    ctx = SimpleNamespace(exhibits=SimpleNamespace(table=Mock(return_value='table'), image=Mock(return_value='thumb')))
    return {'detectors': [detector_entry(C2PADetector.spec, c2pa_result(ctx, data), 'authored-crypto-validation')]}


def test_actual_trusted_asset_bound_declaration_activates_gate(signed_asset, monkeypatch):
    data, _, bundle = signed_asset
    report = verifier.read(data, 'image/png', bundle=bundle, legacy=bundle)
    assert report.state == 'trusted', (report.state, report.failure_codes, report.error)
    assert report.ai_declared and report.covers_bytes
    assert decide(actual_entry(data, bundle, monkeypatch), 'image')['action'] == 'stop_statistical_detection'


def test_metadata_stripping_loses_the_record_without_clearing_the_content(signed_asset, monkeypatch):
    _, pixels, bundle = signed_asset
    output = io.BytesIO()
    Image.fromarray(pixels).save(output, format='PNG')
    report = verifier.read(output.getvalue(), 'image/png', bundle=bundle, legacy=bundle)
    assert report.state == 'absent'
    assert decide(actual_entry(output.getvalue(), bundle, monkeypatch), 'image')['action'] == 'continue_detection'


def test_pixel_tampering_with_old_credentials_cannot_trigger_gate(signed_asset, monkeypatch):
    data, pixels, bundle = signed_asset
    changed = pixels.copy()
    changed[0, 0] ^= 255
    output = io.BytesIO()
    Image.fromarray(changed).save(output, format='PNG')
    # Retain the old C2PA chunk in a well-formed PNG with different pixels.
    chunks, offset = [], 8
    while offset < len(data):
        size = int.from_bytes(data[offset:offset + 4], 'big')
        kind = data[offset + 4:offset + 8]
        if kind == b'caBX':
            chunks.append(data[offset:offset + size + 12])
        offset += size + 12
    assert chunks, 'Actual signed PNG must contain credentials'
    raw = output.getvalue()
    # Insert immediately after IHDR, retaining valid original chunk CRCs.
    position = 8 + int.from_bytes(raw[8:12], 'big') + 12
    tampered = raw[:position] + b''.join(chunks) + raw[position:]
    report = verifier.read(tampered, 'image/png', bundle=bundle, legacy=bundle)
    assert report.state == 'invalid', (report.state, report.failure_codes)
    assert decide(actual_entry(tampered, bundle, monkeypatch), 'image')['action'] == 'continue_detection'


@pytest.mark.parametrize('kind', ['audio', 'video'])
def test_actual_audio_and_video_credentials_are_validated_before_gate(kind, request, monkeypatch):
    from app.detectors.audio.provenance_adapter import AudioCredentialsDetector
    from app.detectors.video.adapters.metadata import C2PAVideoDetector
    from app.engine.stages.common import detector_entry

    data, bundle = request.getfixturevalue('signed_' + kind)
    report = verifier.read(data, verifier.sniff_mime(data), bundle=bundle, legacy=bundle)
    assert report.state == 'trusted' and report.ai_declared and report.covers_bytes
    monkeypatch.setattr(verifier, 'load_bundle', lambda _: bundle)
    ctx = SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: data),
        exhibits=SimpleNamespace(table=Mock(return_value='table'), image=Mock(return_value='thumb')))
    adapter = AudioCredentialsDetector() if kind == 'audio' else C2PAVideoDetector()
    result = adapter.run(ctx)
    entry = detector_entry(adapter.spec, result, 'authored-validation')
    assert decide({'detectors': [entry]}, kind)['action'] == 'stop_statistical_detection'


@pytest.mark.parametrize('kind,audio_mode,encoding', [('image', 'legacy', 'png'), ('video', 'legacy', 'mp4'),
    *[('audio', mode, encoding) for mode in ('legacy', 'candidate') for encoding in ('wav', 'mp3', 'flac', 'm4a')]])
@pytest.mark.parametrize('profile', ['quick', 'full', 'deep'])
def test_full_pipeline_signed_record_reaches_result_without_model_acquisition(
        kind, audio_mode, encoding, profile, compressed_audio_assets, request, tmp_path, monkeypatch):
    """Isolated research DB and authored assets; never a customer/payment route."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.core.config import settings
    from app.detectors.image import copy_similarity
    from app.engine import pipelines
    from app.evidence import image_packets_v7, service
    from app.evidence.image_copy_similarity_binding import (
        SKIPPED_CREDENTIALS,
        credential_skip,
        validate_copy_scope,
    )
    from app.reports.public_package import public_result
    from app.storage.artefacts import register_artefact
    from app.storage.db import Analysis, Base, MediaObject
    from app.storage.objects import ObjectStore

    if kind == 'audio' and encoding != 'wav':
        data, bundle, mime = compressed_audio_assets[encoding]
    elif kind == 'image':
        data, _, bundle = request.getfixturevalue('signed_asset')
        mime = 'image/png'
        reflected = Mock(side_effect=AssertionError('Signed record must bypass the reflection matcher'))
        translated = Mock(side_effect=AssertionError('Signed record must bypass the translation matcher'))
        monkeypatch.setattr(copy_similarity, 'analyse', reflected)
        monkeypatch.setattr(image_packets_v7, 'selected_translation_pairs', translated)
    else:
        data, bundle = request.getfixturevalue('signed_' + kind)
        mime = {'audio': 'audio/wav', 'video': 'video/mp4'}[kind]
    suffix = '.' + encoding
    monkeypatch.setattr(verifier, 'load_bundle', lambda _: bundle)
    monkeypatch.setattr(settings, 'data_dir', tmp_path)
    monkeypatch.setattr(settings, 'min_free_bytes', 0)
    monkeypatch.setattr(settings, 'audio_detector_mode', audio_mode)
    db_engine = create_engine(f'sqlite:///{tmp_path / "authored.sqlite"}', connect_args={'check_same_thread': False})
    Base.metadata.create_all(db_engine)
    store = ObjectStore(tmp_path / 'objects')

    @contextmanager
    def sessions():
        with Session(db_engine, expire_on_commit=False) as db:
            yield db

    analysis_id = str(uuid.uuid4())
    digest = hashlib.sha256(data).hexdigest()
    with sessions() as db:
        row = Analysis(id=analysis_id, kind=kind, filename='authored-test' + suffix,
            content_type=mime, size_bytes=len(data), sha256=digest, status='queued')
        db.add(row)
        db.flush()
        artifact = register_artefact(db, store, data, media_type=mime, role='original',
            created_by='authored-local-test', analysis_id=analysis_id)
        media = MediaObject(id=str(uuid.uuid4()), sha256=digest, size_bytes=len(data),
            mime_sniffed=mime, original_filename=row.filename, storage_artefact_id=artifact.id)
        db.add(media)
        row.media_id = media.id
        db.commit()

    class NeverAcquire:
        def lease(self, *args, **kwargs):
            raise AssertionError('A signed-record fast path must never request a model lease')

    outcome = pipelines.run_analysis(analysis_id, profile=profile, sessions=sessions, store=store, models=NeverAcquire())
    assert outcome.status == 'done', (outcome.status, outcome.error, outcome.stages)
    assert outcome.outputs['provenance_gate']['action'] == 'stop_statistical_detection'
    assert all(r['status'] == 'skipped' for r in outcome.outputs['detect']['detectors'])
    assert outcome.outputs['fuse']['decision']['provenance_record'] is not None
    from app.engine.customer_assessment import customer_assessment

    assessment = customer_assessment({**outcome.outputs['fuse'], 'kind': kind, 'status': 'done'})
    assert assessment['confidence_pct'] is None
    with sessions() as db:
        result = db.get(Analysis, analysis_id)
        assert result.result_hash and result.status == 'done'
        if kind == 'image':
            reflected.assert_not_called()
            translated.assert_not_called()
            document = service.load_packet_set(db, result, store)
            assert (document['source']['width'], document['source']['height']) == (768, 512)
            assert all(document[key]['status'] == SKIPPED_CREDENTIALS and not document[key]['pairs']
                       for key in ('copy_similarity', 'copy_translation'))
            assert document['copy_translation']['basis'] is None
            raw = json.loads(store.read_bytes(result.result_hash))
            scope = public_result(raw)['decision']['image_copy_coverage']
            assert scope == credential_skip(digest)
            validate_copy_scope(document, scope)
            with pytest.raises(ValueError, match='differs from the source-bound result'):
                validate_copy_scope(document, None)
            if profile == 'quick':
                from test_video_nested_evidence import _check_public_zip

                from app.reports import engine, image_evidence, snapshot

                frozen = snapshot.build(db, result, raw, report_id=str(uuid.uuid4()), revision=1, profile='standard',
                                        redacted=False, supersedes_report_id=None, revision_reason=None,
                                        requested_at=datetime.now(UTC), requested_by='authored-local-test',
                                        template_set=engine.TEMPLATE_SET, veritas_version=settings.version, store=store)
                monkeypatch.setattr(image_evidence, 'default_store', lambda: store)
                pdf = engine.render_pdf(frozen)
                text = ' '.join(engine.pdf_text(pdf.pdf).split())
                assert text.count('The two copy checks were skipped.') == 1
                _check_public_zip(SimpleNamespace(db=db, store=store, analysis=result, frozen=frozen, document=document),
                                  tmp_path, monkeypatch, pdf.pdf)
