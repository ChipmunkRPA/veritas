"""Unsigned labels survive real publication/exports without deciding authorship."""
import hashlib
import io
import json
import subprocess
import uuid
import wave
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock
from xml.sax.saxutils import escape

import pytest
from PIL import Image, PngImagePlugin
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.provenance import source_labels


def authored(kind, folder):
    label = json.dumps({'Label': '1', 'ContentProducer': 'Gemini', 'ProduceID': 'private-test-id'})
    if kind == 'image':
        info = PngImagePlugin.PngInfo()
        info.add_text('AIGC', label)
        out = io.BytesIO()
        Image.new('RGB', (96, 96), '#aa8877').save(out, format='PNG', pnginfo=info)
        return out.getvalue(), 'image/png', '.png'
    if kind == 'audio':
        out = io.BytesIO()
        with wave.open(out, 'wb') as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b'\0' * 32000)
        raw, data = label.encode(), out.getvalue()
        chunk = b'AIGC' + len(raw).to_bytes(4, 'little') + raw + b'\0' * (len(raw) % 2)
        return data[:4] + (len(data) + len(chunk) - 8).to_bytes(4, 'little') + data[8:] + chunk, 'audio/wav', '.wav'
    if kind == 'video':
        path = folder / 'authored.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-f', 'lavfi', '-i', 'color=s=96x96:r=24:d=1',
            '-c:v', 'libx264', '-threads', '1', '-movflags', 'use_metadata_tags', '-metadata', 'AIGC=' + label,
            str(path)], check=True, timeout=30)
        return path.read_bytes(), 'video/mp4', '.mp4'
    from docx import Document

    path = folder / 'authored.docx'
    doc = Document()
    doc.add_paragraph('This is an authored software test document. It describes ordinary local observations and contains no statement about who created the words. ' * 12)
    doc.save(path)
    xml = ('<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/custom-properties" '
        'xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">'
        '<property name="AIGC"><vt:lpwstr>' + escape(label) + '</vt:lpwstr></property></Properties>')
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr('docProps/custom.xml', xml)
    return path.read_bytes(), 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx'


@pytest.mark.parametrize('kind', ['image', 'video', 'audio', 'document'])
def test_full_unsigned_label_flow_api_json_and_report(kind, tmp_path, monkeypatch):
    from app.api import routes, v2_evidence
    from app.core.config import settings
    from app.engine import pipelines
    from app.engine.customer_assessment import customer_assessment
    from app.reports import engine as reports
    from app.reports.specimens import specimen
    from app.storage.artefacts import register_artefact
    from app.storage.db import Analysis, Base, MediaObject
    from app.storage.objects import ObjectStore

    data, mime, suffix = authored(kind, tmp_path)
    digest, aid = hashlib.sha256(data).hexdigest(), str(uuid.uuid4())
    monkeypatch.setattr(settings, 'data_dir', tmp_path)
    monkeypatch.setattr(settings, 'min_free_bytes', 0)
    db_engine = create_engine(f'sqlite:///{tmp_path / "authored.sqlite"}', connect_args={'check_same_thread': False})
    Base.metadata.create_all(db_engine)
    store = ObjectStore(tmp_path / 'objects')

    @contextmanager
    def sessions():
        with Session(db_engine, expire_on_commit=False) as db:
            yield db

    with sessions() as db:
        row = Analysis(id=aid, kind=kind, filename='authored-label-test' + suffix, content_type=mime,
            size_bytes=len(data), sha256=digest, status='queued')
        db.add(row)
        db.flush()
        asset = register_artefact(db, store, data, media_type=mime, role='original', created_by='authored-test', analysis_id=aid)
        media = MediaObject(id=str(uuid.uuid4()), sha256=digest, size_bytes=len(data), mime_sniffed=mime,
            original_filename=row.filename, storage_artefact_id=asset.id)
        db.add(media)
        row.media_id = media.id
        db.commit()

    class NoWeights:
        calls = 0
        default_device = 'cpu'

        def activation_bytes_for_items(self, counts):
            return 0

        def lease(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError('Weights deliberately unavailable in this authored flow test')

    models = NoWeights()
    outcome = pipelines.run_analysis(aid, profile='quick', sessions=sessions, store=store, models=models)
    assert outcome.status == 'done', (outcome.status, outcome.error, outcome.stages)
    assert outcome.outputs['provenance_gate']['action'] == 'continue_detection'
    packet = outcome.outputs['source_labels']
    assert packet['records'][0]['claimed_provider'] == 'Google / Gemini'
    assert packet['used_in_confidence'] is False
    if kind in ('image', 'video', 'document'):
        assert models.calls > 0, 'Unsigned metadata must not bypass applicable model checks'
    with sessions() as db:
        row = db.get(Analysis, aid)
        original_hash = row.result_hash
        monkeypatch.setattr(v2_evidence, 'default_store', lambda: store)
        result = v2_evidence._result(db, row)
        assert result['manifest']['sampling']['source_labels'] == packet
        assert all(s.get('family') != 'source-labels' for s in result['signals'])
        assessment = customer_assessment({**result, 'status': 'done'})
        assert assessment['confidence_pct'] is None and assessment['direction'] in ('inconclusive', 'unavailable')
        assert result['decision'].get('provenance_record') is None
    monkeypatch.setattr(v2_evidence, 'get_session', sessions)
    monkeypatch.setattr(v2_evidence, '_analysis', lambda db, ident: db.get(Analysis, ident))
    detail = json.loads(v2_evidence.get_analysis(aid).body)
    assert detail['manifest']['sampling']['source_labels'] == packet
    monkeypatch.setattr(routes, '_get', lambda _: row)
    monkeypatch.setattr(routes, 'get_session', sessions)
    monkeypatch.setattr(routes, '_current_revision', lambda _: None)
    monkeypatch.setattr(routes, '_verified', lambda *_: None)
    monkeypatch.setattr(routes.audit, 'record', Mock())
    response = routes.export_json(aid, SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id='authored-test'))))
    exported = json.loads(response.body)
    assert exported['result']['source_labels'] == packet
    assert 'private-test-id' not in json.dumps(packet)
    assert exported['result_hash'] == original_hash
    for profile in ('summary', 'standard', 'forensic', 'newsroom'):
        snapshot = specimen(kind, profile)
        snapshot = snapshot.model_copy(update={'result': result, 'media': snapshot.media.model_copy(update={'sha256': digest})})
        html = reports.render_html(snapshot, embed_fonts=False)
        assert 'SL01' in html and 'Source labeling - unverified metadata' in html
        assert 'Provider identity is not authenticated' in html and 'No confidence percentage is assigned' in html
        assert packet['records'][0]['payload_sha256'] in html and digest in html


def test_historical_refusion_does_not_invent_new_metadata_checks(monkeypatch):
    from app.engine.stages import common

    monkeypatch.setattr(common, 'media_path', Mock(side_effect=AssertionError('Historical input must not be re-examined')))
    got = source_labels.stage(SimpleNamespace(state={'refusion': True}, outputs={'ingest': {'sha256': 'a' * 64}}))
    assert got['status'] == 'not_assessed' and got['records'] == []
    common.media_path.assert_not_called()
