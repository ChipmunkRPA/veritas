"""The generic evidence endpoint keeps the actual authentication and case policies."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_image_evidence_access import evidence_http  # noqa: F401 - reuse real credential/case infrastructure

from app.api import routes, tokens
from app.core.canonical import canonical_hash, canonical_json
from app.detectors.audio.decode import SAMPLE_RATE, wav_bytes
from app.evidence import service
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact, CaseMemberRow


def _add_audio(db, store, aid, case_id, frequency):
    pcm = np.round(np.sin(np.arange(SAMPLE_RATE * 4) * frequency / SAMPLE_RATE) * 10000).astype("<i2").astype(np.float32) / 32768
    wav = wav_bytes(pcm)
    source = store.put_bytes(wav)
    analysis = Analysis(id=aid, kind="audio", filename="voice.wav", content_type="audio/wav", case_id=case_id,
                        size_bytes=len(wav), sha256=source.sha256, status="done", revision=1, result_schema_version=2)
    db.add(analysis)
    db.flush()
    wave_asset = register_artefact(db, store, wav, media_type="audio/wav", role="working_copy", created_by="audio-input/1", analysis_id=aid)
    pcm_sha = hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
    mapping = {"version": "audio-input/1", "source_sha256": source.sha256, "pcm_sha256": pcm_sha,
               "sample_rate": SAMPLE_RATE, "channels": 1, "samples": len(pcm), "source_stream_index": 0,
               "source_channels": 1, "source_sample_rate": SAMPLE_RATE, "source_duration_s": 4,
               "transform": "Decoded mono PCM", "timeline_time_base": "1/16000", "limit_samples": 9600000,
               "timeline": [{"start_sample": 0, "end_sample": len(pcm), "source_pts": 0}]}
    mapping_asset = register_artefact(db, store, canonical_json(mapping), media_type="application/json", role="stage_output", created_by="audio-input/1", analysis_id=aid)
    binding = {"version": "audio-input/1", "pcm_sha256": pcm_sha, "samples": len(pcm), "sample_rate": SAMPLE_RATE,
               "wav_sha256": wave_asset.sha256, "mapping_sha256": mapping_asset.sha256}
    sid = "aud.gustking_xlsr:global"
    entry = {"detector_id": "aud.gustking_xlsr", "status": "ok", "family": "voice", "details": {"audio_input": binding},
             "exhibit_ids": [], "unit_scores": [{"unit_ref": "window:0", "raw": 0.8, "raw_kind": "probability",
                "quality": {"start_s": 0.5, "end_s": 3.5, "start_sample": 8000, "end_sample": 56000}}]}
    raw = register_artefact(db, store, canonical_json(entry), media_type="application/json", role="raw_output", created_by="voice-check", analysis_id=aid)
    result = {"input": {"sha256": source.sha256}, "manifest": {"sampling": {"audio_evidence": {
        "input": binding, "checks": [{"detector_id": "aud.gustking_xlsr", "raw_output_sha256": raw.sha256}]}}},
        "signals": [{"id": sid, "detector_id": "aud.gustking_xlsr", "display_id": "S01", "status": "ok"}],
        "findings": [{"id": "F1", "evidence_ids": [sid]}]}
    analysis.result_hash = canonical_hash(result)
    analysis.result_json = json.dumps(result)
    register_artefact(db, store, canonical_json(result), media_type="application/json", role="stage_output",
                      created_by="publish-result", analysis_id=aid)
    return analysis, service.persist_packet_set(db, analysis, store, result=result)


@pytest.fixture
def audio_http(evidence_http):  # noqa: F811
    f = evidence_http
    with Session(f.engine) as db:
        _, first = _add_audio(db, f.store, "audio-a", "visible-case", 400)
        _, second = _add_audio(db, f.store, "audio-b", "visible-case", 550)
        _, private = _add_audio(db, f.store, "audio-private", "private-case", 700)
        unrelated = register_artefact(db, f.store, wav_bytes(np.zeros(16000, dtype=np.float32)), media_type="audio/wav",
                                      role="working_copy", created_by="unrelated-audio", analysis_id="audio-a")
        unrelated_sha = unrelated.sha256
        db.commit()
    return SimpleNamespace(**vars(f), audio=first, other_audio=second, private_audio=private, unrelated_audio_sha=unrelated_sha)


def _clip(document):
    return document["packets"][0]["artifacts"]["selected_clip"]["url"]


def test_generic_collection_and_wav_have_matching_bytes_and_scope(audio_http):
    f = audio_http
    response = f.client.get("/api/v2/analyses/audio-a/evidence-packets", headers=f.viewer)
    assert response.status_code == 200 and response.json() == f.audio
    for url in (_clip(f.audio), f.audio["overview"]["waveform"]["url"]):
        assert f.client.get(url).status_code == 401
        assert f.client.get(url, headers=f.outsider).status_code == 404
        assert f.client.get(url, headers=f.viewer).status_code == 200
    clip = f.client.get(_clip(f.audio), headers=f.viewer)
    assert clip.headers["content-type"] == "audio/wav"
    assert clip.headers.get("content-disposition") != "attachment"
    assert hashlib.sha256(clip.content).hexdigest() == f.audio["packets"][0]["artifacts"]["selected_clip"]["sha256"]
    assert f.client.get(_clip(f.private_audio), headers={**f.viewer, "If-None-Match": "*"}).status_code == 404


def test_wrong_set_cross_analysis_unlisted_or_wrong_extension_refused(audio_http):
    f = audio_http
    sha = f.audio["packets"][0]["artifacts"]["selected_clip"]["sha256"]
    base = f"/api/v2/analyses/audio-a/evidence-packets/{f.audio['id']}/assets"
    for url in (f"{base}/{sha}.png", f"{base}/{sha}.html", f"{base}/{f.unrelated_audio_sha}.wav",
                f"{base}/{f.audio['source']['mapping_sha256']}.wav",
                f"/api/v2/analyses/audio-b/evidence-packets/{f.audio['id']}/assets/{sha}.wav"):
        assert f.client.get(url, headers=f.viewer).status_code == 404
    assert f.client.get(_clip(f.other_audio), headers=f.viewer).status_code == 200


def test_media_ranges_are_exact_hash_verified_and_still_case_scoped(audio_http):
    f = audio_http
    url = _clip(f.audio)
    full = f.client.get(url, headers=f.viewer)
    assert full.headers['accept-ranges'] == 'bytes'
    for header, start, end in [('bytes=0-127', 0, 127), ('bytes=123-', 123, len(full.content) - 1),
                               ('bytes=-17', len(full.content) - 17, len(full.content) - 1)]:
        response = f.client.get(url, headers={**f.viewer, 'Range': header})
        assert response.status_code == 206 and response.content == full.content[start:end + 1]
        assert response.headers['content-range'] == f'bytes {start}-{end}/{len(full.content)}'
        assert int(response.headers['content-length']) == end - start + 1
    assert f.client.get(url, headers={'Range': 'bytes=0-127'}).status_code == 401
    assert f.client.get(url, headers={**f.outsider, 'Range': 'bytes=0-127'}).status_code == 404
    for header in ('bytes=99999999-', 'bytes=50-10', 'bytes=-0', 'bytes=0-1,4-5', 'bytes=' + '9' * 100 + '-'):
        refused = f.client.get(url, headers={**f.viewer, 'Range': header})
        assert refused.status_code == 416 and refused.headers['content-range'] == f'bytes */{len(full.content)}'
    changed = f.client.get(url, headers={**f.viewer, 'Range': 'bytes=0-127', 'If-Range': '"different"'})
    assert changed.status_code == 200 and changed.content == full.content
    assert f.client.get(url, headers={**f.viewer, 'Range': 'bytes=0-127', 'If-None-Match': full.headers['etag']}).status_code == 304
    sha = f.audio['packets'][0]['artifacts']['selected_clip']['sha256']
    f.store.path_for(sha).chmod(0o600)  # isolated test object: simulate storage corruption
    f.store.path_for(sha).write_bytes(b'corrupt')
    assert f.client.get(url, headers={**f.viewer, 'Range': 'bytes=0-127'}).status_code == 500


def test_wav_tokens_keep_case_membership_and_original_read_is_pure(audio_http):
    f = audio_http
    with Session(f.engine) as db:
        before = list(db.execute(select(Artefact.id, Artefact.sha256)).all())
        originals = list(db.execute(select(Analysis.id, Analysis.result_hash, Analysis.result_json)).all())
    token, _ = tokens.make_token("analysis:audio-a", f.viewer_id, 60)
    assert f.client.get(_clip(f.audio), params={"token": token}).status_code == 200
    assert f.client.get(_clip(f.other_audio), params={"token": token}).status_code == 401
    with Session(f.engine) as db:
        assert list(db.execute(select(Artefact.id, Artefact.sha256)).all()) == before
        assert list(db.execute(select(Analysis.id, Analysis.result_hash, Analysis.result_json)).all()) == originals
        db.delete(db.get(CaseMemberRow, ("visible-case", f.viewer_id)))
        db.commit()
    assert f.client.get(_clip(f.audio), params={"token": token}).status_code == 404


def test_audio_customer_export_joins_the_same_set_and_public_signal_ids(audio_http, monkeypatch):
    f = audio_http
    with Session(f.engine) as db:
        analysis = db.get(Analysis, "audio-a")
        @contextmanager
        def sessions():
            yield db
        monkeypatch.setattr(routes, "_get", lambda _: analysis)
        monkeypatch.setattr(routes, "_current_revision", lambda _: None)
        monkeypatch.setattr(routes, "_verified", lambda *_: None)
        monkeypatch.setattr(routes, "get_session", sessions)
        monkeypatch.setattr("app.storage.objects.default_store", lambda: f.store)
        events = []
        monkeypatch.setattr(routes.audit, "record", lambda *a, **kw: events.append((a, kw)))
        request = SimpleNamespace(state=SimpleNamespace(principal=SimpleNamespace(id=f.viewer_id)))
        response = routes.export_json(analysis.id, request)
        exported = json.loads(response.body)
        assert exported["evidence"] == f.audio
        assert "image_evidence" not in exported
        assert exported["evidence"]["packets"][0]["signal_refs"][0]["id"] == exported["result"]["signals"][0]["id"]
        assert events[0][1]["after_hash"] == hashlib.sha256(response.body).hexdigest()


def test_unavailable_modality_is_explicit_and_does_not_build(monkeypatch):
    analysis = SimpleNamespace(kind="unimplemented", id="future", revision=1, sha256="a" * 64, result_hash="b" * 64)
    assert service.load_packet_set(None, analysis, None) is None
    assert service.persist_packet_set(None, analysis, None) is None
    unavailable = service.unavailable_packet_set(analysis)
    assert unavailable["status"] == "not_generated" and unavailable["packets"] == []
    with pytest.raises(ValueError):
        service.build_packet_set(None, analysis, None, {})
