"""Image evidence HTTP reads keep real authentication, case scope and set membership checks."""
from __future__ import annotations

import hashlib
import io
import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.api import auth, security, tokens, v2_evidence, v2_evidence_packets
from app.api.v2 import install_problem_handlers
from app.core.canonical import canonical_hash, canonical_json
from app.core.config import settings
from app.evidence.image_packets import build_packet_set
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, Artefact, Base, CaseMemberRow, CaseRow
from app.storage.objects import ObjectStore


def _png(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@pytest.fixture
def evidence_http(tmp_path, monkeypatch):
    # Only infrastructure is substituted. HTTP authentication resolves real hashed API keys / signed tokens,
    # and CheckedRoute runs the production role and case-membership policies against this isolated database.
    engine = create_engine(f"sqlite:///{tmp_path / 'access.sqlite'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    store = ObjectStore(tmp_path / "objects")
    monkeypatch.setattr(settings, "single_user", False)
    monkeypatch.setattr(settings, "api_key", "test-bootstrap-not-used")
    monkeypatch.setattr(settings, "api_key_pepper", "isolated-test-pepper")
    monkeypatch.setattr(settings, "signing_key", "isolated-resource-token-signing-key")

    @contextmanager
    def sessions():
        with Session(engine) as db:
            yield db

    for module in (auth, security, v2_evidence, v2_evidence_packets):
        monkeypatch.setattr(module, "get_session", sessions)
    monkeypatch.setattr(v2_evidence, "default_store", lambda: store)
    monkeypatch.setattr(v2_evidence_packets, "default_store", lambda: store)
    monkeypatch.setattr(security, "limiter", security.RateLimiter(settings.rate_limit_per_minute))
    monkeypatch.setattr(security, "credential_limiter", security.RateLimiter(settings.credential_failures_per_minute))

    with Session(engine) as db:
        viewer = auth.new_principal("evidence-viewer", "viewer")
        outsider = auth.new_principal("evidence-outsider", "examiner")
        db.add_all([viewer, outsider])
        db.flush()
        viewer_key, _ = auth.issue_api_key(db, viewer)
        outsider_key, _ = auth.issue_api_key(db, outsider)
        for cid in ("visible-case", "private-case"):
            db.add(CaseRow(id=cid, reference=cid, title=cid, classification="internal", status="open", created_by=outsider.id))
        db.flush()
        db.add(CaseMemberRow(case_id="visible-case", principal_id=viewer.id, role="viewer", added_by=outsider.id))
        db.flush()

        def add_image(aid, case_id, offset, prepared=True):
            yy, xx = np.mgrid[:80, :100]
            source = store.put_bytes(_png(Image.fromarray(np.stack([(xx + offset) % 256, yy, (xx + yy) % 256], axis=2).astype(np.uint8))))
            analysis = Analysis(id=aid, kind="image", filename=f"{aid}.png", content_type="image/png", case_id=case_id,
                                size_bytes=source.size_bytes, sha256=source.sha256, status="done", revision=1, result_schema_version=2)
            db.add(analysis)
            db.flush()
            scalar = register_artefact(db, store, _png(Image.fromarray(np.arange(80, dtype=np.uint8).reshape(8, 10))),
                                       media_type="image/png", role="exhibit_payload", created_by="img.forensic.ela", analysis_id=aid)
            space = {"width": 100, "height": 80, "scale_to_original": 1.0}
            signal = "img.forensic.ela:global"
            result = {
                "input": {"sha256": source.sha256},
                "exhibits": [
                    {"id": f"{aid}-map", "kind": "heatmap", "created_by": "img.forensic.ela", "media_ref": "original",
                     "semantics": "error_level", "calibrated": False, "grid": [10, 8], "value_range": [0.0, 10.0],
                     "coordinate_space": space, "payload_artefact_id": scalar.id, "sha256": scalar.sha256,
                     "source_signal_ids": [signal]},
                    {"id": f"{aid}-regions", "kind": "regions", "created_by": "localise", "media_ref": "original",
                     "coordinate_space": space, "source_signal_ids": [signal],
                     "regions": [{"region": {"kind": "bbox", "bbox": {"x": 20, "y": 20, "w": 30, "h": 30}}, "score": 8.0}]},
                ],
                "signals": [{"id": signal, "detector_id": "img.forensic.ela", "display_id": "S01", "status": "ok",
                             "exhibit_ids": [f"{aid}-map"]}],
                "findings": [{"id": "F1", "family": "F5_compression_history", "evidence_ids": [signal]}],
            }
            analysis.result_hash = canonical_hash(result)
            analysis.result_json = json.dumps(result)
            store.put_bytes(canonical_json(result))
            return build_packet_set(db, analysis, store, result) if prepared else None

        first = add_image("image-a", "visible-case", 0)
        second = add_image("image-b", "visible-case", 50)
        private = add_image("image-private", "private-case", 100)
        add_image("image-unprepared", "visible-case", 150, prepared=False)
        unrelated = register_artefact(db, store, _png(Image.new("RGB", (5, 5), (0, 0, 0))), media_type="image/png",
                                      role="preview", created_by="unrelated-render", analysis_id="image-a")
        outsider_sha = unrelated.sha256
        viewer_id = viewer.id
        db.commit()

    app = FastAPI()
    install_problem_handlers(app)
    app.include_router(v2_evidence_packets.router)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, engine=engine, store=store, first=first, second=second, private=private,
                              viewer_id=viewer_id, viewer={"X-API-Key": viewer_key}, outsider={"X-API-Key": outsider_key},
                              unrelated_sha=outsider_sha, objects=tmp_path / "objects")
    engine.dispose()


def _collection(aid):
    return f"/api/v2/analyses/{aid}/evidence-packets"


def _plate(evidence):
    return evidence["packets"][0]["artifacts"]["crop"]["url"]


def test_viewer_member_can_read_but_missing_credentials_cannot(evidence_http):
    f = evidence_http
    for url in (_collection("image-a"), _plate(f.first)):
        assert f.client.get(url).status_code == 401
    collection = f.client.get(_collection("image-a"), headers=f.viewer)
    assert collection.status_code == 200
    assert collection.json() == f.first
    assert collection.headers["cache-control"] == "no-store"
    image = f.client.get(_plate(f.first), headers=f.viewer)
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert hashlib.sha256(image.content).hexdigest() == f.first["packets"][0]["artifacts"]["crop"]["sha256"]


def test_other_cases_and_nonmembers_are_hidden_even_with_a_known_etag(evidence_http):
    f = evidence_http
    for url in (_collection("image-a"), _plate(f.first)):
        response = f.client.get(url, headers=f.outsider)
        assert response.status_code == 404
        assert response.json()["title"] == "Analysis not found"
        assert f.first["sha256"] not in response.text
    for url in (_collection("image-private"), _plate(f.private)):
        response = f.client.get(url, headers={**f.viewer, "If-None-Match": '"' + f.private["packets"][0]["artifacts"]["crop"]["sha256"] + '"'})
        assert response.status_code == 404  # Authorization runs before conditional 304 handling.


def test_set_and_plate_membership_cannot_be_bypassed_by_an_existing_object(evidence_http):
    f = evidence_http
    crop_sha = f.first["packets"][0]["artifacts"]["crop"]["sha256"]
    assert f.store.read_bytes(crop_sha)
    assert f.store.read_bytes(f.unrelated_sha)
    rejected = [
        f"{_collection('image-a')}?set_id={f.second['id']}",
        f"{_collection('image-b')}/{f.first['id']}/images/{crop_sha}.png",
        f"{_collection('image-a')}/missing-set/images/{crop_sha}.png",
        f"{_collection('image-a')}/{f.first['id']}/images/{f.unrelated_sha}.png",
    ]
    for url in rejected:
        assert f.client.get(url, headers=f.viewer).status_code == 404
    # Both analyses are visible to this viewer, so denial above is exact-set binding, not just case scoping.
    assert f.client.get(_plate(f.second), headers=f.viewer).status_code == 200


def test_resource_token_is_analysis_scoped_and_still_checks_current_membership(evidence_http):
    f = evidence_http
    token, _ = tokens.make_token("analysis:image-a", f.viewer_id, 60)
    assert f.client.get(_plate(f.first), params={"token": token}).status_code == 200
    assert f.client.get(_plate(f.second), params={"token": token}).status_code == 401
    with Session(f.engine) as db:
        membership = db.get(CaseMemberRow, ("visible-case", f.viewer_id))
        db.delete(membership)
        db.commit()
    assert f.client.get(_plate(f.first), params={"token": token}).status_code == 404


def test_http_reads_never_generate_sidecars_or_change_original_results(evidence_http):
    f = evidence_http

    def snapshot():
        with Session(f.engine) as db:
            return (
                sorted(db.execute(select(Artefact.id, Artefact.sha256, Artefact.created_by)).all()),
                sorted(db.execute(select(Analysis.id, Analysis.result_hash, Analysis.result_json)).all()),
                sorted(str(p.relative_to(f.objects)) for p in f.objects.rglob("*") if p.is_file()),
            )

    before = snapshot()
    for _ in range(2):
        response = f.client.get(_collection("image-unprepared"), headers=f.viewer)
        assert response.status_code == 200
        assert response.json()["status"] == "not_generated"
        assert response.json()["packets"] == []
        assert f.client.get(_collection("image-a"), headers=f.viewer).status_code == 200
        assert f.client.get(_plate(f.first), headers=f.viewer).status_code == 200
    assert snapshot() == before
