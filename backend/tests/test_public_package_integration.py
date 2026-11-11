"""Real migration, media-packet ZIPs and CheckedRoute access for public derivatives.

Detector scores are controlled fixtures; these tests establish export integrity and
authorization, not model accuracy. PDF rendering has separate report/visual tests.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from alembic import command
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.orm import Session
from test_audio_evidence import audio_fixture as audio_fixture
from test_document_packets import packet_factory as packet_factory
from test_image_evidence_access import evidence_http as evidence_http
from test_video_evidence_packets import vfr_source as vfr_source
from test_video_evidence_packets import video_fixture as video_fixture

from app.api import auth, security, tokens, v2_evidence_packets, v2_reports
from app.api.v2 import install_problem_handlers
from app.core.canonical import canonical_hash
from app.core.config import settings
from app.evidence import audio_packets, document_packets, service, video_packets
from app.reports import engine, package, signing, snapshot, verify
from app.reports.specimens import specimen
from app.storage.artefacts import register_artefact
from app.storage.db import Analysis, CaseMemberRow, MediaObject, ReportRow
from app.storage.migrations import alembic_config, current_revision, migration
from app.storage.objects import ObjectStore


def _zip_bytes(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_actual_0023_to_0024_migration_preserves_legacy_package_bytes_and_every_field(tmp_path):
    db_engine = create_engine(f"sqlite:///{tmp_path / 'migration.sqlite'}")
    with migration(db_engine) as connection:
        command.upgrade(alembic_config(connection), "0023")
    assert current_revision(db_engine) == "0023"
    assert "schema_id" not in {column["name"] for column in inspect(db_engine).get_columns("packages")}

    key = signing.instance_key(tmp_path / "legacy-key.pem")
    store = ObjectStore(tmp_path / "objects")
    manifest = f"{hashlib.sha256(b'legacy content').hexdigest()}  original.txt\n".encode()
    signature = key.sign(manifest)
    original = _zip_bytes({"original.txt": b"legacy content", "manifest.sha256": manifest,
                           "manifest.sha256.sig": signature.encode()})
    with Session(db_engine) as db:
        analysis = Analysis(id="legacy-analysis", kind="image", filename="source.png", content_type="image/png",
                            size_bytes=1, sha256="a" * 64, status="done", result_hash="b" * 64)
        db.add(analysis)
        db.flush()
        art = register_artefact(db, store, original, media_type="application/zip", role="package",
                               created_by="legacy-export", analysis_id=analysis.id)
        old_table = Table("packages", MetaData(), autoload_with=db.connection())
        db.execute(old_table.insert().values(
            id="legacy-package", analysis_id=analysis.id, report_id="legacy-report", case_id=None,
            kind="draft", include_original=True, status="done", error=None, artefact_id=art.id,
            sha256=art.sha256, manifest_sha256=hashlib.sha256(manifest).hexdigest(), signature=signature,
            signing_key_id=key.key_id, file_count=3, size_bytes=len(original), created_by="legacy-examiner",
            created_at=datetime(2026, 9, 12, 9, 0, tzinfo=UTC), finished_at=datetime(2026, 9, 12, 9, 1, tzinfo=UTC),
            audit_event_seq=42))
        db.commit()
        before = dict(db.execute(select(old_table)).mappings().one())
    before_hash = canonical_hash({k: v.isoformat() if isinstance(v, datetime) else v for k, v in before.items()})

    with migration(db_engine) as connection:
        command.upgrade(alembic_config(connection), "0024")
    assert current_revision(db_engine) == "0024"
    new_table = Table("packages", MetaData(), autoload_with=db_engine)
    with db_engine.connect() as connection:
        after = dict(connection.execute(select(new_table)).mappings().one())
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    assert after.pop("schema_id") == package.SCHEMA
    assert after == before
    assert canonical_hash({k: v.isoformat() if isinstance(v, datetime) else v for k, v in after.items()}) == before_hash
    assert store.read_bytes(after["sha256"]) == original
    with zipfile.ZipFile(io.BytesIO(original)) as archive:
        frozen_manifest = archive.read("manifest.sha256")
        assert hashlib.sha256(frozen_manifest).hexdigest() == after["manifest_sha256"]
        assert signing.verify(key.public, frozen_manifest, after["signature"])
    db_engine.dispose()


@pytest.fixture(params=["video", "audio", "document"])
def multimedia_package(request, tmp_path, monkeypatch):
    kind = request.param
    if kind == "video":
        db, store, analysis, result, *_ = request.getfixturevalue("video_fixture")
        document = video_packets.build_packet_set(db, analysis, store, result)
    elif kind == "audio":
        db, store, analysis, result, *_ = request.getfixturevalue("audio_fixture")
        document = audio_packets.build_packet_set(db, analysis, store, result)
    else:
        db, store, analysis, result, *_ = request.getfixturevalue("packet_factory")(".pdf")
        document = document_packets.build_packet_set(db, analysis, store, result)
    key = signing.instance_key(tmp_path / "multimedia-key.pem")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    # Use the real source/result identity and the exact builder-produced set.
    # Only the report scaffolding and its small signed PDF are synthetic.
    frozen = specimen(kind, "standard")
    frozen = frozen.model_copy(update={
        "result": copy.deepcopy(result), "evidence": copy.deepcopy(document),
        "analysis": frozen.analysis.model_copy(update={"id": analysis.id, "revision": analysis.revision}),
        "media": frozen.media.model_copy(update={"sha256": analysis.sha256, "size_bytes": analysis.size_bytes,
                                                 "filename": analysis.filename, "content_type": analysis.content_type}),
        "report": frozen.report.model_copy(update={"template_set": engine.TEMPLATE_SET}),
    })
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 70), f"Chipmunk Verify {kind} export integrity fixture")
        pdf_bytes = pdf.tobytes()
    pdf_art = register_artefact(db, store, pdf_bytes, media_type="application/pdf", role="report",
                               created_by="report-fixture", analysis_id=analysis.id)
    snap_art = register_artefact(db, store, snapshot.dump(frozen), media_type="application/json", role="stage_output",
                                created_by="report-snapshot", analysis_id=analysis.id)
    report = ReportRow(id=frozen.report.id, display_id=frozen.report.display_id, revision=frozen.report.revision,
                       analysis_id=analysis.id, analysis_revision=analysis.revision, profile="standard", redacted=False,
                       review_state="machine", issuable=False, template_set=engine.TEMPLATE_SET, engine="test",
                       status="done", snapshot_hash=snapshot.snapshot_hash(frozen), snapshot_artefact_id=snap_art.id,
                       artefact_id=pdf_art.id, sha256=pdf_art.sha256, signature=key.sign(pdf_bytes), signing_key_id=key.key_id,
                       requested_by="test", requested_at=datetime.now(UTC), page_count=1)
    db.add(report)
    row = package.prepare(db, analysis, report, kind="draft", include_original=False, created_by="test",
                          schema_id=package.PUBLIC_SCHEMA)
    # The ordinary original-source lookup uses the existing content-addressed source.
    source = register_artefact(db, store, store.read_bytes(analysis.sha256), media_type=analysis.content_type,
                              role="original", created_by="intake-fixture", analysis_id=analysis.id)
    media = MediaObject(id=f"media-{kind}", sha256=analysis.sha256, size_bytes=analysis.size_bytes,
                        mime_sniffed=analysis.content_type, original_filename=analysis.filename,
                        storage_artefact_id=source.id)
    db.add(media)
    db.flush()
    analysis.media_id = media.id
    return SimpleNamespace(db=db, store=store, analysis=analysis, frozen=frozen, report=report, row=row, key=key)


@pytest.mark.parametrize("include_original", [False, True])
def test_multimedia_zip_matches_frozen_set_and_verifies_without_application_runtime(multimedia_package, tmp_path, include_original):
    f = multimedia_package
    f.row.include_original = include_original
    built = package.build(f.db, f.row, store=f.store, key=f.key, exported_at=datetime.now(UTC))
    pinned = verify.verify_package(built.path, public_key_pem=f.key.public_pem())
    assert pinned.ok, pinned.as_dict()
    assert verify.verify_package(built.path).outcome == "self_consistent"
    assert hashlib.sha256(built.path.read_bytes()).hexdigest() == built.sha256
    with zipfile.ZipFile(built.path) as archive:
        frozen = json.loads(archive.read("evidence.json"))
        bundle = json.loads(archive.read("bundle.json"))
        assert frozen == f.frozen.evidence
        assert bundle["evidence"]["set_sha256"] == frozen["sha256"]
        assert bundle["analysis"]["original_result_sha256"] == f.analysis.result_hash
        assert bundle["reports"][0]["snapshot_sha256"] == snapshot.snapshot_hash(f.frozen)
        expected = service.asset_refs(frozen)
        assert {entry["sha256"] for entry in bundle["artifacts"]} == set(expected)
        assert expected and all(archive.read(entry["file"]) == f.store.read_bytes(entry["sha256"])
                                for entry in bundle["artifacts"])
        assert {asset["media_type"] for asset in expected.values()} == (
            {"audio/wav", "image/png"} if f.analysis.kind == "audio" else
            {"video/mp4", "image/png"} if f.analysis.kind == "video" else {"image/png"})
        assert bundle["reproduction_supported"] is False
        assert bundle["custody"]["complete_event_chain_included"] is False
        public = json.loads(archive.read("result.json"))
        assert "manifest" not in public and "exhibits" not in public
        assert public["original_result_sha256"] == canonical_hash(f.frozen.result)
        leaked = [name for name in ("vid.swap.private", "aud.gustking_xlsr", "PRIVATE MODEL")
                  if name in json.dumps(public) + json.dumps(frozen)]
        check_ids = {check["id"] for check in public["checks"]}
        assert all(set(finding["evidence_ids"]) <= check_ids for finding in public["findings"])
        if include_original:
            assert archive.read(bundle["source"]["file"]) == f.store.read_bytes(f.analysis.sha256)
        else:
            assert bundle["source"] is None
            assert not any(name.startswith("source/") for name in archive.namelist())

    trusted_key = tmp_path / "trusted-public.pem"
    trusted_key.write_bytes(f.key.public_pem())
    # A fresh process blocks runtime imports and networking: verification consumes
    # only the ZIP and independently supplied signing key, with no live API/DB/model.
    offline = subprocess.run([sys.executable, "-c", """
import importlib.abc, json, socket, sys
from pathlib import Path
class RuntimeUnavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # This bounded MP4 metadata reader is deliberately pure stdlib. The
        # package initializer is empty; all model/runtime evidence modules stay blocked.
        if fullname in ('app.evidence', 'app.evidence.mp4', 'app.evidence.image_eligibility', 'app.evidence.document_context', 'app.evidence.audio_binding', 'app.evidence.av_binding', 'app.evidence.audio_background_binding', 'app.evidence.face_interval_binding', 'app.evidence.face_alignment_context', 'app.evidence.video_packet_versions', 'app.evidence.sound_event_binding'):
            return None
        if fullname.startswith(('app.storage', 'app.core.config', 'app.detectors', 'app.evidence', 'app.api',
                                'torch', 'numpy', 'cv2', 'transformers', 'fastapi', 'sqlalchemy')):
            raise RuntimeError('Offline verification imported runtime: ' + fullname)
sys.meta_path.insert(0, RuntimeUnavailable())
def no_network(*args, **kwargs):
    raise RuntimeError('Offline verification attempted networking')
socket.socket = no_network
from app.reports.verify import verify_package
result = verify_package(Path(sys.argv[1]), public_key_pem=Path(sys.argv[2]).read_bytes())
print(json.dumps(result.as_dict()))
sys.exit(0 if result.ok else 1)
""", str(built.path), str(trusted_key)], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=20)
    assert offline.returncode == 0, offline.stdout + offline.stderr
    assert json.loads(offline.stdout)["result"] == "verified"
    assert leaked == [], f"Public derivative exposes internal check identities: {leaked}"


@pytest.mark.parametrize('name', ['face_interval_binding.py', 'sound_event_binding.py'])
def test_new_offline_binding_helpers_remain_standard_library_only(name):
    import ast

    path = Path(__file__).parents[1] / 'app/evidence' / name
    imports = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imports.update(n.name.split('.')[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imports.add(node.module.split('.')[0])
    assert imports <= sys.stdlib_module_names


@pytest.fixture
def package_http(evidence_http, monkeypatch):
    f = evidence_http

    @contextmanager
    def sessions():
        with Session(f.engine) as db:
            yield db

    monkeypatch.setattr(v2_reports, "get_session", sessions)
    zip_bytes = _zip_bytes({"README.txt": b"HTTP authorization fixture"})
    with Session(f.engine) as db:
        examiner = auth.new_principal("package-examiner", "examiner")
        db.add(examiner)
        db.flush()
        examiner_key, _ = auth.issue_api_key(db, examiner)
        db.add(CaseMemberRow(case_id="visible-case", principal_id=examiner.id, role="examiner", added_by=examiner.id))
        for aid in ("image-a", "image-b", "image-private"):
            analysis = db.get(Analysis, aid)
            report = ReportRow(id=f"report-{aid}", display_id=f"report-{aid}", revision=1, analysis_id=aid,
                               analysis_revision=1, profile="standard", redacted=False, review_state="machine",
                               issuable=False, template_set=engine.TEMPLATE_SET, engine="test", status="done",
                               snapshot_hash="a" * 64, snapshot_artefact_id="unused-http-snapshot",
                               requested_by=examiner.id, requested_at=datetime.now(UTC))
            db.add(report)
            artifact = register_artefact(db, f.store, zip_bytes, media_type="application/zip", role="package",
                                        created_by="http-test", analysis_id=aid)
            row = package.prepare(db, analysis, report, kind="draft", include_original=False, created_by=examiner.id,
                                  schema_id=package.PUBLIC_SCHEMA)
            row.id, row.status, row.artefact_id = f"package-{aid}", "done", artifact.id
            row.sha256, row.size_bytes = artifact.sha256, len(zip_bytes)
            # A stale/forged denormalized case id must never grant access: the
            # authoritative resource scope is the owning analysis's current case.
            row.case_id = "visible-case"
        examiner_id = examiner.id
        db.commit()
    app = FastAPI()
    install_problem_handlers(app)
    app.include_router(v2_reports.router)
    app.include_router(v2_evidence_packets.router)
    assert all(isinstance(route, security.CheckedRoute) for route in v2_reports.router.routes)
    with TestClient(app) as client:
        yield SimpleNamespace(client=client, engine=f.engine, examiner_id=examiner_id,
                              examiner={"X-API-Key": examiner_key}, viewer=f.viewer, outsider=f.outsider,
                              zip=zip_bytes, evidence=f.first)


def test_package_http_authentication_role_case_scope_and_etag(package_http):
    f = package_http
    base = "/api/v2/packages/package-image-a"
    for path in (base, base + "/file"):
        assert f.client.get(path).status_code == 401
        assert f.client.get(path, headers=f.viewer).status_code == 403
        hidden = f.client.get(path, headers=f.outsider)
        assert hidden.status_code == 404 and hidden.json()["title"] == "Package not found"
        assert f.client.get(path, headers=f.examiner).status_code == 200
    downloaded = f.client.get(base + "/file", headers=f.examiner)
    assert downloaded.content == f.zip and downloaded.headers["content-type"] == "application/zip"
    assert "chipmunk-evidence-" in downloaded.headers["content-disposition"]
    assert f.client.get(base + "/file", headers={**f.examiner, "If-None-Match": downloaded.headers["etag"]}).status_code == 304
    for path in ("/api/v2/packages/package-image-private", "/api/v2/packages/package-image-private/file"):
        hidden = f.client.get(path, headers={**f.examiner, "If-None-Match": downloaded.headers["etag"]})
        assert hidden.status_code == 404 and hidden.json()["title"] == "Package not found"
    # CheckedRoute authorizes before parsing even a malformed create request.
    path = "/api/v2/analyses/image-a/packages"
    assert f.client.post(path, content=b"invalid JSON").status_code == 401
    assert f.client.post(path, content=b"invalid JSON", headers=f.outsider).status_code == 404
    assert f.client.post(path, content=b"invalid JSON", headers=f.viewer).status_code == 403


def test_package_http_rejects_other_resource_tokens_and_rechecks_revoked_membership(package_http):
    f = package_http
    plate = f.evidence["packets"][0]["artifacts"]["crop"]["url"]
    token, _ = tokens.make_token("analysis:image-a", f.examiner_id, 60)
    assert f.client.get(plate, params={"token": token}).status_code == 200
    for resource in ("analysis:image-a", "analysis:image-b", "report:report-image-a"):
        token, _ = tokens.make_token(resource, f.examiner_id, 60)
        for path in ("/api/v2/packages/package-image-a", "/api/v2/packages/package-image-a/file"):
            assert f.client.get(path, params={"token": token}).status_code == 401
    with Session(f.engine) as db:
        db.delete(db.get(CaseMemberRow, ("visible-case", f.examiner_id)))
        db.commit()
    for path in ("/api/v2/packages/package-image-a", "/api/v2/packages/package-image-a/file"):
        assert f.client.get(path, headers=f.examiner).status_code == 404
