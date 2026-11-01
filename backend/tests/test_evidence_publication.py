"""Real modality sidecars and their publication events commit together.

Packet-builder tests alone do not exercise the audit action allowlist. These
tests use the same commit boundary as common.publish, with real source decoding,
the real audit validator and a separate database for every case.
"""
from __future__ import annotations

import json
from copy import deepcopy

import pytest
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from test_audio_evidence import audio_fixture  # noqa: F401 - real decoded PCM fixture
from test_document_packets import packet_factory  # noqa: F401 - real extracted document fixture
from test_video_evidence_packets import (
    vfr_source,  # noqa: F401 - dependency of video_fixture
    video_fixture,  # noqa: F401 - real VFR source with nonzero timestamp origin
)

from app.audit import chain as audit
from app.core.canonical import canonical_hash, canonical_json
from app.evidence import service
from app.storage.db import Analysis, Artefact, AuditEvent


@pytest.fixture(params=["audio", "video", "document-text", "document-pdf"])
def publication_source(request):
    if request.param.startswith("document-"):
        factory = request.getfixturevalue("packet_factory")
        return factory(".pdf" if request.param.endswith("pdf") else ".txt")
    return request.getfixturevalue(f"{request.param}_fixture")


def test_real_evidence_publication_commits_once_and_preserves_result(publication_source):
    fixture_db, store, analysis, result, *_ = publication_source
    # Commit the fixture's existing source and result before the publication
    # transaction. No live application database or global session is used.
    analysis.result_json = json.dumps(result)
    fixture_db.commit()
    sessions = sessionmaker(bind=fixture_db.get_bind())
    analysis_id, kind = analysis.id, analysis.kind
    original_hash, source_hash = analysis.result_hash, analysis.sha256
    original_projection = analysis.result_json
    original_result, original_bytes = deepcopy(result), store.read_bytes(original_hash)
    original_source = store.read_bytes(source_hash)
    fixture_db.rollback()  # Release the fixture's refresh transaction before writing.

    def publish():
        with audit.writing(sessions) as db:
            current = db.get(Analysis, analysis_id)
            previous = service.load_packet_set(db, current, store, set_id=service.current_set_id(current))
            packet_set = service.persist_packet_set(db, current, store, result=result)
            assert packet_set is not None and packet_set["status"] == "ready"
            assert packet_set["packets"], "The fixture must exercise actual evidence generation"
            events = []
            if previous is None:
                events.append({"action": f"{kind}_evidence_generated", "actor_id": audit.SYSTEM_ACTOR,
                               "analysis_id": analysis_id, "object_ids": [packet_set["id"]],
                               "after_hash": packet_set["sha256"], "details": {"version": packet_set["version"]}})
            # Do not mock ACTIONS, _add or commit_with: an omitted modality
            # registration must fail here, as it would in common.publish.
            audit.commit_with(db, events)
            return packet_set

    first = publish()
    with sessions() as db:
        indexed_before = list(db.execute(select(Artefact.id, Artefact.sha256).order_by(Artefact.id)))
        events_before = [audit.exported(row) for row in db.scalars(select(AuditEvent).order_by(AuditEvent.seq))]

    assert publish() == first
    with sessions() as db:
        current = db.get(Analysis, analysis_id)
        assert service.load_packet_set(db, current, store, set_id=first["id"]) == first
        sidecars = list(db.scalars(select(Artefact).where(
            Artefact.analysis_id == analysis_id, Artefact.role == "exhibit_payload",
            Artefact.created_by == service.module_for_kind(kind).CREATOR,
            Artefact.media_type == "application/json")))
        assert len(sidecars) == 1
        sidecar = sidecars[0]
        assert sidecar.id == first["id"]
        assert sidecar.sha256 == canonical_hash(first)
        assert store.read_bytes(sidecar.sha256) == canonical_json(first)
        service.validate_packet_set(first, analysis_id=analysis_id, revision=current.revision,
                                    original_result_hash=original_hash, source_sha256=source_hash)

        events = list(db.scalars(select(AuditEvent).order_by(AuditEvent.seq)))
        assert len(events) == 1
        event = events[0]
        assert event.action == f"{kind}_evidence_generated"
        assert event.actor_id == audit.SYSTEM_ACTOR and event.analysis_id == analysis_id
        assert event.object_ids == [sidecar.id]
        assert event.after_hash == first["sha256"]
        assert event.details == {"version": first["version"]}
        chain = audit.verify_chain(db=db)
        assert chain.valid and chain.checked == 1 and chain.head == event.event_hash
        assert [audit.exported(row) for row in events] == events_before
        assert list(db.execute(select(Artefact.id, Artefact.sha256).order_by(Artefact.id))) == indexed_before
        assert current.result_hash == original_hash
        assert current.result_json == original_projection
        assert current.sha256 == source_hash

    assert result == original_result
    assert store.read_bytes(original_hash) == original_bytes == canonical_json(original_result)
    assert store.read_bytes(source_hash) == original_source
