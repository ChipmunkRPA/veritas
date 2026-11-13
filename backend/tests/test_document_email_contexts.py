from copy import deepcopy
from email.message import EmailMessage
from zipfile import ZipFile

import pytest
from test_document_native_locations import _snapshot
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash, canonical_json
from app.evidence import document_email_contexts as scope
from app.evidence.document_packets import build_packet_set
from app.reports import engine, multimedia_evidence, public_package, public_verify, verify_document_source

BODY = "Please bring the current agenda when you arrive. I added the meeting location in the invitation. I can review the minutes later today."
QUOTE = "The earlier plan was to gather by the auditorium."
SIGNATURE = "The records team will review this in the next session."


@pytest.mark.parametrize("layout", ["plain", "mapped_html", "unmapped_html"])
def test_email_scope_counts_exact_roles_and_replays_report(packet_factory, tmp_path, monkeypatch, layout):
    message = EmailMessage()
    if layout == "plain":
        message.set_content(f"{BODY}\n\n> {QUOTE}\n\n-- \n{SIGNATURE}")
    else:
        body = f"<p>{BODY}</p><blockquote><p>{QUOTE}</p></blockquote>"
        if layout == "unmapped_html":
            body = '<div style="color: black">' + body + "</div>"
        message.set_content(body, subtype="html", cte="base64")
    message.add_attachment("Content not in the selected message.", filename="private-file.txt")
    original = tmp_path / "email.eml"
    original.write_bytes(message.as_bytes())
    scores = (0.2, 0.2, 0.7, 0.2, 0.2) if layout == "plain" else (0.2, 0.2, 0.7, 0.2)
    db, store, analysis, result, decoded, _ = packet_factory(".eml", source_pdf=original, scores=scores)
    evidence = build_packet_set(db, analysis, store, result)
    record = evidence["coverage"][scope.KEY]
    (row,) = record["checks"]
    assert record["mime_part"] == "1.1"
    expected = {
        "plain": {"body": 2, "quoted": 0, "signature": 0, "mixed": 3, "unmapped": 0},
        "mapped_html": {"body": 2, "quoted": 0, "signature": 0, "mixed": 2, "unmapped": 0},
        "unmapped_html": {"body": 0, "quoted": 0, "signature": 0, "mixed": 0, "unmapped": 4},
    }
    assert row["counts"] == expected[layout]
    assert row["unlisted"] == 0
    assert scope.verify_source(evidence, decoded["text"], decoded["source_map"], result=result) == len(scores)
    snapshot = _snapshot(evidence, analysis, result)
    monkeypatch.setattr(multimedia_evidence, "default_store", lambda: store)
    export = public_package._evidence(snapshot)
    assert export["coverage"][scope.KEY] == record
    assert "txt.desklib" not in str(export) and "PRIVATE MODEL NAME" not in str(export)
    assert "private-file.txt" not in str(record) and BODY not in str(record)
    pdf = " ".join(engine.pdf_text(engine.render_pdf(snapshot).pdf).split())
    assert "What the email writing checks read" in pdf and "Mixed roles" in pdf
    assert ("The source mapping could not separate" in pdf) is (layout == "unmapped_html")
    archive_path = tmp_path / "source-proof.zip"
    bundle = {
        "schema": public_verify.SCHEMA,
        "analysis": {"kind": "document"},
        "source_commitment": {"sha256": analysis.sha256, "bytes": analysis.size_bytes},
    }
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("evidence.json", canonical_json(export))
    with ZipFile(archive_path) as archive:
        assert f"Verified source roles and text hashes for {len(scores)} classifier inputs" in (
            verify_document_source.check(archive, bundle, original)
        )


def test_email_scope_distinguishes_no_complete_inputs_and_rejects_forged_roles(packet_factory):
    body = f"{BODY}\n\n> {QUOTE}"
    db, store, analysis, result, decoded, _ = packet_factory(".eml", scores=(0.2, 0.2, 0.7, 0.2), text=body)
    evidence = build_packet_set(db, analysis, store, result)
    checks = {evidence["coverage"][scope.KEY]["checks"][0]["signal_ref"]["id"]}
    commitment = canonical_hash(evidence["coverage"][scope.KEY])
    scope.validate_commitment(evidence, commitment, published_checks=checks)
    forged = deepcopy(evidence)
    row = forged["coverage"][scope.KEY]["checks"][0]
    row["windows"][0]["role"] = "quoted"
    row["counts"]["body"] -= 1
    row["counts"]["quoted"] += 1
    with pytest.raises(ValueError, match="published result commitment"):
        scope.validate_commitment(forged, commitment, published_checks=checks)
    with pytest.raises(ValueError, match="differs from the original source"):
        scope.verify_source(forged, decoded["text"], decoded["source_map"])
    del forged["coverage"][scope.KEY]
    with pytest.raises(ValueError, match="published result commitment"):
        scope.validate_commitment(forged, commitment, published_checks=checks)

    windows = result["signals"][0]["details"]["text_evidence"]["windows"]
    for window in windows:
        window["token_coverage"]["omitted_tokens"] = 1
    incomplete = scope.build(result, decoded["text"], decoded["source_map"], decoded["meta"]["email"])
    (row,) = incomplete["checks"]
    assert row["windows"] == [] and row["unlisted"] == 4
    assert (
        scope.build({"signals": []}, decoded["text"], decoded["source_map"], decoded["meta"]["email"])[
            "checks"
        ]
        == []
    )
