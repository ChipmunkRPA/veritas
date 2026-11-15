"""Replay one ordinary and one rewritten input with both writing flags."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from zipfile import ZipFile

from scripts import video_short_return_encoded_app as paid
from scripts import writing_joint_tail_transfer as study
from scripts.writing_revision_worker_transfer import upload


def examples(bench, source):
    study.validate(bench)
    lookup = {row["sha256"]: row for row in source["rows"]}
    choices = {}
    for row in bench["rows"]:
        for name, locations, _ in study.policies(row, bench["frozen"]):
            if name != "raw_agreement" or not locations:
                continue
            is_edit = bool(row["truth"])
            if is_edit and row["position"] not in locations:
                continue
            if is_edit not in choices:
                original = lookup[row["sha256"]]
                assert study.pool.sha(original["text"]) == row["sha256"] and original["spans"] == row["spans"]
                choices[is_edit] = original, list(locations.values())
    return [choices[is_edit] for is_edit in (False, True)]


def report_roundtrip(client, worker, analysis, source, evidence, output):
    import pymupdf

    from app.evidence.document_scope import DIFFERENT_WRITING_RANGES
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    report = paid.body(client.post(f"/api/v2/analyses/{analysis}/reports", json={"profile": "standard"}))
    assert worker.run_once()
    report = paid.wait(client, f"/api/v2/reports/{report['id']}")
    pdf = paid.fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as pages:
        words = " ".join(" ".join(page.get_text().split()) for page in pages)
        assert DIFFERENT_WRITING_RANGES in words
        scan(words)
    (output / "standard.pdf").write_bytes(pdf)
    package = paid.body(client.post(f"/api/v2/analyses/{analysis}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False,
    }))
    assert worker.run_once()
    package = paid.wait(client, f"/api/v2/packages/{package['id']}")
    archive_path = output / "signed-report.zip"
    archive_path.write_bytes(paid.fetch(client, f"/api/v2/packages/{package['id']}/file"))
    key = paid.body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    replay = verify_package(archive_path, public_key_pem=key, source_document=source)
    assert replay.outcome == "verified", replay.as_dict()
    with ZipFile(archive_path) as archive:
        projected = archive.read("evidence.json")
        assert json.loads(projected) == evidence
        scan(projected.decode())


def run(roster, benchmark, output):
    assert study.sources.read(benchmark)["roster_sha256"] == study.pool.file_hash(roster)
    selected = examples(study.sources.read(benchmark), study.sources.read(roster))
    paid.configure(output, study.pool.ROOT)
    os.environ.update(VERITAS_DEVICE="mps", VERITAS_RUN_WORKER="false", VERITAS_ENVIRONMENT="dev",
                      VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_WEBHOOKS_ENABLED="false",
                      VERITAS_RATE_LIMIT_PER_MINUTE="1800", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth, jobs
    from app.billing import service
    from app.evidence import document_scope
    from app.main import app
    from app.storage.db import get_session

    assert torch.backends.mps.is_available()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = paid.PaymentStub()
    service._stripe = stub
    completed = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("writing-range-local-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = key
        worker = jobs.build_worker()
        for row, expected in selected:
            aid = upload(client, stub, row, worker)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            exported = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
            assert exported["evidence"] == evidence
            assert evidence["source"]["sha256"] == evidence["source"]["extracted_text_sha256"] == row["sha256"]
            positions = []
            for packet in evidence["packets"]:
                if packet.get("review_basis") != "different_writing_ranges":
                    continue
                focus = packet["focus"]
                assert document_scope.writing_review_basis(focus, packet["inspection_scope"], packet["supporting_windows"]) == packet["review_basis"]
                positions.append([focus["start"], focus["end"]])
            assert any(position in expected for position in positions)
            folder = output / ("rewrite" if row["truth"] else "ordinary")
            folder.mkdir(mode=0o700)
            source = folder / "submitted.txt"
            source.write_text(row["text"], encoding="utf-8")
            report_roundtrip(client, worker, aid, source, evidence, folder)
            completed.append({"sha256": row["sha256"], "group": row["group"], "truth": row["truth"], "different_ranges": positions})
            print(folder.name, positions, flush=True)
    study.write(output / "receipt.json", {
        "schema": "chipmunk.writing-different-ranges-worker/1", "benchmark_sha256": study.pool.file_hash(benchmark),
        "checkout": {"simulated": True, "currency": "USD", "amount_minor": 999, "jobs_before_payment": 0},
        "source_verified_pdf_and_export": True, "rows": completed,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.roster, args.benchmark, args.output)
