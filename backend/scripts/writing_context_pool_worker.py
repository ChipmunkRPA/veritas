"""Replay a position- and generator-balanced essay subset through the local app."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_context_pool_transfer as study
from scripts.writing_revision_worker_transfer import assess, upload


def sample(roster):
    held = [row for row in roster if (row["domain"], row["role"]) == ("essays", "held")]
    groups = list(study.group_rows(held).values())[:6]
    if len(groups) != 6:
        raise ValueError("Six held essay works are required")
    chosen = [
        next(row for row in group if row["family"] == family)
        for i, group in enumerate(groups)
        for family in ("original", study.GENERATORS[i % len(study.GENERATORS)])
    ]
    positions = {(row["family"], row["spans"].index(row["truth"])) for row in chosen if row["truth"]}
    if positions != {(family, i) for family in study.GENERATORS for i in (1, 2, 3)}:
        raise ValueError("Each generator must be represented at each edit position")
    return chosen


def check_report(client, worker, aid, source):
    import pymupdf

    from app.reports.verify import verify_package

    report = paid.body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    assert worker.run_once()
    report = paid.wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
    pdf = paid.fetch(client, f"/api/v2/reports/{report['id']}/file")
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        words = " ".join(" ".join(page.get_text().split()) for page in doc)
    if ("Passage classifier. Completed." not in words or "Writing-pattern ratio. Completed." not in words):
        raise ValueError("The PDF did not report both completed writing checks")
    archive = paid.body(client.post(f"/api/v2/analyses/{aid}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False,
    }))
    assert worker.run_once()
    archive = paid.wait(client, f"/api/v2/packages/{archive['id']}", timeout=120)
    package = source.with_name("signed-report.zip")
    package.write_bytes(paid.fetch(client, f"/api/v2/packages/{archive['id']}/file"))
    pem = paid.body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    verification = verify_package(package, public_key_pem=pem, source_document=source)
    if verification.outcome != "verified":
        raise ValueError(verification.as_dict())


def replay(roster, public, output):
    bench = json.loads(public.read_text())
    data = json.loads(roster.read_text())
    if study.file_hash(roster) != bench["input_roster_sha256"]:
        raise ValueError("Worker source roster differs from the numeric study")
    selected = sample(data["rows"])
    measured = {(row["group"], row["family"]): row for row in bench["held"]}
    paid.configure(output, study.ROOT)
    os.environ.update(
        VERITAS_DEVICE="mps", VERITAS_RUN_WORKER="false", VERITAS_ENVIRONMENT="dev",
        VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
        VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_WEBHOOKS_ENABLED="false",
        VERITAS_TEXT_ISOLATED_PARAGRAPH_REVIEW="true", VERITAS_TEXT_PARAGRAPH_EVIDENCE_PRIORITY="true",
        VERITAS_TEXT_MATCHED_CONTEXT_REVIEW="true", VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="true",
        VERITAS_TEXT_PASSAGE_POSITION_REVIEW="true", VERITAS_TEXT_LONG_PASSAGE_POSITION_REVIEW="true",
        VERITAS_RATE_LIMIT_PER_MINUTE="1800", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
    )
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth, jobs
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import get_session

    if (settings.data_dir != output.resolve() / "app" or settings.run_worker or not settings.offline
            or not torch.backends.mps.is_available()):
        raise ValueError("An isolated offline worker with Apple GPU access is required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = paid.PaymentStub()
    service._stripe = stub
    records = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("context-pool-eval-" + uuid.uuid4().hex[:8], "examiner")
            db.add(owner)
            db.flush()
            secret, _ = auth.issue_api_key(db, owner)
            db.commit()
        client.headers["X-API-Key"] = secret
        worker = jobs.build_worker()
        for index, row in enumerate(selected):
            query = row | {"text_sha256": row["sha256"], "record_id": str(index),
                           "ranges": [{"start": a, "end": b} for a, b in row["spans"]],
                           "context": [text for text, _ in list(study.inputs(row))[:5]]}
            aid = upload(client, stub, query, worker)
            result = assess(client, query, aid, export=True)
            packet = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            recorded = measured[row["group"], row["family"]]
            if [pin for pin, _ in result["raw"]] != recorded["input_sha256"][:5]:
                raise ValueError("Worker contexts differ from the benchmark")
            checks = packet["coverage"]["writing_check_coverage"]["checks"]
            if checks != {"passage_classifier": "completed", "writing_pattern_ratio": "completed"}:
                raise ValueError("Required writing checks did not finish")
            source = output / "submitted.txt"
            source.write_text(row["text"], encoding="utf-8")
            check_report(client, worker, aid, source)
            ranges = study.baseline(recorded)
            combined = bench["parameters"]["combined"]
            top = study.rank(recorded, "combined", combined)
            proposed = row["spans"][top[1]] if top and top[0] > combined["matched"] else None
            records.append({
                "sha256": row["sha256"], "source_group": row["group"], "family": row["family"],
                "truth": row["truth"], "worker_cards": result["packets"],
                "worker_useful": any(study.useful(row, span) for card in result["packets"] for span in card["spans"]),
                "proxy_ranges": ranges, "proxy_useful": any(study.useful(row, span) for span in ranges),
                "candidate_range": proposed, "candidate_useful": proposed is not None and study.useful(row, proposed),
                "result": result["document_decision"], "checks": checks, "report_signature_replayed": True,
            })
            print(f"Completed {index + 1}/{len(selected)} {row['family']}", flush=True)
    study.write(output / "receipt.json", {
        "schema": "chipmunk.writing-context-pool-worker/1", "benchmark_sha256": study.file_hash(public),
        "runner_sha256": study.file_hash(Path(__file__)),
        "selection": "First six held essay works. One genuine excerpt and one edit from alternating generators.",
        "billing": {"provider": "simulated Stripe", "amount_minor": 999, "currency": "usd", "queued_before_paid": 0},
        "device": "Apple MPS", "rows": records,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("public", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    replay(args.roster, args.public, args.output)
