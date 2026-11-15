"""Compare the frozen sentence probes to actual prepaid government-report evidence."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_sentence_probe as study
from scripts.writing_context_pool_worker import check_report
from scripts.writing_revision_worker_transfer import assess, upload

SCHEMA = "chipmunk.writing-sentence-probe-worker/1"
CHECKOUT = {"simulated": True, "usd_minor": 999, "jobs_before_payment": 0}
POLICIES = ("conservative", "matched")


def report(receipt, benchmark):
    if receipt["schema"] != SCHEMA or receipt["checkout"] != CHECKOUT:
        raise ValueError("Worker audit scope differs")
    expected = {(r["group"], r["family"]): r for r in benchmark["held"] if r["domain"] == "reports"}
    counts = {}
    signatures = Counter()
    for row in receipt["rows"]:
        peer = expected.pop((row["group"], row["family"]))
        if row["sha256"] != peer["sha256"] or row["checks"] != {
            "passage_classifier": "completed", "writing_pattern_ratio": "completed",
        }:
            raise ValueError("A paid run has incomplete writing evidence or a different source")
        if row["signature_verified"]:
            signatures[row["family"]] += 1
        ordinary = peer["truth"] is None
        ranges = [span for card in row["cards"] for span in card["spans"]]
        if any(not 0 <= a < b <= peer["spans"][-1][1] for a, b in ranges):
            raise ValueError("A report range is outside the compared document")
        base = bool(ranges) if ordinary else study.editorial.covers(ranges, peer["truth"], set(peer["spaces"]))
        cell = counts.setdefault(f'{peer["pair_family"]}/{"ordinary" if ordinary else "rewritten"}', Counter())
        cell.update(total=1, worker=base, inconclusive=row["decision"] == "inconclusive")
        for mode, points in benchmark["points"].items():
            for policy in POLICIES:
                label = mode + "/" + policy
                span = study.selected(peer, mode, points[policy])
                found = bool(span) if ordinary else span == peer["truth"]
                cell[label] += found
                cell[label + "/combined"] += found or base
                cell[label + "/wrong_location"] += bool(span and not ordinary and span != peer["truth"])
    if expected or signatures != {family: 1 for family in study.holdout.FAMILIES}:
        raise ValueError("The worker audit is missing evidence or a signed report replay")
    return counts


def replay(roster, benchmark, output):
    roster_pin = study.pool.file_hash(roster)
    bench_pin = study.pool.file_hash(benchmark)
    data = study.holdout.read(roster)
    comparison = study.holdout.read(benchmark)
    if roster_pin != comparison["roster_sha256"]:
        raise ValueError("Native sources differ from the held evaluation")
    study.review(comparison)
    rows = [r for r in data["rows"] if r["role"] == "held" and r["domain"] == "reports"]
    paid.configure(output, study.pool.ROOT)
    os.environ.update(VERITAS_DEVICE="mps", VERITAS_RUN_WORKER="false", VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_WEBHOOKS_ENABLED="false",
                      VERITAS_TEXT_ISOLATED_PARAGRAPH_REVIEW="true", VERITAS_TEXT_PARAGRAPH_EVIDENCE_PRIORITY="true",
                      VERITAS_TEXT_MATCHED_CONTEXT_REVIEW="true", VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="true",
                      VERITAS_TEXT_PASSAGE_POSITION_REVIEW="true", VERITAS_TEXT_LONG_PASSAGE_POSITION_REVIEW="true",
                      VERITAS_RATE_LIMIT_PER_MINUTE="1800", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth, jobs
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import get_session

    if (not torch.backends.mps.is_available() or settings.data_dir != output.resolve() / "app"
            or settings.run_worker or not settings.offline):
        raise RuntimeError("Use the isolated local Apple GPU worker")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()
    signatures = set()
    measured = []
    with (output / "progress.jsonl").open("x") as log, TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-sentence-probe-report-audit", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        worker = jobs.build_worker()
        for index, row in enumerate(rows):
            case = row | {"text_sha256": row["sha256"], "record_id": str(index),
                          "ranges": [{"start": a, "end": b} for a, b in row["spans"]],
                          "context": [text for text, _ in list(study.pool.inputs(row))[:5]]}
            aid = upload(client, stub, case, worker)
            observed = assess(client, case, aid, export=True)
            packet = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            family = row["family"]
            signature = family != "original" and family not in signatures
            if signature:
                source = output / "submitted.txt"
                source.write_text(row["text"], encoding="utf-8")
                check_report(client, worker, aid, source)
                signatures.add(family)
            result = {"group": row["group"], "family": family, "sha256": row["sha256"], "cards": observed["packets"],
                      "decision": observed["document_decision"], "signature_verified": signature,
                      "checks": packet["coverage"]["writing_check_coverage"]["checks"]}
            measured.append(result)
            print(json.dumps(result, sort_keys=True), file=log, flush=True)
            print(f"Verified {len(measured)}/{len(rows)} paid reports", flush=True)
    receipt = {"schema": SCHEMA, "benchmark_sha256": bench_pin, "checkout": CHECKOUT, "rows": measured}
    receipt["results"] = report(receipt, comparison)
    study.write(output / "receipt.json", receipt)
    print(json.dumps(receipt["results"], sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "review"))
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--roster", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        if not args.roster:
            parser.error("An exact source roster is required")
        replay(args.roster, args.benchmark, args.output)
    else:
        receipt = study.holdout.read(args.output)
        counts = report(receipt, study.holdout.read(args.benchmark))
        if counts != receipt["results"] or receipt["benchmark_sha256"] != study.pool.file_hash(args.benchmark):
            raise ValueError("The worker recount or study binding differs")
        print(json.dumps(counts, sort_keys=True))
