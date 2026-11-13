"""Replay pinned OpAI-Bench text through the isolated paid Full document worker."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import time
import uuid
from collections import Counter
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_revision_trajectory_transfer as source

SCHEMA = "chipmunk.writing-revision-worker/1"
CHECKPOINT = "txt.four_class"
CHECKS = ("txt.four_class", "txt.binoculars", "txt.stylometry")
OPTIONAL = ("matched_context_review", "checkpoint_agreement_review", "passage_position_review")


def roster(prior):
    saved = source.read(prior / "freeze.json")
    require = source.require
    require(saved["schema"] == source.SCHEMA and saved["revision"] == source.REVISION
            and source.digest(prior / "roster.json") == saved["roster_sha256"], "Prior source freeze differs")
    cases = source.read(prior / "roster.json")
    originals = {row["group"]: row["text_sha256"] for row in cases if row["family"] == "original"}
    require(Counter(row["role"] for row in cases) == {"fit": 16, "held": 140} and len(originals) == 36,
            "Original roster must retain every report and delivery")
    require(all(row["text_sha256"] == source.sha(row["text"]) for row in cases), "Changed text input")
    return [dict(row, changed=row["text_sha256"] != originals[row["group"]]) for row in cases]


def phase_rows(rows, phase):
    return [row for row in rows if row["role"] == phase]


def upload(client, stub, row, worker):
    from app.billing import purchase_notice

    payload = row["text"].encode()
    before = paid.counts()
    order = paid.body(client.post("/api/v2/billing/checkout", headers={"Idempotency-Key": str(uuid.uuid4())},
        data={"kind": "document", "profile": "full", "risk_accepted": "true", "risk_notice_version": purchase_notice.VERSION,
              "risk_notice_sha256": purchase_notice.SHA256}, files={"file": ("report.txt", payload, "text/plain")}))
    sid = next(reversed(stub.sessions))
    nonce = uuid.uuid4().hex
    source.require((stub.sessions[sid]["amount_total"], stub.sessions[sid]["currency"]) == (999, "usd"), "Wrong price")
    source.require(paid.counts() == before and paid.event(client, sid, nonce + "unpaid")["received"]
                   and paid.counts() == before and not worker.run_once(), "A job ran before payment")
    stub.sessions[sid].update(payment_status="paid", status="complete")
    source.require(paid.event(client, sid, nonce + "paid")["received"] and paid.event(client, sid, nonce + "paid")["received"],
                   "Paid fulfillment was not idempotent")
    order = paid.body(client.get(f"/api/v2/billing/orders/{order['order_id']}"))
    source.require(order["status"] == "fulfilled" and paid.counts() == tuple(n + 1 for n in before) and worker.run_once(),
                   "Paid analysis did not queue and run exactly once")
    return order["analysis_id"]


def assess(client, row, aid, *, export):
    from app.api.public_projection import alias_signal
    from app.evidence import document_scope
    from app.storage.db import Analysis, AnalysisStage, Artefact, get_session
    from app.storage.objects import default_store

    evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
    if export:
        receipt = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
        source.require(receipt["evidence"] == evidence, "Published evidence and export disagree")
    with get_session() as db:
        analysis = db.get(Analysis, aid)
        source.require(analysis.status == "done" and analysis.sha256 == row["text_sha256"], "Expected completed source-bound analysis")
        store = default_store()
        result = json.loads(store.read_bytes(analysis.result_hash))
        stage = db.get(AnalysisStage, (aid, "detect"))
        source.require(stage.status == "done", "Detector stage did not complete")
        detect = json.loads(store.read_bytes(db.get(Artefact, stage.output_artefact_id).sha256))
    runs = {entry["detector_id"]: entry for entry in detect["detectors"]}
    source.require(set(runs) == set(CHECKS), "Unexpected Full detector set")
    primary = runs[CHECKPOINT]
    source.require(primary["status"] == "ok" and evidence["status"] == "ready"
                   and evidence["source"]["sha256"] == evidence["source"]["extracted_text_sha256"] == row["text_sha256"],
                   "Primary or exact packet source unavailable")
    windows = primary["details"]["text_evidence"]["windows"]
    texts = row["context"]
    source.require(len(windows) == len(row["ranges"]) == len(texts), "Incomplete native window coverage")
    scores = []
    for native, expected, text in zip(windows, row["ranges"], texts, strict=True):
        focus = native["focus"]
        ctx_text = " ".join(row["text"][r["start"]:r["end"]] for r in native["spans"])
        source.require(focus == {k: expected[k] for k in ("start", "end")} and text == ctx_text
                       and native["input_sha256"] == source.sha(text) and math.isfinite(native["score"])
                       and 0 <= native["score"] <= 1, "Native response does not bind the planned focus and exact context")
        scores.append([native["input_sha256"], native["score"].hex()])
    ratio = runs["txt.binoculars"]
    ratio_windows = ratio.get("details", {}).get("text_evidence", {}).get("windows", [])
    ratio_scores = {}
    for native in ratio_windows:
        focus = native["focus"]
        spans = [[item["start"], item["end"]] for item in native["scored_spans"]]
        observed = float.fromhex(native["score_exact_hex"])
        source.require(math.isfinite(observed) and abs(observed - native["score"]) <= 0.00000051
                       and any(a <= focus["start"] < focus["end"] <= b for a, b in spans), "Invalid measured ratio scope")
        ratio_scores[focus["start"], focus["end"]] = {"raw": observed.hex(), "spans": spans,
            "neighbor_expansion": native["neighbor_expansion"], "input_sha256": native["input_sha256"]}
    source.require(len(ratio_scores) == len(ratio_windows), "Repeated native ratio focus")
    primary_ids = {alias_signal(signal["id"]) for signal in result["signals"] if signal["detector_id"] == CHECKPOINT}
    packets = []
    for packet in evidence["packets"]:
        focus = packet["focus"]
        source.require(focus["text"] == row["text"][focus["start"]:focus["end"]], "Public focus mismatches the source")
        scope = packet["inspection_scope"]
        source.require(document_scope.build(focus, packet["supporting_windows"], len(row["text"])) == scope,
                       "Public highlight is not the actual inspected span")
        packets.append({"focus": [focus["start"], focus["end"]],
                        "spans": [[r["start"], r["end"]] for r in scope["spans"]],
                        "basis": scope["basis"],
                        "primary_support": any(w["signal_ref"]["id"] in primary_ids and document_scope.supports_inspection(w)
                                               for w in packet["supporting_windows"])})
    exact_ratio = [ratio_scores.get((r["start"], r["end"])) for r in row["ranges"]]
    source.require(sum(x is not None for x in exact_ratio) == len(ratio_scores), "Ratio focus differs from source positions")
    return {"record_id": row["record_id"], "raw": scores, "ratio": exact_ratio, "packets": packets,
            "document_decision": result["decision"]["code"],
            "runs": {key: {"status": entry["status"], "reason": entry.get("reason"), "versions": entry["model_revisions"]}
                     for key, entry in runs.items()},
            "optional_reviews": [key for key in OPTIONAL if key in evidence["coverage"]]}


def recover(client, log, worker, rows):
    from sqlalchemy import select

    from app.api import auth
    from app.storage.db import Analysis, BillingOrderRow, Job, PrincipalRow, get_session

    log.seek(0)
    measured = [json.loads(line) for line in log]
    source.require([row["record_id"] for row in measured] == [row["record_id"] for row in rows[:len(measured)]],
                   "The saved journal is not a complete prefix of the frozen batch")
    with get_session() as db:
        orders = db.execute(select(BillingOrderRow, Analysis, Job).join(Analysis, Analysis.id == BillingOrderRow.analysis_id)
            .join(Job, (Job.subject_id == Analysis.id) & (Job.type == "analysis"))
            .order_by(BillingOrderRow.created_at, BillingOrderRow.id)).all()
        source.require(len(measured) <= len(orders) <= min(len(measured) + 1, len(rows))
                       and len(orders) == paid.counts()[0] == paid.counts()[1], "Isolated batch orders differ from the saved prefix")
        owners = {order.owner_id for order, _, _ in orders}
        source.require(len(owners) == 1, "The retained batch must have one evaluation account")
        owner = db.get(PrincipalRow, owners.pop())
        source.require(owner is not None and owner.username.startswith("opai-worker-eval-"), "Use an isolated writing study account")
        for index, (order, analysis, job) in enumerate(orders):
            source.require(order.status == "fulfilled" and order.mode == "test" and order.paid_at is not None
                           and (order.amount_minor, order.currency, order.kind, order.profile) == (999, "usd", "document", "full")
                           and order.sha256 == analysis.sha256 == rows[index]["text_sha256"]
                           and analysis.submitted_by == owner.id and (order.stripe_session_id or "").startswith("cs_test_"),
                           "Saved analysis or payment does not match the frozen input")
            if index < len(measured):
                source.require((analysis.status, job.status) == ("done", "done"), "A journaled case is incomplete")
        retained = [(analysis.id, job.status) for _, analysis, job in orders]
        key, _ = auth.issue_api_key(db, owner)
        db.commit()
    client.headers["X-API-Key"] = key
    for row, saved, (aid, _) in zip(rows[:len(measured)], measured, retained[:len(measured)], strict=True):
        source.require(assess(client, row, aid, export=True) == saved, "Journal and retained API evidence disagree")
    if len(retained) > len(measured):
        aid, status = retained[-1]
        if status != "done":
            worker.sweep(force=True)
            source.require(worker.run_once(), "The interrupted paid job is not ready for a retry")
        observed = assess(client, rows[len(measured)], aid, export=True)
        measured.append(observed)
        write_progress(log, observed)
    return measured


def write_progress(log, observed):
    print(json.dumps(observed, sort_keys=True, allow_nan=False), file=log, flush=True)
    os.fsync(log.fileno())


def run(prior, output, phase, *, device, fit_receipt=None, points_path=None, batches_path=None, batch=None, limit=None,
        resume=False):
    rows = phase_rows(roster(prior), phase)
    if phase == "held":
        source.require(fit_receipt is not None and points_path is not None and batches_path is not None and batch is not None,
                       "Freeze the fit and complete-group batches before held scoring")
        fit = source.read(fit_receipt)
        source.require(fit["schema"] == SCHEMA and fit["phase"] == "fit" and fit["device"] == device
                       and fit["source_roster_sha256"] == source.digest(prior / "roster.json")
                       and len(fit["rows"]) == 16 and not fit["limited"], "Complete matching fit run required")
        from scripts.review_writing_revision_worker_transfer import matches_points
        from scripts.writing_revision_worker_priority import plan

        source.require(matches_points(source.read(points_path), fit_receipt), "The fixed genuine fit differs")
        batches = source.read(batches_path)
        source.require(batches == plan(prior) and 0 <= batch < len(batches["batches"]), "The source-group batches differ")
        by_id = {row["record_id"]: row for row in rows}
        rows = [by_id[record_id] for record_id in batches["batches"][batch]]
    if limit is not None:
        rows = rows[:limit]
    journal = output / "progress.jsonl"
    receipt = output / "worker.json"
    if resume:
        source.require(phase == "held" and limit is None and journal.is_file()
                       and (output / "app" / "veritas.db").is_file() and not receipt.exists(),
                       "Resume requires an incomplete held batch in an existing isolated output directory")
    paid.configure(output, source.ROOT, resume=resume)
    os.environ.update(VERITAS_DEVICE=device, VERITAS_RUN_WORKER="false", VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING",
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

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source.require(settings.data_dir == output.resolve() / "app" and not settings.run_worker and settings.offline,
                   "Use isolated state and offline models")
    stub = paid.PaymentStub()
    service._stripe = stub
    start = time.monotonic()
    with journal.open("a+" if resume else "x") as log:
        fcntl.flock(log.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with TestClient(app, client=("127.0.0.1", 50000)) as client:
            worker = jobs.build_worker()
            if resume:
                measured = recover(client, log, worker, rows)
                print(phase, "recovered", len(measured), len(rows), flush=True)
            else:
                measured = []
                with get_session() as db:
                    owner = auth.new_principal("opai-worker-eval-" + uuid.uuid4().hex[:8], "examiner")
                    db.add(owner)
                    db.flush()
                    key, _ = auth.issue_api_key(db, owner)
                    db.commit()
                client.headers["X-API-Key"] = key
            recovered = len(measured)
            for index, row in enumerate(rows[recovered:], start=recovered):
                aid = upload(client, stub, row, worker)
                observed = assess(client, row, aid, export=True)
                measured.append(observed)
                write_progress(log, observed)
                print(phase, index + 1, len(rows), row["family"], len(observed["packets"]), observed["document_decision"], flush=True)
        source.seal(receipt, {"schema": SCHEMA, "phase": phase, "limited": limit is not None, "device": device,
            "fit_worker_sha256": source.digest(fit_receipt) if fit_receipt else None,
            "points_sha256": source.digest(points_path) if points_path else None,
            "batch": batch, "batches_sha256": source.digest(batches_path) if batches_path else None,
            "source_roster_sha256": source.digest(prior / "roster.json"), "runner_sha256": source.digest(Path(__file__)),
            "worker_duration_s": round(time.monotonic() - start, 3), "recovered_cases": recovered, "rows": measured})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fit", "held"))
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--fit-receipt", type=Path)
    parser.add_argument("--points", type=Path)
    parser.add_argument("--batches", type=Path)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.prior, args.output, args.phase, device=args.device, fit_receipt=args.fit_receipt,
        points_path=args.points, batches_path=args.batches, batch=args.batch, limit=args.limit, resume=args.resume)


if __name__ == "__main__":
    main()
