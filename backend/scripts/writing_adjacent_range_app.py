# SPDX-License-Identifier: MIT
"""Verify the existing app's passage selection on a frozen mixed-writing sample."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from scripts import writing_adjacent_range_transfer as trial

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.writing-adjacent-range-app/1"
BILLING = {"price_usd_cents": 999, "simulated": True, "jobs_started_before_payment": 0}


def sample(rows):
    chosen = {}
    originals = {}
    for row in rows:
        if row["family"] == "calibration":
            continue
        if row["delivery"] == "original":
            originals[row["group"]] = row
            continue
        key = row["genre"], row["family"], len(row["target"])
        group = row["group"]
        if key not in chosen or group < chosen[key]["group"]:
            chosen[key] = row
    trial.require(len(chosen) == 8, "The app sample lacks a genre, generator, or edit width")
    return [row for _, edited in sorted(chosen.items()) for row in (originals[edited["group"]], edited)]


def outcomes(rows, ledger):
    sources = {(row["group"], row["delivery"]): row for row in sample(ledger["rows"])}
    checks = ("passage_classifier", "writing_pattern_ratio")
    result = defaultdict(Counter)
    for row in rows:
        source = sources.pop((row["group"], row["delivery"]))
        ordinary = source["delivery"] == "original"
        trial.require(row["sha256"] == source["sha256"] and set(row["checks"]) == set(checks),
                      "The worker report does not bind to the frozen source and checks")
        words = source["words"]
        target = set(source["target"]) if not ordinary else set()
        truth = {i for i, item in enumerate(words) if item[2] in target}
        useful, exact = False, False
        for card in row["spans"]:
            trial.require(bool(card) and all(0 <= a < b <= source["spans"][-1][1] for a, b in card),
                          "A displayed range is outside the analyzed excerpt")
            picked = {i for i, (a, b, _pos) in enumerate(words) if any(x <= a < b <= y for x, y in card)}
            useful |= trial.words.useful_words(picked, set(source["changed"]))
            exact |= bool(truth) and picked == truth
        lane = source["genre"], str(len(source["target"])), source["delivery"]
        result["/".join(lane)].update(files=1, displayed_review=bool(row["spans"]), useful=useful, exact=exact,
                                     inconclusive=row["decision"] == "inconclusive",
                                     classifier_completed=row["checks"]["passage_classifier"] == "completed",
                                     ratio_completed=row["checks"]["writing_pattern_ratio"] == "completed")
    trial.require(not sources and len(rows) == 16, "A matched delivery is missing from the app sample")
    return result


def app(plan, ledger, output, device):
    from scripts import video_short_return_encoded_app as paid
    from scripts.writing_revision_worker_transfer import upload

    trial.review(ledger)
    trial.require(list(map(trial.public, plan["rows"])) == ledger["rows"], "Native prose differs from the frozen ledger")
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_RUN_WORKER="false", VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_DEVICE=device,
                      VERITAS_WEBHOOKS_ENABLED="false", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth, jobs
    from app.billing import service
    from app.engine.models import accelerator_kind
    from app.main import app as server
    from app.storage.db import Analysis, get_session
    from app.storage.objects import default_store

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()
    measured = []
    with (output / "observations.jsonl").open("x") as progress, TestClient(server, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-writing-range-report-audit", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        worker = jobs.build_worker()
        for row in sample(plan["rows"]):
            trial.require(trial.pool.sha(row["text"]) == row["sha256"], "Native excerpt bytes changed")
            aid = upload(client, stub, row, worker)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            exported = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
            trial.require(evidence == exported["evidence"] and evidence["status"] == "ready"
                          and evidence["source"]["sha256"] == evidence["source"]["extracted_text_sha256"] == row["sha256"],
                          "The app and JSON must agree on the source")
            packets = []
            for packet in evidence["packets"]:
                focus = packet["focus"]
                trial.require(focus["text"] == row["text"][focus["start"]:focus["end"]],
                              "A finding does not match the original excerpt")
                packets.append([[span["start"], span["end"]] for span in packet["inspection_scope"]["spans"]])
            with get_session() as db:
                record = db.get(Analysis, aid)
                trial.require(record.status == "done", "The prepaid analysis did not complete")
                decision = json.loads(default_store().read_bytes(record.result_hash))["decision"]["code"]
            observation = {key: row[key] for key in ("group", "delivery", "sha256")} | {
                "spans": packets, "decision": decision, "checks": evidence["coverage"]["writing_check_coverage"]["checks"],
            }
            measured.append(observation)
            print(json.dumps(observation, sort_keys=True), file=progress, flush=True)
            print(f"Verified {len(measured)}/16 app reports.", flush=True)
    return {"schema": SCHEMA, "billing": BILLING, "device": accelerator_kind(),
            "rows": measured, "results": outcomes(measured, ledger)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("directory_or_receipt", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--device", choices=("cpu", "auto"), default="cpu")
    args = parser.parse_args()
    ledger = trial.read(args.ledger)
    pin = trial.pool.file_hash(args.ledger)
    if args.plan:
        result = app(trial.read(args.plan), ledger, args.directory_or_receipt, args.device)
        result["requested_device"] = args.device
        result["ledger_sha256"] = pin
        output = args.directory_or_receipt / "receipt.json"
        with output.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
            stream.write("\n")
    else:
        receipt = json.loads(args.directory_or_receipt.read_bytes())
        trial.review(ledger)
        trial.require(receipt["schema"] == SCHEMA and receipt["billing"] == BILLING
                      and receipt["ledger_sha256"] == pin and outcomes(receipt["rows"], ledger) == receipt["results"],
                      "The app receipt does not reproduce")
        print(json.dumps(receipt["results"], sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
