# SPDX-License-Identifier: Apache-2.0
"""Replay dated editorial controls and matched rewrites through the paid Full worker."""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audio_vocal_event_mixtures import digest, read, save, sha

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "docs/evaluation-panels/writing-journal-body-transfer-2026-11-11.json"
SCHEMA = "chipmunk.writing-editorial-worker-audit/1"
ROLES = {"before", "edited", "qwen", "smol2"}
CHECKS = {"passage_classifier", "writing_pattern_ratio"}
CHECKOUT = {"simulated": True, "usd_minor": 999, "jobs_before_payment": 0}
METHOD = "Use every exact earlier, ordinary revised and model-rewritten text from the pinned journal receipt. Report actual cards and whole-document decisions separately. Measure overlap with the central sentence and compare selected sentence indexes across ordinary revisions. An ordinary selection is a review lead on expected-genuine text. No decision points are fitted."


def plan(path):
    source_pin = read(PARENT)["commitments"]["scored/cases.json"]
    if sha(path) != source_pin:
        raise ValueError("Text inputs differ from the pinned journal body panel")
    native = read(path)
    cases = []
    for row in native:
        text = row["text"]
        spans = sorted({tuple(pair) for win in row["windows"] for pair in win["spans"]})
        if not (len(spans) == 5 and row["target"] == 2 and list(spans[2]) == row["focus"]
                and row["role"] in ROLES and digest(text.encode()) == row["text_sha256"]):
            raise ValueError("Text or review geometry differs")
        target = row["focus"]
        cases.append({key: row[key] for key in ("group", "publisher", "role", "text", "text_sha256")} | {
            "ranges": [{"start": a, "end": b} for a, b in spans], "focus": target,
            "context": [" ".join(text[a:b] for a, b in win["spans"]) for win in row["windows"]],
            "spaces": [i for i, ch in enumerate(text) if ch.isspace()], "length": len(text),
            "record_id": str(len(cases)),
        })
    return cases


def covers(spans, focus, spaces):
    a, b = focus
    target = set(range(a, b)) - spaces
    if not target:
        return False
    for lo, hi in spans:
        area = set(range(lo, hi)) - spaces
        overlap = len(target & area)
        if overlap * 2 >= len(target) and overlap * 10 >= 3 * len(target | area):
            return True
    return False


def review(document):
    if document["schema"] != SCHEMA or document["parent_sha256"] != sha(PARENT) or document["method"] != METHOD:
        raise ValueError("Audit source or method changed")
    if document["checkout"] != CHECKOUT:
        raise ValueError("Simulated checkout contract changed")
    rows = document["rows"]
    expected = {"before": 21, "edited": 21, "qwen": 16, "smol2": 16}
    if Counter(row["role"] for row in rows) != expected or len({(row["group"], row["role"]) for row in rows}) != len(rows):
        raise ValueError("Audit must retain all planned cases")
    outcomes, ordinary, lanes = defaultdict(Counter), defaultdict(dict), defaultdict(set)
    for row in rows:
        ranges, focus, spaces, size = row["ranges"], row["focus"], set(row["spaces"]), row["length"]
        if (len(ranges) != 5 or focus != ranges[2] or not all(0 <= a < b <= size for a, b in ranges)
                or not all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:])) or not all(0 <= i < size for i in spaces)
                or not re.fullmatch(r"[a-f0-9]{64}", row["text_sha256"]) or set(row["checks"]) != CHECKS):
            raise ValueError("Invalid source extent or recorded check coverage")
        cards = row["cards"]
        if any(card["focus"] not in ranges or not card["spans"]
               or not all(0 <= a < b <= size for a, b in card["spans"]) for card in cards):
            raise ValueError("Invalid public card position")
        useful = any(covers(card["spans"], focus, spaces) for card in cards)
        kind = row["role"]
        cell = outcomes[row["publisher"], kind]
        cell.update(files=1, any_card=bool(cards), changed_center_covered=useful, cards_elsewhere=bool(cards) and not useful,
                    primary_completed=row["checks"]["passage_classifier"] == "completed",
                    ratio_completed=row["checks"]["writing_pattern_ratio"] == "completed")
        cell[f"result/{row['decision']}"] += 1
        if kind in ("before", "edited"):
            ordinary[row["group"]][kind] = {ranges.index(card["focus"]) for card in cards}
            if cards:
                lanes[row["publisher"]].add(row["group"])
    if any(set(values) != {"before", "edited"} for values in ordinary.values()) or len(ordinary) != 21:
        raise ValueError("Ordinary revision pairs are incomplete")
    pairs = Counter()
    for values in ordinary.values():
        former, later = values["before"], values["edited"]
        pairs.update(groups=1, flagged_either=bool(former or later), unchanged_focus_selected_in_both=bool((former & later) - {2}),
                     new_later_focus=bool(later - former), new_later_center=2 in later - former)
    return {"by_role": {"/".join(key): dict(value) for key, value in outcomes.items()}, "ordinary_pairs": dict(pairs),
            "ordinary_groups_flagged_by_publisher": {name: len(value) for name, value in lanes.items()}}


def run(source, output, *, resume):
    inputs = plan(source)
    from scripts import video_short_return_encoded_app as paid
    from scripts.writing_revision_worker_transfer import assess, upload

    progress = output / "progress.jsonl"
    records = []
    if resume:
        records = [json.loads(line) for line in progress.read_text().splitlines()]
        keys = ("group", "publisher", "role", "text_sha256")
        if (output / "receipt.json").exists() or len(records) >= len(inputs) or any(
            any(native[key] != row[key] for key in keys) for native, row in zip(inputs, records)
        ):
            raise ValueError("Progress differs from the pinned input prefix or is already complete")
    paid.configure(output, ROOT, resume=resume)
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
    from app.main import app
    from app.storage.db import get_session

    if not torch.backends.mps.is_available():
        raise RuntimeError("The audit is pinned to the local Apple GPU worker")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    stub = paid.PaymentStub()
    service._stripe = stub
    with progress.open("a" if resume else "x") as log, TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal(f"local-editorial-audit-{len(records)}", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        worker = jobs.build_worker()
        for row in inputs[len(records):]:
            analysis_id = upload(client, stub, row, worker)
            observed = assess(client, row, analysis_id, export=True)
            evidence = paid.body(client.get(f"/api/v2/analyses/{analysis_id}/evidence-packets"))
            checks = evidence["coverage"]["writing_check_coverage"]["checks"]
            record = {key: row[key] for key in ("group", "publisher", "role", "focus", "spaces", "length", "text_sha256")}
            record.update(ranges=[[r["start"], r["end"]] for r in row["ranges"]], checks=checks,
                          cards=observed["packets"], decision=observed["document_decision"])
            records.append(record)
            print(json.dumps(record, sort_keys=True), file=log, flush=True)
            print(f"Verified {len(records)}/{len(inputs)} paid analyses", flush=True)
    receipt = {"schema": SCHEMA, "parent_sha256": sha(PARENT), "method": METHOD, "rows": records,
               "checkout": CHECKOUT}
    receipt["results"] = review(receipt)
    save(output / "receipt.json", receipt)
    print(json.dumps(receipt["results"], sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "run", "review"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "review":
        receipt = read(args.path)
        observed = review(receipt)
        if observed != receipt["results"]:
            raise ValueError("Saved audit results do not recount")
        print(json.dumps(observed, indent=2, sort_keys=True))
    elif args.phase == "preflight":
        print(dict(Counter(row["role"] for row in plan(args.path))))
    elif args.source is not None:
        run(args.source, args.path, resume=args.resume)
    else:
        parser.error("Specify the exact retained private journal input file")


if __name__ == "__main__":
    main()
