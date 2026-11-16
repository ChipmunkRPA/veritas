# SPDX-License-Identifier: MIT
"""Replay the paid CPU writing flow under the local two-model memory limit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_structured_range_transfer as study

ROOT = Path(__file__).resolve().parents[2]
LIMIT_MB = 7600


def sampled(plan_path, receipt_path):
    plan, receipt = study.read(plan_path), study.read(receipt_path)
    assert study.pool.file_hash(plan_path) == receipt["plan_sha256"]
    assert list(map(study.adjacent.public, plan["rows"])) == receipt["rows"]
    selected = []
    for genre, family in (("reports", "gemini-2.5-flash"), ("news", "gpt-5.4")):
        group = min(row["group"] for row in plan["rows"] if row["genre"] == genre and row["family"] == family)
        paired = [row for row in plan["rows"] if row["group"] == group]
        assert [row["delivery"] for row in paired] == ["original", "edited"]
        selected.extend(paired)
    return selected


def main(plan_path, receipt_path, output):
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE="cpu", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_MODEL_MEMORY_BUDGET_MB=str(LIMIT_MB), VERITAS_JOB_TIMEOUT_S="420",
                      VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_TEXT_WINDOW_SENTENCES="3",
                      VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="true", VERITAS_WEBHOOKS_ENABLED="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import numpy as np
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.detectors.text.adapters.learned import BinocularsDetector
    from app.detectors.text.binoculars import Binoculars, document_score, sentence_score_records
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from app.engine.models import get_model_manager
    from app.evidence.document_check_coverage import KEY as COVERAGE_KEY
    from app.main import app
    from app.reports.engine import pdf_text
    from app.storage.db import AnalysisStage, Artefact, get_session
    from app.storage.objects import default_store
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    detector = BinocularsDetector
    original_fallback = detector.run_after_memory_refusal
    service._stripe = stub = paid.PaymentStub()

    def run_entry(aid):
        with get_session() as db:
            row = db.get(AnalysisStage, (aid, "detect"))
            raw = db.get(Artefact, row.output_artefact_id)
            stage = json.loads(default_store().read_bytes(raw.sha256))
        return next(item for item in stage["detectors"] if item["detector_id"] == detector.spec.id)

    def checkout(client, path):
        return paid.checkout(client, stub, path, kind="document", media_type="text/plain", filename=path.name)

    sources = sampled(plan_path, receipt_path)
    # Preserve one real multi-window case with entirely ordinary words.
    seen = {row["group"] for row in sources}
    left = [row["text"] for row in sources if row["delivery"] == "original"]
    remaining = [row["text"] for row in study.read(plan_path)["rows"]
                 if row["delivery"] == "original" and row["group"] not in seen]
    long_text = "\n\n".join([*left, *remaining[:4]])
    sources += [{"group": None, "genre": "mixed", "family": "ordinary", "delivery": "long_control",
                 "sha256": study.pool.sha(long_text), "text": long_text}]
    observations = []
    baseline_windows = None
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-writing-cpu-replay", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        for i, source in enumerate(sources):
            path = output / f"sample-{i}.txt"
            path.write_bytes(source["text"].encode())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
            if i == 0:
                try:
                    detector.run_after_memory_refusal = lambda *args, **kwargs: None
                    entry = run_entry(checkout(client, path))
                    assert (entry["status"], entry["reason"]) == ("skipped", "memory budget")
                finally:
                    detector.run_after_memory_refusal = original_fallback
            aid = checkout(client, path)
            entry = run_entry(aid)
            assert entry["status"] == "ok" and len(entry["model_revisions"]) == 2, entry["reason"]
            raw = entry["details"]["text_evidence"]
            score = raw["document_score"]
            assert raw["shared_tokenization_verified"] and score["threshold"] == .9505
            assert raw["windows"] and sum(w["scored_token_range"][1] - w["scored_token_range"][0]
                                          for w in raw["contexts"]) == score["tokens_scored"]
            if source["delivery"] == "long_control":
                assert len(raw["contexts"]) >= 2
            if i == 0:
                baseline_windows = {w["unit_ref"]: w["score_exact_hex"] for w in raw["windows"]}
            route = f"/api/v2/analyses/{aid}"
            packets = paid.body(client.get(f"{route}/evidence-packets"))
            export = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
            assert export["evidence"] == packets
            assert packets["coverage"][COVERAGE_KEY]["checks"]["writing_pattern_ratio"] == "completed"
            scan(json.dumps(export))
            if i in (0, 4):
                report = paid.body(client.post(f"{route}/reports", json={"profile": "standard"}))
                report = paid.wait(client, f"/api/v2/reports/{report['id']}", timeout=120)
                pdf = paid.fetch(client, f"/api/v2/reports/{report['id']}/file")
                text = pdf_text(pdf)
                scan(text)
                assert "Writing-pattern ratio" in text
                path.with_suffix(".pdf").write_bytes(pdf)
            observations.append({key: source[key] for key in ("group", "genre", "family", "delivery", "sha256")} | {
                "raw": score["score"].hex(), "review_side": "ai_like" if score["score"] < score["threshold"] else "other",
                "contexts": len(raw["contexts"]), "tokens": score["tokens_scored"], "scored_sentences": len(raw["windows"]),
                "context_input_sha256": [w["token_ids_sha256"] for w in raw["contexts"]],
                "direction": export["customer_assessment"]["direction"], "report_exercised": i in (0, 4),
            })
            print(i, source["delivery"], len(raw["contexts"]), score["score"], flush=True)

        # Apply the original CPU joint-logit routine to the actual native forwards.
        # The proxies make each forward use a separate managed lease because this
        # 18 GiB machine does not have RAM available for a native simultaneous CPU pass.
        models = get_model_manager()
        model_ids = (detector.OBSERVER, detector.PERFORMER)
        with models.lease(model_ids[:1], prefer="cpu") as lease:
            tokenizer = lease.get(model_ids[0]).obj[0]

        def native_forward(model_id):
            def forward(ids):
                allowance = models.activation_bytes_for_items({model_id: 1})
                with models.lease((model_id,), prefer="cpu", activation_bytes=allowance) as lease:
                    return lease.get(model_id).obj[1](ids)
            return forward

        reference = Binoculars.from_parts(tokenizer, *map(native_forward, model_ids), "cpu")
        count = len(tokenizer(sources[0]["text"], add_special_tokens=False)["input_ids"]) + 1
        with models.lease((), prefer="cpu", activation_bytes=count * detector.VOCABULARY_SIZE * 4):
            baseline = reference.token_scores(sources[0]["text"])
        actual, tokens = document_score(baseline)
        assert round(actual, 6).hex() == observations[0]["raw"] and tokens == observations[0]["tokens"]
        measured = sentence_score_records(baseline, segment(from_string(sources[0]["text"])))
        assert {f"sentence:{i}": item["score"].hex() for i, item in enumerate(measured)
                if item["score"] is not None} == baseline_windows
        assert np.isfinite(baseline.xent).all()
        assert [w["token_ids_sha256"] for w in baseline.input_contexts] == observations[0]["context_input_sha256"]
    proof = {"schema": "chipmunk.text-ratio-cpu-memory-app/1", "cpu_model_budget_mb": LIMIT_MB,
             "before_status": "skipped_memory_budget", "prepayment_jobs": 0, "simulated_usd_cents": 999,
             "legacy_cpu_arithmetic_matched": True, "exact_sentence_matches": len(baseline_windows),
             "cases": observations, "checkouts": len(stub.sessions)}
    assert len(stub.sessions) == 6
    (output / "receipt.json").write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "receipt", "output"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    main(args.plan, args.receipt, args.output)
