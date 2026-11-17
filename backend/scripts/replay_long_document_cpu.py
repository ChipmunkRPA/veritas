"""Replay long-document writing analysis under the local CPU memory limit."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from scripts import video_short_return_encoded_app as paid
from scripts.audio_stereo_timing_change_app import local_model_view

ROOT = Path(__file__).resolve().parents[2]
MODEL_MEMORY_MB = 7600
APP_TIMEOUT_S = 180
CALCULATION_TIMEOUT_S = 600


def configure(output, compact):
    paid.configure(output, ROOT)
    models = local_model_view(output)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE="cpu", VERITAS_LOG_LEVEL="WARNING",
                      VERITAS_MODELS_DIR=str(models), VERITAS_HF_HOME=str(models / "hf"),
                      VERITAS_MODEL_MEMORY_BUDGET_MB=str(MODEL_MEMORY_MB), VERITAS_JOB_TIMEOUT_S=str(APP_TIMEOUT_S),
                      VERITAS_TEXT_RATIO_COMPACT_CPU=str(compact).lower(), VERITAS_WEBHOOKS_ENABLED="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)


def source_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def calculate(source, output, *, compact):
    configure(output, compact)

    from app.core.canonical import canonical_hash
    from app.core.config import settings
    from app.detectors.text.adapters.learned import BinocularsDetector
    from app.detectors.text.binoculars import Binoculars
    from app.detectors.text.pipeline import decode_document
    from app.engine.boundary import Deadline, InMemoryRunStore, run_detector
    from app.engine.detector import Profile, RunContext, StageOutputs
    from app.engine.models import get_model_manager

    doc, sentences = decode_document(source, source.suffix, prose_only=settings.text_bibliography_scope_review)
    outputs = StageOutputs(document={"text": doc.text, "paragraphs": doc.paragraphs, "meta": doc.meta,
                                     "source_map": doc.source_map, "sentences": [asdict(item) for item in sentences]})
    detector = BinocularsDetector()
    models = get_model_manager()
    keys = (detector.OBSERVER, detector.PERFORMER)
    limits = models.lease_terms(keys, prefer="cpu")
    assert limits.weight_bytes > models.budget_bytes == MODEL_MEMORY_MB * 1024**2
    invoke = models._load
    trace, token_readout = [], {}

    def load(entry, device, dtype):
        loaded = invoke(entry, device, dtype)
        if entry.id in keys:
            trace.append(keys.index(entry.id))
        return loaded

    token_scores = Binoculars.token_scores

    def observe(runtime, *args, **kwargs):
        result = token_scores(runtime, *args, **kwargs)
        token_readout.update({
            "token_count": int(result.valid_mask().sum()),
            "context_count": len(result.input_contexts),
            "contexts_sha256": canonical_hash(result.input_contexts),
            "nll_sha256": hashlib.sha256(result.obs_nll.astype("<f4", copy=False).tobytes()).hexdigest(),
            "cross_entropy_sha256": hashlib.sha256(result.xent.astype("<f4", copy=False).tobytes()).hexdigest(),
        })
        return result

    ctx = RunContext("long-writing-public-replay", None, Profile.FULL, "cpu", "float32", None,
                     outputs, Deadline(CALCULATION_TIMEOUT_S), None, None)
    with patch.object(models, "_load", side_effect=load), patch.object(Binoculars, "token_scores", observe):
        result = run_detector(detector, ctx, store=InMemoryRunStore(), models=models, timeout_s=CALCULATION_TIMEOUT_S)
    assert result.status == "ok" and token_readout, result.reason
    assert all(row.pins == 0 for row in models.snapshot())
    assert models.free_bytes() == models.budget_bytes - sum(row.bytes_declared for row in models.snapshot())
    proof = {"source_sha256": source_hash(source), "mode": "grouped" if compact else "per_window",
             "calculation_timeout_s": CALCULATION_TIMEOUT_S, "memory_limit_mb": MODEL_MEMORY_MB, "cpu_threads": 2,
             "writing_model_loads": len(trace), "writing_model_load_order": trace,
             "elapsed_s": result.timings["total_s"], "document_evidence_sha256": canonical_hash(result.details["text_evidence"]),
             "document_score_hex": result.aggregate.raw.hex(), "sentence_scores": len(result.unit_scores), **token_readout}
    save(output / "receipt.json", proof)
    print(json.dumps(proof, sort_keys=True), flush=True)


def compare(reference, compact):
    left, right = (json.loads(path.read_text()) for path in (reference, compact))
    return compare_results(left, right)


def compare_results(left, right):
    assert (left["mode"], right["mode"]) == ("per_window", "grouped")
    measured = ("source_sha256", "memory_limit_mb", "cpu_threads", "context_count", "token_count",
                "contexts_sha256", "nll_sha256", "cross_entropy_sha256", "document_evidence_sha256",
                "document_score_hex", "sentence_scores")
    assert all(left[key] == right[key] for key in measured)
    assert right["writing_model_loads"] < left["writing_model_loads"]
    return {key: right[key] for key in measured} | {
        "writing_model_loads": [left["writing_model_loads"], right["writing_model_loads"]],
        "writing_check_seconds": [left["elapsed_s"], right["elapsed_s"]],
    }


def verify_panel(path):
    panel = json.loads(path.read_text())
    assert panel["schema"] == "chipmunk.writing-cpu-window-reuse/1"
    comparisons = {}
    for item in panel["calculations"]:
        name = item["case"]
        assert name not in comparisons
        comparisons[name] = compare_results(item["per_window"], item["grouped"])
    assert set(comparisons) == {"federal_register", "original_text", "edited_text"}
    previous = panel["earlier_app_failure"]
    long = panel["long_app"]
    assert (previous["source_sha256"], previous["status"], previous["writing_check_status"]) == (
        long["sha256"], "failed", "timeout")
    assert long["app_timeout_s"] == panel["additional_app"]["app_timeout_s"] == APP_TIMEOUT_S
    assert long["unpaid_jobs"] == panel["additional_app"]["unpaid_jobs"] == 0
    assert long["simulated_usd_cents"] == panel["additional_app"]["simulated_usd_cents"] == 999
    cases = {value["sha256"]: value for value in [long, *panel["additional_app"]["cases"]]}
    assert len(cases) == 4
    for proof in comparisons.values():
        delivery = cases[proof["source_sha256"]]
        assert all(delivery[key] == proof[key] for key in ("token_count", "context_count", "document_evidence_sha256"))
    for delivery in cases.values():
        assert 0 < delivery["analysis_seconds"] <= APP_TIMEOUT_S
        assert delivery["signed_source_replay"] and len(delivery["pdf_sha256"]) == 64
        assert delivery["writing_coverage"]["writing_pattern_ratio"] == "completed"
    return {"exact_comparisons": comparisons, "app_deliveries": len(cases), "unpaid_jobs": 0}


def app_exports(client, aid, source, output):
    from zipfile import ZipFile

    from app.api.public_projection import _identities
    from app.core.config import settings
    from app.detectors.text.pipeline import decode_document
    from app.evidence.document_check_coverage import KEY
    from app.reports.engine import pdf_text
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    doc, _ = decode_document(source, source.suffix, prose_only=settings.text_bibliography_scope_review)
    _, names = _identities()
    source_names = {name for name in names if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", doc.text, re.I)}
    route = f"/api/v2/analyses/{aid}"
    evidence = paid.body(client.get(f"{route}/evidence-packets"))
    exported = paid.fetch(client, f"/api/analyses/{aid}/export.json")
    assert json.loads(exported)["evidence"] == evidence
    for response in (exported, paid.fetch(client, route), paid.fetch(client, f"{route}/detector-runs")):
        scan(response.decode(), source_names=source_names)
    assert evidence["coverage"][KEY]["checks"]["writing_pattern_ratio"] == "completed"

    report = paid.body(client.post(f"{route}/reports", json={"profile": "standard"}))
    report = paid.wait(client, f"/api/v2/reports/{report['id']}", timeout=180)
    pdf = paid.fetch(client, f"/api/v2/reports/{report['id']}/file")
    words = pdf_text(pdf)
    scan(words, source_names=source_names)
    assert "Writing-pattern ratio" in words
    (output / "report.pdf").write_bytes(pdf)
    package = paid.body(client.post(f"{route}/packages",
                                    json={"report_id": report["id"], "kind": "draft", "include_original": False}))
    package = paid.wait(client, f"/api/v2/packages/{package['id']}", timeout=180)
    path = output / "evidence.zip"
    path.write_bytes(paid.fetch(client, f"/api/v2/packages/{package['id']}/file"))
    key = paid.body(client.get("/api/v2/instance/public-key"))
    signed = verify_package(path, public_key_pem=key["public_key_pem"].encode(), source_document=source)
    assert signed.outcome == "verified", signed.as_dict()
    with ZipFile(path) as archive:
        assert json.loads(archive.read("evidence.json")) == evidence and archive.read("report.pdf") == pdf
    return {"pdf_sha256": hashlib.sha256(pdf).hexdigest(), "signed_source_replay": True,
            "writing_coverage": evidence["coverage"][KEY]["checks"]}


def run_app(sources, output):
    configure(output, True)

    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.canonical import canonical_hash
    from app.main import app
    from app.storage.db import AnalysisStage, Artefact, get_session
    from app.storage.objects import default_store

    service._stripe = stub = paid.PaymentStub()
    receipts = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-long-document-replay", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for index, source in enumerate(sources):
            media_type = mimetypes.guess_type(source.name)[0]
            start = time.monotonic()
            aid = paid.checkout(client, stub, source, kind="document", media_type=media_type, filename=source.name)
            elapsed = round(time.monotonic() - start, 3)
            with get_session() as db:
                stage = db.get(AnalysisStage, (aid, "detect"))
                output_blob = db.get(Artefact, stage.output_artefact_id)
                entries = json.loads(default_store().read_bytes(output_blob.sha256))["detectors"]
            [ratio] = [row for row in entries if row["detector_id"] == "txt.binoculars"]
            assert ratio["status"] == "ok", ratio["reason"]
            raw = ratio["details"]["text_evidence"]
            token_count = raw["document_score"]["tokens_scored"]
            assert sum(c["scored_token_range"][1] - c["scored_token_range"][0] for c in raw["contexts"]) == token_count
            delivery_dir = output / f"case-{index + 1}"
            delivery_dir.mkdir()
            receipts.append({"filename": source.name, "sha256": source_hash(source), "analysis_seconds": elapsed,
                             "token_count": token_count, "context_count": len(raw["contexts"]),
                             "document_evidence_sha256": canonical_hash(raw),
                             **app_exports(client, aid, source, delivery_dir)})
            print(source.name, elapsed, token_count, len(raw["contexts"]), flush=True)
    assert len(stub.sessions) == len(receipts)
    save(output / "receipt.json", {"app_timeout_s": APP_TIMEOUT_S, "memory_limit_mb": MODEL_MEMORY_MB,
                                   "unpaid_jobs": 0, "simulated_usd_cents": 999, "cases": receipts})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path, nargs="?")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--per-window", action="store_true")
    mode.add_argument("--app", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--verify-panel", action="store_true")
    cli.add_argument("--also", type=Path, action="append", default=[])
    args = cli.parse_args()
    if args.verify_panel:
        print(json.dumps(verify_panel(args.source), indent=2))
    elif args.output is None:
        cli.error("an output path or a second receipt is required")
    elif args.verify:
        print(json.dumps(compare(args.source, args.output), indent=2))
    elif args.app:
        run_app([args.source, *args.also], args.output)
    else:
        calculate(args.source, args.output, compact=not args.per_window)
