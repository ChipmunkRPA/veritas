"""Compare body-only and reference-inclusive inputs through the paid local worker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from zipfile import ZipFile

from scripts import video_short_return_encoded_app as paid
from scripts import writing_bibliography_scope as study
from scripts.audio_vocal_event_mixtures import save, sha
from scripts.writing_editorial_worker_audit import plan as revision_cases

MEDIA_TYPES = {"txt": "text/plain", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
               "pdf": "application/pdf"}
ROLES = ("reference_included", "reference_excluded", "body_control")


def examples(args):
    held = list(study.cases(args.elife, args.journals))
    selected = []
    for publisher, ext in zip(study.PUBLISHERS, ("docx", "pdf", "txt"), strict=True):
        item, body, refs = next(group for group in held if group[0]["publisher"] == publisher)
        selected.append({"publisher": publisher, "source": item["doi"], "role": "ordinary", "extension": ext,
                         "body": body, "refs": refs})
    revisions = revision_cases(args.revisions)
    for technology in ("qwen", "smol2"):
        row = next(case for case in revisions if case["role"] == technology
                   and study.article(args.journals / f"{case['group']}-after.xml", case["publisher"]) is not None)
        article, _, refs = study.article(args.journals / f"{row['group']}-after.xml", row["publisher"])
        selected.append({"publisher": article["publisher"], "source": article["doi"], "role": technology, "extension": "txt",
                         "body": [row["text"]], "refs": refs, "focus": row["focus"]})
    return selected


def worker_record(aid):
    from sqlalchemy import select

    from app.storage.db import AnalysisStage, Artefact, get_session
    from app.storage.objects import default_store

    with get_session() as db:
        store = default_store()
        outputs = {}
        for name in ("decode", "detect"):
            stage = db.scalar(select(AnalysisStage).where(AnalysisStage.analysis_id == aid, AnalysisStage.stage == name))
            assert stage.status == "done"
            artifact = db.get(Artefact, stage.output_artefact_id)
            outputs[name] = json.loads(store.read_bytes(artifact.sha256))
        return outputs["decode"], outputs["detect"]


def payload(client, aid, path, role):
    from app.evidence import document_bibliography, document_check_coverage
    from scripts.verify_multimedia_exports import scan

    evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
    exported = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
    assert evidence == exported["evidence"] and evidence["source"]["sha256"] == sha(path)
    scan(json.dumps(exported))
    record = evidence["coverage"].get(document_bibliography.KEY)
    assert (record is not None) == (role == "reference_excluded")
    bibliography = record["body_chars"] if record else None
    decoded, detected = worker_record(aid)
    assert record == decoded["meta"].get(document_bibliography.KEY)
    windows = [{"focus": [item["focus"]["start"], item["focus"]["end"]], "ranges": [
        [span["start"], span["end"]] for window in item["supporting_windows"] for span in window["spans"]]}
        for item in evidence["packets"]]
    if bibliography is not None:
        assert all(end <= bibliography for block in windows for _, end in block["ranges"])
    # Bind immutable per-unit responses instead of depending on report navigation order.
    responses = {entry["detector_id"]: {"status": entry["status"], "aggregate": entry["aggregate"], "units": entry["unit_scores"]}
                 for entry in detected["detectors"] if (entry.get("details") or {}).get("text_evidence")}
    checks = evidence["coverage"][document_check_coverage.KEY]["checks"]
    assert checks.get("passage_classifier") == "completed"
    return evidence, decoded["text"], responses, windows, checks


def report(client, aid, source, evidence, output):
    import pymupdf

    from app.evidence import document_bibliography
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    info = paid.body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    info = paid.wait(client, f"/api/v2/reports/{info['id']}")
    data = paid.fetch(client, f"/api/v2/reports/{info['id']}/file")
    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        text = " ".join(" ".join(page.get_text().split()) for page in pdf)
        record = evidence["coverage"][document_bibliography.KEY]
        assert document_bibliography.notice(record) in text and document_bibliography.location_label(record) in text
        scan(text)
    bundle = paid.body(client.post(f"/api/v2/analyses/{aid}/packages", json={"report_id": info["id"], "kind": "draft",
                                                                       "include_original": False}))
    bundle = paid.wait(client, f"/api/v2/packages/{bundle['id']}")
    path = output / "signed-package.zip"
    path.write_bytes(paid.fetch(client, f"/api/v2/packages/{bundle['id']}/file"))
    key = paid.body(client.get("/api/v2/instance/public-key"))["public_key_pem"].encode()
    checked = verify_package(path, public_key_pem=key, source_document=source)
    assert checked.outcome == "verified", checked.as_dict()
    with ZipFile(path) as package:
        assert json.loads(package.read("evidence.json")) == evidence and package.read("report.pdf") == data
    (output / "standard.pdf").write_bytes(data)


def run(args):
    paid.configure(args.output, study.ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DEVICE="mps", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_WEBHOOKS_ENABLED="false", VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_TEXT_BIBLIOGRAPHY_SCOPE_REVIEW="false",
                      VERITAS_RATE_LIMIT_PER_MINUTE="1800", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    selected = examples(args)

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.main import app
    from app.storage.db import get_session

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available()
    stub = paid.PaymentStub()
    service._stripe = stub
    outcomes = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-bibliography-review", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        for index, case in enumerate(selected):
            folder = args.output / f"{index:02}"
            folder.mkdir()
            ext = case["extension"]
            entire, body = folder / f"with-references.{ext}", folder / f"body.{ext}"
            study.render(entire, [*case["body"], "References", *case["refs"]])
            study.render(body, case["body"])
            observed = {}
            for role, source in zip(ROLES, (entire, entire, body), strict=True):
                settings.text_bibliography_scope_review = role != "reference_included"
                aid = paid.checkout(client, stub, source, kind="document", media_type=MEDIA_TYPES[ext], filename=source.name)
                evidence, text, responses, windows, checks = payload(client, aid, source, role)
                if role == "reference_excluded" and case["role"] == "ordinary":
                    report(client, aid, source, evidence, folder)
                observed[role] = {"text": text, "responses": responses, "windows": windows, "checks": checks}
            scoped, clean = observed["reference_excluded"], observed["body_control"]
            assert scoped == clean, {"source": case["source"], "parity": {key: scoped[key] == clean[key] for key in clean}}
            included = observed["reference_included"]
            cutoff = len(clean["text"])
            assert included["text"].startswith(clean["text"] + "\n\nReferences\n\n")
            outcomes.append({key: case[key] for key in ("publisher", "source", "role", "extension")}
                            | {"original_sha256": sha(entire), "body_characters": cutoff, "focus": case.get("focus"),
                               "body_control_exact_parity": True, "checks": scoped["checks"],
                               "scope_on": scoped["windows"], "scope_off": included["windows"]})
    save(args.output / "receipt.json", {"cases": outcomes, "unpaid_jobs": 0, "simulated_paid_jobs": len(stub.sessions),
                                        "signed_exports_checked": 3})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    journal = study.DATA / "writing-journal-body-transfer-2026-11-11"
    parser.add_argument("--elife", type=Path, default=study.DATA / "writing-elife-body-replication-2026-11-11/publisher")
    parser.add_argument("--journals", type=Path, default=journal / "publisher")
    parser.add_argument("--revisions", type=Path, default=journal / "scored/cases.json")
    run(parser.parse_args())
