"""Replay paid document analysis, archive evidence, and signed exports locally."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

from scripts import video_short_return_encoded_app as paid
from scripts import writing_multi_passage_panel as panel
from scripts import writing_native_passage_panel as documents
from scripts.writing_multi_passage_app import PORTABLE_CHECK, ROOT
from scripts.writing_native_passage_app import MIME
from scripts.writing_report_app import sample
from scripts.writing_report_sources import GAO_SHA256

CASE = "GAO_GAO-15-149"


def exports(client, aid, source, directory, evidence):
    import pymupdf

    from app.evidence import document_published_wording as wording
    from app.reports.verify import verify_package
    from scripts.verify_multimedia_exports import scan

    comparison = evidence["coverage"][wording.KEY]
    assert comparison["authorship_confidence_contribution"] == 0
    public = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
    assert public["evidence"]["coverage"][wording.KEY] == comparison
    scan(json.dumps(public))
    report = paid.body(client.post(f"/api/v2/analyses/{aid}/reports", json={"profile": "standard"}))
    paid.wait(client, f"/api/v2/reports/{report['id']}")
    pdf = paid.fetch(client, f"/api/v2/reports/{report['id']}/file")
    (directory / "report.pdf").write_bytes(pdf)
    with pymupdf.open(stream=pdf, filetype="pdf") as rendered:
        text = " ".join(" ".join(page.get_text().split()) for page in rendered)
        scan(text)
        assert wording.TITLE in text and comparison["reference"]["published_date"] in text
        assert comparison["reference"]["id"] in text
        for row in comparison["changes"]:
            start, end = row["start"], row["end"]
            position = f"[{start}, {end})" if start != end else f"Omission at extracted character {start}."
            assert position in text
            if source.suffix == ".pdf":
                pages = ", ".join(map(str, sorted({place["page"] for place in row["locations"]})))
                assert f"Original PDF pages {pages}." in text
            else:
                label = "Original text characters" if source.suffix == ".txt" else f"Word paragraph {row['locations'][0]['paragraph']} characters"
                assert label in text
        if comparison["published_style_matches"]:
            assert "Writing highlights with exact published wording" in text
            assert "Inspect the highlighted passages in context." not in text
    package = paid.body(client.post(f"/api/v2/analyses/{aid}/packages", json={
        "report_id": report["id"], "kind": "draft", "include_original": False,
    }))
    paid.wait(client, f"/api/v2/packages/{package['id']}")
    archive = directory / "evidence.zip"
    archive.write_bytes(paid.fetch(client, f"/api/v2/packages/{package['id']}/file"))
    key = directory / "public-key.pem"
    key.write_text(paid.body(client.get("/api/v2/instance/public-key"))["public_key_pem"])
    replay = verify_package(archive, public_key_pem=key.read_bytes(), source_document=source)
    assert replay.ok, replay.as_dict()
    assert any("Verified earlier wording against the retained transcript" in check.detail for check in replay.checks)
    subprocess.run([sys.executable, "-B", "-c", PORTABLE_CHECK, str(archive), str(key)], check=True, timeout=20)
    with ZipFile(archive) as packaged:
        frozen = json.loads(packaged.read("evidence.json"))
        assert frozen["coverage"][wording.KEY] == comparison
        scan(packaged.read("result.json").decode())
    return panel.digest(pdf)


def run(plan, font, output, browser):
    paid.configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_TEXT_DETECTOR_MODE="candidate",
        VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_TEXT_PASSAGE_INTERVALS_REVIEW="true",
        VERITAS_TEXT_PASSAGE_DOCUMENT_BASELINE_REVIEW="true", VERITAS_TEXT_PASSAGE_REPORT_REVIEW="true",
        VERITAS_DOCUMENT_GAO_AUTOMATIC_REVIEW="true", VERITAS_DOCUMENT_GAO_ARCHIVE_SHA256=GAO_SHA256,
        VERITAS_DOCUMENT_GAO_ARCHIVE_PATH=str(ROOT / "backend/data/gao-archive/gao-pack.json"),
        VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="false", VERITAS_WEBHOOKS_ENABLED="false",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence import document_passage_intervals
    from app.evidence import document_published_wording as wording
    from app.main import app
    from app.storage.db import get_session
    from scripts.verify_multimedia_exports import scan
    from scripts.writing_character_view_app import tokens_assessed

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    source_plan = panel.read(plan)
    service._stripe = stub = paid.PaymentStub()
    rows, browser_cases = {}, []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-published-wording-test", "examiner")
            db.add(owner)
            db.flush()
            key, _ = auth.issue_api_key(db, owner)
            client.headers["X-API-Key"] = key
            db.commit()

        def analyze(carrier, version, *, case=None):
            case = case or f"{carrier}-{version}"
            directory = output / case
            directory.mkdir()
            source = directory / ("upload" + documents.EXTENSION[carrier])
            source.write_bytes(documents.fixture(carrier, sample(source_plan, CASE, version), font))
            aid = paid.checkout(client, stub, source, kind="document", media_type={"text": "text/plain", **MIME}[carrier], filename=source.name)
            evidence = paid.body(client.get(f"/api/v2/analyses/{aid}/evidence-packets"))
            scan(json.dumps(evidence))
            return aid, evidence, source, directory

        for carrier, version in (("text", "mixed"), ("word", "mixed"), ("pdf", "mixed"), ("text", "ordinary")):
            aid, evidence, source, directory = analyze(carrier, version)
            received = evidence["coverage"].get(wording.KEY)
            assert received is not None, (carrier, version, evidence["source"]["method"])
            assert received["reference"]["id"] == CASE.removeprefix("GAO_")
            assert len(received["changes"]) == (6 if version == "mixed" else 0), (carrier, len(received["changes"]))
            assert len(received["published_style_matches"]) == (1 if version == "ordinary" else 0)
            baseline = evidence["coverage"].get(document_passage_intervals.KEY)
            if version == "ordinary":
                assert len(baseline["ranges"]) == 1 and wording.display_intervals(baseline, received) == []
                ordinary_assessment, ordinary_ranges = tokens_assessed(aid), baseline
            name = f"{carrier}-{version}"
            rows[name] = {"file_sha256": panel.digest(source.read_bytes()), "source": received["reference"]["id"],
                          "changes": len(received["changes"]), "published_style_matches": len(received["published_style_matches"]),
                          "pdf_sha256": exports(client, aid, source, directory, evidence)}
            browser_cases.append({"analysis_id": aid, "case": name, "reference": received["reference"]["id"],
                                  "date": received["reference"]["published_date"], "changes": len(received["changes"]),
                                  "cleared": bool(received["published_style_matches"])})
            print(name, rows[name]["changes"], rows[name]["published_style_matches"], flush=True)
        settings.document_gao_automatic_review = False
        aid, control, _, _ = analyze("text", "ordinary", case="control")
        assert wording.KEY not in control["coverage"]
        assert tokens_assessed(aid) == ordinary_assessment
        assert control["coverage"][document_passage_intervals.KEY] == ordinary_ranges
        audit = {"cases": rows, "source_plan_sha256": panel.digest(plan.read_bytes()), "unchanged_overall_assessment": True,
                 "signed_packages_verified_with_original": len(rows), "simulated_prepaid_analyses": len(stub.sessions), "unpaid_jobs": 0}
        if browser:
            from scripts.image_jpeg_turned_copy_app import browser_check

            audit["browser"] = browser_check(output, app, key, browser_cases, scenario="writing-published-wording-replay.mjs")
    (output / "receipt.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "font", "output"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("--browser", action="store_true")
    args = cli.parse_args()
    run(args.plan, args.font, args.output, args.browser)
