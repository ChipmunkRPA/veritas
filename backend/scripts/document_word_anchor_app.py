"""Verify a matched Word revision and its report through the local prepaid analysis flow."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path

from scripts.audio_vocal_event_mixtures import digest, save
from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure

ROOT = Path(__file__).resolve().parents[2]
MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def run(output, root, news):
    configure(output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_DOCUMENT_REVISION_COMPARISON_ENABLED="true",
                      VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="true", VERITAS_TEXT_DETECTOR_MODE="candidate",
                      VERITAS_TEXT_WINDOW_SENTENCES="3", VERITAS_DOCUMENT_PAIRED_SEPARATE_REVIEW="true",
                      VERITAS_DOCUMENT_PAIRED_INDEPENDENT_REVIEW="true",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")

    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.document_revision_writing import WORD_INDEPENDENT_SCHEMA
    from app.main import app
    from app.storage.db import Analysis, get_session
    from scripts.document_word_anchor_transfer import PREFIX, frozen_readings, native_pairs
    from scripts.rebuild_writing_word_boundary_sources import docx
    from scripts.review_document_revision_export import review
    from scripts.verify_multimedia_exports import scan
    from scripts.writing_paired_scale_transfer import increases

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fixtures = {}
    for before, after, frozen in native_pairs(root, news, "news"):
        readings = frozen_readings(frozen)
        generated = {tuple(item["later"]) for item in frozen["truth"] if item["family"] != "ordinary"}
        if generated and "generated" not in fixtures:
            for (_, span), values in readings.items():
                if span in generated and values and min(increases(values)) >= math.log(2):
                    fixtures["generated"] = (before, after, list(span), frozen["source_sha256"])
                    break
        if not generated and readings and all(values and min(increases(values)) < math.log(2) for values in readings.values()):
            fixtures.setdefault("ordinary", (before, after, None, frozen["source_sha256"]))
    assert set(fixtures) == {"ordinary", "generated"}

    service._stripe = stub = PaymentStub()
    results = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-independent-word-review", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for kind, (before, after, truth, source_pins) in fixtures.items():
            expected = "joint_increase" if truth else "no_joint_increase"
            older, newer = (output / f"{kind}-{side}.docx" for side in ("earlier", "later"))
            older.write_bytes(docx(before))
            newer.write_bytes(docx(PREFIX + after))
            aid = checkout(client, stub, newer, kind="document", media_type=MIME, filename=newer.name)
            order = next(reversed(stub.sessions.values()))
            assert (order["amount_total"], order["currency"]) == (999, "usd")
            with get_session() as db:
                overall = db.get(Analysis, aid).result_hash
            endpoint = f"/api/v2/analyses/{aid}/document-revision-comparison"
            assert body(client.get(endpoint))["paired_check_available"]
            files = {"earlier": (older.name, older.read_bytes(), MIME)}
            settings.document_paired_independent_review = False
            previous = body(client.post(endpoint, files=files, data={"include_paired_check": "true"}))["comparison"]
            assert previous["paired_check"]["status"] == "outside_scope"
            settings.document_paired_independent_review = True
            data = {"include_pdf": "true", "include_paired_check": "true"}
            payload = body(client.post(endpoint, files=files, data=data))
            comparison = payload["comparison"]
            check = comparison["paired_check"]
            assert check["schema"] == WORD_INDEPENDENT_SCHEMA and check["status"] == expected
            assert check["authorship_confidence_contribution"] == 0 and comparison["changes"] == previous["changes"]
            scan(json.dumps(check))
            export = output / f"{kind}.json"
            save(export, comparison)
            checked = review(export, older, newer)
            assert checked["paired_word_sentence_ranges_verified"] == 6 * len(check["revisions"])
            if truth:
                changed = [value + len(PREFIX) for value in truth]
                assert any(row["later"]["context_spans"][1] == changed and row["status"] == expected for row in check["revisions"])
            pdf = base64.b64decode(payload["pdf"], validate=True)
            (output / f"{kind}.pdf").write_bytes(pdf)
            with pymupdf.open(stream=pdf, filetype="pdf") as document:
                pages = [" ".join(page.get_text().split()) for page in document]
                scan(" ".join(pages))
                for index, row in enumerate(check["revisions"], 1):
                    positions = []
                    for side in ("earlier", "later"):
                        location = row[side]["context_paragraphs"][1]
                        positions.append(f"Paragraph {location['paragraph']} [{location['start']}, {location['end']})")
                    assert any(all(phrase in page for phrase in (f"Changed sentence {index}", *positions)) for page in pages)
            with get_session() as db:
                assert db.get(Analysis, aid).result_hash == overall
            results[kind] = {"plain_text_sha256": source_pins, "word_sha256": [digest(p.read_bytes()) for p in (older, newer)],
                             "status_without_fallback": previous["paired_check"]["status"],
                             "status_with_fallback": expected, "verified": checked,
                             "paragraphs": [{side: row[side]["context_paragraphs"][1] for side in ("earlier", "later")}
                                            for row in check["revisions"]]}
    save(output / "receipt.json", {"cases": results, "paid_runs": len(results), "unpaid_jobs": 0,
                                    "overall_assessments_preserved": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "root", "news"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    run(args.output, args.root, args.news)
