# SPDX-License-Identifier: MIT
"""Test real earlier-draft writing checks across TXT, DOCX and PDF."""

from __future__ import annotations

import argparse
import base64
import json
import os
from itertools import permutations
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_structured_range_transfer as study

ROOT = Path(__file__).resolve().parents[2]
FORMATS = {".txt": "text/plain", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
           ".pdf": "application/pdf"}
SOURCE_ROLES = (("reports", "gemini-2.5-flash"), ("news", "gpt-5.4"))


def native_pairs(plan, receipt, output):
    from app.evidence.document_revision import compare_with_sources
    from app.evidence.document_revision_writing import mixed_anchored_revisions
    from scripts.replay_document_mixed_formats import rendition

    private = study.read(plan)
    public = study.read(receipt)
    assert study.pool.file_hash(plan) == public["plan_sha256"]
    assert list(map(study.adjacent.public, private["rows"])) == public["rows"]
    earlier_by_group = {row["group"]: row for row in private["rows"] if row["delivery"] == "original"}
    for genre, family in SOURCE_ROLES:
        edits = [row for row in private["rows"] if row["delivery"] == "edited" and row["genre"] == genre
                 and row["family"] == family and len(row["target"]) == 1]
        for after in sorted(edits, key=lambda row: row["group"]):
            before = earlier_by_group[after["group"]]
            index, = after["target"]
            contexts = [row["text"][row["spans"][index-1][0]:row["spans"][index+1][1]] for row in (before, after)]
            try:
                files = [{suffix: rendition(text, suffix) for suffix in FORMATS} for text in contexts]
            except ValueError:
                continue
            scopes = []
            for a, b in permutations(FORMATS, 2):
                comparison, old, new, layouts = compare_with_sources(
                    files[0][a], files[1][b], b, earlier_extension=a, mixed_paired=True,
                )
                scopes.append(mixed_anchored_revisions(comparison, old, new, layouts))
            if not all(len(scope) == 1 and scope[0].geometry.get("change_ids") for scope in scopes):
                continue
            inputs = [(scope[0].earlier, scope[0].later) for scope in scopes]
            if len(set(inputs)) != 1:
                continue
            folder = output / after["group"]
            folder.mkdir(mode=0o700)
            paths = [{suffix: folder / (side+suffix) for suffix in FORMATS} for side in ("earlier", "later")]
            for path_set, bytes_set in zip(paths, files, strict=True):
                for suffix, path in path_set.items():
                    path.write_bytes(bytes_set[suffix])
            yield after, paths
            break
        else:
            raise ValueError(f"No carrier-equivalent known rewrite for {genre}/{family}")


def run(args):
    paid.configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_DEVICE=args.device,
                      VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_TEXT_WINDOW_SENTENCES="3",
                      VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="true", VERITAS_DOCUMENT_REVISION_COMPARISON_ENABLED="true",
                      VERITAS_DOCUMENT_PAIRED_INDEPENDENT_REVIEW="true", VERITAS_WEBHOOKS_ENABLED="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import pymupdf
    import torch
    from fastapi.testclient import TestClient

    from app.api import auth
    from app.billing import service
    from app.evidence.document_revision_writing import MIXED_DRAFT_SCHEMA
    from app.main import app
    from app.storage.db import get_session
    from scripts.review_document_revision_export import review
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()
    observations = []
    pairs = list(native_pairs(args.plan, args.receipt, args.output))
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            principal = auth.new_principal("local-mixed-draft-check", "examiner")
            db.add(principal)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, principal)
            db.commit()
        for source, (earlier, later) in pairs:
            source_results = []
            for current, path in later.items():
                aid = paid.checkout(client, stub, path, kind="document", media_type=FORMATS[current], filename=path.name)
                analysis = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
                before_counts = paid.counts()
                route = f"/api/v2/analyses/{aid}/document-revision-comparison"
                available = paid.body(client.get(route))
                assert available["paired_mixed_formats_available"] and available["paired_check_available"]
                for old, original in earlier.items():
                    if old == current:
                        continue
                    payload = paid.body(client.post(route, data={"include_paired_check": "true", "include_pdf": "true"},
                                                    files={"earlier": (original.name, original.read_bytes(), FORMATS[old])}))
                    report = payload["comparison"]
                    check = report["paired_check"]
                    assert check["schema"] == MIXED_DRAFT_SCHEMA and check["authorship_confidence_contribution"] == 0
                    revision, = check["revisions"]
                    assert len(revision.get("responses", ())) == 2, f"Both writing checks required: {old} to {current}."
                    assert all(revision[side]["text"] for side in ("earlier", "later"))
                    assert revision["change_ids"] and set(revision) >= {"wording_changes", "earlier", "later"}
                    destination = path.parent / f"{old[1:]}-to-{current[1:]}"
                    exported = destination.with_suffix(".json")
                    exported.write_text(json.dumps(report, sort_keys=True))
                    scan(exported.read_text())
                    verified = review(exported, original, path)
                    assert verified["paired_scope_and_saved_response_arithmetic"] == check["status"]
                    pdf = base64.b64decode(payload["pdf"])
                    destination.with_suffix(".pdf").write_bytes(pdf)
                    with pymupdf.open(stream=pdf, filetype="pdf") as rendered:
                        text = " ".join(" ".join(page.get_text().split()) for page in rendered)
                        scan(text)
                        assert "Experimental paired writing check" in text and "Linked wording changes" in text
                        assert "Check 1" in text and "Check 2" in text and any(page.get_links() for page in rendered)
                        if ".pdf" in (old, current):
                            assert "highlighted source crop" in text and verified["paired_pdf_word_regions_verified"]
                    source_results.append({"earlier_format": old, "later_format": current, "status": check["status"],
                                           "input_hashes": [revision[side]["input_sha256"] for side in ("earlier", "later")],
                                           "responses": revision["responses"], "verification": verified})
                    print(source["group"][:12], old, current, check["status"], flush=True)
                assert paid.counts() == before_counts and analysis == paid.body(client.get(f"/api/analyses/{aid}/export.json"))
            assert len(source_results) == 6 and len({tuple(row["input_hashes"]) for row in source_results}) == 1
            first = source_results[0]
            assert all(row["status"] == first["status"] and row["responses"] == first["responses"] for row in source_results)
            observations.append({"group": source["group"], "genre": source["genre"], "generator": source["family"],
                                 "comparisons": source_results})
    assert len(stub.sessions) == 6
    result = {"schema": "chipmunk.mixed-paired-writing-app/1", "billing_jobs_before_payment": 0,
              "simulated_usd_cents": 999, "checkouts": len(stub.sessions),
              "source_plan_sha256": study.pool.file_hash(args.plan), "observations": observations}
    (args.output / "receipt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("plan", "receipt", "output"):
        cli.add_argument(arg, type=Path)
    cli.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    run(cli.parse_args())
