# SPDX-License-Identifier: Apache-2.0
"""Replay the source mapping for short reflowed PDF revision pairs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from scripts.writing_editorial_worker_audit import plan

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "chipmunk.pdf-reflow-revision/1"
ROLE_COUNTS = {"edited": 21, "qwen": 16, "smol2": 16, "unchanged": 21, "large_gap": 21}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, result):
    with path.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


def split_position(row):
    start, stop = row["focus"]
    text = row["text"]
    midpoint = (start + stop) // 2
    eligible = [i for i in range(start + 10, stop - 10)
                if text[i].isspace() and text[i-1].isalnum() and text[i+1].islower()]
    return min(eligible, key=lambda i: (abs(i - midpoint), i), default=None)


def pdf(text, font_path, *, split=None, large_gap=False):
    import pymupdf

    font = pymupdf.Font(fontfile=str(font_path))
    size, pitch = 10, 12.2

    def lines(body):
        output, current = [], ""
        for word in body.split():
            candidate = f"{current} {word}".strip()
            if current and font.text_length(candidate, fontsize=size) > 476:
                output.append(current)
                current = word
            else:
                current = candidate
        return [*output, current]

    def insert(page, content, *, bottom=None):
        wrapped = lines(content)
        y = 68 if bottom is None else page.rect.height - bottom - (len(wrapped) - 1) * pitch
        for line in wrapped:
            page.insert_text((54, y), line, fontsize=size, fontname="reviewbody", fontfile=str(font_path))
            y += pitch

    with pymupdf.open() as doc:
        first = doc.new_page()
        if split is None:
            insert(first, text)
        else:
            insert(first, text[:split], bottom=230 if large_gap else 67)
            insert(doc.new_page(), text[split:])
        return doc.tobytes(garbage=3, deflate=True, no_new_id=True)


def submissions(inputs):
    earlier = {item["group"]: item for item in inputs if item["role"] == "before"}
    for item in inputs:
        before = earlier[item["group"]]
        if item["role"] == "before":
            yield "unchanged", before, item
        else:
            yield item["role"], before, item
        if item["role"] == "edited":
            yield "large_gap", before, item


def evidence(candidate, expected):
    focused = candidate.geometry["later"]
    return (" ".join(focused["text"].split()) == " ".join(expected.split())
            and [page["page"] for page in focused["pdf_pages"]] == [1, 2])


def summaries(rows):
    tallies = defaultdict(Counter)
    for row in rows:
        counts = tallies[row["role"]]
        counts.update(files=1, assessable_layout=row["assessable"], current_matches=row["current"],
                      extended_matches=row["extended"], exact_two_page_centers=row["correct"],
                      false_or_misplaced=row["invalid"], unchanged_wording=row["same_wording"])
    return dict(tallies)


def transport(source, font_path, output):
    from app.evidence import document_revision as draft
    from app.evidence import document_revision_writing as writing

    rows = []
    for role, before, after in submissions(plan(source)):
        split = split_position(after)
        comparison, prior, current, layouts = draft.compare_with_sources(
            pdf(before["text"], font_path), pdf(after["text"], font_path, split=split, large_gap=role == "large_gap"),
            ".pdf", pdf_paired_text_only=True, pdf_independent=True,
        )
        # The renderer must preserve the exact words and their order before its layout counts.
        exact = (layouts is not None and " ".join(prior.split()) == " ".join(before["text"].split())
                 and " ".join(current.split()) == " ".join(after["text"].split()))
        base = writing.independently_anchored_revisions(comparison, prior, current, pdf_layouts=layouts) if exact else []
        candidates, extended = writing.pdf_anchored_revisions(comparison, prior, current, layouts) if exact else ([], False)
        a, b = after["focus"]
        correct = sum(evidence(candidate, after["text"][a:b]) for candidate in candidates)
        target = role not in ("unchanged", "large_gap") and split is not None
        rows.append({"role": role, "group": after["group"], "text_sha256": [before["text_sha256"], after["text_sha256"]],
                     "split_offset": split, "assessable": exact, "same_wording": not comparison["changes"],
                     "current": len(base), "extended": len(candidates) if extended else 0,
                     "correct": correct if target else 0,
                     "invalid": (len(candidates) - correct if target else len(candidates)) if extended else 0})
        if len(rows) % 16 == 0:
            print(f"Compared {len(rows)} reflowed pairs", flush=True)
    receipt = {"schema": SCHEMA, "source_sha256": digest(source.read_bytes()), "font_sha256": digest(font_path.read_bytes()),
               "rows": rows, "summary": summaries(rows)}
    check_receipt(receipt)
    write(output, receipt)


def check_receipt(receipt):
    rows = receipt["rows"]
    if (receipt["schema"] != SCHEMA or summaries(rows) != receipt["summary"]
            or Counter(row["role"] for row in rows) != ROLE_COUNTS
            or len({(row["role"], row["group"]) for row in rows}) != len(rows)):
        raise ValueError("The frozen comparison does not reproduce")


def paid_flow(source, font_path, transport_path, output):
    from scripts.video_short_return_encoded_app import PaymentStub, body, checkout, configure

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
    from app.evidence.document_revision_writing import PDF_CONTINUATION_SCHEMA
    from app.main import app
    from app.storage.db import Analysis, get_session
    from scripts.review_document_revision_export import review
    from scripts.verify_multimedia_exports import scan

    receipt = json.loads(transport_path.read_bytes())
    check_receipt(receipt)
    assert receipt["source_sha256"] == digest(source.read_bytes()) and receipt["font_sha256"] == digest(font_path.read_bytes())
    mapped = {(r["role"], r["group"]) for r in receipt["rows"] if r["extended"] == r["correct"] == 1}
    fixtures = {}
    used = set()
    for role, before, after in submissions(plan(source)):
        if role not in fixtures and after["group"] not in used and (role, after["group"]) in mapped:
            fixtures[role] = (before, after)
            used.add(after["group"])
    assert set(fixtures) == {"edited", "qwen", "smol2"}
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = payment = PaymentStub()
    records = {}
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-pdf-reflow-analysis", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for role, (before, after) in fixtures.items():
            older, newer = (output / f"{role}-{side}.pdf" for side in ("earlier", "later"))
            older.write_bytes(pdf(before["text"], font_path))
            newer.write_bytes(pdf(after["text"], font_path, split=split_position(after)))
            aid = checkout(client, payment, newer, kind="document", media_type="application/pdf", filename=newer.name)
            with get_session() as db:
                saved = db.get(Analysis, aid).result_hash
            endpoint = f"/api/v2/analyses/{aid}/document-revision-comparison"
            files = {"earlier": (older.name, older.read_bytes(), "application/pdf")}
            standard = body(client.post(endpoint, files=files))["comparison"]
            payload = body(client.post(endpoint, files=files, data={"include_paired_check": "true", "include_pdf": "true"}))
            comparison = payload["comparison"]
            check = comparison["paired_check"]
            assert check["schema"] == PDF_CONTINUATION_SCHEMA and check["authorship_confidence_contribution"] == 0
            assert comparison["changes"] == standard["changes"]
            revision, = check["revisions"]
            assert len(revision["responses"]) == 2
            a, b = after["focus"]
            assert " ".join(revision["later"]["text"].split()) == " ".join(after["text"][a:b].split())
            assert [row["page"] for row in revision["later"]["pdf_pages"]] == [1, 2]
            for side, item in (("earlier", before), ("later", after)):
                text = " ".join(item["text"][location["start"]:location["end"]] for location in item["ranges"][1:4])
                assert revision[side]["input_sha256"] == digest(" ".join(text.split()).encode())
            exported = output / f"{role}.json"
            write(exported, comparison)
            replay = review(exported, older, newer)
            report = base64.b64decode(payload["pdf"], validate=True)
            (output / f"{role}.pdf").write_bytes(report)
            with pymupdf.open(stream=report, filetype="pdf") as doc:
                report_text = " ".join(" ".join(page.get_text().split()) for page in doc)
                scan(report_text)
                assert all(phrase in report_text for phrase in ("Changed sentence 1", "Page 1", "Page 2", "Linked wording changes"))
            scan(json.dumps(check))
            with get_session() as db:
                assert db.get(Analysis, aid).result_hash == saved
            records[role] = {"group": after["group"], "source_text_sha256": [before["text_sha256"], after["text_sha256"]],
                             "source_pdf_sha256": [digest(older.read_bytes()), digest(newer.read_bytes())],
                             "status": check["status"], "replay": replay}
    write(output / "receipt.json", {"cases": records, "simulated_paid_runs": len(records), "jobs_before_payment": 0,
                                    "transport_sha256": digest(transport_path.read_bytes())})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    modes = cli.add_subparsers(dest="mode", required=True)
    for action, fields in (("transport", ("source", "font", "output")),
                           ("app", ("source", "font", "transport", "output")),
                           ("review", ("receipt",))):
        command = modes.add_parser(action)
        for field in fields:
            command.add_argument(field, type=Path)
    args = cli.parse_args()
    if args.mode == "transport":
        transport(args.source, args.font, args.output)
    elif args.mode == "app":
        paid_flow(args.source, args.font, args.transport, args.output)
    else:
        receipt = json.loads(args.receipt.read_bytes())
        check_receipt(receipt)
        print(receipt["summary"])


if __name__ == "__main__":
    main()
