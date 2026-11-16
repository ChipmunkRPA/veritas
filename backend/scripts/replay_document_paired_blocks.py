# SPDX-License-Identifier: MIT
"""Replay licensed earlier drafts with joined, split and rewritten sentences."""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from scripts import video_short_return_encoded_app as paid
from scripts import writing_function_sequence_transfer as grammar

ROOT = Path(__file__).resolve().parents[2]
FORMATS = {".txt": "text/plain", ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
           ".pdf": "application/pdf"}
SCHEMA = "chipmunk.paired-document-block-replay/1"


def prior_rewrites(path, receipt):
    local, measured = grammar.read(path), grammar.read(receipt)
    assert measured["fresh_intake"]["plan_sha256"] == grammar.parent.sha(path.read_bytes())
    pairs = defaultdict(dict)
    for row in local["rows"]:
        changed = {row["words"][index][2] for index in row["changed"]}
        if row["delivery"] == "edited" and changed != {2, 3}:
            continue
        pairs[row["group"]][row["delivery"]] = row
    counts = Counter()
    for group, pair in sorted(pairs.items()):
        if set(pair) != {"original", "edited"}:
            continue
        before, after = (pair[key] for key in ("original", "edited"))
        category = after["genre"], after["family"]
        if counts[category] >= 2:
            continue
        texts = [[unit["text"] for unit in row["sentences"]] for row in (before, after)]
        if not all(len(value) == 4 for value in texts) or any(texts[0][i] != texts[1][i] for i in (0, -1)):
            continue
        counts[category] += 1
        yield {"source": group, "role": "known_rewrite", "genre": after["genre"], "family": after["family"],
               "words": [sum(len(unit.split()) for unit in side) for side in texts],
               "contexts": [" ".join(side) for side in texts], "counts": [2, 2],
               "parent_files": [before["sha256"], after["sha256"]]}


def prose_windows(body):
    from app.detectors.text.extract import from_string
    from app.detectors.text.segment import segment
    from scripts.writing_reference_revision_sources import blocks

    by_anchors = defaultdict(list)
    for block in blocks(body, expanded=True):
        for paragraph in block.split("\n\n"):
            units = segment(from_string(paragraph))
            for width in (3, 4):
                for i in range(len(units) - width + 1):
                    window = units[i:i+width]
                    if any(unit.n_words < 6 for unit in window) or sum(unit.n_words for unit in window) > 100:
                        continue
                    text = " ".join(unit.text for unit in window)
                    by_anchors[(window[0].text, window[-1].text)].append((text, width-2))
    return by_anchors


def genuine_revisions(folder):
    from scripts import writing_elife_version_body_holdout as elife

    pool = elife.roster(folder)["pool"]
    publisher = folder / "publisher"
    for article, versions in pool.items():
        try:
            dates, _ = elife.dated_api(publisher / f"{article}-versions.json", article)
            old, new = [elife.body(publisher / versions[key]["path"], versions[key], article) for key in ("2", "3")]
        except ValueError:
            continue
        if old[1] != new[1]:
            continue
        left, right = (prose_windows(root) for root, _ in (old, new))
        for key in sorted(left.keys() & right.keys(), key=lambda value: grammar.parent.sha("/".join(value))):
            if len(left[key]) != 1 or len(right[key]) != 1:
                continue
            before, after = left[key][0], right[key][0]
            if before[1] == after[1] or before[0] == after[0]:
                continue
            urls = [f"{elife.RAW}/{versions[key]['path']}" for key in ("2", "3")]
            yield {"source": grammar.parent.sha(f"elife:{article}"), "role": "genuine_editorial_revision",
                   "doi": f"10.7554/eLife.{article}", "version_dates": [date.date().isoformat() for date in dates],
                   "source_urls": urls, "license": elife.LICENSE, "contexts": [before[0], after[0]],
                   "counts": [before[1], after[1]],
                   "parent_files": [grammar.parent.sha((publisher / versions[key]["path"]).read_bytes()) for key in ("2", "3")]}
            break


def condition_controls():
    before = ("The inspectors collected the original records before they met with the district staff. "
              "The warehouse stored thirty boxes, and the delivery wagon departed in the afternoon. "
              "All later adjustments must be entered in the original register by the supervisors.")
    punctuation = before.replace("boxes, and the", "boxes. And the")
    replacement = before.replace("stored thirty boxes, and the delivery wagon departed in the afternoon.",
                                 "stored thirty-two boxes when the inspection began. The trucks left after the evening crew arrived.")
    yield {"source": grammar.parent.sha("punctuation-only"), "role": "punctuation_only", "contexts": [before, punctuation], "counts": [1, 2]}
    yield {"source": grammar.parent.sha("split"), "role": "authored_split", "contexts": [before, replacement], "counts": [1, 2]}
    yield {"source": grammar.parent.sha("merge"), "role": "authored_merge", "contexts": [replacement, before], "counts": [2, 1]}
    repeat = before.split(". ", 1)[0] + ". " + before
    yield {"source": grammar.parent.sha("duplicate"), "role": "duplicate_anchor", "contexts": [repeat, repeat.replace(
        "stored thirty boxes, and the delivery wagon departed in the afternoon.",
        "stored thirty-two boxes when the inspection began. The trucks left after the evening crew arrived.")], "counts": [1, 2]}


def make_files(item):
    from scripts.replay_document_mixed_formats import rendition

    return [{fmt: rendition(text, fmt) for fmt in FORMATS} for text in item["contexts"]]


def mapping(files, a, b):
    from app.evidence.document_revision import compare_with_sources
    from app.evidence.document_revision_writing import (
        block_anchored_revisions,
        independently_anchored_revisions,
        mixed_anchored_revisions,
        paired_input,
        pdf_anchored_revisions,
    )

    comparison, earlier, later, layouts = compare_with_sources(
        files[0][a], files[1][b], b, earlier_extension=a, mixed_paired=a != b,
        pdf_paired_text_only=True, pdf_independent=a == b == ".pdf",
    )
    block = block_anchored_revisions(comparison, earlier, later, layouts)
    if a != b:
        prior = mixed_anchored_revisions(comparison, earlier, later, layouts)
    elif (candidate := paired_input(comparison, earlier, later, word_layouts=layouts if b == ".docx" else None)) is not None:
        prior = [candidate]
    elif b == ".pdf":
        prior, _ = pdf_anchored_revisions(comparison, earlier, later, layouts)
    else:
        prior = independently_anchored_revisions(comparison, earlier, later, word_layouts=layouts if b == ".docx" else None)
    return comparison, block, prior


def proof(item, files):
    rows = {}
    for a, b in product(FORMATS, repeat=2):
        _, current, previous = mapping(files, a, b)
        current = [candidate for candidate in current if [len(candidate.geometry[side]["context_spans"])-2
                   for side in ("earlier", "later")] == item["counts"]]
        validate = next(iter(current), None)
        rows[f"{a}/{b}"] = {"before": bool(previous), "mapped": validate is not None,
                           "input_hashes": [validate.geometry[side]["input_sha256"] for side in ("earlier", "later")] if validate else [],
                           "files": [grammar.parent.sha(files[0][a]), grammar.parent.sha(files[1][b])]}
    return {key: value for key, value in item.items() if key != "contexts"} | {"deliveries": rows}


def freeze(args):
    research = list(prior_rewrites(args.plan, args.receipt))
    genuine = list(genuine_revisions(args.elife)) if args.elife else []
    cases, evidence, exclusions = [], [], []
    for item in (*research, *genuine, *condition_controls()):
        try:
            files = make_files(item)
            row = proof(item, files)
        except ValueError:
            if item["role"] not in {"known_rewrite", "genuine_editorial_revision"}:
                raise
            exclusions.append({"source": item["source"], "role": item["role"], "reason": "delivery_outside_scope"})
            continue
        evidence.append(row)
        print(item["role"], item["counts"], sum(entry["mapped"] for entry in row["deliveries"].values()), flush=True)
        cases.append(item)
    values = {"schema": SCHEMA, "plan_sha256": grammar.parent.sha(args.plan.read_bytes()),
              "writing_receipt_sha256": grammar.parent.sha(args.receipt.read_bytes()), "sources": evidence, "excluded": exclusions}
    values["summary"] = recount(values)
    (args.output / "mapping.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    grammar.save(args.output / "cases.json.gz", {"schema": SCHEMA, "rows": cases, "proof_sha256": grammar.parent.sha(
        (args.output / "mapping.json").read_bytes())})


def recount(data):
    assert data["schema"] == SCHEMA
    outcomes = defaultdict(Counter)
    sources = set()
    formats = {f"{a}/{b}" for a, b in product(FORMATS, repeat=2)}
    for row in data["sources"]:
        assert row["source"] not in sources and set(row["deliveries"]) == formats
        sources.add(row["source"])
        bucket = outcomes[row["role"]]
        bucket["works"] += 1
        selected = []
        for delivery in row["deliveries"].values():
            assert len(delivery["input_hashes"]) == (2 if delivery["mapped"] else 0)
            bucket.update(comparisons=1, before=delivery["before"], mapped=delivery["mapped"])
            if delivery["input_hashes"]:
                selected.append(tuple(delivery["input_hashes"]))
        assert len(set(selected)) <= 1
        if row["role"] in ("duplicate_anchor", "punctuation_only"):
            assert not selected
    return {role: dict(counts) for role, counts in outcomes.items()}


def app_examples(folder):
    mapping_receipt = folder / "mapping.json"
    numeric, raw = json.loads(mapping_receipt.read_bytes()), grammar.read(folder / "cases.json.gz")
    assert raw["proof_sha256"] == grammar.parent.sha(mapping_receipt.read_bytes())
    assert numeric["summary"] == recount(numeric)
    sources = {row["source"]: row for row in raw["rows"]}
    for fmt, counts in ((".docx", [2, 2]), (".pdf", [1, 2]), (".txt", [2, 1])):
        options = [row for row in numeric["sources"] if row["counts"] == counts
                   and all(item["mapped"] for item in row["deliveries"].values())]
        assert options, counts
        case = min(options, key=lambda row: (row["role"].startswith("authored"), row["source"]))
        yield fmt, case, make_files(sources[case["source"]])


def run_app(args):
    paid.configure(args.output, ROOT)
    os.environ.update(VERITAS_ENVIRONMENT="dev", VERITAS_LOG_LEVEL="WARNING", VERITAS_DEVICE="cpu",
                      VERITAS_TEXT_DETECTOR_MODE="candidate", VERITAS_TEXT_WINDOW_SENTENCES="3",
                      VERITAS_TEXT_CHECKPOINT_AGREEMENT_REVIEW="true", VERITAS_DOCUMENT_REVISION_COMPARISON_ENABLED="true",
                      VERITAS_DOCUMENT_PAIRED_INDEPENDENT_REVIEW="true", VERITAS_DOCUMENT_PAIRED_BLOCK_REVIEW="true",
                      VERITAS_WEBHOOKS_ENABLED="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    import pymupdf
    import torch
    from app.api import auth
    from app.billing import service
    from app.core.config import settings
    from app.evidence.document_revision_writing import BLOCK_DRAFT_SCHEMA
    from app.main import app
    from app.storage.db import get_session
    from fastapi.testclient import TestClient
    from scripts.review_document_revision_export import review
    from scripts.verify_multimedia_exports import scan

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    service._stripe = stub = paid.PaymentStub()
    verified = []
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        with get_session() as db:
            owner = auth.new_principal("local-document-passage-check", "examiner")
            db.add(owner)
            db.flush()
            client.headers["X-API-Key"], _ = auth.issue_api_key(db, owner)
            db.commit()
        for fmt, case, files in app_examples(args.fixtures):
            folder = args.output / fmt[1:]
            folder.mkdir(mode=0o700)
            latest = folder / ("analyzed" + fmt)
            latest.write_bytes(files[1][fmt])
            aid = paid.checkout(client, stub, latest, kind="document", media_type=FORMATS[fmt], filename=latest.name)
            existing = paid.body(client.get(f"/api/analyses/{aid}/export.json"))
            jobs = paid.counts()
            route = f"/api/v2/analyses/{aid}/document-revision-comparison"
            assert paid.body(client.get(route))["paired_mixed_formats_available"]
            for old_col, old_body in files[0].items():
                before = folder / ("earlier" + old_col)
                before.write_bytes(old_body)
                upload = {"earlier": (before.name, old_body, FORMATS[old_col])}
                if fmt == ".docx" and old_col == ".txt":
                    settings.document_paired_block_review = False
                    prior = paid.body(client.post(route, data={"include_paired_check": "true"}, files=upload))
                    assert prior["comparison"]["paired_check"]["status"] == "outside_scope"
                    settings.document_paired_block_review = True
                response = paid.body(client.post(route, data={"include_paired_check": "true", "include_pdf": "true"}, files=upload))
                report = response["comparison"]
                check = report["paired_check"]
                assert check["schema"] == BLOCK_DRAFT_SCHEMA and check["authorship_confidence_contribution"] == 0
                revision, = check["revisions"]
                assert len(revision.get("responses", [])) == 2, f"Missing local writing scores for {old_col}/{fmt}"
                expected = case["deliveries"][f"{old_col}/{fmt}"]
                assert expected["input_hashes"] == [revision[side]["input_sha256"] for side in ("earlier", "later")]
                assert [len(revision[side]["context_spans"])-2 for side in ("earlier", "later")] == case["counts"]
                assert set(revision["change_ids"]) <= {item["id"] for item in report["changes"]}
                stem = folder / f"{old_col[1:]}-to-{fmt[1:]}"
                exported = stem.with_suffix(".json")
                exported.write_text(json.dumps(report, sort_keys=True))
                scan(exported.read_text())
                audit = review(exported, before, latest)
                assert audit["paired_scope_and_saved_response_arithmetic"] == check["status"]
                if not verified:
                    tampered = folder / "altered-scope.json"
                    revision["earlier"]["start"] += 1
                    tampered.write_text(json.dumps(report, sort_keys=True))
                    try:
                        review(tampered, before, latest)
                    except ValueError as error:
                        assert "Mapped revisions differ" in str(error)
                    else:
                        raise AssertionError("A changed source position passed JSON verification")
                    revision["earlier"]["start"] -= 1
                pdf = base64.b64decode(response["pdf"], validate=True)
                stem.with_suffix(".pdf").write_bytes(pdf)
                with pymupdf.open(stream=pdf, filetype="pdf") as rendered:
                    text = " ".join(" ".join(page.get_text().split()) for page in rendered)
                    scan(text)
                    for label in ("Changed passage 1", "Linked wording changes", "Check 1", "Check 2"):
                        assert label in text, label
                    for side, label in (("earlier", "Earlier"), ("later", "Analyzed")):
                        unit = "sentences" if len(revision[side]["context_spans"]) == 4 else "sentence"
                        assert f"{label} {unit} [{revision[side]['start']}, {revision[side]['end']})" in text
                    if ".pdf" in (fmt, old_col):
                        assert "highlighted source crop" in text and audit["paired_pdf_word_regions_verified"]
                verified.append({"earlier_format": old_col, "analyzed_format": fmt, "source": case["source"],
                                 "role": case["role"], "sentences": case["counts"], "status": check["status"],
                                 "responses": revision["responses"], "verification": audit})
                print(case["role"], old_col, fmt, case["counts"], check["status"], flush=True)
            assert jobs == paid.counts() and existing == paid.body(client.get(f"/api/analyses/{aid}/export.json"))
    (args.output / "receipt.json").write_text(json.dumps({
        "schema": SCHEMA, "paid_analyses": len(stub.sessions), "unpaid_jobs": 0, "price_usd": 9.99,
        "initial_analysis_preserved": True, "altered_scope_rejected": True, "results": verified,
    }, indent=2, sort_keys=True) + "\n")


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scope = commands.add_parser("freeze")
    for name in ("plan", "receipt", "output"):
        scope.add_argument(name, type=Path)
    scope.add_argument("--elife", type=Path)
    verify = commands.add_parser("review")
    verify.add_argument("mapping", type=Path)
    app = commands.add_parser("app")
    app.add_argument("fixtures", type=Path)
    app.add_argument("output", type=Path)
    return parser


def main():
    args = parser().parse_args()
    if args.command == "freeze":
        args.output.mkdir(mode=0o700)
        freeze(args)
    elif args.command == "app":
        run_app(args)
    else:
        data = json.loads(args.mapping.read_bytes())
        result = recount(data)
        assert result == data["summary"]
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
