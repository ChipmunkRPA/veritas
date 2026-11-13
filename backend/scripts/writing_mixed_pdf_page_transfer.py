# SPDX-License-Identifier: Apache-2.0
"""Check whether native PDF writing contexts survive an unassessed intervening page."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.writing-mixed-pdf-page-transfer/1"
HUMAN = {"original", "human_splice"}
PARENT = "writing-agreement-document-format-2026-11-12"
SOURCE = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
BASELINE = SOURCE / "writing-agreement-document-format-transfer-2026-11-12.json"
METHOD = {
    "cases": "Insert one blank or image-only page after the first source page of all 100 native PDF fixtures. Alternate page types by stable case index.",
    "window": "Reuse prior positive first responses only after checking the whole extracted text, every ordered input hash and complete native boxes. Keep only first six positive inputs wholly within a single selectable-text page.",
    "second": "Rerun the pinned second check on each exact ordered PDF request for both mixed and intact source scope. Collapse identical complete requests only.",
    "controls": "Exclude the inserted page and independently reject every mixed-page input. App regression checks malformed or missing native word boxes.",
    "gate": "Keep at least four useful generated locations across two groups, with no new false group and complete scope and original coordinates for all retained contexts.",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_key(hashes):
    return hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()


def frozen_hash(document):
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def save(path, document):
    with path.open("x") as output:
        json.dump(document, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


def context_page(window, blocks):
    pages = []
    for span in [window["focus"], *window["spans"]]:
        containing = [block["page"] for block in blocks if block["start"] <= span["start"] < span["end"] <= block["end"]]
        if len(containing) != 1:
            return None
        pages.extend(containing)
    return pages[0] if len(set(pages)) == 1 else None


def fixture(source, path, visual):
    import pymupdf

    with pymupdf.open(source) as pdf:
        page = pdf.new_page(pno=1, width=612, height=792)
        if visual:
            width, height = 32, 32
            # Repeated wide bands have no letters. The image is outside the text layer.
            rgb = bytes(v for y in range(height) for _ in range(width) for v in ([255]*3 if y < 16 else [44, 75, 91]))
            page.insert_image(page.rect, pixmap=pymupdf.Pixmap(pymupdf.csRGB, width, height, rgb, False))
        pdf.save(path, garbage=4, deflate=True, no_new_id=True)


def freeze(run, panels):
    from scripts.writing_agreement_document_format_transfer import source_inputs

    rows, _, _ = source_inputs(panels)
    source_dir = panels / PARENT / "inputs"
    prior = read(BASELINE)["measurements"]
    require(len(prior["cases"]) == len(rows) == 100, "Native PDF cohort changed")
    plan = []
    run.mkdir(mode=0o700)
    for index, (row, reference) in enumerate(zip(rows, prior["cases"], strict=True)):
        path = source_dir / f"{index:03d}" / "excerpt.pdf"
        witness = next(item for item in reference["formats"] if item["format"] == "pdf")
        require(witness["source_sha256"] == digest(path), "Frozen native PDF changed")
        output = run / f"{index:03d}.pdf"
        visual = bool(index % 2)
        fixture(path, output, visual)
        plan.append({"index": index, "sha256": digest(output), "native_sha256": witness["source_sha256"],
                     "text_sha256": row["text_sha256"], "kind": row["kind"], "group": row["group"],
                     "inserted": "image_only" if visual else "blank"})
    save(run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "native_receipt_sha256": digest(BASELINE), "plan": plan})


def primary_result(windows):
    from app.evidence import document_matched_context

    return {"detector_id": document_matched_context.DETECTOR, "version": "1.0.0",
            "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
            "details": {"text_evidence": {"windows": windows}}}


def options(doc, reference):
    from dataclasses import asdict

    from scripts.writing_agreement_document_format_transfer import windows_from

    windows, sentences = windows_from(doc, reference)
    for index, (actual, saved) in enumerate(zip(windows, reference["windows"], strict=True)):
        require(actual["sha256"] == saved["sha256"] and actual["spans"] == saved["spans"],
                "Candidate context differs from native format")
        actual.update(focus=dict(zip(("start", "end"), actual["focus"], strict=True)),
                      spans=[dict(zip(("start", "end"), span, strict=True)) for span in actual["spans"]],
                      input_sha256=actual["sha256"], unit_ref=index)
    decoded = {"text": doc.text, "meta": doc.meta, "paragraphs": doc.paragraphs,
               "sentences": [asdict(sentence) for sentence in sentences], "source_map": doc.source_map}
    return decoded, windows


def coverage(text, truth, spans):
    target = {i for i in range(*truth) if not text[i].isspace()}
    selected = {i for span in spans for i in range(span["start"], span["end"]) if not text[i].isspace()}
    return len(selected & target), len(selected | target), len(target)


def measure(run, panels, checkpoint):
    import os

    from app.detectors.text.adapters import agreement
    from app.detectors.text.extract import extract
    from app.evidence import document_scope
    from scripts.writing_agreement_document_format_transfer import native_positions, source_inputs
    from scripts.writing_paired_email_revision import SECONDARY
    from scripts.writing_reply_native_batches import measure as score_batches

    require(checkpoint is not None and {name: digest(checkpoint / name) for name in SECONDARY} == SECONDARY,
            "The second checkpoint does not match the licensed files")
    require(os.getenv("HF_HUB_OFFLINE") == os.getenv("TRANSFORMERS_OFFLINE") == "1", "Only local inference is supported")
    rows, _, _ = source_inputs(panels)
    plan = read(run / "freeze.json")
    require(plan["schema"] == SCHEMA and plan["method"] == METHOD and plan["native_receipt_sha256"] == digest(BASELINE),
            "Comparison freeze changed")
    records, batches = [], {}
    for row, case in zip(rows, plan["plan"], strict=True):
        index = case["index"]
        path = run / f"{index:03d}.pdf"
        require(digest(path) == case["sha256"], "Mixed PDF differs from its freeze")
        doc = extract(path, ".pdf")
        require(doc.text == row["text"] and doc.meta["unmapped_pages"] == [2]
                and len(doc.meta["ocr"]["pages"]) == 1 and doc.meta["ocr"]["pages"][0]["status"] == "unchecked",
                "The inserted page altered the native transcript")
        decoded, windows = options(doc, row)
        require(not agreement._supported_source(decoded), "Mixed source unexpectedly qualified through the old path")
        valid = document_scope.pdf_agreement_text_spans(doc.source_map, doc.meta, doc.text)
        require(valid is not None and len(valid) == len(doc.paragraphs), "Intact native page scope failed")
        chosen = agreement.candidates(decoded, primary_result(windows))
        require(chosen is not None, "Intact native text review did not qualify")
        native_positions_count = native_positions(doc, "pdf", path)
        values = []
        for window in windows:
            page = context_page(window, valid)
            require(page == document_scope.pdf_agreement_window_page(window, valid), "Independent page check differs")
            values.append({"focus": window["focus"], "spans": window["spans"], "sha256": window["input_sha256"],
                           "page": page, "first_hex": window["score"].hex(),
                           "overlap": coverage(doc.text, row["truth"], window["spans"])})
        positives = [i for i, value in enumerate(values) if float.fromhex(value["first_hex"]) >= .5]
        expected = [i for i in positives if values[i]["page"] is not None][:6]
        require([window["unit_ref"] for window in chosen] == expected, "Ordered PDF context rule differs")
        for selection in (expected, positives[:6]):
            hashes = [values[i]["sha256"] for i in selection]
            texts = [" ".join(doc.text[span["start"]:span["end"]] for span in values[i]["spans"]) for i in selection]
            request = request_key(hashes)
            require(request not in batches or batches[request] == texts, "Exact requests differ for one input digest")
            batches[request] = texts
        records.append({"index": index, "blocks": valid, "positions": native_positions_count, "windows": values, "selected": expected})
    score_batches(run, checkpoint, batches)
    files = [{"filename": name, "sha256": checksum} for name, checksum in sorted(SECONDARY.items())]
    payload = {"freeze": plan, "outcomes": records, "measured": read(run / "responses.json"), "checkpoint_files": files}
    payload["summary"] = review(payload)
    save(run / "receipt.json", payload)


def review(payload):
    plan = payload["freeze"]
    require(plan["schema"] == SCHEMA and plan["method"] == METHOD and plan["native_receipt_sha256"] == digest(BASELINE)
            and len(plan["plan"]) == len(payload["outcomes"]) == 100, "Published PDF comparison differs")
    files = payload["checkpoint_files"]
    require([item["filename"] for item in files] == ["LICENSE", "config.json", "detector-config.json", "model.safetensors", "tokenizer.json"]
            and all(re.fullmatch("[a-f0-9]{64}", item["sha256"]) for item in files), "Incomplete checkpoint file commitments")
    recorded = payload["measured"]
    measured = recorded["readings"]
    require(recorded["freeze_sha256"] == frozen_hash(plan)
            and recorded["actual_forwards"] == sum((len(batch) + 3) // 4 for batch in measured.values()),
            "The measured batches do not match the frozen PDF panel")
    observed_batches = set()

    def corroborated(selection, values):
        key = request_key([values[i]["sha256"] for i in selection])
        observed_batches.add(key)
        hits = []
        for i, reading in zip(selection, measured[key], strict=True):
            score = float.fromhex(reading["score_hex"])
            require(0 <= score <= 1 and 0 < reading["tokens"] <= 512, "Incomplete or invalid second response")
            if score >= .5:
                hits.append(values[i])
        return hits

    totals, groups, gain_groups = Counter(), set(), set()
    for index, (case, record) in enumerate(zip(plan["plan"], payload["outcomes"], strict=True)):
        require(index == case["index"] == record["index"] and record["positions"] > 0, "Incomplete PDF case order or positions")
        blocks = record["blocks"]
        require(len(blocks) in {1, 4} and all(block["page"] in {1, 3} for block in blocks), "PDF page scope differs")
        expected, native_marks = [], []
        values = record["windows"]
        for i, row in enumerate(values):
            page = context_page(row, blocks)
            require(page == row["page"], "Scored context page differs")
            first = float.fromhex(row["first_hex"])
            require(0 <= first <= 1, "Invalid saved score")
            shared, union, truth = row["overlap"]
            require(0 <= shared <= truth <= union and union > 0, "Invalid known-edit overlap")
            if first >= .5:
                native_marks.append(i)
                if page is not None:
                    expected.append(i)
        expected, native_marks = expected[:6], native_marks[:6]
        require(expected == record["selected"], "Selected PDF contexts differ")
        hits = corroborated(expected, values)
        native_hits = corroborated(native_marks, values)
        human = case["kind"] in HUMAN
        useful = any(10 * row["overlap"][0] >= 3 * row["overlap"][1] and 2 * row["overlap"][0] >= row["overlap"][2] for row in hits)
        native_useful = any(10 * row["overlap"][0] >= 3 * row["overlap"][1] and 2 * row["overlap"][0] >= row["overlap"][2] for row in native_hits)
        if human and hits and not native_hits:
            groups.add(case["group"])
        if not human and useful:
            gain_groups.add(case["group"])
        totals.update(files=1, known_edits=not human, genuine=human, candidate_contexts=len(expected),
                      useful=useful and not human, native_useful=native_useful and not human,
                      genuine_flagged=human and bool(hits), native_genuine_flagged=human and bool(native_hits),
                      checked_words=record["positions"], cross_page_contexts=sum(row["page"] is None for row in values),
                      **{case["inserted"]: 1})
    require(observed_batches == set(measured), "The recorded model batches differ from the PDF requests")
    return dict(totals) | {"useful_groups": len(gain_groups), "new_genuine_groups": len(groups), "distinct_requests": len(measured),
                           "passed": totals["useful"] >= 4 and len(gain_groups) >= 2 and not groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("target", type=Path)
    parser.add_argument("--panels", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.phase == "review":
        payload = read(args.target)
        summary = review(payload)
        require(summary == payload["summary"], "Published summary differs")
        print(summary)
    else:
        if args.panels is None:
            parser.error("--panels is required for freeze and measure")
        if args.phase == "freeze":
            freeze(args.target, args.panels)
        else:
            measure(args.target, args.panels, args.checkpoint)


if __name__ == "__main__":
    main()
