# SPDX-License-Identifier: Apache-2.0
"""Replay pinned writing agreement contexts after native document repackaging."""

from __future__ import annotations

import argparse
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pymupdf

from app.detectors.text.adapters import agreement
from app.detectors.text.extract import extract
from app.detectors.text.segment import segment
from app.evidence import document_matched_context, document_scope
from app.evidence.document_packets import _source_anchors
from scripts import writing_agreement_document_format_fixtures as fixtures
from scripts import writing_checkpoint_agreement as prior
from scripts.image_small_boundary_screenshot_admission import save_new, sha

SCHEMA = "chipmunk.writing-agreement-document-format-transfer/1"
RECEIPT = Path(__file__).resolve().parents[2] / "docs/evaluation-panels/writing-checkpoint-agreement-2026-11-11.json"
FORMATS = ("txt", "docx", "pdf", "eml")
CONTROL_GROUPS = 8
WORD_XML = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def digest(data) -> str:
    return prior.text_sha(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def source_inputs(panels):
    public = prior.load(RECEIPT)
    reference = panels / "writing-checkpoint-agreement-2026-11-11"
    scores = reference / "scores.json"
    saved_readout = reference / "private-readout.json"
    prior.require(sha(scores) == public["scores_sha256"] and sha(saved_readout) == public["readout_sha256"]
                  and sha(reference / "freeze.json") == public["freeze_sha256"], "The pinned agreement result changed")
    rows, _ = prior.frozen_inputs(panels)
    responses = prior.load(scores)["responses"]
    marks = prior.readout(rows, responses)
    prior.require(marks == prior.load(saved_readout), "Pinned agreement locations changed")
    return rows, responses, marks


def freeze(panels, font):
    rows, _, _ = source_inputs(panels)
    controls = []
    seen = set()
    for index, row in enumerate(rows):
        if row["kind"] == "original" and row["group"] not in seen:
            seen.add(row["group"])
            controls.append(index)
        if len(controls) == CONTROL_GROUPS:
            break
    return {"schema": SCHEMA, "parent_agreement_sha256": sha(RECEIPT), "font_sha256": sha(font), "font": font.name,
            "formats": list(FORMATS), "control_indices": controls,
            "cases": [{"index": i, "panel": row["panel"], "kind": row["kind"], "group": row["group"], "truth": row["truth"],
                       "text_sha256": row["text_sha256"], "contexts": [w["sha256"] for w in row["windows"]]}
                      for i, row in enumerate(rows)],
            "code_sha256": {path.name: sha(path) for path in (Path(__file__), Path(fixtures.__file__), Path(prior.__file__),
                            Path(agreement.__file__))}}


def supported_mapping(document, fmt: str) -> bool:
    meta = document.meta
    mapping = document.source_map
    if not mapping or len(mapping) != len(document.paragraphs):
        return False
    if fmt == "docx":
        return meta.get("method") == "python-docx" and all(
            row["kind"] == "docx" and row.get("run_mapping_complete") is True and len(row["body_path"]) == 1 for row in mapping)
    if fmt == "pdf":
        return (meta.get("method") == "pymupdf" and not meta.get("unmapped_pages") and not meta.get("partially_checked_pages")
                and meta.get("ocr", {}).get("status") == "not_needed"
                and all(row["kind"] == "pdf" and row["extraction"] == "text_layer" and row.get("fragments") for row in mapping))
    if fmt == "eml":
        email = meta.get("email", {})
        return (meta.get("method") == "email" and email.get("body_content_type") == "text/plain"
                and not email.get("attachments") and all(row["kind"] == "email" and row["mime_part"] == email["body_part"]
                and row["quote_role"] == "body" for row in mapping))
    return fmt == "txt" and meta.get("method") == "plain" and all(row["kind"] == "text" for row in mapping)


def _decoded_ranges(document, row, raw, origin):
    positions = row["source_ranges"]
    cursor = row["start"]
    for start, end, a, b in positions:
        local_a, local_b = a - origin, b - origin
        token = raw[local_a:local_b]
        current = document.text[start:end]
        prior.require(start == cursor and end > start and 0 <= local_a < local_b <= len(raw)
                      and (token == current or current == " " and token.isspace()), "Mapped decoded text differs from source characters")
        cursor = end
    prior.require(cursor == row["end"], "Incomplete decoded source range")
    return len(positions)


def native_positions(document, fmt, path):
    mapping = document.source_map
    if fmt in {"txt", "eml"}:
        if fmt == "txt":
            body = path.read_bytes().decode("utf-8")
        else:
            message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
            body = message.get_body(preferencelist=("plain",)).get_content()
        return sum(_decoded_ranges(document, row, body, 0) for row in mapping)
    if fmt == "docx":
        with ZipFile(path) as archive:
            tree = ElementTree.fromstring(archive.read("word/document.xml"))
        body = list(tree.find(WORD_XML + "body"))
        count = 0
        for row in mapping:
            [position] = row["body_path"]
            native = body[position]
            prior.require(native.tag == WORD_XML + "p", "DOCX paragraph path differs")
            runs = ["".join(node.text or "" for node in run.iter(WORD_XML + "t")) for run in native.findall(WORD_XML + "r")]
            offsets, cursor = [], 0
            for index, run in enumerate(runs):
                offsets.append({"run": index, "start": cursor, "end": cursor + len(run)})
                cursor += len(run)
            prior.require(row["runs"] == offsets, "Exact DOCX run offsets differ")
            count += len(offsets) + _decoded_ranges(document, row, "".join(runs), 0)
        return count

    count = 0
    with pymupdf.open(path) as pdf:
        # Word extraction is independent of the character-glyph grouping in the app.
        indexed = [{(tuple(round(value, 3) for value in word[:4]), word[4]) for word in page.get_text("words")}
                   for page in pdf]
        for row in mapping:
            page = row["page"]
            prior.require(1 <= page <= len(pdf) and row["page_rotation"] == 0, "Invalid native fixture page")
            words = indexed[page - 1]
            for fragment in row["fragments"]:
                text = document.text[fragment["start"]:fragment["end"]]
                box = tuple(round(value, 3) for value in fragment["bbox"])
                prior.require((box, text) in words, "Mapped PDF word geometry differs from the native page")
                count += 1
    return count


def windows_from(document, pinned):
    sentences = segment(document)
    contexts = prior.native_contexts(sentences)
    windows = []
    for i, (focus, context) in enumerate(zip(sentences, contexts, strict=True)):
        offsets = [[sentence.start, sentence.end] for sentence in context]
        text = " ".join(document.text[a:b] for a, b in offsets)
        windows.append({"focus": [focus.start, focus.end], "spans": offsets, "sha256": prior.text_sha(text),
                        "score": pinned["windows"][i]["current"] if i < len(pinned["windows"]) else None})
    return windows, sentences


def route_scope(document, windows, sentences):
    decoded = {"text": document.text, "paragraphs": document.paragraphs, "sentences": sentences,
               "meta": document.meta, "source_map": document.source_map}
    primary = {"detector_id": document_matched_context.DETECTOR, "version": "1.0.0",
               "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
               "details": {"text_evidence": {"windows": windows}}}
    return agreement.candidates(decoded, primary)


def score_file(path, fmt, pinned, responses, previous):
    document = extract(path, f".{fmt}", allow_ocr=False)
    source_hash = sha(path)
    prior.require(source_hash == document.meta["original_sha256"], "The decoded document is not bound to the source bytes")
    windows, sentences = windows_from(document, pinned)
    equal_text = prior.text_sha(document.text) == pinned["text_sha256"]
    prior_windows = [{key: w[key] for key in ("spans", "sha256")} for w in pinned["windows"]]
    actual_windows = [{key: w[key] for key in ("spans", "sha256")} for w in windows]
    equal_windows = prior_windows == actual_windows
    bound = supported_mapping(document, fmt)
    positions, contexts_bound = 0, 0
    marks = None
    current, prospective = False, False
    position_error = None
    app_source = agreement._supported_source({"paragraphs": document.paragraphs, "meta": document.meta, "source_map": document.source_map})
    if bound and equal_text and equal_windows:
        try:
            positions = native_positions(document, fmt, path)
        except ValueError as error:
            position_error = str(error)
    if positions:
        ranges = []
        for window in windows:
            for a, b in window["spans"]:
                anchors = _source_anchors(document.source_map, a, b, len(document.text))
                prior.require(len(anchors) == 1 and (anchors[0]["selected_start"], anchors[0]["selected_end"]) == (a, b)
                              and (fmt != "pdf" or anchors[0]["highlight_boxes"]), "Context span has no native source location")
                contexts_bound += 1
            if window["score"] >= .5:
                a, b = window["focus"]
                ranges.append({"start": a, "end": b, "spans": [{"start": a, "end": b} for a, b in window["spans"]],
                               "input_sha256": window["sha256"], "state": "unavailable"})
        document_scope.validate_checkpoint_agreement({"source": {"chars": len(document.text)},
            "coverage": {document_scope.CHECKPOINT_AGREEMENT_KEY: {"schema": document_scope.CHECKPOINT_AGREEMENT_SCHEMA,
                "ranges": ranges[:document_scope.CHECKPOINT_AGREEMENT_MAX], "authorship_confidence_contribution": 0}}}, document.text)
        prospective = route_scope(document, windows, sentences) is not None
        current = fmt == "txt" and prospective
        marks = prior.select(pinned["windows"], [responses[row["sha256"]]["score"] for row in windows])
        prior.require(marks == previous["marks"], "A format changed the recorded detector selection")
    return {"format": fmt, "source_sha256": source_hash, "text_sha256": prior.text_sha(document.text),
            "source_map_sha256": digest(document.source_map), "ordered_contexts_sha256": digest(actual_windows),
            "native_windows": len(windows), "verified_format_positions": positions, "context_span_anchors": contexts_bound,
            "source_scope_qualified": bound, "app_source_gate": app_source, "position_error": position_error,
            "text_equal": equal_text, "context_equal": equal_windows,
            "existing_gate": current, "prospective_gate": prospective, "selections": marks}


def measure(panels, font, run, plan):
    rows, responses, marks = source_inputs(panels)
    folder = run / "inputs"
    folder.mkdir()
    cases, controls = [], []
    for index, (row, saved) in enumerate(zip(rows, marks, strict=True)):
        case_dir = folder / f"{index:03d}"
        paths = fixtures.build(case_dir, row["text"], font)
        by_format = [score_file(paths[fmt], fmt, row, responses, saved) for fmt in FORMATS]
        cases.append({"index": index, "formats": by_format, "baseline_useful": saved["useful"],
                      "baseline_selections": saved["marks"]})
        if index in plan["control_indices"]:
            quote, html, raster = fixtures.controls(case_dir, paths["pdf"], row["text"])
            for role, path in (("quoted_eml", quote), ("html_eml", html), ("raster_pdf", raster)):
                try:
                    document = extract(path, path.suffix, allow_ocr=False)
                    admitted = supported_mapping(document, path.suffix[1:])
                    app_source = agreement._supported_source({"paragraphs": document.paragraphs, "meta": document.meta,
                                                              "source_map": document.source_map})
                    record = {"extraction": "completed", "admitted": admitted,
                              "app_source_gate": app_source, "text_equal": prior.text_sha(document.text) == row["text_sha256"]}
                except ValueError as error:
                    prior.require(role == "raster_pdf" and "requires OCR" in str(error), "Unexpected control extraction failure")
                    record = {"extraction": "requires_ocr", "admitted": False, "app_source_gate": False, "text_equal": False}
                controls.append({"index": index, "role": role, "source_sha256": sha(path), **record})
    return {"freeze_sha256": sha(run / "freeze.json"), "cases": cases, "controls": controls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    for key in ("panels", "font", "run"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    plan = freeze(args.panels, args.font)
    path = args.run / "freeze.json"
    if args.stage == "freeze":
        args.run.mkdir(mode=0o700)
        save_new(path, plan)
        print(sha(path), flush=True)
        return
    prior.require(prior.load(path) == plan, "The frozen format plan changed")
    outcomes = measure(args.panels, args.font, args.run, plan)
    save_new(args.run / "measurements.json", outcomes)
    print(f"{len(outcomes['cases'])} compared texts and {len(outcomes['controls'])} negative controls", flush=True)


if __name__ == "__main__":
    main()
