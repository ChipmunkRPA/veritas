# SPDX-License-Identifier: Apache-2.0
"""Verify exact writing context after repackaging replies with more paragraph counts."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.text.adapters import agreement
from app.detectors.text.extract import _ocr_language, extract, from_string
from app.detectors.text.segment import segment
from app.evidence import document_matched_context, document_scope
from scripts import writing_email_body_scope_transfer as base
from scripts.writing_agreement_document_format_transfer import RECEIPT, native_positions, source_inputs
from scripts.writing_checkpoint_agreement import HUMAN, native_contexts, text_sha, useful

LAYOUTS = (2, 3, 5, 6)
SCHEMA = "chipmunk.writing-reply-paragraph-transfer/1"


def freeze(panels, rows):
    return {
        "schema": SCHEMA, "layouts": list(LAYOUTS), "source_results": base.sha(RECEIPT),
        "source_scores": base.sha(panels / base.PANEL / "scores.json"),
        "script": base.sha(Path(__file__)), "extraction": base.sha(Path(extract.__code__.co_filename)),
        "incumbent": base.sha(Path(agreement.__file__)), "scope": base.sha(Path(document_scope.__file__)),
        "source_cases": [row["text_sha256"] for row in rows],
        "gate": {"min_context_parity": .96, "min_generated_locations_retained": .90, "max_added_human_flags": 0,
                 "controls": "Reject every alternate HTML-only and interleaved quoted source fixture"},
        "segmentation": "Keep all planned deliveries. If reformatting changes the ordered exact source sentences, "
                        "record the unmeasured inputs as unavailable and never reuse their responses.",
    }


def repack(sentences, count):
    cuts = [i * len(sentences) // count for i in range(count + 1)]
    return "\n\n".join(" ".join(s.text for s in sentences[a:b]) for a, b in zip(cuts, cuts[1:]))


def mapped_body(doc, count):
    meta = doc.meta.get("email") or {}
    if doc.meta.get("method") != "email" or meta.get("body_content_type") != "text/plain" or not meta.get("body_part"):
        return None
    previous, body, trailing = 0, [], False
    for span in doc.source_map:
        a, b = span["start"], span["end"]
        if (span.get("mime_part") != meta["body_part"] or span["kind"] != "email" or not span["source_ranges"]
                or a != (previous + 2 if previous else 0) or not a < b <= len(doc.text)):
            return None
        if span["quote_role"] == "body" and not trailing:
            body.append({"start": a, "end": b})
        elif span["quote_role"] in {"signature", "quoted"}:
            trailing = True
        else:
            return None
        previous = b
    if trailing and len(body) == count and previous == len(doc.text):
        return body
    return None


def outcome(path, row, original_sentences, responses, count):
    doc = extract(path, ".eml")
    body = mapped_body(doc, count)
    generated = row["kind"] not in HUMAN
    possible = range(len(original_sentences) - 1)
    positive = [i for i in possible if row["windows"][i]["current"] >= .5][:document_scope.CHECKPOINT_AGREEMENT_MAX]
    expected = [i for i in positive if responses[row["windows"][i]["sha256"]]["score"] >= .5]
    reference_located = any(useful(row["text"], row["truth"], row["windows"][i]["spans"]) for i in expected)
    summary = {"file_sha256": base.sha(path), "eligible": len(possible), "generated": generated,
               "baseline_useful": generated and reference_located, "baseline_human_flag": not generated and bool(expected)}
    if body is None:
        return {**summary, "admitted": False, "same_inputs": 0, "useful": False,
                "human_flag": False, "unavailable": "native_body_scope"}
    decoded_sentences = segment(doc)
    body_sentences = [s for s in decoded_sentences if s.end <= body[-1]["end"]]
    language = _ocr_language(doc.text[:body[-1]["end"]])
    base.require(language.get("code") == "en" and language.get("status") == "supported"
                 and 68 <= len(doc.text[:body[-1]["end"]].split(None, 420)) <= 420, "Source left the pinned text scope")
    original_text = [s.text for s in original_sentences]
    current_text = [s.text for s in body_sentences]
    if current_text != original_text:
        return {**summary, "admitted": True, "same_inputs": 0, "useful": False,
                "human_flag": False, "unavailable": "sentences_changed"}
    exact, matched, windows = 0, [], []
    for i, (sentence, nearby) in enumerate(zip(decoded_sentences, native_contexts(decoded_sentences), strict=True)):
        if sentence.end > body[-1]["end"] or any(s.end > body[-1]["end"] for s in nearby):
            continue
        key = text_sha(" ".join(s.text for s in nearby))
        base.require(i in possible and key == row["windows"][i]["sha256"], "Reformatted context differs from the pinned input")
        exact += 1
        source = {"focus": {"start": sentence.start, "end": sentence.end},
                  "spans": [{"start": s.start, "end": s.end} for s in nearby], "input_sha256": key,
                  "score": row["windows"][i]["current"]}
        base.require(document_scope.checkpoint_window_in_scope(source, body), "Quote or signature entered the checked context")
        windows.append(source)
        if i in expected:
            matched.append(i)
    primary = {"detector_id": document_matched_context.DETECTOR, "version": "1.0.0",
               "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
               "details": {"text_evidence": {"windows": windows}}}
    decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "meta": doc.meta, "source_map": doc.source_map,
               "sentences": [{"start": s.start, "end": s.end} for s in decoded_sentences]}
    baseline = agreement.candidates(decoded, primary)
    base.require(baseline is None or count in {1, 4}, "Current reply gate changed unexpectedly")
    located = any(useful(row["text"], row["truth"], row["windows"][i]["spans"]) for i in matched)
    return {**summary, "admitted": True, "same_inputs": exact, "useful": generated and located,
            "human_flag": not generated and bool(matched),
            "source_ranges": native_positions(doc, "eml", path), "body": body,
            "selected": [item["input_sha256"] for item in windows if item["score"] >= .5][:6]}


def run(folder, panels, rows, responses):
    frozen = json.loads((folder / "freeze.json").read_bytes())
    base.require(frozen == freeze(panels, rows), "Writing layout freeze changed")
    media = folder / "eml"
    media.mkdir(mode=0o700)
    counts = defaultdict(Counter)
    records = []
    quote = "This paragraph is earlier correspondence. Its text is excluded from the reply check."
    for i, row in enumerate(rows):
        originals = segment(from_string(row["text"]))
        for count in LAYOUTS:
            if len(originals) < count:
                counts[count]["source_unavailable"] += 1
                continue
            path = media / f"{i:03d}-{count}.eml"
            base.author(path, repack(originals, count), quote, "signature_and_quote")
            measured = outcome(path, row, originals, responses, count)
            records.append({"index": i, "paragraph_count": count, **measured})
            cell = counts[count]
            cell["files"] += 1
            for key in ("admitted", "same_inputs", "eligible", "generated", "useful", "human_flag",
                        "baseline_useful", "baseline_human_flag"):
                cell[key] += measured[key]
            if unavailable := measured.get("unavailable"):
                cell[unavailable] += 1
    controls = []
    for i in range(12):
        for delivery in ("inline_quote", "html"):
            path = media / f"control-{i:02d}-{delivery}.eml"
            base.author(path, repack(segment(from_string(rows[i]["text"])), 2), quote, delivery)
            base.require(mapped_body(extract(path, ".eml"), 2) is None, "An excluded control entered the broader scope")
            controls.append(base.sha(path))
    gates = frozen["gate"]
    passed = all(cell["files"] > 0 and cell["same_inputs"] >= gates["min_context_parity"] * cell["eligible"]
                 and cell["useful"] >= gates["min_generated_locations_retained"] * cell["baseline_useful"]
                 and cell["human_flag"] <= cell["baseline_human_flag"] for cell in counts.values())
    base.write(folder / "result.json", {"schema": SCHEMA, "freeze_sha256": base.sha(folder / "freeze.json"),
               "result": counts, "controls": controls, "passed": passed, "cases": records})
    print(json.dumps({"result": counts, "controls": len(controls), "passed": passed}, sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "run"))
    cli.add_argument("folder", type=Path)
    cli.add_argument("panels", type=Path)
    args = cli.parse_args()
    rows, responses, _ = source_inputs(args.panels)
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        base.write(args.folder / "freeze.json", freeze(args.panels, rows))
    else:
        run(args.folder, args.panels, rows, responses)


if __name__ == "__main__":
    main()
