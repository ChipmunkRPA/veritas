# SPDX-License-Identifier: Apache-2.0
"""Check the exact writing inputs that remain inside a mapped email reply body."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from pathlib import Path

from app.detectors.text.adapters import agreement
from app.detectors.text.extract import extract, from_string
from app.detectors.text.segment import segment
from app.evidence import document_matched_context, document_scope
from scripts.writing_agreement_document_format_transfer import RECEIPT, native_positions, source_inputs
from scripts.writing_checkpoint_agreement import HUMAN, native_contexts, text_sha, useful

SCHEMA = "chipmunk.writing-email-body-scope-transfer/1"
DELIVERIES = ("quote", "signature", "signature_and_quote")
CONTROLS = ("quote_only", "inline_quote", "html")
PANEL = "writing-checkpoint-agreement-2026-11-11"
SCOPE = ("Use only the first contiguous plain body in a single MIME part. Retain only already measured primary contexts "
         "whose full sentence spans map inside body paragraphs. Never copy a response from a newly composed context.")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")


def plan(rows):
    originals = [i for i, row in enumerate(rows) if row["kind"] == "original"]
    require(len(rows) == 100 and len(originals) == 28, "Pinned writing source roster differs")
    cases = []
    for i, row in enumerate(rows):
        donors = [j for j in originals if rows[j]["group"] != row["group"]]
        donor = donors[int(row["text_sha256"], 16) % len(donors)]
        first = rows[donor]["windows"][0]["spans"]
        quote = " ".join(rows[donor]["text"][a:b] for a, b in first)
        cases.append({"index": i, "text_sha256": row["text_sha256"], "source_group": row["group"], "kind": row["kind"],
                      "quote_source_index": donor, "quote_text_sha256": text_sha(quote),
                      "input_hashes": [w["sha256"] for w in row["windows"]]})
    return cases


def lock(panels, rows):
    return {"schema": SCHEMA, "scope": SCOPE, "deliveries": list(DELIVERIES), "controls": list(CONTROLS),
            "parent_receipt_sha256": sha(RECEIPT), "parent_scores_sha256": sha(panels / PANEL / "scores.json"),
            "cases": plan(rows), "code_sha256": {path.name: sha(path) for path in (
                Path(__file__), Path(agreement.__file__), Path(document_scope.__file__),
            )}}


def email_body(source, quote, delivery):
    quoted = f"On a prior date someone wrote:\n> {quote}"
    signature = "-- \nAlex Example\nDocumentation Team"
    if delivery == "quote_only":
        return quoted
    if delivery == "inline_quote":
        return f"{source}\n\n> {quote}\n\nA reply follows the inline quotation."
    tails = {"quote": [quoted], "signature": [signature], "signature_and_quote": [signature, quoted]}
    return "\n\n".join([source, *tails.get(delivery, [])])


def author(path, text, quote, delivery):
    from html import escape

    message = EmailMessage(policy=policy.SMTP)
    message["From"] = "reviewer@example.invalid"
    message["To"] = "recipient@example.invalid"
    message["Subject"] = "Writing review evaluation"
    body = email_body(text, quote, delivery)
    if delivery == "html":
        message.set_content("".join(f"<p>{escape(p)}</p>" for p in body.split("\n\n")), subtype="html")
    else:
        message.set_content(body, charset="utf-8", cte="quoted-printable")
        if delivery == "signature_and_quote":
            message.add_alternative("<p>This alternate message body is outside the selected text part.</p>", subtype="html")
            message.add_attachment(b"Unrelated test attachment.", maintype="application", subtype="octet-stream",
                                   filename="excluded.txt")
            # EmailMessage otherwise inserts random MIME boundaries on serialization.
            for number, part in enumerate(p for p in message.walk() if p.is_multipart()):
                part.set_boundary(f"writing-reply-scope-{number}")
    path.write_bytes(message.as_bytes())


def current_windows(doc, row):
    sentences = segment(doc)
    originals = segment(from_string(row["text"]))
    saved = {((focus.start, focus.end), tuple(map(tuple, w["spans"])), w["sha256"]): w
             for focus, w in zip(originals, row["windows"], strict=True)}
    windows, pinned = [], {}
    for i, (focus, context) in enumerate(zip(sentences, native_contexts(sentences), strict=True)):
        spans = [[s.start, s.end] for s in context]
        key = (focus.start, focus.end), tuple(map(tuple, spans)), text_sha(" ".join(s.text for s in context))
        matched = saved.get(key)
        ref = f"sentence:{i}"
        if matched is not None:
            pinned[ref] = matched
        windows.append({"focus": {"start": focus.start, "end": focus.end},
                        "spans": [{"start": a, "end": b} for a, b in spans], "input_sha256": key[2],
                        "unit_ref": ref, "score": matched["current"] if matched else .999})
    return windows, pinned, [asdict(s) for s in sentences]


def measure(path, row, responses, previous, delivery):
    doc = extract(path, ".eml")
    windows, pinned, sentences = current_windows(doc, row)
    decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "sentences": sentences,
               "meta": doc.meta, "source_map": doc.source_map}
    primary = {"detector_id": document_matched_context.DETECTOR, "version": "1.0.0",
               "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
               "details": {"text_evidence": {"windows": windows}}}
    selected = agreement.candidates(decoded, primary)
    if delivery in CONTROLS:
        require(selected is None, "Unsupported source entered the second check")
        return {"delivery": delivery, "file_sha256": sha(path), "admitted": False}

    require(selected is not None and not agreement._supported_source(decoded), "Mixed MIME role was not admitted by the reply gate")
    bodies = [block for block in doc.source_map if block["quote_role"] == "body"]
    expected_body = [[a, b] for a, b in doc.paragraphs[:len(bodies)]]
    end = expected_body[-1][1]
    require(doc.text[:end] == row["text"], "MIME packaging changed the original reply body")
    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    raw = message.get_body(preferencelist=("plain",)).get_content()
    require(raw.replace("\r\n", "\n").startswith(row["text"]), "Exact MIME body position differs")
    for window in selected:
        require(window["unit_ref"] in pinned and all(any(a <= s["start"] < s["end"] <= b for a, b in expected_body)
                                                     for s in window["spans"]), "A second-check context includes material outside its reply")
    selected_hits = [pinned[w["unit_ref"]] for w in selected if responses[w["input_sha256"]]["score"] >= .5]
    synthetic = row["kind"] not in HUMAN
    found = synthetic and any(useful(row["text"], row["truth"], w["spans"]) for w in selected_hits)
    require(not found or previous["useful"]["agreement"], "Reused evidence unexpectedly introduced a useful location")
    require(synthetic or not selected_hits or previous["marks"]["agreement"], "Reused evidence unexpectedly added a human alert")
    return {"delivery": delivery, "file_sha256": sha(path), "admitted": True, "body_spans": expected_body,
            "excluded_roles": [b["quote_role"] for b in doc.source_map if b["quote_role"] != "body"],
            "source_positions": native_positions(doc, "eml", path), "all_primary_contexts": len(windows),
            "reply_eligible_contexts": len(pinned), "selected_contexts": [w["input_sha256"] for w in selected],
            "corroborated_contexts": [w["sha256"] for w in selected_hits], "synthetic": synthetic,
            "baseline_useful": previous["useful"]["agreement"], "useful": found,
            "baseline_human_flag": not synthetic and bool(previous["marks"]["agreement"]),
            "human_flag": not synthetic and bool(selected_hits)}


def summarize(results):
    counts = defaultdict(Counter)
    for row in results:
        cell = counts[row["delivery"]]
        cell["files"] += 1
        cell["admitted"] += row["admitted"]
        if row["admitted"]:
            for name in ("synthetic", "baseline_useful", "useful", "baseline_human_flag", "human_flag"):
                cell[name] += row[name]
            for name in ("selected_contexts", "corroborated_contexts"):
                cell[name] += len(row[name])
            cell["reply_eligible_contexts"] += row["reply_eligible_contexts"]
            cell["source_positions"] += row["source_positions"]
    return {key: dict(value) for key, value in counts.items()}


def run(panels, directory, rows, responses, previous):
    protocol = lock(panels, rows)
    require(json.loads((directory / "freeze.json").read_bytes()) == protocol, "Writing study freeze differs")
    fixtures = directory / "fixtures"
    fixtures.mkdir()
    results = []
    checked_controls = set()
    for i, (row, mark, case) in enumerate(zip(rows, previous, protocol["cases"], strict=True)):
        donor = rows[case["quote_source_index"]]
        quote = " ".join(donor["text"][a:b] for a, b in donor["windows"][0]["spans"])
        require(text_sha(quote) == case["quote_text_sha256"], "Quote source differs from the protocol")
        deliveries = list(DELIVERIES)
        if row["kind"] == "original" and row["group"] not in checked_controls and len(checked_controls) < 8:
            checked_controls.add(row["group"])
            deliveries.extend(CONTROLS)
        for delivery in deliveries:
            path = fixtures / f"{i:03d}-{delivery}.eml"
            author(path, row["text"], quote, delivery)
            result = measure(path, row, responses, mark, delivery)
            results.append({"case": i, **result})
    require(len(results) == 324, "Incomplete EML comparison")
    write(directory / "result.json", {"schema": SCHEMA, "freeze_sha256": sha(directory / "freeze.json"),
          "source_groups": 28, "aggregate": summarize(results), "outcomes": results})
    print(json.dumps(summarize(results), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "run"))
    parser.add_argument("panels", type=Path)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows, responses, previous = source_inputs(args.panels)
    if args.phase == "freeze":
        args.directory.mkdir()
        write(args.directory / "freeze.json", lock(args.panels, rows))
    else:
        run(args.panels, args.directory, rows, responses, previous)


if __name__ == "__main__":
    main()
