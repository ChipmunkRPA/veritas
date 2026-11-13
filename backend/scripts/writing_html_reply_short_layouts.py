"""Replay known plain-text writing responses on mapped one- and four-paragraph HTML replies."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from email import policy
from email.parser import BytesParser
from html import unescape
from pathlib import Path
from xml.etree import ElementTree

from app.detectors.text.adapters import agreement
from app.detectors.text.extract import extract
from app.evidence import document_matched_context
from scripts.writing_agreement_document_format_transfer import source_inputs
from scripts.writing_checkpoint_agreement import HUMAN, useful
from scripts.writing_email_body_scope_transfer import current_windows
from scripts.writing_html_reply_transfer import digest, marked_paragraph, mime, read, save

SCHEMA = "chipmunk.writing-html-reply-short-layouts/1"
PARENT = "writing-email-body-scope-2026-11-12"
METHOD = {
    "transfer": "Replay all 100 earlier measured plain replies, with one or four body paragraphs, as exact HTML paragraphs and a trailing blockquote. Alternate the two MIME encodings, retain explicit quote roles, and independently map decoded source characters.",
    "evidence": "Each HTML case must return the same ordered candidate inputs as its original measured plain reply. Use saved responses only on identical text and native input geometry.",
    "controls": "Use unsupported no-quote, interleaved, styled, unmarked, unbalanced and table variants on one genuine source of each body count. Require zero secondary-check admissions.",
    "advance": "Reproduce every admitted plain-reply location and genuine file flag, keep the one-paragraph four-sentence condition, and reproduce every decoded MIME range. This measures format transfer, with zero overall confidence weight.",
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def author(run, panels):
    previous = panels / PARENT
    original = read(previous / "result.json")
    known = {row["case"]: row for row in original["outcomes"] if row["delivery"] == "quote"}
    require(len(known) == 100, "Expected every prior plain reply")
    rosters, extra_controls = [], set()
    for i, old in known.items():
        name = f"{i:03d}-quote.eml"
        path = previous / "fixtures" / name
        require(old["admitted"] and digest(path) == old["file_sha256"], "Prior file commitment differs")
        doc = extract(path, ".eml", allow_ocr=False)
        body, quoted = [], []
        for part in doc.source_map:
            target = body if part["quote_role"] == "body" else quoted
            target.append(marked_paragraph(doc.text[part["start"]:part["end"]]))
        require(len(body) in {1, 4} and quoted, "Unexpected native reply layout")
        reply = "\n".join(body)
        quote = "<blockquote>\n" + "\n".join(quoted) + "\n</blockquote>"
        path = run / name
        mime(path, "<!doctype html><html><body>\n" + reply + "\n" + quote + "\n</body></html>", alternate=bool(i % 2))
        rosters.append({"case": i, "filename": name, "plain_sha256": old["file_sha256"], "html_sha256": digest(path),
                        "paragraphs": len(body), "control": None})
        if old["synthetic"] or len(body) in extra_controls:
            continue
        extra_controls.add(len(body))
        layouts = {
            "no_quote": reply,
            "interleaved": quote + reply,
            "styled": reply.replace("<strong>", '<strong style="color:red">', 1) + quote,
            "unmarked": reply + "<p>On Friday someone wrote:</p>" + quote,
            "unbalanced": reply.replace("</strong>", "</em>", 1) + quote,
            "table": "<table><tr><td>" + reply + "</td></tr></table>" + quote,
        }
        for role, markup in layouts.items():
            path = run / f"control-{len(body)}-{role}.eml"
            mime(path, markup, alternate=bool(i % 2))
            rosters.append({"case": i, "filename": path.name, "plain_sha256": None, "html_sha256": digest(path),
                            "paragraphs": len(body), "control": role})
    require(extra_controls == {1, 4} and len(rosters) == 112, "Incomplete HTML layout freeze")
    return rosters


def independently_mapped(path, doc, body_spans):
    raw_message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    mime_body, attachment = raw_message.iter_parts()
    require(mime_body.get_content_type() == "text/html" and attachment.get_content_disposition() == "attachment",
            "Wrong selected message part")
    raw = mime_body.get_content()
    blocks = ElementTree.fromstring(raw.removeprefix("<!doctype html>")).find("body")
    require(blocks is not None and blocks[-1].tag == "blockquote", "Missing trailing blockquote")
    expected = [(block, "body") for block in list(blocks)[:-1]] + [(block, "quoted") for block in blocks[-1]]
    measured = 0
    native_body = []
    for (block, role), mapping in zip(expected, doc.source_map, strict=True):
        start, stop = mapping["start"], mapping["end"]
        exact = " ".join("".join(block.itertext()).split())
        require(mapping["quote_role"] == role and mapping["mime_part"] == "1.1" and doc.text[start:stop] == exact,
                "HTML block text or source role differs")
        if role == "body":
            native_body.append([start, stop])
        cursor, source_cursor = start, 0
        for a, b, c, d in mapping["source_ranges"]:
            piece = unescape(re.sub(r"</?strong>", "", raw[c:d]))
            target = doc.text[a:b]
            require(cursor == a < b and source_cursor <= c < d <= len(raw)
                    and (piece == target or target == " " and piece.isspace()), "HTML coordinate mismatch")
            cursor, source_cursor = b, d
            measured += 1
        require(cursor == stop, "Unmapped HTML block characters")
    require(native_body == body_spans, "Plain and HTML reply boundaries differ")
    return measured, mime_body["Content-Transfer-Encoding"]


def measure(run, panels, frozen):
    prior = panels / PARENT
    require(frozen["prior_result"] == digest(prior / "result.json"), "Prior reply outcomes changed")
    earlier = {row["case"]: row for row in read(prior / "result.json")["outcomes"] if row["delivery"] == "quote"}
    rows, responses, _ = source_inputs(panels)
    controls, aggregate, outcomes = Counter(), defaultdict(Counter), []
    for case in frozen["cases"]:
        path = run / case["filename"]
        require(digest(path) == case["html_sha256"], "Frozen HTML bytes differ")
        doc = extract(path, ".eml", allow_ocr=False)
        source = rows[case["case"]]
        windows, pinned, sents = current_windows(doc, source)
        decoded = {"text": doc.text, "paragraphs": doc.paragraphs, "sentences": sents,
                   "meta": doc.meta, "source_map": doc.source_map}
        primary = {"detector_id": document_matched_context.DETECTOR, "version": "1.0.0",
                   "model_revisions": {document_matched_context.MODEL: document_matched_context.REVISION},
                   "details": {"text_evidence": {"windows": windows}}}
        selected = agreement.candidates(decoded, primary)
        if case["control"] is not None:
            require(selected is None, "Unsupported HTML control entered the second check")
            controls[case["control"]] += 1
            continue
        old = earlier[case["case"]]
        plain_path = prior / "fixtures" / path.name
        require(digest(plain_path) == case["plain_sha256"], "Frozen plain bytes differ")
        plain = extract(plain_path, ".eml", allow_ocr=False)
        require(doc.text == plain.text and doc.paragraphs == plain.paragraphs, "Full extracted reply differs")
        position_count, encoding = independently_mapped(path, doc, old["body_spans"])
        require(selected is not None, "Verified HTML reply unexpectedly skipped")
        ordered = [w["input_sha256"] for w in selected]
        require(ordered == old["selected_contexts"] and all(w["unit_ref"] in pinned for w in selected),
                "HTML reply did not select the same complete primary contexts")
        hits = [pinned[w["unit_ref"]] for w in selected if responses[w["input_sha256"]]["score"] >= .5]
        hit_hashes = [row["sha256"] for row in hits]
        synthetic = source["kind"] not in HUMAN
        found = synthetic and any(useful(source["text"], source["truth"], row["spans"]) for row in hits)
        false_flag = not synthetic and bool(hits)
        require(hit_hashes == old["corroborated_contexts"] and found == old["useful"] and false_flag == old["human_flag"],
                "The same complete contexts did not reproduce the previous location outcomes")
        cell = aggregate[str(case["paragraphs"])]
        cell.update(deliveries=1, generated=synthetic, useful=found, human=not synthetic, human_flags=false_flag,
                    checked_inputs=len(ordered), verified_html_ranges=position_count, **{encoding: 1})
        outcomes.append({**case, "body_spans": old["body_spans"], "selected": ordered, "corroborated": hit_hashes,
                         "exact_selected_spans": [w["spans"] for w in selected], "useful": found,
                         "human_flag": false_flag, "synthetic": synthetic, "verified_html_ranges": position_count,
                         "encoding": encoding})
    require(len(outcomes) == 100 and sum(controls.values()) == 12, "Incomplete replay outcome")
    return {"schema": SCHEMA, "method": METHOD, "prior_result": frozen["prior_result"],
            "freeze_sha256": digest(run / "freeze.json"), "counts": aggregate, "excluded": controls, "cases": outcomes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "measure", "review"))
    parser.add_argument("run", type=Path)
    parser.add_argument("panels", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "freeze":
        args.run.mkdir(mode=0o700)
        save(args.run / "freeze.json", {"schema": SCHEMA, "method": METHOD, "cases": author(args.run, args.panels),
                                      "prior_result": digest(args.panels / PARENT / "result.json"),
                                      "runner": digest(Path(__file__))})
        return
    frozen = read(args.run / "freeze.json")
    require(frozen["runner"] == digest(Path(__file__)), "Frozen runner changed")
    result = measure(args.run, args.panels, frozen)
    if args.phase == "measure":
        save(args.receipt, result)
    else:
        require(read(args.receipt) == result, "Recorded transfer changed")
    print(json.dumps({key: result[key] for key in ("counts", "excluded")}, sort_keys=True))


if __name__ == "__main__":
    main()
