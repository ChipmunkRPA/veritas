"""Replay exact plain-text reply inputs through a source-mapped HTML-only MIME body."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from email.message import EmailMessage
from hashlib import sha256
from html import escape
from pathlib import Path

from app.detectors.text.extract import extract, text_hash
from app.detectors.text.segment import segment
from app.evidence import document_scope
from scripts.writing_paired_email_revision import native_contexts

SCHEMA = "chipmunk.writing-html-reply-transfer/1"
DESIGN = {
    "formats": "Re-encode 308 earlier measured plain replies with 2, 3, 5, or 6 body paragraphs as HTML-only MIME. Use explicit trailing blockquote, inline emphasis, and alternate base64/quoted-printable transfer encoding. Keep decoded text exactly the same.",
    "measure": "Replay only identical ordered primary contexts and complete per-reply second-check requests. Any source, text, request, or selected location mismatch fails.",
    "controls": "Six held controls per unsupported structure: no quote, interleaved quote, styled text, unmarked quote, malformed nesting and unsupported markup. The second check must abstain on every control.",
    "advance": "Preserve every useful known-edit location and human flag count. Verify every selected context against original decoded HTML ranges. Zero authorship confidence weight.",
}


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save(path, data):
    with path.open("x") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def marked_paragraph(text):
    word = re.search(r"[^\W_]+", text)
    assert word is not None
    start, end = word.span()
    return f"<p>{escape(text[:start])}<strong>&#{ord(text[start])};{escape(text[start+1:end])}</strong>{escape(text[end:])}</p>"


def mime(path, markup, *, alternate):
    msg = EmailMessage()
    msg["From"] = "fixture@example.invalid"
    msg["To"] = "review@example.invalid"
    msg["Subject"] = "Locally generated HTML reply fixture"
    msg.set_content(markup, subtype="html", cte="base64" if alternate else "quoted-printable")
    msg.add_attachment(b"Unrelated test attachment.", maintype="application", subtype="octet-stream", filename="ignore.bin")
    msg.set_boundary("chipmunk-html-reply-" + text_hash(markup)[:32])
    path.write_bytes(msg.as_bytes())


def author(run, plain, previous):
    rows = read(previous / "decisions.json")
    folders = run / "html"
    folders.mkdir(mode=0o700)
    cases = []
    for i, row in enumerate(rows):
        name = f'{row["source"]:03d}-{row["layout"]}.eml'
        source = plain / name
        doc = extract(source, ".eml", allow_ocr=False)
        body, tail = [], []
        for part in doc.source_map:
            target = body if part["quote_role"] == "body" else tail
            target.append(marked_paragraph(doc.text[part["start"]:part["end"]]))
        assert len(body) == row["layout"] and tail
        reply, quote = "\n".join(body), "<blockquote>\n" + "\n".join(tail) + "\n</blockquote>"
        path = folders / name
        mime(path, "<!doctype html><html><body>\n" + reply + "\n" + quote + "\n</body></html>", alternate=i % 2)
        cases.append({"name": name, "html": digest(path), "plain": digest(source), "control": None})
        if i < 6:
            variants = {
                "no_quote": reply,
                "interleaved": "\n".join([body[0], quote, *body[1:]]),
                "styled": reply.replace("<strong>", '<strong style="color: red">', 1) + quote,
                "unmarked": reply + "<p>On Friday someone wrote:</p>" + quote,
                "unbalanced": reply.replace("</strong>", "</em>", 1) + quote,
                "other_markup": "<table><tr><td>" + reply + "</td></tr></table>" + quote,
            }
            for role, markup in variants.items():
                path = folders / f"control-{i:02d}-{role}.eml"
                mime(path, markup, alternate=i % 2)
                cases.append({"name": path.name, "html": digest(path), "plain": None, "control": role})
    return cases


def measurements(run, plain, previous, cases):
    earlier = read(previous / "freeze.json")
    responses = read(previous / "responses.json")
    assert responses["freeze_sha256"] == digest(previous / "freeze.json")
    earlier_decisions = {f'{row["source"]:03d}-{row["layout"]}.eml': row for row in read(previous / "decisions.json")}
    totals, controls, records = {}, Counter(), []
    for case in cases:
        name = case["name"]
        path = run / "html" / name
        assert digest(path) == case["html"]
        html = extract(path, ".eml", allow_ocr=False)
        body = document_scope.email_agreement_body_spans(html.source_map, html.meta["email"], len(html.text))
        if case["control"]:
            assert body is None, case
            controls[case["control"]] += 1
            continue
        assert body is not None, case
        prior_path = plain / name
        assert digest(prior_path) == case["plain"]
        before = extract(prior_path, ".eml", allow_ocr=False)
        old_body = document_scope.email_agreement_body_spans(before.source_map, before.meta["email"], len(before.text))
        assert html.text == before.text and body == old_body and html.paragraphs == before.paragraphs
        old_sentences, new_sentences = segment(before), segment(html)
        assert old_sentences == new_sentences
        windows = []
        for focus, context in zip(new_sentences, native_contexts(new_sentences), strict=True):
            spans = [{"start": s.start, "end": s.end} for s in context]
            if document_scope.checkpoint_window_in_scope({"focus": {"start": focus.start, "end": focus.end}, "spans": spans}, body):
                windows.append({"sha256": text_hash(" ".join(s.text for s in context)), "spans": spans,
                                "focus": [focus.start, focus.end]})
        old = earlier_decisions[name]
        ordered = earlier["requests"][old["request"]]
        assert all(sum(row["sha256"] == key for row in windows) == 1 for key in ordered)
        selected = []
        for row in windows:
            if len(selected) < len(ordered) and row["sha256"] == ordered[len(selected)]:
                selected.append(row)
        assert len(selected) == len(ordered)
        secondary = responses["readings"][old["request"]]
        corroborated = [item["sha256"] for item, response in zip(selected, secondary, strict=True)
                        if float.fromhex(response["score_hex"]) >= .5]
        assert corroborated == old["corroborated"]
        counts = totals.setdefault(str(old["layout"]), Counter())
        counts.update(replies=1, exact_contexts=len(windows), second_inputs=len(selected), useful=old["useful"],
                      human_flags=old["human_flag"], corroborated=len(corroborated))
        records.append({"name": name, "body": body, "source": html.meta["email"]["body_part"], "selected": selected,
                        "complete_contexts": len(windows), "corroborated": corroborated})
    return {"by_layout": totals, "controls": controls, "records": records}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("freeze", "measure", "replay"))
    for name in ("run", "plain", "previous"):
        cli.add_argument(name, type=Path)
    args = cli.parse_args()
    run = args.run
    paths = (Path(__file__), Path(extract.__code__.co_filename), Path(document_scope.__file__))
    code = {path.name: digest(path) for path in paths}
    previous = {name: digest(args.previous / name) for name in ("freeze.json", "decisions.json", "responses.json")}
    if args.phase == "freeze":
        run.mkdir(mode=0o700)
        cases = author(run, args.plain, args.previous)
        save(run / "cases.json", cases)
        save(run / "freeze.json", {"schema": SCHEMA, "design": DESIGN, "code": code, "earlier": previous,
                                    "cases": digest(run / "cases.json")})
        print(f"Froze {len(cases)} MIME deliveries", flush=True)
        return
    lock = read(run / "freeze.json")
    assert lock["code"] == code and lock["earlier"] == previous and lock["cases"] == digest(run / "cases.json")
    result = measurements(run, args.plain, args.previous, read(run / "cases.json"))
    if args.phase == "measure":
        save(run / "result.json", result)
    else:
        assert read(run / "result.json") == result
    print(json.dumps({key: value for key, value in result.items() if key != "records"}), flush=True)


if __name__ == "__main__":
    main()
