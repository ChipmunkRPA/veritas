# SPDX-License-Identifier: MIT
"""Transfer the frozen passage review across native Word, email and PDF deliveries."""

from __future__ import annotations

import argparse
import html
import math
from collections import Counter, defaultdict
from email.message import EmailMessage
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import writing_multi_passage_panel as source
from scripts.writing_multi_passage_selection import _match

FORMATS = ("text", "word", "email_plain", "email_html", "pdf")
EXTENSION = {"text": ".txt", "word": ".docx", "email_plain": ".eml", "email_html": ".eml", "pdf": ".pdf"}
QUOTE = "The previous message included these figures for reference. Its wording belongs to the earlier correspondence."


def _page_pdf(text, font_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen.canvas import Canvas

    font = "WritingSourceBody"
    if font not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(font, str(font_path)))
    wrapped = []
    for paragraph in text.split("\n\n"):
        lines, line, width = [], [], 0.0
        for word in paragraph.split():
            advance = pdfmetrics.stringWidth(word + " ", font, 10)
            if width + advance > 500 and line:
                lines.append(" ".join(line))
                line, width = [], 0.0
            line.append(word)
            width += advance
        if line:
            lines.append(" ".join(line))
        if lines:
            wrapped.append(lines)
    output = BytesIO()
    document = Canvas(output, pagesize=A4, invariant=1, pageCompression=1)
    y = 58
    for paragraph in wrapped:
        while paragraph:
            available = 1 + int((735 - y) // 14)
            if available <= 0:
                document.showPage()
                y = 58
                continue
            lines, paragraph = paragraph[:available], paragraph[available:]
            block = document.beginText(54, A4[1] - y)
            block.setFont(font, 10)
            block.setLeading(14)
            block.textLines(lines)
            document.drawText(block)
            y += 14 * len(lines) + 12
    document.save()
    return output.getvalue()


def fixture(kind, text, font, *, quote=QUOTE):
    if kind == "text":
        return text.encode()
    if kind == "word":
        from scripts.rebuild_writing_word_boundary_sources import docx

        return docx(text)
    if kind == "pdf":
        return _page_pdf(text, font)
    message = EmailMessage()
    message["Subject"] = "Correspondence for review"
    if kind == "email_plain":
        message.set_content(text + "\n\n-- \nCorrespondence desk\n\nOn Monday, the editor wrote:\n> " + quote,
                            cte="quoted-printable")
    elif kind == "email_html":
        paragraphs = "".join(f"<p>{html.escape(p)}</p>" for p in text.split("\n\n"))
        body = "<html><body>" + paragraphs + f"<blockquote><p>{html.escape(quote)}</p></blockquote></body></html>"
        message.set_content(body.encode("ascii", "xmlcharrefreplace").decode(), subtype="html", cte="quoted-printable")
    else:
        raise ValueError("Unknown native writing fixture")
    return message.as_bytes()


def inputs(sentences):
    return {"isolated": [row.text for row in sentences],
            "context": [" ".join(x.text for x in sentences[max(0, i-1):i+2]) for i in range(len(sentences))]}


def prepare(variant, kind, font, path):
    from app.detectors.text.extract import extract
    from app.detectors.text.segment import segment
    from app.evidence import document_interval_locations as location
    from app.evidence import document_passage_intervals as proof

    path.write_bytes(fixture(kind, variant["text"], font))
    doc = extract(path, path.suffix, allow_ocr=False)
    method = doc.meta["method"]
    end = len(doc.text) if kind == "text" else location.checked_end(method, doc.text, doc.source_map, doc.meta, doc.word_review)
    body = doc.text[:end] if end is not None else ""
    compact = source.compact(body) == source.compact(variant["text"])
    sents = [row for row in segment(doc) if end is not None and row.end <= end]
    prepared = inputs(sents) if proof.MIN_SENTENCES <= len(sents) <= proof.MAX_SENTENCES else {}
    expected = [variant["text"][a:b] for a, b, *_ in variant["sentences"]]
    case = {"carrier": kind, "version": variant["version"], "native_sha256": source.digest(path.read_bytes()),
            "body_sha256": source.digest(body.encode()), "text_order_preserved": compact,
            "in_scope": bool(prepared), "sentences": len(sents), "pages": doc.meta.get("pages"),
            "same_reading_inputs": bool(prepared) and prepared["isolated"] == expected,
            "excluded_characters": len(doc.text) - len(body) if end is not None else None}
    return case, doc, sents, end, prepared


def run(args):
    import torch

    from app.core.canonical import FLOAT_DP, canonical_hash
    from app.detectors.text.classifier import FourClassTextClassifier
    from app.evidence import document_passage_intervals as proof
    from scripts.audio_strong_traffic_view import network_free
    from scripts.writing_word_edit_transfer import model_assets

    plan, measured = source.read(args.plan), source.read(args.measured)
    known = {}
    for work, previous in zip(plan["works"], measured["works"], strict=True):
        assert work["document"] == previous["document"]
        for variant, reading in zip(work["variants"], previous["variants"], strict=True):
            assert variant["native_text_sha256"] == reading["native_text_sha256"]
            iso, context = source._inputs(variant)
            for text, value in [*zip(iso, reading["isolated"], strict=True), *zip(context, reading["context"], strict=True)]:
                assert known.setdefault(text, value) == value
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    network_free()
    model_assets(args.snapshot)
    classifier = FourClassTextClassifier.from_snapshot(args.snapshot, "cpu")
    prior_readings = len(known)
    output = []
    with TemporaryDirectory(prefix="writing-carriers-", dir=args.output.parent) as temporary:
        for work in (row for row in plan["works"] if row["split"] == "test"):
            for variant in (v for v in work["variants"] if v["layout"] == "published"):
                for kind in FORMATS:
                    path = Path(temporary) / ("source" + EXTENSION[kind])
                    case, doc, sents, end, prepared = prepare(variant, kind, args.font, path)
                    case.update(document=work["document"], genre=work["genre"], targets=[row["compact"] for row in variant["truth"]])
                    if prepared:
                        extra = sorted(set(prepared["isolated"] + prepared["context"]) - known.keys())
                        if extra:
                            raw = classifier.predict(extra, batch=12, check=lambda _: None)
                            known.update((text, round(float(score), 8)) for text, score in zip(extra, raw, strict=True))
                        scores = {key: [round(known[text], FLOAT_DP) for text in series] for key, series in prepared.items()}
                        assert all(math.isfinite(value) for arm in scores.values() for value in arm)
                        positions = [[row.start, row.end, row.n_words] for row in sents]
                        proof_value = proof.capture(doc.text, doc.source_map, positions, scores,
                                                    method=doc.meta["method"], checked_end=end)
                        source_evidence = {"source": {"method": doc.meta["method"], "chars": len(doc.text),
                                                     proof.COMMITMENT: canonical_hash(proof_value)},
                                           "coverage": doc.meta | {proof.KEY: proof_value}}
                        proof.verify_source(source_evidence, doc.text, doc.source_map,
                                            [{"start": a, "end": b, "n_words": words} for a, b, words in positions],
                                            metadata=doc.meta, word_review=doc.word_review)
                        mapped = source._prefix(doc.text)
                        intervals = [[mapped[row["start"]], mapped[row["end"]]] for row in proof_value["ranges"]]
                        case.update(status=proof_value["status"], selected=intervals)
                    else:
                        case.update(status="unassessed", selected=[])
                    output.append(case)
            if len(output) % 50 == 0:
                print(f"Replayed {len(output)} file deliveries.", flush=True)
    receipt = {"schema": "chipmunk.writing-native-passages/1", "input_plan_sha256": source.digest(args.plan.read_bytes()),
               "previous_readings_sha256": source.digest(args.measured.read_bytes()), "font_sha256": source.digest(args.font.read_bytes()),
               "rows": output, "new_inference_inputs": len(known) - prior_readings}
    receipt["summary"] = summarize(receipt)
    source.save(args.output, receipt)
    print(receipt["summary"], flush=True)


def summarize(receipt):
    results = defaultdict(Counter)
    originals = {(row["document"], row["version"]): row for row in receipt["rows"] if row["carrier"] == "text"}
    for item in receipt["rows"]:
        key = item["carrier"]
        group = results[key]
        comparable = item["text_order_preserved"]
        matches = _match([{"compact": row} for row in item["selected"]], item["targets"]) if comparable else ()
        assert item["status"] in {"flagged", "no_ranges", "unassessed"} and (item["status"] == "flagged") == bool(item["selected"])
        group["deliveries"] += 1
        group["ordinary_alerts"] += item["version"] == "ordinary" and bool(item["selected"])
        group["selected_passages"] += len(matches)
        group["labeled_passages"] += len(item["targets"])
        group["unassessed"] += item["status"] == "unassessed"
        if item["version"] == "mixed" and comparable:
            group["unmatched_mixed_ranges"] += len(item["selected"]) - len(matches)
        original = originals[item["document"], item["version"]]
        group["exact_match_to_plain"] += comparable and (item["selected"], item["status"]) == (original["selected"], original["status"])
        group["same_reading_inputs"] += item["same_reading_inputs"]
        group["text_order_preserved"] += item["text_order_preserved"]
    return {key: dict(counts) for key, counts in results.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    score = phases.add_parser("score")
    for key in ("plan", "measured", "snapshot", "font", "output"):
        score.add_argument(key, type=Path)
    review = phases.add_parser("review")
    review.add_argument("receipt", type=Path)
    args = parser.parse_args()
    if args.phase == "score":
        run(args)
    else:
        read = source.read(args.receipt)
        assert summarize(read) == read["summary"]
        print(read["summary"])
