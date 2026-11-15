import json
from copy import deepcopy
from email.message import EmailMessage
from html import escape

import docx
import pymupdf
import pytest
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import extract
from app.evidence import document_chat_markers as chat
from app.evidence import document_packets
from app.reports.multimedia_evidence import _chat_markers_view

FORMS = (
    ("<|start|>assistant<|channel|>final<|message|>", "<|return|>"),
    ("<|im_start|>assistant", "<|im_end|>"),
    ("<|start_header_id|>assistant<|end_header_id|>", "<|eot_id|>"),
    ("<start_of_turn>model", "<end_of_turn>"),
    ("<|turn>model", "<turn|>"),
    ("<｜Assistant｜>", "<｜end▁of▁sentence｜>"),
    ("]~b]ai", "[e~["),
    ("<|im_assistant|>assistant<|im_middle|>", "<|im_end|>"),
)
START, END = FORMS[1]
TURN = START + "\nLeave the ledger on the north desk." + END


def measured(source):
    doc = extract(source, source.suffix)
    return doc, chat.measure(doc.text, doc.source_map, doc.meta)


@pytest.mark.parametrize(("start", "end"), FORMS)
def test_exact_plain_markers_and_source_ranges(tmp_path, start, end):
    raw = "  🧪 Inventory note.\r\n\r\n" + start + "\r\nThank you for the note.\r\n" + end
    source = tmp_path / "upload.txt"
    source.write_bytes(b"\xef\xbb\xbf" + raw.encode())
    doc, record = measured(source)
    hit, = record["locations"]
    for role, marker in (("opener", start), ("closer", end)):
        location = hit[role]
        assert doc.text[location["start"]:location["end"]] == marker
        assert location["ranges"] == [[raw.index(marker), raw.index(marker) + len(marker)]]
    assert record["turns"] == 1 and record["authorship_confidence_contribution"] == 0


@pytest.mark.parametrize("control", [
    "The legal assistant: the records are already on file. User: Thank you.",
    START + "ship code by Friday" + END,
    START + "\nNo closing delimiter.",
    "The pricing write-up uses only " + END,
    START + "\nTwo closing types do not pair.<end_of_turn>",
    "<|im_start|>user\nThe training example asks a question." + END,
    START + "\nThe next conversation starts.<|im_start|>user\nThe other reply never ran." + END,
    START + "\n⚠️" + END,
    "This is a literal code sample.\n\n```text\n\n" + TURN + "\n\n```",
])
def test_incomplete_and_ordinary_turns_are_not_listed(tmp_path, control):
    source = tmp_path / "control.txt"
    source.write_text(control, encoding="utf-8")
    _, record = measured(source)
    assert record["turns"] == 0 and record["locations"] == []


def test_source_anchors_for_word_email_and_selectable_pdf(tmp_path):
    word = tmp_path / "note.docx"
    document = docx.Document()
    document.add_paragraph(START)
    document.add_paragraph("Leave the ledger on the north desk.")
    document.add_paragraph(END)
    document.save(word)
    _, record = measured(word)
    assert [(place["anchor"]["body_path"], place["ranges"]) for place in record["locations"][0].values()] == [
        ([0], [[0, len(START)]]), ([2], [[0, len(END)]])]

    email = tmp_path / "mail.eml"
    message = EmailMessage()
    html = f"<html><body><p>{escape(START)}</p><p>Leave the ledger on the north desk.{escape(END)}</p></body></html>"
    message.set_content(html, subtype="html")
    email.write_bytes(message.as_bytes())
    _, record = measured(email)
    for place in record["locations"][0].values():
        [(a, b)] = place["ranges"]
        assert html[a:b] == escape(place["literal"]) and place["anchor"]["mime_part"] == "1"
    message.set_content(f"<html><body><blockquote><p>{escape(TURN)}</p></blockquote></body></html>", subtype="html")
    email.write_bytes(message.as_bytes())
    _, record = measured(email)
    assert record is None or record["turns"] == 0

    pdf = tmp_path / "note.pdf"
    with pymupdf.open() as file:
        page = file.new_page()
        page.insert_text((72, 120), TURN)
        file.save(pdf)
    doc, record = measured(pdf)
    assert record["turns"] == 1
    assert json.loads(canonical_json(record)) == record
    for place in record["locations"][0].values():
        assert place["anchor"]["page"] == 1 and "ranges" not in place
        assert doc.text[place["start"]:place["end"]] == place["literal"]


def test_bounded_locations_and_rebuilt_evidence(packet_factory, tmp_path):
    source = tmp_path / "many.txt"
    source.write_text("\n\n".join([TURN] * (chat.MAX_TURNS + 1)), encoding="utf-8")
    _, record = measured(source)
    assert record["turns"] == chat.MAX_TURNS + 1
    assert len(record["locations"]) == chat.MAX_TURNS

    db, store, analysis, result, decoded, _ = packet_factory(text=(
        "This first line mentions the inventory. " + TURN + " The following sentence states the delivery date."
    ))
    older = document_packets.build_packet_set(db, analysis, store, result)
    assert chat.KEY not in older["coverage"]
    record = chat.measure(decoded["text"], decoded["source_map"], decoded["meta"])
    assert record["turns"] == 1
    result["manifest"]["sampling"][chat.COMMITMENT] = canonical_hash(record)
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    evidence = document_packets.persist_packet_set(db, analysis, store, result=result)
    assert document_packets.load_packet_set(db, analysis, store) == evidence
    view = _chat_markers_view(evidence["coverage"])
    assert view["locations"][0]["opener"]["literal"] == START
    assert "Decoded source" in view["locations"][0]["opener"]["label"]
    forged = deepcopy(evidence)
    marker = forged["coverage"][chat.KEY]["locations"][0]["opener"]
    marker["ranges"] = [[a + 1, b + 1] for a, b in marker["ranges"]]
    forged["source"][chat.COMMITMENT] = canonical_hash(forged["coverage"][chat.KEY])
    with pytest.raises(ValueError, match="source text"):
        chat.verify_source(forged, decoded["text"], decoded["source_map"], decoded["meta"])
