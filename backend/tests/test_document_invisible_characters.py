from __future__ import annotations

from copy import deepcopy
from email.message import EmailMessage

import docx
import pytest
from test_document_packets import packet_factory as packet_factory

from app.core.canonical import canonical_hash, canonical_json
from app.detectors.text.extract import extract
from app.evidence import document_invisible_characters as controls
from app.evidence import document_packets
from app.reports import multimedia_evidence


def measure(source, *, schema=controls.SCHEMA):
    document = extract(source, source.suffix)
    return controls.measure(document.text, document.source_map, document.meta, schema=schema)


def test_plain_coordinates_are_decoded_codepoints_and_preserve_original(tmp_path):
    raw = "  🧪 Alpha\u200bBeta.\r\n\r\nCheck 31\u2060\u00ad25, then \u202einvoice\u202c and \ufeff a label."
    source = tmp_path / "unicode.txt"
    source.write_bytes(b"\xef\xbb\xbf" + raw.encode())
    record = measure(source)
    assert record["checked_blocks"] == 2
    matched = "\u200b\u2060\u00ad\u202e\ufeff"
    assert [r["source_start"] for r in record["locations"]] == [raw.index(c) for c in matched]
    assert [r["codepoint"] for r in record["locations"]] == [f"U+{ord(c):04X}" for c in matched]
    assert all(row["source_end"] == row["source_start"] + 1 for row in record["locations"])
    original = extract(source, ".txt")
    assert [original.text[r["text_offset"]] for r in record["locations"]] == list(matched)
    assert record["authorship_confidence_contribution"] == 0


def test_normal_unicode_formatting_and_bounded_locations(tmp_path):
    source = tmp_path / "controls.txt"
    source.write_text("العربية\u200bنص עברית \u2067עברית\u2069 zero\u200b width 👩\u200d🔬 نرم\u200cافزار\u200e.")
    assert measure(source)["matches"] == 0
    source.write_text("a" + "\u200b" * 31 + "z")
    record = measure(source)
    assert record["matches"] == 31 and len(record["locations"]) == controls.MAX_LOCATIONS
    assert record["locations"][-1]["source_start"] == controls.MAX_LOCATIONS


def test_mixed_lookalikes_require_a_mostly_basic_latin_letter_run(tmp_path):
    source = tmp_path / "glyphs.txt"
    raw = "🧪 It shows the c\u0430se, later \u03bfption and an in\u200bvoice. café c\u0430fé c\u0430\u0301se A\u03b1 δelta \u0441лово αλφα."
    source.write_text(raw)
    report = measure(source, schema=controls.GLYPH_SCHEMA)
    assert [(row["codepoint"], row.get("looks_like"), row["source_start"]) for row in report["locations"]] == [
        ("U+0430", "a", raw.index("\u0430")), ("U+03BF", "o", raw.index("\u03bf")), ("U+200B", None, raw.index("\u200b")),
    ]
    assert report["authorship_confidence_contribution"] == 0
    assert measure(source)["matches"] == 1


@pytest.mark.parametrize("codepoint,ascii_letter", controls.LOOKALIKES.items())
def test_each_pinned_lookalike_preserves_the_terminal_token_and_excludes_whole_script(codepoint, ascii_letter):
    letter = chr(codepoint)
    assert list(controls._lookalikes("nav" + letter)) == [(3, letter)]
    assert not list(controls._lookalikes(letter * 4))
    assert controls.GLYPHS[f"U+{codepoint:04X}"] == ascii_letter


def test_word_paragraph_and_email_html_entity_keep_native_locations(tmp_path):
    word = tmp_path / "report.docx"
    content = docx.Document()
    content.add_paragraph("The signed original uses ordinary text.")
    content.add_table(rows=1, cols=1).cell(0, 0).text = "🧪 Acct\u200bNumber is on the first line."
    content.save(word)
    item, = measure(word)["locations"]
    assert item["body_path"] == [1, "row", 0, "cell", 0, 1]
    assert (item["source_start"], item["source_end"]) == (6, 7)
    email = tmp_path / "mail.eml"
    message = EmailMessage()
    decoded = "<html><body><p>Update the Acct&#x200B;Number carefully and keep a copy.</p></body></html>"
    message.set_content(decoded, subtype="html")
    email.write_bytes(message.as_bytes())
    report = measure(email)
    assert report["coordinate_space"] == "decoded_email_html"
    exhibit, = report["locations"]
    assert exhibit["mime_part"] == "1"
    assert decoded[exhibit["source_start"]:exhibit["source_end"]] == "&#x200B;"
    message = EmailMessage()
    message.set_content("<p class='styled'>The Acct&#x200B;Number is included.</p>", subtype="html")
    email.write_bytes(message.as_bytes())
    assert measure(email) is None


@pytest.mark.parametrize("schema", [controls.SCHEMA, controls.GLYPH_SCHEMA])
def test_evidence_binding_and_report_reject_rehashed_forged_positions(packet_factory, schema):
    db, store, analysis, result, decoded, _ = packet_factory(
        text="First sentence contains Acct\u200bNumber and the c\u0430se terms. Second sentence has more words. Third sentence confirms.")
    baseline = document_packets.build_packet_set(db, analysis, store, result)
    assert controls.KEY not in baseline["coverage"]
    record = controls.measure(decoded["text"], decoded["source_map"], decoded["meta"], schema=schema)
    result["manifest"]["sampling"][controls.COMMITMENT] = canonical_hash(record)
    analysis.result_hash = canonical_hash(result)
    store.put_bytes(canonical_json(result))
    evidence = document_packets.persist_packet_set(db, analysis, store, result=result)
    assert evidence["coverage"][controls.KEY] == record
    assert document_packets.load_packet_set(db, analysis, store) == evidence
    report = multimedia_evidence._invisible_characters_view(evidence["coverage"])
    assert report["locations"][0]["name"] == "Zero-width space"
    assert report["locations"][0]["codepoint"] == "U+200B"
    if schema == controls.GLYPH_SCHEMA:
        assert report["locations"][1]["name"] == "Cyrillic small letter a"
        assert report["locations"][1]["looks_like"] == "a"
    assert "zero AI-authorship weight" in report["limitation"]
    assert "\u200b" not in str(report)
    forged = deepcopy(evidence)
    forged["coverage"][controls.KEY]["locations"][0]["source_start"] += 2
    forged["coverage"][controls.KEY]["locations"][0]["source_end"] += 2
    forged["source"][controls.COMMITMENT] = canonical_hash(forged["coverage"][controls.KEY])
    with pytest.raises(ValueError, match="source text"):
        controls.verify_source(forged, decoded["text"], decoded["source_map"], decoded["meta"])
