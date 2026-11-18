import hashlib
import json
import socket
from collections import Counter
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from html import escape, unescape
from io import BytesIO
from zipfile import ZipFile

import pytest
from docx import Document
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree
from test_strict_word import _strict_package

from app.core.canonical import canonical_hash
from app.detectors.text.extract import extract
from app.evidence import document_saved_link_hosts as reader
from app.evidence import document_saved_link_hosts_binding as binding
from app.evidence.document_saved_link_hosts import inspect

# The host pairs, outcomes and exact source spellings were committed before running the scanner.
CASES = (
    (
        "literal_difference",
        "https://records.example.org/manual?example-label-token=hidden",
        "https://relay.example.net/mail?example-target-token=secret",
        "different_host",
        "records.example.org",
        "records.example.org",
        "relay.example.net",
    ),
    (
        "shorthand_difference",
        "www.example.com/instructions",
        "https://catalog.example.net/?example-target-token=private",
        "different_host",
        "www.example.com",
        "www.example.com",
        "catalog.example.net",
    ),
    (
        "long_public_suffix",
        "https://docs.example.co.uk/archive",
        "https://backup.example.org/",
        "different_host",
        "docs.example.co.uk",
        "docs.example.co.uk",
        "backup.example.org",
    ),
    (
        "unicode_difference",
        "https://bücher.example.org/files",
        "https://store.example.com/news#hidden",
        "different_host",
        "bücher.example.org",
        "xn--bcher-kva.example.org",
        "store.example.com",
    ),
    (
        "ordinary_mailing_redirect",
        "https://bulletin.example.org/info",
        "https://campaign.example.net/forward?u=https%3A%2F%2Fbulletin.example.org%2Finfo",
        "different_host",
        "bulletin.example.org",
        "bulletin.example.org",
        "campaign.example.net",
    ),
    (
        "protected_redirect",
        "https://records.example.org/manual",
        "https://nam01.safelinks.protection.outlook.com/?url=https%3A%2F%2Frecords.example.org%2Fmanual",
        "registered_wrapper",
        None,
        None,
        None,
    ),
    (
        "related_service_hosts",
        "https://docs.example.co.uk/manual",
        "https://support.example.co.uk/manual",
        "related_hostname",
        None,
        None,
        None,
    ),
    (
        "www_alias",
        "www.example.org/manual",
        "https://example.org/manual",
        "related_hostname",
        None,
        None,
        None,
    ),
    (
        "idna_equivalence",
        "https://bücher.example.org",
        "https://xn--bcher-kva.example.org/download",
        "same_host",
        None,
        None,
        None,
    ),
    (
        "host_case_and_port",
        "http://EXAMPLE.ORG:80/info",
        "https://example.org:443/manual?example-target-token=private",
        "same_host",
        None,
        None,
        None,
    ),
    (
        "regular_label",
        "The attached records",
        "https://example.net/manual",
        "unsupported_label",
        None,
        None,
        None,
    ),
    (
        "unmarked_bare_host",
        "example.org/manual",
        "https://example.net/manual",
        "unsupported_label",
        None,
        None,
        None,
    ),
    (
        "unresolved_relative",
        "https://example.org/manual",
        "../manual",
        "unsupported_target",
        None,
        None,
        None,
    ),
    (
        "unsupported_scheme",
        "https://example.org/manual",
        "mailto:editor@example.net",
        "unsupported_target",
        None,
        None,
        None,
    ),
    (
        "ip_literal",
        "https://example.org/manual",
        "https://127.0.0.1/manual",
        "unsupported_target",
        None,
        None,
        None,
    ),
    (
        "missing_idna_equivalence",
        "https://example.org.",
        "https://example.org/",
        "same_host",
        None,
        None,
        None,
    ),
)
PREFIX = "Report 😀 label: "


def _html(cases):
    rows = []
    for name, label, target, _, raw, _, _ in cases:
        markup = escape(label)
        if name == "literal_difference":
            assert raw in label
            markup = markup.replace(raw, "records<span>&#46;example</span>.org", 1)
        rows.append(f'<p>{PREFIX}<a href="{escape(target, quote=True)}">{markup}</a> in the source.</p>')
    return (
        '<!doctype html><html><head><base href="https://base.example.org/"></head><body>'
        + "".join(rows)
        + "</body></html>"
    )


def _hyperlink(paragraph, label, target, hidden=False):
    connection = paragraph.part.relate_to(target, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), connection)
    for text in (label[: len(label) // 2], label[len(label) // 2 :]):
        run = OxmlElement("w:r")
        t = OxmlElement("w:t")
        t.set(qn("xml:space"), "preserve")
        t.text = text
        if hidden:
            properties = OxmlElement("w:rPr")
            properties.append(OxmlElement("w:vanish"))
            run.append(properties)
        run.append(t)
        link.append(run)
    paragraph._p.append(link)


def _word(cases, strict=False):
    document = Document()
    for _, label, target, _, _, _, _ in cases:
        p = document.add_paragraph(PREFIX)
        _hyperlink(p, label, target)
        p.add_run(" in the source.")
    p = document.sections[0].header.paragraphs[0]
    _hyperlink(p, "https://header.example.org", "https://header.example.net")
    p = document.add_table(1, 1).cell(0, 0).paragraphs[0]
    _hyperlink(p, "https://table.example.org", "https://table.example.net")
    p = document.add_paragraph(PREFIX)
    _hyperlink(p, "https://hidden.example.org", "https://hidden.example.net", hidden=True)
    output = BytesIO()
    document.save(output)
    return _strict_package(output.getvalue()) if strict else output.getvalue()


def _email(raw, plain=False):
    message = EmailMessage()
    message["From"] = "author@example.org"
    message["Subject"] = "One locally authored example for source verification"
    if plain:
        message.set_content("This selected plain body contains no saved hyperlink elements.")
        message.add_alternative(raw, subtype="html", cte="quoted-printable")
    else:
        message.set_content(raw, subtype="html", cte="quoted-printable")
    message.add_attachment(
        '<a href="https://attachment.example.net">https://attachment.example.org</a>',
        filename="attachment.html",
        subtype="html",
    )
    return message.as_bytes()


def test_saved_host_commitment_and_original_replay_reject_changes(tmp_path):
    path = tmp_path / "site.html"
    path.write_text(_html(CASES[:1]), encoding="utf-8")
    capture = inspect(path, ".html")
    packet = {"source": {"method": "html", binding.COMMITMENT: canonical_hash(capture)}, binding.KEY: capture}
    binding.validate(packet)
    reader.verify_original(packet, path, ".html")

    path.write_text(path.read_text().replace("relay.example.net", "changed.example.net"), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen original-file inspection"):
        reader.verify_original(packet, path, ".html")

    capture["findings"][0]["saved_host"] += "/hidden?token=dropped"
    packet["source"][binding.COMMITMENT] = canonical_hash(capture)
    with pytest.raises(ValueError, match="invalid hostname"):
        binding.validate(packet)


def test_optional_reader_discards_results_when_its_subprocess_exceeds_budget(tmp_path, monkeypatch):
    import subprocess

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(reader.subprocess, "run", timed_out)
    assert reader.inspect_within(tmp_path / "site.html", timeout=0.05, source_sha256="a" * 64) is None


def test_optional_reader_checks_the_exact_extracted_source(tmp_path):
    path = tmp_path / "site.html"
    path.write_text(_html(CASES[:1]), encoding="utf-8")
    source = hashlib.sha256(path.read_bytes()).hexdigest()
    assert reader.inspect_within(path, timeout=4, source_sha256=source) == inspect(path, ".html")
    path.write_text(_html(CASES[:2]), encoding="utf-8")
    assert reader.inspect_within(path, timeout=4, source_sha256=source) is None


@pytest.mark.parametrize("format_name", ("html", "html_utf16", "eml", "docx", "strict_docx"))
def test_native_saved_link_fixtures(tmp_path, format_name, monkeypatch):
    def network_denied(*args, **kwargs):
        pytest.fail("Link inspection attempted network access")

    monkeypatch.setattr(socket.socket, "connect", network_denied)
    monkeypatch.setattr(socket, "getaddrinfo", network_denied)
    format = "docx" if "docx" in format_name else "html" if format_name.startswith("html") else "eml"
    path = tmp_path / f"saved.{format}"
    markup = _html(CASES)
    if format == "docx":
        path.write_bytes(_word(CASES, strict=format_name == "strict_docx"))
    elif format == "eml":
        path.write_bytes(_email(markup))
        markup = (
            BytesParser(policy=policy.default)
            .parsebytes(path.read_bytes())
            .get_body(preferencelist=("plain", "html"))
            .get_content()
        )
    else:
        path.write_text(markup, encoding="utf-16" if format_name == "html_utf16" else "utf-8")
    before = path.read_bytes()
    decoded = extract(path, ".docx") if format == "docx" else None
    report = inspect(path, "." + format)
    assert path.read_bytes() == before
    assert (
        report["status"] == "checked"
        and report["destinations_contacted"] == 0
        and report["authorship_confidence_contribution"] == 0
    )
    assert report["counts"] == dict(Counter(case[3] for case in CASES)), report
    expected = [(i, case) for i, case in enumerate(CASES) if case[3] == "different_host"]
    assert len(report["findings"]) == len(expected) == 5
    for row, (i, case) in zip(report["findings"], expected, strict=True):
        raw, visible, saved = case[4:]
        where = row["location"]
        assert (row["label_host"], row["saved_host"]) == (visible, saved)
        assert set(row) == {"label_host", "saved_host", "location"}
        assert all(type(a) is int and b > a for a, b in where["ranges"])
        if format == "docx":
            with ZipFile(path) as document:
                xml = etree.fromstring(document.read("word/document.xml"))
            ns = xml.tag.split("}")[0] + "}"
            p = xml.find(ns + "body")[i]
            unified = "".join(p.itertext())
            assert where["body_path"] == [i] and where["kind"] == "word_paragraph"
            (block,) = [a for a in decoded.source_map if a["body_path"] == [i]]
            assert block["kind"] == "docx" and block["run_mapping_complete"] is True
            assert "".join(unified[a:b] for a, b in where["ranges"]) == raw
        else:
            assert where["kind"] == ("decoded_email_html" if format == "eml" else "decoded_html")
            if format == "eml":
                assert where["mime_part"] == "1.1"
            assert "".join(unescape(markup[a:b]) for a, b in where["ranges"]) == raw
            if i == 0:
                assert len(where["ranges"]) == 3
    result = json.dumps(report)
    assert "example-label-token" not in result and "example-target-token" not in result
    assert "campaign.example.net" in result and "safelinks.protection" not in result


def test_ambiguous_and_hidden_markup_is_not_reported(tmp_path):
    path = tmp_path / "controls.html"
    path.write_text("""<html><head><a href="https://target.example.net">https://head.example.org</a></head>
    <body><p hidden><a href="https://target.example.net">https://hidden.example.org</a></p>
    <p style="display:none!important"><a href="https://target.example.net">https://other.example.org</a></p>
    <p><a href="https://target.example.net" href="https://branch.example.net">https://dup.example.org</a></p>
    <p><a href="https://target.example.net">https://overlap.example.org<a href="https://branch.example.net">nested</a></a></p>
    <blockquote><a href="https://target.example.net">https://quote.example.org</a></blockquote>
    <p><a href="https://target.example.net">https://shape.<br>example.org</a></p>
    <p><a href="https://target.example.net"><em>https://unclosed.example.org</a></p>
    <p><a href="https://target.example.net">https://pending.example.org</body></html>""")
    assert inspect(path, ".html")["findings"] == []


def test_selected_plain_email_body_excludes_alternative_and_attachment(tmp_path):
    path = tmp_path / "controls.eml"
    path.write_bytes(_email(_html(CASES), plain=True))
    result = inspect(path, ".eml")
    assert result["status"] == "outside_scope" and result["findings"] == []


def test_unreadable_archive_has_no_word_evidence(tmp_path):
    path = tmp_path / "broken.docx"
    path.write_bytes(b"not an Office package")
    record = inspect(path, ".docx")
    assert record["status"] == "outside_scope" and not record["findings"]


def test_body_path_preserves_blocks_before_link(tmp_path):
    path = tmp_path / "with-table.docx"
    word = Document()
    word.add_paragraph("Ordinary heading before a table.")
    word.add_table(1, 1).cell(0, 0).text = "An ordinary native table cell."
    paragraph = word.add_paragraph(PREFIX)
    _hyperlink(paragraph, "www.example.com/archive", "https://publisher.example.net/registry")
    word.save(path)
    (row,) = inspect(path, ".docx")["findings"]
    assert row["location"] == {
        "kind": "word_paragraph",
        "body_path": [2],
        "ranges": [[len(PREFIX), len(PREFIX) + 15]],
    }
    decoded = extract(path, ".docx")
    assert [2] in [part["body_path"] for part in decoded.source_map]
