# SPDX-License-Identifier: Apache-2.0
"""Inspect stored link labels and destinations offline. No URLs are followed."""

from __future__ import annotations

import codecs
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from email import policy
from email.parser import BytesParser
from html import unescape
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from app.detectors.text import email_body, html_content
from app.evidence import document_saved_link_hosts_binding as binding

METHODS = {".html": "html", ".htm": "html", ".xhtml": "html", ".eml": "email", ".docx": "python-docx"}
READ_BUDGET_S = 4.0
_SAFE_LINKS = ".safelinks.protection.outlook.com"
_URL = re.compile(
    r"(?P<scheme>https?://)?(?P<host>[^\s/:@%?#\\]+)(?::(?P<port>[0-9]{1,5}))?(?P<tail>[/#?][^\s\\]*)?\Z",
    re.IGNORECASE,
)
_VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_INLINE = frozenset(
    {
        "b",
        "bdi",
        "bdo",
        "cite",
        "code",
        "em",
        "i",
        "mark",
        "s",
        "small",
        "span",
        "strong",
        "u",
        "var",
        "wbr",
    }
)
_HIDE = frozenset(
    {
        "head",
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "math",
        "iframe",
        "object",
        "audio",
        "video",
        "canvas",
        "dialog",
        "details",
        "blockquote",
    }
)
_VISIBILITY = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse))\s*(?:!important\s*)?(?:;|$)",
    re.IGNORECASE,
)


def _host(value: str, *, label: bool) -> tuple[str, int, int] | None:
    import idna

    left = len(value) - len(value.lstrip(" \t\n\r"))
    value = value.strip(" \t\n\r")
    if not 0 < len(value) <= binding.MAX_URL or any(not character.isprintable() for character in value):
        return None
    match = _URL.fullmatch(value)
    if match is None or (
        not match["scheme"] and (not label or not match["host"].casefold().startswith("www."))
    ):
        return None
    if match["port"] and not 1 <= int(match["port"]) <= 65535:
        return None
    try:
        hostname = (
            idna.encode(match["host"], uts46=True, transitional=False, std3_rules=True)
            .decode("ascii")
            .removesuffix(".")
        )
    except (ValueError, idna.IDNAError):
        return None
    pieces = hostname.split(".")
    if (
        len(pieces) < 2
        or not any(c.isalpha() for c in pieces[-1])
        or pieces[-1] in {"local", "localhost", "internal"}
    ):
        return None
    start, stop = match.span("host")
    return hostname, left + start, left + stop


def _classify(label: str, target: str) -> tuple[str, str | None, str | None, tuple[int, int] | None]:
    visible = _host(label, label=True)
    if visible is None:
        return "unsupported_label", None, None, None
    destination = _host(target, label=False)
    if destination is None:
        return "unsupported_target", None, None, None
    source_host, start, stop = visible
    saved_host = destination[0]
    if source_host == saved_host:
        return "same_host", None, None, None
    if saved_host.endswith(_SAFE_LINKS) and len(saved_host.split(".")) == 5:
        return "registered_wrapper", None, None, None
    if binding.related_root(source_host) == binding.related_root(saved_host):
        return "related_hostname", None, None, None
    return "different_host", source_host, saved_host, (start, stop)


def _hidden(tag, attrs):
    return (
        tag in _HIDE
        or "hidden" in attrs
        or "inert" in attrs
        or (attrs.get("aria-hidden") or "").strip().casefold() == "true"
        or _VISIBILITY.search(attrs.get("style") or "")
    )


class _Links(HTMLParser):
    """Record static visible link labels and their decoded HTML source positions."""

    def __init__(self, source: str):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = [0] + [match.end() for match in re.finditer("\n", source)]
        self.stack = []
        self.links = []
        self.active = None

    def _location(self):
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        hidden = (self.stack and self.stack[-1][1]) or _hidden(tag, values)
        if self.active is not None and (hidden or tag not in _INLINE):
            self.active["invalid"] = True
        if tag == "a" and self.active is None and not hidden:
            targets = [target for name, target in attrs if name == "href"]
            if len(targets) == 1 and isinstance(targets[0], str):
                self.active = {
                    "depth": len(self.stack),
                    "target": targets[0],
                    "chunks": [],
                    "origins": [],
                    "invalid": False,
                }
        if tag not in _VOID:
            if len(self.stack) >= 256:
                raise ValueError("HTML nesting exceeds the local bound")
            self.stack.append((tag, hidden))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        position = next(
            (i for i in reversed(range(len(self.stack))) if self.stack[i][0] == tag),
            None,
        )
        if position is None:
            return
        if self.active is not None and position <= self.active["depth"]:
            if (
                position == self.active["depth"] == len(self.stack) - 1
                and tag == "a"
                and not self.active["invalid"]
            ):
                if len(self.links) == binding.MAX_LINKS:
                    raise ValueError("HTML link count exceeds the local bound")
                self.links.append(self.active)
            self.active = None
        del self.stack[position:]

    def _append(self, value, locations):
        if self.active is not None and not self.active["invalid"]:
            self.active["chunks"].append(value)
            self.active["origins"].extend(locations)
            if len(self.active["origins"]) > binding.MAX_URL:
                self.active["invalid"] = True

    def handle_data(self, value):
        if self.active is not None and not self.active["invalid"]:
            start = self._location()
            self._append(value, ((start + i, start + i + 1) for i in range(len(value))))

    def _entity(self, literal):
        if self.active is not None and not self.active["invalid"]:
            start, value = self._location(), unescape(literal)
            if len(value) != 1 or self.source[start : start + len(literal)] != literal:
                self.active["invalid"] = True
            else:
                self._append(value, [(start, start + len(literal))])

    def handle_entityref(self, name):
        self._entity(f"&{name};")

    def handle_charref(self, name):
        self._entity(f"&#{name};")


def _ranges(locations):
    output = []
    for a, b in locations:
        if output and output[-1][1] == a:
            output[-1][1] = b
        else:
            output.append([a, b])
    return output


def _result(method, counts=None, findings=(), *, status="checked"):
    recorded = dict(counts or {})
    return {
        "schema": binding.SCHEMA,
        "status": status,
        "method": method,
        "offset_unit": "unicode_codepoint",
        "counts": recorded,
        "findings": list(findings),
        "omitted": max(0, recorded.get("different_host", 0) - len(findings)),
        "authorship_confidence_contribution": 0,
        "destinations_contacted": 0,
    }


def _consider(counts, findings, label, target, locate):
    status, visible, saved, location = _classify(label, target)
    counts[status] += 1
    if status == "different_host" and len(findings) < binding.MAX_FINDINGS:
        findings.append({"label_host": visible, "saved_host": saved, "location": locate(*location)})


def _inspect_html(raw: str, method: str, part: str | None = None):
    if len(raw) > binding.MAX_HTML_CHARS or any(marker in raw for marker in ("\ufffd", "\x00")):
        return _result(method, status="outside_scope")
    parser = _Links(raw)
    try:
        parser.feed(raw)
        parser.close()
    except ValueError:
        return _result(method, status="outside_scope")
    counts, findings = Counter(), []
    for item in parser.links:

        def location(a, b, origins=item["origins"]):
            result = {
                "kind": "decoded_email_html" if part else "decoded_html",
                "ranges": _ranges(origins[a:b]),
            }
            if part:
                result["mime_part"] = part
            return result

        _consider(counts, findings, "".join(item["chunks"]), item["target"], location)
    return _result(method, counts, findings)


def _email_body(data: bytes, *, use_current_email_body: bool, use_static_html: bool):
    message = BytesParser(policy=policy.default).parsebytes(data)

    def find(node, path="1", depth=0):
        if node is body:
            return path
        if depth < 24 and node.is_multipart():
            for i, child in enumerate(node.iter_parts(), 1):
                if found := find(child, f"{path}.{i}", depth + 1):
                    return found
        return None

    if use_current_email_body:
        part, body, _ = email_body.select(message)
    else:
        body = message.get_body(preferencelist=("plain", "html"))
        part = find(message)
    if body is None or body.get_content_type() != "text/html" or part is None or part.count(".") > 24:
        return _result("email", status="outside_scope")
    raw = body.get_content()
    if not isinstance(raw, str):
        return _result("email", status="outside_scope")
    return _inspect_html(html_content.mask(raw) if use_static_html else raw, "email", part)


def _word_relationships(path, name):
    from app.detectors.text.strict_word import (
        RELATIONS,
        RELATIONSHIP,
        TRANSITIONAL_RELATIONSHIP,
        _read_xml,
    )

    main = PurePosixPath(name)
    entry = str(main.parent / "_rels" / (main.name + ".rels"))
    with ZipFile(path) as archive:
        if entry not in archive.namelist():
            return {}
        root = _read_xml(archive, entry, 512 * 1024)
    if root.tag != RELATIONS + "Relationships":
        raise ValueError("Unexpected Word relationships element")
    rows = {}
    for item in root.findall(RELATIONS + "Relationship"):
        key = item.get("Id")
        if not key or key in rows:
            raise ValueError("Ambiguous Word relationship identifier")
        rows[key] = (
            item.get("Target", "")
            if (
                item.get("Type") in {ns + "hyperlink" for ns in (RELATIONSHIP, TRANSITIONAL_RELATIONSHIP)}
                and item.get("TargetMode") == "External"
            )
            else ""
        )
    return rows


def _word_text(path):
    from docx.oxml.ns import qn
    from docx.text.paragraph import Paragraph
    from docx.text.run import Run

    from app.detectors.text import word_saved_fields as visibility
    from app.detectors.text.extract import docx_body
    from app.storage.files import check_docx_safe

    check_docx_safe(path)
    document, name, revisions, _ = docx_body(
        path, project_word_revisions=False, include_word_content_controls=False
    )
    if (
        not revisions
        or revisions.get("status") != "no_match"
        or not visibility.defaults_allow_visible_runs(document)
    ):
        return _result("python-docx", status="outside_scope")
    targets = _word_relationships(path, name)
    counts, findings = Counter(), []
    run, hyperlink, word_text = qn("w:r"), qn("w:hyperlink"), qn("w:t")
    accepted = {run, hyperlink, qn("w:pPr"), qn("w:bookmarkStart"), qn("w:bookmarkEnd")}
    formatting = qn("w:rPr")
    complex_fields = visibility.complex_field_blocks(document.element.body)
    for index, element in enumerate(document.element.body):
        if element.tag != qn("w:p") or element in complex_fields or not element.findall(hyperlink):
            continue
        paragraph = Paragraph(element, document._body)
        if (
            any(child.tag not in accepted for child in element)
            or not visibility.paragraph_allows_visible_runs(paragraph)
            or paragraph.text != "".join(child.text for child in element if child.tag in {run, hyperlink})
        ):
            continue
        cursor = 0
        for child in element:
            if child.tag not in {run, hyperlink}:
                continue
            value = child.text
            if child.tag == hyperlink and all(
                item.tag == run
                and all(node.tag in {formatting, word_text} for node in item)
                and not item.xpath("./w:rPr/w:vanish | ./w:rPr/w:specVanish | ./w:rPr/w:webHidden")
                and visibility.style_allows_visible_runs(Run(item, paragraph).style)
                for item in child
            ):
                if sum(counts.values()) >= binding.MAX_LINKS:
                    return _result("python-docx", status="outside_scope")
                _consider(
                    counts,
                    findings,
                    value,
                    targets.get(child.get(qn("r:id")), ""),
                    lambda a, b, index=index, cursor=cursor: {
                        "kind": "word_paragraph",
                        "body_path": [index],
                        "ranges": [[cursor + a, cursor + b]],
                    },
                )
            cursor += len(value)
    return _result("python-docx", counts, findings)


def inspect(path: Path, extension: str, *, use_current_email_body: bool = True, use_static_html: bool = True):
    """Read only a bounded supported source and report saved hostname differences."""
    ext = extension.casefold()
    method = METHODS.get(ext)
    if method is None or not 0 < path.stat().st_size <= binding.MAX_FILE_BYTES:
        return _result(method, status="outside_scope")
    if ext == ".docx":
        from docx.opc.exceptions import OpcError
        from lxml.etree import XMLSyntaxError

        try:
            return _word_text(path)
        except (BadZipFile, KeyError, OpcError, ValueError, XMLSyntaxError):
            return _result(method, status="outside_scope")
    if ext != ".eml" and path.stat().st_size > binding.MAX_HTML_CHARS * 4 + 4:
        return _result(method, status="outside_scope")
    data = path.read_bytes()
    try:
        if ext == ".eml":
            return _email_body(data, use_current_email_body=use_current_email_body, use_static_html=use_static_html)
        encoding = "utf-16" if data.startswith((codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)) else "utf-8-sig"
        raw = data.decode(encoding)
        return _inspect_html(html_content.mask(raw, xhtml=ext == ".xhtml") if use_static_html else raw, "html")
    except (LookupError, ValueError, RecursionError):
        return _result(method, status="outside_scope")


def inspect_within(path: Path, *, timeout: float, source_sha256: str):
    """Keep optional original-file parsing outside the analysis worker."""
    try:
        result = subprocess.run(  # noqa: S603 - fixed module and server-staged document path
            [sys.executable, "-I", "-m", "app.evidence.document_saved_link_hosts", str(path), source_sha256],
            capture_output=True, check=True, timeout=timeout,
        )
        record = json.loads(result.stdout)
        return record if isinstance(record, dict) and record.get("findings") else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def verify_original(document, original, extension, *, use_current_email_body: bool = True, use_static_html: bool = True):
    if binding.KEY in document or binding.COMMITMENT in document["source"]:
        result = inspect(original, extension, use_current_email_body=use_current_email_body, use_static_html=use_static_html)
        binding.verify_record(document, result, commitment=document["source"].get(binding.COMMITMENT))


def _file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bound_inspection(original, source_sha256):
    if not 0 < original.stat().st_size <= binding.MAX_FILE_BYTES or _file_sha256(original) != source_sha256:
        return None
    record = inspect(original, original.suffix)
    return record if _file_sha256(original) == source_sha256 else None


if __name__ == "__main__":
    original = Path(sys.argv[1])
    source = sys.argv[2] if len(sys.argv) == 3 else _file_sha256(original)
    print(json.dumps(_bound_inspection(original, source)))
