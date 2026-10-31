"""Document text extraction preserving paragraph boundaries and character offsets (PDF, DOCX, TXT, MD, HTML).

This file imports nothing of the application: `pipeline.extract_within` runs it as a script in a child process
(`python -I extract.py <path> <ext>`), which prints the document, or the error, as one JSON object (N-10; SEC-04)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path


@dataclass
class Document:
    text: str
    paragraphs: list[tuple[int, int]]  # (start, end) char offsets into text
    meta: dict = field(default_factory=dict)


_WS = re.compile(r"[ \t ]+")
# Every pattern is linear in the length of the file (N-10; SEC-04). A bracket class that could run to the end of the file
# from each `[`, or `\s*` that could run over every blank line from each line start, made three of them quadratic: 40,000
# characters of `[` took 2.6 s and 200,000 took 69 s, inside one call that holds the interpreter. The text of a link or an
# image cannot hold `[`, and its address cannot either, so the scan from one opener ends at the next; the rule and table
# patterns stay on their line.
_MD_INLINE = [
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"!\[[^\[\]]*\]\([^)\[]*\)"), " "),
    (re.compile(r"\[([^\[\]]+)\]\([^)\[]*\)"), r"\1"),
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.MULTILINE), ""),
    (re.compile(r"^[ \t]{0,3}(?:[-*+]|\d+[.)])[ \t]+", re.MULTILINE), ""),
    (re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE), ""),
    (re.compile(r"(\*\*|__)(.*?)\1"), r"\2"),
    (re.compile(r"(\*|_)(.*?)\1"), r"\2"),
    (re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$", re.MULTILINE), ""),
    (re.compile(r"^\|.*\|[ \t]*$", re.MULTILINE), lambda m: m.group(0).replace("|", " ")),
]


def _join_paragraphs(paras: list[str]) -> Document:
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for p in paras:
        p = _WS.sub(" ", p).strip()
        if not p:
            continue
        if text_parts:
            text_parts.append("\n\n")
            pos += 2
        text_parts.append(p)
        spans.append((pos, pos + len(p)))
        pos += len(p)
    return Document("".join(text_parts), spans)


def _dehyphenate(s: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", s)


def extract_pdf(path: Path) -> Document:
    import pymupdf

    paras: list[str] = []
    pages = 0
    with pymupdf.open(path) as doc:
        for page in doc:
            pages += 1
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (round(b[1] / 5), b[0]))
            for b in blocks:
                if b[6] != 0:  # image block
                    continue
                t = _dehyphenate(b[4]).replace("\n", " ").strip()
                if len(t) >= 2:
                    paras.append(t)
    d = _join_paragraphs(paras)
    d.meta = {"pages": pages, "method": "pymupdf"}
    if len(d.text.split()) < 5 and pages:
        d.meta["warning"] = "The PDF contains little or no extractable text (scanned image?). OCR is not performed."
    return d


def extract_docx(path: Path) -> Document:
    import docx

    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                paras.append(" | ".join(c for c in cells if c))
    doc = _join_paragraphs(paras)
    doc.meta = {"method": "python-docx", "paragraph_count": len(d.paragraphs)}
    return doc


class _HTMLText(HTMLParser):
    BLOCK = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article", "blockquote", "pre"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def extract_text_like(path: Path, ext: str) -> Document:
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        raw = data.decode("utf-16", errors="replace")
    elif data.startswith(b"\xef\xbb\xbf"):
        raw = data.decode("utf-8-sig", errors="replace")
    else:
        raw = data.decode("utf-8", errors="replace")
    if ext in (".html", ".htm"):
        h = _HTMLText()
        h.feed(raw)
        raw = "".join(h.parts)
        method = "html"
    elif ext in (".md", ".markdown"):
        for rx, rep in _MD_INLINE:
            raw = rx.sub(rep, raw)
        method = "markdown"
    else:
        method = "plain"
    paras = re.split(r"\n\s*\n", raw.replace("\r\n", "\n"))
    paras = [p.replace("\n", " ") for p in paras]
    d = _join_paragraphs(paras)
    d.meta = {"method": method}
    return d


def extract(path: Path, ext: str) -> Document:
    ext = ext.lower()
    if ext == ".pdf":
        d = extract_pdf(path)
    elif ext == ".docx":
        d = extract_docx(path)
    else:
        d = extract_text_like(path, ext)
    d.meta.update({"chars": len(d.text), "words": len(d.text.split()), "paragraphs": len(d.paragraphs)})
    return d


def from_string(text: str) -> Document:
    paras = [p.replace("\n", " ") for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n"))]
    d = _join_paragraphs(paras)
    d.meta = {"method": "string", "chars": len(d.text), "words": len(d.text.split()), "paragraphs": len(d.paragraphs)}
    return d


def _child(argv: list[str]) -> int:
    """`python -I extract.py <path> <ext>`: the document as JSON on stdout, or the error it raised (its class name, its text
    and whether it is a `ValueError`, which the application shows to the person as it is)."""
    import json
    import sys

    try:
        d = extract(Path(argv[1]), argv[2])
        out: dict = {"text": d.text, "paragraphs": d.paragraphs, "meta": d.meta}
    except Exception as e:  # noqa: BLE001 - handed to the parent, which raises it again
        out = {"error": {"type": type(e).__name__, "message": str(e), "value_error": isinstance(e, ValueError)}}
    sys.stdout.write(json.dumps(out))  # ASCII only: a lone surrogate of a damaged file crosses as an escape
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_child(sys.argv))
