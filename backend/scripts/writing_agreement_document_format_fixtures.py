# SPDX-License-Identifier: Apache-2.0
"""Create controlled native containers around an unchanged writing excerpt."""

from __future__ import annotations

import io
from email import policy
from email.message import EmailMessage
from html import escape
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import docx
import pymupdf


def _docx(path: Path, paragraphs: list[str]) -> None:
    document = docx.Document()
    for body in paragraphs:
        paragraph = document.add_paragraph()
        words = body.split(" ")
        for index in range(0, len(words), 7):
            start = " " if index else ""
            run = paragraph.add_run(start + " ".join(words[index:index + 7]))
            run.bold = index % 14 == 7
    buffer = io.BytesIO()
    document.save(buffer)
    with ZipFile(buffer) as source, ZipFile(path, "w") as output:
        for name in sorted(source.namelist()):
            output.writestr(ZipInfo(name), source.read(name), compress_type=ZIP_DEFLATED)


def _pdf(path: Path, paragraphs: list[str], font: Path) -> None:
    with pymupdf.open() as document:
        pages = [paragraphs[:2], paragraphs[2:]] if len(paragraphs) == 4 else [paragraphs]
        for group in pages:
            page = document.new_page(width=612, height=792)
            page.insert_font(fontname="BodyText", fontfile=font)
            top = 55.
            for body in group:
                bounds = pymupdf.Rect(48, top, 558, 725)
                unused = page.insert_textbox(bounds, body, fontname="BodyText", fontsize=10, lineheight=1.35)
                if unused < 0:
                    raise ValueError("The held paragraph does not fit the native fixture page.")
                top = bounds.y1 - unused + 28
        document.subset_fonts()
        document.save(path, garbage=4, deflate=True, no_new_id=True)


def _email(path: Path, body: str, *, quote: bool = False) -> None:
    message = EmailMessage(policy=policy.SMTP)
    message["From"] = "reviewer@example.invalid"
    message["To"] = "recipient@example.invalid"
    message["Subject"] = "Document format parity fixture"
    if quote:
        body += "\n\nOn a prior date someone wrote:\n> This is a synthetic quotation from a different message."
    message.set_content(body, charset="utf-8", cte="quoted-printable")
    path.write_bytes(message.as_bytes())


def build(folder: Path, text: str, font: Path) -> dict[str, Path]:
    folder.mkdir()
    paragraphs = text.split("\n\n")
    paths = {name: folder / f"excerpt.{name}" for name in ("txt", "docx", "pdf", "eml")}
    paths["txt"].write_bytes(text.encode("utf-8"))
    _docx(paths["docx"], paragraphs)
    _pdf(paths["pdf"], paragraphs, font)
    _email(paths["eml"], text)
    return paths


def controls(folder: Path, source_pdf: Path, text: str) -> tuple[Path, Path, Path]:
    quote = folder / "quoted.eml"
    _email(quote, text, quote=True)
    html = folder / "html.eml"
    message = EmailMessage(policy=policy.SMTP)
    message["Subject"] = "HTML parity control"
    message.set_content("".join(f"<p>{escape(part)}</p>" for part in text.split("\n\n")), subtype="html", charset="utf-8")
    html.write_bytes(message.as_bytes())
    raster = folder / "image-only.pdf"
    with pymupdf.open(source_pdf) as source, pymupdf.open() as image_only:
        for original in source:
            page = image_only.new_page(width=original.rect.width, height=original.rect.height)
            page.insert_image(page.rect, stream=original.get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).tobytes("png"))
        image_only.save(raster, garbage=4, deflate=True, no_new_id=True)
    return quote, html, raster
