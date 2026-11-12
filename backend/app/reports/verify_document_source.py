"""Compare signed document evidence with a local original without scoring it."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from app.core.canonical import canonical_hash
from app.detectors.text.extract import extract, text_hash
from app.evidence import (
    document_context,
    document_matched_context,
    document_paragraph_priority,
    document_passage_coverage,
    document_repeat,
    document_scope,
)
from app.evidence.document_packets import (
    _inspection_anchors,
    _pdf_plates,
    _range,
    _source_anchors,
    verify_context_window,
)
from app.reports.public_verify import SCHEMA
from app.storage.files import check_docx_safe

EXTENSIONS = {"plain": ".txt", "python-docx": ".docx", "email": ".eml", "pymupdf": ".pdf"}
MAX_SOURCE_BYTES = 25 * 1024 * 1024


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pdf_images(archive: ZipFile, packet: dict, source: bytes) -> int:
    expected = _pdf_plates(source, packet["anchors"])
    artifacts = packet["artifacts"]
    if expected.keys() != artifacts.keys():
        raise ValueError(f"{packet['id']} PDF images differ from the source pages and regions")
    for key, (image, metadata) in expected.items():
        asset = artifacts[key]
        if any(canonical_hash(asset.get(field)) != canonical_hash(value)
               for field, value in metadata.items() if field != "renderer"):
            raise ValueError(f"{packet['id']} PDF image geometry differs from the source page")
        with archive.open(f"assets/{asset['sha256']}.png") as stream, Image.open(stream) as recorded:
            if recorded.format != "PNG" or recorded.size != image.size or recorded.convert("RGB").tobytes() != image.tobytes():
                raise ValueError(f"{packet['id']} PDF image pixels differ from the rendered source page")
    return len(expected)


def check(archive: ZipFile, bundle: dict, original: Path) -> str:
    if bundle.get("schema") != SCHEMA or bundle.get("analysis", {}).get("kind") != "document":
        raise ValueError("Original-document verification requires a public document evidence package")
    commitment = bundle["source_commitment"]
    size = original.stat().st_size
    if not 0 < size <= MAX_SOURCE_BYTES:
        raise ValueError("The supplied document exceeds the local 25 MiB limit or is empty")
    if size != commitment["bytes"] or _sha(original) != commitment["sha256"]:
        raise ValueError("The supplied document size or SHA-256 differs from the committed original")

    evidence = json.loads(archive.read("evidence.json"))
    source, coverage = evidence["source"], evidence.get("coverage") or {}
    extension = EXTENSIONS.get(source.get("method"))
    if extension is None:
        raise ValueError("Original-document verification supports plain text, DOCX, EML and native PDF text")
    if any(row.get("status") == "extracted" for row in (coverage.get("ocr") or {}).get("pages", [])):
        raise ValueError("Original-document replay supports native PDF text only. This package contains OCR evidence.")
    if extension == ".docx":
        from docx.opc.exceptions import OpcError
        from lxml.etree import XMLSyntaxError

        check_docx_safe(original)
        try:
            doc = extract(original, extension, allow_ocr=False)
        except (OpcError, XMLSyntaxError) as error:
            raise ValueError("The supplied DOCX cannot be parsed locally") from error
    else:
        doc = extract(original, extension, allow_ocr=False)
    if extension == ".eml" and doc.meta["email"]["body_content_type"] != "text/plain":
        raise ValueError("Original-document replay requires a plain-text EML body. HTML positions refer to converted text.")
    text, mapping = doc.text, doc.source_map
    if (doc.meta["original_sha256"] != source["sha256"] or len(text) != source["chars"]
            or doc.meta.get("method") != source["method"] or text_hash(text) != source["extracted_text_sha256"]
            or canonical_hash(mapping) != source["source_map_sha256"]):
        raise ValueError("Re-extracted text or source mapping differs from the committed extraction")
    fields = {"email": None, "ocr": None, "unmapped_pages": [], "partially_checked_pages": []}
    if any(canonical_hash(coverage.get(key, default)) != canonical_hash(doc.meta.get(key, default))
           for key, default in fields.items()):
        raise ValueError("Recorded document coverage differs from the re-extracted original")
    document_repeat.verify_source(evidence, text, mapping)
    document_matched_context.verify_source(evidence, text, mapping)
    document_paragraph_priority.verify_source(evidence, text, mapping)
    document_passage_coverage.verify_source(evidence, text, mapping)
    document_scope.validate_checkpoint_agreement(evidence, text)

    pdf_bytes = original.read_bytes() if extension == ".pdf" else None
    packet_count = window_count = images = 0
    for packet in evidence["packets"]:
        for key in ("focus", "context"):
            a, b = _range(packet[key], len(text))
            if packet[key]["text"] != text[a:b]:
                raise ValueError(f"{packet['id']} {key} excerpt differs from its exact source range")
        focus = packet["focus"]
        if evidence["version"] in document_scope.SCOPE_VERSIONS:
            document_scope.validate(packet, len(text))
            expected = _inspection_anchors(mapping, packet["inspection_scope"], len(text))
        else:
            expected = _source_anchors(mapping, focus["start"], focus["end"], len(text))
        if canonical_hash(packet["anchors"]) != canonical_hash(expected):
            raise ValueError(f"{packet['id']} source positions differ from the original extraction")
        for window in packet["supporting_windows"]:
            verify_context_window(window, text)
        if "context_comparison" in packet:
            document_context.validate(packet["context_comparison"], focus=focus, chars=len(text), text=text)
        if pdf_bytes is not None:
            images += _pdf_images(archive, packet, pdf_bytes)
        packet_count += 1
        window_count += len(packet["supporting_windows"])
    if _sha(original) != commitment["sha256"]:
        raise ValueError("The supplied original changed during document replay")
    return (f"Original SHA-256, extracted text and source mapping matched. Verified passages ({packet_count}), "
            f"recorded input windows ({window_count}) and native PDF images ({images}).")
