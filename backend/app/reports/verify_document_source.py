"""Compare signed document evidence with a local original without scoring it."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable
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
    document_passage_position,
    document_repeat,
    document_scope,
    document_unselected_responses,
    email_alternatives,
)
from app.evidence.document_packets import (
    DocumentPacketIntegrityError,
    _inspection_anchors,
    _pdf_plates,
    _range,
    _source_anchors,
    public_ocr_coverage,
    verify_context_window,
)
from app.reports.public_verify import SCHEMA
from app.storage.files import check_docx_safe

EXTENSIONS = {"plain": ".txt", "python-docx": ".docx", "email": ".eml", "pymupdf": ".pdf"}
MAX_SOURCE_BYTES = 25 * 1024 * 1024


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pdf_images(archive: ZipFile, packet: dict, source: bytes,
                read_ocr_raster: Callable[[str], bytes] | None) -> int:
    try:
        expected = _pdf_plates(source, packet["anchors"], read_ocr_raster=read_ocr_raster)
    except DocumentPacketIntegrityError as error:
        raise ValueError(str(error)) from error
    artifacts = packet["artifacts"]
    if expected.keys() != artifacts.keys():
        raise ValueError(f"{packet['id']} PDF images differ from the source pages and regions")
    for key, (image, metadata) in expected.items():
        asset = artifacts[key]
        exact = metadata.pop("_exact_png", None)
        if any(canonical_hash(asset.get(field)) != canonical_hash(value)
               for field, value in metadata.items() if field != "renderer"):
            raise ValueError(f"{packet['id']} PDF image geometry differs from the source page")
        with archive.open(f"assets/{asset['sha256']}.png") as stream:
            if exact is not None and stream.read(len(exact) + 1) != exact:
                raise ValueError(f"{packet['id']} OCR input raster differs from the original scan")
            stream.seek(0)
            with Image.open(stream) as recorded:
                if (recorded.format != "PNG" or recorded.size != image.size
                        or recorded.convert("RGB").tobytes() != image.tobytes()):
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
        raise ValueError("Original-document verification supports plain text, DOCX, EML and PDF")
    ocr_pages = (coverage.get("ocr") or {}).get("pages", [])
    page_coverage = (coverage.get(document_passage_coverage.KEY) or {}).get(document_scope.PDF_PAGE_COVERAGE_KEY)
    used_ocr = extension == ".pdf" and any(row.get("status") == "extracted" for row in ocr_pages)
    replay_ocr = used_ocr or (extension == ".pdf" and page_coverage is not None and bool(ocr_pages))
    if extension == ".docx":
        from docx.opc.exceptions import OpcError
        from lxml.etree import XMLSyntaxError

        check_docx_safe(original)
        try:
            doc = extract(original, extension, allow_ocr=False)
        except (OpcError, XMLSyntaxError) as error:
            raise ValueError("The supplied DOCX cannot be parsed locally") from error
    else:
        doc = extract(original, extension, allow_ocr=replay_ocr)
    if used_ocr and not doc.source_rasters:
        raise ValueError("The local OCR runtime could not reproduce the original scan. Check that the verified OCR data are installed.")
    if extension == ".eml" and doc.meta["email"]["body_content_type"] != "text/plain" and not (
        document_scope.mapped_email_html(doc.meta["email"])
    ):
        raise ValueError("Original-document replay requires a plain-text or source-mapped HTML EML body.")
    text, mapping = doc.text, doc.source_map
    if (doc.meta["original_sha256"] != source["sha256"] or len(text) != source["chars"]
            or doc.meta.get("method") != source["method"] or text_hash(text) != source["extracted_text_sha256"]
            or canonical_hash(mapping) != source["source_map_sha256"]):
        raise ValueError("Re-extracted text or source mapping differs from the committed extraction")
    fields = {"email": None, "ocr": None, "unmapped_pages": [], "partially_checked_pages": []}
    local_coverage = {**doc.meta, "ocr": public_ocr_coverage(doc.meta)}
    if any(canonical_hash(coverage.get(key, default)) != canonical_hash(local_coverage.get(key, default))
           for key, default in fields.items()):
        raise ValueError("Recorded document coverage differs from the re-extracted original")
    document_repeat.verify_source(evidence, text, mapping)
    document_matched_context.verify_source(evidence, text, mapping)
    document_paragraph_priority.verify_source(evidence, text, mapping)
    document_passage_coverage.verify_source(evidence, text, mapping, meta=doc.meta)
    document_scope.validate_checkpoint_agreement(evidence, text, mapping=mapping)
    unselected = document_unselected_responses.verify_source(evidence, text, mapping)
    position = document_passage_position.verify_source(evidence, text, mapping)
    alternatives = email_alternatives.verify_source(evidence, original.read_bytes()) if email_alternatives.KEY in coverage else None

    pdf_bytes = original.read_bytes() if extension == ".pdf" else None

    def read_ocr_raster(digest: str) -> bytes:
        encoded = doc.source_rasters.get(digest)
        if encoded is None:
            raise ValueError("The OCR input raster differs from the re-extracted original")
        data = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("The re-extracted OCR raster does not match its recorded hash")
        return data

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
            images += _pdf_images(archive, packet, pdf_bytes, read_ocr_raster if replay_ocr else None)
        packet_count += 1
        window_count += len(packet["supporting_windows"])
    if _sha(original) != commitment["sha256"]:
        raise ValueError("The supplied original changed during document replay")
    image_label = "PDF images, including OCR source rasters" if used_ocr else "native PDF images"
    page_detail = f" Verified mapped text coverage across {page_coverage['page_count']} PDF pages." if page_coverage else ""
    unselected_detail = f" Verified {unselected} additional recorded response locations and inputs." if unselected else ""
    alternative_detail = " Verified both decoded email body renditions and reported source positions." if alternatives and "parts" in alternatives else ""
    position_detail = " Verified all three writing position inputs and their source ranges." if position else ""
    return (f"Original SHA-256, extracted text and source mapping matched. Verified passages ({packet_count}), "
            f"recorded input windows ({window_count}) and {image_label} ({images})."
            f"{page_detail}{unselected_detail}{alternative_detail}{position_detail}")
