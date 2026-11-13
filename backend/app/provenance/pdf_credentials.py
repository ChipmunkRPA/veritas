"""Qualify a PDF Content Credentials store for whole-file attribution."""

from __future__ import annotations

import io
import json
import re
import subprocess
import sys

from app.provenance import c2pa as verifier

MIME = "application/pdf"
MAX_STORE = 1024 * 1024
MAX_XREF = 100_000
REFERENCE = r"([1-9][0-9]{0,8}) 0 R"
CATALOG_ASSOCIATION = re.compile(rf"\[\s*{REFERENCE}\s*\]")
SIMPLE_NAME_TREE = re.compile(rf"\[\([^()\\]{{1,128}}\)\s*{REFERENCE}\s*\]")
TIMEOUT = 8


def catalog_store(data: bytes) -> bytes | None:
    """Read the narrow, direct catalog association used by the qualified PDFs."""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as pdf:
        if not pdf.is_pdf or pdf.is_repaired or pdf.is_encrypted or not 0 < pdf.xref_length() <= MAX_XREF:
            return None
        catalog = pdf.pdf_catalog()
        association = CATALOG_ASSOCIATION.fullmatch(pdf.xref_get_key(catalog, "AF")[1])
        attachment = SIMPLE_NAME_TREE.fullmatch(pdf.xref_get_key(catalog, "Names/EmbeddedFiles/Names")[1])
        if association is None or attachment is None or association[1] != attachment[1]:
            return None
        filespec = int(association[1])
        if pdf.xref_get_key(filespec, "AFRelationship") != ("name", "/C2PA_Manifest"):
            return None
        stream = re.fullmatch(REFERENCE, pdf.xref_get_key(filespec, "EF/F")[1])
        if stream is None or not pdf.xref_is_stream(number := int(stream[1])):
            return None
        # Cover the original bytes precisely. Compressed, encrypted and indirect-length stores
        # can continue through the ordinary analysis while this restricted reader abstains.
        length_type, length = pdf.xref_get_key(number, "Length")
        if (length_type != "int" or not 0 < int(length) <= MAX_STORE
                or pdf.xref_get_key(number, "Filter")[0] != "null"):
            return None
        manifest = pdf.xref_stream_raw(number)
        if (manifest is None or len(manifest) != int(length) or manifest[4:8] != b"jumb"
                or int.from_bytes(manifest[:4], "big") != len(manifest)):
            return None
        return manifest


def _detail(data: bytes, mime: str, context):
    import c2pa

    with c2pa.Reader(mime, io.BytesIO(data), context=context) as reader:
        return json.loads(reader.detailed_json()), reader.is_embedded()


def _verified_document_binding(data: bytes, bundle: verifier.TrustBundle) -> bool:
    """Require the SDK-validated active manifest to be the unique catalog store.

    The signed data hash may exclude that store alone. All other bytes must be
    covered, including the catalog association and the current PDF revision.
    """
    import c2pa

    try:
        manifest = catalog_store(data)
        if manifest is None or data.count(manifest) != 1:
            return False
        with c2pa.Context.from_dict(verifier.settings(bundle)) as context:
            native, embedded = _detail(data, MIME, context)
            associated, _ = _detail(manifest, "application/c2pa", context)
        active = native["active_manifest"]
        if (not embedded or native["validation_state"].lower() != "trusted" or active != associated["active_manifest"]
                or len(native["manifests"]) != 1 or native["manifests"] != associated["manifests"]):
            return False
        native_assertions = native["manifests"][active]["assertion_store"]
        exclusions = native_assertions["c2pa.hash.data"]["exclusions"]
        start = data.index(manifest)
        # Verify that the independently recovered bytes begin at the stream boundary.
        begins_stream = data[max(0, start - 9):start].endswith((b"stream\n", b"stream\r\n", b"stream\r"))
        return begins_stream and exclusions == [{"start": start, "length": len(manifest)}]
    except (c2pa.C2paError, AttributeError, LookupError, RuntimeError, TypeError, ValueError):
        return False


def read(data: bytes) -> tuple[verifier.C2PAReport, bool]:
    """Validate a PDF in a time-bounded process without opening its attachments in the worker."""
    from dataclasses import fields

    try:
        reply = subprocess.run(  # noqa: S603 - fixed interpreter and module, bytes passed on stdin
            [sys.executable, "-B", "-m", "app.provenance.pdf_credentials"], input=data, capture_output=True,
            check=True, timeout=TIMEOUT,
        )
        packet = json.loads(reply.stdout)
        payload = packet["report"]
        allowed = {item.name for item in fields(verifier.C2PAReport)}
        if payload.keys() != allowed or payload["thumbnails"]:
            raise ValueError("Unexpected PDF credentials response")
        payload["codes"] = [verifier.ValidationCode(**row) for row in payload["codes"]]
        payload["source_types"] = [verifier.SourceType(**row) for row in payload["source_types"]]
        payload["chain"] = [verifier.ChainNode(**row) for row in payload["chain"]]
        return verifier.C2PAReport(**payload), packet["verified_document_binding"] is True
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, ValueError):
        return verifier.C2PAReport(state="unreadable"), False


def _main():
    from dataclasses import asdict

    data = sys.stdin.buffer.read()
    official = verifier.load_bundle(verifier.OFFICIAL)
    report = verifier.read(data, MIME, bundle=official, thumbnails=False)
    bound = report.state == "trusted" and report.covers_bytes and _verified_document_binding(data, official)
    print(json.dumps({"report": asdict(report), "verified_document_binding": bound}, ensure_ascii=True))


if __name__ == "__main__":
    _main()
