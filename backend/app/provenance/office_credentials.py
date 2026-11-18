"""Offline validation of the active credential in a Word or OpenDocument package."""

from __future__ import annotations

import io
from zipfile import ZIP_STORED, BadZipFile, ZipFile

from app.provenance import c2pa
from app.storage.files import DOCUMENT_ARCHIVE_EXTENSIONS, check_document_archive_safe

STORE = "META-INF/content_credential.c2pa"
COLLECTION_MATCH = "assertion.collectionHash.match"


def read(data: bytes, mime: str, *, bundle: c2pa.TrustBundle, legacy: c2pa.TrustBundle) -> tuple[c2pa.C2PAReport, bool]:
    extension = DOCUMENT_ARCHIVE_EXTENSIONS.get(mime)
    if extension is None or not data.startswith(b"PK\x03\x04"):
        return c2pa.C2PAReport(state="unreadable"), False
    try:
        with io.BytesIO(data) as stream:
            check_document_archive_safe(stream, extension)
            with ZipFile(stream) as archive:
                entry = archive.getinfo(STORE)
                if entry.compress_type != ZIP_STORED or entry.flag_bits != 0 or entry.file_size == 0:
                    return c2pa.C2PAReport(state="unreadable"), False
    except KeyError:
        return c2pa.C2PAReport(state="absent"), False
    except (ValueError, BadZipFile, OSError):
        return c2pa.C2PAReport(state="unreadable"), False

    report = c2pa.read(data, mime, bundle=bundle, legacy=legacy, thumbnails=False)
    # A data hash for an embedded attachment cannot establish the provenance of the package.
    bound = report.covers_bytes and any(code.scope == "active manifest" and code.level == "success"
                                       and code.code == COLLECTION_MATCH for code in report.codes)
    return report, bound
