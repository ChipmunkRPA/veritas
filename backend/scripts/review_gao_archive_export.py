"""Verify a GAO archive comparison JSON from the submitted document and pinned pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from app.detectors.text.extract import extract
from app.evidence import document_archive_pdf
from app.evidence import document_gao_archive as gao
from app.evidence.document_archive_word import compare_gao
from app.evidence.document_revision import MAX_BYTES
from app.evidence.document_revision_pdf import MAX_PREVIEW_BYTES
from scripts.review_document_revision_export import _read, _unique

MAX_EXPORT = 256 * 1024


def review(export: Path, submitted: Path, pack: Path, sha256: str):
    extension = submitted.suffix.lower()
    if extension not in {".txt", ".docx", ".pdf"}:
        raise ValueError("A retained TXT, Word or PDF original is required")
    original = _read(submitted, MAX_BYTES)
    limit = {".txt": MAX_EXPORT, ".docx": MAX_BYTES, ".pdf": MAX_BYTES + 2 * MAX_PREVIEW_BYTES}[extension]
    proof = json.loads(_read(export, limit), object_pairs_hook=_unique)
    if not isinstance(proof, dict):
        raise ValueError("Expected a GAO comparison object")
    catalog, index = gao.library(pack, sha256)
    upload_sha256 = hashlib.sha256(original).hexdigest()
    if extension == ".pdf":
        expected = document_archive_pdf.compare(original, upload_sha256, gao_archive=(sha256, catalog, index))
    elif extension == ".docx":
        expected = compare_gao(original, upload_sha256, sha256, catalog, index)
    else:
        with TemporaryDirectory(prefix="chipmunk-gao-review-") as directory:
            staged = Path(directory) / "retained.txt"
            staged.write_bytes(original)
            doc = extract(staged, ".txt")
        if "\ufffd" in doc.text or "\x00" in doc.text:
            raise ValueError("The file contains undecodable text")
        expected = gao.compare(doc, upload_sha256, sha256, catalog, index)
        if (proof != expected and proof.get("status") == "no_qualified_reference"
                and expected["status"] == "reference_bound" and not expected["changes"]):
            # Previously, the same exact excerpt could have received an abstention.
            expected = gao.compare(doc, upload_sha256, sha256, catalog, index, accept_exact_excerpt=False)
    if proof != expected:
        raise ValueError("The GAO comparison does not replay from these source bytes and positions")
    return {"status": expected["status"], "source": (expected["reference"] or {}).get("id"),
            "exact_changes_verified": len(expected["changes"]),
            "sentence_ranges_verified": len(expected["submitted_sentence_ranges"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("export", "submitted", "pack"):
        parser.add_argument(field, type=Path)
    parser.add_argument("pack_sha256")
    args = parser.parse_args()
    print(json.dumps(review(args.export, args.submitted, args.pack, args.pack_sha256), indent=2))


if __name__ == "__main__":
    main()
