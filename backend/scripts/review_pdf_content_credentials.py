"""Replay offline whole-document credentials and report only verification facts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from app.core.config import settings
from app.provenance import c2pa, pdf_credentials

SCHEMA = "chipmunk.pdf-content-credentials-replay/1"


def review(path: Path, expected_sha256: str | None = None) -> dict:
    with path.open("rb") as stream:
        content = stream.read(settings.max_document_bytes + 1)
    if not 0 < len(content) <= settings.max_document_bytes or not content.startswith(b"%PDF-"):
        raise ValueError("Use an original PDF within the local document limit")
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("The original does not match the expected SHA-256")
    report, bound = pdf_credentials.read(content)
    return {"schema": SCHEMA, "source_sha256": digest, "source_bytes": len(content), "sdk": c2pa.sdk_version(),
            "trust_sha256": c2pa.load_bundle(c2pa.OFFICIAL).anchors_sha256, "status": report.state,
            "active_byte_binding": report.binding_status, "whole_document_binding": bound,
            "signed_ai_declaration": bool(bound and report.ai_declared and not report.failure_codes),
            "failure_codes": sorted(set(report.failure_codes))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("--expect-sha256")
    args = parser.parse_args()
    print(json.dumps(review(args.original, args.expect_sha256), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
