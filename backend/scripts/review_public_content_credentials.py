"""Inspect original files or replay public receipts with offline Content Credentials checks."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from app.core.config import settings
from app.detectors.audio.provenance_adapter import AudioCredentialsDetector
from app.detectors.image.adapters.provenance import C2PADetector
from app.detectors.text.adapters.provenance import OfficeCredentialsDetector, PdfCredentialsDetector
from app.detectors.video.adapters.metadata import C2PAVideoDetector
from app.engine.detector import StageOutputs
from app.engine.provenance_gate import decide
from app.engine.stages.common import detector_entry
from app.provenance import c2pa, office_credentials, pdf_credentials
from app.storage.files import DOCUMENT_ARCHIVE_EXTENSIONS, DOCUMENT_ARCHIVES, validate_kind

SCHEMA = "chipmunk.public-credentials-replay/1"
MAX_FIXTURE_BYTES = 20_000_000
ADAPTERS = {"image": C2PADetector, "video": C2PAVideoDetector, "audio": AudioCredentialsDetector}


def _discard_exhibit(*args, **kwargs):
    return "local-replay"


def inspect(path: Path, expected_sha256: str | None = None, *, office_extension: str | None = None) -> dict:
    extension = office_extension or path.suffix.lower()
    limit = settings.max_document_bytes if extension in DOCUMENT_ARCHIVES else MAX_FIXTURE_BYTES
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if not 0 < len(data) <= limit:
        raise ValueError("The file is empty or exceeds the inspection limit")
    digest = sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("The file differs from the expected original SHA-256")

    if data.startswith(b"%PDF-"):
        mime, kind = "application/pdf", "document"
        report, document_binding = pdf_credentials.read(data)
        adapter = PdfCredentialsDetector()
    elif extension in DOCUMENT_ARCHIVES:
        mime, _ = validate_kind("document", data, f"original{extension}", len(data))
        if mime not in DOCUMENT_ARCHIVE_EXTENSIONS:
            raise ValueError("The file does not have a recognized DOCX or ODT format")
        kind = "document"
        report, document_binding = office_credentials.read(
            data, mime, bundle=c2pa.load_bundle(c2pa.OFFICIAL), legacy=c2pa.load_bundle(c2pa.LEGACY),
        )
        adapter = OfficeCredentialsDetector()
    else:
        mime = c2pa.sniff_mime(data)
        if mime is None or mime.split("/", 1)[0] not in ADAPTERS:
            raise ValueError("The file is outside the supported media containers")
        kind = mime.split("/", 1)[0]
        report = c2pa.read(data, mime, bundle=c2pa.load_bundle(c2pa.OFFICIAL),
                           legacy=c2pa.load_bundle(c2pa.LEGACY), thumbnails=False)
        document_binding = None
        adapter = ADAPTERS[kind]()

    ctx = SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: data),
                          outputs=StageOutputs(probe={"mime": mime}),
                          exhibits=SimpleNamespace(table=_discard_exhibit, image=_discard_exhibit))
    entry = detector_entry(adapter.spec, adapter.run(ctx), None)
    action = decide({"detectors": [entry]}, kind)["action"]
    return {
        "sha256": digest,
        "bytes": len(data),
        "media_type": mime,
        "state": report.state,
        "signer": report.signer.get("common_name") or None,
        "claim_generator": report.claim_generator or None,
        "legacy_trusted": report.legacy_trusted,
        "binding": report.binding_status,
        "document_binding": document_binding,
        "ai_declared": report.ai_declared,
        "source_types": sorted({source.term for source in report.source_types}),
        "failures": sorted(set(report.failure_codes)),
        "actions": [action["action"] for action in report.actions],
        "pipeline_action": action,
    }


def trust_context() -> dict:
    return {
        "sdk": c2pa.sdk_version(),
        "trust_snapshot_sha256": sha256((c2pa.TRUST_DIR / "snapshot.json").read_bytes()).hexdigest(),
        "trust_anchors_sha256": {name: c2pa.load_bundle(name).anchors_sha256 for name in (c2pa.OFFICIAL, c2pa.LEGACY)},
    }


def verify(receipt: Path, directory: Path, sample: str | None) -> None:
    document = json.loads(receipt.read_text())
    context = trust_context()
    if (document["schema"] != SCHEMA or document["trust_snapshot_sha256"] != context["trust_snapshot_sha256"]
            or document["trust_anchors_sha256"] != context["trust_anchors_sha256"]):
        raise ValueError("The replay requires the recorded receipt schema and both trust bundles")
    if context["sdk"] != document["sdk"]:
        print(f"Comparing current SDK {context['sdk']} with receipt SDK {document['sdk']}")
    rows = document["samples"]
    selected = [row for row in rows if sample is None or row["id"] == sample]
    if not selected:
        raise ValueError("The requested fixture is not in the receipt")
    for row in selected:
        name = row["id"]
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid fixture identifier")
        expected = row["verification"]
        actual = inspect(directory / f"{name}.asset", expected["sha256"],
                         office_extension=DOCUMENT_ARCHIVE_EXTENSIONS.get(expected["media_type"]))
        if actual != expected:
            raise ValueError(f"{name}: live validation differs from the historical receipt")
        print(name, actual["state"], actual["binding"], actual["pipeline_action"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path)
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--sample")
    parser.add_argument("--file", type=Path, help="Inspect one original file, including DOCX and ODT")
    parser.add_argument("--expect-sha256", help="Require this original byte hash when using --file")
    args = parser.parse_args()
    try:
        if args.file is not None:
            if args.receipt is not None or args.directory is not None or args.sample is not None:
                parser.error("--file cannot be combined with a receipt, directory, or sample")
            result = inspect(args.file, args.expect_sha256)
            print(json.dumps({**trust_context(), "verification": result}, indent=2))
        else:
            if args.receipt is None or args.directory is None or args.expect_sha256 is not None:
                parser.error("Provide a receipt and directory to replay, or use --file")
            verify(args.receipt, args.directory, args.sample)
    except (OSError, ValueError, c2pa.TrustBundleError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
