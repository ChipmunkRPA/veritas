"""Read exact public fixtures offline with the serving Content Credentials checks."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from app.detectors.audio.provenance_adapter import AudioCredentialsDetector
from app.detectors.image.adapters.provenance import C2PADetector
from app.detectors.text.adapters.provenance import PdfCredentialsDetector
from app.detectors.video.adapters.metadata import C2PAVideoDetector
from app.engine.provenance_gate import decide
from app.engine.stages.common import detector_entry
from app.provenance import c2pa, pdf_credentials

SCHEMA = "chipmunk.public-credentials-replay/1"
MAX_FIXTURE_BYTES = 20_000_000
ADAPTERS = {"image": C2PADetector, "video": C2PAVideoDetector, "audio": AudioCredentialsDetector,
            "document": PdfCredentialsDetector}


def _discard_exhibit(*args, **kwargs):
    return "local-replay"


def inspect(path: Path, expected_sha256: str) -> dict:
    with path.open("rb") as stream:
        data = stream.read(MAX_FIXTURE_BYTES + 1)
    if not 0 < len(data) <= MAX_FIXTURE_BYTES or sha256(data).hexdigest() != expected_sha256:
        raise ValueError("The fixture differs from the expected original SHA-256 or exceeds the replay limit")

    if data.startswith(b"%PDF-"):
        mime, kind = "application/pdf", "document"
        report, document_binding = pdf_credentials.read(data)
    else:
        mime = c2pa.sniff_mime(data)
        if mime is None or mime.split("/", 1)[0] not in ADAPTERS:
            raise ValueError("The fixture is outside the supported media containers")
        kind = mime.split("/", 1)[0]
        report = c2pa.read(data, mime, bundle=c2pa.load_bundle(c2pa.OFFICIAL),
                           legacy=c2pa.load_bundle(c2pa.LEGACY), thumbnails=False)
        document_binding = None

    ctx = SimpleNamespace(media=SimpleNamespace(original_bytes=lambda: data),
                          exhibits=SimpleNamespace(table=_discard_exhibit, image=_discard_exhibit))
    adapter = ADAPTERS[kind]()
    entry = detector_entry(adapter.spec, adapter.run(ctx), None)
    action = decide({"detectors": [entry]}, kind)["action"]
    return {
        "sha256": expected_sha256,
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


def verify(receipt: Path, directory: Path, sample: str | None) -> None:
    document = json.loads(receipt.read_text())
    expected_trust = {name: c2pa.load_bundle(name).anchors_sha256 for name in (c2pa.OFFICIAL, c2pa.LEGACY)}
    snapshot = sha256((c2pa.TRUST_DIR / "snapshot.json").read_bytes()).hexdigest()
    if (document["schema"] != SCHEMA or document["trust_snapshot_sha256"] != snapshot
            or document["trust_anchors_sha256"] != expected_trust):
        raise ValueError("The replay requires the recorded receipt schema and both trust bundles")
    if (sdk := c2pa.sdk_version()) != document["sdk"]:
        print(f"Comparing current SDK {sdk} with receipt SDK {document['sdk']}")
    rows = document["samples"]
    selected = [row for row in rows if sample is None or row["id"] == sample]
    if not selected:
        raise ValueError("The requested fixture is not in the receipt")
    for row in selected:
        name = row["id"]
        if not name.replace("_", "").isalnum():
            raise ValueError("Invalid fixture identifier")
        actual = inspect(directory / f"{name}.asset", row["verification"]["sha256"])
        if actual != row["verification"]:
            raise ValueError(f"{name}: live validation differs from the historical receipt")
        print(name, actual["state"], actual["binding"], actual["pipeline_action"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--sample")
    args = parser.parse_args()
    verify(args.receipt, args.directory, args.sample)


if __name__ == "__main__":
    main()
