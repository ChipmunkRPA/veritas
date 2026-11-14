"""Recompute a signed short speech review's coverage from the original audio."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from app.evidence.audio_short_review_binding import COVERAGE_SCHEMA, RATE, speech_counts, window_coverage
from app.reports.verify import verify_package

ACTIVITY_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"


def verify_coverage(package, original, public_key, weights):
    from app.detectors.audio.decode import decode_audio
    from app.detectors.audio.upgraded import SpeechActivity

    result = verify_package(package, public_key_pem=public_key.read_bytes(), source_audio=original)
    if not result.ok or result.bundle.get("analysis", {}).get("kind") != "audio":
        raise ValueError("Original and package must verify against the supplied trusted key for a standalone audio analysis")
    with ZipFile(package) as archive:
        evidence = json.loads(archive.read("evidence.json"))
    review = evidence.get("short_voice_review", {})
    if review.get("schema") != COVERAGE_SCHEMA:
        raise ValueError("This package has no measured short speech-window coverage")
    with weights.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != ACTIVITY_SHA256:
            raise ValueError("Local speech activity weights differ from the pinned asset")
    source = evidence["source"]
    decoded = decode_audio(original, source_sha256=source["sha256"], max_seconds=6.0, inspect_stereo=False)
    if decoded.mapping["pcm_sha256"] != source["pcm_sha256"]:
        raise ValueError("Re-decoded short speech review input differs from the original commitment")
    spans = SpeechActivity(weights).spans(decoded.pcm)
    expected = window_coverage(speech_counts(spans), decoded.mapping)
    if expected != review["window_coverage"]:
        raise ValueError("Measured speech coverage differs from the signed evidence")
    return {"status": "verified", "sample_rate": RATE, "checked": sum(row["status"] == "checked" for row in expected),
            "windows": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("original", type=Path)
    parser.add_argument("--public-key", type=Path, required=True)
    parser.add_argument("--activity-weights", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_coverage(args.package, args.original, args.public_key, args.activity_weights), sort_keys=True))
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Short speech-window verification failed. {error}\n")


if __name__ == "__main__":
    main()
