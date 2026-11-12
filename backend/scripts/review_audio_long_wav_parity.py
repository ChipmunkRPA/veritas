# SPDX-License-Identifier: Apache-2.0
"""Reopen both lossless containers and verify the frozen long-speech inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from app.detectors.audio.decode import decode_audio, wav_bytes
from app.evidence.audio_long_review_binding import intervals, supported_source
from app.evidence.audio_short_review_binding import source_metadata

PANELS = Path(__file__).resolve().parents[2] / "docs/evaluation-panels"
PANELS_TO_CHECK = {
    "audio-radar-transfer-2026-11-12.json": ("panel/clips", 32),
    "audio-radar-long-confirmation-2026-11-12.json": ("confirmation", 40),
}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def pcm_digest(decoded, start, stop):
    return digest(decoded.pcm[start:stop].astype("<f4").tobytes())


def check(root):
    groups = {}
    with tempfile.TemporaryDirectory() as temp:
        scratch = Path(temp) / "source.wav"
        for receipt, (directory, expected) in PANELS_TO_CHECK.items():
            body = (PANELS / receipt).read_bytes()
            rows = [row for row in json.loads(body)["rows"] if row.get("stratum", "long") == "long"]
            if len(rows) != expected:
                raise ValueError("Publisher comparison roster changed")
            windows = 0
            for row in rows:
                flac = decode_audio(root / directory / f"{row['id']}.flac", source_sha256=row["flac_sha256"])
                wav = wav_bytes(flac.pcm)
                scratch.write_bytes(wav)
                second = decode_audio(scratch, source_sha256=digest(wav))
                expected_scope = {"source_codec": "pcm_s16le", "source_container": "wav", "source_bit_depth": 16}
                if (not supported_source(flac.mapping) or not supported_source(second.mapping)
                        or source_metadata(second.mapping) != expected_scope or flac.wav != second.wav
                        or second.mapping["pcm_sha256"] != row["pcm_sha256"] or len(second.pcm) != row["samples"]
                        or flac.mapping["timeline"] != second.mapping["timeline"]):
                    raise ValueError(f"Lossless delivery changed exact input geometry for {row['id']}")
                if [list(span) for span in intervals(len(second.pcm))] != [item["interval"] for item in row["portions"]]:
                    raise ValueError(f"Frozen long speech schedules differ for {row['id']}")
                for item in row["portions"]:
                    if pcm_digest(second, *item["interval"]) != item["pcm_sha256"]:
                        raise ValueError(f"Frozen long speech input differs for {row['id']}")
                    windows += 1
            groups[receipt] = {"source_receipt_sha256": digest(body), "paired_containers": len(rows), "window_inputs": windows}
    return {"schema": "chipmunk.audio-long-wav-parity/1", "source_panels": groups,
            "paired_containers": sum(row["paired_containers"] for row in groups.values()),
            "window_inputs": sum(row["window_inputs"] for row in groups.values()), "authorship_confidence_weight": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("radar_source", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.radar_source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
