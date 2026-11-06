"""Read a fixed additional public-domain noise row group from verified originals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from app.detectors.audio.decode import decode_audio
from scripts.plan_audio_multisource_v8 import digest
from scripts.prepare_audio_confirmation_inputs_v7 import store
from scripts.prepare_audio_guard_v7 import write


def materialize(root, output):
    plan_path = root / "source-planning/additional-noise-acquisition-plan.json"
    plan = json.loads(plan_path.read_text())
    archive_root = root / "reserved-sources/additional-noise"
    receipt_path = archive_root / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["plan_sha256"] != digest(plan_path) or len(receipt["archives"]) != 1:
        raise ValueError("Fixed complete noise-original receipt required")
    asset, record = plan["assets"][0], receipt["archives"][0]
    path = archive_root / asset["name"]
    if (
        record["status"] != "verified"
        or not record["publisher_sha256_verified"]
        or digest(path) != asset["sha256"]
    ):
        raise ValueError("Original noise Parquet publisher checksum failed")
    reader, rows = pq.ParquetFile(path), []
    for group in plan["selected_row_groups"]:
        for offset, row in enumerate(
            reader.read_row_group(group, columns=["path", "source", "audio"]).to_pylist()
        ):
            if not row["path"].startswith("noise/free-sound/"):
                continue
            ref = store(archive_root, row["audio"]["bytes"])
            source = archive_root / ref["path"]
            audio = decode_audio(source, source_sha256=ref["sha256"])
            rows.append(
                {
                    "path": row["path"],
                    "source": row["source"],
                    "ref": {
                        **ref,
                        "source_path": str(source.resolve()),
                        "dataset": asset["source"],
                        "revision": asset["revision"],
                        "url": asset["url"],
                        "parquet": asset["source_path"],
                        "parquet_row_group": group,
                        "parquet_group_row": offset,
                        "decoded_pcm_sha256": audio.mapping["pcm_sha256"],
                        "decoded_samples": len(audio.pcm),
                        "original_sample_rate": audio.mapping["source_sample_rate"],
                        "whole_parquet_sha256": asset["sha256"],
                        "publisher_whole_parquet_sha256_verified": True,
                    },
                }
            )
    if not rows:
        raise ValueError("Fixed row group contains no eligible public-domain subset records")
    write(
        output,
        {
            "schema": "chipmunk.audio-v8-additional-noise-inventory/1",
            "code_sha256": digest(Path(__file__)),
            "plan_sha256": digest(plan_path),
            "receipt_sha256": digest(receipt_path),
            "rows": rows,
            "detector_inference_run": False,
            "deployment_eligible": False,
        },
    )
    print(
        json.dumps(
            {
                "inventory_sha256": digest(output),
                "public_domain_subset_originals": len(rows),
                "at_least_eight_seconds": sum(r["ref"]["decoded_samples"] >= 128000 for r in rows),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    materialize(args.root, args.output)
