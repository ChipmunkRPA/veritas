"""Offline public-only reconstruction of the corrected v9 source/role panel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from app.detectors.audio.decode import read_wav, wav_bytes
from scripts.rebuild_audio_multisource_inputs_v8 import cached_source, digest, pcm_hash, source_variants
from scripts.review_audio_identity_plan_v9 import review


def rebuild(capsule, cache, output, expected_sha):
    if digest(capsule) != expected_sha:
        raise ValueError("Corrected public source capsule checksum differs")
    body = json.loads(gzip.decompress(capsule.read_bytes()))
    if body["original_source_catalogue"] != "audio-v8-source-plan.json.gz":
        raise ValueError("Expected the original public catalogue beside the repair")
    original_path = capsule.parent / body["original_source_catalogue"]
    if digest(original_path) != body["original_source_catalogue_sha256"]:
        raise ValueError("Original public source catalogue checksum differs")
    metadata = review(body, json.loads(gzip.decompress(original_path.read_bytes())))
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    backgrounds, source_errors = {}, {}
    for row in body["backgrounds"]:
        try:
            backgrounds[row["sha256"]] = cached_source(cache, row)
        except Exception as error:
            source_errors[row["sha256"]] = type(error).__name__
    by_source = defaultdict(list)
    for rows in body["variants"].values():
        for row in rows:
            by_source[row["source_sha256"]].append(row)
    journal = output / "receipts.jsonl"
    header = {
        "schema": "chipmunk.audio-v9-public-reconstruction/1",
        "capsule_sha256": expected_sha,
        "code_sha256": digest(Path(__file__)),
        "planned": 5328,
        "metadata_review": metadata,
        "scope": "Local corrected public-capsule/source-cache reconstruction; no model, private manifest, paid analysis or outside reviewer.",
    }
    outcomes = []
    with journal.open("x") as stream:
        stream.write(json.dumps(header) + "\n")
        for index, source in enumerate(body["sources"], 1):
            rows = by_source[source["sha256"]]
            try:
                values = source_variants(cached_source(cache, source)[:64000], rows, backgrounds)
                source_error = None
            except Exception as error:
                values, source_error = {}, type(error).__name__
            for row in rows:
                result = {
                    "id": row["id"],
                    "role": row["role"],
                    "sha256": row["sha256"],
                    "pcm_sha256": row["pcm_sha256"],
                    "status": "failed",
                }
                try:
                    if source_error:
                        raise ValueError(source_error)
                    pcm = values[row["condition"]]
                    if pcm_hash(pcm) != row["payload_pcm_sha256"]:
                        raise ValueError("Corrected float waveform payload differs")
                    raw = wav_bytes(pcm)
                    if (
                        hashlib.sha256(raw).hexdigest() != row["sha256"]
                        or pcm_hash(read_wav(raw)) != row["pcm_sha256"]
                    ):
                        raise ValueError("Corrected WAV file/PCM differs")
                    path = output / "inputs" / (row["sha256"] + ".wav")
                    if path.exists():
                        if path.read_bytes() != raw:
                            raise ValueError("Repeated output hash has different bytes")
                    else:
                        with path.open("xb") as target:
                            target.write(raw)
                    result["status"] = "ok"
                except Exception as error:
                    result["error_type"] = source_error or type(error).__name__
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                outcomes.append(result)
            if index % 60 == 0 or index == len(body["sources"]):
                print(json.dumps({"foregrounds_rebuilt": index, "planned": len(body["sources"])}), flush=True)
    counts = dict(Counter(r["status"] for r in outcomes))
    result = {
        "passed": len(outcomes) == 5328 and counts.get("failed", 0) == 0,
        "counts": counts,
        "capsule_sha256": expected_sha,
        "receipts_sha256": digest(journal),
        "background_source_errors": source_errors,
        "scope": header["scope"],
    }
    with (output / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capsule", type=Path)
    parser.add_argument("source_cache", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capsule-sha256", required=True)
    args = parser.parse_args()
    rebuild(args.capsule, args.source_cache, args.output, args.capsule_sha256)
