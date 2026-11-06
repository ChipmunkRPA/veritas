"""Rebuild public v8 inputs offline from separately acquired SHA-named sources.

Uses the public capsule only; no private manifest, transcript, provider profile,
credentials, network access or model is needed. Source data keeps upstream terms.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, read_wav, wav_bytes
from scripts.review_audio_multisource_plan_v8 import CONDITIONS, review


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pcm_hash(pcm):
    return hashlib.sha256(np.asarray(pcm, "<f4").tobytes()).hexdigest()


def cached_source(cache, row):
    sha = row["sha256"]
    if not re.fullmatch("[0-9a-f]{64}", sha):
        raise ValueError("Source cache accepts SHA-256 basenames only")
    path = cache / (sha + ".audio")
    if digest(path) != sha:
        raise ValueError("Separately acquired original source bytes differ")
    audio = decode_audio(path, source_sha256=sha)
    if audio.mapping["pcm_sha256"] != row["production_pcm_sha256"] or len(audio.pcm) != row["full_samples"]:
        raise ValueError("Original production decode differs from public source reference")
    if row.get("prefix_pcm_sha256") and pcm_hash(audio.pcm[:64000]) != row["prefix_pcm_sha256"]:
        raise ValueError("Original public foreground prefix differs")
    return audio.pcm


def source_variants(clean, rows, background_pcm):
    clean = np.asarray(clean, np.float64)
    if clean.shape != (64000,) or not np.isfinite(clean).all():
        raise ValueError("Four finite seconds of original foreground required")
    level = float(np.sqrt(np.mean(clean**2)))
    if level < 1e-4:
        raise ValueError("Silent original prefix")
    signal = clean * (0.1 / level)
    values = {"clean": signal}
    for row in rows:
        if row["condition"] == "clean":
            continue
        full = background_pcm[row["background_sha256"]]
        start = row["background_start_sample"]
        interval = full[start : start + 64000]
        if len(interval) != 64000 or pcm_hash(interval) != row["background_interval_pcm_sha256"]:
            raise ValueError("Public background interval differs")
        noise = interval.astype(np.float64)
        rms = float(np.sqrt(np.mean(noise**2)))
        if rms < 1e-5:
            raise ValueError("Silent original background interval")
        values[row["condition"]] = signal + noise * (0.1 / rms / 10 ** (row["snr_db"] / 20))
    if set(values) != set(CONDITIONS):
        raise ValueError("Every original condition required before common gain")
    gain = min(1.0, 0.98 / max(float(np.max(np.abs(value))) for value in values.values()))
    if any(not math.isclose(row["common_peak_gain"], gain, abs_tol=1e-12, rel_tol=1e-12) for row in rows):
        raise ValueError("Independently recomputed common peak gain differs")
    return {name: (values[name] * gain).astype(np.float32) for name in CONDITIONS}


def rebuild(capsule, cache, output, expected_sha):
    if digest(capsule) != expected_sha:
        raise ValueError("Public capsule checksum differs")
    body = json.loads(gzip.decompress(capsule.read_bytes()))
    metadata = review(body)
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs").mkdir()
    backgrounds, source_errors = {}, {}
    for row in body["backgrounds"]:
        try:
            backgrounds[row["sha256"]] = cached_source(cache, row)
        except Exception as error:
            source_errors[row["sha256"]] = type(error).__name__
    by_source = defaultdict(list)
    for row in body["variants"]["development"] + body["variants"]["confirmation"]:
        by_source[row["source_sha256"]].append(row)
    outcomes = []
    journal = output / "receipts.jsonl"
    header = {
        "schema": "chipmunk.audio-v8-offline-reconstruction/1",
        "capsule_sha256": expected_sha,
        "code_sha256": digest(Path(__file__)),
        "planned": 5364,
        "metadata_review": metadata,
        "scope": "Local public-capsule/source-cache reconstruction only. No pretrained model, paid analysis or outside reviewer implied.",
    }
    with journal.open("x") as stream:
        stream.write(json.dumps(header) + "\n")
        for index, source in enumerate(body["sources"], 1):
            rows = by_source[source["sha256"]]
            try:
                clean = cached_source(cache, source)[:64000]
                values = source_variants(clean, rows, backgrounds)
                source_error = None
            except Exception as error:
                values, source_error = {}, type(error).__name__
            for row in rows:
                record = {
                    "id": row["id"],
                    "role": row["role"],
                    "sha256": row["sha256"],
                    "pcm_sha256": row["pcm_sha256"],
                    "status": "failed",
                }
                try:
                    if source_error is not None:
                        raise ValueError(source_error)
                    pcm = values[row["condition"]]
                    if pcm_hash(pcm) != row["payload_pcm_sha256"]:
                        raise ValueError("Reconstructed float payload differs")
                    raw = wav_bytes(pcm)
                    if (
                        hashlib.sha256(raw).hexdigest() != row["sha256"]
                        or pcm_hash(read_wav(raw)) != row["pcm_sha256"]
                    ):
                        raise ValueError("Reconstructed WAV file/PCM differs")
                    target = output / "inputs" / (row["sha256"] + ".wav")
                    if target.exists():
                        if target.read_bytes() != raw:
                            raise ValueError("Repeated output hash has different waveform bytes")
                    else:
                        with target.open("xb") as stream_out:
                            stream_out.write(raw)
                    record["status"] = "ok"
                except Exception as error:
                    record["error_type"] = source_error or type(error).__name__
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                outcomes.append(record)
            if index % 60 == 0 or index == len(body["sources"]):
                print(json.dumps({"source_foregrounds": index, "planned": len(body["sources"])}), flush=True)
    counts = dict(Counter(r["status"] for r in outcomes))
    result = {
        "passed": len(outcomes) == 5364 and counts.get("failed", 0) == 0,
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
