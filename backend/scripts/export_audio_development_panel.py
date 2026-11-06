"""Export allowlisted public-source metadata for independent audio review, no media."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from urllib.parse import urlsplit

from scripts.audio_quality_v5 import digest

SOURCE_FIELDS = (
    "id",
    "dataset",
    "revision",
    "url",
    "archive_member",
    "upstream_split",
    "sha256",
    "bytes",
    "decoded_pcm_sha256",
    "decoded_samples",
    "original_sample_rate",
    "label",
    "speaker",
    "cluster_id",
    "generator",
    "generator_voice",
    "generator_engine",
    "source_group",
    "transcript_sha256",
    "seconds",
    "license",
    "license_url",
    "label_source",
    "voice_identity_limit",
    "production_pcm_sha256",
    "source_prefix_pcm_sha256",
)
BACKGROUND_FIELDS = (
    "dataset",
    "revision",
    "url",
    "parquet",
    "parquet_row_group",
    "parquet_group_row",
    "sha256",
    "bytes",
    "decoded_pcm_sha256",
    "decoded_samples",
    "original_sample_rate",
    "kind",
    "source_id",
    "license",
    "license_url",
    "attribution",
)
INPUT_FIELDS = (
    "id",
    "base_id",
    "label",
    "cluster_id",
    "generator",
    "dataset",
    "seconds",
    "condition",
    "sha256",
    "pcm_sha256",
    "samples",
    "sample_rate",
    "source_interval_samples",
    "background_sha256",
    "background_id",
    "background_kind",
    "background_start_sample",
    "snr_db",
    "common_peak_gain",
    "measured_snr_db",
)


def wav_metadata(blob):
    if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE" or struct.unpack_from("<I", blob, 4)[0] + 8 != len(blob):
        raise ValueError("Unsupported WAV container")
    offset, timestamp, payload, fmt = 12, None, None, False
    while offset + 8 <= len(blob):
        kind, size = struct.unpack_from("<4sI", blob, offset)
        data = blob[offset + 8 : offset + 8 + size]
        if len(data) != size:
            raise ValueError("Truncated WAV chunk")
        if kind == b"fmt ":
            fmt = size == 16 and struct.unpack("<HHIIHH", data) == (3, 1, 16000, 64000, 4, 32)
        elif kind == b"PEAK":
            if size != 16 or struct.unpack_from("<I", data)[0] != 1 or timestamp is not None:
                raise ValueError("Unsupported WAV PEAK metadata")
            timestamp = struct.unpack_from("<I", data, 4)[0]
        elif kind == b"data":
            if payload is not None:
                raise ValueError("Multiple WAV payloads")
            payload = hashlib.sha256(data).hexdigest()
        offset += 8 + size + size % 2
    if not fmt or timestamp is None or payload is None or offset != len(blob):
        raise ValueError("Incomplete float WAV container")
    return {"container_peak_timestamp": timestamp, "float_payload_sha256": payload}


def public_reference(value):
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or parts.hostname
        not in {
            "huggingface.co",
            "github.com",
            "www.openslr.org",
            "openslr.org",
            "www.cstr.ed.ac.uk",
            "datashare.ed.ac.uk",
        }
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Source reference is not an approved unsigned public URL")


def choose(row, fields):
    out = {key: row[key] for key in fields if key in row}
    for key in ("url", "license_url"):
        if out.get(key):
            public_reference(out[key])
    return out


def export(source_path, development_path, output):
    source, development = json.loads(source_path.read_text()), json.loads(development_path.read_text())
    if development["role"] != "consumed v3 development; not independent confirmation":
        raise ValueError("Only the identified consumed panel can be exported here")
    if digest(source_path) != development["source_manifest_file_sha256"]:
        raise ValueError("Development and source metadata differ")
    ids = {r["base_id"] for r in development["variants"]}
    inputs = {r["id"] for r in development["variants"]}
    bgs = {r["background_id"] for r in development["variants"] if r["background_id"]}
    original = [choose(r, SOURCE_FIELDS) for r in source["base_files"] if r["id"] in ids]
    backgrounds = [choose(r, BACKGROUND_FIELDS) for r in source["backgrounds"] if r["source_id"] in bgs]
    variants = []
    frozen = {r["id"]: r for r in development["variants"]}
    for row in source["variants"]:
        if row["id"] not in inputs:
            continue
        r = frozen[row["id"]]
        if row["sha256"] != r["sha256"] or row["pcm_sha256"] != r["pcm_sha256"]:
            raise ValueError("Input bytes differ from the frozen panel")
        original_input = Path(development["source_root"]) / row["path"]
        if digest(original_input) != row["sha256"]:
            raise ValueError("Original input container changed")
        variants.append(
            {
                **choose(row, INPUT_FIELDS),
                **wav_metadata(original_input.read_bytes()),
                "development_fold": r["fold"],
                "connected_group": r["group"],
                "baseline_raw_response": r["raw"],
                "baseline_original_windows": [
                    {k: w[k] for k in ("start_sample", "end_sample")} for w in r["original_windows"]
                ],
                "guarded_reference_probability_ai": r["serving_p"],
                "guarded_reference_direction": r["serving_decision"],
            }
        )
    if (
        len(original) != len(ids)
        or len(variants) != len(inputs)
        or {r["source_id"] for r in backgrounds} != bgs
    ):
        raise ValueError("Source catalogue would omit planned provenance")
    body = {
        "schema": "chipmunk.public-audio-development-panel/2",
        "role": "consumed development; not independent confirmation",
        "source_manifest_file_sha256": digest(source_path),
        "development_manifest_file_sha256": digest(development_path),
        "sources": original,
        "backgrounds": backgrounds,
        "inputs": variants,
        "source_normalization": source["source_normalization"],
        "terms_and_attribution": [
            "This catalogue contains no source audio, checkpoints, customer uploads or full transcripts.",
            "Source metadata retains its publisher terms; project MIT code does not relicense datasets.",
            "DFADD: isjwdu/DFADD publisher, Diffusion-based Fake Audio Detection Dataset, April 2025 revised release; MIT declaration, underlying VCTK attribution and LJ Speech public-domain sources.",
            "VCTK: Junichi Yamagishi, Christophe Veaux and Kirsten MacDonald, Edinburgh CSTR, CC BY 4.0.",
            "LibriSpeech: Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur, CC BY 4.0; underlying LibriVox recordings and public-domain texts.",
            "MUSAN: David Snyder, Guoguo Chen and Daniel Povey; selected FreeSound recordings declared public domain; selected RFM music Kevin MacLeod/Incompetech under CC BY 3.0, retaining each record's attribution.",
        ],
        "limits": development["limitations"]
        + [
            "These exact inputs and responses were already consumed; reuse is replication or development, never fresh confirmation.",
            "Publisher class and rights declarations were retained, not independently authenticated for every recording.",
            "Known source/identity/prompt groups and input variants must stay together; no outside reviewer has yet validated these results.",
            "URLs and hashes locate public sources; recreating matching input bytes also requires the recorded v3 mixing/decoding method and runtime. This catalogue has not been independently reconstructed end to end.",
            "WAV PEAK timestamps are recorded container metadata, not audio evidence. Restore these original timestamps after independently recomputing and verifying the float sample payload to reproduce exact file hashes.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "sources": len(original),
                "backgrounds": len(backgrounds),
                "inputs": len(variants),
                "sha256": digest(output),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source_manifest", type=Path)
    p.add_argument("development_manifest", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    export(a.source_manifest, a.development_manifest, a.output)
