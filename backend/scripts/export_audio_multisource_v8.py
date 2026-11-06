"""Publish explicit source/role/input metadata; exclude text, profiles and paths."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from scripts.plan_audio_multisource_v8 import digest
from scripts.prepare_audio_confirmation_v7 import text_keys


def hashed(value):
    return hashlib.sha256(value.encode()).hexdigest()


def public_source(row):
    fields = (
        "sha256",
        "bytes",
        "dataset",
        "revision",
        "role",
        "group",
        "label",
        "generator",
        "kind",
        "production_pcm_sha256",
        "prefix_pcm_sha256",
        "full_samples",
        "license",
        "license_url",
        "license_sha256",
        "attribution",
        "parquet",
        "parquet_row_group",
        "parquet_group_row",
        "parquet_local_row",
        "row_index",
        "parquet_row",
        "whole_parquet_sha256",
        "publisher_whole_parquet_sha256_verified",
    )
    result = {key: row[key] for key in fields if key in row}
    result["id_sha256"] = hashed(row["id"])
    for key in ("voice_key", "source_key"):
        if row.get(key):
            result[key + "_sha256"] = hashed(row[key])
    result["text_connection_keys"] = sorted(hashed(key) for key in text_keys(row.get("transcript") or ""))
    if row.get("url"):
        url = urlparse(row["url"])
        if (
            url.scheme != "https"
            or url.query
            or url.fragment
            or url.hostname not in ("huggingface.co", "openslr.trmal.net")
        ):
            raise ValueError("Public pinned publisher URL without credentials/query required")
        result["url"] = row["url"]
    for key in ("archive_member", "member"):
        if row.get(key):
            result["archive_member_sha256"] = hashed(row[key])
    if row.get("archive_name") or row.get("archive"):
        result["archive_name"] = Path(row.get("archive_name") or row["archive"]).name
    return result


def public_variant(row):
    fields = (
        "id",
        "group",
        "role",
        "label",
        "generator",
        "dataset",
        "condition",
        "source_sha256",
        "source_prefix_pcm_sha256",
        "source_interval_samples",
        "sha256",
        "payload_pcm_sha256",
        "pcm_sha256",
        "samples",
        "sample_rate",
        "seconds",
        "common_peak_gain",
        "background_role",
        "background_kind",
        "background_sha256",
        "background_start_sample",
        "background_interval_pcm_sha256",
        "snr_db",
        "measured_snr_db",
    )
    result = {key: row[key] for key in fields}
    result["base_id_sha256"] = hashed(row["base_id"])
    return result


def catalogue(root):
    prepared_path = root / "prepared-sources.json"
    prepared = json.loads(prepared_path.read_text())
    variants = {}
    manifest_hashes = {}
    for name in ("development", "confirmation"):
        path = root / name / "variants.json"
        body = json.loads(path.read_text())
        if body["prepared_sources_sha256"] != digest(prepared_path):
            raise ValueError("Exact variants use different prepared originals")
        variants[name] = [public_variant(row) for row in body["variants"]]
        manifest_hashes[name] = digest(path)
    assets = []
    for name in (
        "confirmation-acquisition-plan.json",
        "libri-acquisition-plan.json",
        "additional-noise-acquisition-plan.json",
    ):
        body = json.loads((root / "source-planning" / name).read_text())
        assets.extend(
            {
                k: row[k]
                for k in ("name", "url", "bytes", "sha256", "license", "source", "revision")
                if k in row
            }
            for row in body["assets"]
        )
    return {
        "schema": "chipmunk.audio-multisource-public-source-plan/8",
        "prepared_sources_sha256": digest(prepared_path),
        "variants_manifest_sha256": manifest_hashes,
        "sources": [public_source(r) for r in prepared["sources"]],
        "backgrounds": [public_source(r) for r in prepared["backgrounds"]],
        "variants": variants,
        "acquisition_assets": assets,
        "exclusions": [
            {"id_sha256": hashed(row["id"]), "reason": row["reason"]} for row in prepared["exclusions"]
        ],
        "role_counts": prepared["role_counts"],
        "role_generator_counts": prepared["role_generator_counts"],
        "limitations": prepared["limitations"],
        "detector_inference_run": False,
        "fitting_started": False,
        "deployment_eligible": False,
        "data_terms": "Upstream sources retain their separate licenses/attributions, including CC-BY-SA4 for Dictation Speech-derived metadata. No raw audio, source text, customer material or pretrained weights are bundled. The project software license does not override upstream terms.",
        "reconstruction": "Use publisher revision and Parquet coordinates or archive basename/member hash/original-byte SHA to locate originals. Decode with the pinned production decoder. Every variant records its exact recipe and byte/PCM hashes; profile IDs, transcripts and local paths are intentionally omitted.",
    }


def export(root, output):
    body = catalogue(root)
    lock_path = root / "development/lock.json"
    lock = json.loads(lock_path.read_text())
    body["original_lock_sha256"] = digest(lock_path)
    body["lock"] = {
        k: lock[k]
        for k in (
            "schema",
            "variants_sha256",
            "prepared_sources_sha256",
            "reserved_confirmation_variants_sha256",
            "method",
            "implementation_sha256",
            "runtime",
            "weights",
            "serving_calibration_sha256",
            "serving_bundle",
            "planned",
            "deployment_eligible",
        )
    }
    raw = (json.dumps(body, indent=2, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(gzip.compress(raw, mtime=0))
    print(
        json.dumps(
            {
                "capsule_sha256": digest(output),
                "bytes": output.stat().st_size,
                "expanded_bytes": len(raw),
                "detector_inference_run": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.root, args.output)
