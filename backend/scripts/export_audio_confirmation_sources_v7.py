"""Export fresh-source references and hashes, never audio or full transcripts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.prepare_audio_confirmation_v7 import DUPLEX_REVISION, digest, text_keys


def catalogue(prepared):
    sources = []
    for row in prepared["sources"]:
        keep = (
            "id",
            "role",
            "label",
            "generator",
            "dataset",
            "source_group",
            "source_path",
            "revision",
            "sha256",
            "bytes",
            "production_pcm_sha256",
            "source_prefix_pcm_sha256",
            "member",
            "meta_sha256",
            "voice_id_sha256",
            "seconds",
            "sample_rate",
            "channels",
            "accent",
            "script_type",
            "license",
            "license_url",
        )
        public = {k: row[k] for k in keep if k in row}
        public["known_text_key_sha256"] = sorted(
            hashlib.sha256(k.encode()).hexdigest() for k in text_keys(row["transcript"])
        )
        public["intended_interval_samples"] = [0, 64000]
        public["intended_sample_rate"] = 16000
        if row["generator"] == "ElevenLabs":
            public["url"] = row["asset"]["url"]
            public["publisher_sha256_verified"] = True
        else:
            name = Path(row["archive"]).name
            public["archive_name"] = name
            public["url"] = (
                f"https://huggingface.co/datasets/DuplexGen/duplexgen-spoken/resolve/{DUPLEX_REVISION}/shards/PLN/{name}"
                if row["generator"] == "Chatterbox"
                else "https://openslr.trmal.net/resources/12/" + name
            )
        if row.get("prompt_speaker_id") is not None:
            public["publisher_prompt_speaker_id_sha256"] = hashlib.sha256(
                row["prompt_speaker_id"].encode()
            ).hexdigest()
        if row["id"] in prepared["assignments"]:
            public["connected_group"] = prepared["assignments"][row["id"]]
        sources.append(public)
    background_fields = (
        "id",
        "source_id",
        "kind",
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
        "license",
        "license_url",
        "license_sha256",
        "attribution",
    )
    return {
        "schema": "chipmunk.audio-v7-fresh-source-catalogue/1",
        "foreground_plan_sha256": prepared["foreground_plan_sha256"],
        "preparation_code_sha256": prepared["preparation_code_sha256"],
        "sources": sources,
        "backgrounds": [
            {k: row[k] for k in background_fields if k in row} for row in prepared["backgrounds"]
        ],
        "groups": prepared["groups"],
        "counts": prepared["counts"],
        "detector_inference_run": False,
        "deployment_eligible": False,
        "scope": "Fixed source preparation only. No detector scores, exact variant freeze, passing confirmation or outside review. Source media must be acquired separately under their declared terms.",
        "limitations": prepared["limitations"],
    }


def export(source, output):
    body = catalogue(json.loads(source.read_text()))
    body["original_prepared_sources_sha256"] = digest(source)
    fresh = source.parent.parent
    used = {r["archive_name"] for r in body["sources"] if "archive_name" in r}
    archives = []
    for name in ("archives", "supplement"):
        receipt = json.loads((fresh / "fresh-sources" / name / "receipt.json").read_text())
        archives.extend(r for r in receipt["archives"] if r["name"] in used)
    if {r["name"] for r in archives} != used or any(r["status"] != "verified" for r in archives):
        raise ValueError("Verified archive integrity receipt required for each selected source")
    body["archive_integrity"] = archives
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "catalogue_sha256": digest(output),
                "sources": len(body["sources"]),
                "backgrounds": len(body["backgrounds"]),
                "connected_groups": len(body["groups"]),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
