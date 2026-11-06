"""Materialize the sixty fixed AI confirmation sources; never run a classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path

from app.detectors.audio.decode import decode_audio
from scripts.plan_audio_multisource_v8 import DICTATION_REVISION, LAION_REVISION, digest
from scripts.prepare_audio_confirmation_inputs_v7 import store
from scripts.prepare_audio_confirmation_v7 import tar_bytes, text_keys
from scripts.prepare_audio_guard_v7 import write


def spoken_request(text):
    # Provider bracketed delivery controls are not assumed to be spoken words.
    # This is an intended-text connection only, not an ASR transcript/alignment.
    return " ".join(re.sub(r"\[[^\]]*\]", " ", text).split())


def materialize(root, output):
    import pyarrow.parquet as pq

    plan_path = root / "source-planning/data-plan.json"
    plan = json.loads(plan_path.read_text())
    sources_root = root / "reserved-sources"
    archives = sources_root / "archives"
    receipt = json.loads((archives / "receipt.json").read_text())
    if len(receipt["archives"]) != 18 or any(
        r["status"] != "verified" or not r["publisher_sha256_verified"] for r in receipt["archives"]
    ):
        raise ValueError("Every reserved acquisition asset requires its publisher SHA256")
    for row in receipt["archives"]:
        if digest(archives / row["name"]) != row["sha256"]:
            raise ValueError("Reserved publisher asset changed")
    chosen = {r["id"]: r for r in plan["confirmation_laion"]}
    manifests, originals = {}, {}
    for path in sorted(archives.glob("audio-*.tar")):
        with tarfile.open(path, "r:") as tar:
            for member in tar:
                identifier = Path(member.name).stem
                if identifier not in chosen:
                    continue
                if member.name.endswith(".json"):
                    manifests[identifier] = {
                        "body": json.loads(tar_bytes(tar, member)),
                        "archive": path.name,
                        "member": member.name,
                        "sha256": hashlib.sha256(tar_bytes(tar, member)).hexdigest(),
                    }
                elif member.name.endswith(".mp3"):
                    originals[identifier] = {
                        **store(sources_root, tar_bytes(tar, member)),
                        "archive_name": path.name,
                        "archive_member": member.name,
                    }
    if set(manifests) != set(chosen) or set(originals) != set(chosen):
        raise ValueError("Every fixed provider profile/recording must be present")
    rows = []
    for identifier, expected in chosen.items():
        record = manifests[identifier]["body"]
        manifest, generation = record["manifest"], record["generation"]
        if (
            manifest["language"] != "en"
            or generation["voice_id"] != expected["voice"]
            or generation["model_id"] != "eleven_v4"
            or generation["audio_sha256"] != expected["audio_sha256"]
            or originals[identifier]["sha256"] != expected["audio_sha256"]
        ):
            raise ValueError("Raw provider generation differs from reserved identity/label/hash")
        intended = spoken_request(generation["request"]["text"])
        rows.append(
            {
                **originals[identifier],
                "id": "laion:" + identifier,
                "dataset": "laion/elevenlabs-v4-five-language-voice-acting",
                "revision": LAION_REVISION,
                "generator": "ElevenLabs-v4",
                "label": 1,
                "role": "confirmation",
                "provider_voice_sha256": hashlib.sha256(expected["voice"].encode()).hexdigest(),
                "speaker_alias": expected["speaker_alias"],
                "source_group": expected["challenge_group"],
                "voice_key": "elevenlabs-profile:" + hashlib.sha256(expected["voice"].encode()).hexdigest(),
                "transcript": intended,
                "transcript_kind": "Intended provider text, bracket controls removed; actual words/alignment unverified",
                "generation_metadata_sha256": manifests[identifier]["sha256"],
                "license": "CC-BY-4.0",
                "license_url": f"https://huggingface.co/datasets/laion/elevenlabs-v4-five-language-voice-acting/resolve/{LAION_REVISION}/README.md",
                "url": f"https://huggingface.co/datasets/laion/elevenlabs-v4-five-language-voice-acting/resolve/{LAION_REVISION}/raw/{originals[identifier]['archive_name']}",
            }
        )
    frame = pq.read_table(archives / "kokoro-directed-test.parquet").to_pylist()
    for expected in plan["confirmation_kokoro"]:
        row = frame[expected["parquet_row"]]
        if any(
            row[k] != expected[k]
            for k in ("id", "tts_voice", "tts_engine", "language", "transcript", "license")
        ):
            raise ValueError("Kokoro fixed row/voice/content/permission differs")
        rows.append(
            {
                **store(sources_root, row["audio"]["bytes"]),
                "id": "dictation:" + row["id"],
                "dataset": "ductai199x/dictation-speech",
                "revision": DICTATION_REVISION,
                "generator": "Kokoro-82M",
                "label": 1,
                "role": "confirmation",
                "voice_key": "kokoro:" + row["tts_voice"],
                "source_group": "dictation-passage:" + row["rendition_id"].split("-", 1)[0],
                "transcript": row["transcript"],
                "transcript_kind": "Publisher spoken rendition; alignment unverified",
                "parquet": "data/kokoro-directed/test-00000-of-00001.parquet",
                "parquet_row": expected["parquet_row"],
                "license": "CC-BY-SA-4.0",
                "passage_license": row["license"],
                "passage_source_url": row["source_url"],
                "license_url": f"https://huggingface.co/datasets/ductai199x/dictation-speech/resolve/{DICTATION_REVISION}/README.md",
                "url": f"https://huggingface.co/datasets/ductai199x/dictation-speech/resolve/{DICTATION_REVISION}/data/kokoro-directed/test-00000-of-00001.parquet",
            }
        )
    by_archive = {}
    for row in plan["confirmation_chatterbox"]:
        by_archive.setdefault(row["archive"], []).append(row)
    for name, selected in sorted(by_archive.items()):
        with tarfile.open(name, "r:") as tar:
            for row in selected:
                ref = store(sources_root, tar_bytes(tar, tar.getmember(row["member"])))
                rows.append(
                    {
                        **row,
                        **ref,
                        "role": "confirmation",
                        "voice_key": "duplex-prompt:" + row["prompt_speaker_id"],
                        "archive_name": Path(name).name,
                        "archive_member": row["member"],
                    }
                )
    if len(rows) != 60 or len({r["sha256"] for r in rows}) != 60:
        raise ValueError("Every sixty distinct original source recordings required")
    issues, fingerprints = [], set()
    for row in rows:
        decoded = decode_audio(sources_root / row["path"], source_sha256=row["sha256"])
        if len(decoded.pcm) < 64000:
            issues.append({"id": row["id"], "issue": "Selected source shorter than four seconds"})
        row["production_pcm_sha256"] = decoded.mapping["pcm_sha256"]
        row["full_samples"] = len(decoded.pcm)
        row["prefix_pcm_sha256"] = hashlib.sha256(decoded.pcm[:64000].astype("<f4").tobytes()).hexdigest()
        row["source_production_input"] = decoded.mapping
        if row["prefix_pcm_sha256"] in fingerprints:
            issues.append({"id": row["id"], "issue": "Duplicate selected decoded prefix"})
        fingerprints.add(row["prefix_pcm_sha256"])
    # Preserve explicit metadata connections for the final role/eligibility
    # amendment; never silently remove a reserved source after observing scores.
    prior = {}
    for row in plan["development_sources"]:
        for key in text_keys(row.get("transcript") or ""):
            prior.setdefault(key, set()).add(row["group"])
    connections = []
    for row in rows:
        overlaps = set().union(*(prior.get(key, set()) for key in text_keys(row["transcript"])))
        if overlaps:
            connections.append({"reserved_id": row["id"], "connected_consumed_groups": sorted(overlaps)})
    result = {
        "schema": "chipmunk.audio-v8-reserved-ai-materialization/1",
        "data_plan_sha256": digest(plan_path),
        "acquisition_receipt_sha256": digest(archives / "receipt.json"),
        "code_sha256": digest(Path(__file__)),
        "sources": rows,
        "eligibility_issues": issues,
        "consumed_prompt_connections": connections,
        "neural_inference_run": False,
        "fitting_started": False,
        "deployment_eligible": False,
        "scope": "Fixed raw source/decoded-prefix preparation only; resolve eligibility and role connections before final freeze or fitting. No deepfake scores or learned annotation targets used.",
    }
    write(output, result)
    print(
        json.dumps(
            {
                "sources": len(rows),
                "full4_sources": sum(r["full_samples"] >= 64000 for r in rows),
                "eligibility_issues": issues,
                "consumed_prompt_connections": len(connections),
                "materialization_sha256": digest(output),
                "neural_inference_run": False,
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
