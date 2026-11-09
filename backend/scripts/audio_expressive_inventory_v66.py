"""Inventory archived expressive speech using provenance, never annotation scores.

Keep global recording/text/voice novelty separate from within-study role
independence. This is preparation, not source-decode or model qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath

from scripts.audio_history_v58 import text_keys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-expressive-inventory-v66"
ARCHIVES = BASE / "audio-v8/reserved-sources/archives"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def intended_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Declared generation text required")
    return " ".join(re.sub(r"\[[^\]]*\]", " ", text).split())


def candidate(body, reference, history):
    manifest, generation = body["manifest"], body["generation"]
    if manifest["language"] != "en" or generation["model_id"] != "eleven_v4":
        raise ValueError("Exact declared English generator provenance required")
    if (generation["audio_sha256"] != reference["audio_sha256"]
            or generation["voice_id"] != reference["voice"]
            or manifest["id"] != reference["id"]):
        raise ValueError("Raw generation metadata differs from publisher index")
    text = intended_text(generation["request"]["text"])
    keys = text_keys(text)
    if not keys:
        raise ValueError("Nonempty normalized intended speech required")
    voice = "elevenlabs:" + hashlib.sha256(generation["voice_id"].encode()).hexdigest()
    return {"id": reference["id"], "sha256": reference["audio_sha256"],
            "canonical_speaker": voice, "transcript": text,
            "text_identity": sorted(keys), "declared_script_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_subset": reference["source_subset"], "challenge_group": reference["challenge_group"],
            "duration_estimate_s": reference["duration_s"],
            "novelty": {"recorded_bytes_seen": "bytes:" + reference["audio_sha256"] in history,
                        "declared_text_seen": bool(keys & history),
                        "declared_voice_seen": "voice:" + voice in history},
            "generator": "ElevenLabs-v4", "label": 1, "license": "CC-BY-4.0",
            "scope": "Declared synthetic recording and intended text, not ASR alignment, biometric identity or absence from pretraining."}


def member_bytes(tar, member):
    path = PurePosixPath(member.name)
    if (not member.isfile() or path.is_absolute() or ".." in path.parts
            or member.size <= 0 or member.size > 1_000_000):
        raise ValueError("Unsafe/oversized metadata sidecar")
    with tar.extractfile(member) as stream:
        raw = stream.read(1_000_001)
    if len(raw) != member.size:
        raise ValueError("Incomplete archived metadata")
    return raw


def inventory():
    TRIAL.mkdir(exist_ok=False)
    history_path = BASE / "audio-history-v58/catalogue.json"
    history = set(json.loads(history_path.read_text())["keys"])
    index_path = BASE / "audio-v8/source-planning/laion-index.jsonl"
    rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    references = {row["id"]: row for row in rows if row["language"] == "en"}
    if len(references) != 432:
        raise ValueError("Complete 432-recording English metadata bank required")
    receipt_path = ARCHIVES / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    archive_rows = [row for row in receipt["archives"] if row["name"].startswith("audio-")]
    if len(archive_rows) != 17:
        raise ValueError("Complete 17-archive source bank required")
    artifacts = {str(path.relative_to(ROOT)): sha(path) for path in (history_path, index_path, receipt_path)}
    result, sidecars, payloads = {}, set(), {}
    for row in archive_rows:
        path = ARCHIVES / row["name"]
        if row["status"] != "verified" or not row["publisher_sha256_verified"] or sha(path) != row["sha256"]:
            raise ValueError("Publisher-verified immutable archive required")
        artifacts[str(path.relative_to(ROOT))] = row["sha256"]
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                identifier = PurePosixPath(member.name).stem
                if identifier not in references:
                    continue
                if member.name.endswith(".mp3"):
                    if identifier in payloads:
                        raise ValueError("Duplicate declared audio member")
                    payloads[identifier] = {"archive": row["name"], "member": member.name, "bytes": member.size}
                elif member.name.endswith(".json"):
                    if identifier in sidecars:
                        raise ValueError("Duplicate generation sidecar")
                    raw = member_bytes(archive, member)
                    value = candidate(json.loads(raw), references[identifier], history)
                    value.update(metadata_archive=row["name"], metadata_member=member.name,
                                 metadata_sha256=hashlib.sha256(raw).hexdigest())
                    result[identifier] = value
                    sidecars.add(identifier)
    if set(result) != set(references) or set(payloads) != set(references):
        raise ValueError("Every English source and sidecar must be retained")
    for identifier, value in result.items():
        value.update(payloads[identifier])
    candidates = sorted(result.values(), key=lambda row: row["id"])
    write(TRIAL / "candidates.json", candidates)
    write(TRIAL / "input-bindings.json", artifacts)
    strata = Counter(tuple(row["novelty"][key] for key in ("recorded_bytes_seen", "declared_text_seen", "declared_voice_seen"))
                     for row in candidates)
    summary = {"schema": "chipmunk.audio-expressive-metadata-inventory/66", "recordings": len(candidates),
               "declared_voices": len({row["canonical_speaker"] for row in candidates}),
               "declared_scripts": len({tuple(row["text_identity"]) for row in candidates}),
               "byte_unseen_candidates": sum(not row["novelty"]["recorded_bytes_seen"] for row in candidates),
               "byte_and_text_unseen_candidates": sum(not row["novelty"]["recorded_bytes_seen"] and not row["novelty"]["declared_text_seen"] for row in candidates),
               "byte_text_voice_unseen_candidates": sum(not any(row["novelty"].values()) for row in candidates),
               "novelty_strata": [{"recorded_bytes_seen": key[0], "declared_text_seen": key[1],
                                    "declared_voice_seen": key[2], "recordings": value} for key, value in sorted(strata.items())],
               "archives_verified": len(archive_rows), "archive_bytes_verified": sum(row["bytes"] for row in archive_rows),
               "candidate_catalogue_sha256": sha(TRIAL / "candidates.json"),
               "input_bindings_sha256": sha(TRIAL / "input-bindings.json"), "code_sha256": sha(Path(__file__)),
               "annotation_scores_used": False, "roles_assigned": False, "audio_decodes": 0,
               "new_model_calls": 0, "scope": "Metadata candidates only. Byte novelty needs decoded-PCM verification. Historical voice/text reuse is reported, not silently relabelled as unseen. No within-study role, source or calibrated-confidence qualification yet."}
    write(TRIAL / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    inventory()
