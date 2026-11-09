"""Joint genuine/clone candidate inventory, using text, provenance and headers only."""
from __future__ import annotations

import hashlib
import json
import re
import tarfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from scripts.audio_expressive_inventory_v66 import member_bytes, sha, write
from scripts.audio_genuine_sources_v67 import DESTINATION, REVISION
from scripts.audio_history_v58 import records, text_keys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
TRIAL = BASE / "audio-source-inventory-v68"


def history_seen(voice, text, history):
    keys = text_keys(text)
    if not keys:
        raise ValueError("Declared source text required")
    return {"declared_voice_seen": "voice:" + voice in history,
            "declared_text_seen": bool(keys & history)}


def parent_voice(value):
    text = str(value)
    if not re.fullmatch(r"[1-9][0-9]*", text):
        raise ValueError("Publisher-declared LibriSpeech parent identity required")
    return "librispeech:" + text


def inventory():
    TRIAL.mkdir(exist_ok=False)
    history_path = BASE / "audio-history-v58/catalogue.json"
    history = set(json.loads(history_path.read_text())["keys"])
    binding_path = BASE / "audio-history-v58/input-bindings.json"
    bindings = json.loads(binding_path.read_text())
    old_members = set()
    for relative, expected in bindings.items():
        path = BASE / relative
        if sha(path) != expected:
            raise ValueError("Historical source metadata changed")
        for _, row in records(json.loads(path.read_text())):
            for key in ("member", "archive_member"):
                if isinstance(row.get(key), str):
                    old_members.add(row[key])
    artifacts = {str(path.relative_to(ROOT)): sha(path) for path in (history_path, binding_path)}
    download_path = BASE / "audio-genuine-sources-v67/receipt.json"
    download = json.loads(download_path.read_text())
    if download["revision"] != REVISION or len(download["archives"]) != 4:
        raise ValueError("Exact complete genuine source bank required")
    artifacts[str(download_path.relative_to(ROOT))] = sha(download_path)
    genuine = []
    for entry in download["archives"]:
        path = DESTINATION / entry["name"]
        if entry["status"] != "publisher-verified" or sha(path) != entry["sha256"]:
            raise ValueError("Original genuine corpus file changed")
        reader = pq.ParquetFile(path)
        for index in range(reader.num_row_groups):
            for offset, row in enumerate(reader.read_row_group(index, columns=["id", "speaker_id", "text"]).to_pylist()):
                voice = parent_voice(row["speaker_id"])
                genuine.append({"id": "libri:" + row["id"], "canonical_speaker": voice,
                                "transcript": row["text"], "text_identity": sorted(text_keys(row["text"])),
                                "generator": "human", "label": 0, "license": "CC-BY-4.0",
                                "dataset": "openslr/librispeech_asr", "revision": REVISION,
                                "parquet": entry["name"], "source_path": entry["source_path"],
                                "parquet_sha256": entry["sha256"], "parquet_row_group": index,
                                "parquet_group_row": offset, "novelty": history_seen(voice, row["text"], history),
                                "audio_header_status": "not_inspected", "decoded_pcm_status": "not_inspected"})
    publisher_path = BASE / "audio-v4/fresh-sources/DuplexGen--duplexgen-spoken/metadata.jsonl"
    references = {row["key"]: row for line in publisher_path.read_text().splitlines()
                  if (row := json.loads(line))["scenario"] == "PLN"}
    receipt_path = BASE / "audio-v7/fresh-sources/archives/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    artifacts.update({str(path.relative_to(ROOT)): sha(path) for path in (publisher_path, receipt_path)})
    cloned, errors = [], []
    archive_bytes = 0
    for entry in receipt["archives"]:
        path = receipt_path.parent / entry["name"]
        if not re.fullmatch(r"PLN-0000[0-3]\.tar", entry["name"]):
            continue
        if entry["status"] != "verified" or not entry["publisher_sha256_verified"] or sha(path) != entry["sha256"]:
            raise ValueError("Exact publisher-verified clone source archive required")
        archive_bytes += path.stat().st_size
        artifacts[str(path.relative_to(ROOT))] = entry["sha256"]
        with tarfile.open(path, "r:") as archive:
            members = {member.name: member for member in archive.getmembers()}
            for name, member in sorted(members.items()):
                if not name.endswith("/meta.json"):
                    continue
                key = name.removesuffix("/meta.json")
                reference = references[key]
                if reference["license"] != "MIT":
                    raise ValueError("Publisher MIT PLN scenario required")
                raw = member_bytes(archive, member)
                body = json.loads(raw)
                user = body["speakers"].index("user")
                voice = parent_voice(body["user_prompt_speaker_id"])
                for ordinal, utterance in enumerate(body["speech_meta"]):
                    if utterance["speaker"] != user:
                        continue
                    audio_member = f"{key}/utterances/{ordinal:02d}.wav"
                    if audio_member not in members:
                        raise ValueError("Declared synthetic user utterance missing")
                    text = utterance["tts_text"]
                    row = {"id": "chatterbox:" + audio_member, "archive": entry["name"],
                           "archive_sha256": entry["sha256"], "member": audio_member,
                           "metadata_member": name, "metadata_sha256": hashlib.sha256(raw).hexdigest(),
                           "canonical_speaker": voice, "source_group": "duplex-dialogue:" + reference["example_id"],
                           "transcript": text, "text_identity": sorted(text_keys(text)),
                           "label": 1, "generator": "Chatterbox", "dataset": "DuplexGen/duplexgen-spoken",
                           "license": "MIT (publisher PLN scenario declaration)",
                           "novelty": {**history_seen(voice, text, history), "archive_member_seen": audio_member in old_members},
                           "decoded_pcm_status": "not_inspected"}
                    try:
                        with archive.extractfile(members[audio_member]) as stream:
                            info = sf.info(stream)
                        row.update(audio_header_status="ready", seconds=info.duration, channels=info.channels,
                                   sample_rate=info.samplerate, header_frames=info.frames)
                    except Exception as error:
                        row.update(audio_header_status="failed", error_type=type(error).__name__)
                        errors.append({"id": row["id"], "type": type(error).__name__})
                    cloned.append(row)
    if len({row["id"] for row in genuine}) != len(genuine) or len({row["id"] for row in cloned}) != len(cloned):
        raise ValueError("Unique declared candidate identities required")
    write(TRIAL / "genuine-candidates.json", genuine)
    write(TRIAL / "clone-candidates.json", cloned)
    write(TRIAL / "input-bindings.json", artifacts)
    write(TRIAL / "header-errors.json", errors)
    useful_clones = [row for row in cloned if row["audio_header_status"] == "ready" and row["seconds"] >= 4
                     and not row["novelty"]["archive_member_seen"] and not row["novelty"]["declared_text_seen"]]
    summary = {"schema": "chipmunk.audio-joint-source-candidate-inventory/68",
               "genuine_records": len(genuine), "genuine_voices": len({r["canonical_speaker"] for r in genuine}),
               "genuine_history_unseen_voices": len({r["canonical_speaker"] for r in genuine if not r["novelty"]["declared_voice_seen"]}),
               "genuine_declared_text_unseen_records": sum(not r["novelty"]["declared_text_seen"] for r in genuine),
               "clone_records": len(cloned), "clone_voices": len({r["canonical_speaker"] for r in cloned}),
               "clone_header_failures": len(errors), "clone_eligible4_metadata_candidates": len(useful_clones),
               "clone_eligible4_history_unseen_voices": len({r["canonical_speaker"] for r in useful_clones if not r["novelty"]["declared_voice_seen"]}),
               "clone_eligible4_declared_voices": len({r["canonical_speaker"] for r in useful_clones}),
               "clone_eligible12_metadata_candidates": sum(r["seconds"] >= 12 for r in useful_clones),
               "clone_archive_bytes_verified": archive_bytes,
               "genuine_catalogue_sha256": sha(TRIAL / "genuine-candidates.json"),
               "clone_catalogue_sha256": sha(TRIAL / "clone-candidates.json"),
               "input_bindings_sha256": sha(TRIAL / "input-bindings.json"), "code_sha256": sha(Path(__file__)),
               "candidate_generators": dict(Counter(r["generator"] for r in cloned)),
               "roles_assigned": False, "audio_pcm_decodes": 0, "new_model_calls": 0,
               "scope": "Candidate provenance/text and clone WAV headers only. Full selected bytes and production PCM must be verified before model calls; historical metadata novelty is not biometric or pretrained-corpus independence. Errors and short/seen candidates are retained."}
    write(TRIAL / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    inventory()
