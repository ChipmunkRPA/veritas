"""Canonical source identities across genuine speech and cloned-voice corpora.

Publisher code maps DuplexGen user_prompt_speaker_id to the original LibriSpeech
speaker ID. Distinct dataset prefixes must not split that known parent identity.
No detector output, biometric model or candidate parameter is used here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hashed(text):
    return hashlib.sha256(text.encode()).hexdigest()


def text_keys(text):
    words = re.findall(r"\w+", (text or "").casefold())
    return {"full:" + " ".join(words), "prefix8:" + " ".join(words[:8])} if words else set()


def speaker_key(row):
    if row.get("kind") in ("noise", "music"):
        return None
    dataset = row["dataset"].casefold()
    if dataset == "duplexgen/duplexgen-spoken":
        identity = str(row.get("prompt_speaker_id", ""))
        if not re.fullmatch(r"[0-9]+", identity):
            raise ValueError("Published clone prompt requires its numeric LibriSpeech speaker ID")
        return "librispeech:" + identity
    if "librispeech" in dataset:
        for key in ("speaker_id", "speaker", "cluster_id", "voice_key", "source_group"):
            value = str(row.get(key, ""))
            match = re.fullmatch(r"(?:(?:librispeech|libri-speaker|libri):)?([0-9]+)", value)
            if match:
                return "librispeech:" + match[1]
        match = re.search(r"(?:^|:)([0-9]+)-[0-9]+-[0-9]+$", row["id"])
        if match:
            return "librispeech:" + match[1]
        raise ValueError("Known genuine LibriSpeech source lacks its speaker identity")
    for key in ("provider_voice_sha256", "voice_id_sha256"):
        if row.get(key):
            if not re.fullmatch(r"[0-9a-f]{64}", row[key]):
                raise ValueError("Provider-profile identity must be its documented SHA256")
            return "elevenlabs:" + row[key]
    if str(row.get("generator") or "").casefold().startswith("kokoro"):
        for key in ("tts_voice", "voice_key", "speaker", "cluster_id"):
            value = str(row.get(key, ""))
            match = re.fullmatch(r"(?:(?:kokoro|kokoro-82m):)?([a-z]{2}_[a-z0-9_]+)", value.casefold())
            if match:
                return "kokoro-82m:" + match[1]
        raise ValueError("Kokoro source lacks its known voice-profile identity")
    for key in ("cluster_id", "voice_key", "speaker"):
        value = str(row.get(key, ""))
        match = re.fullmatch(r"(?:vctk:)?(p[0-9]{3})", value.casefold())
        if match:
            return "vctk:" + match[1]
    for key in ("cluster_id", "voice_key", "speaker"):
        if row.get(key) is not None:
            return "declared:" + dataset + ":" + str(row[key])
    return None


def connection_keys(row):
    keys = {"bytes:" + row["sha256"]} | text_keys(row.get("transcript"))
    voice = speaker_key(row)
    if voice:
        keys.add("voice:" + voice)
    source = row.get("source_key") or row.get("source_group")
    if source:
        keys.add("source:" + row["dataset"].casefold() + ":" + source)
    for key in ("production_pcm_sha256", "prefix_pcm_sha256"):
        if row.get(key):
            keys.add(key + ":" + row[key])
    if row.get("prompt_index") is not None and row["dataset"] == "isjwdu/DFADD" and row["label"] == 1:
        keys.add("dfadd-prompt:" + str(row["prompt_index"]))
    return keys


def connected_groups(rows):
    parent = {r["id"]: r["id"] for r in rows}
    if len(parent) != len(rows):
        raise ValueError("Duplicate source IDs cannot form independent groups")

    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    owners = {}
    for row in rows:
        for key in sorted(connection_keys(row)):
            if key in owners:
                parent[find(row["id"])] = find(owners[key])
            else:
                owners[key] = row["id"]
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row)
    result = []
    for members in groups.values():
        identifiers = sorted(r["id"] for r in members)
        result.append(
            {
                "group": "connected:" + hashed(json.dumps(identifiers, separators=(",", ":"))),
                "sources": identifiers,
                "stratum": "+".join(sorted({str(r["label"]) + ":" + r["generator"] for r in members})),
                "class_counts": {str(c): sum(r["label"] == c for r in members) for c in (0, 1)},
            }
        )
    return sorted(result, key=lambda r: r["group"])


def role_violations(rows):
    voices = defaultdict(list)
    for row in rows:
        voice = speaker_key(row)
        if voice:
            voices[voice].append(row)
    records = []
    for voice, members in sorted(voices.items()):
        roles = {r["role"] for r in members}
        mixed_use = any(r.get("kind") == "speech" for r in members) and any(
            r.get("kind") != "speech" for r in members
        )
        split_group = len({r.get("group") for r in members if r.get("group")}) > 1
        if len(roles) > 1 or mixed_use or split_group:
            records.append(
                {
                    "speaker_fingerprint": hashed(voice),
                    "roles": sorted(roles),
                    "cross_role": len(roles) > 1,
                    "foreground_background_shared": mixed_use,
                    "known_speaker_split_across_groups": split_group,
                    "sources": [
                        {
                            "source_sha256": r["sha256"],
                            "role": r["role"],
                            "kind": r.get("kind", "foreground"),
                            "dataset": r["dataset"],
                            "label": r.get("label"),
                            "generator": r.get("generator"),
                        }
                        for r in members
                    ],
                }
            )
    return records


def prior_sources(data):
    rows, pins = [], {}
    for version in ("audio", "audio-v2", "audio-v3", "audio-v7/confirmation"):
        path = (
            data
            / version
            / ("prepared-sources.json" if version.endswith("confirmation") else "manifest.json")
        )
        body = json.loads(path.read_text())
        pins[version] = digest(path)
        for row in body.get("base_files", body.get("sources", [])) + body["backgrounds"]:
            rows.append({**row, "history_version": version, "source_path": str(data / version / row["path"])})
    return rows, pins


def audio_fingerprints(row):
    import numpy as np

    from app.detectors.audio.decode import decode_audio

    path = Path(row["source_path"])
    if digest(path) != row["sha256"]:
        raise ValueError("Historical original source bytes changed")
    audio = decode_audio(path, source_sha256=row["sha256"])
    prefixes = {}
    for samples in range(32000, 64001, 4000):
        if len(audio.pcm) >= samples:
            prefix = audio.pcm[:samples]
            if float(np.sqrt(np.mean(prefix.astype(float) ** 2))) >= 1e-4:
                prefixes[str(samples)] = hashlib.sha256(prefix.astype("<f4").tobytes()).hexdigest()
    return {"pcm_sha256": audio.mapping["pcm_sha256"], "prefixes": prefixes, "samples": len(audio.pcm)}


def audit(data, root, output):
    if output.exists():
        raise ValueError("Preserve existing identity audit")
    source_path = root / "prepared-sources.json"
    prepared = json.loads(source_path.read_text())
    current = prepared["sources"] + prepared["backgrounds"]
    prior, pins = prior_sources(data)
    proof_path = root / "source-planning/duplex-primary-identity-provenance.json"
    proof = json.loads(proof_path.read_text())
    code_path = root / "source-planning/duplex-primary-identity-code.py"
    raw = code_path.read_bytes()
    if (
        digest(code_path) != proof["sha256"]
        or hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != proof["git_blob_sha1"]
    ):
        raise ValueError("Publisher identity-mapping code differs from captured Git blob")
    if (
        'total_speech_meta["user_prompt_speaker_id"] = libri_spk' not in raw.decode()
        or "def list_librispeech_speakers(" not in raw.decode()
    ):
        raise ValueError("Publisher mapping evidence missing; do not invent cross-corpus joins")
    unique = {r["sha256"]: r for r in prior + current}
    fingerprints, failures = {}, []

    def one(item):
        sha, row = item
        try:
            return sha, audio_fingerprints(row), None
        except Exception as error:
            return sha, None, type(error).__name__

    with ThreadPoolExecutor(4) as pool:
        for index, (sha, record, error) in enumerate(pool.map(one, sorted(unique.items())), 1):
            if error:
                failures.append({"source_sha256": sha, "error_type": error})
            else:
                fingerprints[sha] = record
            if index % 100 == 0 or index == len(unique):
                print(
                    json.dumps(
                        {"originals_audited": index, "planned": len(unique), "failures": len(failures)}
                    ),
                    flush=True,
                )
    lookup = defaultdict(set)
    voice_history = defaultdict(list)
    for row in prior:
        for key in text_keys(row.get("transcript")) | {"bytes:" + row["sha256"]}:
            lookup[key].add(row["sha256"])
        record = fingerprints.get(row["sha256"])
        if record:
            lookup["pcm:" + record["pcm_sha256"]].add(row["sha256"])
            for length, value in record["prefixes"].items():
                lookup["prefix:" + length + ":" + value].add(row["sha256"])
        voice = speaker_key(row)
        if voice:
            voice_history[voice].append(row)
    lj = set()
    with (data / "audio-v3/research/lj-speech-original-metadata.csv").open() as stream:
        for row in csv.reader(stream, delimiter="|"):
            for text in row[1:]:
                lj.update(text_keys(text))
    reserved = [r for r in prepared["sources"] if r["role"] == "confirmation"] + prepared["backgrounds"]
    reuse, voices, prompt_bank = [], [], []
    for row in reserved:
        keys = text_keys(row.get("transcript")) | {"bytes:" + row["sha256"]}
        record = fingerprints.get(row["sha256"])
        if record:
            keys.add("pcm:" + record["pcm_sha256"])
            keys.update("prefix:" + length + ":" + value for length, value in record["prefixes"].items())
        matches = {key: sorted(lookup[key]) for key in sorted(keys) if key in lookup}
        if matches:
            reuse.append(
                {
                    "source_sha256": row["sha256"],
                    "role": row["role"],
                    "kind": row.get("kind", "foreground"),
                    "matches": [
                        {
                            "connection_type": key.split(":", 1)[0],
                            "connection_fingerprint": hashed(key),
                            "prior_sources": refs,
                        }
                        for key, refs in matches.items()
                    ],
                }
            )
        voice = speaker_key(row)
        if voice in voice_history:
            voices.append(
                {
                    "source_sha256": row["sha256"],
                    "role": row["role"],
                    "kind": row.get("kind", "foreground"),
                    "generator": row.get("generator"),
                    "speaker_fingerprint": hashed(voice),
                    "prior_sources": sorted({r["sha256"] for r in voice_history[voice]}),
                    "prior_versions": sorted({r["history_version"] for r in voice_history[voice]}),
                }
            )
        if text_keys(row.get("transcript")) & lj:
            prompt_bank.append({"source_sha256": row["sha256"]})
    violations = role_violations(current)
    extraction = root / "development/representations.jsonl"
    with extraction.open() as stream:
        next(stream)
        terminal = [json.loads(line) for line in stream]
    result = {
        "schema": "chipmunk.audio-cross-corpus-identity-audit/9",
        "code_sha256": digest(Path(__file__)),
        "prepared_sources_sha256": digest(source_path),
        "historical_manifest_pins": pins,
        "historical_references": len(prior),
        "historical_unique_originals": len({r["sha256"] for r in prior}),
        "all_unique_originals_audited": len(unique),
        "publisher_mapping": proof,
        "source_decode_failures": failures,
        "reserved_recording_or_text_reuse": reuse,
        "reserved_declared_prompt_bank_connections": prompt_bank,
        "reserved_voices_seen_in_history": voices,
        "current_identity_violations": violations,
        "current_cross_role_voice_groups": sum(r["cross_role"] for r in violations),
        "current_foreground_background_voice_groups": sum(
            r["foreground_background_shared"] for r in violations
        ),
        "v8_voice_separation_valid": not violations and not failures,
        "reserved_recording_novelty_valid": not reuse and not prompt_bank and not failures,
        "v8_fitting_started": False,
        "v8_extraction_stopped": True,
        "retained_extraction_records": len(terminal),
        "retained_extraction_sha256": digest(extraction),
        "retained_extraction_status": dict(Counter(r["status"] for r in terminal)),
        "deployment_eligible": False,
        "scope": "Audits original recordings in the audio/v2/v3/v7 evaluation manifests, not all pretraining or every runtime fixture. Known publisher identities are conservatively joined; no biometric inference or detector responses select sources. Voice-history exposure differs from exact recording/script reuse. Original v8 inputs and completed feature records remain preserved; do not fit its invalid roles.",
    }
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "audit_sha256": digest(output),
                "historical_references": len(prior),
                "all_originals_audited": len(unique),
                "decode_failures": len(failures),
                "recording_or_text_reuse": len(reuse),
                "prompt_bank_connections": len(prompt_bank),
                "voices_seen_in_history": dict(Counter(r["generator"] or r["kind"] for r in voices)),
                "cross_role_voice_groups": result["current_cross_role_voice_groups"],
                "foreground_background_voice_groups": result["current_foreground_background_voice_groups"],
                "v8_voice_separation_valid": result["v8_voice_separation_valid"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    audit(args.data, args.root, args.output)
