"""Freeze and qualify existing publisher-linked same-speaker clone sources."""

from __future__ import annotations

import argparse
import io
import json
import re
import wave
from collections import Counter, defaultdict
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from app.detectors.audio.decode import decode_audio, source_ranges, wav_bytes
from scripts import audio_short_edit_materialize_v169c as source_reader
from scripts import audio_short_edit_sources_v169 as sources

BASE, ROOT, RATE = sources.BASE, sources.ROOT, 16_000
sha, read, write, require = source_reader.sha, source_reader.read, source_reader.write, source_reader.require
digest, pcm_digest = source_reader.digest, source_reader.pcm_digest
SCHEMA = "chipmunk.audio-same-speaker-clone-sources/1"
CLONES = "audio-source-inventory-v68/clone-candidates.json"
PARQUET = "audio-v8/reserved-sources/libri-parquet"
ARCHIVES = "audio-v7/fresh-sources/archives"
ROLES = {"selection": (6, 10, ("ES2004a", "IS1009a")), "confirmation": (8, 16, ("ES2010d", "IS1007a"))}
HISTORY = (
    "audio-source-qualification-v72/manifest.json",
    "audio-confidence-mixtures-v82/variants.json",
    "audio-spectra-long-room-final-2026-11-12/roster.json",
    "audio-enrolled-target-2026-11-12/roster.json",
)
TAR_LICENSE = "MIT (publisher PLN scenario declaration)"
CUTS = {"host": 8*RATE, "enrollment": 4*RATE, "clone": 3*RATE}
METHOD = {
    "identity": "Compare publisher PLN isolated user-turn sidecar's LibriSpeech user_prompt_speaker_id against exact original LibriSpeech Parquet speaker ID and utterance path for both genuine sources. These are declared corpus identities. Original synthesis prompt paths are stripped.",
    "order": "Before reading any selected audio, salt-order canonical identities and choose one unique dialogue/script clone per identity from metadata. Freeze up to ten point and sixteen held candidates, alternating source-role allocation while both have space. Use the first six/eight source-qualified speakers, respectively. Within each identity retain ordered metadata candidates, pair different chapters, and reject any known exact prior source/script or byte/PCM match. Previously known speaker or dialogue context can enter a role and is disclosed. Never read detector scores.",
    "reference": "Bind four exact seconds from an original whose full duration is at least four and at most five seconds and which differs from the host utterance and chapter. The public renderer requires source prompts longer than five seconds. The exact archived synthesis prompt cannot be verified independently.",
    "clocks": "Retain first eight seconds of host, first four seconds of enrollment and first three seconds of generated donor. The donor is the sidecar-declared native user channel only, at its original clock before normalization. Original normalized decode and WAV round trip are mono 16 kHz s16 with a single continuous source range. Six seconds can be cut from host after a two-second same-host genuine donor reserve. Future clip mix and edit ranges require a separate freeze.",
    "scope": "One documented synthesis family and no detector or acoustic measurements. Room names allocate four separate meeting cohorts prospectively; this stage does not read or qualify any room audio or claim full-panel independence from model training.",
}


def rank(value):
    return digest(f"{SCHEMA}:{value}".encode())


def links(body, ids, keys, voices):
    if isinstance(body, list):
        for item in body:
            links(item, ids, keys, voices)
    if not isinstance(body, dict):
        return
    for name, value in body.items():
        if isinstance(value, dict | list):
            links(value, ids, keys, voices)
        elif isinstance(value, str):
            if name in {"transcript", "declared_text"}:
                keys.update(sources.text_keys(value))
            if name in {"id", "source_id", "candidate"}:
                ids.add(value)
            if name in {"voice", "canonical_speaker"} and re.fullmatch(r"librispeech:\d+", value):
                voices.add(value)
            if re.fullmatch(r"[a-f0-9]{64}", value):
                if "pcm" in name and name.endswith("sha256"):
                    keys.add(f"pcm:{value}")
                if name in {"original_sha256", "raw_sha256", "source_sha256", "wav_sha256", "host_sha256", "ordinary_sha256",
                            "donor_sha256", "generated_sha256", "target_reference", "sham_reference"}:
                    keys.add(f"bytes:{value}")


def history(meta):
    keys, ids, contexts = sources.exclude_history(meta)
    voices = {value.removeprefix("voice:") for value in keys if value.startswith("voice:")}
    for i, relative in enumerate(HISTORY):
        links(meta.source(f"prior_{i}", relative), ids, keys, voices)
    prior = meta.read("v169c_summary", ROOT / "docs/evaluation-panels/audio-short-edit-materialization-complete-v169c.json")
    for role in ("development", "holdout"):
        name = f"v169c_{role}"
        roster = meta.read(name, source_reader.MEDIA / role / "private-manifest.json")
        require(meta.bindings[name] == prior["roles"][role]["private_manifest_sha256"], "Prior qualified sources changed")
        links(roster, ids, keys, voices)
    return ids, contexts, keys, voices


def genuine_candidates(files, speakers, ids, history_keys):
    values = defaultdict(lambda: defaultdict(list))
    for entry in files.values():
        with pq.ParquetFile(entry["file"]) as parquet:
            for group in range(parquet.num_row_groups):
                fields = ["id", "speaker_id", "chapter_id", "text"]
                rows = parquet.read_row_group(group, columns=fields, use_threads=False).to_pylist()
                for offset, row in enumerate(rows):
                    voice = f"librispeech:{row['speaker_id']}"
                    identity = f"libri:{row['id']}"
                    if voice not in speakers or identity in ids or row["id"] in ids:
                        continue
                    words = len(row["text"].split())
                    purpose = "enrollment" if 7 <= words <= 13 else "host" if 19 <= words <= 42 else None
                    if purpose is None:
                        continue
                    candidate = sources.candidate("genuine", identity, voice, row["text"],
                                                  sources.parquet_locator(entry, group, offset), "CC-BY-4.0",
                                                  chapter=row["chapter_id"])
                    if not set(candidate["texts"]) & history_keys:
                        values[voice][purpose].append(candidate)
    return values


def freeze_plan():
    meta = sources.Metadata()
    ids, contexts, seen, voices = history(meta)
    clone_files = sources.assets(meta, "clone", ARCHIVES)
    parquets = sources.assets(meta, "libri100", PARQUET, "audio-v8/source-planning/libri-acquisition-plan.json")
    raw_clone = meta.source("clone_candidates", CLONES)
    meta.source("expressive_candidates", "audio-expressive-inventory-v66/candidates.json")
    meta.source("stock_expressive_receipt", "audio-v8/reserved-sources/archives/receipt.json")
    clones = defaultdict(list)
    for row in raw_clone:
        if (row["archive"] not in clone_files or row["archive"][:4] != "PLN-" or row["audio_header_status"] != "ready"
                or not 3 <= row["seconds"] <= 12 or row["id"] in ids
                or row["novelty"]["archive_member_seen"] or set(row["text_identity"]) & seen):
            continue
        entry = clone_files[row["archive"]]
        require(row["archive_sha256"] == entry["sha256"] and row["license"] == TAR_LICENSE
                and row["generator"] == "Chatterbox" and row["member"].startswith("PLN/"), "Source lane terms changed")
        voice = row["canonical_speaker"]
        clones[voice].append(sources.candidate("clone", row["id"], voice, row["transcript"],
                                              sources.tar_locator(entry, row), TAR_LICENSE, context=row["source_group"], donor=voice,
                                              context_known_in_prior_project=row["source_group"] in contexts,
                                              native=[row["channels"], row["sample_rate"], row["header_frames"]]))
    genuine = genuine_candidates(parquets, set(clones), ids, seen)
    slots, used_dialogues, used_texts = [], set(), set()
    capacities = Counter()
    role_order = tuple(ROLES)
    for voice in sorted(genuine, key=rank):
        if all(capacities[role] == ROLES[role][1] for role in ROLES):
            break
        arms = genuine[voice]
        candidates = sorted(clones[voice], key=lambda row: rank(row["id"]))
        donor = next((row for row in candidates if row["context"] not in used_dialogues and not set(row["texts"]) & used_texts), None)
        if donor is None or not any(h["chapter"] != e["chapter"] for h in arms["host"] for e in arms["enrollment"]):
            continue
        next_role = role_order[len(slots) % len(role_order)]
        role = next_role if capacities[next_role] < ROLES[next_role][1] else next(r for r in ROLES if capacities[r] < ROLES[r][1])
        used_dialogues.add(donor["context"])
        used_texts.update(donor["texts"])
        slots.append({"slot": f"{role}-{capacities[role]:02d}", "role": role, "voice": voice,
                      "voice_known_in_prior_project": voice in voices, "clone": donor,
                      "host": sorted(arms["host"], key=lambda row: rank(row["id"]))[:12],
                      "enrollment": sorted(arms["enrollment"], key=lambda row: rank(row["id"]))[:20]})
        capacities[role] += 1
    require(capacities == {role: cap for role, (_, cap, _) in ROLES.items()}, f"Insufficient metadata groups for frozen roles: {capacities}")
    return {"schema": SCHEMA, "method": METHOD, "metadata": meta.bindings, "slots": slots,
            "historical_key_count": len(seen), "runtime_sha256": sha(Path(__file__))}, seen


def genuine_frames(raw):
    require(raw[:8] == b"fLaC\x00\x00\x00\x22", "Expected publisher LibriSpeech FLAC STREAMINFO")
    info = int.from_bytes(raw[18:26], "big")
    rate, channels = info >> 44, ((info >> 41) & 7)+1
    require((channels, rate) == (1, RATE), "Genuine native source clock differs")
    return info & ((1 << 36)-1)


def clone_channel(raw, row, loader):
    locator = row["locator"]
    sidecar = loader.clone[row["id"]]
    metadata = loader.member(locator["file"], locator["archive_sha256"], sidecar["metadata_member"])
    speakers = json.loads(metadata)["speakers"]
    require(set(speakers) == {"assistant", "user"} and len(speakers) == 2, "PLN speaker channels differ")
    user_channel = speakers.index("user")
    with wave.open(io.BytesIO(raw), "rb") as file:
        native = [file.getnchannels(), file.getframerate(), file.getnframes()]
        require(file.getsampwidth() == 2 and native == row["native"] and native[:2] == [2, 24_000], "PLN WAV header differs")
        frames = file.readframes(native[2])
    require(len(frames) == native[2]*4, "PLN source WAV is truncated")
    channels = np.frombuffer(frames, "<i2").reshape(-1, 2)
    require(np.any(channels[:, user_channel]), "PLN user channel is empty")
    output = io.BytesIO()
    with wave.open(output, "wb") as file:
        file.setparams((1, 2, native[1], native[2], "NONE", "not compressed"))
        file.writeframes(channels[:, user_channel].tobytes())
    details = {"original_channels": 2, "user_channel": user_channel,
               "other_channel_active_frames": int(np.count_nonzero(channels[:, 1-user_channel]))}
    return Fraction(native[2]*RATE, native[1]), output.getvalue(), details


def exact(row, raw, arm, temp, seen, loader):
    if arm == "clone":
        full, payload, native = clone_channel(raw, row, loader)
    else:
        full, payload, native = genuine_frames(raw), raw, {"original_channels": 1}
    count = CUTS[arm]
    if full < count or (arm == "enrollment" and full > 5*RATE):
        return None, "duration"
    raw_hash = digest(raw)
    if f"bytes:{raw_hash}" in seen:
        return None, "exact_prior_source"
    source_path = temp / ("source.wav" if arm == "clone" else "source.flac")
    source_path.write_bytes(payload)
    decoder_hash = digest(payload)
    audio = decode_audio(source_path, source_sha256=decoder_hash)
    pcm, mapping = audio.pcm, audio.mapping
    rate = 24_000 if arm == "clone" else RATE
    require((mapping["source_sample_rate"], mapping["source_channels"]) == (rate, 1)
            and abs(full-len(pcm)) <= 1 and mapping["pcm_sha256"] == pcm_digest(pcm), "Native and decoded source clocks differ")
    cut = pcm[:count]
    ranges = source_ranges(mapping, 0, count)
    expected = {"start_sample": 0, "end_sample": count, "source_start_pts": 0, "source_end_pts": count, "time_base": "1/16000"}
    require(ranges == [expected], "Exact source range is discontinuous or shifted")
    if not active(cut):
        return None, "insufficient_speech_energy"
    if arm == "host" and not active(cut[:2*RATE]):
        return None, "inactive_same_host_reserve"
    out = wav_bytes(cut)
    fingerprints = {"original_sha256": raw_hash, "decoder_input_sha256": decoder_hash,
                    "full_pcm_sha256": mapping["pcm_sha256"], "wav_sha256": digest(out),
                    "prefix_pcm_sha256": {str(n): pcm_digest(pcm[:n]) for n in sorted({47_648, 48_000, 64_000, count}) if n <= len(pcm)}}
    keys = {f"bytes:{raw_hash}", f"bytes:{decoder_hash}", f"bytes:{fingerprints['wav_sha256']}", f"pcm:{fingerprints['full_pcm_sha256']}",
            *(f"pcm:{pin}" for pin in fingerprints["prefix_pcm_sha256"].values()), *row["texts"]}
    if keys & seen:
        return None, "prior_pcm_or_text"
    proof_path = temp / "cut.wav"
    proof_path.write_bytes(out)
    proof = decode_audio(proof_path, source_sha256=fingerprints["wav_sha256"])
    require(np.array_equal(proof.pcm, cut) and source_ranges(proof.mapping, 0, count) == [expected], "WAV cut changed PCM or clock")
    return {"id": row["id"], "voice": row["voice"], "source_samples": len(pcm), "sample_rate": RATE,
            "chapter": row.get("chapter"), "license": row["license"], "source_range": expected, **native, **fingerprints}, (out, keys)


def active(values):
    blocks = values[:len(values)//320*320].reshape(-1, 320).astype(np.float64)
    return np.count_nonzero(np.mean(blocks*blocks, axis=1) > 1e-5) >= len(blocks)//2


def qualify(plan, seen):
    outcomes, files = [], {}
    accepted = Counter()
    with ExitStack() as resources, TemporaryDirectory() as folder:
        loader = source_reader.SourceReader(resources, plan["metadata"])
        temp = Path(folder)
        for slot in plan["slots"]:
            role = slot["role"]
            if accepted[role] == ROLES[role][0]:
                continue
            reasons = Counter()

            def check(arm, row, previous, reasons=reasons):
                proof, result = exact(row, loader.original(row), arm, temp, previous, loader)
                if proof is None:
                    reasons[f"{arm}:{result}"] += 1
                return proof, result

            donor, synth = check("clone", slot["clone"], seen)
            if donor is not None:
                paired = qualify_genuine(slot, check, seen | synth[1])
                if paired is not None:
                    host, host_audio, enrolled, enrolled_audio = paired
                    label = slot["slot"]
                    for arm, material in (("clone", synth), ("host", host_audio), ("enrollment", enrolled_audio)):
                        files[f"{label}-{arm}.wav"] = material[0]
                        seen.update(material[1])
                    rooms = ROLES[role][2]
                    outcomes.append({"slot": label, "role": role, "status": "qualified", "future_room": rooms[accepted[role] % 2],
                                     "voice_known_in_prior_project": slot["voice_known_in_prior_project"],
                                     "dialogue_known_in_prior_project": slot["clone"]["context_known_in_prior_project"],
                                     "host": host, "enrollment": enrolled, "clone": donor, "screening_reasons": dict(reasons)})
                    accepted[role] += 1
                    continue
            outcomes.append({"slot": slot["slot"], "role": role, "status": "unavailable", "screening_reasons": dict(reasons)})
    return outcomes, files


def qualify_genuine(slot, check, seen):
    for row in slot["enrollment"]:
        enrollment, ref = check("enrollment", row, seen)
        if enrollment is None:
            continue
        for host_row in slot["host"]:
            if host_row["chapter"] == row["chapter"]:
                continue
            host, foreground = check("host", host_row, seen | ref[1])
            if host is not None:
                return host, foreground, enrollment, ref
    return None


def aggregate(plan, outcomes, run):
    accepted = [item for item in outcomes if item["status"] == "qualified"]
    counts = Counter(item["role"] for item in accepted)
    require(len({item["host"]["voice"] for item in accepted}) == len(accepted)
            and all(item["host"]["voice"] == item["enrollment"]["voice"] == item["clone"]["voice"]
                    and item["host"]["chapter"] != item["enrollment"]["chapter"] for item in accepted), "Host roles or provenance changed")
    return {"schema": SCHEMA, "method": METHOD, "generator_families": ["DuplexGen PLN / Chatterbox"],
            "metadata_slots_frozen": dict(Counter(item["role"] for item in plan["slots"])), "groups_attempted": len(outcomes),
            "historical_exclusion_keys": plan["historical_key_count"], "qualified_groups": dict(counts),
            "qualified_by_future_room": dict(Counter(item["future_room"] for item in accepted)),
            "known_previous_speaker_identity_groups": sum(item["voice_known_in_prior_project"] for item in accepted),
            "known_previous_dialogue_context_groups": sum(item["dialogue_known_in_prior_project"] for item in accepted),
            "clone_sources_with_activity_in_discarded_channel": sum(item["clone"]["other_channel_active_frames"] > 0 for item in accepted),
            "unavailable_groups": len(outcomes)-len(accepted), "source_wav_files": 3*len(accepted),
            "proof_source_minimum_met": counts == {role: n for role, (n, _, _) in ROLES.items()},
            "new_neural_calls": 0, "private_commitments": {name: sha(run / f"{name}.json") for name in ("freeze", "sources")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "qualify", "verify"))
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    plan, seen = freeze_plan()
    run = args.run
    if args.phase == "freeze":
        run.mkdir(mode=0o700)
        write(run / "freeze.json", plan)
        print(f"Frozen {len(plan['slots'])} metadata-only candidate speaker groups", flush=True)
        return
    require(plan == read(run / "freeze.json"), "Metadata selection freeze changed")
    outcomes, files = qualify(plan, seen)
    if args.phase == "qualify":
        (run / "wav").mkdir(mode=0o700)
        for name, wav in files.items():
            (run / "wav" / name).write_bytes(wav)
        write(run / "sources.json", outcomes)
        write(run / "result.json", aggregate(plan, outcomes, run))
        print(read(run / "result.json")["qualified_groups"], flush=True)
    else:
        require(outcomes == read(run / "sources.json") and aggregate(plan, outcomes, run) == read(run / "result.json"),
                "Qualified source receipt did not replay")
        require(all((run / "wav" / name).read_bytes() == raw for name, raw in files.items())
                and len(list((run / "wav").iterdir())) == len(files), "Saved source WAV differs from source replay")
        print(f"Verified {len(files)} exact source WAVs and disjoint publisher-declared source groups", flush=True)


if __name__ == "__main__":
    main()
