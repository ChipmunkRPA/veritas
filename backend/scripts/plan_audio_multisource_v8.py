"""Reserve untouched confirmation identities and split consumed sources before fitting.

Metadata/header planning only. No classifier, source selection by response or
claim of biometric cross-corpus independence. Audio eligibility must complete
before the final method/input freeze and any new representations or fitting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from scripts.prepare_audio_confirmation_v7 import duplex_inventory, text_keys

SEED = "chipmunk-audio-multisource-v8-20261106"
LAION_REVISION = "4b24030e508c0e42b49647bddb96598a7aed884b"
DICTATION_REVISION = "692f01ad28a5d6f26bde093b6d717e89cfae0067"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable(value):
    return hashlib.sha256((SEED + "\0" + value).encode()).hexdigest()


def alias(text):
    return re.sub(r"[^a-z]", "", text.casefold())


def consumed_sources(data):
    v3 = json.loads((data / "audio-v3/manifest.json").read_text())["base_files"]
    recent = json.loads((data / "audio-v7/confirmation/prepared-sources.json").read_text())["sources"]
    names = list(
        csv.DictReader((data / "audio-v4/fresh-sources/accent-synthetic-voices/speaker_ids.csv").open())
    )
    voice_names = {hashlib.sha256(v["voice_id"].encode()).hexdigest(): v["name"] for v in names}
    result = []
    for row in v3:
        result.append(
            {
                **row,
                "source_path": str(data / "audio-v3" / row["path"]),
                "voice_key": row["cluster_id"],
                "source_key": row["source_group"],
                "full_samples": row["decoded_samples"],
                "consumed_from": "v3",
            }
        )
    for row in recent:
        if row["role"] != "foreground":
            continue
        voice = (
            "duplex-prompt:" + row["prompt_speaker_id"]
            if row["generator"] == "Chatterbox"
            else row["source_group"]
        )
        name = voice_names.get(row.get("voice_id_sha256"))
        result.append(
            {
                **row,
                "source_path": str(data / "audio-v7/confirmation" / row["path"]),
                "voice_key": voice,
                "source_key": row["source_group"],
                "full_samples": row["production_input"]["samples"],
                "provider_alias": name,
                "consumed_from": "v7-failed-confirmation",
            }
        )
    if len(result) != 540 or len({r["id"] for r in result}) != 540:
        raise ValueError("Every540 consumed source reference required before exclusions")
    return result


def connected_sources(rows):
    parent = {r["id"]: r["id"] for r in rows}
    if len(parent) != len(rows):
        raise ValueError("Duplicate original source identity")

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for row in rows:
        keys = {"voice:" + row["voice_key"], "source:" + row["source_key"], "bytes:" + row["sha256"]}
        keys |= text_keys(row.get("transcript") or "")
        if row.get("decoded_pcm_sha256"):
            keys.add("pcm:" + row["decoded_pcm_sha256"])
        if row.get("production_pcm_sha256"):
            keys.add("production_pcm:" + row["production_pcm_sha256"])
        if row.get("prompt_index") is not None and row["dataset"] == "isjwdu/DFADD" and row["label"] == 1:
            keys.add("dfadd_prompt:" + str(row["prompt_index"]))
        for key in sorted(keys):
            if key in owners:
                parent[find(row["id"])] = find(owners[key])
            else:
                owners[key] = row["id"]
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row)
    result = []
    for members in groups.values():
        ids = sorted(r["id"] for r in members)
        group = "connected:" + hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        result.append(
            {
                "group": group,
                "sources": ids,
                "stratum": "+".join(sorted({str(r["label"]) + ":" + r["generator"] for r in members})),
                "class_counts": {str(c): sum(r["label"] == c for r in members) for c in (0, 1)},
            }
        )
    return sorted(result, key=lambda r: r["group"])


def roles(groups):
    strata = defaultdict(list)
    for row in groups:
        strata[row["stratum"]].append(row)
    assignments = {}
    for name, members in sorted(strata.items()):
        members = sorted(members, key=lambda r: stable(name + r["group"]))
        if len(members) < 4:
            # Tiny connected strata cannot be split honestly. Keep them in
            # development, never slice a shared identity to fill a role count.
            split = ["development"] * len(members)
        else:
            nfit = len(members) // 2
            ncal = max(1, len(members) // 4)
            split = ["fit"] * nfit + ["cal"] * ncal + ["development"] * (len(members) - nfit - ncal)
        for row, role in zip(members, split, strict=True):
            assignments[row["group"]] = role
    return assignments


def reserve_laion(planning, prior_provider_ids):
    repository = json.loads(
        (planning / "laion--elevenlabs-v4-five-language-voice-acting-repository.json").read_text()
    )
    if repository["sha"] != LAION_REVISION or repository["cardData"]["license"] != "cc-by-4.0":
        raise ValueError("Pinned CC-BY4 LAION declaration required")
    voices = list(csv.DictReader((planning / "laion-metadata--voice_aliases.csv").open()))
    if any(r["provider_voice"] in prior_provider_ids for r in voices):
        raise ValueError("A reserved provider profile was already used")
    index = [json.loads(line) for line in (planning / "laion-index.jsonl").read_text().splitlines()]
    selected, used = [], set()
    for voice in sorted(voices, key=lambda r: stable(r["provider_voice"])):
        pool = sorted(
            [
                r
                for r in index
                if r["voice"] == voice["provider_voice"] and r["language"] == "en" and r["duration_s"] >= 4
            ],
            key=lambda r: stable(r["id"]),
        )
        choices = [r for r in pool if r["challenge_group"] not in used]
        if not choices:
            raise ValueError("A fixed new provider profile lacks a distinct English challenge")
        row = choices[0]
        used.add(row["challenge_group"])
        selected.append(
            {
                k: row[k]
                for k in (
                    "id",
                    "voice",
                    "speaker_alias",
                    "speaker_id",
                    "challenge_group",
                    "duration_s",
                    "audio_sha256",
                    "language",
                )
            }
        )
    if len(selected) != 24:
        raise ValueError("Reserve all24 new provider profiles")
    # Full intended scripts are checked after raw JSON acquisition; the index's
    # learned annotation/review scores never select cases or serve as labels.
    return selected


def reserve_kokoro(planning, prior_text):
    metadata = json.loads((planning / "kokoro-directed-metadata.json").read_text())
    if metadata["revision"] != DICTATION_REVISION:
        raise ValueError("Pinned dictation metadata required")
    voices = sorted({r["tts_voice"] for r in metadata["rows"]}, key=stable)
    selected, used_text, passages = [], set(prior_text), set()
    for voice in voices:
        eligible = sorted(
            [
                r
                for r in metadata["rows"]
                if r["tts_voice"] == voice
                and r["duration_s"] >= 4
                and r["language"] == "en"
                and r["tts_engine"] == "kokoro-82m"
            ],
            key=lambda r: stable(r["id"]),
        )
        for row in eligible:
            keys = text_keys(row["transcript"])
            passage = row["rendition_id"].split("-", 1)[0]
            if keys and not keys & used_text and passage not in passages:
                selected.append({**row, "parquet_row": metadata["rows"].index(row)})
                used_text.update(keys)
                passages.add(passage)
                break
        else:
            raise ValueError("New voice lacks a distinct eligible script; no model-based substitutions")
    if len(selected) != 12:
        raise ValueError("Reserve all12 Kokoro voice profiles")
    return selected


def reserve_chatter(data, prior_text, other_text):
    old = json.loads((data / "audio-v7/confirmation/prepared-sources.json").read_text())["sources"]
    old = [r for r in old if r["generator"] == "Chatterbox"]
    denied_voice = {r["prompt_speaker_id"] for r in old}
    denied_dialogue = {r["source_group"] for r in old}
    with (data / "audio-v4/fresh-sources/DuplexGen--duplexgen-spoken/metadata.jsonl").open() as stream:
        publisher = [json.loads(line) for line in stream]
    rows = duplex_inventory(data / "audio-v7/fresh-sources/archives", publisher)
    selected, voices, dialogues, texts = [], set(), set(), set(prior_text) | set(other_text)
    for row in sorted(rows, key=lambda r: (stable(r["source_group"]), r["member"])):
        keys = text_keys(row["transcript"])
        if (
            row["seconds"] < 4
            or row["prompt_speaker_id"] in denied_voice | voices
            or row["source_group"] in denied_dialogue | dialogues
            or not keys
            or keys & texts
        ):
            continue
        selected.append(row)
        voices.add(row["prompt_speaker_id"])
        dialogues.add(row["source_group"])
        texts.update(keys)
        if len(selected) == 24:
            return selected
    raise ValueError("Reserve24 distinct unused dialogue/prompt identities before training")


def plan(data, output):
    planning = data / "audio-v8/source-planning"
    consumed = consumed_sources(data)
    prior_text = set().union(*(text_keys(r.get("transcript") or "") for r in consumed))
    with (data / "audio-v3/research/lj-speech-original-metadata.csv").open() as stream:
        for row in csv.reader(stream, delimiter="|"):
            for text in row[1:]:
                prior_text.update(text_keys(text))
    prior_provider_ids = {
        r["voice_id"]
        for r in csv.DictReader(
            (data / "audio-v4/fresh-sources/accent-synthetic-voices/speaker_ids.csv").open()
        )
    }
    laion = reserve_laion(planning, prior_provider_ids)
    kokoro = reserve_kokoro(planning, prior_text)
    chatter = reserve_chatter(data, prior_text, set().union(*(text_keys(r["transcript"]) for r in kokoro)))
    reserved_aliases = {alias(r["speaker_alias"].removeprefix("Eleven")) for r in laion} | {
        alias(r["tts_voice"].split("_", 1)[1]) for r in kokoro
    }
    selected, exclusions = [], []
    for row in consumed:
        reason = (
            "source shorter than four seconds"
            if row["full_samples"] < 64000
            else "conservative alias collision with reserved confirmation profile"
            if row.get("provider_alias") and alias(row["provider_alias"]) in reserved_aliases
            else None
        )
        if reason:
            exclusions.append({"id": row["id"], "reason": reason})
        else:
            selected.append(row)
    groups = connected_sources(selected)
    assignments = roles(groups)
    group_by_source = {identifier: row["group"] for row in groups for identifier in row["sources"]}
    for row in selected:
        row["group"] = group_by_source[row["id"]]
        row["role"] = assignments[row["group"]]
    prior_libri = set(
        json.loads((data / "audio-v7/fresh-planning/genuine-speaker-eligibility.json").read_text())[
            "prior_librispeech_speakers"
        ]
    )
    index = json.loads((data / "audio-v3/metadata/libri-train100-index.json").read_text())
    available = sorted({str(r["speaker_id"]) for r in index} - prior_libri, key=stable)
    if len(available) != 91:
        raise ValueError("All91 known-unused train100 speakers must be accounted for")
    humans = []
    for i, speaker in enumerate(available[:78]):
        role = (
            "confirmation_foreground"
            if i < 60
            else "confirmation_background"
            if i < 66
            else ("fit_background", "cal_background", "development_background")[(i - 66) // 4]
        )
        candidates = sorted(
            [
                r
                for r in index
                if str(r["speaker_id"]) == speaker
                and text_keys(r["text"])
                and not text_keys(r["text"]) & prior_text
            ],
            key=lambda r: r["id"],
        )
        if not candidates:
            raise ValueError("Reserved speaker has no unconsumed known script")
        humans.append(
            {
                "speaker_id": speaker,
                "role": role,
                "ordered_candidates": candidates,
                "minimum_seconds": 4 if role == "confirmation_foreground" else 8,
            }
        )
    body = {
        "schema": "chipmunk.audio-multisource-data-plan/8",
        "seed": SEED,
        "planning_code_sha256": digest(Path(__file__)),
        "consumed_sources_total": len(consumed),
        "development_sources": selected,
        "exclusions": exclusions,
        "groups": groups,
        "role_counts": dict(Counter(r["role"] for r in selected)),
        "role_generator_counts": {
            role: dict(Counter(r["generator"] for r in selected if r["role"] == role))
            for role in ("fit", "cal", "development")
        },
        "confirmation_laion": laion,
        "confirmation_kokoro": kokoro,
        "confirmation_chatterbox": chatter,
        "new_genuine_speaker_reservations": humans,
        "fresh_foreground_planned": 120,
        "fresh_conditions_planned": 9,
        "fresh_inputs_planned": 1080,
        "metadata_pins": {str(p.relative_to(data)): digest(p) for p in planning.iterdir() if p.is_file()},
        "fitting_started": False,
        "neural_inference_run": False,
        "deployment_eligible": False,
        "scope": "Identity/metadata reservation and connected consumed-source roles only; finish audio eligibility and exact role-disjoint backgrounds before final method/input freeze.",
        "limitations": [
            "Provider profile IDs, known aliases and corpus speaker IDs do not establish biometric cross-corpus identity or pretrained disjointness.",
            "LAION scripts require raw generation-JSON checks; learned annotation scores are not used as truth or selectors.",
            "Dictation evaluation audio retains CC-BY-SA4 attribution/share-alike terms; its dataset license does not relicense upstream weights.",
            "Some small connected strata remain entirely in development, rather than splitting a shared identity.",
        ],
    }
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "plan_sha256": digest(output),
                "consumed_sources": len(selected),
                "excluded": len(exclusions),
                "connected_groups": len(groups),
                "role_counts": body["role_counts"],
                "role_generator_counts": body["role_generator_counts"],
                "confirmation_profiles": {
                    "ElevenLabs_v4": len(laion),
                    "Kokoro": len(kokoro),
                    "Chatterbox": len(chatter),
                },
                "fitting_started": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan(args.data, args.output)
