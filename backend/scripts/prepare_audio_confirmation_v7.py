"""Plan fresh licensed speech sources using metadata/headers, never a detector.

Acquisition/eligibility only. A separate exact variant/method/gate freeze must
precede confirmation inference. Publisher IDs are not biometric identity proof.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import soundfile as sf

SEED = "chipmunk-audio-v7-fresh-source-eligibility-20261106"
EL_REVISION = "9e41afe1a8b995d2091b602b424c11e4de8e27f6"
DUPLEX_REVISION = "7b7904b73b10ba272145606a06afb6e65bfac985"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stable(value):
    return hashlib.sha256((SEED + "\0" + value).encode()).hexdigest()


def text_keys(text):
    words = re.findall(r"\w+", text.casefold())
    if not words:
        return set()
    return {"full:" + " ".join(words), "prefix8:" + " ".join(words[:8])}


def tar_bytes(tar, member, maximum=10_000_000):
    path = PurePosixPath(member.name)
    if not member.isfile() or path.is_absolute() or ".." in path.parts or member.size > maximum:
        raise ValueError("Unsafe or oversized source archive member")
    with tar.extractfile(member) as stream:
        raw = stream.read(maximum + 1)
    if len(raw) != member.size:
        raise ValueError("Incomplete source archive member")
    return raw


def user_turns(meta):
    """File indices follow speech_meta, which can be shorter than utterances."""
    user = meta["speakers"].index("user")
    return [(i, row["tts_text"]) for i, row in enumerate(meta["speech_meta"]) if row["speaker"] == user]


def original_metadata(root):
    bases, backgrounds = [], []
    for version in ("audio", "audio-v2", "audio-v3"):
        body = json.loads((root / version / "manifest.json").read_text())
        bases.extend(body["base_files"])
        backgrounds.extend(body["backgrounds"])
    prior_text = set()
    for row in bases:
        prior_text.update(text_keys(row.get("transcript") or ""))
    # All DFADD synthetic prompts are declared to come from LJ Speech, although
    # the exact per-file prompt mapping is unavailable. Exclude known prefixes.
    lj = root / "audio-v3/research/lj-speech-original-metadata.csv"
    for row in csv.reader(lj.open(), delimiter="|"):
        for text in row[1:]:
            prior_text.update(text_keys(text))
    return bases, backgrounds, prior_text


def libri_inventory(archives):
    rows = []
    for archive in archives:
        texts, audio = {}, []
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if member.name.endswith(".trans.txt"):
                    for line in tar_bytes(tar, member).decode().splitlines():
                        identifier, text = line.split(" ", 1)
                        texts[identifier] = text
                elif member.name.endswith(".flac"):
                    raw = tar_bytes(tar, member)
                    info = sf.info(io.BytesIO(raw))
                    audio.append(
                        {
                            "id": "libri:" + Path(member.name).stem,
                            "speaker": Path(member.name).stem.split("-")[0],
                            "archive": str(archive),
                            "member": member.name,
                            "sha256": hashlib.sha256(raw).hexdigest(),
                            "seconds": info.duration,
                            "sample_rate": info.samplerate,
                            "channels": info.channels,
                        }
                    )
        for row in audio:
            row["transcript"] = texts[row["id"].split(":", 1)[1]]
            rows.append(row)
    return rows


def choose_libri(rows, prior_speakers, prior_text):
    voices = defaultdict(list)
    for row in rows:
        if (
            row["speaker"] not in prior_speakers
            and row["seconds"] >= 4
            and row["channels"] == 1
            and text_keys(row["transcript"])
            and not text_keys(row["transcript"]) & prior_text
        ):
            voices[row["speaker"]].append(row)
    if len(voices) != 66:
        raise ValueError(
            "Expected all66 known-unused LibriSpeech speakers; do not substitute after inference"
        )
    selected, seen = [], set()
    for i, voice in enumerate(sorted(voices, key=stable)):
        candidates = sorted(voices[voice], key=lambda r: r["id"])
        eligible = [r for r in candidates if not text_keys(r["transcript"]) & seen]
        if not eligible:
            raise ValueError("Known-unused speaker lacks a distinct eligible utterance")
        row = dict(eligible[0])
        seen.update(text_keys(row["transcript"]))
        row.update(
            role="background" if i < 6 else "foreground",
            label=0,
            generator="human",
            dataset="OpenSLR12/LibriSpeech",
            source_group="libri-speaker:" + voice,
            license="CC-BY-4.0",
            license_url="https://www.openslr.org/12",
        )
        selected.append(row)
    return selected


def choose_elevenlabs(metadata, repository, prior_text):
    records = list(csv.DictReader((metadata / "metadata.csv").open()))
    voices = list(csv.DictReader((metadata / "speaker_ids.csv").open()))
    if len(voices) != 30 or repository["sha"] != EL_REVISION:
        raise ValueError("Pinned30-voice ElevenLabs source required")
    files = {r["rfilename"]: r for r in repository["siblings"]}
    selected, used_scripts = [], set()
    for voice in sorted(voices, key=lambda r: stable(r["voice_id"])):
        candidates = sorted(
            [r for r in records if r["voice_id"] == voice["voice_id"]],
            key=lambda r: stable(r["script_filename"]),
        )
        eligible = [
            r
            for r in candidates
            if r["script_filename"] not in used_scripts
            and not text_keys(r["script_text"]) & prior_text
            and r["file_name"] in files
        ]
        if not eligible:
            raise ValueError("No unconsumed script for a fixed ElevenLabs voice")
        row = eligible[0]
        used_scripts.add(row["script_filename"])
        asset = files[row["file_name"]]
        if asset["size"] > 20_000_000 or not re.fullmatch(r"[0-9a-f]{64}", asset["lfs"]["sha256"]):
            raise ValueError("Bounded publisher LFS checksum required")
        selected.append(
            {
                "id": "elevenlabs:" + stable(row["file_name"]),
                "source_group": "elevenlabs-voice:" + stable(voice["voice_id"]),
                "voice_id_sha256": hashlib.sha256(voice["voice_id"].encode()).hexdigest(),
                "source_path": row["file_name"],
                "transcript": row["script_text"],
                "generator": "ElevenLabs",
                "dataset": "multispeak/accent-synthetic-voices",
                "revision": EL_REVISION,
                "label": 1,
                "role": "foreground",
                "accent": row["accent"],
                "script_type": row["script_type"],
                "license": "CC-BY-4.0",
                "license_url": f"https://huggingface.co/datasets/multispeak/accent-synthetic-voices/resolve/{EL_REVISION}/README.md",
                "asset": {
                    "name": asset["lfs"]["sha256"] + ".wav",
                    "url": f"https://huggingface.co/datasets/multispeak/accent-synthetic-voices/resolve/{EL_REVISION}/"
                    + quote(row["file_name"], safe="/"),
                    "bytes": asset["size"],
                    "max_bytes": asset["size"],
                    "sha256": asset["lfs"]["sha256"],
                },
            }
        )
    return selected


def duplex_inventory(archive_dir, publisher_rows):
    references = {r["key"]: r for r in publisher_rows if r["scenario"] == "PLN"}
    rows = []
    for archive in sorted(archive_dir.glob("PLN-0000[0-3].tar")):
        with tarfile.open(archive, "r:") as tar:
            members = {r.name: r for r in tar.getmembers()}
            for name, member in sorted(members.items()):
                if not name.endswith("/meta.json"):
                    continue
                key = name.removesuffix("/meta.json")
                reference = references[key]
                if reference["license"] != "MIT":
                    raise ValueError("Only publisher-declared MIT PLN scenarios allowed")
                meta = json.loads(tar_bytes(tar, member))
                for i, text in user_turns(meta):
                    path = f"{key}/utterances/{i:02d}.wav"
                    if path not in members:
                        raise ValueError("Declared synthesized utterance missing")
                    info = sf.info(io.BytesIO(tar_bytes(tar, members[path])))
                    if info.channels not in (1, 2):
                        raise ValueError("Unexpected isolated utterance channel count")
                    rows.append(
                        {
                            "id": "chatterbox:" + path,
                            "source_group": "duplex-dialogue:" + reference["example_id"],
                            "prompt_speaker_id": str(meta["user_prompt_speaker_id"]),
                            "archive": str(archive),
                            "member": path,
                            "meta_sha256": hashlib.sha256(tar_bytes(tar, member)).hexdigest(),
                            "transcript": text,
                            "seconds": info.duration,
                            "sample_rate": info.samplerate,
                            "channels": info.channels,
                            "label": 1,
                            "role": "foreground",
                            "generator": "Chatterbox",
                            "dataset": "DuplexGen/duplexgen-spoken",
                            "revision": DUPLEX_REVISION,
                            "license": "MIT (publisher PLN scenario declaration)",
                            "license_url": f"https://huggingface.co/datasets/DuplexGen/duplexgen-spoken/resolve/{DUPLEX_REVISION}/README.md",
                        }
                    )
    return rows


def choose_duplex(rows, prior_text):
    selected, dialogues, voices, known_text = [], set(), set(), set(prior_text)
    for row in sorted(rows, key=lambda r: (stable(r["source_group"]), r["member"])):
        if (
            row["seconds"] < 4
            or row["source_group"] in dialogues
            or row["prompt_speaker_id"] in voices
            or not text_keys(row["transcript"])
            or text_keys(row["transcript"]) & known_text
        ):
            continue
        selected.append(row)
        dialogues.add(row["source_group"])
        voices.add(row["prompt_speaker_id"])
        known_text.update(text_keys(row["transcript"]))
        if len(selected) == 30:
            return selected
    raise ValueError(
        "Insufficient distinct four-second user voices/dialogues; no detector-based substitutions"
    )


def plan(root, output):
    fresh = root / "audio-v7"
    planning = fresh / "fresh-planning"
    prior_bases, _, prior_text = original_metadata(root)
    prior_speakers = set(
        json.loads((planning / "genuine-speaker-eligibility.json").read_text())["prior_librispeech_speakers"]
    )
    libri = libri_inventory(
        [
            fresh / "fresh-sources/archives/test-other.tar.gz",
            fresh / "fresh-sources/supplement/dev-other.tar.gz",
        ]
    )
    humans = choose_libri(libri, prior_speakers, prior_text)
    el_metadata = root / "audio-v4/fresh-sources/accent-synthetic-voices"
    elevenlabs = choose_elevenlabs(
        el_metadata, json.loads((planning / "elevenlabs-files.metadata").read_text()), prior_text
    )
    with (root / "audio-v4/fresh-sources/DuplexGen--duplexgen-spoken/metadata.jsonl").open() as stream:
        publisher = [json.loads(line) for line in stream]
    duplex = duplex_inventory(fresh / "fresh-sources/archives", publisher)
    blocked = prior_text | set().union(*(text_keys(r["transcript"]) for r in humans + elevenlabs))
    chatterbox = choose_duplex(duplex, blocked)
    sources = humans + elevenlabs + chatterbox
    prior_hashes = {r["sha256"] for r in prior_bases}
    if any(r.get("sha256", r.get("asset", {}).get("sha256")) in prior_hashes for r in sources):
        raise ValueError("Selected source was already consumed")
    body = {
        "schema": "chipmunk.audio-v7-fresh-foreground-plan/1",
        "seed": SEED,
        "source_selection_code_sha256": digest(Path(__file__)),
        "sources": sources,
        "counts": dict(Counter((r["role"] + ":" + r["generator"]) for r in sources)),
        "eligible_libri_files": len(libri),
        "eligible_duplex_user_turns": len(duplex),
        "source_window_samples": [0, 64000],
        "elevenlabs_asset_plan": {"assets": [r["asset"] for r in elevenlabs]},
        "detector_inference_run": False,
        "deployment_eligible": False,
        "scope": "Metadata/header selection, not final background/variant/method/gate freeze. Preserve every selection and failure; never choose by detector output.",
        "limitations": [
            "Publisher voice/speaker identifiers do not establish biometric identity disjointness across corpora.",
            "Known full texts and eight-word prefixes are checked; exact alignment of the first four seconds is unknown.",
            "Some ElevenLabs scripts share openings; connect known prefixes in the confirmation uncertainty groups.",
            "Publisher scenario permission declarations do not independently establish every underlying prompt right.",
        ],
    }
    with output.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"plan_sha256": digest(output), "counts": body["counts"], "detector_inference_run": False})
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan(args.root, args.output)
