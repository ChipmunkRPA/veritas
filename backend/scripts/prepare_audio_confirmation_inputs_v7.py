"""Materialize the fixed fresh sources and unused backgrounds, without models."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts.prepare_audio_confirmation_v7 import digest, original_metadata, stable, tar_bytes, text_keys


def store(root, raw):
    sha = hashlib.sha256(raw).hexdigest()
    path = root / "sources" / (sha + ".audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("xb") as stream:
            stream.write(raw)
    if digest(path) != sha:
        raise ValueError("Materialized source bytes changed")
    return {"path": str(path.relative_to(root)), "sha256": sha, "bytes": len(raw)}


def materialize(plan, root):
    sources = plan["sources"]
    references = {}
    archives = sorted({r["archive"] for r in sources if "archive" in r})
    for name in archives:
        selected = {r["member"]: r for r in sources if r.get("archive") == name}
        mode = "r|gz" if name.endswith(".gz") else "r:"
        with tarfile.open(name, mode) as tar:
            for member in tar:
                if member.name in selected:
                    row = selected[member.name]
                    ref = store(root, tar_bytes(tar, member))
                    if row.get("sha256") and row["sha256"] != ref["sha256"]:
                        raise ValueError("Selected publisher source bytes changed")
                    references[row["id"]] = ref
        if len([r for r in selected.values() if r["id"] in references]) != len(selected):
            raise ValueError("Fixed archive selection is incomplete")
    for row in sources:
        if row["generator"] == "ElevenLabs":
            path = root.parent / "fresh-sources/elevenlabs" / row["asset"]["name"]
            ref = store(root, path.read_bytes())
            if ref["sha256"] != row["asset"]["sha256"] or ref["bytes"] != row["asset"]["bytes"]:
                raise ValueError("ElevenLabs publisher checksum differs")
            references[row["id"]] = ref
    return [{**row, **references[row["id"]]} for row in sources]


def group_sources(sources):
    foreground = [r for r in sources if r["role"] == "foreground"]
    parent = {r["id"]: r["id"] for r in foreground}
    if len(parent) != len(foreground):
        raise ValueError("Repeated foreground source")

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for row in foreground:
        keys = {"source:" + row["source_group"]} | text_keys(row["transcript"])
        if row["generator"] == "Chatterbox":
            keys.add("duplex-prompt:" + row["prompt_speaker_id"])
        for key in keys:
            if key in owners:
                parent[find(row["id"])] = find(owners[key])
            else:
                owners[key] = row["id"]
    grouped = defaultdict(list)
    for row in foreground:
        grouped[find(row["id"])].append(row["id"])
    assignments, records = {}, []
    for members in grouped.values():
        members = sorted(members)
        group = "connected:" + hashlib.sha256(json.dumps(members, separators=(",", ":")).encode()).hexdigest()
        records.append({"group": group, "sources": members})
        assignments.update({key: group for key in members})
    return assignments, sorted(records, key=lambda r: r["group"])


def unused_backgrounds(data, root, prior_backgrounds):
    from scripts.evaluate_audio_candidates import decode

    denied_ids = {r.get("source_id") for r in prior_backgrounds}
    denied_hashes = {r["sha256"] for r in prior_backgrounds}
    denied_pcm = {r["decoded_pcm_sha256"] for r in prior_backgrounds if r.get("decoded_pcm_sha256")}
    denied_stems = {Path(str(p)).stem for p in denied_ids if p}
    available = []
    for path in sorted((data / "audio-v3/metadata").glob("parquet-audio-group-*.json")):
        for row in json.loads(path.read_text())["rows"]:
            source_id = row.get("path", "")
            if source_id.startswith(("noise/free-sound/", "music/rfm/")):
                available.append(row)
    selected = []
    for kind, count, subset, license_name in (
        ("noise", 8, "free-sound", "Public-domain"),
        ("music", 4, "rfm", "CC-BY-3.0"),
    ):
        license_path = data / f"audio-v3/research/musan-{kind}-{subset}-LICENSE.txt"
        declaration = license_path.read_text()
        if ("Public Domain" if kind == "noise" else "CC BY 3.0") not in declaration:
            raise ValueError("Known commercial-permission declaration required")
        pool = sorted(
            [r for r in available if r["path"].startswith(kind + "/")], key=lambda r: stable(r["path"])
        )
        eligible = []
        for row in pool:
            ref = row["ref"]
            if (
                row["path"] in denied_ids
                or Path(row["path"]).stem in denied_stems
                or ref["sha256"] in denied_hashes
                or ref["decoded_pcm_sha256"] in denied_pcm
            ):
                continue
            if kind == "music" and Path(row["path"]).stem not in declaration:
                raise ValueError("Music attribution missing")
            source = data / "audio-v3" / ref["path"]
            if digest(source) != ref["sha256"]:
                raise ValueError("Cached original background hash differs")
            pcm = decode(source)
            pcm_sha = hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
            if pcm_sha != ref["decoded_pcm_sha256"]:
                raise ValueError("Cached background decoding differs")
            if len(pcm) < 8 * 16000 or float(np.sqrt(np.mean(pcm.astype(float) ** 2))) < 1e-4:
                continue
            current = store(root, source.read_bytes())
            eligible.append(
                {
                    **ref,
                    **current,
                    "id": row["path"],
                    "source_id": row["path"],
                    "kind": kind,
                    "license": license_name,
                    "license_url": f"https://huggingface.co/datasets/corypaik/musan/raw/{ref['revision']}/licenses/{kind}/{subset}/LICENSE",
                    "license_sha256": digest(license_path),
                    "attribution": "Kevin MacLeod / Incompetech via MUSAN"
                    if kind == "music"
                    else "MUSAN / FreeSound public-domain credits in retained LICENSE",
                }
            )
            denied_ids.add(row["path"])
            denied_hashes.add(ref["sha256"])
            denied_pcm.add(pcm_sha)
            if len(eligible) == count:
                break
        if len(eligible) != count:
            raise ValueError("Insufficient unused permitted backgrounds; no outcome-based substitutions")
        selected.extend(eligible)
    return selected


def prepare(data, plan_path, root):
    plan = json.loads(plan_path.read_text())
    code = Path(__file__).with_name("prepare_audio_confirmation_v7.py")
    if digest(code) != plan["source_selection_code_sha256"]:
        raise ValueError("Source selection implementation changed after planning")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "prepared-sources.json").exists():
        raise ValueError("Preserve existing source preparation")
    prior_bases, prior_backgrounds, _ = original_metadata(data)
    denied_bytes = {r["sha256"] for r in prior_bases + prior_backgrounds}
    denied_pcm = {
        row[key]
        for row in prior_bases
        for key in ("production_pcm_sha256", "source_prefix_pcm_sha256", "decoded_pcm_sha256")
        if row.get(key)
    }
    sources = materialize(plan, root)
    decoded, prefixes = set(), set()
    for row in sources:
        if row["sha256"] in denied_bytes:
            raise ValueError("Previously consumed source bytes")
        audio = decode_audio(root / row["path"], source_sha256=row["sha256"])
        pcm = audio.pcm
        if (
            len(pcm) < 64000
            or not np.isfinite(pcm).all()
            or float(np.sqrt(np.mean(pcm[:64000].astype(float) ** 2))) < 1e-4
        ):
            raise ValueError("Selected source is short, silent or invalid; preserve selection and stop")
        full_sha = audio.mapping["pcm_sha256"]
        prefix_sha = hashlib.sha256(pcm[:64000].astype("<f4").tobytes()).hexdigest()
        if full_sha in denied_pcm | decoded or prefix_sha in denied_pcm | prefixes:
            raise ValueError("Source/full-prefix decoded PCM overlaps prior/current recordings")
        row.update(
            production_input=audio.mapping,
            production_pcm_sha256=full_sha,
            source_prefix_pcm_sha256=prefix_sha,
        )
        decoded.add(full_sha)
        prefixes.add(prefix_sha)
    assignments, groups = group_sources(sources)
    foreground_keys = set().union(*(text_keys(r["transcript"]) for r in sources if r["role"] == "foreground"))
    if any(text_keys(r["transcript"]) & foreground_keys for r in sources if r["role"] == "background"):
        raise ValueError("Known foreground/background prompt connection")
    backgrounds = unused_backgrounds(data, root, prior_backgrounds)
    body = {
        "schema": "chipmunk.audio-v7-fresh-prepared-sources/1",
        "foreground_plan_sha256": digest(plan_path),
        "preparation_code_sha256": digest(Path(__file__)),
        "sources": sources,
        "backgrounds": backgrounds,
        "groups": groups,
        "assignments": assignments,
        "counts": dict(Counter(r["role"] + ":" + r["generator"] for r in sources)),
        "detector_inference_run": False,
        "scope": "Source/decoded-prefix novelty, known connected groups and unused backgrounds. Exact variant/method/gate freeze remains required before scoring.",
        "limitations": [
            "Novelty checks cover known recording IDs, metadata, bytes and decoded prefixes, not unknown pretrained overlap or biometric identity across corpora.",
            "Noise contributor/session identities are unpublished; selected music tracks share a composer with previous tracks.",
            "Known script prefixes connect ElevenLabs sources; intervals must condition on the selected backgrounds and respect connected foreground groups.",
        ],
    }
    out = root / "prepared-sources.json"
    with out.open("x") as stream:
        json.dump(body, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "prepared_sources_sha256": digest(out),
                "counts": body["counts"],
                "foreground_connected_groups": len(groups),
                "unused_noise_backgrounds": 8,
                "unused_music_backgrounds": 4,
                "detector_inference_run": False,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.data, args.plan, args.output)
