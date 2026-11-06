"""Verify reserved originals and separate all four roles before model extraction.

Only decoding, known identity/text connections and deterministic eligibility are
used here. No learned responses, VAD or source substitution by detector output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio
from scripts.plan_audio_multisource_v8 import digest, stable
from scripts.prepare_audio_confirmation_inputs_v7 import store
from scripts.prepare_audio_confirmation_v7 import text_keys
from scripts.prepare_audio_guard_v7 import write

ROLES = ("fit", "cal", "development", "confirmation")


def pcm_hash(pcm):
    return hashlib.sha256(np.asarray(pcm, dtype="<f4").tobytes()).hexdigest()


def checked_source(row, path):
    if digest(path) != row["sha256"]:
        raise ValueError("Original source bytes changed")
    audio = decode_audio(path, source_sha256=row["sha256"])
    expected = row.get("production_pcm_sha256")
    if expected and audio.mapping["pcm_sha256"] != expected:
        raise ValueError("Original production PCM changed")
    if len(audio.pcm) < 64000 or not np.isfinite(audio.pcm).all():
        raise ValueError("Reserved original lacks four finite seconds; preserve the failure")
    prefix = pcm_hash(audio.pcm[:64000])
    if row.get("prefix_pcm_sha256") and row["prefix_pcm_sha256"] != prefix:
        raise ValueError("Reserved four-second prefix changed")
    rms = float(np.sqrt(np.mean(audio.pcm[:64000].astype(float) ** 2)))
    if rms < 1e-4:
        raise ValueError("Reserved foreground prefix is silent; do not substitute another case")
    return {
        **row,
        "source_path": str(path.resolve()),
        "production_pcm_sha256": audio.mapping["pcm_sha256"],
        "full_samples": len(audio.pcm),
        "prefix_pcm_sha256": prefix,
        "source_interval_samples": [0, 64000],
        "prefix_rms": rms,
        "source_production_input": audio.mapping,
    }


def connected_roles(rows):
    """Keep known voice, prompt, byte and actual-prefix connections together."""
    parent = {r["id"]: r["id"] for r in rows}
    if len(parent) != len(rows):
        raise ValueError("Duplicate source identity")

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    owners = {}
    for row in rows:
        if row["role"] not in ROLES:
            raise ValueError("Explicit research role required")
        keys = {
            "voice:" + row["voice_key"],
            "source:" + row["source_key"],
            "bytes:" + row["sha256"],
            "pcm:" + row["production_pcm_sha256"],
            "prefix:" + row["prefix_pcm_sha256"],
        } | text_keys(row.get("transcript") or "")
        if row.get("group"):
            keys.add("planned-group:" + row["group"])
        if row.get("prompt_index") is not None and row["dataset"] == "isjwdu/DFADD" and row["label"] == 1:
            keys.add("dfadd-prompt:" + str(row["prompt_index"]))
        for key in sorted(keys):
            if key in owners:
                parent[find(row["id"])] = find(owners[key])
            else:
                owners[key] = row["id"]
    groups = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row)
    assignments, records = {}, []
    for members in groups.values():
        roles = {r["role"] for r in members}
        if len(roles) != 1:
            raise ValueError("Known original/voice/prompt/prefix connection crosses frozen roles")
        if any(r.get("kind") == "speech" for r in members) and len(members) > 1:
            raise ValueError("Background speaker/prompt connects to a foreground or another background")
        identifiers = sorted(r["id"] for r in members)
        group = (
            "connected:" + hashlib.sha256(json.dumps(identifiers, separators=(",", ":")).encode()).hexdigest()
        )
        assignments.update({r["id"]: group for r in members})
        records.append({"group": group, "role": next(iter(roles)), "sources": identifiers})
    return assignments, sorted(records, key=lambda r: r["group"])


def select_backgrounds(data, root):
    prior = []
    for version in ("audio", "audio-v2", "audio-v3", "audio-v7/confirmation"):
        name = "prepared-sources.json" if version.endswith("confirmation") else "manifest.json"
        body = json.loads((data / version / name).read_text())
        for row in body["backgrounds"]:
            path = data / version / row["path"]
            audio = decode_audio(path, source_sha256=row["sha256"])
            if digest(path) != row["sha256"]:
                raise ValueError("Prior background source changed")
            prior.append({**row, "production_pcm_sha256": audio.mapping["pcm_sha256"]})
    denied_ids = {r["source_id"] for r in prior}
    denied_stems = {Path(r).stem for r in denied_ids}
    denied_bytes = {r["sha256"] for r in prior}
    denied_pcm = {r["production_pcm_sha256"] for r in prior}
    denied_pcm.update(r["decoded_pcm_sha256"] for r in prior if r.get("decoded_pcm_sha256"))
    inventory = {}
    for path in sorted((data / "audio-v3/metadata").glob("parquet-audio-group-*.json")):
        for row in json.loads(path.read_text())["rows"]:
            if row.get("path", "").startswith(("noise/free-sound/", "music/rfm/")):
                if row["path"] in inventory and row != inventory[row["path"]]:
                    raise ValueError("Conflicting cached background identity")
                inventory[row["path"]] = row
    extra_path = data / "audio-v8/source-planning/additional-noise-inventory.json"
    extra = json.loads(extra_path.read_text())
    if extra["code_sha256"] != digest(Path(__file__).with_name("materialize_background_audio_v8.py")):
        raise ValueError("Additional background materialization code changed")
    for row in extra["rows"]:
        if row["path"] in inventory:
            raise ValueError("Additional background inventory repeats an existing source identity")
        inventory[row["path"]] = row
    selected, ineligible = [], []
    for kind, subset, per_role, license_name in (
        ("noise", "free-sound", 8, "Public-domain"),
        ("music", "rfm", 4, "CC-BY-3.0"),
    ):
        license_path = data / f"audio-v3/research/musan-{kind}-{subset}-LICENSE.txt"
        declaration = license_path.read_text()
        if ("Public Domain" if kind == "noise" else "CC BY 3.0") not in declaration:
            raise ValueError("Pinned commercial-use-compatible background declaration required")
        pool = sorted(
            (r for r in inventory.values() if r["path"].startswith(kind + "/")),
            key=lambda r: stable(r["path"]),
        )
        count = 0
        for row in pool:
            ref, identifier = row["ref"], row["path"]
            if (
                identifier in denied_ids
                or Path(identifier).stem in denied_stems
                or ref["sha256"] in denied_bytes
            ):
                continue
            if kind == "music" and Path(identifier).stem not in declaration:
                raise ValueError("Selected musical work lacks its per-file attribution")
            path = Path(ref["source_path"]) if ref.get("source_path") else data / "audio-v3" / ref["path"]
            if digest(path) != ref["sha256"]:
                raise ValueError("Cached background original changed")
            audio = decode_audio(path, source_sha256=ref["sha256"])
            pcm_sha = audio.mapping["pcm_sha256"]
            if pcm_sha in denied_pcm or ref.get("decoded_pcm_sha256") in denied_pcm:
                continue
            if len(audio.pcm) < 128000 or float(np.sqrt(np.mean(audio.pcm.astype(float) ** 2))) < 1e-4:
                ineligible.append(
                    {
                        "source_id": identifier,
                        "reason": "Fewer than eight seconds or low whole-recording energy",
                    }
                )
                continue
            role = ROLES[count // per_role]
            stored = store(root, path.read_bytes())
            selected.append(
                {
                    **ref,
                    **stored,
                    "id": identifier,
                    "source_id": identifier,
                    "source_path": str((root / stored["path"]).resolve()),
                    "role": role,
                    "kind": kind,
                    "production_pcm_sha256": pcm_sha,
                    "source_production_input": audio.mapping,
                    "full_samples": len(audio.pcm),
                    "license": license_name,
                    "license_sha256": digest(license_path),
                    "license_url": f"https://huggingface.co/datasets/corypaik/musan/raw/{ref['revision']}/licenses/{kind}/{subset}/LICENSE",
                    "attribution": "Kevin MacLeod / Incompetech via MUSAN"
                    if kind == "music"
                    else "MUSAN / FreeSound public-domain per-file credits in retained LICENSE",
                }
            )
            denied_ids.add(identifier)
            denied_stems.add(Path(identifier).stem)
            denied_bytes.add(ref["sha256"])
            denied_pcm.add(pcm_sha)
            count += 1
            if count == per_role * len(ROLES):
                break
        if count != per_role * len(ROLES):
            raise ValueError("Insufficient unused permitted backgrounds for all four roles")
    return selected, ineligible


def prepare(root, output):
    if output.exists():
        raise ValueError("Preserve existing source preparation")
    planning = root / "source-planning"
    plan_path, ai_path, human_path = (
        planning / name for name in ("data-plan.json", "reserved-ai-sources.json", "new-genuine-sources.json")
    )
    plan, ai, human = (json.loads(p.read_text()) for p in (plan_path, ai_path, human_path))
    project = Path(__file__).resolve().parents[2]
    if plan["planning_code_sha256"] != digest(project / "backend/scripts/plan_audio_multisource_v8.py"):
        raise ValueError("Frozen selection code changed")
    for body, code in ((ai, "materialize_reserved_audio_v8.py"), (human, "acquire_genuine_parquet_v8.py")):
        if body["data_plan_sha256"] != digest(plan_path) or body["code_sha256"] != digest(
            project / "backend/scripts" / code
        ):
            raise ValueError("Reserved source bindings changed")
    if (
        human["ai_sources_sha256"] != digest(ai_path)
        or ai["eligibility_issues"]
        or ai["consumed_prompt_connections"]
    ):
        raise ValueError("Unresolved reserved-source eligibility/connection")
    sources = []
    for row in plan["development_sources"]:
        sources.append(checked_source(row, Path(row["source_path"])))
    for original in ai["sources"] + human["sources"]:
        row = {**original, "source_key": original["source_group"]}
        if row["role"] == "confirmation_foreground":
            row["role"] = "confirmation"
        elif row["role"].endswith("_background"):
            row["role"] = row["role"].removesuffix("_background")
            row["kind"] = "speech"
            row["source_id"] = row["id"]
        sources.append(checked_source(row, root / "reserved-sources" / row["path"]))
    assignments, groups = connected_roles(sources)
    for row in sources:
        row["group"] = assignments[row["id"]]
    foreground = [r for r in sources if r.get("kind") != "speech"]
    speech = [r for r in sources if r.get("kind") == "speech"]
    if Counter(r["role"] for r in foreground) != {
        "fit": 219,
        "cal": 99,
        "development": 158,
        "confirmation": 120,
    }:
        raise ValueError("Every reserved eligible foreground must remain present")
    if Counter(r["role"] for r in speech) != {"fit": 4, "cal": 4, "development": 4, "confirmation": 6}:
        raise ValueError("Every reserved background speaker must remain present")
    backgrounds, ineligible = select_backgrounds(root.parent, root / "prepared")
    all_bytes = [r["sha256"] for r in foreground + speech + backgrounds]
    if len(all_bytes) != len(set(all_bytes)):
        raise ValueError("Duplicate original foreground/background source")
    body = {
        "schema": "chipmunk.audio-multisource-prepared-sources/8",
        "data_plan_sha256": digest(plan_path),
        "reserved_ai_sha256": digest(ai_path),
        "reserved_genuine_sha256": digest(human_path),
        "additional_noise_inventory_sha256": digest(planning / "additional-noise-inventory.json"),
        "preparation_code_sha256": digest(Path(__file__)),
        "sources": foreground,
        "backgrounds": backgrounds + speech,
        "groups": groups,
        "exclusions": plan["exclusions"],
        "background_ineligible": ineligible,
        "role_counts": dict(Counter(r["role"] for r in foreground)),
        "role_generator_counts": {
            role: dict(Counter(r["generator"] for r in foreground if r["role"] == role)) for role in ROLES
        },
        "detector_inference_run": False,
        "fitting_started": False,
        "deployment_eligible": False,
        "limitations": plan["limitations"]
        + [
            "Only known metadata/text/byte/PCM connections are excluded; biometric identity and pretrained-corpus disjointness remain unverified.",
            "MUSAN noise contributor/session identities are unpublished, and distinct selected music works share a composer across roles.",
            "LAION scripts are intended provider text with bracket controls removed, not verified transcripts or speech alignment.",
            "All476 fitting/calibration/development sources were consumed in earlier experiments; only the120 reserved foreground originals are prospective confirmation.",
        ],
    }
    write(output, body)
    print(
        json.dumps(
            {
                "prepared_sources_sha256": digest(output),
                "sources": len(foreground),
                "backgrounds": len(backgrounds + speech),
                "role_counts": body["role_counts"],
                "known_connection_audit": "passed",
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
    prepare(args.root, args.output)
