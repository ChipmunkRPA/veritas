"""Metadata-only history exclusions for fresh speech testing; no customer files or inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from scripts.audio_source_identity_v9 import speaker_key

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / "backend/data/independent-evaluation/audio-history-v58"
SHA = re.compile(r"[0-9a-f]{64}")
PROFILE = re.compile(r"(?:kokoro(?:-82m)?:)?([ab][fm]_[a-z0-9_]+)")
PCM_FIELDS = ("decoded_pcm_sha256", "production_pcm_sha256", "prefix_pcm_sha256",
              "source_prefix_pcm_sha256", "pcm_sha256", "production_input_sha256")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def text_keys(text):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text or "").casefold())
    if not words:
        return set()
    return {"text:" + " ".join(words), "prefix8:" + " ".join(words[:8])}


def voice_key(row):
    if row.get("kind") in ("noise", "music"):
        return None
    generator = str(row.get("generator") or row.get("generator_engine") or "").casefold()
    if "kokoro" in generator:
        for field in ("tts_voice", "generator_voice", "voice_key", "speaker", "cluster_id"):
            match = PROFILE.fullmatch(str(row.get(field) or "").casefold())
            if match:
                return "kokoro-82m:" + match[1]
        raise ValueError("Known stock generator has no usable voice-profile identity")
    dataset = row.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        return None
    normalized = dict(row)
    normalized["id"] = str(row.get("id") or "")
    if dataset.casefold() == "duplexgen/duplexgen-spoken" and not row.get("prompt_speaker_id"):
        normalized["prompt_speaker_id"] = row.get("user_prompt_speaker_id")
    declared = row.get("canonical_speaker")
    if declared and re.fullmatch(r"(?:librispeech:[0-9]+|vctk:p[0-9]{3}|elevenlabs:[0-9a-f]{64})", str(declared)):
        return str(declared)
    return speaker_key(normalized)


def row_keys(row):
    keys = set()
    for field in ("sha256", "source_sha256", "original_sha256"):
        value = row.get(field)
        if isinstance(value, str) and SHA.fullmatch(value):
            keys.add("bytes:" + value)
    for field in PCM_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and SHA.fullmatch(value):
            # Equal full/prefix hashes conservatively connect even if old schemas named them differently.
            keys.add("pcm:" + value)
    keys.update(text_keys(row.get("transcript") if isinstance(row.get("transcript"), str) else ""))
    voice = voice_key(row)
    if voice:
        keys.add("voice:" + voice)
    return keys


def records(body, pointer=""):
    if isinstance(body, dict):
        if (any(isinstance(body.get(k), str) and SHA.fullmatch(body[k])
                for k in ("sha256", "source_sha256", "original_sha256"))
                and any(key in body for key in ("dataset", "id", "base_id", "source_id", "path", "source_path", "transcript"))):
            yield pointer, body
        for key, value in body.items():
            if isinstance(value, dict | list):
                yield from records(value, pointer + "/" + key)
    elif isinstance(body, list):
        for index, value in enumerate(body):
            yield from records(value, pointer + "/" + str(index))


def select_manifests(base):
    names = {"manifest.json", "prepared-sources.json", "source-plan.json", "audio-manifest.json",
             "api-parity-manifest.json", "challenge-manifest.json", "video-manifest.json"}
    return sorted(path for path in base.rglob("*.json") if path.name in names
                  and path.relative_to(base).parts[0].startswith(("audio", "lipsync", "mouth", "video", "multimodal", "rule80", "public-watermark")))


def audit():
    base = ROOT / "backend/data/independent-evaluation"
    paths = select_manifests(base)
    TRIAL.mkdir(exist_ok=False)
    plan = {str(path.relative_to(base)): digest(path) for path in paths}
    (TRIAL / "input-bindings.json").write_text(json.dumps(plan, indent=2) + "\n")
    keys, voices, byte_hashes, unresolved, references = set(), set(), set(), [], 0
    by_history = Counter()
    for path in paths:
        relative = str(path.relative_to(base))
        body = json.loads(path.read_text())
        for pointer, row in records(body):
            references += 1
            by_history[path.relative_to(base).parts[0]] += 1
            try:
                found = row_keys(row)
            except ValueError as error:
                # Keep all byte/text exclusions even when known voice metadata is unresolved.
                safe = {k: v for k, v in row.items() if k in {"sha256", "source_sha256", "original_sha256", "transcript", *PCM_FIELDS}}
                found = row_keys(safe)
                unresolved.append({"manifest": relative, "pointer": pointer, "reason": str(error)})
            keys.update(found)
            voices.update(k[6:] for k in found if k.startswith("voice:"))
            byte_hashes.update(k[6:] for k in found if k.startswith("bytes:"))
        if digest(path) != plan[relative]:
            raise ValueError("A historical source manifest changed while reading")
    private = {"schema": "chipmunk.audio-history-private-catalogue/58", "keys": sorted(keys),
               "voice_keys": sorted(voices), "unresolved": unresolved,
               "scope": "Declared historical metadata only, including contextual media references; not biometric or pretraining novelty proof."}
    (TRIAL / "catalogue.json").write_text(json.dumps(private, indent=2) + "\n")
    metadata = ROOT / "backend/data/independent-evaluation/audio-fresh-source-preparation-v53/hexgrad--Kokoro-82M/metadata.json"
    available = {Path(row["rfilename"]).stem for row in json.loads(metadata.read_text())["siblings"]
                 if row["rfilename"].startswith("voices/") and PROFILE.fullmatch(Path(row["rfilename"]).stem)}
    used = {voice.split(":", 1)[1] for voice in voices if voice.startswith("kokoro-82m:")}
    result = {"schema": "chipmunk.audio-history-summary/58", "status": "complete", "historical_manifests": len(paths),
              "input_bindings_sha256": digest(TRIAL / "input-bindings.json"), "private_catalogue_sha256": digest(TRIAL / "catalogue.json"),
              "metadata_references": references, "unique_recorded_byte_hashes": len(byte_hashes), "exclusion_keys": len(keys),
              "declared_voice_keys": len(voices), "voice_keys_by_family": dict(Counter(voice.split(":", 1)[0] for voice in voices)),
              "references_by_history": dict(by_history), "known_voice_metadata_unresolved": len(unresolved),
              "kokoro_english_stock_profiles_in_pinned_metadata": len(available), "kokoro_known_historical_profiles": len(used),
              "kokoro_pinned_english_profiles_seen_in_history": len(available & used),
              "kokoro_pinned_english_profiles_without_declared_history_match": len(available - used),
              "generator_metadata_sha256": digest(metadata), "code_sha256": digest(Path(__file__)),
              "canonical_identity_helper_sha256": digest(Path(__file__).with_name("audio_source_identity_v9.py")),
              "new_model_calls": 0, "research_audio_downloaded": False, "new_source_roles_assigned": False,
              "scope_limits": ["Reference counts include mixtures and contextual records, not independent original recordings.",
                               "Metadata without a resolved speaker identity cannot establish voice independence.",
                               "No biometric matching, pretraining audit, customer upload or unmanifested runtime fixture is covered.",
                               "No unmatched stock profile is yet a qualified fresh source or assigned benchmark role."],
              "defaults_changed": False, "workflow_runs": 0, "outreach_sent": False, "deployment_done": False}
    (TRIAL / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"scope_limits", "references_by_history"}}), flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    audit()
