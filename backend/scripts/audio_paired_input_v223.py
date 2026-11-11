"""Read the three consumed v84 roles without importing an audio model."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scripts.audio_score_fusion_v83 import predict

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "backend/data/independent-evaluation"
PANEL = ROOT / "docs/evaluation-panels"
TRIAL = BASE / "audio-combined-confidence-v84"
FULL = Path(
    "/Volumes/SEAGATE/Codex-Development-Archive/2026-11-08-Chipmunk-Verify/audio-confidence-extraction-v84"
)
VARIANTS = BASE / "audio-confidence-mixtures-v82/variants.json"
MANIFEST = BASE / "audio-complete-sources-v81/manifest.json"
TRAINED = TRIAL / "trained.json"
FREEZE = "audio-combined-confidence-v84-freeze.json"
TRAINING = "audio-combined-confidence-v84-training.json"
SELECTION = "audio-combined-confidence-v84-selection.json"
PUBLIC = {
    FREEZE: "5a6cefd6d09b7a78b5699f2a625aeaff3d0886d57621be916df04be134e31b17",
    TRAINING: "b30169d22d5ce6a4b4b97f6262b8f3fc9bd1c363bab6037e697e3b8c737204d1",
    SELECTION: "25aa0c565e003b9b578046aa155be24e37d3667f75d7d8ebe09a94b52aa209e5",
}
VARIANTS_SHA = "8411c76325bc4dcde649db08ed05747c5eb917df13d811d1068b7395ee562d24"
MANIFEST_SHA = "8b667675a803bd03851c27251e08a63b040b2bd29474462f9f0a4e3af9108ea6"
TRAINED_SHA = "0aa21efdad351eebbc2390f241c49f1e40d55d188277db442f092c7b1ef668ae"
ROLES = ("fusion_fit", "confidence_cal", "selection")
CONDITIONS = ("clean", "noise_0", "noise_5")
CHUNK, ROLE_CHUNKS, SAMPLES, WIDTH, FRAMES = 87, 12, 64000, 4608, 199
META = ("id", "base_id", "group", "role", "label", "kind", "condition", "sha256", "pcm_sha256")
FIELDS = (*META[:-2], "raw", "original_p", "original_decision", "candidates", "wavlm", "status")
SOURCE_KEYS = (
    "raw_sha256",
    "pcm_sha256",
    "prefix_pcm_sha256",
    "first4_pcm_sha256",
    "source_id",
    "source_group",
    "study_group",
    "canonical_speaker",
    "text_identity",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def chunk_paths(index):
    require(
        type(index) is int and 0 <= index < len(ROLES) * ROLE_CHUNKS,
        "Only the three consumed v84 roles are permitted",
    )
    tag = f"chunk-{index:02d}"
    return TRIAL / f"{tag}.jsonl", FULL / f"{tag}-full.jsonl", TRIAL / f"{tag}-profile"


def check_profile(index, path, expected):
    profile = read(path)
    require(
        sha(path) == expected
        and profile["schema"] == "chipmunk.audio-shared-fixed-monitor-profile/84"
        and profile["module"] == "scripts.audio_combined_confidence_v84"
        and profile["arguments"] == ["extract", str(index)]
        and profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and not profile["resource_limit_observations"]
        and profile["resource_journal_sha256"] == sha(path.parent / "resource-samples.jsonl"),
        "Exact successful original v84 resource profile required",
    )


def source_identity():
    """Verify metadata and file hashes only, without parsing feature rows."""
    require(
        all(sha(PANEL / name) == digest for name, digest in PUBLIC.items())
        and sha(TRIAL / "freeze.json") == PUBLIC[FREEZE]
        and sha(TRIAL / "training-summary.json") == PUBLIC[TRAINING]
        and sha(TRIAL / "selection.json") == PUBLIC[SELECTION]
        and sha(VARIANTS) == VARIANTS_SHA
        and sha(MANIFEST) == MANIFEST_SHA
        and sha(TRAINED) == TRAINED_SHA,
        "Exact v81/v82 metadata and v84 public/private lineage required",
    )
    freeze, training, selection = (read(PANEL / name) for name in (FREEZE, TRAINING, SELECTION))
    require(
        freeze["inputs_sha256"] == VARIANTS_SHA
        and freeze["source_manifest_sha256"] == MANIFEST_SHA
        and selection["freeze_sha256"] == PUBLIC[FREEZE]
        and selection["trained_sha256"] == training["trained_sha256"] == TRAINED_SHA
        and training["selection_inputs_used"]
        == training["confirmation_inputs_used"]
        == selection["confirmation_inputs_used"]
        == 0,
        "Exact role-specific v84 fitting and selection receipts required",
    )
    stages = {
        "fusion_fit": training["fit_stages"],
        "confidence_cal": training["calibration_stages"],
        "selection": selection["source_stages"],
    }
    for number, (role, bindings) in enumerate(stages.items()):
        indices = range(number * ROLE_CHUNKS, (number + 1) * ROLE_CHUNKS)
        require(set(bindings) == {str(i) for i in indices}, f"Complete {role} chunk roster required")
        for i in indices:
            compact, full, guard = chunk_paths(i)
            expected = bindings[str(i)]
            hashes = {"completed": CHUNK, "compact_sha256": sha(compact), "full_sha256": sha(full)}
            require(
                {**hashes, "profile_sha256": expected["profile_sha256"]} == expected
                and read(TRIAL / f"chunk-{i:02d}-complete.json") == hashes,
                "Complete exact v84 compact and full journal digests required",
            )
            check_profile(i, guard / "profile.json", expected["profile_sha256"])
    return {
        "schema": "chipmunk.audio-paired-inputs/223",
        "v84_public_sha256": PUBLIC.copy(),
        "v81_source_manifest_sha256": MANIFEST_SHA,
        "v82_variants_sha256": VARIANTS_SHA,
        "v83_global_trained_sha256": TRAINED_SHA,
        "v84_consumed_stages": stages,
        "roles": list(ROLES),
        "conditions": list(CONDITIONS),
        "planned_per_role": 348,
        "confirmation_features_read": False,
    }


def role_sources(rows):
    owners = {}
    indexed = {}
    for row in rows:
        raw = row["raw_sha256"]
        require(raw not in indexed, "Duplicate source bytes in the consumed roles")
        indexed[raw] = row
        for field in SOURCE_KEYS:
            value = row.get(field)
            for identity in value if isinstance(value, list) else (value,):
                if identity:
                    key = (field, identity)
                    require(
                        owners.setdefault(key, row["role"]) == row["role"],
                        "Known source, speaker, group, text or background identity crosses v223 roles",
                    )
    return indexed


def check_variant(row, foregrounds, backgrounds):
    source = foregrounds[row["source_sha256"]]
    require(
        row["role"] == source["role"]
        and row["group"] == source["study_group"]
        and row["label"] == source["label"]
        and row["kind"] == source["kind"]
        and row["base_id"] == "foreground:" + row["source_sha256"]
        and row["samples"] == SAMPLES
        and row["sample_rate"] == 16000
        and row["source_interval_samples"] == [0, SAMPLES],
        "Exact original foreground identity, role and four-second clock required",
    )
    background = row["background"]
    if row["condition"] == "clean":
        require(background is None, "The matched clean source may not have a background")
        return None
    noise = backgrounds[background["raw_sha256"]]
    require(
        noise["role"] == row["role"]
        and noise["study_group"] == background["group"]
        and background["end_sample"] - background["start_sample"] == SAMPLES
        and 0 <= background["start_sample"] < background["end_sample"] <= noise["samples"],
        "Noise recording, role and exact excerpt required",
    )
    return background["raw_sha256"]


def valid_score(value):
    return type(value) is float and math.isfinite(value) and 0 <= value <= 1


def observation(full, meta, background_sha256, global_head):
    if full["status"] == "not_applicable":
        require(
            all(
                full[key] is None
                for key in (
                    "raw",
                    "original_p",
                    "wavlm",
                    "wavlm_sha256",
                    "feature_frames",
                    "valid_feature_frames",
                    "input_pcm_sha256",
                )
            )
            and all(candidate["p"] is None for candidate in full["candidates"].values())
            and full["windows"] == []
            and full["original_decision"] == "inconclusive",
            "Unassessed original inputs cannot supply a voice feature or score",
        )
    else:
        feature = np.asarray(full["wavlm"], dtype="<f4")
        require(
            full["status"] == "ok"
            and valid_score(full["raw"])
            and valid_score(full["original_p"])
            and valid_score(full["candidates"]["complement"]["p"])
            and feature.shape == (WIDTH,)
            and np.isfinite(feature).all()
            and feature.tolist() == full["wavlm"]
            and hashlib.sha256(feature.tobytes()).hexdigest() == full["wavlm_sha256"]
            and full["feature_frames"] == full["valid_feature_frames"] == FRAMES
            and full["input_pcm_sha256"] == meta["pcm_sha256"]
            and full["windows"] == [{"start_sample": 0, "end_sample": SAMPLES}],
            "Exact finite WavLM feature vector and original input sample clock required",
        )
    score = predict(full, global_head)["probability_ai"]
    require(
        (full["status"] == "ok" and valid_score(score))
        or (full["status"] == "not_applicable" and score is None),
        "Frozen v83 global response must respect the original speech gate",
    )
    return {
        **{key: full[key] for key in FIELDS},
        "background_sha256": background_sha256,
        "v83_global_p": score,
    }


def read_chunk(index, expected, bindings, foregrounds, backgrounds, global_head):
    compact, full, _ = chunk_paths(index)
    rows = []
    ch, fh = hashlib.sha256(), hashlib.sha256()
    count = 0
    with compact.open("rb") as compact_stream, full.open("rb") as full_stream:
        for short_line, full_line, meta in zip(compact_stream, full_stream, expected, strict=True):
            ch.update(short_line)
            fh.update(full_line)
            short, complete = json.loads(short_line), json.loads(full_line)
            require(
                short["full_record_sha256"] == hashlib.sha256(full_line).hexdigest()
                and all(complete[key] == value for key, value in short.items() if key != "full_record_sha256")
                and all(complete[key] == meta[key] for key in META),
                "Exact compact/full record and variant lineage required",
            )
            count += 1
            if meta["condition"] in CONDITIONS:
                background_sha256 = check_variant(meta, foregrounds, backgrounds)
                rows.append(observation(complete, meta, background_sha256, global_head))
    require(
        count == bindings["completed"] == CHUNK
        and ch.hexdigest() == bindings["compact_sha256"]
        and fh.hexdigest() == bindings["full_sha256"],
        "Complete unchanged v84 journal streams required",
    )
    return rows


def check_role(role, rows, planned):
    require(
        len(rows) == 348
        and all(row["role"] == role for row in rows)
        and Counter((row["condition"], row["label"]) for row in rows)
        == {(condition, label): count for condition in CONDITIONS for label, count in ((0, 56), (1, 60))},
        "Every fixed clean and environmental-noise slot is required",
    )
    groups = defaultdict(dict)
    for row in planned:
        if row["condition"] in CONDITIONS:
            groups[row["base_id"]][row["condition"]] = row
    require(
        len(groups) == 116
        and all(
            set(group) == set(CONDITIONS)
            and group["noise_0"]["background"] == group["noise_5"]["background"]
            and group["noise_0"]["background_interval_pcm_sha256"]
            == group["noise_5"]["background_interval_pcm_sha256"]
            for group in groups.values()
        )
        and len({row["id"] for row in rows}) == len(rows)
        and len({row["background_sha256"] for row in rows if row["background_sha256"]}) == 4,
        "Exact distinct foregrounds and within-source paired noise excerpts required",
    )


def load_roles():
    """Return eligible conditions by consumed role and the public-safe source commitment."""
    commitment = source_identity()
    manifest = read(MANIFEST)
    foregrounds = role_sources([row for row in manifest["foregrounds"] if row["role"] in ROLES])
    backgrounds = role_sources(
        [row for row in manifest["backgrounds"] if row["role"] in ROLES and row["kind"] == "noise"]
    )
    require(
        Counter(row["role"] for row in foregrounds.values()) == {role: 116 for role in ROLES}
        and Counter(row["role"] for row in backgrounds.values()) == {role: 4 for role in ROLES},
        "Exact independent foreground and environmental-noise rosters required",
    )
    variants = read(VARIANTS)["variants"]
    global_head = read(TRAINED)["global"]
    roles = {}
    for number, role in enumerate(ROLES):
        start, end = number * ROLE_CHUNKS, (number + 1) * ROLE_CHUNKS
        selected = []
        planned = variants[start * CHUNK : end * CHUNK]
        require(
            len(planned) == ROLE_CHUNKS * CHUNK and all(row["role"] == role for row in planned),
            "Exact role order required",
        )
        bindings = commitment["v84_consumed_stages"][role]
        for index in range(start, end):
            chunk = variants[index * CHUNK : (index + 1) * CHUNK]
            selected.extend(
                read_chunk(index, chunk, bindings[str(index)], foregrounds, backgrounds, global_head)
            )
        check_role(role, selected, planned)
        roles[role] = selected
    return roles, commitment
