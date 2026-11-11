"""Freeze and prepare a matched three-second English babble development panel."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import types
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from app.detectors.audio.decode import decode_audio, source_ranges
from scripts import audio_short_edit_controls as controls
from scripts.audio_short_edit_cases_v172 import LEVELS, float_pcm_sha, write_case
from scripts.video_appearance_corrected_v112 import scoped
from scripts.video_appearance_native_v108 import ARCHIVE, BASE, ROOT, read, require, sha, write

NAME = "audio-matched-babble-v181"
MODULE = "scripts.audio_matched_babble_v181"
TRIAL, MEDIA = BASE / NAME, ARCHIVE / NAME
STAGE = TRIAL / "bind-00"
OUTPUT = STAGE / "output"
PANELS = ROOT / "docs/evaluation-panels"
PUBLIC = PANELS / "audio-matched-babble-freeze-v181.json"
V81 = BASE / "audio-complete-sources-v81/manifest.json"
V81_RESULT = PANELS / "audio-complete-sources-v81.json"
HISTORY = BASE / "audio-history-v58/catalogue.json"
HOLDOUT = BASE / "audio-short-edit-cases-v172/holdout/private-plan.json"
V172_FREEZE = PANELS / "audio-short-edit-case-freeze-v172.json"
PACC = BASE / "audio-pacc-p-independent-sample-v176"
PACC_PUBLISHER = ARCHIVE / PACC.name
V176_FREEZE = PANELS / "audio-pacc-sample-freeze-v176.json"
V176_RESULT = PANELS / "audio-pacc-sample-result-v176.json"
V177_FREEZE = PANELS / "audio-pacc-false-alarm-freeze-v177.json"
VCTK = BASE / "audio-v3"
VCTK_METADATA = BASE / "audio-v2/metadata"
VCTK_TEXT_REVISION = "fb74847570d78d2b23e83193d8e55df80e6271b2"
DFADD_REVISION = "dfc1eeab3cb0068db8e87a2b89a1ebd103665b1f"
FOREGROUND_ROOT = ARCHIVE / "audio-calibration-sourcebank-v67"
RATE, SAMPLES = controls.SAMPLE_RATE, controls.SAMPLES
ARMS = ("genuine", "synthetic")
POOLS, TALKERS, PAIRS = 4, 6, 16
CASE_COUNT = PAIRS * len(ARMS) * len(LEVELS)
MAX_SOURCE_BYTES = 2_000_000
MAX_SOURCE_SECONDS = 30
VOICE_ID = re.compile(r"p[0-9]+")
RECORDING_ID = re.compile(r"(p[0-9]+)_[0-9]+")
HASH = re.compile(r"[0-9a-f]{64}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def ranked(key):
    return digest(f"chipmunk.audio-matched-babble/181:{key}".encode())


def voice_id(value):
    value = value.strip().casefold()
    return f"vctk:{value}" if VOICE_ID.fullmatch(value) else value


def declared_hashes(rows):
    return {
        value
        for row in rows
        for key, value in row.items()
        if key.endswith("_sha256") and isinstance(value, str) and HASH.fullmatch(value)
    }


def reference_metadata():
    holdout_freeze, holdout = read(V172_FREEZE), read(HOLDOUT)
    holdout_manifest = holdout["source_manifest"]
    require(
        holdout["role"] == holdout_manifest["role"] == "holdout"
        and holdout_manifest["scored"] is False
        and holdout_freeze["holdout_scored"] is False
        and holdout_freeze["roles"]["holdout"]["private_plan_sha256"] == sha(HOLDOUT),
        "Exact v172 holdout exclusion metadata required",
    )
    plan, observations = PACC / "private-plan.json", PACC / "bind-00/output/private-observations.json"
    pacc_freeze, pacc_result, observation_freeze = read(V176_FREEZE), read(V176_RESULT), read(V177_FREEZE)
    pacc_plan, pacc_rows = read(plan), read(observations)
    credits = [PACC_PUBLISHER / name for name in ("README.md", "attribution.csv")]
    require(
        observation_freeze["v176_public_freeze_sha256"] == sha(V176_FREEZE)
        and observation_freeze["v176_public_result_sha256"] == sha(V176_RESULT)
        and pacc_result["freeze_sha256"] == sha(V176_FREEZE)
        and pacc_result["private_observations_sha256"] == sha(observations)
        and pacc_freeze["private_plan_sha256"] == sha(plan)
        and len(pacc_plan) == 14
        and len(pacc_rows) == 98
        and all(row["status"] == "qualified" for row in pacc_rows)
        and all(sha(path) == pacc_freeze["publisher_metadata_sha256"][path.name] for path in credits)
        and "MUSAN's LibriVox speech" in credits[0].read_text(),
        "Exact v177 observation inputs and publisher babble attribution required",
    )
    holdout_rows = [*holdout_manifest["foregrounds"], *holdout_manifest["noise"]]
    exclusions = {
        "v172_holdout": {
            "voices": sorted(
                {voice_id(row["voice"]) for row in holdout_manifest["foregrounds"] if row.get("voice")}
            ),
            "hashes": sorted(declared_hashes(holdout_rows)),
        },
        "v177_observation": {
            "voices": sorted({voice_id(row["speaker"]) for row in pacc_plan}),
            "hashes": sorted(declared_hashes(pacc_rows)),
        },
    }
    return exclusions, [
        V172_FREEZE,
        HOLDOUT,
        V176_FREEZE,
        V176_RESULT,
        V177_FREEZE,
        plan,
        observations,
        *credits,
    ]


def matched_foregrounds(manifest, voices, fingerprints):
    catalogue = {row["source_id"]: row for row in manifest["foregrounds"]}
    pairs = []
    for ai in catalogue.values():
        if ai["role"] != "fusion_fit" or ai["kind"] != "stock":
            continue
        genuine = catalogue[ai["matched_genuine_source_id"]]
        require(
            genuine["role"] == "fusion_fit"
            and genuine["kind"] == "genuine"
            and genuine["label"] == 0
            and genuine["license"] == "CC-BY-4.0"
            and genuine["dataset"] == "openslr/librispeech_asr"
            and ai["label"] == 1
            and ai["license"] == "Apache-2.0"
            and ai["generator"] == "Kokoro-82M"
            and genuine["pcm_sha256"] == ai["matched_genuine_pcm_sha256"]
            and genuine["transcript"] == ai["transcript"]
            and genuine["study_group"] == ai["study_group"],
            "Exact matched development foreground ancestry required",
        )
        pair = {"component": genuine["study_group"]}
        for arm, row in zip(ARMS, (genuine, ai), strict=True):
            identity = voice_id(row["canonical_speaker"])
            require(
                row["status"] == "ready"
                and SAMPLES <= row["samples"] < MAX_SOURCE_SECONDS * RATE
                and identity not in voices
                and not declared_hashes([row]) & fingerprints,
                "Foreground is unqualified or overlaps protected source metadata",
            )
            pair[arm] = {
                "id": row["source_id"],
                "voice": identity,
                "source_path": row["source_path"],
                "source_bytes": row["source_bytes"],
                "original_sha256": row["raw_sha256"],
                "prior_pcm_sha256": row["pcm_sha256"],
                "prior_samples": row["samples"],
                "license": row["license"],
            }
        pairs.append(pair)
    components = Counter(row["component"] for row in pairs)
    require(
        len(pairs) == PAIRS
        and sorted(components.values()) == [POOLS] * POOLS
        and all(len({pair[arm]["voice"] for pair in pairs}) == POOLS for arm in ARMS)
        and len({pair[arm]["original_sha256"] for pair in pairs for arm in ARMS}) == PAIRS * len(ARMS),
        "Exact four connected foreground components and 16 matched pairs required",
    )
    pool_for_component = {component: i for i, component in enumerate(sorted(components, key=ranked))}
    for index, pair in enumerate(sorted(pairs, key=lambda row: ranked(row["genuine"]["id"]))):
        pair["pair"] = f"pair-{index:02d}"
        pair["pool"] = f"pool-{pool_for_component[pair['component']]:02d}"
    return sorted(pairs, key=lambda row: row["pair"])


def babble_pools(candidates):
    selected, voices, fingerprints = [], set(), set()
    for row in sorted(candidates, key=lambda item: ranked(item["original_sha256"])):
        hashes = declared_hashes([row])
        if row["voice"] in voices or fingerprints & hashes:
            continue
        selected.append(row)
        voices.add(row["voice"])
        fingerprints.update(hashes)
        if len(selected) == POOLS * TALKERS:
            break
    require(len(selected) == POOLS * TALKERS, "24 distinct eligible VCTK talkers required")
    return {f"pool-{i:02d}": selected[i * TALKERS : (i + 1) * TALKERS] for i in range(POOLS)}


def babble_candidates(voices, fingerprints, history):
    vctk_index = VCTK_METADATA / "vctk-index.json"
    member_index = VCTK / "metadata/dfadd-VCTK_BONAFIDE-index.json"
    vctk_card = VCTK_METADATA / "jspaulsen_vctk-publisher.json"
    vctk_license = VCTK_METADATA / "VCTK-LICENSE.txt"
    dfadd_license = VCTK_METADATA / "DFADD-LICENSE.txt"
    dfadd_readme = VCTK_METADATA / "DFADD-README.md"
    publisher = read(vctk_card)
    require(
        publisher["id"] == "jspaulsen/vctk"
        and publisher["sha"] == VCTK_TEXT_REVISION
        and publisher["cardData"]["license"] == "cc-by-4.0"
        and "creativecommons.org/licenses/by/4.0/legalcode" in vctk_license.read_text()
        and "MIT License" in dfadd_license.read_text(),
        "Pinned VCTK CC BY 4.0 and DFADD MIT credit records required",
    )
    transcripts = {row["row_idx"]: row for row in read(vctk_index)}
    members = {row["name"]: row for row in read(member_index)}
    caches = sorted((VCTK / "metadata").glob("vctk-source-*.json"))
    eligible = []
    for path in caches:
        row = read(path)
        recording = path.stem.removeprefix("vctk-source-")
        match = RECORDING_ID.fullmatch(recording)
        require(match is not None, "Exact VCTK recording identifier required")
        voice = voice_id(match.group(1))
        raw, prior_pcm = row["sha256"], row["decoded_pcm_sha256"]
        if (
            voice in voices
            or row["decoded_samples"] < SAMPLES
            or {raw, prior_pcm} & fingerprints
            or {f"bytes:{raw}", f"pcm:{prior_pcm}"} & history
        ):
            continue
        member, transcript = members[row["archive_member"]], transcripts[row["transcript_row_index"]]
        require(
            row["dataset"] == "isjwdu/DFADD"
            and row["revision"] == DFADD_REVISION
            and row["url"]
            == f"https://huggingface.co/datasets/isjwdu/DFADD/resolve/{DFADD_REVISION}/DATASET_VCTK_BONAFIDE.zip"
            and Path(member["name"]).stem == recording
            and member["bytes"] == row["bytes"]
            and transcript["id"] == f"{recording}_mic1"
            and voice_id(transcript["speaker_id"]) == voice
            and row["transcript_dataset"] == "jspaulsen/vctk"
            and row["transcript_revision"] == VCTK_TEXT_REVISION
            and row["microphone"] == "not documented by archive publisher"
            and HASH.fullmatch(raw)
            and Path(row["path"]) == Path("sources") / f"{raw}.audio",
            "VCTK publisher archive member, transcript row or byte identity differs",
        )
        eligible.append(
            {
                "id": recording,
                "voice": voice,
                "source_path": str(VCTK / row["path"]),
                "source_bytes": row["bytes"],
                "original_sha256": raw,
                "prior_pcm_sha256": prior_pcm,
                "publisher_member": member["name"],
                "publisher_member_crc32": member["crc32"],
                "transcript_row_index": row["transcript_row_index"],
                "license": "CC-BY-4.0",
            }
        )
    metadata = [vctk_index, member_index, vctk_card, vctk_license, dfadd_license, dfadd_readme, *caches]
    return eligible, metadata


def frozen_inputs():
    exclusions, references = reference_metadata()
    protected_voices = {voice for row in exclusions.values() for voice in row["voices"]}
    protected_hashes = {value for row in exclusions.values() for value in row["hashes"]}
    v81_result = read(V81_RESULT)
    require(v81_result["manifest_sha256"] == sha(V81), "Exact qualified v81 source manifest required")
    recorded_history = read(HISTORY)
    history = set(recorded_history["keys"])
    pairs = matched_foregrounds(read(V81), protected_voices, protected_hashes)
    foregrounds = [pair[arm] for pair in pairs for arm in ARMS]
    candidates, metadata = babble_candidates(
        protected_voices | {row["voice"] for row in foregrounds},
        protected_hashes | declared_hashes(foregrounds),
        history,
    )
    pools = babble_pools(candidates)
    private = {
        "role": "development",
        "pairs": pairs,
        "pools": pools,
        "exclusions": exclusions,
        "noise_history_bytes_pcm_only": sorted(key for key in history if key.startswith(("bytes:", "pcm:"))),
    }
    code = [
        Path(__file__),
        ROOT / "backend/tests/test_audio_matched_babble_v181.py",
        ROOT / "docs/audio-matched-babble-v181.md",
        Path(controls.__file__),
        ROOT / "backend/scripts/audio_short_edit_cases_v172.py",
        ROOT / "backend/scripts/video_appearance_runtime_v108.py",
        ROOT / "backend/app/detectors/audio/decode.py",
        ROOT / "backend/app/detectors/video/decode.py",
    ]
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(tools.values()), "Existing production FFmpeg tools required")
    commitments = [(path.name, sha(path)) for path in [V81, V81_RESULT, HISTORY, *references, *metadata]]
    public = {
        "schema": "chipmunk.audio-matched-babble-freeze/181",
        "role": "development",
        "source_metadata_sha256": digest(json.dumps(commitments, separators=(",", ":")).encode()),
        "source_metadata_records": len(commitments),
        "v81_manifest_sha256": sha(V81),
        "v172_holdout_plan_sha256": sha(HOLDOUT),
        "v177_input_observations_sha256": read(V176_RESULT)["private_observations_sha256"],
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in code},
        "ffmpeg_tool_sha256": {name: sha(Path(path)) for name, path in tools.items()},
        "planned_cases": CASE_COUNT,
        "planned_pairs": PAIRS,
        "foreground_components": POOLS,
        "foregrounds_by_arm": {arm: PAIRS for arm in ARMS},
        "cases_by_condition": {condition: PAIRS * len(ARMS) for condition, _ in LEVELS},
        "nominal_snr_db": dict(LEVELS),
        "babble_pools": POOLS,
        "talkers_per_pool": TALKERS,
        "eligible_vctk_recordings": len(candidates),
        "eligible_vctk_declared_voices": len({row["voice"] for row in candidates}),
        "babble_voices_previously_recorded": len(
            {row["voice"] for pool in pools.values() for row in pool}
            & {voice_id(value) for value in recorded_history["voice_keys"]}
        ),
        "sample_rate": RATE,
        "samples_per_case": SAMPLES,
        "input_decode_limit_seconds": MAX_SOURCE_SECONDS,
        "overlap_scope": "Declared voices and available original/production PCM receipts for v172 holdout and v177 inputs. VCTK babble parents also exclude v58 byte/PCM hits.",
        "new_neural_calls": 0,
        "holdout_media_opened": False,
        "holdout_scored": False,
    }
    return public, private


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists() and not MEDIA.exists(), "Preserve every prior attempt")
    public, private = frozen_inputs()
    TRIAL.mkdir(mode=0o700)
    plan = TRIAL / "private-plan.json"
    write(plan, private)
    public["private_plan_sha256"] = sha(plan)
    write(TRIAL / "freeze.json", public)
    write(PUBLIC, public)
    return public


def verify():
    public, private = frozen_inputs()
    plan = TRIAL / "private-plan.json"
    public["private_plan_sha256"] = sha(plan)
    require(
        read(PUBLIC) == read(TRIAL / "freeze.json") == public and read(plan) == private,
        "Exact metadata, plan, exclusion scope and code freeze required",
    )
    return public, private


def admitted_path(row, kind):
    path = Path(row["source_path"])
    banks = (
        {VCTK / "sources"}
        if kind == "babble"
        else {FOREGROUND_ROOT / f"qualified-{version}/sources" for version in ("v72", "v81")}
    )
    require(
        path.parent in banks
        and path.resolve(strict=True) == path
        and path.is_file()
        and path.stat().st_size == row["source_bytes"] <= MAX_SOURCE_BYTES,
        "Exact direct source in the admitted development bank required",
    )
    return path


def production_prefix(row, kind, excluded):
    path = admitted_path(row, kind)
    original = row["original_sha256"]
    if original in excluded:
        raise ValueError("original_bytes_overlap")
    result = decode_audio(path, source_sha256=original, max_seconds=MAX_SOURCE_SECONDS)
    mapping = result.mapping
    require(
        mapping["source_sha256"] == original
        and mapping["sample_rate"] == RATE
        and mapping["sample_format"] == "s16le"
        and mapping["channels"] == 1
        and len(result.pcm) == mapping["samples"]
        and SAMPLES <= mapping["samples"] < MAX_SOURCE_SECONDS * RATE,
        "Source is short, reached its limit or differs from the production input format",
    )
    if kind != "babble":
        require(
            (mapping["samples"], mapping["pcm_sha256"]) == (row["prior_samples"], row["prior_pcm_sha256"]),
            "Qualified foreground full production PCM changed",
        )
    ranges = source_ranges(mapping, 0, SAMPLES)
    require(len(ranges) == 1, "Discontinuous original three-second source clock")
    with wave.open(io.BytesIO(result.wav), "rb") as stream:
        raw_pcm = stream.readframes(SAMPLES)
    require(len(raw_pcm) == SAMPLES * 2, "Short production PCM prefix")
    prefix_sha = float_pcm_sha(raw_pcm)
    signed_sha = digest(raw_pcm)
    fingerprints = {original, mapping["pcm_sha256"], prefix_sha, signed_sha}
    require(not fingerprints & excluded, "Known source byte or exact PCM overlap")
    return (
        raw_pcm,
        {
            "source_range": ranges[0],
            "full_pcm_sha256": mapping["pcm_sha256"],
            "prefix_pcm_sha256": prefix_sha,
            "prefix_s16le_sha256": signed_sha,
        },
        fingerprints,
    )


def make_babble(parents):
    require(len(parents) == TALKERS, "Exact six-talker babble required")
    levels = []
    for payload in parents:
        require(len(payload) == SAMPLES * 2, "Expected an exact three-second talker")
        samples = np.frombuffer(payload, dtype="<i2").astype(np.float64)
        rms = float(np.sqrt(np.mean(samples**2)))
        if not np.ptp(samples) or rms < 3.2768:
            raise controls.UnsupportedControl("silent_or_constant_babble_talker")
        levels.append(samples / rms)
    total = np.sum(levels, axis=0)
    peak = float(np.max(np.abs(total)))
    if not np.ptp(total):
        raise controls.UnsupportedControl("constant_babble_pool")
    return np.rint(total * (controls.PEAK_TARGET / peak)).astype("<i2").tobytes()


def cases(pair, foregrounds, babble):
    """Return all six receipts and optional PCM for one immutable comparison."""
    ready = {arm: foregrounds[arm] for arm in ARMS if arm in foregrounds}
    treatment = None
    if len(ready) != len(ARMS):
        reason = "paired_foreground_unavailable"
    elif babble is None:
        reason = "fixed_babble_pool_unavailable"
    else:
        try:
            treatment = controls.mix_paired_noise(ready, reference="genuine", noise_pcm_s16le=babble)
            reason = None
        except controls.UnsupportedControl as error:
            reason = str(error)
    for arm in ARMS:
        for condition, level in LEVELS:
            row = {
                "case": f"{pair['pair']}-{arm}-{condition}",
                "pair": pair["pair"],
                "component": pair["component"],
                "pool": pair["pool"],
                "arm": arm,
                "condition": condition,
                "nominal_snr_db": level,
                "label": int(arm == "synthetic"),
                "ai_support_samples": [0, SAMPLES] if arm == "synthetic" else None,
                "background_speech_origin": "genuine_competing_speech" if level is not None else None,
            }
            if arm not in ready or (level is not None and treatment is None):
                yield {**row, "status": "unavailable", "reason": reason}, None
                continue
            payload, residual, measured, gain = ready[arm], None, None, 1.0
            if treatment:
                variant = treatment.variants[arm]
                payload, gain = variant.clean_s16le, treatment.common_gain
                if level is not None:
                    payload = variant.mixed_s16le[level]
                    residual = treatment.noise_s16le[level]
                    observed = np.frombuffer(payload, dtype="<i2").astype(np.int32) - np.frombuffer(
                        variant.clean_s16le, dtype="<i2"
                    )
                    require(
                        np.array_equal(observed, np.frombuffer(residual, dtype="<i2")),
                        "Babble residual differs",
                    )
                    measured = variant.measured_snr_db[level]
            yield (
                {
                    **row,
                    "status": "constructed",
                    "gain": gain,
                    "shared_noise_pcm_sha256": digest(residual) if residual else None,
                    "measured_snr_db": measured,
                    "complete_paired_treatment": treatment is not None,
                },
                payload,
            )


def worker():
    frozen, plan = verify()
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and not OUTPUT.exists()
        and not MEDIA.exists(),
        "One exclusive guarded data-only preparation is required",
    )
    OUTPUT.mkdir(mode=0o700)
    MEDIA.mkdir(mode=0o700)
    exclusions = {value for row in plan["exclusions"].values() for value in row["hashes"]}
    history = {key.split(":", 1)[1] for key in plan["noise_history_bytes_pcm_only"]}
    accepted, source_receipts = set(), []

    def prepare(row, kind):
        source = {"id": row["id"], "voice": row["voice"], "kind": kind}
        excluded = exclusions | accepted
        if kind == "babble":
            excluded.update(history)
        try:
            payload, evidence, fingerprints = production_prefix(row, kind, excluded)
        except (OSError, ValueError, EOFError, wave.Error) as error:
            source_receipts.append({**source, "status": "unavailable", "private_reason": str(error)})
            return None
        accepted.update(fingerprints)
        source_receipts.append({**source, **evidence, "status": "qualified"})
        return payload

    foregrounds = {
        pair["pair"]: {arm: pcm for arm in ARMS if (pcm := prepare(pair[arm], arm)) is not None}
        for pair in plan["pairs"]
    }
    pools, pool_outcomes = {}, {}
    for pool, sources in plan["pools"].items():
        payloads = [prepare(row, "babble") for row in sources]
        if any(payload is None for payload in payloads):
            pool_outcomes[pool] = {"status": "unavailable", "reason": "incomplete_fixed_talker_pool"}
            continue
        try:
            pools[pool] = make_babble(payloads)
        except controls.UnsupportedControl as error:
            pool_outcomes[pool] = {"status": "unavailable", "reason": str(error)}
            continue
        pool_outcomes[pool] = {"status": "constructed", "pcm_s16le_sha256": digest(pools[pool])}
    receipts = OUTPUT / "private-source-receipts.json"
    write(receipts, {"sources": source_receipts, "pools": pool_outcomes})

    journal_path = OUTPUT / "private-cases.jsonl"
    outcomes = Counter()
    by_condition = {condition: Counter() for condition, _ in LEVELS}
    with journal_path.open("x") as journal:
        for pair in plan["pairs"]:
            for row, payload in cases(pair, foregrounds[pair["pair"]], pools.get(pair["pool"])):
                if payload is not None:
                    try:
                        row.update(write_case(MEDIA / f"{row['case']}.wav", payload))
                    except (OSError, ValueError, wave.Error) as error:
                        row |= {
                            "status": "failed",
                            "reason": "wav_write_or_roundtrip",
                            "private_error": str(error),
                        }
                outcomes[row["status"]] += 1
                by_condition[row["condition"]][row["status"]] += 1
                journal.write(json.dumps(row, allow_nan=False) + "\n")
                journal.flush()
    require(sum(outcomes.values()) == CASE_COUNT, "Every planned case must have a receipt")
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-matched-babble-result/181",
            "status": "complete",
            "role": "development",
            "freeze_sha256": sha(PUBLIC),
            "private_plan_sha256": frozen["private_plan_sha256"],
            "private_case_journal_sha256": sha(journal_path),
            "private_source_receipt_sha256": sha(receipts),
            "planned_cases": CASE_COUNT,
            "case_outcomes": outcomes,
            "condition_outcomes": by_condition,
            "source_outcomes_by_kind": {
                kind: Counter(row["status"] for row in source_receipts if row["kind"] == kind)
                for kind in (*ARMS, "babble")
            },
            "babble_pool_outcomes": Counter(row["status"] for row in pool_outcomes.values()),
            "new_neural_calls": 0,
            "holdout_media_opened": False,
            "holdout_scored": False,
        },
    )


def run():
    from scripts import video_appearance_runtime_v108 as runtime

    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
            "Only the fixed babble data worker may run",
        )
        return subprocess.Popen([*argv[:3], MODULE, "worker"], **kwargs)

    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda *_: not MEDIA.exists(),
            "subprocess": types.SimpleNamespace(
                Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
            ),
            "__file__": __file__,
        },
    )("bind", 0)


def readout():
    frozen, plan = verify()
    profile = read(STAGE / "profile.json")
    result_path, journal = OUTPUT / "result.json", OUTPUT / "private-cases.jsonl"
    result = read(result_path)
    require(
        profile["child_started"] is True
        and profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["worker_receipt_sha256"] == sha(result_path)
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and result["schema"] == "chipmunk.audio-matched-babble-result/181"
        and result["status"] == "complete"
        and result["role"] == "development"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_plan_sha256"] == frozen["private_plan_sha256"]
        and result["private_case_journal_sha256"] == sha(journal)
        and result["private_source_receipt_sha256"] == sha(OUTPUT / "private-source-receipts.json")
        and result["planned_cases"] == CASE_COUNT
        and result["new_neural_calls"] == profile["new_neural_calls"] == 0
        and result["holdout_media_opened"] is result["holdout_scored"] is False,
        "Exact completed guarded development preparation required",
    )
    with journal.open() as stream:
        rows = [json.loads(line) for line in stream]
    expected = [
        (pair["pair"], pair["component"], pair["pool"], arm, condition)
        for pair in plan["pairs"]
        for arm in ARMS
        for condition, _ in LEVELS
    ]
    require(
        [(row["pair"], row["component"], row["pool"], row["arm"], row["condition"]) for row in rows]
        == expected
        and all(
            row["case"] == f"{row['pair']}-{row['arm']}-{row['condition']}"
            and row["status"] in {"constructed", "unavailable", "failed"}
            and row["label"] == int(row["arm"] == "synthetic")
            and row["ai_support_samples"] == ([0, SAMPLES] if row["label"] else None)
            for row in rows
        )
        and Counter(row["status"] for row in rows) == result["case_outcomes"]
        and {
            condition: Counter(row["status"] for row in rows if row["condition"] == condition)
            for condition, _ in LEVELS
        }
        == result["condition_outcomes"],
        "Every fixed comparison arm and condition must be accounted for",
    )
    for row in rows:
        if row["status"] == "constructed":
            path = MEDIA / f"{row['case']}.wav"
            require(
                Path(row["file"]) == path
                and path.resolve(strict=True) == path
                and sha(path) == row["wav_sha256"],
                "Prepared development WAV differs",
            )
    return {
        **result,
        "resource_profile_sha256": sha(STAGE / "profile.json"),
        "worker_duration_s": profile["duration_s"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "readout", "worker"))
    command = parser.parse_args().command
    outcome = {"freeze": freeze, "run": run, "readout": readout, "worker": worker}[command]()
    if outcome is not None:
        print(json.dumps(outcome, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
