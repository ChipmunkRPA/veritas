"""Prepare a frozen three-arm Turkish panel using shared recorded-noise residuals."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import wave
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np

from scripts import audio_alania_fresh_v213 as alania
from scripts import audio_drifting_turkish_v215 as drifting
from scripts import audio_shared_noise_v216 as shared
from scripts import audio_voxforge_tr_genuine_v212 as genuine
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, ARCHIVE = runtime.ROOT, runtime.ROOT / "backend/data/independent-evaluation", runtime.ARCHIVE
TRIAL = BASE / "audio-fresh-shared-noise-v217"
STAGE, OUTPUT = TRIAL / "bind-00", TRIAL / "bind-00/output"
PANEL = ROOT / "docs/evaluation-panels"
PUBLIC = PANEL / "audio-fresh-shared-noise-freeze-v217.json"
ASSET_RECORD = PANEL / "audio-demand-kitchen-traffic-assets-v212.json"
ASSET_RECORD_SHA = "70473811ceefa9063977400b070b49870229256180ed9f586055bb0ea73d6705"
SOURCE_RESULTS = {"genuine": PANEL / "audio-voxforge-tr-genuine-complete-v212.json",
                  "alania": PANEL / "audio-alania-fresh-complete-v213.json",
                  "drifting": PANEL / "audio-drifting-turkish-complete-v215.json"}
TAPE_ROOT = ARCHIVE / "audio-demand-sources-v212"
TAPE_FILES = {"kitchen": "DKITCHEN_16k.zip", "traffic": "STRAFFIC_16k.zip"}
MODULE = "scripts.audio_fresh_shared_noise_v217"
CONDITIONS = ("matched_clean", "kitchen_0db", "kitchen_5db", "traffic_0db", "traffic_5db")
CODE = ("backend/scripts/audio_fresh_shared_noise_v217.py", "backend/tests/test_audio_fresh_shared_noise_v217.py",
        "docs/audio-fresh-shared-noise-v217-method.md", "backend/scripts/audio_shared_noise_v216.py",
        "backend/scripts/audio_short_edit_controls.py", "backend/scripts/video_appearance_runtime_v108.py")
read, write, sha, require = runtime.read, runtime.write, runtime.sha, runtime.require


def model_free():
    require(not {"torch", "transformers", "onnxruntime", "app.detectors.audio.upgraded"} & sys.modules.keys(),
            "Fresh model-free preparation process required")


def publisher_assets():
    require(ASSET_RECORD.is_file() and sha(ASSET_RECORD) == ASSET_RECORD_SHA, "Exact published DEMAND asset record required")
    record = read(ASSET_RECORD)
    assets = {row["file"]: row for row in record["archives"]}
    require(record["schema"] == "chipmunk.audio-demand-source-admission/212" and record["status"] == "verified_archives_and_channel"
            and record["license"] == "CC-BY-SA-3.0" and record["channel"] == 1 and record["sample_rate_hz"] == 16000
            and set(assets) == set(TAPE_FILES.values()) and record["model_calls"] == 0,
            "Exact published noise source, license and channel required")
    for name in TAPE_FILES.values():
        path, row = TAPE_ROOT / name, assets[name]
        require(path.is_file() and not path.is_symlink() and path.resolve(strict=True) == path
                and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"], "Exact publisher tape archive required")
        with path.open("rb") as stream:
            require(hashlib.file_digest(stream, "md5").hexdigest() == row["publisher_md5"], "Publisher tape MD5 differs")
    return record


def tape(name, record):
    require(name in TAPE_FILES, "Only the two committed noise tapes are eligible")
    info = next(row for row in record["archives"] if row["file"] == TAPE_FILES[name])
    with zipfile.ZipFile(TAPE_ROOT / TAPE_FILES[name]) as archive:
        entries = [item for item in archive.infolist() if not item.is_dir() and PurePosixPath(item.filename).name == "ch01.wav"]
        require(len(entries) == 1 and entries[0].file_size <= 12_000_000, "One bounded publisher channel-one file required")
        body = archive.read(entries[0])
    with wave.open(io.BytesIO(body), "rb") as reader:
        require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype()) ==
                (1, 2, shared.SAMPLE_RATE, "NONE") and reader.getnframes() == info["channel_samples"],
                "Exact publisher mono 16 kHz channel required")
        pcm = reader.readframes(reader.getnframes())
        require(len(pcm) == 2 * info["channel_samples"] and not reader.readframes(1)
                and hashlib.sha256(pcm).hexdigest() == info["channel_pcm_sha256"], "Publisher channel PCM differs")
    return pcm


def source_rows():
    """Read only three completed preparation receipts, never detector scores."""
    model_free()
    require(genuine.readout() == read(SOURCE_RESULTS["genuine"]), "Exact published v212 genuine preparation required")
    genuine_rows = read(genuine.OUTPUT / "private-cases.json")
    alania_rows, alania_receipt = alania.verified_preparation()
    require(alania_receipt == read(SOURCE_RESULTS["alania"]), "Exact published v213 preparation required")
    require(drifting.readout() == read(SOURCE_RESULTS["drifting"]), "Exact published v215 preparation required")
    drift_rows = [json.loads(line) for line in (drifting.OUTPUT / "private-cases.jsonl").read_text().splitlines()]
    rosters = {"genuine": genuine_rows, "alania": alania_rows, "drifting": drift_rows}
    prefixes = {"genuine": "genuine", "alania": "positive", "drifting": "synthetic"}
    for arm, rows in rosters.items():
        require(len(rows) == 16 and [row["slot"] for row in rows] == [f"{prefixes[arm]}-{i:02d}" for i in range(16)]
                and all(row["status"] in {"qualified", "unavailable"} for row in rows),
                "Exact fixed 16-slot source roster required")
    model_free()
    return rosters


def identity():
    rosters, record = source_rows(), publisher_assets()
    return {"schema": "chipmunk.audio-fresh-shared-noise-freeze/217", "planned_triplets": 16, "planned_cases": 240,
            "arms": list(shared.ARMS), "conditions": list(CONDITIONS), "source_range": [0, shared.SAMPLES],
            "tape_start_seconds": 4, "tape_step_seconds": 6, "sample_rate": shared.SAMPLE_RATE,
            "source_receipts_sha256": {arm: sha(path) for arm, path in SOURCE_RESULTS.items()},
            "demand_asset_record_sha256": ASSET_RECORD_SHA, "publisher_record": record["publisher_record"],
            "noise_license": record["license"], "code_sha256": {name: sha(ROOT / name) for name in CODE},
            "source_dispositions": {arm: dict(Counter(row["status"] for row in rows)) for arm, rows in rosters.items()},
            "model_calls_permitted": False, "detector_calls_permitted": False, "holdout_access_permitted": False}, rosters


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists(), "Preserve every earlier attempt")
    lock, _ = identity()
    record = publisher_assets()
    for name in shared.TAPES:
        tape(name, record)
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", lock)
    write(PUBLIC, lock)
    return lock


def verify():
    lock, rosters = identity()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == lock, "Exact frozen fresh shared-noise panel required")
    return rosters


def signed_prefix(pcm):
    source = np.asarray(pcm, dtype=np.float64)
    require(source.ndim == 1 and len(source) >= shared.SAMPLES and np.isfinite(source).all(), "Three seconds of finite source PCM required")
    values = source[:shared.SAMPLES] * 32768
    require((values == np.rint(values)).all() and np.min(values) >= -32768 and np.max(values) <= 32767,
            "Source prefix must preserve exact signed 16-bit production samples")
    return values.astype("<i2").tobytes()


def offsets(index):
    require(type(index) is int and 0 <= index < 16, "One fixed source roster position required")
    return (4 + 6 * index) * shared.SAMPLE_RATE


def source_readers():
    return {"genuine": genuine.read_prepared_pcm,
            "alania": scoped(alania.transfer.read_pcm, {"prepared_dir": alania.prepared_dir}),
            "drifting": drifting.read_prepared_pcm}


def cases(rosters, channels, readers):
    require(set(rosters) == set(readers) == set(shared.ARMS) and set(channels) == set(shared.TAPES)
            and all(len(rows) == 16 and all(row.get("status") in {"qualified", "unavailable"} for row in rows)
                    for rows in rosters.values()), "All planned source rosters and tapes required")
    for index in range(16):
        start = offsets(index)
        excerpts = {name: pcm[2 * start:2 * (start + shared.SAMPLES)] for name, pcm in channels.items()}
        require(all(len(body) == 2 * shared.SAMPLES for body in excerpts.values()), "Noise tape excerpt is outside its committed extent")
        raw, reasons = {}, {}
        for arm in shared.ARMS:
            row = rosters[arm][index]
            if row["status"] != "qualified":
                reasons[arm] = "source_unavailable"
                continue
            pcm = readers[arm](row)
            if len(pcm) < shared.SAMPLES:
                reasons[arm] = "source_shorter_than_three_seconds"
                continue
            raw[arm] = signed_prefix(pcm)
        treatment = None
        if raw:
            try:
                treatment = shared.mix_shared_noise(raw, excerpts)
            except shared.UnsupportedControl as error:
                for arm in raw:
                    reasons[arm] = "mix_unavailable:" + str(error)
        for arm in shared.ARMS:
            row = rosters[arm][index]
            for condition in CONDITIONS:
                base = {"case": f"triplet-{index:02d}-{arm}-{condition}", "triplet": f"triplet-{index:02d}",
                        "arm": arm, "condition": condition}
                if arm in reasons:
                    yield {**base, "status": "unavailable", "reason": reasons[arm]}, None
                    continue
                variant = treatment.variants[arm]
                key = None if condition == "matched_clean" else (condition.rsplit("_", 1)[0], int(condition.rsplit("_", 1)[1][:-2]))
                payload = variant.clean_s16le if key is None else variant.mixed_s16le[key]
                residual = None if key is None else treatment.noise_s16le[key]
                yield {**base, "status": "prepared", "samples": shared.SAMPLES,
                       "source_wav_sha256": row["wav_sha256"], "source_prefix_sha256": hashlib.sha256(raw[arm]).hexdigest(),
                       "pcm_sha256": hashlib.sha256(payload).hexdigest(), "common_gain": treatment.common_gain,
                       "level_match_gain": variant.level_match_gain, "reference_arm": treatment.reference_arm,
                       "noise_sample_offset": start if key else None,
                       "noise_excerpt_sha256": hashlib.sha256(excerpts[key[0]]).hexdigest() if key else None,
                       "noise_residual_sha256": hashlib.sha256(residual).hexdigest() if residual is not None else None,
                       "measured_snr_db": variant.measured_snr_db[key] if key else None}, payload


def save_wave(path, pcm):
    with path.open("xb") as stream, wave.open(stream, "wb") as writer:
        writer.setparams((1, 2, shared.SAMPLE_RATE, shared.SAMPLES, "NONE", "not compressed"))
        writer.writeframes(pcm)
    body = path.read_bytes()
    with wave.open(io.BytesIO(body), "rb") as reader:
        require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes()) ==
                (1, 2, shared.SAMPLE_RATE, shared.SAMPLES) and reader.readframes(shared.SAMPLES) == pcm,
                "Saved PCM must match the constructed waveform")
    return hashlib.sha256(body).hexdigest()


def summary(rows):
    expected = [(f"triplet-{index:02d}", arm, condition)
                for index in range(16) for arm in shared.ARMS for condition in CONDITIONS]
    require([(row["triplet"], row["arm"], row["condition"]) for row in rows] == expected
            and all(row["status"] in {"prepared", "unavailable"} for row in rows),
            "Every planned ordered triplet, arm and condition must be present")
    lookup = {(row["triplet"], row["arm"], row["condition"]): row for row in rows}
    outcomes = {arm: {condition: dict(Counter(row["status"] for row in rows if row["arm"] == arm and row["condition"] == condition))
                      for condition in CONDITIONS} for arm in shared.ARMS}
    coverage = {}
    for condition in CONDITIONS:
        complete = [[lookup[f"triplet-{index:02d}", arm, condition]["status"] == "prepared" for arm in shared.ARMS]
                    for index in range(16)]
        coverage[condition] = {"planned_triplets": 16, "complete_triplets": sum(all(flags) for flags in complete),
            "genuine_alania": sum(flags[0] and flags[1] for flags in complete),
            "genuine_drifting": sum(flags[0] and flags[2] for flags in complete),
            "alania_drifting": sum(flags[1] and flags[2] for flags in complete)}
    return {"by_arm_and_condition": outcomes, "pair_coverage": coverage,
            "unavailable_reasons": dict(Counter(row["reason"] for row in rows if row["status"] == "unavailable"))}


def worker():
    rosters = verify()
    require((STAGE / "resources.jsonl").is_file() and (STAGE / "worker.log").is_file()
            and not (STAGE / "profile.json").exists(), "Use the owned data-only v108 guard")
    OUTPUT.mkdir(mode=0o700)
    prepared = OUTPUT / "prepared"
    prepared.mkdir(mode=0o700)
    record = publisher_assets()
    channels = {name: tape(name, record) for name in shared.TAPES}
    rows = []
    journal = OUTPUT / "private-cases.jsonl"
    with journal.open("x") as stream:
        for row, payload in cases(rosters, channels, source_readers()):
            if payload is not None:
                row["wav_sha256"] = save_wave(prepared / f"{row['case']}.wav", payload)
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    model_free()
    write(OUTPUT / "result.json", {"schema": "chipmunk.audio-fresh-shared-noise-result/217", "status": "complete",
          "freeze_sha256": sha(PUBLIC), "private_cases_sha256": sha(journal), "planned_cases": len(rows),
          **summary(rows), "new_neural_calls": 0, "detector_calls": 0, "holdout_scored": False})


def run():
    def launch(argv, **kwargs):
        require(argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
                "Exact guarded data-only worker required")
        return subprocess.Popen([*argv[:3], MODULE, "_worker"], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(runtime.run, {"TRIAL": TRIAL, "verify": verify, "subprocess": proxy,
                         "prerequisites": lambda *_: True, "__file__": __file__})("bind", 0)


def readout():
    rosters = verify()
    journal, result_path = OUTPUT / "private-cases.jsonl", OUTPUT / "result.json"
    from scripts.audio_voxforge_tr_transfer_v201 import verify_data_guard

    verify_data_guard(read(STAGE / "profile.json"), STAGE, result_path, sha(Path(__file__)))
    result, rows = read(result_path), [json.loads(line) for line in journal.read_text().splitlines()]
    require(result["schema"] == "chipmunk.audio-fresh-shared-noise-result/217" and result["status"] == "complete"
            and result["new_neural_calls"] == result["detector_calls"] == 0 and result["holdout_scored"] is False
            and result["freeze_sha256"] == sha(PUBLIC) and result["private_cases_sha256"] == sha(journal)
            and result["planned_cases"] == len(rows) == 240, "Exact complete guarded preparation required")
    record = publisher_assets()
    channels = {name: tape(name, record) for name in shared.TAPES}
    for (expected, pcm), row in zip(cases(rosters, channels, source_readers()), rows, strict=True):
        if pcm is not None:
            path = OUTPUT / "prepared" / f"{expected['case']}.wav"
            require(path.is_file() and not path.is_symlink(), "Direct prepared waveform required")
            body = path.read_bytes()
            with wave.open(io.BytesIO(body), "rb") as reader:
                require((reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getnframes()) ==
                        (1, 2, shared.SAMPLE_RATE, shared.SAMPLES) and reader.readframes(shared.SAMPLES) == pcm,
                        "Saved waveform changed")
            expected["wav_sha256"] = hashlib.sha256(body).hexdigest()
        require(row == expected, "Reconstructed shared-noise case differs")
    require(all(result[key] == value for key, value in summary(rows).items()), "Coverage or pair counts differ")
    model_free()
    return {**result, "profile_sha256": sha(STAGE / "profile.json"),
            "scope": "Three prepared Turkish source cohorts and two recorded noise tapes. No detector output or population claim."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "readout", "_worker"))
    args = parser.parse_args()
    result = {"freeze": freeze, "run": run, "readout": readout, "_worker": worker}[args.action]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))
