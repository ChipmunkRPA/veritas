"""Prepare paired environmental-noise controls from fixed Turkish speech sources."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import wave
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np

from scripts import audio_alania_transfer_v179 as synthetic
from scripts import audio_short_edit_controls as controls
from scripts import audio_voxforge_tr_genuine_v199 as genuine
from scripts import audio_voxforge_tr_transfer_v201 as genuine_route
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, ARCHIVE = genuine.ROOT, genuine.BASE, runtime.ARCHIVE
TRIAL = BASE / "audio-demand-controls-v204"
SOURCES = ARCHIVE / "audio-demand-sources-v204"
STAGE = TRIAL / "bind-00"
OUTPUT = STAGE / "output"
PUBLIC = ROOT / "docs/evaluation-panels/audio-demand-controls-freeze-v204.json"
MODULE = "scripts.audio_demand_controls_v204"
RECORD = "https://zenodo.org/records/1227121"
MD5 = {
    "DWASHING_16k.zip": "7e5ee9437ce9409c5f9a779b6212a240",
    "NRIVER_16k.zip": "54264db61d3fe073fb81f2e40e0d19b5",
    "DEMAND.pdf": "a513afe131e08cdacf7c1854b174d5ab",
}
TAPES = {"washing_room": "DWASHING_16k.zip", "river": "NRIVER_16k.zip"}
RATE = controls.SAMPLE_RATE
START_SECONDS, STEP_SECONDS = 4, 6
MIN_SAMPLES, MAX_SAMPLES = 40000, 64000
LEVELS = controls.SNR_LEVELS_DB
CONDITIONS = ("source_clean", "matched_clean", *(f"{noise}_{level}db" for noise in TAPES for level in LEVELS))
read, write, sha, require = genuine.read, genuine.write, genuine.sha, genuine.require


def asset_identity():
    assets = {}
    for name, expected in MD5.items():
        path = SOURCES / name
        require(
            path.is_file() and path.resolve() == path and path.stat().st_size <= 115_000_000,
            "Only the bounded official publisher files are allowed",
        )
        with path.open("rb") as stream:
            md5 = hashlib.file_digest(stream, "md5").hexdigest()
        require(md5 == expected, "Official DEMAND checksum differs")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assets[name] = {"publisher_md5": md5, "sha256": digest, "bytes": path.stat().st_size}
    return assets


def read_wave(body):
    with wave.open(io.BytesIO(body), "rb") as reader:
        require(
            (reader.getnchannels(), reader.getsampwidth(), reader.getframerate(), reader.getcomptype())
            == (1, 2, RATE, "NONE"),
            "Expected mono signed16 PCM at exactly 16 kHz",
        )
        count = reader.getnframes()
        pcm = reader.readframes(count)
        require(len(pcm) == count * 2 and not reader.readframes(1), "Incomplete source PCM")
    return pcm


def tape(name):
    with zipfile.ZipFile(SOURCES / TAPES[name]) as archive:
        members = [
            item
            for item in archive.infolist()
            if not item.is_dir() and PurePosixPath(item.filename).name == "ch01.wav"
        ]
        require(
            len(members) == 1 and members[0].file_size <= 12_000_000,
            "Exactly one bounded publisher channel-one WAV required",
        )
        pcm = read_wave(archive.read(members[0]))
    require(
        len(pcm) >= 2 * RATE * (START_SECONDS + 47 * STEP_SECONDS + 4),
        "The complete frozen series of nonoverlapping source windows is required",
    )
    return pcm


def foregrounds():
    original, _ = genuine_route.verified_preparation()
    selected, _ = synthetic.roster_commitments()
    generated, _ = synthetic.verified_preparation(selected)
    require(len(original) == len(generated) == 24, "Every frozen original and synthetic slot is required")
    return [("genuine", row) for row in original] + [("synthetic", row) for row in generated]


def identity():
    sources = foregrounds()
    paths = (
        Path(__file__),
        ROOT / "backend/tests/test_audio_demand_controls_v204.py",
        ROOT / "docs/audio-demand-controls-v204-method.md",
        ROOT / "backend/scripts/audio_short_edit_controls.py",
        ROOT / "backend/scripts/audio_voxforge_tr_transfer_v201.py",
        ROOT / "backend/scripts/audio_alania_transfer_v179.py",
        ROOT / "backend/scripts/video_appearance_runtime_v108.py",
    )
    fixed = {
        "schema": "chipmunk.audio-demand-controls-freeze/204",
        "publisher_record": RECORD,
        "publisher_assets": asset_identity(),
        "environment_license": "CC-BY-SA-3.0",
        "foreground_results_sha256": {
            path.name: sha(path)
            for path in (
                ROOT / "docs/evaluation-panels/audio-voxforge-tr-transfer-complete-v201.json",
                genuine_route.pilot.ALANIA_RESULT,
            )
        },
        "private_preparation_sha256": {
            "genuine": sha(genuine.OUTPUT / "private-cases.json"),
            "synthetic": sha(synthetic.TRIAL / "bind-00/output/private-preparation.json"),
        },
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in paths},
        "source_slots": len(sources),
        "cases_planned": len(sources) * len(CONDITIONS),
        "source_outcomes": dict(Counter(f"{origin}_{row['status']}" for origin, row in sources)),
        "conditions": list(CONDITIONS),
        "channel": 1,
        "nominal_snr_db": list(LEVELS),
        "window_start_seconds": START_SECONDS,
        "window_step_seconds": STEP_SECONDS,
        "model_calls_permitted": False,
    }
    return fixed, sources


def freeze():
    require(not PUBLIC.exists() and not TRIAL.exists(), "Preserve any prior preparation")
    frozen, _ = identity()
    for name in TAPES:
        tape(name)
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)
    return frozen


def verify():
    frozen, sources = identity()
    require(read(PUBLIC) == read(TRIAL / "freeze.json") == frozen, "Exact frozen preparation required")
    return sources


def signed_pcm(pcm):
    values = np.asarray(pcm, dtype=np.float64) * 32768
    require(
        MIN_SAMPLES <= len(values) <= MAX_SAMPLES
        and np.isfinite(values).all()
        and (values == np.rint(values)).all()
        and np.max(values) <= 32767
        and np.min(values) >= -32768,
        "Prepared foreground must reconstruct exact signed samples",
    )
    return values.astype("<i2").tobytes()


def variants(foreground, background):
    count = len(foreground) // 2
    require(
        MIN_SAMPLES <= count <= MAX_SAMPLES
        and len(foreground) == 2 * count
        and tuple(background) == tuple(TAPES),
        "Exact variable-length foreground and both backgrounds required",
    )
    sampler = scoped(controls._samples, {"SAMPLES": count})
    mix = scoped(controls.mix_paired_noise, {"_samples": sampler})
    one = {"source": foreground}
    gain = min(mix(one, reference="source", noise_pcm_s16le=pcm).common_gain for pcm in background.values())
    result = {"source_clean": (foreground, None, None, 1.0)}
    for name, pcm in background.items():
        treatment = mix(one, reference="source", noise_pcm_s16le=pcm, common_gain=gain)
        arm = treatment.variants["source"]
        clean = (arm.clean_s16le, None, None, gain)
        require(
            result.setdefault("matched_clean", clean) == clean, "Both environments require one common gain"
        )
        clean_samples = np.frombuffer(clean[0], dtype="<i2").astype(np.int32)
        for level in LEVELS:
            payload, residual = arm.mixed_s16le[level], treatment.noise_s16le[level]
            actual = np.frombuffer(payload, dtype="<i2").astype(np.int32) - clean_samples
            measured = arm.measured_snr_db[level]
            require(
                np.array_equal(actual, np.frombuffer(residual, dtype="<i2")) and abs(measured - level) <= 0.1,
                "Exact quantized residual and measured SNR required",
            )
            result[f"{name}_{level}db"] = (payload, residual, measured, gain)
    require(tuple(result) == CONDITIONS, "Fixed condition order required")
    return result


def save_wave(path, pcm):
    with path.open("xb") as stream, wave.open(stream, "wb") as writer:
        writer.setparams((1, 2, RATE, len(pcm) // 2, "NONE", "not compressed"))
        writer.writeframes(pcm)
    body = path.read_bytes()
    require(read_wave(body) == pcm, "Saved mix failed exact PCM verification")
    return hashlib.sha256(body).hexdigest()


def cases(sources, channels):
    for ordinal, (origin, row) in enumerate(sources):
        slot = f"source-{ordinal:02d}"
        offset = (START_SECONDS + STEP_SECONDS * ordinal) * RATE
        if row["status"] == "qualified":
            pcm = genuine.read_prepared_pcm(row) if origin == "genuine" else synthetic.read_pcm(row)
            clean = signed_pcm(pcm)
            noise = {
                name: channel[2 * offset : 2 * offset + len(clean)] for name, channel in channels.items()
            }
            prepared = variants(clean, noise)
        else:
            prepared = None
        for condition in CONDITIONS:
            record = {
                "case": f"{slot}-{condition}",
                "source": slot,
                "origin": origin,
                "condition": condition,
                "source_preparation": row.get("selection", row.get("render_mode")),
            }
            if prepared is None:
                yield (
                    {**record, "status": "unavailable", "reason": "foreground_preparation_unavailable"},
                    None,
                )
                continue
            payload, residual, measured, gain = prepared[condition]
            yield (
                {
                    **record,
                    "status": "prepared",
                    "samples": len(payload) // 2,
                    "foreground_wav_sha256": row["wav_sha256"],
                    "pcm_sha256": hashlib.sha256(payload).hexdigest(),
                    "gain": gain,
                    "background_sample_offset": offset if residual is not None else None,
                    "background_residual_sha256": hashlib.sha256(residual).hexdigest()
                    if residual is not None
                    else None,
                    "measured_snr_db": measured,
                },
                payload,
            )


def worker():
    sources = verify()
    require(
        (STAGE / "resources.jsonl").is_file()
        and (STAGE / "worker.log").is_file()
        and not (STAGE / "profile.json").exists(),
        "Use the owned v108 data-only guard",
    )
    OUTPUT.mkdir(mode=0o700)
    prepared = OUTPUT / "prepared"
    prepared.mkdir(mode=0o700)
    journal = OUTPUT / "private-cases.jsonl"
    channels = {name: tape(name) for name in TAPES}
    counts = Counter()
    with journal.open("x") as stream:
        for row, payload in cases(sources, channels):
            if payload is not None:
                row["wav_sha256"] = save_wave(prepared / f"{row['case']}.wav", payload)
            counts[row["status"]] += 1
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
    write(
        OUTPUT / "result.json",
        {
            "schema": "chipmunk.audio-demand-controls-result/204",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_cases_sha256": sha(journal),
            "outcomes": dict(counts),
            "new_neural_calls": 0,
        },
    )


def run():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
            "Exact owned preparation command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "_worker"], **kwargs)

    proxy = SimpleNamespace(Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired)
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "subprocess": proxy,
            "prerequisites": lambda *_: True,
            "__file__": __file__,
        },
    )("bind", 0)


def readout():
    sources = verify()
    journal = OUTPUT / "private-cases.jsonl"
    result_path, profile_path = OUTPUT / "result.json", STAGE / "profile.json"
    genuine_route.verify_data_guard(read(profile_path), STAGE, result_path, sha(Path(__file__)))
    result = read(result_path)
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    require(
        result["schema"] == "chipmunk.audio-demand-controls-result/204"
        and result["status"] == "complete"
        and result["new_neural_calls"] == 0
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_cases_sha256"] == sha(journal),
        "Prepared case journal differs",
    )
    channels = {name: tape(name) for name in TAPES}
    distributions = {}
    for (expected, payload), row in zip(cases(sources, channels), rows, strict=True):
        if payload is not None:
            path = OUTPUT / "prepared" / f"{expected['case']}.wav"
            body = path.read_bytes()
            require(read_wave(body) == payload, "Exact recorded waveform differs")
            expected["wav_sha256"] = hashlib.sha256(body).hexdigest()
        require(row == expected, "Independent waveform and source replay differs from the private journal")
        statuses = distributions.setdefault(row["origin"], {}).setdefault(row["condition"], Counter())
        statuses[row["status"]] += 1
    require(result["outcomes"] == dict(Counter(row["status"] for row in rows)), "Case coverage differs")
    return {
        **result,
        "by_origin_and_condition": distributions,
        "profile_sha256": sha(STAGE / "profile.json"),
        "scope": "Two recorded environments on fixed Turkish foregrounds. No detector outputs or noisy recording population estimate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "readout", "_worker"))
    action = parser.parse_args().action
    result = {"freeze": freeze, "run": run, "readout": readout, "_worker": worker}[action]()
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
