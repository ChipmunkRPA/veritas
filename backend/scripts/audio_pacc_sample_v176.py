"""Admit matched publisher-prepared genuine audio through the production decoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import types
from collections import Counter
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from app.detectors.audio.decode import SAMPLE_RATE, decode_audio, read_wav, source_ranges
from scripts import video_appearance_runtime_v108 as runtime
from scripts.video_appearance_corrected_v112 import scoped
from scripts.video_appearance_native_v108 import ARCHIVE, BASE, ROOT, read, require, sha, write

NAME = "audio-pacc-p-independent-sample-v176"
MODULE = "scripts.audio_pacc_sample_v176"
TRIAL, SOURCE = BASE / NAME, ARCHIVE / NAME
MEDIA = SOURCE / "prepared"
SAMPLE = SOURCE / "pacc-p_sample.tar"
STAGE = TRIAL / "bind-00"
OUTPUT = STAGE / "output"
PUBLIC = ROOT / "docs/evaluation-panels/audio-pacc-sample-freeze-v176.json"
METHOD = ROOT / "docs/audio-pacc-sample-v176-method.md"
HISTORY = BASE / "audio-history-v58/catalogue.json"
PUBLISHER_REVISION = "b6c9e5f2ff0e3d60f90ade6d37d79245207a4f64"
SAMPLE_SIZE = 154_777_600
SAMPLE_SHA256 = "878caa7fbc67c06380f99c2e165fd8d306c7c0229ce00e03c475e35076c14b99"
CONDITIONS = {
    "clean": None,
    "babble_snr00": 0,
    "babble_snr05": 5,
    "env_noise_snr00": 0,
    "env_noise_snr05": 5,
    "music_instrumental_snr00": 0,
    "music_instrumental_snr05": 5,
}
POOLS = {"AMI": 4, "VCTK_mic1": 5, "VCTK_mic2": 5}
SOURCE_LICENSES = {"CC-BY-3.0", "CC0-1.0", "CC-BY", "PD", "public domain"}
SAMPLES = 3 * SAMPLE_RATE
MAX_MEMBER = 2_000_000


def digest(data):
    return hashlib.sha256(data).hexdigest()


def member(archive, name):
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts, "Unsafe archive path")
    entry = archive.getmember(name)
    require(entry.isfile() and 0 < entry.size <= MAX_MEMBER, "Expected a bounded regular archive member")
    with archive.extractfile(entry) as stream:
        raw = stream.read(MAX_MEMBER + 1)
    require(len(raw) == entry.size, "Truncated archive member")
    return raw


def rows(archive, name):
    with io.StringIO(member(archive, name).decode("utf-8-sig"), newline="") as stream:
        return list(csv.DictReader(stream))


def match_member(name, folder, file_id):
    path = PurePosixPath(name)
    require(
        tuple(path.parts[:-1]) == folder and path.stem == file_id and path.suffix in {".wav", ".flac"},
        "Unexpected publisher path for the matched audio",
    )
    return path.as_posix()


def aligned_params(params, condition, source, snr):
    pool, file_id = source["pool"], source["file_id"]
    require(
        params["pool"] == pool
        and params["file_id"] == file_id
        and params["condition"] == condition
        and params["sr_in"] == params["sr_out"] == source["source_sample_rate"]
        and int(params["n_in"]) == int(params["n_out"]) >= 3 * int(params["sr_in"])
        and float(params["target_snr_db"]) == float(params["achieved_snr_db"]) == snr
        and int(params["written_clipped_samples"]) == 0,
        "Published sample clock, treatment or level differs",
    )
    return match_member(f"pacc-p/{params['output']}", ("pacc-p", condition, pool), file_id)


def private_plan():
    known_voices = set(read(HISTORY)["voice_keys"])
    with tarfile.open(SAMPLE, "r:") as archive:
        entries = archive.getmembers()
        require(
            len(entries) == len({entry.name for entry in entries}) == 838
            and all(entry.isfile() for entry in entries),
            "Exact regular-member publisher sample layout required",
        )
        sources = rows(archive, "sources/metadata.csv")
        keys = {(row["pool"], row["file_id"]) for row in sources}
        require(
            len(keys) == len(sources) == 14 and Counter(row["pool"] for row in sources) == POOLS,
            "Exact publisher source roster required",
        )
        params = {}
        credit_digest = sha(SOURCE / "attribution.csv")
        for condition, snr in CONDITIONS.items():
            if snr is None:
                continue
            treatment = rows(archive, f"pacc-p/{condition}/params.csv")
            params[condition] = {(row["pool"], row["file_id"]): row for row in treatment}
            credits = f"pacc-p/{condition}/attribution.csv"
            require(
                len(treatment) == len(params[condition]) == len(keys)
                and params[condition].keys() == keys
                and digest(member(archive, credits)) == credit_digest,
                "Exact matched roster and publisher credit table required",
            )
        credit_rows = rows(archive, "pacc-p/babble_snr00/attribution.csv")
        require(
            len(credit_rows) == 4696 and {row["licence"] for row in credit_rows} == SOURCE_LICENSES,
            "Publisher credit inventory differs",
        )
        plan = []
        for index, source in enumerate(sorted(sources, key=lambda row: (row["pool"], row["file_id"]))):
            pool, file_id = source["pool"], source["file_id"]
            family = "AMI" if pool == "AMI" else "VCTK"
            context = source["ami_meeting"] if family == "AMI" else source["vctk_utterance"]
            require(bool(context and source["speaker_id"]), "Source dependence metadata missing")
            path = match_member(source["file"], ("sources", pool), file_id)
            plan.append(
                {
                    "slot": f"source-{index:02d}",
                    "pool": pool,
                    "family": family,
                    "source_context": context,
                    "speaker": source["speaker_id"],
                    "known_historical_voice": f"vctk:{source['speaker_id'].casefold()}" in known_voices,
                    "source_rate": int(source["source_sample_rate"]),
                    "members": {
                        "clean": path,
                        **{
                            name: aligned_params(params[name][pool, file_id], name, source, snr)
                            for name, snr in CONDITIONS.items()
                            if snr is not None
                        },
                    },
                }
            )
    return plan


def identity():
    require(
        SAMPLE.is_file()
        and not SAMPLE.is_symlink()
        and SAMPLE.stat().st_size == SAMPLE_SIZE
        and sha(SAMPLE) == SAMPLE_SHA256,
        "Pinned PACC-P sample bytes required",
    )
    listed = (SOURCE / "SHA256SUMS").read_text().splitlines()
    require(f"{SAMPLE_SHA256}  {SAMPLE.name}" in listed, "Publisher sample checksum record differs")
    ffmpeg = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(ffmpeg.values()), "Production decoder tools required")
    files = (
        Path(__file__),
        METHOD,
        ROOT / "backend/tests/test_audio_pacc_sample_v176.py",
        ROOT / "backend/app/detectors/audio/decode.py",
        Path(runtime.__file__),
    )
    return {
        "schema": "chipmunk.audio-pacc-sample-freeze/176",
        "publisher_revision": PUBLISHER_REVISION,
        "sample_sha256": SAMPLE_SHA256,
        "publisher_metadata_sha256": {
            name: sha(SOURCE / name)
            for name in (
                "README.md",
                "SHA256SUMS",
                "TARBALL_INDEX.csv",
                "attribution.csv",
                "condition_index.csv",
            )
        },
        "declared_history_sha256": sha(HISTORY),
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in files},
        "ffmpeg_tool_sha256": {name: sha(Path(path)) for name, path in ffmpeg.items()},
        "planned_source_inputs": POOLS,
        "conditions": CONDITIONS,
        "planned_cases": sum(POOLS.values()) * len(CONDITIONS),
        "source_sample_range": [0, SAMPLES],
        "production_sample_rate": SAMPLE_RATE,
        "new_neural_calls": 0,
    }


def freeze():
    require(not TRIAL.exists() and not PUBLIC.exists() and not MEDIA.exists(), "Preserve every prior attempt")
    bound, plan = identity(), private_plan()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "private-plan.json", plan)
    bound["private_plan_sha256"] = sha(TRIAL / "private-plan.json")
    write(TRIAL / "freeze.json", bound)
    write(PUBLIC, bound)


def verify():
    frozen = read(PUBLIC)
    expected = {**identity(), "private_plan_sha256": sha(TRIAL / "private-plan.json")}
    require(frozen == read(TRIAL / "freeze.json") == expected, "Exact publisher decode freeze required")
    return read(TRIAL / "private-plan.json")


def prepare(path, pcm_path, raw, rate):
    path.write_bytes(raw)
    decoded = decode_audio(path, source_sha256=digest(raw), max_seconds=3.0)
    expected_range = {
        "start_sample": 0,
        "end_sample": SAMPLES,
        "source_start_pts": 0,
        "source_end_pts": SAMPLES,
        "time_base": f"1/{SAMPLE_RATE}",
    }
    require(
        len(decoded.pcm) == SAMPLES
        and decoded.mapping["source_sample_rate"] == rate
        and source_ranges(decoded.mapping, 0, SAMPLES) == [expected_range],
        "Exactly three continuous seconds on the zero-origin production sample clock required",
    )
    require(
        digest(read_wav(decoded.wav).astype("<f4").tobytes()) == decoded.mapping["pcm_sha256"],
        "WAV does not contain the production-decoded samples",
    )
    with pcm_path.open("xb") as stream:
        stream.write(decoded.wav)
    return {
        "status": "qualified",
        "source_sha256": digest(raw),
        "wav_sha256": sha(pcm_path),
        "pcm_sha256": decoded.mapping["pcm_sha256"],
        "source_range": expected_range,
    }


def aggregate(plan, observed):
    require(
        [(row["slot"], row["condition"]) for row in observed]
        == [(source["slot"], condition) for source in plan for condition in CONDITIONS],
        "Every planned input must have one ordered receipt",
    )
    outcomes = {
        family: {
            name: dict(
                Counter(
                    row["status"] for row in observed if row["family"] == family and row["condition"] == name
                )
            )
            for name in CONDITIONS
        }
        for family in ("AMI", "VCTK")
    }
    return {
        "source_inputs": len(plan),
        "declared_source_contexts": dict(
            Counter(family for family, _ in {(row["family"], row["source_context"]) for row in plan})
        ),
        "inputs_with_historical_voice_match": sum(row["known_historical_voice"] for row in plan),
        "outcomes_by_source_family_and_condition": outcomes,
    }


def worker(output):
    require(
        output == OUTPUT and (STAGE / "resources.jsonl").is_file() and (STAGE / "worker.log").is_file(),
        "Exact guarded decoder worker required",
    )
    plan = verify()
    output.mkdir(mode=0o700)
    MEDIA.mkdir(mode=0o700)
    observed = []
    with tarfile.open(SAMPLE, "r:") as archive, TemporaryDirectory() as scratch:
        for source in plan:
            for condition in CONDITIONS:
                name = source["members"][condition]
                incoming = Path(scratch) / f"input{Path(name).suffix}"
                outgoing = MEDIA / f"{source['slot']}-{condition}.wav"
                row = {"slot": source["slot"], "family": source["family"], "condition": condition}
                try:
                    row.update(prepare(incoming, outgoing, member(archive, name), source["source_rate"]))
                except (OSError, ValueError) as error:
                    row.update(status="unavailable", private_error=str(error))
                observed.append(row)
    journal = output / "private-observations.json"
    write(journal, observed)
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-pacc-sample-result/176",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_observations_sha256": sha(journal),
            "new_neural_calls": 0,
            **aggregate(plan, observed),
        },
    )


def run():
    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
            "Exact monitored decoder command required",
        )
        return subprocess.Popen([*argv[:3], MODULE, "worker", str(OUTPUT)], **kwargs)

    proxy = types.SimpleNamespace(
        Popen=launch, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda *_: not MEDIA.exists(),
            "subprocess": proxy,
            "__file__": __file__,
        },
    )("bind", 0)


def readout():
    plan = verify()
    receipt_path = OUTPUT / "result.json"
    receipt = read(receipt_path)
    profile = read(STAGE / "profile.json")
    journal = OUTPUT / "private-observations.json"
    rows = read(journal)
    require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(receipt_path)
        and profile["runtime_sha256"] == sha(Path(__file__))
        and profile["resource_journal_sha256"] == sha(STAGE / "resources.jsonl")
        and receipt["freeze_sha256"] == sha(PUBLIC)
        and receipt["private_observations_sha256"] == sha(journal)
        and receipt["new_neural_calls"] == profile["new_neural_calls"] == 0
        and all(receipt[key] == value for key, value in aggregate(plan, rows).items()),
        "Exact completed guarded decoder receipt required",
    )
    for row in rows:
        if row["status"] == "qualified":
            wav = MEDIA / f"{row['slot']}-{row['condition']}.wav"
            require(
                wav.is_file() and not wav.is_symlink() and sha(wav) == row["wav_sha256"],
                "Prepared audio bytes differ",
            )
    return {
        **receipt,
        "resource_profile_sha256": sha(STAGE / "profile.json"),
        "maximum_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
        "worker_duration_seconds": profile["duration_s"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("output", type=Path, nargs="?")
    args = parser.parse_args()
    require((args.command == "worker") == (args.output is not None), "Only the worker takes an output")
    if args.command == "worker":
        worker(args.output)
    elif args.command == "readout":
        print(json.dumps(readout(), indent=2, sort_keys=True, allow_nan=False))
    else:
        (freeze if args.command == "freeze" else run)()
