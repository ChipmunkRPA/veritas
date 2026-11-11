"""Qualify the two frozen short-edit source roles without detector inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import types
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from app.detectors.audio.decode import decode_audio, source_ranges, wav_bytes
from scripts import audio_short_edit_sources_v169 as roster
from scripts import audio_short_edit_sources_v169b as selected
from scripts.audio_expressive_inventory_v66 import intended_text
from scripts.video_appearance_corrected_v112 import scoped

TRIAL = roster.BASE / "audio-short-edit-materialization-v169c"
MEDIA = roster.ARCHIVE / "audio-short-edit-materialization-v169c"
PUBLIC = roster.ROOT / "docs/evaluation-panels/audio-short-edit-materialization-freeze-v169c.json"
METHOD = roster.ROOT / "docs/audio-short-edit-materialization-v169c.md"
MODULE = "scripts.audio_short_edit_materialize_v169c"
OUTPUT = TRIAL / "bind-00/output"
SOURCE_SAMPLES = 48_000
GRID_PREFIX = roster.SOURCE_SAMPLES
RATE = 16_000
MAX_PAYLOAD = 16_000_000
require, sha, write = roster.require, roster.sha, roster.write


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(value).hexdigest()


def pcm_digest(pcm):
    return digest(np.asarray(pcm, dtype="<f4").tobytes())


def source_plan():
    source = selected.OUTPUT / "private-roster.json"
    public = read(selected.PUBLIC)
    body = read(source)
    require(
        sha(source) == public["private_roster_sha256"]
        and body["schema"] == "chipmunk.audio-short-edit-private-source-plan/169b"
        and not body["exact_production_decode_complete"]
        and not body["holdout_scored"]
        and public["holdout_scored"] is False
        and public["model_calls"] == 0,
        "The exact unscored v169b roster is required",
    )
    roster.validate_roles(body["groups"], body["noise_parents"])
    return body, public


def identity():
    _, public = source_plan()
    paths = (
        Path(__file__),
        METHOD,
        roster.ROOT / "backend/tests/test_audio_short_edit_materialize_v169c.py",
        roster.ROOT / "backend/app/detectors/audio/decode.py",
        roster.ROOT / "backend/app/detectors/video/decode.py",
        roster.ROOT / "backend/scripts/video_appearance_runtime_v108.py",
        roster.ROOT / "backend/scripts/audio_expressive_inventory_v66.py",
    )
    tools = {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
    require(all(tools.values()), "The existing FFmpeg decoder tools are required")
    return {
        "schema": "chipmunk.audio-short-edit-materialization-freeze/169c",
        "source_public_plan_sha256": sha(selected.PUBLIC),
        "private_roster_sha256": public["private_roster_sha256"],
        "code_sha256": {str(path.relative_to(roster.ROOT)): sha(path) for path in paths},
        "ffmpeg_tool_sha256": {name: sha(Path(path)) for name, path in tools.items()},
        "source_samples": SOURCE_SAMPLES,
        "sample_rate": RATE,
        "grid_prefix_samples_for_exclusion": GRID_PREFIX,
        "planned_roles": public["roles"],
        "model_calls": 0,
        "holdout_scored": False,
        "scope": "One bounded production decode and WAV materialization. No analysis, scoring, model, or fitting calls.",
    }


def freeze():
    require(
        not TRIAL.exists() and not MEDIA.exists() and not PUBLIC.exists(), "Preserve every source attempt"
    )
    frozen = identity()
    TRIAL.mkdir(mode=0o700)
    write(TRIAL / "freeze.json", frozen)
    write(PUBLIC, frozen)


def verify():
    frozen = read(PUBLIC)
    require(frozen == read(TRIAL / "freeze.json") == identity(), "Exact source and decode freeze required")
    return frozen


class SourceReader:
    def __init__(self, stack, metadata):
        self.stack = stack
        self.checked = {}
        self.archives = {}
        self.parquets = {}
        self.table = None
        self.table_key = None
        self.clone = self.catalogue(
            "audio-source-inventory-v68/clone-candidates.json", metadata["clone_candidates"]
        )
        self.expressive = self.catalogue(
            "audio-expressive-inventory-v66/candidates.json", metadata["expressive_candidates"]
        )
        receipt = roster.BASE / "audio-v8/reserved-sources/archives/receipt.json"
        require(sha(receipt) == metadata["stock_expressive_receipt"], "Expressive archive index differs")
        self.expressive_hashes = {row["name"]: row["sha256"] for row in read(receipt)["archives"]}

    @staticmethod
    def catalogue(relative, expected):
        path = roster.BASE / relative
        require(sha(path) == expected, "Frozen private source catalogue differs")
        return {row["id"]: row for row in read(path)}

    def asset(self, file, expected):
        path = Path(file).resolve(strict=True)
        require(
            path.is_file() and (path.is_relative_to(roster.BASE) or path.is_relative_to(roster.ARCHIVE)),
            "Source asset must stay in the admitted source banks",
        )
        if path not in self.checked:
            self.checked[path] = sha(path)
        require(self.checked[path] == expected, "Publisher whole-archive SHA-256 differs")
        return path

    def member(self, file, expected, name):
        path = self.asset(file, expected)
        member_name = PurePosixPath(name)
        require(not member_name.is_absolute() and ".." not in member_name.parts, "Unsafe archive member")
        if path not in self.archives:
            self.archives[path] = self.stack.enter_context(tarfile.open(path, "r:"))
        archive = self.archives[path]
        try:
            member = archive.getmember(name)
        except KeyError as error:
            raise ValueError("Frozen archive member unavailable") from error
        require(member.isfile() and 0 < member.size <= MAX_PAYLOAD, "Bounded archive payload required")
        with archive.extractfile(member) as stream:
            raw = stream.read(MAX_PAYLOAD + 1)
        require(len(raw) == member.size, "Archive payload is incomplete")
        return raw

    def from_parquet(self, row):
        loc = row["locator"]
        path = self.asset(loc["file"], loc["archive_sha256"])
        if path not in self.parquets:
            reader = pq.ParquetFile(path)
            self.stack.callback(reader.close)
            self.parquets[path] = reader
        key = (path, loc["row_group"])
        if self.table_key != key:
            self.table = None
            names = (
                ["id", "speaker_id", "text", "audio"]
                if row["lane"] == "genuine"
                else [
                    "rendition_id",
                    "tts_voice",
                    "transcript",
                    "final_text",
                    "audio",
                    "tts_engine",
                    "license",
                ]
            )
            self.table = self.parquets[path].read_row_group(key[1], columns=names, use_threads=False)
            self.table_key = key
        [value] = self.table.slice(loc["row_offset"], 1).to_pylist()
        if row["lane"] == "genuine":
            require(
                f"libri:{value['id']}" == row["id"]
                and f"librispeech:{value['speaker_id']}" == row["voice"]
                and roster.text_keys(value["text"]) == set(row["texts"]),
                "LibriSpeech source row, reader or text differs",
            )
        else:
            require(
                value["rendition_id"] == row["id"]
                and f"kokoro-82m:{value['tts_voice']}" == row["voice"]
                and value["tts_engine"] == "kokoro-82m"
                and value["license"] == "cc-by-sa-4.0"
                and roster.text_keys(value["transcript"]) | roster.text_keys(value["final_text"])
                == set(row["texts"]),
                "Stock source row, voice, terms or text differs",
            )
        return value["audio"]["bytes"]

    def from_tar(self, row):
        loc = row["locator"]
        catalogue = self.clone if row["lane"] == "clone" else self.expressive
        declared = catalogue[row["id"]]
        require(
            declared["member"] == loc["member"]
            and declared["archive"] == Path(loc["file"]).name
            and declared["canonical_speaker"] == row["voice"]
            and roster.text_keys(declared["transcript"]) == set(row["texts"]),
            "Frozen archived member, voice or text differs",
        )
        if row["lane"] == "clone":
            metadata_file, metadata_hash = loc["file"], loc["archive_sha256"]
        else:
            name = declared["metadata_archive"]
            metadata_file = Path(loc["file"]).parent / name
            metadata_hash = self.expressive_hashes[name]
        metadata = self.member(metadata_file, metadata_hash, declared["metadata_member"])
        require(digest(metadata) == declared["metadata_sha256"], "Exact publisher sidecar bytes required")
        body = json.loads(metadata)
        if row["lane"] == "clone":
            utterance = body["speech_meta"][int(PurePosixPath(loc["member"]).stem)]
            voice = f"librispeech:{body['user_prompt_speaker_id']}"
            require(
                voice == row["voice"] == row["donor"]
                and declared["source_group"] == row["context"]
                and utterance["speaker"] == body["speakers"].index("user")
                and roster.text_keys(utterance["tts_text"]) == set(row["texts"]),
                "Exact publisher clone donor, dialogue and utterance required",
            )
        else:
            generation = body["generation"]
            voice = "elevenlabs:" + digest(generation["voice_id"].encode())
            require(
                body["manifest"]["id"] == row["id"]
                and body["manifest"]["language"] == "en"
                and generation["model_id"] == "eleven_v4"
                and voice == row["voice"]
                and declared["challenge_group"] == row["context"]
                and generation["audio_sha256"] == row["raw_sha256"]
                and roster.text_keys(intended_text(generation["request"]["text"])) == set(row["texts"]),
                "Exact publisher expressive generation sidecar required",
            )
        return self.member(loc["file"], loc["archive_sha256"], loc["member"])

    def original(self, row):
        if row["locator"]["format"] == "parquet":
            payload = self.from_parquet(row)
        else:
            require(row["locator"]["format"] == "tar", "Frozen source locator must be Parquet or tar")
            payload = self.from_tar(row)
        require(
            type(payload) is bytes and 0 < len(payload) <= MAX_PAYLOAD,
            "Bounded complete audio bytes required",
        )
        if expected := row.get("raw_sha256"):
            require(digest(payload) == expected, "Selected raw payload SHA-256 differs")
        return payload


def qualification(audio, raw_sha, excluded):
    pcm = audio.pcm
    mapping = audio.mapping
    require(
        mapping["source_sha256"] == raw_sha
        and mapping["samples"] == len(pcm)
        and mapping["sample_rate"] == RATE
        and mapping["channels"] == 1
        and mapping["sample_format"] == "s16le"
        and mapping["pcm_sha256"] == pcm_digest(pcm)
        and pcm.ndim == 1
        and pcm.dtype == np.float32
        and np.isfinite(pcm).all(),
        "Decoded production audio mapping or sample commitment differs",
    )
    if len(pcm) < SOURCE_SAMPLES:
        return None, "shorter_than_exact_three_second_prefix"
    prefix = pcm[:SOURCE_SAMPLES]
    ranges = source_ranges(mapping, 0, SOURCE_SAMPLES)
    if len(ranges) != 1:
        return None, "discontinuous_original_prefix_clock"
    if float(np.sqrt(np.mean(prefix.astype(np.float64) ** 2))) < 1e-4:
        return None, "silent_three_second_prefix"
    fingerprints = {
        "original_sha256": raw_sha,
        "full_pcm_sha256": mapping["pcm_sha256"],
        "prefix_pcm_sha256": pcm_digest(prefix),
        "grid_length_prefix_pcm_sha256": pcm_digest(prefix[:GRID_PREFIX]),
    }
    keys = {
        f"bytes:{raw_sha}",
        *(f"pcm:{value}" for name, value in fingerprints.items() if name != "original_sha256"),
    }
    if keys & excluded:
        return None, "recorded_original_or_pcm_identity_overlap"
    return (prefix, ranges[0], fingerprints, keys), None


def materialize(raw, temp, wav, used):
    sha_raw = digest(raw)
    try:
        with temp.open("xb") as stream:
            stream.write(raw)
        audio = decode_audio(temp, source_sha256=sha_raw)
        ready, reason = qualification(audio, sha_raw, used)
    finally:
        temp.unlink(missing_ok=True)
    if ready is None:
        return None, reason
    prefix, source_range, fingerprints, keys = ready
    payload = wav_bytes(prefix)
    wav_sha = digest(payload)
    if f"bytes:{wav_sha}" in used:
        return None, "recorded_prefix_container_overlap"
    with wav.open("xb") as stream:
        stream.write(payload)
    replay = decode_audio(wav, source_sha256=wav_sha, max_seconds=3)
    expected = {
        "start_sample": 0,
        "end_sample": SOURCE_SAMPLES,
        "source_start_pts": 0,
        "source_end_pts": SOURCE_SAMPLES,
        "time_base": "1/16000",
    }
    require(
        replay.mapping["pcm_sha256"] == fingerprints["prefix_pcm_sha256"]
        and np.array_equal(replay.pcm, prefix)
        and source_ranges(replay.mapping, 0, SOURCE_SAMPLES) == [expected],
        "Materialized WAV changed its exact sample bytes or clock",
    )
    used.update(keys | {f"bytes:{wav_sha}"})
    return {
        **fingerprints,
        "wav_sha256": wav_sha,
        "file": str(wav),
        "source_range": source_range,
        "wav_range": expected,
    }, None


def worker(output):
    frozen = verify()
    require(
        output == OUTPUT
        and (TRIAL / "bind-00/worker.log").is_file()
        and (TRIAL / "bind-00/resources.jsonl").is_file()
        and not MEDIA.exists(),
        "Only the exclusive monitored preparation output is permitted",
    )
    plan, _ = source_plan()
    used = set(plan["excluded_keys"])
    output.mkdir()
    MEDIA.mkdir(mode=0o700)
    for role in roster.ROLES:
        (MEDIA / role).mkdir(mode=0o700)
    manifests = {role: {"foregrounds": [], "noise": []} for role in roster.ROLES}
    counts = {role: Counter() for role in roster.ROLES}
    with ExitStack() as stack:
        journal = stack.enter_context((output / "attempts.jsonl").open("x"))
        temp = Path(stack.enter_context(TemporaryDirectory(prefix=".original-", dir=output))) / "source.audio"
        reader = SourceReader(stack, plan["input_metadata_sha256"])

        def record(row):
            journal.write(json.dumps(row, allow_nan=False) + "\n")
            journal.flush()

        def prepare(role, kind, slot, row, read_payload):
            wav = MEDIA / role / f"{slot}.wav"
            try:
                result, reason = materialize(read_payload(row), temp, wav, used)
                status = "qualified" if result else "ineligible"
            except (OSError, ValueError, RuntimeError, IndexError) as error:
                # A failed WAV is retained at a distinct path before an ordered backup.
                if wav.is_file():
                    wav.rename(wav.with_name(f"{slot}-{counts[role]['attempted']:03d}-failed.wav"))
                result, reason, status = None, str(error), "failed"
            counts[role]["attempted"] += 1
            counts[role][f"{kind}_{status}"] += 1
            record(
                {
                    "role": role,
                    "kind": kind,
                    "slot": slot,
                    "candidate": row["id"],
                    "status": status,
                    "reason": reason,
                }
            )
            return result

        def noise_payload(row):
            path = Path(row["path"])
            require(
                sha(path) == row["raw_sha256"] and 0 < path.stat().st_size <= MAX_PAYLOAD,
                "Frozen original noise bytes differ",
            )
            return path.read_bytes()

        for index, row in enumerate(plan["noise_parents"]):
            role, slot = row["role"], f"noise-{index:02d}"
            accepted = prepare(role, "noise", slot, row, noise_payload)
            manifests[role]["noise"].append(
                {"slot": slot, "source": row["id"], "status": "qualified", **accepted}
                if accepted
                else {"slot": slot, "status": "unavailable"}
            )
        for group in plan["groups"]:
            role, lane = group["role"], group["lane"]
            picked = 0
            for row in group["ordered_candidates"]:
                slot = f"{group['group_id']}-{picked:02d}"
                accepted = prepare(role, lane, slot, row, reader.original)
                if accepted:
                    manifests[role]["foregrounds"].append(
                        {
                            "slot": slot,
                            "group_id": group["group_id"],
                            "lane": lane,
                            "status": "qualified",
                            "candidate": row["id"],
                            "voice": row["voice"],
                            "license": row["license"],
                            **accepted,
                        }
                    )
                    picked += 1
                    if picked == group["required"]:
                        break
            for index in range(picked, group["required"]):
                manifests[role]["foregrounds"].append(
                    {
                        "slot": f"{group['group_id']}-{index:02d}",
                        "group_id": group["group_id"],
                        "lane": lane,
                        "status": "unavailable",
                        "reason": "ordered_component_candidates_exhausted",
                    }
                )
    role_receipts = {}
    for role, manifest in manifests.items():
        target = MEDIA / role / "private-manifest.json"
        write(target, {"role": role, **manifest, "scored": False})
        role_receipts[role] = {
            "private_manifest_sha256": sha(target),
            "foregrounds": {
                lane: dict(Counter(row["status"] for row in manifest["foregrounds"] if row["lane"] == lane))
                for lane in roster.LANES
            },
            "noise": dict(Counter(row["status"] for row in manifest["noise"])),
            "attempt_counts": counts[role],
        }
    write(
        output / "result.json",
        {
            "schema": "chipmunk.audio-short-edit-materialization-result/169c",
            "status": "complete",
            "freeze_sha256": sha(PUBLIC),
            "private_roster_sha256": frozen["private_roster_sha256"],
            "private_attempt_journal_sha256": sha(output / "attempts.jsonl"),
            "roles": role_receipts,
            "source_samples": SOURCE_SAMPLES,
            "sample_rate": RATE,
            "new_neural_calls": 0,
            "holdout_scored": False,
        },
    )


def run():
    from scripts import video_appearance_runtime_v108 as runtime

    def launch(argv, **kwargs):
        require(
            argv[1:] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind", str(OUTPUT), "0"],
            "Only the source materialization child may run",
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
    verify()
    stage = TRIAL / "bind-00"
    profile = read(stage / "profile.json")
    result = read(OUTPUT / "result.json")
    require(
        profile["child_exit_code"] == 0
        and profile["guard"] is None
        and profile["sample_errors"] == 0
        and profile["worker_receipt_sha256"] == sha(OUTPUT / "result.json")
        and profile["resource_journal_sha256"] == sha(stage / "resources.jsonl")
        and result["status"] == "complete"
        and result["freeze_sha256"] == sha(PUBLIC)
        and result["private_attempt_journal_sha256"] == sha(OUTPUT / "attempts.jsonl")
        and profile["new_neural_calls"] == result["new_neural_calls"] == 0
        and result["holdout_scored"] is False,
        "A complete exact guarded preparation receipt is required",
    )
    for role in roster.ROLES:
        path = MEDIA / role / "private-manifest.json"
        body = read(path)
        require(
            sha(path) == result["roles"][role]["private_manifest_sha256"]
            and body["role"] == role
            and body["scored"] is False,
            "Exact private role manifest required",
        )
        for row in [*body["foregrounds"], *body["noise"]]:
            if row["status"] == "qualified":
                wav = Path(row["file"])
                require(
                    wav.parent == path.parent and sha(wav) == row["wav_sha256"], "Exact prepared WAV required"
                )
    print(
        json.dumps(
            {
                **result,
                "resource_profile_sha256": sha(stage / "profile.json"),
                "max_sampled_descendant_rss_bytes": profile["sampled_tree_rss_lower_bound_bytes"],
                "duration_s": profile["duration_s"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "worker", "readout"))
    parser.add_argument("output", nargs="?", type=Path)
    args = parser.parse_args()
    require(
        (args.command == "worker") == (args.output is not None), "Only the guarded worker takes an output"
    )
    if args.command == "worker":
        worker(args.output)
    else:
        {"freeze": freeze, "run": run, "readout": readout}[args.command]()
