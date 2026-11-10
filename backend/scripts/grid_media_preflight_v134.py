"""Decoder-only preflight of two fixed clips in the first fully verified archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import types
import zipfile
from collections import Counter
from fractions import Fraction
from pathlib import Path, PurePosixPath

from scripts import grid_transfer_context_v131 as acquisition
from scripts import video_native_sources_v92 as clocks
from scripts.video_appearance_corrected_v112 import scoped

ROOT, BASE, PANEL = acquisition.original.ROOT, acquisition.original.BASE, acquisition.original.PANEL
TRIAL = BASE / "grid-media-preflight-v134"
CACHE = acquisition.original.runtime.ARCHIVE / "grid-media-preflight-v134"
PUBLIC = PANEL / "grid-media-preflight-freeze-v134.json"
CODE = (
    "backend/scripts/grid_media_preflight_v134.py",
    "backend/tests/test_grid_media_preflight_v134.py",
    "docs/grid-media-preflight-v134-method.md",
    "backend/scripts/video_native_sources_v92.py",
    "backend/app/detectors/video/decode.py",
    "backend/app/detectors/audio/decode.py",
)
read, write, sha, require = (
    acquisition.original.read,
    acquisition.original.write,
    acquisition.original.sha,
    acquisition.original.require,
)


def select_members(members, used_stems=()):
    used = set(used_stems)
    candidates = []
    for row in members:
        name = PurePosixPath(row["name"])
        require(
            not name.is_absolute() and ".." not in name.parts and "\\" not in row["name"],
            "Safe archive member required",
        )
        if "__MACOSX" in name.parts or name.name.startswith("._") or name.suffix.lower() != ".mpg":
            continue
        candidates.append(row)
    ordered = sorted(
        candidates,
        key=lambda r: (
            hashlib.sha256(("chipmunk.grid-clips/134:" + PurePosixPath(r["name"]).stem).encode()).hexdigest(),
            r["name"],
        ),
    )
    selected = []
    for row in ordered:
        stem = PurePosixPath(row["name"]).stem
        if stem not in used:
            selected.append(row)
            used.add(stem)
        if len(selected) == 2:
            break
    require(len(selected) == 2, "Two fixed distinct unused utterance stems required")
    return selected


def plan():
    acquisition.context()["verify"]()
    stage = acquisition.TRIAL / "bind-00"
    profile = read(stage / "profile.json")
    require(
        profile["child_exit_code"] == -15
        and profile["guard"] == "resource_limit"
        and profile["sample_errors"] == 0
        and all(r["violations"] == ["elapsed_time_limit"] for r in profile["resource_limit_observations"]),
        "Preserved elapsed-only stopped acquisition required",
    )
    journal = [json.loads(line) for line in (stage / "output/archives.jsonl").read_text().splitlines()]
    require(
        len(journal) == 1
        and journal[0]["ordinal"] == 0
        and journal[0]["status"] == "publisher_archive_verified",
        "One complete first archive required",
    )
    record = journal[0]
    expected = read(acquisition.TRIAL / "plan.json")["selected"][0]
    require(
        all(record[k] == v for k, v in expected.items())
        and Path(record["file"]).resolve() == (acquisition.CACHE / "batch-00/source-00.zip").resolve(),
        "Unchanged first selected archive required",
    )
    return {
        "archive": record,
        "clips": select_members(record["real_video_members"]),
        "purpose": "First-source decoder/timeline availability probe; not detection or full cohort qualification.",
    }


def identity():
    require(read(TRIAL / "plan.json") == plan(), "Unchanged source/utterance plan required")
    return {
        "schema": "chipmunk.grid-media-preflight-freeze/134",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "private_plan_sha256": sha(TRIAL / "plan.json"),
        "acquisition_freeze_sha256": sha(acquisition.PUBLIC),
        "stopped_acquisition_profile_sha256": sha(acquisition.TRIAL / "bind-00/profile.json"),
        "private_archive_journal_sha256": sha(acquisition.TRIAL / "bind-00/output/archives.jsonl"),
        "binaries_sha256": {name: sha(Path(shutil.which(name))) for name in ("ffmpeg", "ffprobe")},
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless")},
        "selected_archives": 1,
        "selected_clips": 2,
        "source_license": "CC-BY-4.0",
        "scope": "Independently recheck complete archive bytes/checksums and ZIP directory, then two fixed native video/PCM clocks. Partial second archive and all model outputs excluded. Private media/clock/individual fingerprints remain private.",
    }


def freeze():
    body = plan()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "plan.json", body)
    locked = identity()
    write(TRIAL / "freeze.json", locked)
    write(PUBLIC, locked)
    print('{"status":"frozen","selected_clips":2,"new_neural_calls":0}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(), "Unchanged decoder-only preflight required"
    )


def archive_verified(record):
    path = Path(record["file"])
    md5, digest, total = hashlib.md5(), hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            md5.update(chunk)
            digest.update(chunk)
    require(
        total == record["bytes"]
        and md5.hexdigest() == record["md5"]
        and digest.hexdigest() == record["sha256"],
        "Complete independent archive bytes/checksums required",
    )
    require(
        acquisition.original.inspect_archive(path) == record["real_video_members"],
        "Complete original ZIP inventory differs",
    )


def video_supported(table):
    return (
        0 < len(table) <= 500
        and not table.truncated
        and not table.interpolated
        and 0 < table.width <= 640
        and 0 < table.height <= 640
        and all(a < b for a, b in zip(table.pts, table.pts[1:]))
        and Fraction(table.pts[-1] - table.pts[0]) * table.tb <= 10
    )


def probe(output):
    import cv2
    import numpy as np

    from app.detectors.audio.decode import decode_audio, source_ranges
    from app.detectors.video.decode import decode_frames, probe_frames

    chosen = read(TRIAL / "plan.json")
    archive_verified(chosen["archive"])
    CACHE.mkdir(exist_ok=False)
    records = []
    with zipfile.ZipFile(chosen["archive"]["file"]) as archive:
        for ordinal, member in enumerate(chosen["clips"]):
            folder = CACHE / f"clip-{ordinal:02d}"
            folder.mkdir(exist_ok=False)
            info = archive.getinfo(member["name"])
            require(
                info.file_size == member["bytes"]
                and member["crc32"] == info.CRC
                and 1000 <= info.file_size <= 2 * 1024**2,
                "Exact bounded clip member required",
            )
            data = archive.read(info)  # ZIP verifies the uncompressed CRC too.
            require(len(data) == info.file_size, "Complete clip member bytes required")
            path = folder / "source.mpg"
            with path.open("xb") as stream:
                stream.write(data)
            record = {
                "ordinal": ordinal,
                "member": member,
                "source_file": str(path),
                "source_sha256": sha(path),
                "source_license": "CC-BY-4.0",
            }
            table = probe_frames(path, cap=500, timeout_s=60)
            record["native_video_table"] = table.to_json()
            if not video_supported(table):
                record["status"] = "unavailable_video_clock_or_bounds"
                records.append(record)
                continue
            indices = sorted({0, len(table) // 2, len(table) - 1})
            pictures = []
            for frame in decode_frames(path, table, indices, timeout_s=60):
                file = folder / f"frame-{frame.index:04d}.png"
                require(
                    cv2.imwrite(str(file), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)), "Retain source PNG"
                )
                pictures.append(
                    {
                        "frame_index": frame.index,
                        "pts": frame.pts,
                        "file": str(file),
                        "file_sha256": sha(file),
                        "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
                    }
                )
            require(
                [p["frame_index"] for p in pictures] == indices,
                "Complete native picture/PTS support required",
            )
            record["pictures"] = pictures
            info = clocks.probe_json(
                path,
                [
                    "-select_streams",
                    "a:0",
                    "-show_streams",
                    "-show_frames",
                    "-show_entries",
                    "stream=sample_rate,channels,time_base:frame=pts,best_effort_timestamp,nb_samples",
                ],
            )
            streams = info.get("streams") or []
            if not streams:
                record["status"] = "video_only_no_embedded_audio"
            else:
                native = clocks.audio_clocks(streams[0], info.get("frames") or [])
                record["native_audio_clock"] = native
                if (
                    not native["contiguous"]
                    or Fraction(native["frames"][-1]["end"]) - Fraction(native["frames"][0]["start"]) > 10
                ):
                    record["status"] = "unavailable_audio_clock_or_bounds"
                else:
                    sound = decode_audio(path, source_sha256=record["source_sha256"], max_seconds=10)
                    ranges = source_ranges(sound.mapping, 0, len(sound.pcm))
                    require(
                        ranges and np.isfinite(sound.pcm).all(),
                        "Complete finite PCM/sample-clock mapping required",
                    )
                    wav = folder / "source-mono16k.wav"
                    with wav.open("xb") as stream:
                        stream.write(sound.wav)
                    record.update(
                        status="native_video_and_audio_available",
                        pcm_mapping=sound.mapping,
                        whole_pcm_source_ranges=ranges,
                        pcm_file=str(wav),
                        pcm_file_sha256=sha(wav),
                    )
            records.append(record)
    write(output / "private-records.json", records)
    result = {
        "schema": "chipmunk.grid-media-preflight-result/134",
        "status": "complete",
        "selected_archives": 1,
        "independently_verified_archive_bytes": chosen["archive"]["bytes"],
        "planned_clips": 2,
        "recorded_clips": len(records),
        "availability": dict(Counter(r["status"] for r in records)),
        "native_video_frames": sum(len(r["native_video_table"]["pts"]) for r in records),
        "retained_native_pictures": sum(len(r.get("pictures", [])) for r in records),
        "observed_mono16k_samples": sum(r.get("pcm_mapping", {}).get("samples", 0) for r in records),
        "private_records_sha256": sha(output / "private-records.json"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "First-source checksum/decode/native clock availability only; not the complete sixteen-source transfer cohort, person/ancestry independence, generated-edit quality or detection accuracy. Missing audio/clock support retained, no replacement or inference.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def routed_popen(argv, **kwargs):
    require(
        argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"],
        "Only fixed decoder-only stage permitted",
    )
    routed = list(argv)
    routed[3] = "scripts.grid_media_preflight_v134"
    return subprocess.Popen(routed, **kwargs)


def run():
    proxy = types.SimpleNamespace(
        Popen=routed_popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    scoped(
        acquisition.original.runtime.run,
        {
            "TRIAL": TRIAL,
            "verify": verify,
            "prerequisites": lambda c, b: True,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "bind"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command in {"freeze", "run"}:
        require(not args.arguments, "No extra arguments required")
        (freeze if args.command == "freeze" else run)()
    else:
        verify()
        require(len(args.arguments) == 2 and args.arguments[1] == "0", "Single fixed guarded probe required")
        output = Path(args.arguments[0])
        require(
            output.resolve() == (TRIAL / "bind-00/output").resolve(),
            "Exact exclusive guarded output required",
        )
        output.mkdir(exist_ok=False)
        probe(output)
