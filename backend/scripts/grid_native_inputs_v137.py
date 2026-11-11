"""Full native RGB/PCM banks from the fixed fresh-source plan; no model calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import types
from collections import Counter
from fractions import Fraction
from pathlib import Path

from scripts import grid_clip_plan_v136 as planning
from scripts.video_appearance_corrected_v112 import scoped

acquisition, original = planning.acquisition, planning.acquisition.preflight
ROOT, BASE, PANEL = planning.ROOT, planning.BASE, planning.PANEL
TRIAL = BASE / "grid-native-inputs-v137"
CACHE = acquisition.core.runtime.ARCHIVE / "grid-native-inputs-v137"
PUBLIC = PANEL / "grid-native-inputs-freeze-v137.json"
CODE = (
    "backend/scripts/grid_native_inputs_v137.py",
    "backend/tests/test_grid_native_inputs_v137.py",
    "docs/grid-native-inputs-v137-method.md",
    "backend/scripts/grid_media_preflight_v134.py",
    "backend/app/detectors/video/decode.py",
    "backend/app/detectors/audio/decode.py",
)
read, write, sha, require = planning.read, planning.write, planning.sha, planning.require


def archive_record(ordinal):
    if ordinal == 0:
        acquisition.preserved()
        record = read(original.TRIAL / "plan.json")["archive"]
    else:
        folder = acquisition.TRIAL / f"source-{ordinal:02d}/bind-00"
        if not all(p.is_file() for p in (folder / "output/result.json", folder / "profile.json")):
            return None
        result, profile = read(folder / "output/result.json"), read(folder / "profile.json")
        require(
            result["status"] == "complete"
            and profile["child_exit_code"] == 0
            and profile["guard"] is None
            and profile["sample_errors"] == 0
            and profile["worker_receipt_sha256"] == sha(folder / "output/result.json")
            and result["private_archive_journal_sha256"] == sha(folder / "output/archives.jsonl"),
            "Actual complete checksum-verified source acquisition required",
        )
        rows = [json.loads(line) for line in (folder / "output/archives.jsonl").read_text().splitlines()]
        require(len(rows) == 1, "One acquired source archive required")
        record = rows[0]
    expected = read(acquisition.TRIAL / "plan.json")["selected"][ordinal]
    require(
        all(record[k] == v for k, v in expected.items()) and record["status"] == "publisher_archive_verified",
        "Exact originally selected source archive required",
    )
    return record


def selected_members(record, clips):
    require(
        len(clips) == 2 and all(c["source_ordinal"] == record["ordinal"] for c in clips),
        "Both originally planned clips from this source required",
    )
    by_name = {m["name"]: m for m in record["real_video_members"]}
    require(
        all(c["member"] == by_name.get(c["member"]["name"]) for c in clips),
        "Remote planning directory and fully verified archive member identity must match",
    )
    return [c["member"] for c in clips]


def source_origins(table, mapping):
    require(
        len(table) > 0 and mapping["timeline"] and mapping["timeline_time_base"] == "1/16000",
        "Observed nonempty source clocks required; no zero-origin fallback",
    )
    return {
        "video_origin_seconds": str(Fraction(table.pts[0]) * table.tb),
        "audio_origin_seconds": str(Fraction(mapping["timeline"][0]["source_pts"], 16000)),
    }


def identity():
    planning.verify()
    require(planning.completed(2), "Actual complete original 32-clip selection required")
    return {
        "schema": "chipmunk.grid-native-inputs-freeze/137",
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "clip_planning_freeze_sha256": sha(planning.PUBLIC),
        "clip_planning_result_sha256": sha(planning.TRIAL / "bind-02/output/result.json"),
        "private_selected_clips_sha256": sha(planning.TRIAL / "bind-02/output/selected-clips.json"),
        "original_acquisition_continuation_freeze_sha256": sha(acquisition.PUBLIC),
        "source_archives": 16,
        "clips_per_source": 2,
        "planned_clips": 32,
        "stages": 16,
        "binaries_sha256": {n: original.identity()["binaries_sha256"][n] for n in ("ffmpeg", "ffprobe")},
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "opencv-python-headless", "Pillow", "psutil")
        },
        "scope": "Full selected native RGB and canonical PCM/source clocks only after actual full archive acquisition passes. Unavailable sources/decodes/clocks retained; no substitution, face eligibility, inference, fitting, generator quality or detection claim.",
    }


def freeze():
    body = identity()
    TRIAL.mkdir(exist_ok=False)
    write(TRIAL / "freeze.json", body)
    write(PUBLIC, body)
    print('{"status":"frozen","source_archives":16,"planned_clips":32}', flush=True)


def verify():
    require(
        read(TRIAL / "freeze.json") == read(PUBLIC) == identity(),
        "Unchanged native input-bank contract required",
    )


def bind(ordinal, output):
    import zipfile

    import cv2
    import numpy as np
    from PIL import Image

    from app.detectors.audio.decode import decode_audio, source_ranges
    from app.detectors.video.decode import DecodeError, decode_frames, probe_frames

    record = archive_record(ordinal)
    require(record is not None, "Complete archive acquisition required before decoding")
    original.archive_verified(record)
    all_clips = read(planning.TRIAL / "bind-02/output/selected-clips.json")
    clips = [c for c in all_clips if c["source_ordinal"] == ordinal]
    members = selected_members(record, clips)
    folder = CACHE / f"source-{ordinal:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    records = []
    with zipfile.ZipFile(record["file"]) as archive:
        for clip, member in zip(clips, members, strict=True):
            destination = folder / f"clip-{clip['within_source_ordinal']:02d}"
            destination.mkdir(exist_ok=False)
            info = archive.getinfo(member["name"])
            require(
                info.file_size == member["bytes"]
                and member["crc32"] == info.CRC
                and 1000 <= info.file_size <= 2 * 1024**2,
                "Exact bounded planned archive member required",
            )
            payload = archive.read(info)
            require(len(payload) == member["bytes"], "Full CRC-verified member required")
            source = destination / "source.mpg"
            with source.open("xb") as stream:
                stream.write(payload)
            row = {
                "planned": clip,
                "source_file": str(source),
                "source_sha256": sha(source),
                "study_role": "external_transfer",
                "source_license": "CC-BY-4.0",
            }
            records.append(row)
            try:
                table = probe_frames(source, cap=500, timeout_s=60)
            except DecodeError:
                row["status"] = "unavailable_video_probe"
                continue
            row["native_video_table"] = table.to_json()
            if not original.video_supported(table):
                row["status"] = "unavailable_video_clock_or_bounds"
                continue
            pictures = []
            row["pictures"] = pictures
            try:
                for frame in decode_frames(source, table, list(range(len(table))), timeout_s=120):
                    png = destination / f"frame-{frame.index:04d}.png"
                    require(
                        cv2.imwrite(str(png), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)),
                        "Retain exact native source PNG",
                    )
                    with Image.open(png) as image:
                        recovered = np.asarray(image.convert("RGB"))
                    require(recovered.tobytes() == frame.rgb.tobytes(), "Independent PNG round trip differs")
                    pictures.append(
                        {
                            "frame_index": frame.index,
                            "pts": frame.pts,
                            "file": str(png),
                            "file_sha256": sha(png),
                            "rgb_sha256": hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
                        }
                    )
            except DecodeError:
                row["status"] = "unavailable_video_pixels"
                continue
            require(
                [p["frame_index"] for p in pictures] == list(range(len(table))),
                "Complete exact native picture order required",
            )
            audio = original.clocks.probe_json(
                source,
                [
                    "-select_streams",
                    "a:0",
                    "-show_streams",
                    "-show_frames",
                    "-show_entries",
                    "stream=sample_rate,channels,time_base:frame=pts,best_effort_timestamp,nb_samples",
                ],
            )
            streams = audio.get("streams") or []
            if not streams:
                row["status"] = "video_only_no_embedded_audio"
                continue
            native = original.clocks.audio_clocks(streams[0], audio.get("frames") or [])
            row["native_audio_clock"] = native
            if (
                not native["contiguous"]
                or Fraction(native["frames"][-1]["end"]) - Fraction(native["frames"][0]["start"]) > 10
            ):
                row["status"] = "unavailable_audio_clock_or_bounds"
                continue
            try:
                sound = decode_audio(source, source_sha256=row["source_sha256"], max_seconds=10)
            except ValueError:
                require(sha(source) == row["source_sha256"], "Source bytes changed during PCM decode")
                row["status"] = "unavailable_canonical_pcm"
                continue
            ranges = source_ranges(sound.mapping, 0, len(sound.pcm))
            require(ranges and np.isfinite(sound.pcm).all(), "Complete finite sample/clock mapping required")
            pcm = destination / "pcm-f32.npy"
            with pcm.open("xb") as stream:
                np.save(stream, sound.pcm.astype("<f4"), allow_pickle=False)
            wav = destination / "source-mono16k.wav"
            with wav.open("xb") as stream:
                stream.write(sound.wav)
            row.update(
                status="native_video_and_audio_available",
                pcm_mapping=sound.mapping,
                whole_pcm_source_ranges=ranges,
                pcm_file=str(pcm),
                pcm_file_sha256=sha(pcm),
                wav_file=str(wav),
                wav_file_sha256=sha(wav),
                **source_origins(table, sound.mapping),
            )
            require(sha(source) == row["source_sha256"], "Source changed during full input binding")
    write(output / "native-inputs.json", records)
    result = {
        "schema": "chipmunk.grid-native-inputs-result/137",
        "status": "complete",
        "source_archives": 1,
        "planned_clips": len(clips),
        "retained_clips": len(records),
        "availability": dict(Counter(r["status"] for r in records)),
        "native_video_frames": sum(len(r.get("native_video_table", {}).get("pts", [])) for r in records),
        "retained_native_pictures": sum(len(r.get("pictures", [])) for r in records),
        "observed_mono16k_samples": sum(r.get("pcm_mapping", {}).get("samples", 0) for r in records),
        "private_native_inputs_sha256": sha(output / "native-inputs.json"),
        "freeze_sha256": sha(PUBLIC),
        "new_neural_calls": 0,
        "defaults_changed": False,
        "scope": "One source's two original selected clips; full native frames/PNG round trips and canonical PCM/source clocks. Not full cohort completion, independent identity/ancestry proof, geometry, generation or detection accuracy. All media/individual fingerprints private.",
    }
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)


def routed_popen(ordinal, argv, **kwargs):
    expected = TRIAL / f"source-{ordinal:02d}/bind-00/output"
    require(
        type(ordinal) is int
        and 0 <= ordinal <= 15
        and argv[1:5] == ["-B", "-m", "scripts.video_appearance_native_v108", "bind"]
        and argv[5:] == [str(expected), "0"],
        "Only exact explicit native data stage permitted",
    )
    return subprocess.Popen(
        [*argv[:3], "scripts.grid_native_inputs_v137", "bind", str(expected), str(ordinal)], **kwargs
    )


def run(ordinal):
    require(type(ordinal) is int and 0 <= ordinal <= 15, "One original source ordinal required")
    stage = TRIAL / f"source-{ordinal:02d}"
    stage.mkdir(exist_ok=True)
    proxy = types.SimpleNamespace(
        Popen=lambda argv, **kw: routed_popen(ordinal, argv, **kw),
        STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired,
    )
    scoped(
        acquisition.core.runtime.run,
        {
            "TRIAL": stage,
            "verify": verify,
            "prerequisites": lambda c, b: archive_record(ordinal) is not None,
            "subprocess": proxy,
            "__file__": str(Path(__file__)),
        },
    )("bind", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "bind"))
    parser.add_argument("arguments", nargs="*")
    args = parser.parse_args()
    if args.command == "freeze":
        require(not args.arguments, "No extra freeze arguments")
        freeze()
    elif args.command == "run":
        require(len(args.arguments) == 1, "One original source ordinal required")
        run(int(args.arguments[0]))
    else:
        verify()
        require(
            len(args.arguments) == 2 and args.arguments[1] in {str(i) for i in range(16)},
            "Exact source/output required",
        )
        ordinal, output = int(args.arguments[1]), Path(args.arguments[0])
        require(
            output.resolve() == (TRIAL / f"source-{ordinal:02d}/bind-00/output").resolve(),
            "Exclusive guarded output required",
        )
        output.mkdir(exist_ok=False)
        bind(ordinal, output)
