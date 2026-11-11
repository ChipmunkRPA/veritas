"""Compare the full-app video frame routes on an independent genuine occlusion control."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import video_frame_full_app_v168 as original
from scripts import video_frame_full_app_v171 as prior
from scripts.video_appearance_corrected_v112 import scoped

TRIAL = original.BASE / "video-frame-full-app-v175"
WORK = original.BASE / "video-frame-full-app-work-v175"
MODULE = "scripts.video_frame_full_app_v175"
GROUP = "03"
CASE_SHA256 = "501f2f362f48d1e904cf52a75118417b8df6108f1bb2f7ed8bf14ed4048357e2"
CASE_RECEIPT_SHA256 = "25de697b8894b0f5fd31fe3ebefb63aae4885cb8cb8c51493c65f321022a3a09"
PRIOR_RESULT = original.PANEL / "video-frame-full-app-complete-v171.json"
SOURCE_NOTICE = "https://www.usgs.gov/media/videos/working-usgs-md-de-dc-water-science-center"
NO_FACE = "no face detected in the sampled frames"
SCOPE = (
    "One genuine public-domain interview excerpt with a masked speaker. Full-app coverage and evidence only."
)
read, write, sha, require = prior.read, prior.write, prior.sha, prior.require


def preparation():
    file, receipt, receipt_sha = scoped(original.preparation, {"GROUP": GROUP})()
    clock = receipt["clock"]
    require(
        receipt_sha == CASE_RECEIPT_SHA256
        and receipt["file_sha256"] == CASE_SHA256
        and clock["video_time_base"] == "1/30000"
        and clock["picture_pts"] == list(range(0, 180180, 1001))
        and clock["video_end_pts"] == 180180
        and clock["verified_h264_packets"] == 180
        and clock["last_packet_duration"] == 1001
        and clock["audio_time_base"] == "1/48000"
        and clock["audio_samples"] == 288288
        and clock["audio_origin_pts"] == 0,
        "Exact prepared USGS recording and picture/audio clocks required",
    )
    return file, receipt, receipt_sha


def app_inputs(file, receipt):
    clock = receipt["clock"]
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        "file",
        "-threads",
        "1",
        "-copyts",
        "-i",
        str(file),
    ]
    rgb = subprocess.check_output(
        [
            *command,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-fps_mode",
            "passthrough",
            "-vf",
            "format=rgb24",
            "-enc_time_base",
            clock["video_time_base"],
            "-threads",
            "1",
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "pipe:1",
        ],
        timeout=120,
        text=True,
    )
    rows = [line.split(",") for line in rgb.splitlines() if line and not line.startswith("#")]
    require(
        f"#tb 0: {clock['video_time_base']}" in rgb.splitlines()
        and len(rows) == len(clock["picture_pts"])
        and all(
            len(row) == 6
            and int(row[0]) == 0
            and int(row[4]) == 1920 * 1080 * 3
            and re.fullmatch(r"[0-9a-f]{64}", row[5].strip())
            for row in rows
        )
        and [int(row[2]) for row in rows] == clock["picture_pts"],
        "All prepared USGS picture timestamps and native RGB hashes are required",
    )
    pcm = subprocess.check_output(
        [
            *command,
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "aformat=sample_fmts=s16:channel_layouts=mono,aresample=16000,asetpts=N/SR/TB",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "pipe:1",
        ],
        timeout=120,
    )
    samples = clock["audio_samples"] // 3
    require(len(pcm) == 2 * samples, "Complete canonical mono PCM required")
    mono = np.frombuffer(pcm, dtype="<i2").astype("<f4") / 32768.0
    return {
        "picture_pts": clock["picture_pts"],
        "video_time_base": clock["video_time_base"],
        "rgb_sha256": [row[5].strip() for row in rows],
        "audio_samples": samples,
        "pcm_sha256": hashlib.sha256(mono.tobytes()).hexdigest(),
    }


def preflight():
    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import decode_frames, probe_frames

    file, receipt, _ = preparation()
    expected = app_inputs(file, receipt)
    table = probe_frames(file)
    require(
        table.time_base == expected["video_time_base"]
        and list(table.pts) == expected["picture_pts"]
        and (table.width, table.height) == (1920, 1080)
        and not table.truncated
        and not table.interpolated,
        "The production probe must reproduce the complete prepared picture clock",
    )
    indices = (0, 90, 179)
    matched = []
    for frame in decode_frames(file, table, indices):
        require(
            frame.pts == expected["picture_pts"][frame.index]
            and hashlib.sha256(frame.rgb.tobytes()).hexdigest() == expected["rgb_sha256"][frame.index],
            "The production decoder differs from the independently hashed RGB",
        )
        matched.append(frame.index)
    audio = decode_audio(file, source_sha256=receipt["file_sha256"], max_seconds=8)
    samples = expected["audio_samples"]
    require(
        tuple(matched) == indices
        and len(audio.pcm) == audio.mapping["samples"] == samples
        and audio.mapping["pcm_sha256"] == expected["pcm_sha256"]
        and audio.mapping["timeline_time_base"] == "1/16000"
        and audio.mapping["timeline"] == [{"start_sample": 0, "end_sample": samples, "source_pts": 0}],
        "Representative app RGB and the full canonical mono clock and samples must match",
    )
    return {
        "status": "data_only",
        "native_pts_verified": len(table),
        "rgb_frames_replayed": len(matched),
        "mono_samples_verified": samples,
        "face_detection": "unmeasured",
        "model_calls": 0,
    }


def identity(previous, receipt_sha):
    prior_result = read(PRIOR_RESULT)
    prior_freeze = sha(prior.TRIAL / "freeze.json")
    require(
        prior_result["schema"] == "chipmunk.video-frame-full-app-readout/171"
        and prior_result["freeze_sha256"] == prior_freeze
        and prior_result["engineering_passed"] is True,
        "Exact successful v171 comparison required",
    )
    files = (
        Path(__file__),
        original.ROOT / "backend/tests/test_video_frame_full_app_v175.py",
        original.ROOT / "docs/video-frame-full-app-v175-method.md",
    )
    return {
        **previous,
        "schema": "chipmunk.video-frame-full-app-freeze/175",
        "v171_freeze_sha256": prior_freeze,
        "v171_result_sha256": sha(PRIOR_RESULT),
        "successor_code_sha256": {file.name: sha(file) for file in files},
        "private_case_receipt_sha256": receipt_sha,
        "private_inputs_sha256": sha(TRIAL / "inputs.json"),
        "source_notice": SOURCE_NOTICE,
        "input_group": GROUP,
        "scope": SCOPE,
    }


def freeze():
    require(not TRIAL.exists() and not WORK.exists(), "Preserve all prior or incomplete v175 attempts")
    previous, _, _ = prior.verify()
    original.baseline.checked_programs(read(original.baseline.PUBLIC))
    file, receipt, receipt_sha = preparation()
    inputs = app_inputs(file, receipt)
    TRIAL.mkdir(mode=0o700, exist_ok=False)
    for arm in original.ARMS:
        manifest = f"{arm}.yaml"
        (TRIAL / manifest).write_bytes((prior.TRIAL / manifest).read_bytes())
    write(TRIAL / "inputs.json", inputs)
    write(TRIAL / "freeze.json", identity(previous, receipt_sha))
    return {"status": "frozen", "freeze_sha256": sha(TRIAL / "freeze.json"), "arms": list(original.ARMS)}


def verify():
    previous, _, _ = prior.verify()
    file, receipt, receipt_sha = preparation()
    expected = identity(previous, receipt_sha)
    require(
        read(TRIAL / "freeze.json") == expected
        and all(
            (TRIAL / f"{arm}.yaml").read_bytes() == (prior.TRIAL / f"{arm}.yaml").read_bytes()
            for arm in original.ARMS
        ),
        "Exact v175 input and inherited arm manifests required",
    )
    return expected, file, receipt


def unscored(case, status, reason):
    return (
        case["target_status"] == status
        and case["target_reason"] == reason
        and case["target_rows_bound"]
        and case["target_raw_verified"]
        and case["target_scored_frames"] == 0
        and not case["target_signal_published"]
        and not case["target_series_published"]
    )


def verify_input(outputs, expected, digest):
    from app.core.canonical import canonical_hash

    result = original.verify_input(outputs, expected, digest)
    decoded = (outputs.get("decode") or {}).get("frames") or []
    faces = (outputs.get("faces") or {}).get("faces")
    if not isinstance(faces, list) or len(faces) != len(decoded):
        return {**result, "face_inputs_sha256": None}
    # Commit both positions and the actual geometry supplied by the unchanged face stage.
    anchors = [(row["frame_index"], row["pts"], boxes) for row, boxes in zip(decoded, faces, strict=True)]
    return {**result, "face_inputs_sha256": canonical_hash(anchors)}


def compare(reference, candidate):
    inherited = original.compare(reference, candidate)
    reference_faces = reference["input"]["faces"]
    candidate_faces = candidate["input"]["faces"]
    same_coverage = (
        reference["input"]["sampled_frames"] == candidate["input"]["sampled_frames"] > 0
        and 0 <= reference_faces == candidate_faces <= candidate["input"]["sampled_frames"]
        and reference["input"]["face_inputs_sha256"] == candidate["input"]["face_inputs_sha256"] is not None
        and all(
            dict(case["non_target_runs"]).get("vid.faces.yunet") == "ok" for case in (reference, candidate)
        )
    )
    if candidate_faces:
        reference_ok = unscored(reference, "skipped", "memory budget")
        candidate_ok = (
            inherited["candidate_scored_and_published"]
            and candidate["target_scored_frames"] == candidate_faces
        )
    else:
        reference_ok = unscored(reference, "not_applicable", NO_FACE)
        candidate_ok = unscored(candidate, "not_applicable", NO_FACE)
    return {
        **{
            key: inherited[key]
            for key in (
                "both_full_analyses_published",
                "both_exact_input_bindings",
                "same_non_target_route_statuses",
                "same_fuser",
            )
        },
        "same_condition_profile": reference["condition_profile"] == candidate["condition_profile"],
        "same_detected_face_coverage": same_coverage,
        "reference_disposition_verified": bool(reference_ok),
        "candidate_disposition_verified": bool(candidate_ok),
    }


def worker(arm, output):
    inherited = SimpleNamespace(
        serving=original.serving,
        worker=scoped(
            original.worker, {"record_case": scoped(original.record_case, {"verify_input": verify_input})}
        ),
    )
    scoped(prior.worker, {"TRIAL": TRIAL, "WORK": WORK, "verify": verify, "prior": inherited})(arm, output)


def run(arm):
    scoped(prior.run, {"TRIAL": TRIAL, "MODULE": MODULE, "verify": verify, "__file__": __file__})(arm)


def readout():
    summary = scoped(
        original.readout,
        {
            "TRIAL": TRIAL,
            "WORK": WORK,
            "verify": verify,
            "compare": compare,
            "__file__": __file__,
        },
    )()
    validated = summary["comparison"]
    verified_coverage = validated["same_detected_face_coverage"] and validated["both_exact_input_bindings"]
    for arm, case in summary["arms"].items():
        verified = (
            verified_coverage
            and validated[
                "reference_disposition_verified" if arm == "reference" else "candidate_disposition_verified"
            ]
        )
        case["target_coverage"] = (
            {
                "ok": "scored_and_published",
                "skipped": "activation_budget_unavailable",
                "not_applicable": "no_detected_face",
            }[case["target_status"]]
            if verified
            else "unverified"
        )
    return {
        **summary,
        "schema": "chipmunk.video-frame-full-app-readout/175",
        "control_type": "genuine_face_occlusion",
        "candidate_admitted_and_scored": summary["arms"]["batch-one"]["target_coverage"]
        == "scored_and_published",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("freeze", "preflight", "readout"):
        commands.add_parser(action)
    commands.add_parser("run").add_argument("arm", choices=original.ARMS)
    child = commands.add_parser("_worker")
    child.add_argument("arm", choices=original.ARMS)
    child.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "run":
        run(args.arm)
    elif args.action == "_worker":
        worker(args.arm, args.output)
    else:
        action = {"freeze": freeze, "preflight": preflight, "readout": readout}[args.action]
        print(json.dumps(action(), allow_nan=False))


if __name__ == "__main__":
    main()
