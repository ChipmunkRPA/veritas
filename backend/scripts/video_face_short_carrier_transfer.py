# SPDX-License-Identifier: Apache-2.0
"""Compare the fixed eight-second native face check on AVC MP4 and VP9 WebM."""

from __future__ import annotations

import argparse
import subprocess
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import cv2

from scripts import video_face_short_prefix_transfer as prior
from scripts.video_common_sources_v89 import read, require, sha, write

FORMATS = {
    "avc": {
        "suffix": "mp4",
        "codec": "h264",
        "tag": "avc1",
        "carrier": "mp4",
        "options": [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-tag:v",
            "avc1",
            "-video_track_timescale",
            "30000",
            "-movflags",
            "+faststart",
        ],
        "tandem_crf": "35",
    },
    "vp9": {
        "suffix": "webm",
        "codec": "vp9",
        "tag": "[0][0][0][0]",
        "carrier": "webm",
        "options": [
            "-c:v",
            "libvpx-vp9",
            "-deadline",
            "realtime",
            "-cpu-used",
            "5",
            "-crf",
            "30",
            "-b:v",
            "0",
        ],
        "tandem_crf": "41",
    },
}
GATE = {
    "minimum_recovered_late": 9,
    "minimum_groups": 4,
    "minimum_exact": 6,
    "maximum_new_ordinary": 1,
    "maximum_new_cut_or_loop": 0,
    "minimum_full_coverage": 0.75,
    "maximum_wrong_edited": 1,
}
PLAN = {
    "profile": "240 verified upright 720p 8-bit 4:2:0 pictures, nominal 30 fps",
    "clock": "AVC uses exact 1/30 spacing. VP9 WebM uses 1 ms ticks with each relative PTS equal to its nearest 30 fps lattice tick. No tolerance for dropped, duplicated or interpolated timestamps.",
    "method": "Reuse the three windows starting at native pictures 0, 60 and 120. Same appearance and surrounding-motion rule, fixed point .10. No development fit.",
    "sources": "Two development and six held known source/donor components. Exclude every original, donor and component in the previous eight-second study. Retained publisher files have earlier project exposure.",
    "experiment": "Render both carriers independently from the same verified original RGB inputs; tandem is a second encode in the same carrier. Preserve exact frame-index truth and compare current Full on identical delivered bytes.",
    "source_hard_limit_bytes": 96 * 1024**2,
}
OLD_SHA = "1ee86ff12dc7a3099b77072e4b558353b91ed7d8af410b9784d1f04319f9ff26"


def encoder_info():
    return {
        name: sha(Path(subprocess.check_output(["which", name], text=True).strip()))
        for name in ("ffmpeg", "ffprobe")
    }


def freeze(parent):
    from app.detectors.video import face_paired_edge, face_short_proposal
    from app.evidence import face_short_proposal_binding

    old = read(parent / "freeze.json")
    require(
        sha(parent / "freeze.json") == OLD_SHA and old["existing_point"] == face_short_proposal_binding.POINT,
        "Exact previous HEVC roster required",
    )
    _, common = prior.dual.archive_context()
    qualified = prior.NATIVE_TRIAL / "qualified-plan.json"
    qualify_sha = sha(qualified)
    require(
        qualify_sha == read(prior.NATIVE_TRIAL / "summary.json")["qualified_private_plan_sha256"],
        "Native publisher clock qualification changed",
    )
    assets = asset_map()
    meta = prior.corpus.trial.prior.comparison.metadata()
    previous = old["development"] + old["held"]
    source_keys = {r["url"] for r in previous}
    donors = {k for r in previous for k in r["donors"]}
    components = {r["original_component"] for r in previous}
    roster = []
    for file in sorted(common.DATA.glob("group-*-complete.json")):
        row = read(file)
        source = assets[row["group"]]
        host = source["original_videos"]
        clock = host["frame_table"]
        facts = meta[row["group"]]
        component = host["qualification"]["component"]
        if (
            "child" in facts["tags"]
            or facts["url"] in source_keys
            or facts["donors"] & donors
            or component in components
            or len(clock["pts"]) < prior.N
            or clock["width"] not in {1280, 1920}
            or any(
                (p - clock["pts"][0]) * Fraction(clock["time_base"]) != Fraction(i, 30)
                for i, p in enumerate(clock["pts"][: prior.N])
            )
            or any(
                not a["qualification"]["eligible"]
                or a["frame_table"]["pts"][: prior.N] != clock["pts"][: prior.N]
                or a["frame_table"]["time_base"] != clock["time_base"]
                for a in source.values()
            )
        ):
            continue
        roster.append(
            {
                "index": row["index"],
                "role": row["role"],
                "url": facts["url"],
                "donors": sorted(facts["donors"]),
                "source_sha256": sha(file),
                "source_group": row["group"],
                "original_component": component,
            }
        )
    development = [r for r in roster if r["role"] == "selection"][:2]
    require(
        len(development) == 2
        and development[0]["url"] != development[1]["url"]
        and development[0]["original_component"] != development[1]["original_component"]
        and not set(development[0]["donors"]) & set(development[1]["donors"]),
        "Development split overlaps",
    )
    held = []
    reserved = {r["index"] for r in development}
    used = {r["url"] for r in development}
    used_donors = {k for r in development for k in r["donors"]}
    used_components = {r["original_component"] for r in development}
    for row in sorted(roster, key=lambda r: (r["role"] != "confirmation", r["index"])):
        if (
            row["index"] in reserved
            or row["url"] in used
            or set(row["donors"]) & used_donors
            or row["original_component"] in used_components
        ):
            continue
        held.append(row)
        used.add(row["url"])
        used_donors.update(row["donors"])
        used_components.add(row["original_component"])
        if len(held) == 6:
            break
    require(len(held) == 6, "Missing six independent held groups")
    code = (
        Path(__file__),
        Path(prior.__file__),
        Path(face_paired_edge.__file__),
        Path(face_short_proposal.__file__),
        Path(face_short_proposal_binding.__file__),
    )
    return {
        "schema": "chipmunk.video-face-short-carrier-transfer/1",
        "method": PLAN,
        "formats": FORMATS,
        "gate": GATE,
        "prior_hevc_freeze_sha256": OLD_SHA,
        "qualified_plan_sha256": qualify_sha,
        "code": {p.name: sha(p) for p in code},
        "binaries": encoder_info(),
        "opencv": cv2.__version__,
        "weights": prior.earlier.WEIGHTS,
        "point": face_short_proposal_binding.POINT,
        "development": development,
        "held": held,
    }


def asset_map():
    result = {}
    for a in read(prior.NATIVE_TRIAL / "qualified-plan.json")["assets"]:
        if a["kind"] in ("original_videos", "inswapper", "uniface"):
            result.setdefault(a["group"], {})[a["kind"]] = a
    return result


def clock(path, profile):
    from app.detectors.video.decode import probe_frames
    from app.detectors.video.face_return import source_carrier

    table = probe_frames(path, timeout_s=18)
    [probe] = prior.probe_encoding(path)
    require(
        probe["codec_name"] == profile["codec"]
        and probe["codec_tag_string"] == profile["tag"]
        and probe["pix_fmt"] == "yuv420p"
        and probe["r_frame_rate"] == "30/1"
        and source_carrier(path) == profile["carrier"],
        "Unexpected encoding/carrier",
    )
    require(
        len(table) == prior.N
        and not (table.rotation or table.truncated or table.interpolated)
        and (table.width, table.height) == (table.coded_width, table.coded_height) == (1280, 720),
        "Unexpected native decoded profile",
    )
    origin = table.pts[0]
    if profile["carrier"] == "webm":
        require(
            table.tb == Fraction(1, 1000)
            and all(p - origin == (100 * i + 1) // 3 for i, p in enumerate(table.pts)),
            "WebM ticks do not match the rounded native 30 fps lattice",
        )
    else:
        require(
            all((p - origin) * table.tb == Fraction(i, 30) for i, p in enumerate(table.pts)),
            "MP4 clock is not exact 30 fps",
        )
    return table


def encode(path, profile, *, pictures=None, source=None):
    target = path.with_suffix("." + profile["suffix"])
    options = list(profile["options"])
    if source:
        options[options.index("-crf") + 1] = profile["tandem_crf"]
    args = ["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-filter_threads", "1"]
    input_ = (
        ["-i", str(source.with_suffix("." + profile["suffix"]))]
        if source
        else ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "30", "-i", "pipe:0"]
    )
    command = [
        *args,
        *input_,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "2",
        *options,
        str(target),
    ]
    if source:
        subprocess.run(command, check=True, timeout=45)
        return
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        try:
            for picture in pictures:
                proc.stdin.write(memoryview(picture).cast("B"))
            proc.stdin.close()
            require(proc.wait(timeout=45) == 0, "Carrier encode failed")
        finally:
            if proc.poll() is None:
                proc.kill()


@contextmanager
def encoding_hooks(profile, folder):
    # Existing author/measure paths use MP4 names. WebM needs its real suffix for muxing and native reads.
    def native_clock(path):
        return clock(path.with_suffix("." + profile["suffix"]), profile)

    def encoded_hash(path):
        if path.parent == folder and path.suffix == ".mp4":
            path = path.with_suffix("." + profile["suffix"])
        return sha(path)

    def write_video(path, pictures=None, source=None):
        return encode(path, profile, pictures=pictures, source=source)

    with (
        patch.object(prior, "clock", native_clock),
        patch.object(prior, "sha", encoded_hash),
        patch.object(prior, "write_video", write_video),
    ):
        yield


def run(folder, lock, split, carrier):
    from app.detectors.image.faces import load_yunet

    profile = FORMATS[carrier]
    assets = asset_map()
    weights = prior.earlier.ROOT / "models/weights"
    require(
        {name: sha(weights / name) for name in lock["weights"]} == lock["weights"],
        "Pinned face assets changed",
    )
    detector = load_yunet(weights / "face_detection_yunet_2023mar.onnx")
    recognizer = cv2.FaceRecognizerSF.create(str(weights / "face_recognition_sface_2021dec.onnx"), "")
    results = []
    for group, member in enumerate(lock[split], 1):
        local = folder / f"{carrier}-{split}-{group}"
        local.mkdir(mode=0o700)
        partner = lock[split][group % len(lock[split])]
        with encoding_hooks(profile, local):
            rows = prior.prepare_group(local, member, partner, group, assets)
            for row in rows:
                path = local / (row["role"] + "." + profile["suffix"])
                candidate = prior.measure(path, row, detector, recognizer, [])
                results.append({"group": group, "carrier": carrier, **row, **candidate})
                print(
                    carrier,
                    split,
                    group,
                    row["role"],
                    [w["match"]["frames"] for w in candidate["windows"] if w["match"]],
                    round(candidate["elapsed"], 2),
                    flush=True,
                )
        require(
            sum(p.stat().st_size for p in folder.glob("*/*") if p.suffix in (".mp4", ".webm"))
            <= PLAN["source_hard_limit_bytes"],
            "Bounded local media allocation exceeded",
        )
        write(local / "results.json", [r for r in results if r["group"] == group])
    write(folder / f"{carrier}-{split}.json", {"freeze_sha256": sha(folder / "freeze.json"), "rows": results})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("freeze", "development", "held"))
    p.add_argument("folder", type=Path)
    p.add_argument("earlier", type=Path)
    p.add_argument("--carrier", choices=FORMATS, default="avc")
    args = p.parse_args()
    cv2.setNumThreads(2)
    lock = freeze(args.earlier)
    if args.phase == "freeze":
        args.folder.mkdir(mode=0o700)
        write(args.folder / "freeze.json", lock)
        return
    require(read(args.folder / "freeze.json") == lock, "Carrier comparison freeze differs")
    run(args.folder, lock, args.phase, args.carrier)


if __name__ == "__main__":
    main()
