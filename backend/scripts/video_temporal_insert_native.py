# SPDX-License-Identifier: Apache-2.0
"""Reproduce bounded source acquisition and native inserted-video comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
import tempfile
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from scripts import review_video_temporal_insert_sources as source

ROOT = Path(__file__).resolve().parents[2]
INTAKE = ROOT / "docs/evaluation-panels/video-temporal-insert-source-intake-2026-11-13.json"
INTAKE_SHA256 = "7defd929fa0b4a229145810b8c35ae5e3f879367ae59bbd2bd2de24304fb557f"
SCHEMA = "chipmunk.video-temporal-insert-native/1"
MEMBER_PREFIX_BYTES = 512
MEMBER_HEADER = struct.Struct("<4s5H3L2H")
RANGE_ROOT = f"https://zenodo.org/records/{source.RECORD}/files/{source.NAME}"
THUMB_SIDE = 96
THUMB_BYTES = THUMB_SIDE ** 2
PROTOCOL = {
    "clock": "24 fps on both ffprobe's native frame table and an independent FFmpeg framehash decode, with zero native origin.",
    "source_binding": "Search the released triplet for the best global split within four frames of the paper's implied zero-based start. "
                      "Compare aligned authentic prefix, generated segment and authentic suffix at every position in that range.",
    "paired_check": "Compare an earlier video with a longer submitted video. Fit one frame insertion and preserve all earlier frames "
                    "in order. Use 96-square luma standardized per frame. Require at least 24 authentic frames on each side, "
                    "a median retained difference no greater than .10, and a 90th percentile no greater than .22 on each side. "
                    "Withhold if the candidate split is ambiguous by over 3 frames at the fixed profile-distance tolerance.",
    "controls": "Apply the same local H.264 delivery to both members. Test the generated insertion against unchanged delivery, "
                "a genuine insertion of the same length at the same boundary, and a different genuine source as earlier copy. "
                "The genuine insert should be locatable as a physical change but provides no AI authorship evidence.",
    "standalone": "Compare app HSV shot boundaries with robust luma changes using fixed symmetric half-second windows, "
                  "an absolute .10 standardized feature step and a 6 MAD margin. Report endpoints independently "
                  "within three frames and retain ordinary-control boundary observations.",
    "limits": "Four smallest equal-cadence source triplets from a single publisher. No model, calibration or app change.",
}
CONTROL_ROLES = ("same_encode", "ordinary_insert", "wrong_reference")
SHA256 = re.compile(r"[a-f0-9]{64}")


def exact_json_sha(value):
    return source.sha(json.dumps(value, separators=(",", ":")).encode())


def process(args, *, timeout=150):
    return subprocess.run(args, check=True, stdout=subprocess.PIPE, timeout=timeout).stdout


def ffmpeg():
    return ["ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-filter_threads", "1"]


def input_file(path):
    return ["-protocol_whitelist", "file", "-i", "file:" + str(path.resolve())]


def independent_decode(path, table):
    import numpy as np

    cmd = [*ffmpeg(), "-copyts", *input_file(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
           "-vf", f"scale={THUMB_SIDE}:{THUMB_SIDE}:flags=area,format=gray"]
    raw = process([*cmd, "-threads", "1", "-f", "rawvideo", "pipe:1"])
    lines = process([*cmd, "-enc_time_base", table.time_base, "-threads", "1", "-f", "framehash", "-hash", "sha256",
                     "pipe:1"]).decode().splitlines()
    source.require(f"#tb 0: {table.time_base}" in lines, "Independent decode used a different time base")
    rows = [line.replace(" ", "").split(",") for line in lines if line and not line.startswith("#")]
    source.require(len(raw) == THUMB_BYTES * len(rows) == THUMB_BYTES * len(table), "Native frame counts differ")
    hashes = []
    for index, (track, _, pts, _, size, digest) in enumerate(rows):
        block = raw[index*THUMB_BYTES:(index+1)*THUMB_BYTES]
        source.require(track == "0" and int(pts) == table.pts[index] and int(size) == THUMB_BYTES
                       and source.sha(block) == digest, "Independent pixels or timestamp differ")
        hashes.append(digest)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(-1, THUMB_SIDE, THUMB_SIDE)
    return pixels, hashes


def native_clip(path):
    from app.detectors.video.decode import probe_frames

    table = probe_frames(path, cap=600, timeout_s=80)
    stream, = json.loads(process(["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
                                  "-show_entries", "stream=codec_name,pix_fmt,color_space", "-of", "json",
                                  "file:" + str(path.resolve())]))["streams"]
    source.require(not table.truncated and not table.interpolated and table.rotation == 0 and table.pts[0] == 0
                   and all((b-a) * table.tb == Fraction(1, 24) for a, b in zip(table.pts, table.pts[1:])),
                   "Unsupported clock, frame count or orientation")
    pixels, hashes = independent_decode(path, table)
    info = {"sha256": file_sha(path), "frames": len(table), "width": table.width, "height": table.height,
            "time_base": table.time_base, "tick_step": table.pts[1], "pts_sha256": exact_json_sha(list(table.pts)),
            "luma_sha256": exact_json_sha(hashes), "codec": stream["codec_name"], "pixel_format": stream["pix_fmt"],
            "color_space": stream.get("color_space")}
    return table, pixels, hashes, info


def frame_error(first, second):
    import numpy as np

    return np.mean(np.abs(first.astype(np.float32) - second.astype(np.float32)), axis=(1, 2))


def bind_published(case, genuine, generated, merged):
    import numpy as np

    expected, stop = case["publisher_frames"]
    length = stop - expected
    source.require((len(genuine), len(generated), len(merged)) == (case["original_frames"], length, case["merged_frames"]),
                   "Publisher and native picture counts differ")
    trials = []
    for start in range(expected-5, expected+4):
        end = start + length
        errors = [frame_error(genuine[:start], merged[:start]),
                  frame_error(generated, merged[start:end]), frame_error(genuine[start:], merged[end:])]
        means = [round(float(np.mean(block)), 6) for block in errors]
        trials.append({"bounds": [start, end], "mean_luma_errors": means, "median_luma_errors": [
            round(float(np.median(block)), 6) for block in errors]})
    strongest = min(trials, key=lambda row: sum(row["mean_luma_errors"]))
    source.require(max(strongest["mean_luma_errors"]) < 6
                   and sum(row["mean_luma_errors"] == strongest["mean_luma_errors"] for row in trials) == 1,
                   "Publisher interval could not be bound to the three native sources")
    return {"expected_from_one_based_paper": [expected-1, stop-1], "native_frames": strongest["bounds"], "trials": trials}


def normalized(pixels):
    import numpy as np

    luma = pixels.astype(np.float32)
    low, mid, high = np.percentile(luma, (20, 50, 80), axis=(1, 2), keepdims=True)
    return ((luma - mid) / np.maximum(high - low, 24)).astype(np.float32)


def pair_interval(first, second):
    import numpy as np

    added = len(second) - len(first)
    if not 24 <= added <= 240 or len(first) < 49:
        return {"status": "not_a_longer_pair", "bounds": None}
    a, b = normalized(first), normalized(second)
    before = frame_error(a, b[:len(a)])
    after = frame_error(a, b[added:])
    positions = np.arange(24, len(a) - 23)
    profile = np.r_[0., np.cumsum(before - after, dtype=np.float64)]
    best = int(positions[np.argmin(profile[positions])])
    before_stats, after_stats = ([round(float(value), 6) for value in np.percentile(values, (50, 90))]
                                 for values in (before[:best], after[best:]))
    close = positions[profile[positions] <= profile[best] + 0.035]
    selected = [best, best + added]
    details = {"best": selected, "uncertain_starts": [int(close.min()), int(close.max())],
               "before_error_p50_p90": before_stats, "after_error_p50_p90": after_stats}
    if any(mid > .10 or upper > .22 for mid, upper in (before_stats, after_stats)):
        return {"status": "insufficient_matching_frames", "bounds": None, **details}
    if int(close.max() - close.min()) > 3:
        return {"status": "ambiguous_insertion", "bounds": None, **details}
    return {"status": "paired_insertion", "bounds": selected, **details}


def standalone(path, table, pixels):
    import numpy as np

    from app.detectors.video.decode import decode_stream
    from app.detectors.video.shots import detect_shots

    app = detect_shots(decode_stream(path, table, timeout_s=80), table)
    source.require(app.frames_seen == len(table), "App shot analysis did not assess the complete video")
    structural = frame_error(normalized(pixels[1:]), normalized(pixels[:-1]))
    observations = []
    for i in range(12, len(structural)-12):
        near = np.r_[structural[i-12:i], structural[i+1:i+13]]
        median = np.median(near)
        margin = structural[i] - median
        if structural[i] > .10 and margin > 6 * max(.005, float(np.median(abs(near - median)))):
            observations.append(i + 1)
    return {"app": [row["frame_index"] for row in app.boundaries], "structural": observations}


def same_delivery(path, target, *, width, height, insert=None, length=None):
    scale = f"scale={width}:{height}:flags=area,setsar=1"
    command = [*ffmpeg()]
    if length is not None:
        command += ["-stream_loop", "-1"]
    command += input_file(path)
    if insert is None:
        command += ["-map", "0:v:0", "-vf", scale]
    else:
        point, inserted = insert
        graph = (f"[0:v:0]{scale},split=3[a][b][c];"
                 f"[a]trim=end_frame={point},setpts=PTS-STARTPTS[left];"
                 f"[b]trim=end_frame={inserted},setpts=PTS-STARTPTS[insert];"
                 f"[c]trim=start_frame={point},setpts=PTS-STARTPTS[right];"
                 "[left][insert][right]concat=n=3:v=1:a=0[v]")
        command += ["-filter_complex_threads", "1", "-filter_complex", graph, "-map", "[v]"]
    if length is not None:
        command += ["-frames:v", str(length)]
    process([*command, "-an", "-sn", "-dn", "-c:v", "libx264", "-threads", "1", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p", "-g", "48", "-keyint_min", "48", "-sc_threshold", "0",
             "-r", "24", "-video_track_timescale", "12288", "-map_metadata", "-1", "-y", str(target)])


def delivery_size(table):
    if table.width >= table.height:
        return 480, round(240 * table.height / table.width) * 2
    return round(240 * table.width / table.height) * 2, 480


def audit_group(directory, case, scratch, references):
    from app.detectors.video.decode import DecodeError
    from app.evidence import video_reference

    files = {role: directory / row["name"] for role, row in case["members"].items()}
    for role, member in case["members"].items():
        file_receipt(files[role], member)
    original, generated, merged = (native_clip(files[role]) for role in ("authentic", "fake_segments", "merged"))
    binding = bind_published(case, original[1], generated[1], merged[1])
    start, stop = binding["native_frames"]
    width, height = delivery_size(original[0])
    try:
        current_native, _ = video_reference.compare(files["authentic"], files["merged"], include_trimmed=True)
        native_status = current_native["status"]
    except DecodeError as error:
        source.require(str(video_reference.MAX_PIXELS) in str(error).replace(",", ""), "Unexpected native app error")
        native_status = "above_app_pixel_limit"
    control_paths = {role: scratch / f"{case['example']}-{role}.mp4" for role in ("earlier", "edited", "ordinary_insert")}
    for role, target in control_paths.items():
        earlier = role != "edited"
        same_delivery(files["authentic" if earlier else "merged"], target, width=width, height=height,
                      insert=(start, stop-start) if role == "ordinary_insert" else None)
    reference, edited, ordinary = (native_clip(control_paths[role]) for role in ("earlier", "edited", "ordinary_insert"))
    source.require(len(reference[0]) == len(original[0]) and len(edited[0]) == len(ordinary[0]) == len(merged[0]),
                   "Delivery changed the labeled frame clock")
    result_rows = []
    rows = (
        ("same_encode", "earlier", reference, None),
        ("ordinary_insert", "ordinary_insert", ordinary, [start, stop]),
        ("generated_insert", "edited", edited, [start, stop]),
        ("wrong_reference", "edited", edited, None),
    )
    for role, path_key, submitted, known in rows:
        path = control_paths[path_key]
        earlier, earlier_path = reference, control_paths["earlier"]
        if role == "wrong_reference":
            earlier_path = scratch / f"{case['example']}-wrong.mp4"
            same_delivery(directory / references["name"], earlier_path, width=width, height=height, length=len(reference[0]))
            earlier = native_clip(earlier_path)
        proposed = pair_interval(earlier[1], submitted[1])
        if proposed["bounds"] is not None:
            first, last = proposed["bounds"]
            witness = [first-1, (first + last)//2, last]
            proposed["evidence"] = [{"frame": i, "pts": submitted[0].pts[i], "gray_96_sha256": submitted[2][i]} for i in witness]
        existing, _ = video_reference.compare(earlier_path, path, include_trimmed=True, include_soundtrack=False)
        result_rows.append({
            "role": role, "known": known, "submitted": submitted[3],
            "earlier": earlier[3],
            "existing_status": existing["status"], "paired": proposed,
            "standalone": standalone(path, submitted[0], submitted[1]) if role != "wrong_reference" else None,
        })
    return {
        "example": case["example"], "native_pair_app_status": native_status, "source_binding": binding,
        "native": {"authentic": original[3], "fake_segments": generated[3], "merged": merged[3]},
        "native_merged_shots": standalone(files["merged"], merged[0], merged[1]),
        "deliveries": result_rows,
    }


def summarize(groups):
    summary = {key: {} for key in ("native", *CONTROL_ROLES, "generated_insert")}
    for group in groups:
        binding = group["source_binding"]
        native = summary["native"]
        native["clock_verified_files"] = native.get("clock_verified_files", 0) + len(group["native"])
        native["decoded_frames"] = native.get("decoded_frames", 0) + sum(row["frames"] for row in group["native"].values())
        native["publisher_labels_match_picture_bindings"] = native.get("publisher_labels_match_picture_bindings", 0) + (
            binding["native_frames"] == binding["expected_from_one_based_paper"])
        for role, known, current, paired, shots in (
            ("native", binding["native_frames"], group["native_pair_app_status"], {}, group["native_merged_shots"]),
            *((row["role"], row["known"], row["existing_status"], row["paired"], row["standalone"])
              for row in group["deliveries"]),
        ):
            counts = summary[role]
            counts["files"] = counts.get("files", 0) + 1
            counts["current_compared"] = counts.get("current_compared", 0) + (current == "measured")
            proposed = paired.get("bounds")
            counts["paired_reported"] = counts.get("paired_reported", 0) + (proposed is not None)
            counts["paired_exact"] = counts.get("paired_exact", 0) + bool(known and proposed == known)
            counts["paired_within_three_frames"] = counts.get("paired_within_three_frames", 0) + bool(
                known and proposed and max(abs(a-b) for a, b in zip(known, proposed, strict=True)) <= 3)
            if shots is not None:
                for method, boundaries in shots.items():
                    for name, increment in (
                        ("files_with_observations", bool(boundaries)),
                        ("near_boundary_count", sum(any(abs(guess-point) <= 3 for guess in boundaries) for point in (known or []))),
                        ("total_observations", len(boundaries)),
                    ):
                        key = f"{method}_{name}"
                        counts[key] = counts.get(key, 0) + increment
    return summary


def verify_clock(record):
    frame_count = record["frames"]
    source.require(frame_count >= 2 and record["width"] > 0 and record["height"] > 0
                   and SHA256.fullmatch(record["sha256"]) and SHA256.fullmatch(record["luma_sha256"]),
                   "Invalid frame geometry or hash")
    step = record["tick_step"]
    source.require(Fraction(record["time_base"]) * step == Fraction(1, 24)
                   and record["pts_sha256"] == exact_json_sha(list(range(0, frame_count*step, step))),
                   "Frame clock does not match the exact native cadence")


def verify_standalone(record, frames):
    source.require(set(record) == {"app", "structural"}, "Missing standalone route")
    for values in record.values():
        source.require(values == sorted(set(values)) and all(0 < i < frames for i in values),
                       "Standalone observation exceeds the native clock")


def review(document):
    source.require(document["schema"] == SCHEMA and document["intake_sha256"] == INTAKE_SHA256
                   and document["method"] == PROTOCOL, "Native verification scope differs")
    for case, group in zip(intake(), document["groups"], strict=True):
        source.require(case["example"] == group["example"], "Unexpected native triplet")
        native = group["native"]
        generated = case["publisher_frames"][1] - case["publisher_frames"][0]
        for role, length in (("authentic", case["original_frames"]), ("fake_segments", generated),
                             ("merged", case["merged_frames"])):
            verify_clock(native[role])
            source.require(native[role]["frames"] == length, "Unexpected native frame count")
        expected = [value - 1 for value in case["publisher_frames"]]
        binding = group["source_binding"]
        trials = binding["trials"]
        source.require(binding["expected_from_one_based_paper"] == expected and
                       [item["bounds"] for item in trials] == [[start, start + generated] for start in
                                                               range(expected[0]-4, expected[0]+5)],
                       "Unexpected alignment trials")
        selected = min(trials, key=lambda row: sum(row["mean_luma_errors"]))
        source.require(binding["native_frames"] == selected["bounds"]
                       and all(len(row["mean_luma_errors"]) == len(row["median_luma_errors"]) == 3
                               and all(0 <= error <= 255 for error in row["mean_luma_errors"] + row["median_luma_errors"])
                               for row in trials)
                       and max(selected["mean_luma_errors"]) < 6,
                       "Publisher interval does not match the picture measurements")
        verify_standalone(group["native_merged_shots"], native["merged"]["frames"])
        deliveries = group["deliveries"]
        source.require([row["role"] for row in deliveries] == ["same_encode", "ordinary_insert", "generated_insert", "wrong_reference"],
                       "Unexpected matching conditions")
        for row in deliveries:
            earlier, submitted = row["earlier"], row["submitted"]
            verify_clock(earlier)
            verify_clock(submitted)
            source.require((earlier["width"], earlier["height"]) == (submitted["width"], submitted["height"])
                           and all(record["codec"] == "h264" for record in (earlier, submitted)),
                           "Control delivery profile differs")
            role = row["role"]
            source.require(earlier["frames"] == native["authentic"]["frames"], "Earlier delivery frame count differs")
            if role != "wrong_reference":
                verify_standalone(row["standalone"], submitted["frames"])
            if role == "same_encode":
                source.require(earlier == submitted and row["known"] is None, "Unchanged control differs")
            else:
                source.require(submitted["frames"] == native["merged"]["frames"], "Inserted delivery frame count differs")
                source.require(row["known"] == (None if role == "wrong_reference" else binding["native_frames"]),
                               "Delivery truth differs")
            paired = row["paired"]
            if (bounds := paired["bounds"]) is not None:
                start, stop = bounds
                source.require(paired["status"] == "paired_insertion" and bounds == paired["best"]
                               and 24 <= start < stop <= submitted["frames"] - 24
                               and stop - start == submitted["frames"] - earlier["frames"]
                               and paired["uncertain_starts"][0] <= start <= paired["uncertain_starts"][1]
                               and paired["uncertain_starts"][1] - paired["uncertain_starts"][0] <= 3,
                               "Reported insertion does not match both timelines")
                for frame, item in zip((start-1, (start+stop)//2, stop), paired["evidence"], strict=True):
                    source.require(item["frame"] == frame and item["pts"] == frame * submitted["tick_step"]
                                   and SHA256.fullmatch(item["gray_96_sha256"]), "Evidence frame or time differs")
    result = summarize(document["groups"])
    source.require(document["results"] == result, "Native replay counts differ")
    return result


def confirmation_plan(folder):
    """Select untouched source pairs before acquiring or scoring their media."""
    index = source.measure(folder)
    source.require(index["smallest_equal_cadence_triplets"] == intake(), "Public preflight roster differs")
    previous = {row["example"] for row in intake()}
    content = source.archive_part(folder, "tail")
    with zipfile.ZipFile(source.SuffixReader(content, source.SIZE)) as archive:
        members = {item.filename: item for item in archive.infolist()}
    payload = source.archive_part(folder, "metadata")
    label_name = f"{source.ROOT}/metadata/frame_insertion_metadata.csv"
    rows = source.csv_rows(source.stored_member(members[label_name], payload, source.INDEX_PARTS["metadata"][0]))
    selection = []
    for row in rows:
        example = int(row["video_name"].removeprefix("Ornek_"))
        if example in previous or Fraction(row["fps_original"]) != 24 or Fraction(row["fps_fake"]) != 24:
            continue
        sources = {}
        for role in ("authentic", "fake_segments"):
            suffix, ext = source.MEDIA[role]
            name = f"Ornek_{example}{suffix}{ext}"
            item = members[f"{source.ROOT}/{role}/{name}"]
            source.require(item.compress_type == zipfile.ZIP_STORED, "Unexpected stored source")
            sources[role] = {"name": name, "bytes": item.file_size, "header_offset": item.header_offset, "crc32": f"{item.CRC:08x}"}
        selection.append({"example": example, "earlier_frames": int(row["original_frames"]),
                          "generated_frames": int(row["fake_frames_added"]), "sources": sources,
                          "bytes": sum(item["bytes"] for item in sources.values())})
    source.require(len(selection) == index["same_cadence_24fps"] - len(previous), "Missing eligible confirmation source")
    ordered = sorted(selection, key=lambda row: (row["bytes"], row["example"]))[:8]
    for slot, row in enumerate(ordered):
        row["authored_renders"] = [
            {"earlier_split": 48 + 8 * slot, "generated_slice": [12, 12 + (30, 54, 72, 102)[slot % 4]]},
            {"earlier_split": 122 + 4 * slot, "generated_slice": [6, 6 + (102, 72, 54, 30)[slot % 4]]},
        ]
        source.require(all(entry["earlier_split"] < row["earlier_frames"] - 24 and
                           entry["generated_slice"][1] <= row["generated_frames"] for entry in row["authored_renders"]),
                       "Proposed authored geometry exceeds an eligible native source")
    return {
        "schema": "chipmunk.video-temporal-insert-confirmation-plan/1",
        "source_intake_sha256": INTAKE_SHA256,
        "excludes": sorted(previous),
        "cases": ordered,
        "bytes": sum(row["bytes"] for row in ordered),
        "controls": ["untouched_delivery", "matched_ordinary_insertion", "wrong_reference", "equal_length_replacement"],
        "selection": "Eight smallest remaining equal-cadence original/generated pairs. Audit content overlap "
                     "and clocks before authoring; record duplicates and source declines without replacement.",
        "score_before_opening_results": "Keep the fixed paired method. Record exact and ±3-frame counts, input eligibility, "
                                        "wrong-reference and unchanged outcomes, and gain at unique-source level.",
    }


def audit(directory):
    import cv2

    cv2.setNumThreads(1)
    cases = intake()
    results = []
    with tempfile.TemporaryDirectory(prefix="insert-delivery-") as folder:
        for index, case in enumerate(cases):
            scratch = Path(folder)
            row = audit_group(directory, case, scratch, cases[(index+1) % len(cases)]["members"]["authentic"])
            results.append(row)
            print(f"Measured native triplet and four deliveries for {case['example']}", flush=True)
    return {"schema": SCHEMA, "intake_sha256": INTAKE_SHA256, "method": PROTOCOL, "groups": results,
            "results": summarize(results)}


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def intake():
    source.require(file_sha(INTAKE) == INTAKE_SHA256, "Frozen native roster changed")
    return json.loads(INTAKE.read_bytes())["smallest_equal_cadence_triplets"]


def file_receipt(path, member):
    crc = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            crc = zlib.crc32(block, crc)
            digest.update(block)
    size = path.stat().st_size
    source.require(size == member["bytes"] and f"{crc:08x}" == member["crc32"], f"Member differs: {path.name}")
    return {"sha256": digest.hexdigest(), "bytes": size, "crc32": f"{crc:08x}"}


def fetch_member(directory, role, member):
    target = directory / member["name"]
    if target.is_file():
        return file_receipt(target, member)
    start = member["header_offset"]
    stop = start + member["bytes"] + MEMBER_PREFIX_BYTES - 1
    with tempfile.TemporaryDirectory(dir=directory, prefix="range-") as tmp:
        headers = Path(tmp) / "headers"
        part = Path(tmp) / "media"
        url = f"{RANGE_ROOT}?download=1&chipmunk-member={member['name']}"
        command = [
            "curl", "-fsSL", "--retry", "2", "--max-time", "200", "--max-filesize", str(stop-start+1),
            "-H", "Accept-Encoding: identity", "--range", f"{start}-{stop}", "-D", str(headers), url,
        ]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as process, part.open("xb") as output:
            stream = process.stdout
            assert stream is not None
            try:
                raw = stream.read(MEMBER_HEADER.size)
                source.require(len(raw) == MEMBER_HEADER.size, "Incomplete media ZIP header")
                magic, _, flags, method, _, _, _, _, _, name_size, extra_size = MEMBER_HEADER.unpack(raw)
                name = stream.read(name_size)
                extra = stream.read(extra_size)
                prefix_bytes = MEMBER_HEADER.size + name_size + extra_size
                expected_name = f"{source.ROOT}/{role}/{member['name']}".encode()
                source.require(magic == b"PK\x03\x04" and not flags & 1 and method == zipfile.ZIP_STORED
                               and name == expected_name and len(extra) == extra_size and prefix_bytes <= MEMBER_PREFIX_BYTES,
                               "Media ZIP header differs from the roster")
                pending = member["bytes"]
                while pending:
                    data = stream.read(min(1024 * 1024, pending))
                    source.require(bool(data), "Incomplete media ZIP payload")
                    output.write(data)
                    pending -= len(data)
                source.require(len(stream.read()) == MEMBER_PREFIX_BYTES - prefix_bytes, "Unexpected range response length")
                source.require(process.wait(timeout=10) == 0, "Publisher range failed")
            finally:
                if process.poll() is None:
                    process.kill()
        header_lines = headers.read_text().casefold().splitlines()
        source.require(f"content-range: bytes {start}-{stop}/{source.SIZE}" in header_lines, "Publisher served another range")
        record = file_receipt(part, member)
        part.rename(target)
    print(f"Verified {target.name} ({record['bytes']} bytes)", flush=True)
    return record


def acquire(directory):
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    todo = [(role, item) for row in intake() for role, item in row["members"].items()]
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks = [pool.submit(fetch_member, directory, role, item) for role, item in todo]
        for task in tasks:
            task.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("acquire", "audit", "review", "confirmation-plan"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    if args.phase == "acquire":
        acquire(args.directory)
    elif args.phase == "confirmation-plan":
        print(json.dumps(confirmation_plan(args.directory), sort_keys=True, indent=2))
    elif args.phase == "review":
        print(json.dumps(review(json.loads(args.directory.read_bytes())), sort_keys=True))
    else:
        if args.write is None:
            parser.error("audit requires --write")
        result = audit(args.directory)
        review(result)
        with args.write.open("x") as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result["results"], sort_keys=True))


if __name__ == "__main__":
    main()
