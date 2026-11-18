# SPDX-License-Identifier: Apache-2.0
"""Verify brief face changes on an eight-second native portrait clock."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
from bisect import bisect_left
from collections import Counter
from contextlib import closing
from fractions import Fraction
from pathlib import Path

from app.evidence import face_local_return_binding as local
from scripts import video_common_sources_v89 as io
from scripts import video_face_portrait_hevc_transfer as transfer

SCHEMA = "chipmunk.portrait-eight-second-return/1"
SIZE = (360, 640)
CARRIER = ("h264", "yuv420p", "avc1")
GEOMETRY = local.ScanGeometry(Fraction(1, 30), 240, 5, 5, 26)
POINT = "0x1.81f19cb4084c6p-5"
COUNT = 8
GENERATORS = ("inswapper", "uniface")
ROLES = ("original", "gain", "face_blur", "picture_jump", "different_shot", *GENERATORS)
RULE = {
    "minimum_locations": 7,
    "minimum_in_latter_half": 3,
    "minimum_edit_groups": 4,
    "maximum_ordinary_flags": 2,
    "maximum_cut_or_jump_flags": 0,
    "maximum_misplaced": 1,
}
CLOCK = {
    "codec_name": "h264",
    "codec_tag_string": "avc1",
    "width": 360,
    "height": 640,
    "pix_fmt": "yuv420p",
    "start_pts": 0,
    "duration_ts": 240000,
    "time_base": "1/30000",
}


def reserve(base, panels, output):
    from scripts import video_face_portrait_sources as selection
    from scripts import video_face_temporal_panel as source_rules
    from scripts.acquire_video_common_sources_v91 import hf_object

    common = base / "video-common-sources-v89"
    permission = common / "permissions"
    history = io.read(common / "plan.json")
    receipt = io.read(permission / "receipt.json")
    metadata = tuple(f"rtfs-{part}.csv" for part in ("videos", "photos", "inswapper", "uniface"))
    if (
        receipt["rtfs_revision"] != io.RTFS_REV
        or io.sha(permission / "receipt.json") != history["metadata_receipt_sha256"]
        or any(io.sha(permission / name) != receipt["files"][name]["sha256"] for name in metadata)
    ):
        raise ValueError("Publisher metadata differs from the verified source roster")
    parents = []
    for folder, pin in history["historical_acquisition_sha256"].items():
        path = base / folder / "manifest.json"
        if folder == "video-v17-sface/sources":
            path, pin = panels / "video-v17-sface-source-plan.json", selection.V17
        if io.sha(path) != pin:
            raise ValueError("A prior capture list has changed")
        parents.extend(io.read(path)["items"])
    for item in history["groups"]:
        if item["source_family"] == "paired_face_swap":
            parents.extend(item | {"donor_filename": donor} for donor in item["donor_filenames"])
    originals, photo = [io.table(permission / name) for name in metadata[:2]]
    originals_by_name = {row["filename"]: row for row in originals}
    by_photo_hash = {
        source_rules.digest("rtfs-photo-filename:" + row["filename"]): row["filename"] for row in photo
    }
    excluded, seen, earlier_pins = set(), set(), {}
    for path in sorted(panels.glob("video*.json*")):
        if "eight-second" in path.name:
            continue
        body = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        excluded.update(match.decode() for match in selection.FINGERPRINT.findall(body))
        if path.name not in {
            "video-face-shot-continuity-sources-2026-11-17.json",
            "video-face-portrait-sources-2026-11-17.json",
            "video-face-square-sources-2026-11-18.json",
        }:
            continue
        earlier_pins[path.name] = io.sha(path)
        for row in json.loads(body)["groups"]:
            donors = [
                value.removeprefix("rtfs-photo-filename:")
                for value in row.get("donors", [])
                if value.startswith("rtfs-photo-filename:")
            ]
            donors.extend(
                by_photo_hash[key] for key in row.get("donor_key_sha256s", []) if key in by_photo_hash
            )
            asset_name = Path(row["files"]["ordinary"]["path"]).name
            old = {"group": row["group"], "source_video_url": originals_by_name[asset_name]["original_link"]}
            parents.extend(old | {"donor_filename": donor} for donor in (donors or [None]))
            seen.update(asset["sha256"] for asset in row["files"].values())
    asset_index = base / "video-common-acquisition-v91" / "publisher-file-info.json"
    publisher = io.read(asset_index)
    if (
        io.sha(asset_index)
        != io.read(asset_index.parent / "preparation-summary.json")["publisher_file_info_sha256"]
        or publisher["sha"] != io.RTFS_REV
    ):
        raise ValueError("The original media index changed")
    siblings = {row["rfilename"]: row for row in publisher["siblings"]}
    candidates = io.paired_targets(
        [row for row in originals if float(row["end_time"]) - float(row["start_time"]) >= 8.5],
        {arm: io.table(permission / f"rtfs-{arm}.csv") for arm in GENERATORS},
        photo,
        parents,
    )
    rows = []
    for item in candidates:
        target = item["target"]
        group = source_rules.digest(source_rules.RULE + "/" + target["id"])
        if group in excluded:
            continue
        paths = {"ordinary": "original_videos/" + target["filename"]} | {
            item["generator"]: item["generator"] + "/" + item["filename"] for item in item["swaps"]
        }
        files = {}
        for role, path in paths.items():
            digest, length = hf_object(siblings[path])
            files[role] = {"path": path, "sha256": digest, "size": length}
        if all(
            0 < entry["size"] <= source_rules.LIMIT and entry["sha256"] not in seen
            for entry in files.values()
        ):
            footprint = sum(entry["size"] for entry in files.values())
            rows.append(
                (
                    footprint,
                    {
                        "group": group,
                        "files": files,
                        "capture_key_sha256": source_rules.digest(io.source_key(target["original_link"])),
                        "donor_key_sha256s": [source_rules.digest(key) for key in item["donor_source_keys"]],
                    },
                )
            )
    while len(rows) >= COUNT:
        affordable = sorted(rows, key=lambda pair: (pair[0], pair[1]["group"]))[:COUNT]
        if sum(size for size, _ in affordable) <= 67_000_000:
            break
        rows = []
    if len(rows) < COUNT:
        raise ValueError(
            f"Only {len(rows)} eligible unused eight-second original groups fit the scratch reserve"
        )
    groups = [row | {"role": "held", "index": index} for index, (_, row) in enumerate(affordable)]
    if (
        len({group["capture_key_sha256"] for group in groups}) != COUNT
        or len({pin for group in groups for pin in group["donor_key_sha256s"]}) != COUNT * 2
        or len({entry["sha256"] for group in groups for entry in group["files"].values()}) != COUNT * 3
    ):
        raise ValueError("This panel has repeated captures or known donor keys")
    io.write(
        output,
        {
            "recipe": source_rules.RECIPE,
            "rules": source_rules.RULE,
            "revision": io.RTFS_REV,
            "purpose": SCHEMA,
            "groups": groups,
            "selected_groups": {"held": COUNT},
            "publisher_file_info_sha256": io.sha(asset_index),
            "prior_rosters": earlier_pins,
            "permission_card_sha256": receipt["files"]["rtfs-README.md"]["sha256"],
        },
    )
    print(
        f"Reserved {COUNT} new independent capture groups ({sum(size for size, _ in affordable):,} publisher bytes)",
        flush=True,
    )


def source_table(path):
    from app.detectors.video.decode import probe_frames
    from scripts import video_face_portrait_panel as portrait

    table = probe_frames(path, cap=1600)
    if (
        len(table) < 2
        or table.interpolated
        or table.truncated
        or table.rotation
        or any(a >= b for a, b in zip(table.pts, table.pts[1:]))
    ):
        raise ValueError("The publisher source lacks the required continuous native picture track")
    ticks = [
        table.pts[0] + (portrait.START + GEOMETRY.step * index) / table.tb
        for index in range(GEOMETRY.frame_count)
    ]
    if table.pts[-1] < ticks[-1]:
        raise ValueError("The source ends within the eight-second excerpt")
    indices = []
    for tick in ticks:
        right = bisect_left(table.pts, tick)
        closest = min((right - 1, right), key=lambda i: (abs(table.pts[i] - tick), i))
        if abs(table.pts[closest] - tick) * table.tb > Fraction(1, 24):
            raise ValueError("The excerpt would need an interpolated picture")
        indices.append(closest)
    return table, indices


def inputs(roster, sources):
    from scripts import video_face_portrait_panel as portrait

    bank = io.read(roster)
    if bank["purpose"] != SCHEMA or len(bank["groups"]) != COUNT:
        raise ValueError("Use the frozen independent eight-second source roster")
    for group in bank["groups"]:
        paths = {arm: sources / f"{group['index']:02d}-{arm}.mp4" for arm in group["files"]}
        clock = {}
        for arm, path in paths.items():
            if io.sha(path) != group["files"][arm]["sha256"]:
                raise ValueError("A publisher asset differs from its exact roster")
            clock[arm] = source_table(path)
        original, ticks = clock["ordinary"]
        reference = portrait.clock(original, ticks)
        for arm in GENERATORS:
            derived, match = clock[arm]
            if Fraction(derived.width, derived.height) != Fraction(
                original.width, original.height
            ) or not portrait.earlier.compatible_clocks(
                reference,
                portrait.clock(derived, match),
                original.pts[0] * original.tb,
                derived.pts[0] * derived.tb,
            ):
                raise ValueError("The source and its published swap have different clocks or framing")
        yield group, paths, clock


def freeze(roster, sources, output):
    groups = list(inputs(roster, sources))
    plan = {
        "schema": SCHEMA,
        "source_roster_sha256": io.sha(roster),
        "script_sha256": io.sha(__file__),
        "source_groups": len(groups),
        "clock": CLOCK,
        "frames": 240,
        "geometry": {
            "step": "1/30",
            "sampling_stride_pictures": 2,
            "flank_samples": 5,
            "inside_samples": [5, 26],
        },
        "point": POINT,
        "rule": RULE,
        "criterion": "At least 0.50 temporal intersection-over-union on the native delivered clock",
        "comparison": "The existing four-second portrait method is outside scope on every 240-picture file. A separate app replay compares enabled and disabled production evidence.",
        "scope": "Source-disjoint, visually paired controlled replacements and ordinary controls rendered to 360 by 640 H.264 MP4",
    }
    io.write(output, plan)
    print(
        f"Native-source checks completed for all {len(groups) * 3} originals and paired swaps. Froze the 240-picture comparison.",
        flush=True,
    )


def bounds(position, variant):
    late = (position + variant) % 2
    if position in (2, 7) and not late:
        return [108, 132]
    width = (12, 20, 28, 40)[(position + variant * 2) % 4]
    start = (128 if late else 20) + (position * 14 % 52)
    return [start, start + width]


def encode(pictures, path):
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-protocol_whitelist",
        "file,pipe",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "360x640",
        "-framerate",
        "30",
        "-i",
        "pipe:0",
        "-map",
        "0:v:0",
        "-an",
        "-map_metadata",
        "-1",
        "-frames:v",
        "240",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-threads",
        "1",
        "-g",
        "60",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-video_track_timescale",
        "30000",
        "-movflags",
        "+faststart",
        "-fps_mode",
        "passthrough",
        str(path),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        with process.stdin as stream:
            for picture in pictures:
                stream.write(memoryview(picture))
        if process.wait(timeout=80):
            raise ValueError("The native-clocked portrait encode failed")


def check_clock(path):
    found = json.loads(
        transfer.ff(
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-select_streams",
            "v",
            "-show_streams",
            "-show_frames",
            "-show_entries",
            "stream=codec_name,codec_tag_string,width,height,pix_fmt,start_pts,duration_ts,time_base:"
            "frame=best_effort_timestamp:stream_side_data=rotation",
            "-of",
            "json",
            str(path),
        )
    )
    (stream,) = found["streams"]
    if (
        any(stream.get(key) != value for key, value in CLOCK.items())
        or stream.get("side_data_list")
        or [frame["best_effort_timestamp"] for frame in found["frames"]] != list(range(0, 240000, 1000))
    ):
        raise ValueError("Independent readback does not match the measured portrait carrier and clock")


def measure(path, locator, recognizer):
    from app.detectors.video import face_local_detail
    from app.detectors.video.decode import decode_frames, probe_frames

    check_clock(path)
    table = probe_frames(path, cap=GEOMETRY.frame_count + 1)
    if (
        (table.width, table.height) != SIZE
        or table.rotation
        or table.interpolated
        or table.truncated
        or tuple(table.pts) != tuple(range(0, 240000, 1000))
        or table.time_base != CLOCK["time_base"]
    ):
        raise ValueError("The serving frame table differs from FFprobe's native readback")
    readings, frequency, hashes = {}, {}, {}
    with closing(decode_frames(path, table, GEOMETRY.schedule, timeout_s=45)) as iterator:
        for frame in iterator:
            if not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("The detector received an unverified picture timestamp")
            sample, texture, _, _ = face_local_detail.observe(
                frame.rgb, locator, recognizer, with_texture=True
            )
            readings[frame.index] = sample
            if texture is not None:
                frequency[frame.index] = texture
                hashes[frame.index] = hashlib.sha256(frame.rgb.tobytes()).hexdigest()
    if readings.keys() != set(GEOMETRY.schedule):
        raise ValueError("The detector did not access every scheduled native picture")
    proposals = face_local_detail.proposals(readings, frequency, geometry=GEOMETRY)
    if not proposals:
        return {"face_frames": len(frequency), "proposal": None}
    score, proposal = max(
        ((local.response([round(value, 9) for value in item.features[:2]]), item) for item in proposals),
        key=lambda pair: (pair[0], pair[1].end - pair[1].start, -pair[1].start),
    )
    cuts = face_local_detail.shot_changes(
        path, table, proposal.start, proposal.end, timeout_s=25, geometry=GEOMETRY
    )
    reading = {
        "face_frames": len(frequency),
        "proposal": {
            "range": [proposal.start, proposal.end],
            "score": score.hex(),
            "features": [round(value, 9).hex() for value in proposal.features[:2]],
            "cuts": list(cuts),
            "anchors": {key: [i, table.pts[i], hashes[i]] for key, i in proposal.anchors().items()},
        },
    }
    transfer.rgb_anchors(path, reading)
    return reading


def counts(rows, omitted):
    point = float.fromhex(POINT)
    totals = Counter(
        dict.fromkeys(
            (
                "edit_files",
                "ordinary_files",
                "edit_assessed",
                "ordinary_assessed",
                "located",
                "misplaced",
                "ordinary_flags",
                "cut_or_jump_flags",
                "latter_half",
                "latter_half_located",
                "exact",
                "shot_guard_vetoes",
            ),
            0,
        )
    )
    groups = set()
    for row in rows:
        truth = row["truth"]
        picked = transfer.choose(row, point)
        scope = "edit" if truth else "ordinary"
        totals[f"{scope}_files"] += 1
        totals[f"{scope}_assessed"] += row["proposal"] is not None
        totals["shot_guard_vetoes"] += transfer.choose(row, point, guard=False) is not None and picked is None
        if truth:
            later = truth[0] >= 120
            located = transfer.located(truth, picked)
            totals["latter_half"] += later
            totals["latter_half_located"] += later and located
            totals["located"] += located
            totals["exact"] += picked == truth
            totals["misplaced"] += picked is not None and not located
            if located:
                groups.add(row["group"])
        elif picked:
            totals["ordinary_flags"] += 1
            totals["cut_or_jump_flags"] += row["kind"] in ("different_shot", "picture_jump")
    scoped = (
        not omitted
        and totals["located"] >= RULE["minimum_locations"]
        and totals["latter_half_located"] >= RULE["minimum_in_latter_half"]
        and len(groups) >= RULE["minimum_edit_groups"]
        and totals["ordinary_flags"] <= RULE["maximum_ordinary_flags"]
        and totals["cut_or_jump_flags"] <= RULE["maximum_cut_or_jump_flags"]
        and totals["misplaced"] <= RULE["maximum_misplaced"]
    )
    return dict(totals) | {"edit_groups": len(groups), "omissions": len(omitted), "approved_offline": scoped}


def run(plan, roster, media, output):
    import cv2

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video.adapters.loaders import load_face_appearance
    from scripts import video_face_portrait_panel as portrait

    design = io.read(plan)
    if (
        design["schema"] != SCHEMA
        or design["source_roster_sha256"] != io.sha(roster)
        or design["script_sha256"] != io.sha(__file__)
        or design["clock"] != CLOCK
        or design["rule"] != RULE
        or design["point"] != POINT
    ):
        raise ValueError("The approved before-score design or its executable changed")
    cv2.setNumThreads(1)
    assets = portrait.weights.assets()
    locator = load_yunet(portrait.weights.LOCATOR, "cpu", "float32").obj
    recognizer = load_face_appearance(portrait.weights.ENCODER, "cpu", "float32").obj
    groups = list(inputs(roster, media))
    centers = [
        portrait.crop_center(paths["ordinary"], *clock["ordinary"], locator) for _, paths, clock in groups
    ]
    output.mkdir(mode=0o700)
    rows, origins, missed = [], [], []
    with (output / "journal.jsonl").open("x") as stream:
        for position, (group, paths, clock) in enumerate(groups):
            original, indices = clock["ordinary"]
            crop = centers[position]
            original_pictures = portrait.source_pictures(paths["ordinary"], original, indices, crop, SIZE)
            origins.append(
                {
                    "group": group["group"],
                    "source_sha256": group["files"]["ordinary"]["sha256"],
                    "crop_center": crop,
                    "source_clock": portrait.clock(original, indices),
                }
            )
            alternate_index = (position + 1) % len(groups)
            alternate, alternate_paths, alt_clock = groups[alternate_index]
            alt_table, alt_indices = alt_clock["ordinary"]
            begin, end = bounds(position, 0)
            alt = [None] * len(indices)
            alt[begin:end] = portrait.source_pictures(
                alternate_paths["ordinary"], alt_table, alt_indices[begin:end], centers[alternate_index], SIZE
            )
            cases = []
            for kind, pictures in portrait.earlier.camera_treatments(
                original_pictures, alt, begin, end, locator
            ):
                pin = (alternate if kind == "different_shot" else group)["files"]["ordinary"]["sha256"]
                cases.append((kind, pictures, None, [begin, end] if pictures else [], pin, None))
            if not any(item[0] == "face_blur" for item in cases):
                missed.append(
                    {
                        "group": group["group"],
                        "kind": "face_blur",
                        "reason": "No face could anchor the ordinary blur",
                    }
                )
            for variant, kind in enumerate(GENERATORS):
                table, source_indices = clock[kind]
                start, stop = bounds(position, variant)
                swap = portrait.source_pictures(paths[kind], table, source_indices[start:stop], crop, SIZE)
                cases.append(
                    (
                        kind,
                        swap,
                        [start, stop],
                        [start, stop],
                        group["files"][kind]["sha256"],
                        portrait.clock(table, source_indices[start:stop]),
                    )
                )
            for kind, insertion, truth, edited, pin, source_ticks in cases:
                delivery = original_pictures[:]
                if edited:
                    start, stop = edited
                    if len(insertion) != stop - start:
                        raise ValueError("The treatment would change the measured picture count")
                    delivery[start:stop] = insertion
                path = output / f"{group['index']:02d}-{kind}.mp4"
                encode(delivery, path)
                row = {
                    "file": path.name,
                    "group": group["group"],
                    "source_index": group["index"],
                    "kind": kind,
                    "truth": truth,
                    "treatment": edited,
                    "input_sha256": io.sha(path),
                    "treated_sha256": pin,
                    "inserted_source_clock": source_ticks,
                    **measure(path, locator, recognizer),
                }
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
            print(f"Native eight-second panel: {position + 1}/{len(groups)} groups", flush=True)
    result = {
        "schema": SCHEMA,
        "plan_sha256": io.sha(plan),
        "model_weights": assets,
        "origins": origins,
        "rows": rows,
        "omitted": missed,
        "summary": counts(rows, missed),
    }
    transfer.save(output / "receipt.json.gz", result)
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


def review(plan, roster, result):
    design, sources, measured = [transfer.read(path) for path in (plan, roster, result)]
    if (
        design["schema"] != SCHEMA
        or measured["schema"] != SCHEMA
        or design["clock"] != CLOCK
        or design["point"] != POINT
        or design["rule"] != RULE
        or design["script_sha256"] != io.sha(__file__)
        or measured["plan_sha256"] != io.sha(plan)
        or design["source_roster_sha256"] != io.sha(roster)
    ):
        raise ValueError("The panel receipts diverge from their frozen method and roster")
    groups = {row["group"]: row for row in sources["groups"]}
    origins = {row["group"]: row for row in measured["origins"]}
    if set(origins) != set(groups) or any(
        len(record["source_clock"]) != GEOMETRY.frame_count for record in origins.values()
    ):
        raise ValueError("Incomplete original clock map")
    names = {(row["group"], row["kind"]) for row in measured["rows"]}
    absent = {(row["group"], row["kind"]) for row in measured["omitted"]}
    if (
        names & absent
        or names | absent != {(key, role) for key in groups for role in ROLES}
        or len(names) != len(measured["rows"])
        or len(absent) != len(measured["omitted"])
    ):
        raise ValueError("Some source variants are missing, relabeled or duplicated")
    pins = set()
    for row in measured["rows"]:
        kind = row["kind"]
        source = groups[row["group"]]
        position = source["index"]
        label = bounds(position, GENERATORS.index(kind)) if kind in GENERATORS else None
        treatment = label or (bounds(position, 0) if kind != "original" else [])
        if (
            row["truth"] != label
            or row["treatment"] != treatment
            or row["source_index"] != position
            or row["input_sha256"] in pins
        ):
            raise ValueError(
                "A treatment differs from its authored positions or identical input was repeated"
            )
        pins.add(row["input_sha256"])
        alternate = sources["groups"][(position + 1) % COUNT]
        pin = (
            source["files"][kind]["sha256"]
            if label
            else (alternate if kind == "different_shot" else source)["files"]["ordinary"]["sha256"]
        )
        if (
            row["treated_sha256"] != pin
            or source["files"]["ordinary"]["sha256"] != origins[row["group"]]["source_sha256"]
        ):
            raise ValueError("An inserted range was generated from the wrong source")
        if label and len(row["inserted_source_clock"]) != treatment[1] - treatment[0]:
            raise ValueError("The inserted pictures are missing their native source ticks")
        if proposal := row["proposal"]:
            a, b = proposal["range"]
            flank_a, core, flank_b = GEOMETRY.support(a, b)
            anchors = dict(
                zip(
                    ("before", "during", "after"),
                    (flank_a[-1], core[len(core) // 2], flank_b[0]),
                    strict=True,
                )
            )
            if (
                set(proposal["anchors"]) != set(anchors)
                or any(proposal["anchors"][name][:2] != [i, 1000 * i] for name, i in anchors.items())
                or local.response([float.fromhex(f) for f in proposal["features"]]).hex() != proposal["score"]
                or any(not flank_a[0] < i <= flank_b[-1] for i in proposal["cuts"])
            ):
                raise ValueError("The proposal fails its saved score, native support frames, or shot guard")
    summary = counts(measured["rows"], measured["omitted"])
    if summary != measured["summary"]:
        raise ValueError("Independent count differs from the recorded outcome")
    print(json.dumps(summary, sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    choices = cli.add_subparsers(dest="mode", required=True)
    for mode, parameters in (
        ("reserve", ("base", "panels", "output")),
        ("freeze", ("roster", "sources", "output")),
        ("run", ("plan", "roster", "media", "output")),
        ("review", ("plan", "roster", "result")),
    ):
        command = choices.add_parser(mode)
        for attr in parameters:
            command.add_argument(attr, type=Path)
    arguments = vars(cli.parse_args())
    {"reserve": reserve, "freeze": freeze, "run": run, "review": review}[arguments.pop("mode")](**arguments)


if __name__ == "__main__":
    main()
