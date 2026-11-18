# SPDX-License-Identifier: Apache-2.0
"""Render and measure short square video on a frozen native picture clock."""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
from collections import defaultdict
from contextlib import closing
from fractions import Fraction
from pathlib import Path

from scripts import video_common_sources_v89 as io
from scripts import video_face_portrait_hevc_transfer as transfer
from scripts import video_face_portrait_panel as portrait
from scripts import video_face_portrait_sources as selection
from scripts import video_face_temporal_panel as acquisition
from scripts.acquire_video_common_sources_v91 import hf_object

SIZE = (480, 480)
FRESH = 8
SCHEMA = "chipmunk.square-face-transfer/1"
POINT = "0x1.81f19cb4084c6p-5"
ROLES = ("original", "gain", "face_blur", "picture_jump", "different_shot", *portrait.GENERATORS)
RULE = {"minimum_paired_located_retention": 0.70, "maximum_ordinary_paired": 2,
        "minimum_fresh_located": 4, "minimum_fresh_groups": 3, "maximum_ordinary_fresh": 1,
        "maximum_cut_or_jump": 0, "maximum_misplaced_per_cohort": 1}
CLOCK = {"codec_name": "h264", "codec_tag_string": "avc1", "width": SIZE[0], "height": SIZE[1],
         "pix_fmt": "yuv420p", "start_pts": 0, "duration_ts": 120000, "time_base": "1/30000"}


def reserve(base, panels, output):
    """Reserve unseen known source links, donor keys and content hashes before rendering."""
    common = base / "video-common-sources-v89"
    permissions = common / "permissions"
    metadata = ("rtfs-videos.csv", "rtfs-photos.csv", "rtfs-inswapper.csv", "rtfs-uniface.csv")
    historic = io.read(common / "plan.json")
    receipt = io.read(permissions / "receipt.json")
    if (receipt["rtfs_revision"] != io.RTFS_REV
            or io.sha(permissions / "receipt.json") != historic["metadata_receipt_sha256"]
            or any(io.sha(permissions / name) != receipt["files"][name]["sha256"] for name in metadata)):
        raise ValueError("Publisher metadata differs from the retained provenance check")
    parents = []
    for folder, pin in historic["historical_acquisition_sha256"].items():
        filename = base / folder / "manifest.json"
        if folder == "video-v17-sface/sources":
            filename, pin = panels / "video-v17-sface-source-plan.json", selection.V17
        if io.sha(filename) != pin:
            raise ValueError("A historical source list changed")
        parents.extend(io.read(filename)["items"])
    for group in historic["groups"]:
        if group["source_family"] == "paired_face_swap":
            parents.extend(group | {"donor_filename": donor} for donor in group["donor_filenames"])
    known_hashes, excluded_groups = set(), set()
    for path in sorted(panels.glob("video*.json*")):
        content = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
        excluded_groups.update(match.decode() for match in selection.FINGERPRINT.findall(content))
        if "sources-2026-11-17.json" in path.name and path.name.startswith(
                ("video-face-shot-continuity-", "video-face-portrait-")):
            for group in json.loads(content)["groups"]:
                parents.extend(group | {"donor_filename": donor.removeprefix("rtfs-photo-filename:")
                                        if donor.startswith("rtfs-photo-filename:") else None}
                               for donor in group["donors"])
                known_hashes.update(asset["sha256"] for asset in group["files"].values())
    archive = base / "video-common-acquisition-v91"
    asset_index = archive / "publisher-file-info.json"
    publisher = io.read(asset_index)
    if (io.sha(asset_index) != io.read(archive / "preparation-summary.json")["publisher_file_info_sha256"]
            or publisher["sha"] != io.RTFS_REV):
        raise ValueError("The publisher asset index no longer has its pinned identity")
    siblings = {row["rfilename"]: row for row in publisher["siblings"]}
    candidates = io.paired_targets(io.table(permissions / metadata[0]),
                                   {arm: io.table(permissions / f"rtfs-{arm}.csv") for arm in portrait.GENERATORS},
                                   io.table(permissions / metadata[1]), parents)
    rows = []
    for candidate in candidates:
        target = candidate["target"]
        group = acquisition.digest(acquisition.RULE + "/" + target["id"])
        if group in excluded_groups or not 5.0 <= float(target["end_time"]) - float(target["start_time"]) <= 25.0:
            continue
        paths = {"ordinary": "original_videos/" + target["filename"]} | {
            item["generator"]: item["generator"] + "/" + item["filename"] for item in candidate["swaps"]}
        files = {}
        for arm, path in paths.items():
            digest, size = hf_object(siblings[path])
            files[arm] = {"path": path, "sha256": digest, "size": size}
        if all(0 < item["size"] <= 5 * 1024**2 and item["sha256"] not in known_hashes for item in files.values()):
            rows.append({"group": group, "files": files, "capture_key_sha256": acquisition.digest(io.source_key(target["original_link"])),
                         "donor_key_sha256s": [acquisition.digest(key) for key in candidate["donor_source_keys"]]})
    selected = sorted(rows, key=lambda row: row["group"])[:FRESH]
    total = sum(asset["size"] for group in selected for asset in group["files"].values())
    if (len(selected) != FRESH or total > 50_000_000
            or len({asset["sha256"] for group in selected for asset in group["files"].values()}) != FRESH * 3):
        raise ValueError(f"Only {len(rows)} eligible groups for {total:,} selected publisher bytes within the scratch reserve")
    for index, group in enumerate(selected):
        group.update(index=index, role="held")
    io.write(output, {"recipe": acquisition.RECIPE, "rules": acquisition.RULE, "revision": io.RTFS_REV,
                      "groups": selected, "selected_groups": {"held": FRESH}, "bytes": total,
                      "excluded_group_keys": len(excluded_groups), "publisher_file_info_sha256": io.sha(asset_index),
                      "permission_card_sha256": receipt["files"]["rtfs-README.md"]["sha256"]})
    print(f"Reserved {FRESH} fresh groups and {total:,} publisher bytes", flush=True)


def square_pictures(path, table, indices, center):
    import cv2

    from app.detectors.video.decode import decode_frames

    pictures = {}
    with closing(decode_frames(path, table, sorted(set(indices)), max_side=1280, timeout_s=55)) as frames:
        for frame in frames:
            if not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("A decoded publisher picture has an unverified timestamp")
            height, width = frame.rgb.shape[:2]
            side = min(height, width)
            left = max(0, min(round(width * center - side / 2), width - side))
            top = (height - side) // 2
            crop = frame.rgb[top:top + side, left:left + side]
            method = cv2.INTER_AREA if SIZE[1] < side else cv2.INTER_LINEAR
            pictures[frame.index] = cv2.resize(crop, SIZE, interpolation=method)
    if set(pictures) != set(indices):
        raise ValueError("The native source scan missed a sampled square picture")
    return [pictures[index] for index in indices]


def encode(pictures, output):
    command = ["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-filter_threads", "1",
               "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "480x480", "-framerate", "30",
               "-i", "pipe:0", "-map", "0:v:0", "-an", "-map_metadata", "-1", "-frames:v", "120", "-c:v", "libx264",
               "-preset", "fast", "-threads", "1", "-g", "60", "-pix_fmt", "yuv420p", "-crf", "23",
               "-video_track_timescale", "30000", "-movflags", "+faststart", "-fps_mode", "passthrough", str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        with process.stdin as stream:
            for frame in pictures:
                stream.write(memoryview(frame))
        if process.wait(timeout=60):
            raise ValueError("The square carrier could not be encoded")


def audit_clock(path):
    observed = json.loads(transfer.ff(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v",
        "-show_streams", "-show_frames", "-show_entries", "stream=codec_name,codec_tag_string,width,height,"
        "pix_fmt,start_pts,duration_ts,time_base:frame=best_effort_timestamp:stream_side_data=rotation", "-of", "json", str(path)))
    stream, = observed["streams"]
    ticks = [frame["best_effort_timestamp"] for frame in observed["frames"]]
    if (any(stream.get(field) != value for field, value in CLOCK.items()) or stream.get("side_data_list")
            or ticks != list(range(0, 120000, 1000))):
        raise ValueError("The square carrier differs from its promised picture clock or raster")


def rows_for_group(origin, alternate, locator, position):
    group, paths, table, indices, center = origin
    other, other_paths, other_table, other_indices, other_center = alternate
    reference = square_pictures(paths["ordinary"], table, indices, center)
    other_frames = square_pictures(other_paths["ordinary"], other_table, other_indices, other_center)
    low, high = portrait.edit_interval(position)
    for kind, frames in portrait.earlier.camera_treatments(reference, other_frames, low, high, locator):
        pin = (other if kind == "different_shot" else group)["files"]["ordinary"]["sha256"]
        bounds = [low, high] if frames else []
        yield kind, reference[:low] + frames + reference[high:] if frames else reference, None, bounds, pin, None
    del other_frames
    source_clock = portrait.clock(table, indices)
    for variant, kind in enumerate(portrait.GENERATORS):
        changed_table, changed_indices = portrait.source_table(paths[kind])
        changed_clock = portrait.clock(changed_table, changed_indices)
        if (not portrait.earlier.compatible_clocks(source_clock, changed_clock, table.pts[0] * table.tb,
                                                   changed_table.pts[0] * changed_table.tb)
                or Fraction(changed_table.width, changed_table.height) != Fraction(table.width, table.height)):
            continue
        low, high = portrait.edit_interval(position, variant)
        frames = square_pictures(paths[kind], changed_table, changed_indices[low:high], center)
        yield kind, reference[:low] + frames + reference[high:], [low, high], [low, high], group["files"][kind]["sha256"], changed_clock[low:high]


def measure(records, sources, label, output, locator, recognizer):
    origins = []
    for group in records:
        paths = {kind: sources / f'{group["index"]:02d}-{kind}.mp4' for kind in group["files"]}
        if any(io.sha(path) != group["files"][kind]["sha256"] for kind, path in paths.items()):
            raise ValueError("A source is not the pinned publisher asset")
        table, indices = portrait.source_table(paths["ordinary"])
        center = portrait.crop_center(paths["ordinary"], table, indices, locator)
        origins.append((group, paths, table, indices, center))
    bank = []
    output.mkdir(mode=0o700)
    with (output / "measurements.jsonl").open("x") as stream:
        for position, origin in enumerate(origins):
            group, _, table, indices, center = origin
            alternate = origins[(position + 1) % len(origins)]
            rows = []
            for kind, pictures, truth, treatment, treatment_pin, edited_clock in rows_for_group(origin, alternate, locator, position):
                path = output / f'{group["index"]:02d}-{kind}.mp4'
                encode(pictures, path)
                audit_clock(path)
                reading = portrait.measure_clip(path, locator, recognizer, SIZE)
                transfer.rgb_anchors(path, reading)
                result = {"group": group["group"], "index": group["index"], "file": path.name, "kind": kind,
                          "truth": truth, "treatment": treatment, "treated_sha256": treatment_pin, "input_sha256": io.sha(path),
                          "changed_clock": edited_clock, **reading}
                stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                rows.append(result)
            bank.append({"group": group["group"], "source_sha256": group["files"]["ordinary"]["sha256"],
                         "camera_center": center, "source_clock": portrait.clock(table, indices),
                         "rows": rows, "omitted": sorted(set(ROLES) - {row["kind"] for row in rows})})
            print(f"{label} {position + 1}/{len(records)} source groups, {len(rows)} clips", flush=True)
    return bank


def summarize(held_rows, cohorts):
    point = float.fromhex(POINT)
    counts_by_cohort = {}
    for label, groups in cohorts.items():
        rows = [row for group in groups for row in group["rows"]]
        counts = transfer.counts(rows, point)
        flags = defaultdict(int)
        for row in rows:
            flags[row["kind"]] += transfer.choose(row, point) is not None
        correct_groups = {row["group"] for row in rows if transfer.located(row["truth"], transfer.choose(row, point))}
        counts_by_cohort[label] = counts | {"bands": dict(flags), "located_groups": len(correct_groups),
                                          "omitted": sum(len(group["omitted"]) for group in groups)}
    square = {(row["group"], row["kind"]): row for group in cohorts["paired"] for row in group["rows"]}
    baseline = {(row["group"], row["kind"]): row for row in held_rows}
    if set(square) != set(baseline) or any(row["truth"] != baseline[key]["truth"] for key, row in square.items()):
        raise ValueError("The paired native-square comparison cannot change the frozen groups or edit labels")
    old_correct = {key for key, row in baseline.items() if transfer.located(row["truth"], transfer.choose(row, point))}
    new_correct = {key for key, row in square.items() if transfer.located(row["truth"], transfer.choose(row, point))}
    paired, fresh = counts_by_cohort["paired"], counts_by_cohort["fresh"]
    accepted = (len(old_correct & new_correct) >= RULE["minimum_paired_located_retention"] * len(old_correct)
                and paired["ordinary_flags"] <= RULE["maximum_ordinary_paired"]
                and fresh["located"] >= RULE["minimum_fresh_located"] and fresh["located_groups"] >= RULE["minimum_fresh_groups"]
                and fresh["ordinary_flags"] <= RULE["maximum_ordinary_fresh"]
                and all(group["misplaced"] <= RULE["maximum_misplaced_per_cohort"]
                        and group["bands"]["different_shot"] + group["bands"]["picture_jump"] <= RULE["maximum_cut_or_jump"]
                        for group in (paired, fresh))
                and fresh["omitted"] <= 2)
    return counts_by_cohort | {"portrait": transfer.counts(held_rows, point), "paired_retained": len(old_correct & new_correct),
                              "paired_gained": len(new_correct - old_correct), "paired_lost": len(old_correct - new_correct),
                              "accepted_offline": accepted}


def freeze(fresh, roster, calibration, held, output):
    indexes = {group["group"] for group in io.read(roster)["groups"] if group["role"] == "held"}
    additional = io.read(fresh)["groups"]
    if (len(indexes) != 18 or len(additional) != FRESH or indexes & {group["group"] for group in additional}
            or transfer.read(calibration)["points"]["640"]["strict"] != float.fromhex(POINT)):
        raise ValueError("The case rosters or previous threshold are outside the experiment")
    io.write(output, {"schema": SCHEMA, "fresh_sha256": io.sha(fresh), "roster_sha256": io.sha(roster),
                      "calibration_sha256": io.sha(calibration), "held_sha256": io.sha(held), "point": POINT,
                      "size": list(SIZE), "clock": CLOCK, "rules": RULE, "script_sha256": io.sha(__file__),
                      "sample_start": "0.40 seconds into the pinned capture", "minimum_temporal_iou": 0.50,
                      "edits": "Same contiguous H.264 staged replacements and ordinary edits as the portrait panel.",
                      "scope": "Eighteen earlier matched recordings at a new crop, plus eight reserved recordings. "
                               "The inherited portrait point and adjacent picture check are held fixed."})


def prepare_plan(plan, fresh, roster, calibration, held):
    value = io.read(plan)
    pins = {name + "_sha256": io.sha(path)
            for name, path in (("fresh", fresh), ("roster", roster), ("calibration", calibration), ("held", held))}
    if (value["schema"] != SCHEMA or value["point"] != POINT or value["size"] != list(SIZE)
            or value["clock"] != CLOCK or value["rules"] != RULE or value["script_sha256"] != io.sha(__file__)
            or any(value[field] != pin for field, pin in pins.items())):
        raise ValueError("Experiment pins differ from the recorded before-score plan")
    notes = transfer.read(held)
    from scripts.review_video_face_portrait import verify
    verify(notes, calibration)
    rows = [row for row in notes["banks"]["640"]["rows"] if row["delivery"] == "direct"]
    groups = {"paired": [r for r in io.read(roster)["groups"] if r["role"] == "held"], "fresh": io.read(fresh)["groups"]}
    return rows, groups, notes["banks"]["640"]["header"]["weights"]


def run(plan, fresh, roster, calibration, held, fresh_sources, sources, output):
    import cv2

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video import face_local_detail
    from app.detectors.video.adapters.loaders import load_face_appearance

    prior, groups, pinned_weights = prepare_plan(plan, fresh, roster, calibration, held)
    cv2.setNumThreads(1)
    assets = portrait.weights.assets()
    if assets != pinned_weights:
        raise ValueError("The weights must match the earlier picture comparison")
    locator = load_yunet(portrait.weights.LOCATOR, "cpu", "float32").obj
    recognizer = load_face_appearance(portrait.weights.ENCODER, "cpu", "float32").obj
    output.mkdir(mode=0o700)
    cohorts = {label: measure(groups[label], root, label, output / label, locator, recognizer)
               for label, root in (("paired", sources), ("fresh", fresh_sources))}
    result = {"schema": SCHEMA, "plan_sha256": io.sha(plan), "cohorts": cohorts,
              "face_implementation_sha256": io.sha(face_local_detail.__file__), "summary": summarize(prior, cohorts)}
    transfer.save(output / "measurements.json.gz", result)
    print(json.dumps(result["summary"], sort_keys=True), flush=True)


def review(plan, fresh, roster, calibration, held, measurements):
    prior, expected, _ = prepare_plan(plan, fresh, roster, calibration, held)
    result = transfer.read(measurements)
    cohorts = result["cohorts"]
    if result["schema"] != SCHEMA or result["plan_sha256"] != io.sha(plan) or cohorts.keys() != expected.keys():
        raise ValueError("Measurements do not describe the frozen comparison")
    def digest_check(value):
        return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)

    input_pins = set()
    for label, listing in cohorts.items():
        recorded = [group["group"] for group in listing]
        if recorded != [group["group"] for group in expected[label]]:
            raise ValueError("Measurement groups are missing, duplicated or reordered")
        for position, (group, source) in enumerate(zip(listing, expected[label], strict=True)):
            original_pin = source["files"]["ordinary"]["sha256"]
            other_pin = expected[label][(position + 1) % len(listing)]["files"]["ordinary"]["sha256"]
            source_clock = group["source_clock"]
            if (group["source_sha256"] != original_pin or len(source_clock) != 120
                    or any(a[0] > b[0] or a[1] > b[1] or a[2] != b[2]
                           for a, b in zip(source_clock, source_clock[1:]))):
                raise ValueError("A source clock is missing or changes the frozen acquisition")
            kinds = [row["kind"] for row in group["rows"]]
            if set(kinds) & set(group["omitted"]) or sorted(kinds + group["omitted"]) != sorted(ROLES):
                raise ValueError("Both measured and omitted deliveries must be accounted for")
            for row in group["rows"]:
                if row["first_pts"] != 0 or row["step_pts"] != 1000 or row["time_base"] != "1/30000":
                    raise ValueError("The delivered picture clock differs from its verified carrier")
                pin = row["input_sha256"]
                if not digest_check(pin) or pin in input_pins or row["group"] != group["group"]:
                    raise ValueError("A measured delivery lacks a unique input binding")
                input_pins.add(pin)
                kind = row["kind"]
                expected_pin = source["files"][kind]["sha256"] if kind in portrait.GENERATORS else (
                    other_pin if kind == "different_shot" else original_pin)
                if row["treated_sha256"] != expected_pin:
                    raise ValueError("The treatment is not bound to the expected source")
                if kind in portrait.GENERATORS:
                    expected_range = portrait.edit_interval(position, portrait.GENERATORS.index(kind))
                    if row["truth"] != expected_range or row["treatment"] != expected_range:
                        raise ValueError("An edit no longer begins and ends at its authored frames")
                elif row["truth"] is not None or row["treatment"] != (portrait.edit_interval(position) if kind != "original" else []):
                    raise ValueError("An ordinary control has unexpected treatment positions")
                proposal = row["proposal"]
                if proposal:
                    required = {"before", "during", "after"}
                    if (set(proposal["anchors"]) != required or any(
                            pts != index * 1000 or not 0 <= index < 120 or not digest_check(digest)
                            for index, pts, digest in proposal["anchors"].values())):
                        raise ValueError("A proposed range lacks original frame commitments")
                    if portrait.local.response([float.fromhex(x) for x in proposal["features"]]).hex() != proposal["score"]:
                        raise ValueError("The recorded comparison score cannot be reproduced")
    summary = summarize(prior, cohorts)
    if summary != result["summary"]:
        raise ValueError("The result is not reproducible from the retained measurements")
    print(json.dumps(summary, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    command = commands.add_parser("reserve")
    for name in ("base", "panels", "output"):
        command.add_argument(name, type=Path)
    command = commands.add_parser("freeze")
    for name in ("fresh", "roster", "calibration", "held", "output"):
        command.add_argument(name, type=Path)
    for mode in ("run", "review"):
        command = commands.add_parser(mode)
        for name in ("plan", "fresh", "roster", "calibration", "held"):
            command.add_argument(name, type=Path)
        for name in (("fresh_sources", "sources", "output") if mode == "run" else ("measurements",)):
            command.add_argument(name, type=Path)
    args = vars(parser.parse_args())
    {"reserve": reserve, "freeze": freeze, "run": run, "review": review}[args.pop("mode")](**args)


if __name__ == "__main__":
    main()
