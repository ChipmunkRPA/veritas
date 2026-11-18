# SPDX-License-Identifier: Apache-2.0
"""Re-encode a frozen portrait holdout and measure paired localizations."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from app.evidence import face_local_return_binding as local
from scripts.video_stereo_covariance_panel import file_hash, read, save

PLAN_SHA = "b5272eec1b93cffac4510672044818040d9fc98c983124e6460ec0acc9af285f"
SIZE = (360, 640)
RATE = 30000
SCHEMA = "chipmunk.portrait-hevc-transfer/1"
HEVC_NATIVE = {
    "codec_name": "hevc", "codec_tag_string": "hvc1", "profile": "Main", "width": 360, "height": 640,
    "pix_fmt": "yuv420p", "start_pts": 0, "duration_ts": 120000, "time_base": "1/30000",
    "pts": list(range(0, 120000, 1000)),
}
COMMAND = ("-map", "0:v:0", "-an", "-map_metadata", "-1", "-frames:v", "120", "-c:v", "libx265",
           "-preset", "fast", "-profile:v", "main", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-crf", "24",
           "-x265-params", "pools=none:frame-threads=1:log-level=error:keyint=60:min-keyint=30:scenecut=40",
           "-video_track_timescale", "30000", "-movflags", "+faststart", "-brand", "mp42", "-fps_mode", "passthrough")


def ff(tool, *args):
    return subprocess.run([tool, *args], capture_output=True, timeout=60, check=True).stdout


def encode(row, *, root, folder):
    base = root / row["file"]
    if file_hash(base) != row["input_sha256"]:
        raise ValueError("An input does not match the prior held file commitment")
    output = folder / row["file"].replace("-direct.mp4", "-hevc.mp4")
    ff("ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1", "-filter_threads", "1", "-protocol_whitelist",
       "file", "-i", str(base), *COMMAND, str(output))
    return row, output


def clock(path):
    observed = json.loads(ff(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v",
        "-show_streams", "-show_frames", "-show_entries", "stream=codec_name,codec_tag_string,profile,width,height,"
        "pix_fmt,start_pts,duration_ts,time_base:frame=best_effort_timestamp:stream_side_data=rotation",
        "-of", "json", str(path),
    ))
    stream, = observed["streams"]
    ticks = [frame["best_effort_timestamp"] for frame in observed["frames"]]
    if any((stream | {"pts": ticks}).get(field) != value for field, value in HEVC_NATIVE.items()) or stream.get("side_data_list"):
        raise ValueError(f"The encoded file differs from the native main profile or exact picture clock: {path.name}")
    return HEVC_NATIVE


def rgb_anchors(path, reading):
    if not reading["proposal"]:
        return
    anchors = reading["proposal"]["anchors"]
    select = "+".join(f"eq(n,{anchor[0]})" for anchor in anchors.values())
    output = ff("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", "-protocol_whitelist",
                "file", "-i", str(path), "-vf", f"select='{select}',format=rgb24", "-fps_mode", "passthrough",
                "-c:v", "rawvideo", "-threads:v", "1", "-f", "framehash", "-hash", "sha256", "pipe:1")
    hashes = [line.rsplit(",", 1)[1].strip() for line in output.decode().splitlines() if line and not line.startswith("#")]
    if hashes != [item[2] for item in sorted(anchors.values())]:
        raise ValueError(f"The range's independently decoded pictures differ from its frame anchors: {path.name}")


def choose(row, point, *, guard=True):
    proposal = row["proposal"]
    return (proposal["range"] if proposal and float.fromhex(proposal["score"]) > point
            and (not guard or not proposal["cuts"]) else None)


def located(truth, proposal):
    if not (truth and proposal):
        return False
    intersection = min(truth[1], proposal[1]) - max(truth[0], proposal[0])
    union = max(truth[1], proposal[1]) - min(truth[0], proposal[0])
    return 2 * intersection >= union


def counts(rows, point):
    result = Counter(dict.fromkeys(("edit_files", "ordinary_files", "edit_assessed", "ordinary_assessed",
                                    "located", "misplaced", "ordinary_flags", "guard_vetoes"), 0))
    groups = set()
    for row in rows:
        target = row["truth"]
        proposal = choose(row, point)
        scope = "edit" if target else "ordinary"
        result[f"{scope}_files"] += 1
        result[f"{scope}_assessed"] += row["proposal"] is not None
        if target:
            correct = located(target, proposal)
            result["located"] += correct
            result["misplaced"] += proposal is not None and not correct
        elif proposal is not None:
            result["ordinary_flags"] += 1
            groups.add(row["group"])
        result["guard_vetoes"] += choose(row, point, guard=False) is not None and proposal is None
    return dict(result) | {"ordinary_groups": len(groups)}


def summarize(rows, rule, point):
    baseline = [pair["h264"] for pair in rows]
    transported = [pair["hevc"] for pair in rows]
    h264, hevc = [counts(listing, point) for listing in (baseline, transported)]
    categories = dict(retained=0, lost=0, gained=0, new_ordinary=0, new_cut_or_jump=0, new_misplaced=0)
    for old, new in zip(baseline, transported, strict=True):
        truth = old["truth"]
        before, after = choose(old, point), choose(new, point)
        if truth:
            located_before, located_after = (located(truth, proposal) for proposal in (before, after))
            categories["retained"] += located_before and located_after
            categories["lost"] += located_before and not located_after
            categories["gained"] += located_after and not located_before
            categories["new_misplaced"] += after is not None and not located_after and (located_before or before is None)
        elif after is not None and before is None:
            categories["new_ordinary"] += 1
            categories["new_cut_or_jump"] += old["kind"] in {"different_shot", "picture_jump"}
    accepted = (
        categories["retained"] >= rule["minimum_paired_located_retention"] * h264["located"]
        and hevc["edit_assessed"] >= rule["minimum_edit_assessment_retention"] * h264["edit_assessed"]
        and hevc["ordinary_assessed"] >= rule["minimum_ordinary_assessment_retention"] * h264["ordinary_assessed"]
        and categories["new_ordinary"] <= rule["maximum_new_ordinary_file_flags"]
        and categories["new_cut_or_jump"] <= rule["maximum_new_cut_or_jump_flags"]
        and categories["new_misplaced"] <= rule["maximum_new_misplaced_edit_ranges"]
    )
    return {"h264": h264, "hevc": hevc, "paired": categories, "accepted_offline": accepted}


def run(plan, held, calibration, roster, media, output):
    import cv2

    from app.detectors.image.adapters.loaders import load_yunet
    from app.detectors.video.adapters.loaders import load_face_appearance
    from app.detectors.video.face_return import source_carrier
    from scripts import review_video_face_portrait as study
    from scripts import video_face_portrait_panel as panel

    freeze = read(plan)
    if (file_hash(plan) != PLAN_SHA or not held_hashes(freeze, held, calibration, roster)
            or freeze["input_manifest_sha256"] != file_hash(media / "manifest.json")):
        raise ValueError("The previously evaluated inputs differ from the before-score codec plan")
    document = read(held)
    study.verify(document, calibration)
    bank = document["banks"]["640"]
    rows = selected_rows(bank, freeze)
    cv2.setNumThreads(1)
    assets = panel.weights.assets()
    if assets != bank["header"]["weights"]:
        raise ValueError("The signed face assets differ from the prior held run")
    locator = load_yunet(panel.weights.LOCATOR, "cpu", "float32").obj
    recognizer = load_face_appearance(panel.weights.ENCODER, "cpu", "float32").obj
    output.mkdir(mode=0o700)
    result = []
    with ThreadPoolExecutor(max_workers=2) as render, (output / "journal.jsonl").open("x") as stream:
        for index, (base, path) in enumerate(render.map(partial(encode, root=media, folder=output), rows), 1):
            if source_carrier(path) != "mp4":
                raise ValueError("The encoded H.265 file is outside the current MP4 carrier")
            timeline = clock(path)
            start = time.monotonic()
            check = panel.measure_clip(path, locator, recognizer, SIZE)
            elapsed = round(time.monotonic() - start, 4)
            rgb_anchors(path, check)
            pair = {"h264": base, "hevc": {key: base[key] for key in ("group", "kind", "truth")} | check,
                    "hevc_file": path.name, "hevc_sha256": file_hash(path), "native": timeline, "observation_seconds": elapsed}
            result.append(pair)
            stream.write(json.dumps(pair, sort_keys=True) + "\n")
            stream.flush()
            if index % 14 == 0 or index == len(rows):
                print(f"Verified exact clocks and measured {index}/{len(rows)} clips", flush=True)
    table = summarize(result, freeze["comparative_gate"], float.fromhex(freeze["point"]))
    document = {"schema": SCHEMA, "freeze_sha256": PLAN_SHA, "source": freeze, "rows": result, "summary": table,
                "video_encoder_args": list(COMMAND), "assets": assets, "script_sha256": file_hash(Path(__file__)),
                "ffmpeg": ff("ffmpeg", "-version").decode().splitlines()[0]}
    save(output / "result.json.gz", document)
    print(json.dumps(table, sort_keys=True), flush=True)


def selected_rows(bank, freeze):
    rows = [row for row in bank["rows"] if row["delivery"] == "direct"]
    edits = sum(bool(row["truth"]) for row in rows)
    if (len({row["group"] for row in rows}) != freeze["planned_originals"] or edits != freeze["planned_edits"]
            or len(rows) - edits != freeze["planned_ordinary"]):
        raise ValueError("The complete held direct profile is required")
    return rows


def valid_digest(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def inspect(pair, baseline):
    measured, native = pair["hevc"], pair["native"]
    if (pair["h264"] != baseline or native != HEVC_NATIVE
            or pair["hevc_file"] != baseline["file"].replace("-direct.mp4", "-hevc.mp4")
            or not valid_digest(pair["hevc_sha256"])
            or any(measured[key] != baseline[key] for key in ("group", "kind", "truth"))
            or (measured["first_pts"], measured["step_pts"], measured["time_base"]) != (0, 1000, native["time_base"])
            or not 0 <= measured["face_frames"] <= len(local.PORTRAIT.schedule)):
        raise ValueError("A response is outside its original held case or native HEVC profile")
    if matched := measured["proposal"]:
        before, during, after = local.PORTRAIT.support(*matched["range"])
        anchors = {"before": before[-1], "during": during[len(during) // 2], "after": after[0]}
        if (matched["score"] != local.response([float.fromhex(value) for value in matched["features"]]).hex()
                or sorted(set(matched["cuts"])) != matched["cuts"]
                or any(not before[0] < cut <= after[-1] for cut in matched["cuts"])
                or set(matched["anchors"]) != set(anchors)):
            raise ValueError("The selected location differs from its retained measurements")
        for role, index in anchors.items():
            frame, tick, rgb_hash = matched["anchors"][role]
            if (frame, tick) != (index, native["pts"][index]) or not valid_digest(rgb_hash):
                raise ValueError("An exhibit differs from the submitted HEVC picture clock")


def review(receipt, plan, held, calibration, roster):
    freeze = read(plan)
    document = read(receipt)
    if (file_hash(plan) != PLAN_SHA or not held_hashes(freeze, held, calibration, roster)
            or document["freeze_sha256"] != PLAN_SHA or document["schema"] != SCHEMA
            or document["source"] != freeze or document["video_encoder_args"] != list(COMMAND)):
        raise ValueError("A benchmark input differs from the before-score freeze")
    bank = read(held)["banks"]["640"]
    if document["assets"] != bank["header"]["weights"]:
        raise ValueError("The held measurement assets differ")
    for pair, expected in zip(document["rows"], selected_rows(bank, freeze), strict=True):
        inspect(pair, expected)
    summary = summarize(document["rows"], freeze["comparative_gate"], float.fromhex(freeze["point"]))
    if document["summary"] != summary:
        raise ValueError("Retained case responses do not reproduce the recorded outcome")
    return summary


def held_hashes(freeze, held, calibration, roster):
    return all(freeze[f"{field}_sha256"] == file_hash(path)
               for field, path in (("held", held), ("calibration", calibration), ("roster", roster)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="mode", required=True)
    for method, fields in ((run, ("media", "output")), (review, ("receipt",))):
        command = commands.add_parser(method.__name__)
        for field in ("plan", "held", "calibration", "roster", *fields):
            command.add_argument(field, type=Path)
        command.set_defaults(action=method)
    args = vars(cli.parse_args())
    action = args.pop("action")
    args.pop("mode")
    if result := action(**args):
        print(json.dumps(result, sort_keys=True))
