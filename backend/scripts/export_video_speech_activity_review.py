# SPDX-License-Identifier: Apache-2.0
"""Read back the held speech-activity files and export a portable event comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events, measures
from scripts import video_shared_sound_panel as baseline
from scripts import video_speech_activity_gap_panel as panel
from scripts.export_video_covariance_review import SOURCE_SCHEMA, check_counts, prediction
from scripts.video_stereo_covariance_panel import delivery_file, file_hash, read, save

SCHEMA = "chipmunk.video-speech-activity-independent-review/1"
VIDEO_RATE = 30_000
SYSTEMS = ("speech_activity_gap", "prior_stereo", "near_silence")
PRESENTATIONS = {"pcm": ("pcm_s16le", 16_000, 0), "aac48": ("aac", 48_000, 342)}
INPUT_FIELDS = ("group", "snr_db", "carrier", "role", "delivery", "input_sha256", "stereo_pcm_sha256", "samples", "truth")
AUDIO_CLOCK = re.compile(rb"\bpts:(-?\d+)\s.*?\bnb_samples:(\d+)")
REASONS = {
    "speech_activity_gap": "No comparable mouth activity and surrounding speech intervals.",
    "prior_stereo": "No eligible window in the previous stereo scan.",
    "near_silence": "The retained mono comparator omits assessment status when no proposal was made.",
}


def _native_readback(path, row, source):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_frames", "-show_streams",
         "-show_entries", "stream=codec_type,codec_name,profile,sample_rate,channels,time_base,start_pts,duration_ts:"
         "frame=media_type,best_effort_timestamp", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=25,
    )
    native = json.loads(probe.stdout)
    video, audio = native["streams"]
    ticks = [int(frame["best_effort_timestamp"]) for frame in native["frames"] if frame["media_type"] == "video"]
    codec, rate, padding = PRESENTATIONS[row["carrier"]]
    extent = row["samples"]
    if (video["codec_type"] != "video" or video["time_base"] != f"1/{VIDEO_RATE}" or int(video["start_pts"]) != 0
            or int(video["duration_ts"]) != len(ticks) * 1000
            or ticks != source["ticks"] or extent != source["duration_samples"]
            or (audio["codec_type"], audio["codec_name"], audio["sample_rate"], audio["channels"],
                audio["time_base"], int(audio["start_pts"])) != ("audio", codec, str(rate), 2, f"1/{rate}", 0)
            or int(audio["duration_ts"]) * baseline.RATE // rate != extent
            or codec == "aac" and audio["profile"] != "LC"):
        raise ValueError(f"Native picture or soundtrack timeline differs for {path.name}")
    decoded = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-threads", "1", "-filter_threads", "1", "-protocol_whitelist", "file",
         "-copyts", "-i", str(path), "-map", "0:a:0", "-af", f"aresample={baseline.RATE},ashowinfo", "-c:a",
         "pcm_s16le", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True, timeout=25,
    )
    cursor = 0
    for start, count in AUDIO_CLOCK.findall(decoded.stderr):
        if int(start) != cursor or int(count) <= 0:
            raise ValueError(f"Resampled stereo clock is discontinuous for {path.name}")
        cursor += int(count)
    if (len(decoded.stdout) != 4 * cursor or not extent <= cursor <= extent + padding
            or hashlib.sha256(decoded.stdout[:4 * extent]).hexdigest() != row["stereo_pcm_sha256"]):
        raise ValueError(f"Presented stereo samples differ for {path.name}")
    return cursor


def _anchor_clock(reading, source):
    if not (reading and reading.get("picture_frames")):
        return {}
    roles = reading["picture_frames"]
    if set(roles) != {"before", "during", "after"}:
        raise ValueError("A selected range lacks an original picture bracket")
    anchors = {}
    for role, index in roles.items():
        ticks = source["ticks"]
        if type(index) is not int or not 0 <= index < len(ticks):
            raise ValueError("Picture bracket index is outside the source timeline")
        tick = ticks[index]
        anchors[role] = {"frame_index": index, "pts": tick, "sample": round(Fraction(tick * baseline.RATE, VIDEO_RATE))}
    lo, hi = reading["span"]
    if not anchors["before"]["sample"] < lo <= anchors["during"]["sample"] < hi <= anchors["after"]["sample"]:
        raise ValueError("Picture bracket does not surround the reported sample range")
    return anchors


def _choices(row, held):
    selected = held["decisions"][row["carrier"]]["selected"]
    response = row["candidate"][selected["arm"]]
    current = response["reading"]
    yield current if current and current["margin_millionths"] > selected["point"] else None, response["assessed"]
    previous = held["previous"][row["carrier"]]
    old = row["stereo"][previous["arm"]]
    yield old if old and old["margin_millidb"] > previous["point"] else None, row["stereo_assessed"]
    yield {"span": row["mono"]} if row["mono"] else None, row["mono"] is not None


def _correct(target, proposed):
    if not proposed:
        return False
    truth, reported, overlap = measures([target], proposed)
    return 5 * overlap >= 2 * (truth + reported - overlap)


def _paired(manifest, truth, selected, prior):
    labels = {row["id"]: row["regions"] for row in truth["cases"]}
    strata = defaultdict(lambda: defaultdict(set))
    for case, new, old in zip(manifest["cases"], selected["predictions"], prior["predictions"], strict=True):
        if case["id"] != new["id"] or new["id"] != old["id"]:
            raise ValueError("Paired predictions must use the same case order")
        target = labels[case["id"]]
        results = [row["localization"].get("regions", []) for row in (new, old)]
        outcomes = [_correct(target[0], regions) if target else bool(regions) for regions in results]
        coverage = tuple(row["status"] == "ok" for row in (new, old))
        key = case["condition"], case["label"]
        for field, value in (("location_or_alert", tuple(outcomes)), ("coverage", coverage)):
            strata[key][field, value].add((case["id"], case["group"]))
    categories = {(True, True): "both", (True, False): "new_only", (False, True): "prior_only", (False, False): "neither"}
    result = []
    for (condition, label), entries in sorted(strata.items()):
        row = {"condition": condition, "label": label}
        for field in ("location_or_alert", "coverage"):
            counts = row[field] = {}
            for pair, name in categories.items():
                found = entries[field, pair]
                counts[name] = {"cases": len(found), "speaker_groups": len({group for _, group in found})}
        result.append(row)
    return result


def _check_roster(planned, held, reservation, inventory):
    if (planned["schema"] != SOURCE_SCHEMA or planned["split"] != "held" or planned["sources"] != held["sources"]
            or planned["mouth_sha256"] != held["mouth_sha256"]
            or reservation["schema"] != SOURCE_SCHEMA or inventory["schema"] != SOURCE_SCHEMA
            or reservation["evaluation"] != "speech-dropout-activity/1"):
        raise ValueError("Frozen study sources and the reserved holdout disagree")
    speakers = planned["sources"]
    if len(speakers) != 12 or any(len({row[key] for row in speakers}) != 12 for key in ("group", "speaker", "room_sha256", "sha256")):
        raise ValueError("The held panel needs twelve unique speaker, room and media groups")
    for source, inventory_row, reserved in zip(speakers, inventory["entries"], reservation["held"], strict=True):
        if (any(source[key] != inventory_row[key] for key in ("name", "sha256", "speaker", "bytes"))
                or source["speaker"] != reserved["speaker"] or source["name"] not in reserved["options"]
                or inventory_row["room"] != reserved["room"]
                or {"name": source["room"], "sha256": source["room_sha256"]} != reserved["room"]
                or source["ticks"] != list(range(0, len(source["ticks"]) * 1000, 1000))
                or source["duration_samples"] != round(Fraction(len(source["ticks"]) * baseline.RATE, 30))):
            raise ValueError("A held speaker, room take or timestamp table differs from its reservation")


def export(receipt, freeze, inputs, plan, root, output):
    held = panel.verify(receipt, freeze)
    planned, reservation = read(inputs), read(plan)
    inventory_path = root / "sources" / "sources.json.gz"
    inventory = read(inventory_path)
    if (file_hash(inputs) != held["source_receipt_sha256"] or file_hash(plan) != inventory["plan_sha256"]
            or file_hash(inventory_path) != planned["source_sha256"]):
        raise ValueError("The input, source inventory and reservation must match the frozen run")
    _check_roster(planned, held, reservation, inventory)
    parents = {row["group"]: row for row in planned["sources"]}
    expected = {(group, level, carrier, role) for group in parents for level in baseline.SNR_LEVELS
                for carrier in PRESENTATIONS for role in baseline.ROLES}
    cases, truth, files, jobs, anchors = [], [], [], [], []
    predictions = {name: [] for name in SYSTEMS}
    for row, frozen_input in zip(held["rows"], planned["rows"], strict=True):
        treatment = tuple(row[key] for key in ("group", "snr_db", "carrier", "role"))
        if (treatment not in expected or any(row[key] != frozen_input[key] for key in INPUT_FIELDS)
                or row["stereo"] != frozen_input["with_picture"] or row["stereo_assessed"] != frozen_input["audio_eligible"]
                or row["mono"] != frozen_input["mono_baseline"]):
            raise ValueError("A scored treatment differs from its frozen input")
        expected.remove(treatment)
        original = parents[row["group"]]
        path = delivery_file(root / "held", row["delivery"])
        source = delivery_file(root / "sources", original["name"])
        ident = {"id": path.stem, "input_sha256": row["input_sha256"]}
        label = row["truth"]
        cases.append(ident | {
            "split": "confirmation", "modality": "audio", "group": original["speaker"],
            "label": int(label is not None), "source_sha256": original["sha256"], "family": row["role"],
            "condition": f'{row["carrier"]}/{row["snr_db"]}', "lineage": [original["speaker"], original["room_sha256"]],
            "license": {"id": "CREMA-D ODbL 1.0 and DbCL 1.0; DCASE 2013 CC BY 2.0 UK",
                        "evaluation_permitted": True, "redistribution_permitted": False},
            "localization_truth": {"unit": "samples", "source_units": row["samples"],
                                   "target_units": label[1] - label[0] if label else 0},
        })
        truth.append(ident | {"extent": [row["samples"]], "regions": [label] if label else []})
        files.append({"id": path.stem, "input": path.relative_to(root).as_posix(), "source": source.relative_to(root).as_posix()})
        jobs.append((path, row, original))
        anchors.append({})
        for system, (reading, assessed) in zip(SYSTEMS, _choices(row, held), strict=True):
            if reading and not assessed:
                raise ValueError("A selected proposal was marked as unassessed")
            predictions[system].append(prediction(ident, reading["span"] if reading else None, assessed, reason=REASONS[system]))
            if bracket := _anchor_clock(reading, original):
                anchors[-1][system] = bracket
    if expected:
        raise ValueError("Missing a reserved held treatment")
    manifest = {"schema": independent.SCHEMA, "purpose": "Locate staged speech removals in original video soundtracks.",
                "publisher": "https://github.com/CheyneyComputerScience/CREMA-D", "cases": cases}
    binding = {"manifest_sha256": independent.digest(manifest), "split": "confirmation"}
    truth_doc = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    index = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": files}
    verified = verify_inputs(manifest, index, root)
    clock_rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        readbacks = pool.map(lambda job: _native_readback(*job), jobs)
        for (path, row, original), bracket, decoded in zip(jobs, anchors, readbacks, strict=True):
            clock_rows.append({"id": path.stem, "picture_group": original["group"], "decoded_samples": decoded,
                               "samples_presented": row["samples"], "stereo_pcm_sha256": row["stereo_pcm_sha256"],
                               "anchors": bracket})
    result = {"schema": SCHEMA, "held_sha256": file_hash(receipt), "freeze_sha256": file_hash(freeze),
              "inputs_sha256": file_hash(inputs), "reservation_sha256": file_hash(plan), "manifest": manifest,
              "file_index": index, "truth": truth_doc, "inputs_verified": verified,
              "clocks": {"audio_time_base": "1/16000", "picture_time_base": "1/30000", "origin": 0,
                         "picture_pts_by_group": {row["group"]: row["ticks"] for row in planned["sources"]}, "cases": clock_rows},
              "readings": {}, "events": {}}
    # The response identity binds the exact committed measurement replay and operating point.
    for system, summary, rows in zip(SYSTEMS, ("selected", "prior_stereo", "near_silence"), predictions.values(), strict=True):
        weights = [baseline.GEOMETRY_SHA]
        if system == "speech_activity_gap":
            weights += [held["assets"]["vad"], held["assets"]["reducer"]["checkpoints/model_trained_on_dns3.tar"]]
        response = binding | {"system": system, "method_kind": "weighted", "score_kind": "raw",
                              "implementation_sha256": independent.digest([result["held_sha256"], system]),
                              "weight_sha256": weights, "response_units": "A localized interval (1 or 0)",
                              "predictions": rows}
        counts = count_events(manifest, truth_doc, response, minimum_iou=.4, minimum_recall=.4)
        check_counts(counts, held["summary"][summary], coverage=system != "near_silence")
        result["readings"][system], result["events"][system] = response, counts
    result["paired_with_prior_stereo"] = _paired(manifest, truth_doc, *(result["readings"][name] for name in SYSTEMS[:2]))
    save(output, result)
    return verified


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("receipt", "freeze", "inputs", "plan", "root", "output"):
        cli.add_argument(field, type=Path)
    print(json.dumps(export(**{key: path.resolve() for key, path in vars(cli.parse_args()).items()}), sort_keys=True))
