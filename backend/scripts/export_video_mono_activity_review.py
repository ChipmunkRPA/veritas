# SPDX-License-Identifier: Apache-2.0
"""Verify the exact mono-video holdout and export portable speech-gap scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from itertools import accumulate
from pathlib import Path
from zipfile import ZipFile

from app.bench import independent
from app.bench.independent_inputs import INDEX_SCHEMA, verify_inputs
from app.bench.independent_locations import TRUTH_SCHEMA, count_events
from scripts import video_mono_activity_panel as panel
from scripts import video_shared_sound_panel as stereo
from scripts.export_video_covariance_review import SOURCE_SCHEMA, check_counts, prediction
from scripts.export_video_speech_activity_review import (
    AUDIO_CLOCK,
    PRESENTATIONS,
    VIDEO_RATE,
    _anchor_clock,
    _paired,
)
from scripts.video_stereo_covariance_panel import delivery_file, file_hash, read, save

SCHEMA = "chipmunk.video-mono-activity-independent-review/1"
RESERVATION_SCHEMA = "chipmunk.video-mono-activity-sources/1"
MIX_ROLES = ("unmodified_mix", "whole_gain", "local_duck", "local_eq", "background_reduction", "hard", "faded")
CLEAN_ROLES = ("clean_unmodified", "clean_muted")
SYSTEMS = ("candidate", "near_silence", "preserve_near_silence")
REASONS = {
    "candidate": "No comparable speech activity and original picture interval.",
    "near_silence": "The retained comparator does not record assessment status when no interval was reported.",
    "preserve_near_silence": "No candidate assessment or retained near-silence proposal for this input.",
}


def media(tool, *args):
    return subprocess.run([tool, *args], check=True, capture_output=True, timeout=45)


def streams(path):
    result = media(
        "ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_frames", "-show_streams",
        "-show_entries", "stream=codec_type,codec_name,profile,sample_rate,channels,time_base,start_pts,duration_ts:"
        "frame=media_type,best_effort_timestamp", "-of", "json", str(path),
    )
    native = json.loads(result.stdout)
    ticks = [int(frame["best_effort_timestamp"]) for frame in native["frames"] if frame["media_type"] == "video"]
    return native["streams"], ticks


def picture_hash(path):
    return media("ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file",
                 "-i", str(path), *stereo.STREAM_HASH).stdout.decode().strip()


def rgb_hashes(path):
    result = media(
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1", "-protocol_whitelist", "file",
        "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-pix_fmt", "rgb24",
        "-c:v", "rawvideo", "-threads:v", "1", "-f", "framehash", "-hash", "sha256", "pipe:1",
    )
    return [line.rsplit(",", 1)[1].strip() for line in result.stdout.decode().splitlines() if line and not line.startswith("#")]


def picture_clock(video, ticks, source):
    return (video["codec_type"] == "video" and video["codec_name"] == "h264"
            and video["time_base"] == f"1/{VIDEO_RATE}" and int(video["start_pts"]) == 0
            and int(video["duration_ts"]) == len(ticks) * 1000 and ticks == source["ticks"])


def decode_sound(path):
    filters = f"aformat=sample_fmts=s16:channel_layouts=mono,aresample={stereo.RATE},ashowinfo"
    decoded = media(
        "ffmpeg", "-nostdin", "-v", "info", "-threads", "1", "-filter_threads", "1", "-protocol_whitelist",
        "file", "-copyts", "-i", str(path), "-map", "0:a:0", "-af", filters,
        "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
    )
    clock = [(int(start), int(count)) for start, count in AUDIO_CLOCK.findall(decoded.stderr)]
    return decoded.stdout, clock


def native_readback(job):
    path, row, source = job
    (video, audio), ticks = streams(path)
    codec, rate, padding = PRESENTATIONS[row["carrier"]]
    extent = row["samples"]
    if (not picture_clock(video, ticks, source) or picture_hash(path) != source["picture_stream"]
            or extent != source["duration_samples"]
            or (audio["codec_type"], audio["codec_name"], audio["sample_rate"], audio["channels"],
                audio["time_base"], int(audio["start_pts"])) != ("audio", codec, str(rate), 1, f"1/{rate}", 0)
            or int(audio["duration_ts"]) * stereo.RATE != extent * rate
            or codec == "aac" and audio["profile"] != "LC"):
        raise ValueError(f"Native picture or mono soundtrack differs for {path.name}")
    pcm, clock = decode_sound(path)
    cursor = 0
    for start, count in clock:
        if start != cursor or count <= 0:
            raise ValueError(f"Resampled mono clock is discontinuous for {path.name}")
        cursor += count
    if (len(pcm) != 2 * cursor or not extent <= cursor <= extent + padding
            or hashlib.sha256(pcm[:2 * extent]).hexdigest() != row["mono_pcm_sha256"]):
        raise ValueError(f"Presented mono samples differ for {path.name}")
    return cursor


def intervention(pcm, number):
    """Recompute the intervention from source amplitude alone, before reading detector responses."""
    samples = (sample for (sample,) in struct.iter_unpack("<h", pcm))
    energy = list(accumulate((sample * sample for sample in samples), initial=0))
    width = (3840, 5760, 8320)[number % 3]
    limit = len(energy) - 1 - width
    starts = range(2560, limit - 2560 + 1, 80)
    active = [start for start in starts
              if energy[start] - energy[start - 1920] >= 1920 * 192 ** 2
              and energy[start + width + 1920] - energy[start + width] >= 1920 * 192 ** 2
              and energy[start + width] - energy[start] >= width * 384 ** 2]
    start = min(active or starts, key=lambda index: abs(index - limit // 2))
    return [start, start + width]


def verify_source(root, source, number, room_archive):
    path = delivery_file(root / "sources", source["name"])
    group = source["group"]
    picture = delivery_file(root / "held", f"{group}-pictures.mp4")
    if (path.stat().st_size != source["bytes"] or file_hash(path) != source["sha256"]
            or hashlib.sha256(room_archive.read(source["room"]["name"])).hexdigest() != source["room"]["sha256"]):
        raise ValueError(f"Publisher source or room bytes differ for {group}")
    (video,), ticks = streams(picture)
    original, conformed = rgb_hashes(path), rgb_hashes(picture)
    if (not picture_clock(video, ticks, source) or picture_hash(picture) != source["picture_stream"]
            or len(original) != len(ticks) or original != conformed
            or hashlib.sha256("\n".join(original).encode()).hexdigest() != source["rgb_sha256"]):
        raise ValueError(f"Conformed frames differ from the publisher source for {group}")
    clean = delivery_file(root / "held", f"{group}-{panel.CLEAN:+d}-pcm-clean_unmodified.mov")
    muted = delivery_file(root / "held", f"{group}-{panel.CLEAN:+d}-pcm-clean_muted.mov")
    original_streams, original_ticks = streams(path)
    original_picture, = [row for row in original_streams if row["codec_type"] == "video"]
    original_audio, = [row for row in original_streams if row["codec_type"] == "audio"]
    original_pcm, original_clock = decode_sound(path)
    start = round(original_ticks[0] * Fraction(original_picture["time_base"]) * stereo.RATE)
    trim = start - original_clock[0][0]
    extent = 2 * source["duration_samples"]
    foreground, _ = decode_sound(clean)
    edited, _ = decode_sound(muted)
    if (original_audio["channels"] != 1 or not 0 <= trim <= 1600 or trim != source["source_trim_samples"]
            or len(foreground) != extent or abs(len(original_pcm) - 2 * trim - extent) > 3200
            or foreground != original_pcm[2 * trim:2 * trim + extent].ljust(extent, b"\0")):
        raise ValueError(f"The clean foreground differs from its publisher soundtrack for {group}")
    lo, hi = intervention(foreground, number)
    if edited != foreground[:2 * lo] + bytes(2 * (hi - lo)) + foreground[2 * hi:]:
        raise ValueError(f"The clean intervention differs from its recorded treatment for {group}")
    return source | {"conformed_path": picture.relative_to(root).as_posix(), "conformed_sha256": file_hash(picture),
                     "intervention_samples": [lo, hi]}


def source_receipts(held, reservation, inventory, root, archive):
    sources = held["sources"]
    if (len(sources) != 12
            or any(len({source[key] for source in sources}) != 12 for key in ("speaker", "name", "group", "sha256"))
            or len({source["room"]["sha256"] for source in sources}) != 12):
        raise ValueError("The holdout must use twelve different speaker and room recordings")
    with archive.open("rb") as handle:
        archive_sha1 = hashlib.file_digest(handle, "sha1").hexdigest()
    if archive_sha1 != stereo.ARCHIVE_SHA1 or archive_sha1 != reservation["archive_sha1"]:
        raise ValueError("Room archive differs from the reserved publisher release")
    excluded_speakers = set(reservation["prior_speakers_excluded"])
    excluded_sources = set(reservation["prior_source_hashes_excluded"])
    result = []
    with ZipFile(archive) as rooms:
        rows = zip(sources, inventory["entries"], reservation["held"], strict=True)
        for number, (source, recorded, planned) in enumerate(rows, 1):
            if (any(source[key] != value for key, value in recorded.items())
                    or any(source[key] != planned[key] for key in ("speaker", "room", "earlier_project_speaker"))
                    or source["name"] not in planned["options"] or source["speaker"] in excluded_speakers
                    or source["sha256"] in excluded_sources or source["group"] != f"source-{number:02}"
                    or source["ticks"] != list(range(0, len(source["ticks"]) * 1000, 1000))
                    or source["duration_samples"] != round(Fraction(len(source["ticks"]) * stereo.RATE, 30))
                    or source["room_channel"] != number % 2 or source["ambient_start_s"] != 4 + number % 13):
                raise ValueError("A speaker, room recording or picture clock differs from its reservation")
            result.append(verify_source(root, source, number, rooms))
    return archive_sha1, result


def choices(row, selected):
    arm = row["candidate"][selected["arm"]]
    proposal = arm["reading"]
    candidate = proposal if proposal and proposal["margin_millionths"] > selected["point"] else None
    previous = {"span": row["near_silence"]} if row["near_silence"] else None
    return ((candidate, arm["assessed"]), (previous, previous is not None),
            (previous or candidate, previous is not None or arm["assessed"]))


def export(receipt, freeze, plan, inventory_path, archive, root, output):
    held, settings, reservation, inventory = (read(path) for path in (receipt, freeze, plan, inventory_path))
    hashes = {f"{name}_sha256": file_hash(path) for name, path in
              (("held", receipt), ("freeze", freeze), ("reservation", plan), ("inventory", inventory_path))}
    points = {carrier: value["selected"] for carrier, value in settings["decisions"].items()}
    reserved_freeze, = [value for name, value in reservation["comparison_file_sha256"].items()
                        if name.startswith("video-speech-activity-gap-development-")]
    if (held["schema"] != panel.SCHEMA or inventory["schema"] != SOURCE_SCHEMA or reservation["schema"] != SOURCE_SCHEMA
            or reservation["evaluation"] != RESERVATION_SCHEMA or held["points"] != points
            or points != reservation["operating_points"] or set(points) != set(PRESENTATIONS)
            or any(point["arm"] != "both" for point in points.values())
            or hashes["freeze_sha256"] != held["freeze_sha256"] or hashes["freeze_sha256"] != reserved_freeze
            or hashes["reservation_sha256"] != held["plan_sha256"] or hashes["reservation_sha256"] != inventory["plan_sha256"]
            or hashes["inventory_sha256"] != file_hash(root / "sources" / "sources.json.gz")
            or held["summary"] != panel.summary(held["rows"], points)):
        raise ValueError("Reservation, source inventory, measurements and frozen points must agree")
    archive_sha1, sources = source_receipts(held, reservation, inventory, root, archive)
    parents = {source["group"]: source for source in sources}
    conditions = {level: MIX_ROLES for level in stereo.SNR_LEVELS} | {panel.CLEAN: CLEAN_ROLES}
    expected = {(group, level, carrier, role) for group in parents for level, roles in conditions.items()
                for carrier in points for role in roles}
    cases, truth, files, jobs, clocks = [], [], [], [], []
    predictions = {system: [] for system in SYSTEMS}
    for row in held["rows"]:
        group, level, carrier, role = (row[key] for key in ("group", "snr_db", "carrier", "role"))
        original = parents[group]
        target = original["intervention_samples"] if role in panel.EDIT_ROLES else None
        if ((group, level, carrier, role) not in expected or row["truth"] != target
                or row["delivery"] != f"{group}-{level:+d}-{carrier}-{role}.mov"):
            raise ValueError("A scored file differs from its reserved treatment or truth range")
        expected.remove((group, level, carrier, role))
        path = delivery_file(root / "held", row["delivery"])
        source = delivery_file(root / "sources", original["name"])
        ident = {"id": path.stem, "input_sha256": row["input_sha256"]}
        cases.append(ident | {
            "split": "confirmation", "modality": "audio", "group": original["speaker"],
            "label": int(target is not None), "family": role, "condition": f"{carrier}/{level}",
            "source_sha256": original["sha256"], "lineage": [original["speaker"], original["room"]["sha256"]],
            "license": {"id": "CREMA-D ODbL 1.0 and DbCL 1.0; DCASE 2013 CC BY 2.0 UK",
                        "evaluation_permitted": True, "redistribution_permitted": False},
            "localization_truth": {"unit": "samples", "source_units": row["samples"],
                                   "target_units": target[1] - target[0] if target else 0},
        })
        truth.append(ident | {"extent": [row["samples"]], "regions": [target] if target else []})
        files.append({"id": path.stem, "input": path.relative_to(root).as_posix(), "source": source.relative_to(root).as_posix()})
        jobs.append((path, row, original))
        clock = {"id": path.stem, "picture_group": group, "samples_presented": row["samples"],
                 "mono_pcm_sha256": row["mono_pcm_sha256"], "anchors": {}}
        clocks.append(clock)
        for system, (reading, assessed) in zip(SYSTEMS, choices(row, points[carrier]), strict=True):
            if reading and not assessed:
                raise ValueError("A proposed range was marked unassessed")
            predictions[system].append(prediction(ident, reading["span"] if reading else None, assessed, reason=REASONS[system]))
            if bracket := _anchor_clock(reading, original):
                clock["anchors"][system] = bracket
    if expected:
        raise ValueError("Missing a reserved mono treatment")
    manifest = {"schema": independent.SCHEMA, "purpose": "Locate staged speech removals in a mono video soundtrack.",
                "publisher": "https://github.com/CheyneyComputerScience/CREMA-D", "cases": cases}
    binding = {"manifest_sha256": independent.digest(manifest), "split": "confirmation"}
    truth_doc = binding | {"schema": TRUTH_SCHEMA, "cases": truth}
    index = {"schema": INDEX_SCHEMA, "manifest_sha256": binding["manifest_sha256"], "cases": files}
    verified = verify_inputs(manifest, index, root)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for clock, decoded in zip(clocks, pool.map(native_readback, jobs), strict=True):
            clock["samples_decoded"] = decoded
    result = hashes | {
        "schema": SCHEMA, "manifest": manifest, "truth": truth_doc, "file_index": index, "inputs_verified": verified,
        "sources": sources, "archive_sha1": archive_sha1, "points": points, "readings": {}, "events": {},
        "publisher_receipt": {key: reservation[key] for key in ("publisher_revision", "license_sha256", "source_index_sha256")},
        "clocks": {"picture_time_base": "1/30000", "audio_time_base": "1/16000", "origin": 0, "cases": clocks},
        "event_match": {"minimum_iou": 0.4, "minimum_recall": 0.4},
        "coverage_note": REASONS["near_silence"],
    }
    for system, predicted in predictions.items():
        assets = settings["assets"]
        weights = [stereo.GEOMETRY_SHA]
        if system != "near_silence":
            weights += [assets["vad"], assets["reducer"]["checkpoints/model_trained_on_dns3.tar"]]
        response = binding | {"system": system, "method_kind": "weighted", "score_kind": "raw",
                              "implementation_sha256": independent.digest([hashes["held_sha256"], system]),
                              "weight_sha256": weights, "response_units": "A localized interval (1 or 0)",
                              "predictions": predicted}
        counts = count_events(manifest, truth_doc, response, **result["event_match"])
        check_counts(counts, held["summary"][system], coverage=system != "near_silence")
        result["readings"][system], result["events"][system] = response, counts
    result["paired_with_near_silence"] = _paired(manifest, truth_doc, result["readings"]["preserve_near_silence"],
                                                 result["readings"]["near_silence"])
    save(output, result)
    return verified


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("receipt", "freeze", "plan", "inventory_path", "archive", "root", "output"):
        cli.add_argument(field, type=Path)
    print(json.dumps(export(**{name: path.resolve() for name, path in vars(cli.parse_args()).items()}), sort_keys=True))
