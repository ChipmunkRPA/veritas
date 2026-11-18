# SPDX-License-Identifier: Apache-2.0
"""Benchmark brief paired soundtrack differences after a separate picture encode."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from contextlib import closing
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts import video_lip_correspondence as receipt
from scripts.video_visual_speech_evaluation import overlaps

SCHEMA = "chipmunk.video-short-phrase-reencode/1"
ROLES = ("original", "gain", "same_words", "different_words", "global_early", "global_late")
EDITS = frozenset(("same_words", "different_words"))
ENCODES = {"earlier": 23, "moderate": 28, "strong": 34, "short_picture_change": 28, "long_picture_change": 28}
GRIDS = ((80, 60), (160, 90))
INSTANT_CELL_MAX = 13.0
GROUPS = 14
ACCEPTANCE = {"minimum_groups": 14, "minimum_picture_coverage": 0.85, "minimum_audio_coverage": 0.85,
              "minimum_location_fraction_of_exact_copy": 0.80, "minimum_edit_groups": 8,
              "maximum_added_ordinary_groups": 1, "maximum_admitted_picture_edit_controls": 0}
PICTURE_CONTROLS = {
    "short_picture_change": "drawbox=x=270:y=176:w=144:h=104:color=red@0.45:t=fill:enable='between(n,30,31)'",
    "long_picture_change": "drawbox=x=258:y=134:w=168:h=138:color=black@0.4:t=fill:enable='between(n,30,39)'",
}


def read(path):
    return receipt.read(path) if path.suffix == ".gz" else json.loads(path.read_bytes())


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def bindings():
    from app.evidence import (
        video_reference_short_phrases,
        video_reference_sound_difference,
        video_reference_soundtrack,
        video_reference_soundtrack_reencode,
    )
    from scripts import export_video_lip_correspondence, video_lip_articulation_transfer

    return {Path(module.__file__).name: receipt.sha(Path(module.__file__)) for module in (
        video_reference_short_phrases, video_reference_sound_difference, video_reference_soundtrack,
        video_reference_soundtrack_reencode, export_video_lip_correspondence, video_lip_articulation_transfer,
    )} | {Path(__file__).name: receipt.sha(Path(__file__))}


def digit_bounds(phones):
    words, start = [], None
    for phone in phones:
        name = phone["phone"]
        if name.startswith("SIL"):
            if start is not None:
                return None
            continue
        a = round(Fraction(str(phone["offset"])) * 16_000)
        b = round((Fraction(str(phone["offset"])) + Fraction(str(phone["duration"]))) * 16_000)
        boundary = name.rsplit("_", 1)[-1]
        if boundary in {"S", "B"}:
            if start is not None:
                return None
            start = a
        if start is None or b <= a or boundary not in {"S", "B", "I", "E"}:
            return None
        if boundary in {"S", "E"}:
            words.append([start, b])
            start = None
    if len(words) == 6 and start is None and all(a[1] <= b[0] for a, b in zip(words, words[1:])):
        a, b = words[4]
        if 3840 <= b - a <= 8320:
            return (a, b)
    return None


def matched_digit(pool, audio, front, waves, pictures, used_audio, used_pictures):
    from scripts.video_lip_articulation_transfer import source_record

    ordered = sorted(pool, key=lambda uid: sha_bytes(f"{SCHEMA}/digit/{uid}".encode()))
    available = {}

    def member(uid):
        if uid not in available:
            available[uid] = source_record(uid, pool[uid], audio, front, waves, pictures, used_audio, used_pictures)
        return available[uid]

    def sound_only(row):
        return {key: value for key, value in row.items() if key not in {"front", "front_sha256"}}

    def length(uid):
        start, end = pool[uid][0]
        return end - start

    for target in ordered:
        options = [uid for uid in ordered if uid != target and 85 * length(target) <= 100 * length(uid) <= 115 * length(target)]
        same = [uid for uid in options if uid[-2] == target[-2]]
        different = [uid for uid in options if uid[-2] != target[-2]]
        if not same or not different or (origin := member(target)) is None:
            continue
        for choice in same:
            if (keep := member(choice)) is None:
                continue
            for swap in different:
                if abs(length(choice) - length(swap)) <= 960 and (change := member(swap)) is not None:
                    return {"target": origin, "same_words": sound_only(keep), "different_words": sound_only(change)}
    return None


def reserve(sources, panels, output):
    from app.detectors.video.decode import probe_frames
    from app.evidence.video_reference_soundtrack import native_difference_profile
    from scripts.lombard_mouth_source_intake import index_zip, publisher
    from scripts.video_lip_articulation_transfer import keyed

    prior_profile = "video-mouth-patch-sources-2026-11-17.json.gz"
    prior_fresh = "video-mouth-patch-fresh-sources-2026-11-17.json.gz"
    prior_rows = [*read(panels / prior_profile)["rows"], *read(panels / prior_fresh)["rows"]]
    prior_speakers = {row["target"]["id"].split("_")[0] for row in prior_rows
                      if row["split"] not in {"fit", "calibration"}}
    archive_pins = publisher(sources)
    paths = sorted(panels.glob("video-*-sources-*.json.gz"))
    paths += sorted(panels.glob("video-lip-signed-*-plan-*.json.gz"))
    prior_audio, prior_pictures, used_ids, history = set(), set(), set(), {}

    def register(value):
        if isinstance(value, list):
            return sum(register(v) for v in value)
        if not isinstance(value, dict):
            return 0
        matches = 0
        for key, dest in (("audio_sha256", prior_audio), ("front_sha256", prior_pictures),
                          ("video_sha256", prior_pictures), ("picture_sha256", prior_pictures)):
            if isinstance(item := value.get(key), str) and len(item) == 64:
                dest.add(item)
                matches += 1
        if isinstance(item := value.get("id"), str) and item.count("_") == 2:
            used_ids.add(item)
        for key in ("audio_path", "video_path", "front"):
            if isinstance(item := value.get(key), str):
                used_ids.add(Path(item).stem)
        return matches + sum(register(v) for v in value.values())

    for path in paths:
        if register(read(path)):
            history[path.name] = receipt.sha(path)
    available, groups = defaultdict(dict), []
    receiver = Path(output).parent / "scope-check.mov"
    try:
        with (
            ZipFile(sources / "publisher-json.zip") as metadata,
            ZipFile(sources / "alignment.zip") as aligned,
            ZipFile(sources / "lombardgrid_audio.zip") as audio,
            ZipFile(sources / "lombardgrid_front.zip") as front,
        ):
            pictures = index_zip(front, ".mov")
            waves = index_zip(audio, ".wav")
            labels = index_zip(aligned, ".json")
            for path in index_zip(metadata, ".json").values():
                for row in json.loads(metadata.read(path)):
                    who, condition = row["SPKR"], row["COND"]
                    uid = f"{who}_{condition}_{row['UTTERANCE']}"
                    if (who in prior_speakers or condition not in {"p", "l"} or row["STATUS"] != "CORRECT"
                            or uid in used_ids or not all(uid in bank for bank in (pictures, waves, labels))):
                        continue
                    data = aligned.read(labels[uid])
                    if span := digit_bounds(json.loads(data)[uid]):
                        available[who, condition][uid] = span, data

            speakers = sorted({who for who, _ in available}, key=lambda who: sha_bytes(f"{SCHEMA}/{who}".encode()))
            for who in speakers:
                conditions = ("p", "l") if len(groups) % 2 == 0 else ("l", "p")
                for condition in conditions:
                    pool = available.get((who, condition), {})
                    for _ in range(20):
                        match = matched_digit(pool, audio, front, waves, pictures, prior_audio, prior_pictures)
                        if match is None:
                            break
                        target = match["target"]
                        receiver.write_bytes(front.read(target["front"]))
                        table = probe_frames(receiver, cap=100, timeout_s=15, max_pixels=720 * 480)
                        if native_difference_profile((receiver, receiver), (table, table)) is None:
                            pool.pop(target["id"])
                            continue
                        groups.append({"speaker": keyed(who), "condition": condition,
                                       "pictures": len(table), "frame_step": str((table.pts[1] - table.pts[0]) * table.tb),
                                       **match})
                        prior_pictures.add(target["front_sha256"])
                        prior_audio.update(asset["audio_sha256"] for asset in match.values())
                        break
                    if groups and groups[-1]["speaker"] == keyed(who):
                        break
                if len(groups) == GROUPS:
                    break
    finally:
        receiver.unlink(missing_ok=True)

    if len(groups) != GROUPS:
        raise ValueError(f"Only {len(groups)} source and speaker groups outside the earlier paired acoustic comparison")
    plan = {"schema": SCHEMA, "groups": groups, "prior_acoustic_speakers": len(prior_speakers),
            "history": history, "sources": archive_pins, "code": bindings(), "encodes": ENCODES,
            "picture_controls": PICTURE_CONTROLS, "grids": [list(grid) for grid in GRIDS],
            "instant_cell_max": INSTANT_CELL_MAX, "acceptance": ACCEPTANCE}
    receipt.save(output, plan)
    print(f"Frozen {len(groups)} speaker-disjoint groups outside the prior paired acoustic comparison", flush=True)


def encode(source, target, timescale, role):
    argv = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", "1", "-filter_threads", "1",
            "-copyts", "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-fps_mode", "passthrough", "-c:v", "libx264", "-threads", "1",
            "-pix_fmt", "yuv420p", "-crf", str(ENCODES[role]),
            "-preset", "medium" if role == "earlier" else "veryfast",
            "-g", "48" if role == "earlier" else "30", "-video_track_timescale", str(timescale),
            "-map_metadata", "-1", "-map_chapters", "-1"]
    if role in PICTURE_CONTROLS:
        argv.extend(("-vf", PICTURE_CONTROLS[role]))
    subprocess.run([*argv, str(target)], check=True, timeout=20)


def picture_admitted(report):
    return all(grid["within_bounds"] and grid["instant_cell_max"] <= INSTANT_CELL_MAX for grid in report["grids"])


def pictures(earlier, submitted, tables):
    import cv2
    import numpy as np

    from app.detectors.video.decode import decode_frames
    from app.evidence.video_reference_soundtrack_reencode import picture_difference, summarize

    cv2.setNumThreads(1)
    inputs = [(path, table) for path, table in zip((earlier, submitted), tables, strict=True)]
    deltas = [[] for grid in GRIDS]
    anchors = []
    frame_count = len(tables[0])
    with (
        closing(decode_frames(*inputs[0], None, timeout_s=15)) as left,
        closing(decode_frames(*inputs[1], None, timeout_s=15)) as right,
    ):
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            if a.index != i or b.index != i or not a.pts_verified or not b.pts_verified:
                raise ValueError("Could not pair decoded original source timestamps")
            for grid, band in zip(GRIDS, deltas, strict=True):
                band.append(picture_difference(a.rgb, b.rgb, grid=grid))
            anchors.append([a.index, a.pts, b.pts, sha_bytes(a.rgb), sha_bytes(b.rgb)])
    if len(anchors) != frame_count:
        raise ValueError("The picture comparison omitted source frames")
    results = []
    for band in deltas:
        values = np.asarray(band)
        count, height, width = values.shape
        cells = values.reshape(count, height // 5, 5, width // 5, 5).mean(axis=(2, 4))
        results.append(summarize(band) | {"instant_cell_max": round(float(cells.max()), 6)})
    return {"grids": results, "correspondence": sha_bytes(json.dumps(anchors, separators=(",", ":")).encode()),
            "different_frames": sum(a != b for _, _, _, a, b in anchors),
            "time_bases": [table.time_base for table in tables]}


def acoustic(first, samples):
    import numpy as np

    from app.evidence import video_reference_short_phrases as phrases
    from app.evidence import video_reference_sound_difference as difference
    from app.evidence import video_reference_soundtrack as soundtrack

    if len(first) != len(samples) or not difference.SHORT_MIN_SAMPLES <= len(first) <= difference.SAMPLES:
        return {"status": "unassessed_clock"}
    shifted, _ = soundtrack.measure(
        first.astype(np.float32) / 32768, samples.astype(np.float32) / 32768,
        window=soundtrack.SHORT_WINDOW, step=soundtrack.SHORT_STEP, min_rms=soundtrack.QUIET_RMS,
    )
    if shifted:
        return {"status": "unassessed_timing"}
    status, prior = difference.measure(first, samples, recover_supported=True, short_native=True)
    if prior is not None and difference.strong_flanks(prior):
        return {"status": "localized_difference", "basis": "existing", "range": prior["samples"]}
    candidate = phrases.locate(first, samples)
    if candidate is not None and difference.strong_flanks(candidate):
        return {"status": "localized_difference", "basis": phrases.BASIS, "range": candidate["samples"]}
    return {"status": status if prior is None else "weak_flanks"}


def summary(groups):
    output = {}
    for role in ENCODES:
        if role == "earlier":
            continue
        table = defaultdict(Counter)
        admitted = set()
        wrong, flags, good_groups = set(), set(), set()
        exact_hits = hits = assessed = total = changed_picture_flags = 0
        for row in groups:
            picture = row["pictures"][role]
            accept = picture_admitted(picture)
            if accept:
                admitted.add(row["speaker"])
            if role in PICTURE_CONTROLS:
                changed_picture_flags += accept
                continue
            for case in row["audio"]:
                count = table[case["pair"], case["role"]]
                result, truth = case["reading"], case["truth"]
                count["deliveries"] += 1
                count["presented_clock_assessed"] += result["status"] != "unassessed_clock"
                count["picture_pair_admitted"] += accept
                count["combined_assessed"] += result["status"] != "unassessed_clock" and accept
                span = result.get("range")
                exact_hit = bool(span and truth and overlaps(span, truth))
                count["exact_picture_hit"] += exact_hit
                count["reencoded_picture_hit"] += accept and exact_hit
                count["reencoded_audio_alerts"] += accept and bool(span)
                count["misplaced_on_edit"] += bool(accept and span and truth and not exact_hit)
                total += 1
                exact_hits += exact_hit
                hits += accept and exact_hit
                assessed += result["status"] != "unassessed_clock" and accept
                if accept and exact_hit:
                    good_groups.add(row["speaker"])
                if accept and span and not truth:
                    flags.add(row["speaker"])
                if accept and span and truth and not exact_hit:
                    wrong.add(row["speaker"])
        result = {"admitted_picture_groups": len(admitted), "localized_edit_groups": len(good_groups),
                  "ordinary_alert_groups": len(flags), "misplaced_edit_groups": len(wrong),
                  "admitted_picture_edit_controls": changed_picture_flags, "same_picture_hits": exact_hits,
                  "reencoded_picture_hits": hits, "audio_coverage": assessed / total if total else None,
                  "cases": {"/".join(k): dict(c) for k, c in sorted(table.items())}}
        if role not in PICTURE_CONTROLS:
            result["passes_audio_and_coverage"] = (
                len(groups) >= ACCEPTANCE["minimum_groups"]
                and len(admitted) / len(groups) >= ACCEPTANCE["minimum_picture_coverage"]
                and result["audio_coverage"] >= ACCEPTANCE["minimum_audio_coverage"]
                and hits >= ACCEPTANCE["minimum_location_fraction_of_exact_copy"] * exact_hits
                and len(good_groups) >= ACCEPTANCE["minimum_edit_groups"]
                and len(flags) <= ACCEPTANCE["maximum_added_ordinary_groups"]
            )
        output[role] = result
    controls_pass = all(output[role]["admitted_picture_edit_controls"] <= ACCEPTANCE[
        "maximum_admitted_picture_edit_controls"] for role in PICTURE_CONTROLS)
    for role in ENCODES.keys() - (PICTURE_CONTROLS.keys() | {"earlier"}):
        output[role]["advances"] = controls_pass and output[role]["passes_audio_and_coverage"]
    return output


def measure(sources, frozen, output):
    from app.detectors.video.decode import probe_frames
    from app.evidence.video_reference_soundtrack import native_difference_profile
    from scripts.export_video_lip_correspondence import make_soundtracks
    from scripts.video_lip_delivery_transfer import qualified_48k

    plan = read(frozen)
    if plan["schema"] != SCHEMA or plan["code"] != bindings() or plan["acceptance"] != ACCEPTANCE:
        raise ValueError("The frozen pipeline or acceptance rules changed")
    rows = []
    with (
        ZipFile(sources / "lombardgrid_audio.zip") as audio,
        ZipFile(sources / "lombardgrid_front.zip") as front,
        TemporaryDirectory(prefix="chipmunk-short-phrase-encode-", dir=output.parent) as work,
    ):
        base = Path(work)
        source, carrier = base / "source.mov", base / "track.m4a"
        for row in plan["groups"]:
            source.write_bytes(front.read(row["target"]["front"]))
            if receipt.sha(source) != row["target"]["front_sha256"]:
                raise ValueError("Publisher source differs from its frozen digest")
            native = probe_frames(source, cap=100, timeout_s=15)
            delivered = {}
            for enc in ENCODES:
                dest = base / f"{enc}.mov"
                encode(source, dest, native.tb.denominator, enc)
                table = probe_frames(dest, cap=100, timeout_s=15)
                profile = native_difference_profile((source, dest), (native, table))
                if (profile is None or profile["pictures"] != row["pictures"]
                        or profile["frame_step"] != row["frame_step"]):
                    raise ValueError("Encoding changed the supported original native picture clock")
                delivered[enc] = (dest, table)
            first, first_table = delivered["earlier"]
            observations = {}
            for enc, (second, table) in delivered.items():
                if enc == "earlier":
                    continue
                observations[enc] = pictures(first, second, (first_table, table)) | {
                    "earlier_sha256": receipt.sha(first), "submitted_sha256": receipt.sha(second)}
            authored = make_soundtracks(audio, row)
            original = authored["original"]
            earlier_aac, _, earlier_aac_info = qualified_48k(original, carrier)
            comparisons = []
            for role in ROLES:
                current = authored[role]
                submitted_aac, _, submitted_aac_info = qualified_48k(current, carrier)
                for pair, (left, right) in {
                    "pcm_pair": (original, current), "pcm_aac": (original, submitted_aac),
                    "aac_pair": (earlier_aac, submitted_aac),
                }.items():
                    reading = acoustic(left, right)
                    comparisons.append({"role": role, "pair": pair,
                                        "truth": row["target"]["span"] if role in EDITS else None,
                                        "reading": reading, "lengths": [len(left), len(right)],
                                        "earlier_pcm16_sha256": sha_bytes(left.tobytes()),
                                        "submitted_pcm16_sha256": sha_bytes(right.tobytes()),
                                        "earlier_aac_sha256": earlier_aac_info["file_sha256"] if pair == "aac_pair" else None,
                                        "submitted_aac_sha256": submitted_aac_info["file_sha256"] if pair != "pcm_pair" else None})
            rows.append({"speaker": row["speaker"], "source_sha256": row["target"]["front_sha256"],
                         "pictures": observations, "audio": comparisons})
            for path, _ in delivered.values():
                path.unlink()
            print(f"{len(rows)}/{len(plan['groups'])} source groups", flush=True)
    receipt.save(output, {"schema": SCHEMA, "plan_sha256": receipt.sha(frozen), "rows": rows, "results": summary(rows)})


def verify(frozen, result):
    plan, measured = read(frozen), read(result)
    if plan["schema"] != SCHEMA or measured["schema"] != SCHEMA or measured["plan_sha256"] != receipt.sha(frozen):
        raise ValueError("The held output does not use its committed freeze")
    if len({row["speaker"] for row in plan["groups"]}) != GROUPS:
        raise ValueError("The frozen roster does not separate speakers")
    for row, actual in zip(plan["groups"], measured["rows"], strict=True):
        if row["speaker"] != actual["speaker"] or row["target"]["front_sha256"] != actual["source_sha256"]:
            raise ValueError("The held source differs from the frozen roster")
        if any(entry["truth"] != (row["target"]["span"] if entry["role"] in EDITS else None)
               for entry in actual["audio"]):
            raise ValueError("The held edit labels differ from the frozen roster")
    calculated = summary(measured["rows"])
    if calculated != measured["results"]:
        raise ValueError("The held decisions do not recount")
    print(json.dumps(calculated, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    methods = {"reserve": (reserve, ("sources", "panels", "output")),
               "measure": (measure, ("sources", "frozen", "output")),
               "verify": (verify, ("frozen", "result"))}
    for name, (_, arguments) in methods.items():
        cli = stages.add_parser(name)
        for argument in arguments:
            cli.add_argument(argument, type=Path)
    options = vars(parser.parse_args())
    function, _ = methods[options.pop("stage")]
    function(**options)
