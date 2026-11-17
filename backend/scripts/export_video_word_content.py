# SPDX-License-Identifier: Apache-2.0
"""Rebuild the held native videos and verify the whole-word evidence clocks."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path

from app.bench import video_word_soundtrack as soundtrack
from app.bench.video_word_content import ARMS, CARRIERS, MATCH, ROLES, overlap, read, review
from app.detectors.video.decode import decode_frames, probe_frames
from scripts import export_video_lip_correspondence as export
from scripts import export_video_lip_sequence as delivery


def frames(path, chosen):
    table = probe_frames(path, cap=120, timeout_s=12, max_pixels=1280 * 720)
    start, end = chosen
    indices = [max(0, start - 1), (start + end - 1) // 2, min(len(table) - 1, end)]
    source = list(decode_frames(path, table, indices, timeout_s=12, max_pixels=1280 * 720))
    if [picture.index for picture in source] != indices or not all(
        picture.pts_verified for picture in source
    ):
        raise ValueError("The submitted video did not reproduce the selected picture evidence")
    return [
        {"frame": item.index, "pts": item.pts, "rgb_sha256": sha256(item.rgb.tobytes()).hexdigest()}
        for item in source
    ]


def bound_inputs(plan, captured):
    if sha256(plan.read_bytes()).hexdigest() != captured["source"]:
        raise ValueError("The native replay must use the held source plan")
    cases = {}
    for source in captured["rows"]:
        picture = source["source_sha256"]
        for codec in CARRIERS:
            for role in ROLES:
                row = source["cases"][codec][role]
                cases[picture, codec, role] = {
                    "picture_sha256": picture,
                    "carrier": codec,
                    "role": role,
                    "pts": source["source_clock"]["pts"],
                    "time_base": source["source_clock"]["time_base"],
                    "truth": source["truth"],
                    "source_pcm_sha256": row["sha256"],
                    "carrier_sha256": row["encoded_sha256"],
                    "visual_digit": source["expected_visual"],
                    "audible_digit": source["expected_audible"],
                    **row,
                }
    selected = read(plan)["rows"]
    if {row["target"]["front_sha256"] for row in selected} != {
        row["source_sha256"] for row in captured["rows"]
    }:
        raise ValueError("The native replay roster differs from the measured sources")
    return selected, cases


def run(plan, frozen, measured, sources, output):
    review(measured, frozen)
    captured = read(measured)
    selected, cases = bound_inputs(plan, captured)
    output.mkdir(mode=0o700)
    entries, correct = [], Counter()
    for path, original, times, row in delivery._readbacks(selected, cases, sources, output):
        record = {"file": path.name, "file_sha256": sha256(path.read_bytes()).hexdigest(), "references": {}}
        mouth, speech = row["mouth"], row["speech"]
        for arm in ARMS:
            if mouth is None or mouth[arm] <= captured["points"][arm][row["carrier"]]:
                continue
            found = export.frame_range(speech["span"], times)
            truth = export.frame_range(row["truth"], times) if row["role"] == "different_words" else None
            recognized = (
                speech["digit"] == row["audible_digit"] and mouth["visible_digit"] == row["visual_digit"]
            )
            hit = truth is not None and recognized and overlap(found, truth) >= MATCH
            correct[f"{row['carrier']}/{arm}"] += bool(hit)
            record["references"][arm] = {
                "frames_half_open": found,
                "truth_frames_half_open": truth,
                "correct": hit,
            }
        if record["references"]:
            record["before_during_after"] = frames(path, found)
        entries.append(record)
    result = {
        "schema": "chipmunk.video-word-content-native/1",
        "held_receipt_sha256": sha256(measured.read_bytes()).hexdigest(),
        "verified_videos": len(entries),
        "rechecked_sources": len(selected),
        "correct_on_frame_grid": dict(correct),
        "deliveries": entries,
    }
    raw = (json.dumps(result, sort_keys=True, indent=1) + "\n").encode()
    (output / "native.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    print(json.dumps({key: value for key, value in result.items() if key != "deliveries"}, sort_keys=True))


def run_soundtrack(plan, frozen, measured, sources, output, previous):
    soundtrack.review(measured, previous, frozen)
    captured = read(measured)
    selected, cases = bound_inputs(plan, captured)
    output.mkdir(mode=0o700)
    entries, locations = [], Counter()
    for path, _, times, row in delivery._readbacks(selected, cases, sources, output):
        carrier, role = row["carrier"], row["role"]
        selected_points = {
            "primary": 0.0,
            "independent_scan": captured["points"]["independent_scan"][carrier],
        }
        references, evidence = {}, {}
        for method, point in selected_points.items():
            speech, mouth = soundtrack.selected(row, method, point)
            for arm in ARMS:
                if mouth is None or mouth[arm] <= captured["visual_points"][arm][carrier]:
                    continue
                bounds = export.frame_range(speech["span"], times)
                truth = export.frame_range(row["truth"], times) if role == "different_words" else None
                recognized = (
                    speech["digit"] == row["audible_digit"] and mouth["visible_digit"] == row["visual_digit"]
                )
                hit = truth is not None and recognized and overlap(bounds, truth) >= MATCH
                key = f"{method}/{arm}"
                locations[f"{carrier}/{key}"] += bool(hit)
                if tuple(bounds) not in evidence:
                    evidence[tuple(bounds)] = frames(path, bounds)
                references[key] = {
                    "source_audio_samples": speech["span"],
                    "frames_half_open": bounds,
                    "truth_frames_half_open": truth,
                    "correct": hit,
                    "before_during_after": evidence[tuple(bounds)],
                }
        entries.append(
            {
                "file": path.name,
                "file_sha256": sha256(path.read_bytes()).hexdigest(),
                "source_sha256": row["picture_sha256"],
                "source_pcm_sha256": row["source_pcm_sha256"],
                "references": references,
            }
        )
    result = {
        "schema": "chipmunk.video-word-soundtrack-native/1",
        "held_receipt_sha256": sha256(measured.read_bytes()).hexdigest(),
        "verified_videos": len(entries),
        "rechecked_sources": len(selected),
        "correct_on_frame_grid": dict(locations),
        "deliveries": entries,
    }
    raw = (json.dumps(result, sort_keys=True, indent=1) + "\n").encode()
    (output / "native.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    print(json.dumps({key: value for key, value in result.items() if key != "deliveries"}, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "frozen", "measured", "sources", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--soundtrack-previous", type=Path)
    arguments = vars(parser.parse_args())
    previous = arguments.pop("soundtrack_previous")
    if previous is not None:
        run_soundtrack(**arguments, previous=previous)
    else:
        run(**arguments)
