# SPDX-License-Identifier: Apache-2.0
"""Replay each retained 30 fps dropout and ordinary control in 44.1 kHz AAC."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import closing
from pathlib import Path

from scripts.video_lip_multishot_panel import RATE, digest, ffmpeg, overlap, read, save
from scripts.video_silent_mouth_clock_panel import GEOMETRY_SHA, summarize
from scripts.video_silent_mouth_clock_panel import SCHEMA as REFERENCE

SCHEMA = "chipmunk.video-silent-mouth-aac441/1"
CARRIER = "aac441"
RATE_NATIVE = 44_100
STREAM_HASH = ("-map", "0:v:0", "-c:v", "copy", "-f", "streamhash", "pipe:1")


def case_key(row):
    return row["group"], row["role"]


def assess(result, reference):
    prior = {case_key(row): row for row in reference["rows"] if row["carrier"] == "aac48"}
    parents = {case_key(row): row["input_sha256"] for row in reference["rows"] if row["carrier"] == "pcm"}
    observed = {case_key(row): row for row in result["rows"]}
    if result["schema"] != SCHEMA or set(prior) != set(observed) or prior.keys() != parents.keys() or len(observed) != len(result["rows"]):
        raise ValueError("The matched evaluation has missing or repeated rows")
    paired = Counter()
    for key, row in observed.items():
        baseline = prior[key]
        truth, matches = row["truth"], row["matches"]
        extent = row["native_ticks"] * RATE // RATE_NATIVE
        if (
            row["carrier"] != CARRIER or row["clock"] != baseline["clock"] or truth != baseline["truth"]
            or row["parent_sha256"] != parents[key]
            or [shot["frames"] for shot in row["coverage"]] != [shot["frames"] for shot in baseline["coverage"]]
            or row["coverage"][-1]["audio_samples"][1] != extent
            or abs(extent - baseline["coverage"][-1]["audio_samples"][1]) > 1
            or any(not 0 <= start < stop <= extent for start, stop in matches)
            or row["assessed"] != (row["status"] in {"matched", "no_match"})
            or bool(matches) != (row["status"] == "matched")
        ):
            raise ValueError("The measured delivery does not match the reference input or clock")
        if truth:
            before = any(overlap(truth, span) >= .4 for span in baseline["matches"])
            after = any(overlap(truth, span) >= .4 for span in matches)
            paired["retained" if before and after else "gained" if after else "lost" if before else "both_missed"] += 1
        else:
            paired["ordinary"] += 1
            paired["new_ordinary_flags"] += bool(matches) and not baseline["matches"]
    return {"summary": summarize(result["rows"]), "paired": dict(paired)}


def measure(panel, reference_file, assets, output):
    from app.detectors.audio.decode import decode_audio
    from app.detectors.video.decode import decode_stream, probe_frames
    from app.detectors.video.shots import detect_shots
    from app.detectors.video.viseme_review import sound_levels, supported, supported_shots
    from app.detectors.video.viseme_worker import capture_mouth_frames
    from app.evidence import video_lip_agreement as agreement
    from app.evidence import video_silent_mouth as silent
    from app.evidence import video_viseme_binding as binding

    reference = read(reference_file)
    if reference["schema"] != REFERENCE or digest(assets / "face_landmarker.task") != GEOMETRY_SHA:
        raise ValueError("The original comparison and its pinned face geometry are required")
    sys.path.insert(0, str(assets / "runtime"))
    originals = {case_key(row): row for row in reference["rows"] if row["carrier"] == "pcm"}
    output.mkdir(mode=0o700)
    rows = []
    for index, clock in enumerate(reference["clocks"], 1):
        group = clock["group"]
        treatments = [row for key, row in originals.items() if key[0] == group]
        baseline = next(row for row in treatments if row["role"] == "original")
        original = panel / baseline["delivery"]
        if digest(original) != baseline["input_sha256"]:
            raise ValueError("The retained source recording differs from the published input")
        table = probe_frames(original)
        with closing(decode_stream(original, table)) as stream:
            shot_scan = detect_shots(stream, table).as_json()
        shots = shot_scan["ranges"]
        if shot_scan["frames_scanned"] != len(table) or shots != [shot["frames"] for shot in baseline["coverage"]]:
            raise ValueError("The source picture scan differs from the recorded shots")
        frames, _, _ = capture_mouth_frames(original, table, assets / "face_landmarker.task", motion=False, shots=shots)
        picture_hash = ffmpeg("-i", str(original), *STREAM_HASH)
        for item in treatments:
            parent = panel / item["delivery"]
            if digest(parent) != item["input_sha256"] or ffmpeg("-i", str(parent), *STREAM_HASH) != picture_hash:
                raise ValueError("A treatment differs from the original picture stream or committed input")
            destination = output / f"{group}-{CARRIER}-{item['role']}.mov"
            ffmpeg("-i", str(parent), "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy", "-c:a", "aac",
                   "-ar", str(RATE_NATIVE), "-b:a", "112k", "-video_track_timescale", "30000",
                   "-movie_timescale", "35280000", str(destination))
            delivered = probe_frames(destination)
            if ((table.pts, table.time_base) != (delivered.pts, delivered.time_base)
                    or ffmpeg("-i", str(destination), *STREAM_HASH) != picture_hash):
                raise ValueError("The AAC-only conversion changed a source picture or its presentation timestamp")
            input_hash = digest(destination)
            audio = decode_audio(destination, source_sha256=input_hash)
            mapping = audio.mapping
            samples = binding.presented_samples(mapping, max_samples=agreement.MAX_MULTISHOT_SAMPLES)
            if not supported_shots(delivered, mapping, shots) or supported(delivered, mapping, shots):
                raise ValueError(f"{destination.name} did not qualify for the amplitude-only review")
            record = binding.measured(
                {"frames": frames, "phones": [], "sound_levels": sound_levels(audio.pcm, samples), silent.SHOT_KEY: shots},
                timestamps=delivered.pts, time_base=delivered.time_base,
                source={"width": delivered.width, "height": delivered.height}, mapping=mapping,
            )
            rows.append({"group": group, "clock": item["clock"], "role": item["role"], "carrier": CARRIER,
                         "delivery": destination.name, "input_sha256": input_hash, "parent_sha256": item["input_sha256"],
                         "decoded_pcm_sha256": mapping["pcm_sha256"], "native_ticks": mapping["source_duration_pts"],
                         "truth": item["truth"], "coverage": record["shot_coverage"], "status": record["status"],
                         "assessed": record["status"] in {"matched", "no_match"},
                         "matches": [finding["witness"]["audio_source_pts"] for _, finding in binding.exhibits(record)]})
        if index % 8 == 0 or index == len(reference["clocks"]):
            print(f"Verified {index} of {len(reference['clocks'])} source picture presentations", flush=True)
    committable = {"schema": SCHEMA, "reference_sha256": digest(reference_file), "geometry_sha256": GEOMETRY_SHA, "rows": rows}
    outcomes = assess(committable, reference)
    save(output / "result.json.gz", committable | outcomes)
    print(json.dumps(outcomes, sort_keys=True), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    for action, names in (("measure", ("panel", "reference_file", "assets", "output")),
                          ("review", ("result", "reference_file"))):
        sub = commands.add_parser(action)
        for name in names:
            sub.add_argument(name, type=Path)
    args = cli.parse_args()
    if args.action == "measure":
        measure(args.panel, args.reference_file, args.assets, args.output)
    else:
        result = read(args.result)
        outcomes = assess(result, read(args.reference_file))
        if result["reference_sha256"] != digest(args.reference_file) or any(result[key] != out for key, out in outcomes.items()):
            raise ValueError("The paired result does not reproduce the published measurements")
        print(json.dumps(outcomes, sort_keys=True))


if __name__ == "__main__":
    main()
