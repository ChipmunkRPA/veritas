# SPDX-License-Identifier: MIT
"""Transfer the frozen dense lip-timing comparison to native fractional-clock speech."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from bisect import bisect_left
from fractions import Fraction
from pathlib import Path
from statistics import median

from app.evidence import video_viseme_binding as binding
from app.evidence import video_viseme_timing as timing
from scripts import video_lip_dense_posteriors as dense
from scripts import video_lip_native_clock_transfer as native

SCHEMA = "chipmunk.video-lip-dense-native/1"
study = dense.prior


def pixels(source, qualified, geometry, *, measure_width=False):
    from app.detectors.video.decode import FrameTable, decode_stream, probe_frames
    from app.evidence.video_reference_timing import _landmarker, _mesh

    if study.phone.sha(source) != qualified["input_sha256"]:
        raise ValueError("Publisher video bytes differ from the qualified source")
    table = probe_frames(source, cap=121, timeout_s=20)
    if table != FrameTable.from_json(qualified["frame_table"]):
        raise ValueError("Publisher picture timestamps changed")
    frames = []
    with _landmarker(geometry) as (mp, task):
        for frame in decode_stream(source, table, max_side=None, timeout_s=20):
            if frame.index != len(frames) or not frame.pts_verified or frame.pts != table.pts[frame.index]:
                raise ValueError("A decoded picture lacks its verified timestamp")
            record = {"index": frame.index, "pts": frame.pts, "gap_units": None}
            if observed := _mesh(task, mp, frame.rgb):
                features, box = observed
                x, y, right, bottom = box
                record.update(gap_units=round(float(median(features[:3])) * timing.SCORE_UNITS), mouth_box=box,
                              rgb_sha256=study.phone.digest(frame.rgb.tobytes()),
                              mouth_crop_sha256=study.phone.digest(frame.rgb[y:bottom, x:right].tobytes()))
                if measure_width:
                    record["width_units"] = round(float(features[3]) * timing.SCORE_UNITS)
            frames.append(record)
    if len(frames) != len(table):
        raise ValueError("Incomplete publisher picture scan")
    return frames, table


def audio(source, qualified):
    import numpy as np

    decoded = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", "1",
                              "-protocol_whitelist", "file", "-i", f"file:{source.resolve()}",
                              "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "1", "-ar", str(timing.RATE),
                              "-f", "s16le", "pipe:1"], capture_output=True, check=True, timeout=20)
    declared = qualified["canonical_audio"]
    original = np.frombuffer(decoded.stdout, dtype="<i2")
    if (decoded.stderr.strip() or len(original) != declared["samples"]
            or study.phone.digest(decoded.stdout) != declared["pcm_sha256"]):
        raise ValueError("Publisher audio did not decode to the qualified samples")
    return original


def capture(args, primary):
    sys.path.insert(0, str(args.runtime.resolve()))
    if study.phone.sha(args.geometry) != study.parent.FACE_SHA256:
        raise ValueError("Face geometry weights differ from the pinned asset")
    read, device = dense.reader(args.checkpoint)
    qualified = native.roster(args.audit, remaining=args.remaining_actors)
    rows = []
    for item in qualified:
        index = item["index"]
        source = args.media / f"asset-{index:03d}.flv"
        frames, table = pixels(source, item, args.geometry)
        original = audio(source, item)
        origin = round(Fraction(item["audio_clocks"]["frames"][0]["start"]) * timing.RATE)
        edit = [(len(original) - native.EDIT_WIDTH) // 2, (len(original) + native.EDIT_WIDTH) // 2]
        cases = {}
        for role, pcm in study.parent.treatments(original, edit).items():
            groups, phones = read(pcm, include_events=True)
            cases[role] = {"pcm_sha256": study.phone.digest(pcm.tobytes()),
                           "prediction": dense.predict(groups, primary["fit"]["weights"]), "phones": phones}
        rows.append({"source": index, "sha256": item["input_sha256"], "samples": len(original), "edit": edit,
                     "audio_origin": origin, "width": table.width, "height": table.height,
                     "time_base": table.time_base, "frames": frames, "cases": cases})
        print(f"Replayed native pictures and speech for {len(rows)}/{len(qualified)} actors.", flush=True)
    report = {"schema": SCHEMA, "primary_sha256": study.phone.sha(args.primary), "model": dense.MODEL,
              "device": device, "rows": rows, "result": evaluate(qualified, rows, primary)}
    if args.remaining_actors:
        report["remaining_actors"] = True
    return report


def nearest_25(row):
    """Reuse the closest measured picture at each 25 fps tick for an arithmetic-only replay."""
    base = Fraction(row["time_base"]) * timing.RATE
    times = [frame["pts"] * base for frame in row["frames"]]
    start = round(times[0])
    stop = times[-1] + (times[-1] - times[-2])
    frames = []
    for tick in range(start, -(-stop // 1), timing.STEP):
        right = bisect_left(times, tick)
        nearest = min(range(max(0, right - 1), min(len(times), right + 1)),
                      key=lambda index: (abs(times[index] - tick), index))
        original = row["frames"][nearest]
        frames.append({**original, "index": len(frames), "pts": tick,
                       "recorded_frame_index": original["index"], "recorded_pts": original["pts"]})
    return row | {"frames": frames, "time_base": "1/16000"}


def evaluate(qualified, rows, primary, *, replay_25=False):
    held = []
    if len(qualified) != len(rows):
        raise ValueError("The native speaker roster differs")
    for item, row in zip(qualified, rows, strict=True):
        table = item["frame_table"]
        pts = [f["pts"] for f in row["frames"]]
        if (row["source"] != item["index"] or row["sha256"] != item["input_sha256"]
                or (row["time_base"], pts, row["width"], row["height"]) != (
                    table["time_base"], table["pts"], table["width"], table["height"])
                or row["audio_origin"] != round(Fraction(item["audio_clocks"]["frames"][0]["start"]) * timing.RATE)
                or row["samples"] != item["canonical_audio"]["samples"]
                or row["cases"]["original"]["pcm_sha256"] != item["canonical_audio"]["pcm_sha256"]
                or set(row["cases"]) != set(study.parent.ROLES)):
            raise ValueError("Native picture or audio source commitments differ")
        edit = row["edit"]
        if edit != [(row["samples"] - native.EDIT_WIDTH) // 2, (row["samples"] + native.EDIT_WIDTH) // 2]:
            raise ValueError("The edited interval is outside the frozen plan")
        if replay_25:
            row = nearest_25(row)
            pts = [frame["pts"] for frame in row["frames"]]
        for role, values in row["cases"].items():
            samples, origin = row["samples"], row["audio_origin"]
            mapping = {"samples": samples, "source_sample_rate": timing.RATE, "source_channels": 1,
                       "timeline_time_base": "1/16000", "pcm_sha256": values["pcm_sha256"],
                       "timeline": [{"start_sample": 0, "end_sample": samples, "source_pts": origin}]}
            baseline = binding.measured({"frames": row["frames"], "phones": values["phones"]},
                                        timestamps=pts, time_base=row["time_base"],
                                        source={"width": row["width"], "height": row["height"]}, mapping=mapping)
            pictures = [frame | {"gap": None if frame["gap_units"] is None else frame["gap_units"] / timing.SCORE_UNITS}
                        for frame in baseline["inputs"]["frames"]]
            trial = row | {"frames": pictures}
            candidates = dense.case_scores(trial, values, values["pcm_sha256"])
            for arm, result in candidates.items():
                if result["status"] == "measured":
                    plates = timing.picture_evidence(pictures, result, samples)
                    if not all(plates["picture_brackets"].values()):
                        candidates[arm] = {"status": "incomplete_context", "eligible_events": result["eligible_events"]}
            incumbent = baseline["measurement"]
            if baseline["status"] == "incomplete_context":
                incumbent = {"status": "incomplete_context", "eligible_events": incumbent["eligible_events"]}
            elif "score_units" in incumbent:
                incumbent = incumbent | {"score": incumbent["score_units"] / timing.SCORE_UNITS}
            shift = study.parent.EDITS.get(role)
            kind = "local" if shift else "constant" if role.startswith("global_") else "ordinary"
            held.append((trial, role, kind, shift, {"sparse": incumbent} | candidates))
    return dense.summarize(held, primary["result"]["points"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    run = phases.add_parser("capture")
    for name in ("primary", "audit", "media", "checkpoint", "geometry", "runtime", "output"):
        run.add_argument(name, type=Path)
    run.add_argument("--remaining-actors", action="store_true", help="Use the qualified actors after the first native roster")
    replay = phases.add_parser("review")
    for name in ("primary", "audit", "receipt"):
        replay.add_argument(name, type=Path)
    replay.add_argument("--clock-report", type=Path, help="Write a separate arithmetic replay on a virtual 25 fps grid")
    args = parser.parse_args()
    primary = dense.review(args.primary)
    if args.phase == "capture":
        record = capture(args, primary)
        dense.seal(args.output, record)
    else:
        record = study.load(args.receipt)
        qualified = native.roster(args.audit, remaining=record.get("remaining_actors") is True)
        if (record["schema"] != SCHEMA or record["primary_sha256"] != study.phone.sha(args.primary)
                or record["model"] != dense.MODEL
                or record["result"] != evaluate(qualified, record["rows"], primary)):
            raise ValueError("Native transfer results differ from replay")
        if args.clock_report:
            alternate = evaluate(qualified, record["rows"], primary, replay_25=True)
            comparison = {"schema": SCHEMA + "/virtual25", "source_sha256": study.phone.sha(args.receipt),
                          "native": record["result"]["outcomes"], "virtual25": alternate["outcomes"]}
            with args.clock_report.open("x", encoding="utf-8") as output:
                json.dump(comparison, output, indent=2, sort_keys=True)
                output.write("\n")
    print(json.dumps(record["result"]["outcomes"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
