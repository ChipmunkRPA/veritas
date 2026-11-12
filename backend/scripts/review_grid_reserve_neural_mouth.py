# SPDX-License-Identifier: Apache-2.0
"""Replay native neural mouth edits and compare visible closure timing."""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import grid_reserve_neural_mouth as study
from scripts import grid_reserve_word_clocks as clocks
from scripts import review_video_lipsync_local_delay_transfer as audio
from scripts.video_common_sources_v89 import read, require, sha, write

SPEC = {
    "schema": "chipmunk.grid-reserve-neural-mouth-replay/1",
    "observable": "Visible closure and reopening in the eye-normalized inner-lip aperture. "
                  "Select the closest measured local trajectory without using the intended timing.",
    "template_offsets": [-2, 4],
    "native_frame_tolerance": 1,
    "nearest": "Compare the mean-centered gap trajectory with the source closure template, "
               "then compare the chosen native timestamp against the two intended times.",
}


def frozen(out):
    return {"protocol": SPEC, "construction_freeze_sha256": sha(out/"freeze.json"),
            "implementation_sha256": sha(Path(__file__))}


def aperture(series, index):
    value = series[index]
    return None if value is None else float(np.median(value[:3]))


def best_event(source, observed, original, start, stop):
    a, b = SPEC["template_offsets"]
    source_template = [aperture(source, i) for i in range(original+a, original+b+1)]
    if any(x is None for x in source_template):
        return None
    template = np.asarray(source_template) - np.mean(source_template)
    candidates = []
    for index in range(start, stop):
        values = [aperture(observed, i) for i in range(index+a, index+b+1)]
        if any(x is None for x in values):
            continue
        if values[-a] > study.SPEC["maximum_closed_gap"] or max(values[-a+1:])-values[-a] < study.SPEC["minimum_reopening"]:
            continue
        centered = np.asarray(values) - np.mean(values)
        error = float(np.sqrt(np.mean((centered-template)**2)))
        candidates.append((error, index))
    if not candidates:
        return None
    error, selected = min(candidates)
    delta = selected-original
    tolerance = SPEC["native_frame_tolerance"]
    timing = "source_time_compatible" if abs(delta) <= tolerance else (
        "earlier_driver_compatible" if abs(delta-study.DRIVER_SHIFT) <= tolerance else "other_measured_closure")
    return {"observed_native_frame":selected,"observed_seconds":str(Fraction(selected,25)),
            "difference_from_source_frames":delta,"shape_error":error,"timing":timing}


def replay(qualified, out):
    freeze = study.check(qualified,out)
    require(read(out/"review-freeze.json") == frozen(out)
            and (out/"review-freeze.json").stat().st_mtime_ns < (out/"construction.json").stat().st_mtime_ns,
            "Require the predeclared source replay before completed construction")
    construction = read(out/"construction.json")
    journal = out/"render.jsonl"
    require(construction["freeze_sha256"] == sha(out/"freeze.json") and construction["render_journal_sha256"] == sha(journal),
            "Rendered evidence commitments differ")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    authored = {(row["ordinal"],row["mode"],row["frame_index"]):row for row in rows}
    require(len(authored)==len(rows),"Unexpected duplicate source picture")
    sources, _ = clocks.plan()
    module, task = study.mp_task()
    out_evidence = out/"replay"
    out_evidence.mkdir(mode=0o700)
    results, native_count, pcm_count, masks = [],0,0,0
    try:
        for entry in freeze["sources"]:
            ordinal = entry["ordinal"]
            original_table, original, source_audio, origin = clocks.exact_native(sources[ordinal]["input"])
            original_measurement = [study.observed(module, task, image)[1] for image in original]
            require(study.closure(original_measurement,entry["source_word_picture"])==entry["original_closure"],
                    "Independent original closure differs from the frozen source")
            first, stop = entry["span"]
            for case in construction["cases"]:
                if case["ordinal"]!=ordinal:
                    continue
                condition = case["mode"]
                clip = out/"deliveries"/case["file"]
                require(sha(clip)==case["proof"]["control_sha256"],"Frozen delivery bytes differ")
                table = probe_frames(clip,timeout_s=20)
                require([p*table.tb for p in table.pts]==[p*original_table.tb for p in original_table.pts],
                        "Edited and original native clocks differ")
                decoded, measured, observed_edits = [], [], []
                for frame in decode_frames(clip,table,None,timeout_s=30):
                    index = frame.index
                    require(frame.pts_verified and frame.pts==table.pts[index], "Delivered picture lacks exact native PTS")
                    delta = np.any(frame.rgb!=original[index],axis=2)
                    if delta.any():
                        observed_edits.append(index)
                        recorded = authored[ordinal,condition,index]
                        expected_driver = index if condition=="source_time_neural" else index-study.DRIVER_SHIFT
                        require(recorded["driver_frame_index"]==expected_driver
                                and hashlib.sha256(frame.rgb.tobytes()).hexdigest()==recorded["native_rgb_sha256"]
                                and hashlib.sha256((delta.astype(np.uint8)*255).tobytes()).hexdigest()==recorded["changed_mask_sha256"],
                                "Independently decoded RGB or exact changed mask differs from authoring")
                    decoded.append(frame.rgb)
                    measured.append(study.observed(module, task, frame.rgb)[1] if first-2 <= index <= stop+3 else None)
                require(len(decoded)==75 and observed_edits==list(range(first,stop)),"Delivery does not contain the exact edited interval")
                delivered_audio, delivered_origin, dropped = audio.source_pcm(clip,sha(clip))
                require(not dropped and delivered_origin==origin and np.array_equal(delivered_audio,source_audio),
                        "The complete output soundtrack differs from the native source")
                event = best_event(original_measurement,measured,entry["original_closure"],first,stop)
                representative = event["observed_native_frame"] if event else (first+stop)//2
                ids = [first-1,representative,stop]
                target = out_evidence/f"{ordinal:02}-{condition}"
                target.mkdir(mode=0o700)
                for label,index in zip(("before","during","after"),ids,strict=True):
                    Image.fromarray(decoded[index]).save(target/f"{label}-native.png")
                a,b = first*640,stop*640
                require(0<=a<b<=len(source_audio) and table.pts[first]*table.tb==Fraction(a,16000)
                        and table.pts[stop]*table.tb==Fraction(b,16000), "Aligned audio interval differs from native clock")
                (target/"matched-source-pcm.s16le").write_bytes(source_audio[a:b].tobytes())
                results.append({"mode":condition,"source_group":entry["group"],
                    "ordinal":ordinal,"original_closure_s":str(Fraction(entry["original_closure"],25)),
                    "published_word_source_seconds":entry["published_word_source_seconds"],"closure":event,
                    "authored_interval_s":[str(Fraction(i,25)) for i in (first,stop)],
                    "anchors_s":[str(Fraction(i,25)) for i in ids],"aligned_pcm_samples":b-a,
                    "matched_pcm_sha256":hashlib.sha256(source_audio[a:b].tobytes()).hexdigest()})
                native_count += len(decoded)
                pcm_count += len(delivered_audio)
                masks += len(observed_edits)
    finally:
        task.close()
    require(len(results)==6 and masks==len(authored),"Every frozen case must replay")
    write(out/"independent-replay.json",{"status":"verified","native_pictures":native_count,"unchanged_pcm_samples":pcm_count,
          "exact_changed_masks":masks,"results":results,"construction_sha256":sha(out/"construction.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","replay"))
    parser.add_argument("qualification",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.phase=="freeze":
        study.check(args.qualification,args.output)
        require(not (args.output/"render.jsonl").exists(),"Freeze native review before render")
        write(args.output/"review-freeze.json",frozen(args.output))
    else:
        replay(args.qualification,args.output)


if __name__=="__main__":
    main()
