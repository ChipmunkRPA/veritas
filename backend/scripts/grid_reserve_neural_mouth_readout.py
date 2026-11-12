# SPDX-License-Identifier: Apache-2.0
"""Apply earlier frozen timing and visual checks to the fresh GRID redraws."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from app.detectors.image.faces import detect_faces, load_yunet
from app.detectors.video import lipsync
from app.detectors.video.decode import decode_frames, probe_frames
from scripts import grid_reserve_word_clocks as clocks
from scripts import review_video_lipsync_local_delay_transfer as audio
from scripts import video_lipsync_local_contrast_transfer as timing
from scripts import video_mouth_temporal_detail as detail
from scripts.video_common_sources_v89 import read, require, sha, write

GRID_TIMING = clocks.planning.BASE / "video-lipsync-local-contrast-2026-11-12"
VISUAL = clocks.planning.BASE / "video-mouth-temporal-detail-2026-11-12"
SCOPE = {
    "schema": "chipmunk.grid-reserve-neural-mouth-readout/1",
    "cases": "Two held original sources, both source-time and earlier-expression neural versions, and both earlier-pixel ordinary versions.",
    "timing_truth": "Earlier driver moves the measured closure 8 pictures later on unchanged sound. "
                    "Previous signed rule measures the audio minus picture lag, so the known timing shift is -8 ticks. "
                    "Reuse the frozen 0.4 temporal overlap and two-tick lag tolerance.",
    "visual_truth": "For visually edited deliveries, locate the known altered pictures at half temporal IoU. "
                    "Ordinary picture controls remain conventional edits. Keep visual and timing outcomes separate.",
    "models": "No refitting. Use the frozen prior mouth/audio interaction head and visual timing points. "
              "The current timing comparison uses its original arithmetic with a directly observed single face.",
}


def freeze(study):
    construction = read(study/"construction.json")
    audit = read(study/"independent-replay.json")
    require(audit["status"]=="verified" and audit["construction_sha256"]==sha(study/"construction.json"),
            "Complete exact native qualification required")
    sources, _ = clocks.plan()
    trials = []
    for item in read(study/"freeze.json")["sources"]:
        ordinal = item["ordinal"]
        original = sources[ordinal]["input"]
        trials.append({"ordinal":ordinal,"role":"original","path":original["source_file"],"sha256":original["source_sha256"],
                       "picture_edit":None,"known_lag_ticks":None})
        for case in construction["cases"]:
            if case["ordinal"]!=ordinal:
                continue
            role = case["mode"]
            trials.append({"ordinal":ordinal,"role":role,"path":str(study/"deliveries"/case["file"]),
                           "sha256":case["proof"]["control_sha256"],"picture_edit":item["span"],
                           "known_lag_ticks":None if role=="source_time_neural" else -8})
    require(Counter(row["role"] for row in trials)==dict.fromkeys(
        ("original","source_time_neural","earlier_drive_neural","earlier_pixels_ordinary"),2),"The complete role roster is required")
    fit = read(GRID_TIMING/"head.json")
    point = read(GRID_TIMING/"point.json")
    previous = read(GRID_TIMING/"freeze.json")
    prior_roles = {row["ordinal"]:row["split"] for row in previous["roster"]}
    require(all(prior_roles[row["ordinal"]]=="held" for row in trials)
            and fit["freeze_sha256"]==sha(GRID_TIMING/"freeze.json")
            and point["head_sha256"]==sha(GRID_TIMING/"head.json")
            and point["journal_sha256"]==sha(GRID_TIMING/"point.jsonl"),"Prior frozen fit, point or source roles differ")
    visual = read(VISUAL/"aggregate.json")
    require(visual["runtime"]["calibration_sha256"]==sha(VISUAL/"calibration.json"),"Prior visual review point differs")
    bindings = [Path(__file__),Path(timing.__file__),Path(timing.baseline.__file__),Path(detail.__file__),Path(lipsync.__file__),detail.panel.prior.WEIGHT]
    return {"protocol":SCOPE,"sources":trials,"points":{"av_candidate":point["value"],**{arm:visual["points"][arm] for arm in detail.ARMS}},
            "parameters":{"timing_head_sha256":sha(GRID_TIMING/"head.json"),"timing_point_sha256":sha(GRID_TIMING/"point.json"),
                          "visual_calibration_sha256":sha(VISUAL/"calibration.json")},
            "bindings":{path.name:sha(path) for path in bindings}}


def observe(path, digest, detector):
    require(sha(path)==digest,"Source file bytes differ")
    table = probe_frames(path,timeout_s=15)
    require(not (table.rotation or table.interpolated or table.truncated) and (table.width,table.height)==(360,288)
            and [pts*table.tb for pts in table.pts]==[Fraction(i,25) for i in range(75)],"The exact GRID native clock differs")
    pcm, origin, duplicates = audio.source_pcm(path,digest)
    require(origin==0 and not duplicates and len(pcm)==47648,"Canonical source audio clock differs")
    ticks = lipsync.tick_grid(table.times(),3.,len(pcm),audio_start_s=0.)
    require(ticks.frames==tuple(range(74)),"Native timing ticks differ from the prior comparison")
    faces, patches, evidence, rgb = {},{},{},[]
    for frame in decode_frames(path,table,None,timeout_s=25):
        i = frame.index
        require(frame.pts_verified and frame.pts==table.pts[i],"A picture lacks its exact timestamp")
        rgb.append(audio.sha(frame.rgb.tobytes()))
        if i>=ticks.ticks:
            continue
        found = detect_faces(cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR),detector=detector,min_size=48)
        if len(found)!=1:
            continue
        face = found[0]
        record = {"frame_index":i,"pts":frame.pts,"landmarks":face.landmarks,
                  "bbox":dict(zip(("x","y","w","h"),face.box(),strict=True))}
        x,y,w,h = lipsync.mouth_box(record)
        x0,y0 = int(np.clip(np.floor(x),0,table.width-1)),int(np.clip(np.floor(y),0,table.height-1))
        x1,y1 = int(np.clip(np.ceil(x+w),x0+1,table.width)),int(np.clip(np.ceil(y+h),y0+1,table.height))
        faces[i],patches[i] = record,lipsync.mouth_patch(frame.rgb,record)
        evidence[i] = {"frame":i,"pts":frame.pts,"tb":table.time_base,"picture_sha256":rgb[-1],
                       "mouth_patch_sha256":audio.sha(patches[i].tobytes()),"xyxy":[x0,y0,x1,y1]}
    require(len(rgb)==75,"Expected all native picture bytes")
    return pcm,ticks,faces,patches,evidence,{"file_sha256":digest,"expected":{"picture_pts_ms":[40*i for i in range(75)],"rgb_sha256":rgb}}


def run(study, output):
    seal = read(output/"freeze.json")
    require(seal==freeze(study),"Detector readout freeze differs")
    cv2.setNumThreads(1)
    detector = load_yunet(detail.panel.prior.WEIGHT)
    fit = read(GRID_TIMING/"head.json")
    results = []
    for item in seal["sources"]:
        path = Path(item["path"])
        pcm,ticks,faces,patches,evidence,visual_source = observe(path,item["sha256"],detector)
        prior = timing.baseline.current(faces,patches,pcm.astype(np.float32)/32768,ticks)
        research = timing.candidate(timing.visual_features(patches,ticks.ticks),*timing.acoustic_features(pcm),fit)
        best = research["best"]
        av = [best] if best is not None and best["score"]>seal["points"]["av_candidate"] else []
        evidence_pack = timing.exhibit(best,evidence,pcm) if av else None
        visual = detail.select(*detail.edge_observations(path,visual_source,detector,{},{}))
        result = {"ordinal":item["ordinal"],"role":item["role"],"picture_edit":item["picture_edit"],
                  "known_lag_ticks":item["known_lag_ticks"],"native_mouth_pictures":len(evidence),
                  "current_timing":{"measured":prior["measured_windows"],"flags":prior["ranges"]},
                  "av_candidate":{"measured":research["measured"],"flags":av,"native_evidence":evidence_pack},
                  "visual":{arm:{"measured":rec is not None,"range":rec if rec is not None and rec["score"]>seal["points"][arm] else None}
                            for arm,rec in visual.items()}}
        if item["known_lag_ticks"] is not None:
            for name in ("current_timing","av_candidate"):
                result[name]["timing_correct"] = any(timing.qualifies(flag,item["picture_edit"],item["known_lag_ticks"])
                                                     for flag in result[name]["flags"])
        for rec in result["visual"].values():
            span = [str(Fraction(i,25)) for i in item["picture_edit"]] if item["picture_edit"] else None
            rec["picture_range_correct"] = bool(span and rec["range"] and detail.review.overlap(rec["range"]["source_clock_s"],span)>=Fraction(1,2))
        results.append(result)
        print(item["role"],len(evidence),{arm:len(result[arm]["flags"]) for arm in ("current_timing","av_candidate")},
              {arm:bool(rec["range"]) for arm,rec in result["visual"].items()},flush=True)
    write(output/"result.json",{"freeze_sha256":sha(output/"freeze.json"),"cases":results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","run"))
    parser.add_argument("study",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    if args.phase=="freeze":
        args.output.mkdir(mode=0o700)
        write(args.output/"freeze.json",freeze(args.study))
    else:
        run(args.study,args.output)


if __name__=="__main__":
    main()
