# SPDX-License-Identifier: Apache-2.0
"""Describe the matched ordinary-cut failure on all three carriers after scoring."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np

from app.detectors.video import face_return
from scripts import video_face_six_codec_transfer as study

common, require = study.common, study.require


def render_h264(folder, archive, locked, models):
    media = folder / "diagnostic-h264"
    media.mkdir(mode=0o700)
    rows, pair = [], []

    def measure(group, role, pixels, truth):
        path = media / f"g{group}-{role}.mp4"
        study.original.encoded.encode(path, frames=pixels, crf=23)
        recorded = study.original.describe(path, group, "held", role, truth)
        arms, _, _ = study.original.compare(path, recorded, models)
        legacy = arms["eight_hz"]
        found = arms["dense_rank"] if legacy and not legacy["selected"] else None
        rows.append((path, {**recorded, "codec": "h264", "fallback":found,
                            "incumbent_flag":any(arms[k] and arms[k]["selected"] for k in ("coarse","eight_hz"))}))

    for group,pin in enumerate(locked["roster"]):
        if pin["role"] != "held":
            continue
        size,cases,root = study.durations.source_set(pin,archive)
        source = cases["original"]
        raw = study.original.encoded.native_frames(root / source["file"],source,size,range(study.original.N))
        measure(group,"original",raw.values(),None)
        for condition in ("swap_a","swap_b"):
            case = cases[condition]
            a,b = study.span(condition,group)
            edit = study.original.encoded.native_frames(root / case["file"],case,size,range(a,b))
            measure(group,condition,(edit.get(i,rgb) for i,rgb in raw.items()),[a,b])
        pair.append(raw)
        if len(pair) == 2:
            for phase,(base,alternate) in enumerate((pair,pair[::-1])):
                a,b = study.original.SPANS[phase]
                measure(group-1+phase,"hard_cut",(alternate[i] if a <= i < b else base[i] for i in range(study.original.N)),None)
            pair.clear()
        print("matched H.264 source group",group,flush=True)
    return rows


def boundary_profile(path, row, found, detector):
    table = study.clock(path)
    require(table.to_json() == row["clock"], "Boundary source clock differs")
    a,b = found["frames"]
    indices = (a-2,a,b-2,b)
    features = {}
    for frame in common.decode_frames(path,table,list(indices),timeout_s=20):
        i = frame.index
        rgb = frame.rgb
        require(frame.pts_verified and frame.pts == table.pts[i]
                and hashlib.sha256(rgb.tobytes()).hexdigest() == row["rgb_sha256"][i], "Boundary frame binding differs")
        candidates = common.detect_faces(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),detector=detector,min_size=48)
        face = candidates[0] if len(candidates) == 1 else None
        features[i] = face_return.scene(rgb,face)
    require(tuple(features) == indices, "Boundary evidence did not decode at every native picture")
    correlations, quartiles = [],[]
    for first,second in zip(indices[::2],indices[1::2],strict=True):
        p,a = features[first]
        q,b = features[second]
        support = a & b
        require(int(support.sum()) >= 8,"Candidate boundary has insufficient measurable background")
        values = np.sum(p[support]*q[support],axis=1)
        correlations.append(float(np.median(values)))
        quartiles.append(float(np.quantile(values,.25)))
    require(tuple(correlations) == tuple(found["scene_correlations"]), "Recomputed native scene correlations differ")
    return correlations,quartiles,indices


def diagnose(folder, archive):
    lock = study.frozen(folder,archive)
    original = common.read(folder / "media.json")
    held = study.journal(folder,"held")
    manifest = {row["file"]:row for row in original["media"]}
    delivery = [(folder / "media" / row["file"], {**manifest[row["file"]], **row}) for row in held]
    models = common.load_yunet(common.FACE_FILE),cv2.FaceRecognizerSF.create(str(common.APPEARANCE_FILE),"")
    h264 = render_h264(folder,archive,lock,models)
    common.write(folder / "private-h264-cut-diagnostic.json",{"rows":[row for _,row in h264]})
    by_codec = {}
    checked = 0
    details = []
    for codec in ("vp9","hevc","h264"):
        chosen = {"swap":[],"cut":[]}
        for path,row in (*delivery,*h264):
            found = row["fallback"]
            if row["codec"] != codec or not found or not found["selected"]:
                continue
            is_swap = row["truth"] and study.original.iou(found["frames"],row["truth"]) >= .5
            if not (is_swap or row["condition"] == "hard_cut"):
                continue
            median,quartiles,indices = boundary_profile(path,row,found,models[0])
            chosen["swap" if is_swap else "cut"].append((min(median),min(quartiles)))
            checked += len(indices)
            details.append({"codec":codec,"group":row["group"],"condition":row["condition"],"source_sha256":row["sha256"],
                            "boundary_indices":indices,"boundary_rgb_sha256":[row["rgb_sha256"][i] for i in indices],
                            "edge_medians":median,"edge_lower_quartiles":quartiles})
        values = {}
        for group, observations in chosen.items():
            values[group] = {"n":len(observations), **{key:[min(v[i] for v in observations),max(v[i] for v in observations)]
              for i,key in enumerate(("minimum_edge_median","minimum_edge_lower_quartile")) if observations}}
        by_codec[codec] = values
    matched = {}
    for codec in ("vp9","hevc","h264"):
        cases = delivery if codec != "h264" else h264
        matched[codec] = {row["group"] for _,row in cases if row["codec"] == codec and row["condition"] == "hard_cut"
                          and row["fallback"] and row["fallback"]["selected"]}
    common.write(folder / "private-scene-boundary-diagnostic.json",{"rows":details})
    return {"status":"verified","posthoc":True,"threshold_changed":False,"native_boundary_frames_replayed":checked,
            "h264_views":len(h264),"h264_original_flags":sum(row["condition"]=="original" and row["fallback"] is not None
                                                               and row["fallback"]["selected"] for _,row in h264),
            "all_codecs_flag_the_same_cuts":matched["vp9"]==matched["hevc"]==matched["h264"],"scene_correlation_bounds":by_codec}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=study.ARCHIVE)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = diagnose(args.folder,args.archive)
    common.write(args.folder / "posthoc-cut-diagnostic.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    main()
