# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the fine-scene trial and reopen selected native evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.detectors.video import face_return
from app.detectors.video.adapters.face_appearance import _measure, method
from app.detectors.video.decode import decode_frames, probe_frames
from app.evidence.face_interval_binding import face_context_box
from app.evidence.face_return_binding import validate_distinct, validate_record
from scripts import review_video_native_background_authoring as prior
from scripts import video_face_return_fine_scene as fine
from scripts import video_face_return_fine_scene_transfer as study
from scripts import video_face_six_codec_transfer as tooling
from scripts import video_native_background_panel as archive_panel

digest, read, require = study.digest, study.read, study.require


def reconstruct(folder, archive, rows):
    source = study.native_host()
    originals = prior._read(archive, source, "original", range(120))
    copies = 0
    with TemporaryDirectory(prefix="fine-scene-native-rebuild-",dir=folder) as scratch:
        for row in rows:
            if row["bank"] != "authored":
                continue
            a,b=row["truth"]
            alternate=prior._read(archive,source,row["publisher_variant"],range(a,b))
            require(all(not np.array_equal(original,changed) for original,changed in zip(originals[a:b],alternate,strict=True)), "Every exact edited native picture must change")
            pictures=originals.copy()
            pictures[a:b]=alternate
            output=Path(scratch)/row["file"]
            archive_panel.old.encoded.encode(output,frames=pictures,crf=23)
            require(digest(output)==row["sha256"],"Re-encoded publisher swap differs from scored source")
            copies+=1
    return copies


def match_edges(eligible, recorded):
    if recorded is None:
        return eligible is None
    if eligible is None or recorded.keys()!=eligible.keys():
        return False
    return all((math.isclose(value,recorded[key],abs_tol=2e-6,rel_tol=0) if isinstance(value,float) else value==recorded[key])
               for key,value in eligible.items())


def replay_candidate(path, row, table, raw_frames, expected, incumbent, detector, recognizer, example):
    record=expected["record"]
    spec={"width":table.width,"height":table.height,"time_base":table.time_base,"frame_count":len(table)}
    validate_record(record,spec,table_pts=table.pts,interpolated=table.interpolated,rotation=table.rotation,truncated=table.truncated)
    validate_distinct(record,incumbent)
    times,grid=face_return.dense_schedule(table,("h264","yuv420p"))
    require(list(grid)==record["scheduled_frame_indices"],"Candidate did not measure the exact serving sample grid")
    features,finer,measured={}, {},{}
    normal=face_return.scene_correlation
    for i in grid:
        rgb=raw_frames[i]
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        faces=tooling.common.detect_faces(bgr,detector=detector,min_size=48)
        require(len(faces)<=1,"Multiple faces reached candidate verification")
        face=faces[0] if faces else None
        observed=_measure(rgb,bgr,i,table,face,recognizer) if face else None
        features[i]=face_return.Sample(face,method.unit(observed[0]) if observed else None,face_return.scene(rgb,face))
        finer[i]=fine.fine_frame(rgb,face)
        if observed:
            measured[i]=observed[1]
    edges={}
    for a,b in zip(grid,grid[1:]):
        first,last=features[a],features[b]
        value=normal(first,last)
        if value is not None:
            continue
        result=fine.fine_edge(finer[a],finer[b])
        result["old_count"]=int(np.count_nonzero(first.scene[1]&last.scene[1]))
        edges[(id(first.scene),id(last.scene))]=result

    def guarded(first,last):
        report=edges.get((id(first.scene),id(last.scene)))
        return normal(first,last) if report is None else report["median"] if report["passes"] else None

    with patch.object(face_return,"scene_correlation",guarded):
        found=face_return.highest(times,grid,features)
    require(found is not None and face_return.distinct(found,table,incumbent)
            and [found.start,found.end]==[record["start_frame_index"],record["end_frame_index"]]
            and math.isclose(found.response,float.fromhex(record["response_hex"]),abs_tol=2e-6,rel_tol=0)
            and [list(group) for group in found.support]==[record["support_indices"][k] for k in face_return.ROLES]
            and found.anchors()==record["anchors"],"Production ranking or native support differs on readback")
    a,b,c,d=found.scene_frames
    selected_edges=[edges.get((id(features[l].scene),id(features[r].scene))) for l,r in ((a,b),(c,d))]
    require(all(match_edges(calculated,saved) for calculated,saved in zip(selected_edges,expected["fine_edges"],strict=True))
            and any(selected_edges),"Independently measured fine background differs")
    require([i["frame_index"] for i in record["scene_frame_inputs"]]==[a,b,c,d] and all(
            i["decoded_rgb_sha256"]==row["rgb_sha256"][i["frame_index"]] and i["pts"]==table.pts[i["frame_index"]] for i in record["scene_frame_inputs"]),
            "Exact native scene inputs differ")
    for item in record["frame_inputs"]:
        i=item["frame_index"]
        check=measured[i]
        require(all(item[k]==check[k] for k in ("pts","bbox","decoded_rgb_sha256","aligned_rgb_sha256")), "Measured crop pixels or time differ")
    hashes={}
    for role,i in found.anchors().items():
        picture=raw_frames[i]
        box=face_context_box(measured[i]["bbox"],spec)
        x,y,w,h=(box[k] for k in ("x","y","w","h"))
        crop=picture[y:y+h,x:x+w]
        hashes[role]=hashlib.sha256(crop.tobytes()).hexdigest()
        if example:
            Image.fromarray(picture).save(example/f"{role}-picture.png")
            Image.fromarray(crop).save(example/f"{role}-crop.png")
    return {"recorded_face_inputs":len(record["frame_inputs"]),"recorded_scene_edge_pictures":4,"exact_native_crops":len(hashes),
            "selected_edges_with_measured_faces":sum(bool(report and features[l].face and features[r].face)
                for report,(l,r) in zip(selected_edges,((a,b),(c,d)),strict=True)),
            "selected_fine_edges":sum(value is not None for value in selected_edges)}


def overlap(first,second):
    if first is None or second is None:
        return False
    a,b=first
    c,d=second
    return 2*max(0,min(b,d)-max(a,c))>=max(b,d)-min(a,c)


def audit(folder, archive):
    lock,roster=study.frozen(folder)
    result=read(folder/"aggregate.json")
    held=[json.loads(line) for line in (folder/"held.jsonl").read_text().splitlines()]
    require(result["runtime"]["journal_sha256"]==digest(folder/"held.jsonl") and len(held)==len(roster)==31,"Incomplete sealed decisions")
    require(lock["manifest_sha256"]==digest(folder/"manifest.json"),"Scored manifest detached from freeze")
    cv2.setNumThreads(2)
    yunet=tooling.common.load_yunet(tooling.common.FACE_FILE)
    sface=cv2.FaceRecognizerSF.create(str(tooling.common.APPEARANCE_FILE),"")
    totals=Counter(new_byte_identical_reconstructions=reconstruct(folder,archive,roster))
    rates=defaultdict(Counter)
    groups=set()
    public=[]
    for expected,row in zip(roster,held,strict=True):
        require(all(expected[k]==row[k] for k in ("sha256","role","source_scope","truth","cut_truth")),"Candidate used wrong input or label")
        path=study.path_for(folder,expected)
        require(digest(path)==row["sha256"],"Exact held video changed")
        table=probe_frames(path,timeout_s=20)
        require(table.to_json()==expected["clock"] and tooling.codec_of(path)=="h264,yuv420p", "Verified delivery scope changed")
        frames={}
        for picture in decode_frames(path,table,range(len(table)),timeout_s=20):
            i=picture.index
            require(picture.pts_verified and picture.pts==table.pts[i] and hashlib.sha256(picture.rgb.tobytes()).hexdigest()==expected["rgb_sha256"][i],
                    "Native decoder replay differs")
            if row["candidate"]:
                frames[i]=picture.rgb
            totals["native_pictures_reopened"]+=1
        incumbent=row["baseline"]["coarse"]
        baseline=[[table.pts.index(item["start_pts"]),table.pts.index(item["end_pts"])] for item in
                  (incumbent,row["baseline"]["short_range"],row["baseline"]["boundary_refinement"]) if item]
        proof=row["candidate"]
        outcome=rates[row["role"]]
        outcome["cases"]+=1
        outcome["prior_flags"]+=bool(baseline)
        old=any(overlap(value,row["truth"]) for value in baseline)
        outcome["prior_useful"]+=old
        outcome["attempted_retry"]+=row["retry_coverage"] is not None
        outcome["added_flags"]+=proof is not None
        good=False
        if proof:
            record=proof["record"]
            pair=[record["start_frame_index"],record["end_frame_index"]]
            good=overlap(pair,row["truth"])
            require(row["baseline"]["short_range"] is None and row["retry_coverage"]["status"]=="localized","Candidate displaced an incumbent short range")
            example=None
            if good and not old:
                groups.add(row["source_scope"])
                example=folder/f"private-evidence-{len(public)+1}"
                example.mkdir(mode=0o700,exist_ok=True)
            totals.update(replay_candidate(path,expected,table,frames,proof,incumbent,yunet,sface,example))
            public.append({"source_scope":row["source_scope"],"role":row["role"],"authored_frames":row["truth"] or row["cut_truth"],
                           "prior_frames":baseline,"candidate_frames":pair,"useful":good,"added_useful":good and not old,
                           "native_anchor_frames":record["anchors"],"fine_background":proof["fine_edges"]})
        outcome["added_useful"]+=good and not old
        outcome["combined_useful"]+=good or old
        totals["decisions_replayed"]+=1
    require({k:dict(value) for k,value in rates.items()}==result["counts"] and len(groups)==result["useful_source_groups"],"Independent outcome counts differ")
    gate={"added_useful":rates["swap"]["added_useful"]>=3,"positive_groups":len(groups)>=2,"ordinary":rates["ordinary"]["added_flags"]<=1,
          "cuts":rates["cut"]["added_flags"]==0,"preserves_existing_ranges":True}
    require(gate==result["gate"] and all(gate.values())==result["advances"] and totals["native_pictures_reopened"]==3720
            and totals["new_byte_identical_reconstructions"]==2,"Frozen result or native verification differs")
    return {"status":"verified","aggregate_sha256":digest(folder/"aggregate.json"),"counts":dict(totals),"candidate_ranges":public}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=archive_panel.ARCHIVE)
    args=parser.parse_args()
    result=audit(args.folder,args.archive)
    study.write(args.folder/"independent-audit.json",result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
