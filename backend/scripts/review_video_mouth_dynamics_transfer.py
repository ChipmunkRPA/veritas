# SPDX-License-Identifier: Apache-2.0
"""Recompute native mouth-geometry windows and verify selected source pictures."""

from __future__ import annotations

import argparse
import hashlib
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from PIL import Image

from app.detectors.video.decode import decode_frames, probe_frames
from scripts import review_video_mouth_pose_recurrence_transfer as earlier
from scripts import video_mouth_dynamics_transfer as study

read, write, sha, require = study.read, study.write, study.sha, study.require
LIPS=(13,14,78,82,87,308,312,317)
POINTS=(1,13,14,33,78,82,87,133,263,308,312,317,362)


def squared_spread(values):
    mean=math.fsum(values)/len(values)
    return math.fsum((v-mean)**2 for v in values)/len(values)


def statistics(rows):
    if any(row is None for row in rows):
        return None
    columns=[[] for _ in range(8)]
    for row in rows:
        vector=[float.fromhex(v) for v in row["descriptor_hex"]]
        vector += [math.log(float.fromhex(row["eye_distance_px_hex"])),float.fromhex(row["eye_roll_radians_hex"])]
        vector += [float.fromhex(v) for v in row["nose_offset_hex"]]
        for col,value in zip(columns,vector,strict=True):
            col.append(value)
    baseline=[]
    slopes=[]
    for series in columns:
        level=math.log(math.sqrt(squared_spread(series))+.002)
        differences=[]
        for _ in range(3):
            series=[b-a for a,b in zip(series,series[1:])]
            differences.append(math.log(math.sqrt(squared_spread(series))+.002)-level)
        baseline.append(level)
        slopes.append(math.fsum(differences)/3)
    return math.fsum(baseline[:4])/4,math.fsum(slopes[:4])/4,math.fsum(slopes[4:])/4


def rescan(features):
    best=dict.fromkeys(study.ARMS)
    support={}
    for width in (12,16,20):
        for a in range(10,len(features)-width-9):
            b=a+width
            spans=((a-10,a),(a,b),(b,b+10))
            for l,r in spans:
                if (l,r) not in support:
                    support[l,r]=statistics(features[l:r])
            before,middle,after=(support[span] for span in spans)
            if before is None or middle is None or after is None:
                continue
            motion=middle[0]-max(before[0],after[0])
            roughness=middle[1]-max(before[1],after[1])
            pose=max(0.,middle[2]-max(before[2],after[2]))
            for arm, score in zip(study.ARMS,(motion,roughness,roughness-pose),strict=True):
                old=best[arm]
                if old is None or score>old[0]+1e-12:
                    best[arm]=(score,[15+a,15+b])
    return best


def verdict(record,point,truth):
    if record is None:
        return "unassessed"
    if record["score"]<=point:
        return "measured_no_range"
    if truth is None:
        return "false_range"
    left,right=record["frames"]
    a,b=truth
    union=max(b,right)-min(a,left)
    overlap=max(0,min(b,right)-max(a,left))
    return "correct_range" if 2*overlap>=union else "misplaced_range"


def native_exhibit(path, case, records, features, example):
    named=defaultdict(list)
    for arm,record in records.items():
        for label,item in zip(("before","during","after"),record["native_anchors"],strict=True):
            named[item["index"]].append((arm,label,item))
    table=probe_frames(path,timeout_s=15)
    require(not(table.truncated or table.rotation or table.interpolated) and len(table)==75
            and all(pts*table.tb==Fraction(i,25) for i,pts in enumerate(table.pts)),"Exact native 25 fps decoded support required")
    frames=0
    for frame in decode_frames(path,table,sorted(named),timeout_s=20):
        index=frame.index
        require(frame.pts_verified and frame.pts==table.pts[index],"Exhibit PTS not independently verified")
        digest=hashlib.sha256(frame.rgb.tobytes()).hexdigest()
        require(digest==case["expected"]["rgb_sha256"][index],"Exhibit RGB differs from archived native output")
        observed=features[index-15]
        require(observed is not None and observed["rgb_sha256"]==digest,"Exhibit does not bind its lip geometry")
        chosen=[(float.fromhex(x),float.fromhex(y)) for i,(x,y) in zip(POINTS,observed["landmarks_xy_hex"],strict=True) if i in LIPS]
        height,width,_=frame.rgb.shape
        pad=max(2,round(float.fromhex(observed["eye_distance_px_hex"])*.12))
        x,y=zip(*chosen,strict=True)
        x0,y0=max(0,math.floor(min(x))-pad),max(0,math.floor(min(y))-pad)
        x1,y1=min(width,math.ceil(max(x))+pad),min(height,math.ceil(max(y))+pad)
        require(x1>x0 and y1>y0,"Native lip crop is empty")
        for arm,label,item in named[index]:
            require(item["rgb_sha256"]==digest and item["pts"]==frame.pts,"Selected anchor was not found at the recorded time")
            frames+=1
            if example and arm=="slope":
                Image.fromarray(frame.rgb).save(example/f"{label}-picture.png")
                Image.fromarray(frame.rgb[y0:y1,x0:x1]).save(example/f"{label}-mouth.png")
    require(frames==3*len(records),"All selected exhibits must reopen")
    return frames


def review(folder,archive):
    lock=study.frozen(folder,archive)
    result=read(folder/"aggregate.json")
    held=study.prior.jsonl(folder/"held.jsonl")
    points=read(folder/"points.json")["points"]
    require(result["held_runtime"]["journal_sha256"]==sha(folder/"held.jsonl") and len(held)==88,"Incomplete held receipt")
    cases={(row["ordinal"],row["condition"]):row for row in held}
    controls=defaultdict(Counter)
    outcome=defaultdict(lambda:defaultdict(Counter))
    sizes=Counter()
    example_ranges=[]
    for pin in lock["clips"]:
        if pin["role"]!="held":
            continue
        index=pin["ordinal"]
        group=list(study.observations(archive,index))
        original=group[0][0]
        root=archive/"grid-native-clips-v149"/f"clip-{index:02d}"
        by_arm={}
        for case,motion,features,anchors in group:
            kind=case["condition"]
            row=cases[index,kind]
            path=root/Path(case["file"]).name
            require(sha(path)==row["sha256"]==case["file_sha256"] and motion==row["motion"],"Exact held media or visible-motion provenance differs")
            compatible=bool(row["truth"] and motion==study.prior.SOURCE_CLOCK)
            band="compatible" if compatible else "other_neural" if row["truth"] else "control"
            if compatible:
                source_case={**case,"motion_compatible":True}
                masks,frames=earlier.compatible_source(path,source_case,root/Path(original["file"]).name,original)
                sizes["compatible_changed_picture_masks"]+=masks
                sizes["compatible_decoded_pictures"]+=frames
                sizes["compatible_exact_original_pcm_pairs"]+=1
            winners=rescan(features)
            candidates={}
            for arm in study.ARMS:
                observed=row["arms"][arm]
                expected=winners[arm]
                require(bool(expected)==bool(observed),"Observation support differs in independent rescan")
                if observed:
                    require(abs(expected[0]-observed["score"])<1e-10 and expected[1]==observed["frames"],"Independent motion arithmetic or selection differs")
                    a,b=observed["frames"]
                    require(observed["clock_s"]==[str(Fraction(a,25)),str(Fraction(b,25))]
                            and observed["native_anchors"]==[anchors[i-15] for i in (a-1,(a+b)//2,b)],"Native evidence range differs")
                classification=verdict(observed,points[arm],row["truth"])
                outcome[arm][band][classification]+=1
                if classification=="false_range":
                    controls[arm][kind]+=1
                if classification in {"correct_range","misplaced_range","false_range"}:
                    candidates[arm]=observed
                sizes["replayed_candidate_decisions"]+=1
            example=None
            if compatible and "slope" in candidates and verdict(candidates["slope"],points["slope"],row["truth"])=="correct_range":
                example=folder/f"private-compatible-example-{len(example_ranges)+1}"
                example.mkdir(mode=0o700)
                a,b=candidates["slope"]["frames"]
                example_ranges.append({"authored_s":[str(Fraction(i,25)) for i in row["truth"]],"selected_s":[str(Fraction(i,25)) for i in (a,b)],
                                       "native_picture_s":[str(Fraction(x["index"],25)) for x in candidates["slope"]["native_anchors"]]})
            if candidates:
                sizes["selected_native_mouth_crops"]+=native_exhibit(path,case,candidates,features,example)
            by_arm[kind]=row["arms"]
            sizes["verified_held_clips"]+=1
        require(by_arm["original"]==by_arm["audio_delay_320ms"],"An audio-only delivery changed a geometry-only result")
        sizes["audio_only_control_pairs_identical"]+=1
        print("replayed held native group",index,flush=True)
    require(all(result["counts"][arm]=={band:dict(count) for band,count in outcome[arm].items()} for arm in study.ARMS)
            and sizes["verified_held_clips"]==88 and sizes["compatible_exact_original_pcm_pairs"]==7,"Independent decision replay differs")
    return {"status":"verified","aggregate_sha256":sha(folder/"aggregate.json"),"counts":dict(sizes),
            "ordinary_flags_by_condition":{arm:dict(count) for arm,count in controls.items()},"compatible_correct_examples":example_ranges}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=study.ARCHIVE)
    args=parser.parse_args()
    write(args.folder/"independent-audit.json",review(args.folder,args.archive))


if __name__=="__main__":
    main()
