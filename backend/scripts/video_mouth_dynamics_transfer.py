# SPDX-License-Identifier: Apache-2.0
"""Localize relative high-order lip-geometry dynamics on verified GRID edits."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from scripts import video_mouth_reconstruction_transfer as prior

read, write, sha, require = prior.read, prior.write, prior.sha, prior.require
ROOT = prior.ROOT
DATA = ROOT / "backend/data/independent-evaluation"
PREVIOUS = DATA / "video-mouth-reconstruction-transfer-2026-11-11"
ARCHIVE = Path(read(PREVIOUS / "freeze.json")["archive"])
PUBLIC = ROOT / "docs/evaluation-panels"
OBS_FREEZE = PUBLIC / "grid-mouth-observer-freeze-v157.json"
OBS_RESULT = PUBLIC / "grid-mouth-observer-complete-v157.json"
FRAMES = tuple(range(15,59))
VARIANT = tuple(range(25,49))
ARMS = ("amplitude", "slope", "head_adjusted_slope")
SPEC = {
    "schema":"chipmunk.video-mouth-dynamics-transfer/1",
    "scope":"Actual retained native 25 fps GRID pilot pictures and four retained lip geometry measures, three inner-lip gaps and inner-mouth width. Use only the saved geometry of the tested picture itself, including exact RGB and timestamp matches to an earlier observation. Reuse the earlier known source-disjoint development/held component split. General project exposure remains.",
    "scan":"Search 12, 16 and 20 consecutive picture windows with ten consecutive pictures in each flank. Analyze only in-file geometry on one native clock, with no interpolation. Strict ordinary development comparison points are independently fixed from the third-highest case maximum for each arm.",
    "features":"Compute the mean of log standard-deviation ratios between first, second or third temporal differences and position, with a fixed 0.002 eye-normalized geometry floor. Contrast the core against the larger flank. The amplitude comparator uses log position standard deviation. The second candidate subtracts any positive contemporaneous excess of the same slope on log eye scale, eye roll and two nose offsets.",
    "native_step_s":"1/25","widths":[12,16,20],"flank":10,"floor":.002,"minimum_iou":"1/2",
    "gate":{"correct":6,"added_over_direct":2,"source_components":3,"source_compatible_correct":2,
            "maximum_extra_control_ranges":2,"minimum_assessed_fraction":.75},
    "source_compatible":"Existing independently measured visible motion agrees with source time and waveform is original. Achieved phoneme-level timing and listener synchrony have not been established.",
}


def observer_paths(archive,index):
    tag=f"clip-{index:02d}"
    private=archive/"grid-mouth-observer-v157"/tag
    stage=DATA/"grid-mouth-observer-v157"/tag/"authored"
    return {"input":private/"input-binding.json","observations":private/"observations.jsonl","controls":private/"control-checks.json",
            "result":stage/"output/result.json","progress":stage/"output/face-progress.jsonl","profile":stage/"profile.json",
            "resources":stage/"resources.jsonl"}


def sources(archive):
    held=read(PREVIOUS/"freeze.json")
    observer=read(OBS_FREEZE)
    public=read(OBS_RESULT)
    require(public["status"]=="complete" and public["complete_guarded_stages"]==16 and public["actual_face_task_inputs"]==4560
            and public["freeze_sha256"]==sha(OBS_FREEZE),"Original completed observer and exact model contract required")
    face_cards=observer["observer_runtime"]["bundled_face_model_cards"]
    require(len(face_cards)==3 and all(card["reviewed_license"]=="Apache-2.0" for card in face_cards.values()),"Previously reviewed geometry asset licenses required")
    pins=[]
    for item in held["clips"]:
        i=item["ordinal"]
        paths=observer_paths(archive,i)
        actual={key:sha(path) for key,path in paths.items()}
        result,profile=read(paths["result"]),read(paths["profile"])
        cases,motion=prior.files(archive,i)
        require(sha(cases)==item["cases_sha256"] and sha(motion)==item["motion_sha256"],"Previously frozen native case bank changed")
        require(result["clip_ordinal"]==i and result["status"]=="complete" and result["new_neural_calls"]==285
                and result["freeze_sha256"]==public["freeze_sha256"] and result["availability"]=={"geometry_available":285}
                and profile["child_exit_code"]==0 and profile["guard"] is None and profile["sample_errors"]==0
                and profile["worker_receipt_sha256"]==actual["result"] and profile["resource_journal_sha256"]==actual["resources"]
                and profile["runtime_sha256"]==observer["code_sha256"]["backend/scripts/grid_mouth_observer_v157.py"],"Saved observer execution binding differs")
        require(all(result[f"private_{field}_sha256"]==actual[key] for field,key in (("input_binding","input"),
                ("observation_journal","observations"),("control_checks","controls"),("face_progress","progress"))),"Saved source geometry artifact differs")
        check=read(paths["controls"])
        require(check==result["controls"] and all(c["mismatched_pairs"]==0 for c in check.values()),"Exact observer controls differ")
        pins.append({**item,"observer":actual,"role":"held" if item["component"] in held["held_components"] else "development"})
    require(len(pins)==16 and sum(p["role"]=="held" for p in pins)==8,"Exact previous disjoint source-role roster required")
    return {"protocol":SPEC,"clips":pins,"prior_freeze_sha256":sha(PREVIOUS/"freeze.json"),"prior_direct_sha256":sha(PREVIOUS/"held.jsonl"),
            "observer_freeze_sha256":sha(OBS_FREEZE),"observer_result_sha256":sha(OBS_RESULT),"worker_sha256":sha(Path(__file__)),"numpy_version":np.__version__}


def frozen(folder,archive):
    bound=read(folder/"freeze.json")
    require(bound==sources(archive),"Prospective method or saved sources changed")
    return bound


def observations(archive,index):
    cases_file,motion=prior.files(archive,index)
    cases=prior.jsonl(cases_file)
    paths=observer_paths(archive,index)
    binding=read(paths["input"])
    require(binding["bindings"]["construction_case_journal_sha256"]==sha(cases_file) and len(binding["requests"])==285,
            "Geometry was not measured on the exact archived native case bank")
    expected={r["case_number"]:r for r in cases}
    require(tuple(r["condition"] for r in cases)==prior.ORDER and len(expected)==11,"Complete native control roster required")
    observed=prior.jsonl(paths["observations"])
    require(len(observed)==285,"Complete historical observer journal required")
    captured=defaultdict(dict)
    for request,row in zip(binding["requests"],observed,strict=True):
        require(all(row[k]==request[k] for k in ("case_number","frame_index","repeat","source_pts","source_time_base","pts","time_base","rgb_sha256")),
                "Observation differs from exact committed task request")
        i=row["frame_index"]
        case=expected[row["case_number"]]
        require(row["status"]=="geometry_available" and row["observed_faces"]==1
                and row["rgb_sha256"]==case["expected"]["rgb_sha256"][i]
                and row["pts"]*Fraction(row["time_base"])==Fraction(i,25),"Native picture binding differs")
        if not row["repeat"]:
            require(i not in captured[row["case_number"]],"Duplicate native geometry")
            captured[row["case_number"]][i]=row
    original=captured[cases[0]["case_number"]]
    require(tuple(original)==FRAMES,"Original observation support differs")
    status={row["condition"]:row["status"] for row in read(motion)["cases"]}
    for case in cases:
        frames=captured[case["case_number"]]
        if case["condition"]!="original":
            require(tuple(frames)==VARIANT,"Native variant observation support differs")
        feature=[]
        source=[]
        for i in FRAMES:
            item=frames.get(i)
            if item is None and original[i]["rgb_sha256"]==case["expected"]["rgb_sha256"][i]:
                item=original[i]
            feature.append(item)
            source.append({"index":i,"pts":None if item is None else item["pts"],"rgb_sha256":case["expected"]["rgb_sha256"][i]})
        yield case,status[case["condition"]],feature,source


def geometry(rows):
    features=[]
    for row in rows:
        if row is None:
            features.append([float("nan")]*8)
            continue
        mouth=[float.fromhex(v) for v in row["descriptor_hex"]]
        head=[math.log(float.fromhex(row["eye_distance_px_hex"])),float.fromhex(row["eye_roll_radians_hex"]),
              *[float.fromhex(v) for v in row["nose_offset_hex"]]]
        features.append([*mouth,*head])
    return np.asarray(features)


def measures(window):
    sd=np.asarray([np.std(np.diff(window,n=i,axis=0),axis=0) for i in range(4)])
    logs=np.log(sd+SPEC["floor"])
    order=np.mean(logs[1:]-logs[0],axis=0)
    return float(np.mean(logs[0,:4])),float(np.mean(order[:4])),float(np.mean(order[4:]))


def select(geometry,source):
    best=dict.fromkeys(ARMS)
    cache={}

    def support(a,b):
        key=(a,b)
        if key not in cache:
            v=geometry[a:b]
            cache[key]=measures(v) if np.isfinite(v).all() else None
        return cache[key]

    flank=SPEC["flank"]
    for width in SPEC["widths"]:
        for start in range(flank,len(geometry)-width-flank+1):
            end=start+width
            details=(support(start-flank,start),support(start,end),support(end,end+flank))
            if None in details:
                continue
            left,core,right=details
            amp,slope,head=(core[i]-max(left[i],right[i]) for i in range(3))
            for arm,value in zip(ARMS,(amp,slope,slope-max(0.,head)),strict=True):
                if best[arm] is None or value>best[arm]["score"]:
                    best[arm]={"score":value,"frames":[FRAMES[start],FRAMES[end]],"clock_s":[str(Fraction(FRAMES[i],25)) for i in (start,end)],
                               "native_anchors":[source[i] for i in (start-1,(start+end)//2,end)],"support":[list(row) for row in details]}
    return best


def score(folder,archive,role):
    lock=frozen(folder,archive)
    if role=="held":
        point=read(folder/"points.json")
        require(point["freeze_sha256"]==sha(folder/"freeze.json") and point["development_sha256"]==sha(folder/"development.jsonl"),"Seal ordinary calibration before held scoring")
    start=time.monotonic()
    journal=folder/f"{role}.jsonl"
    count=0
    with journal.open("x") as stream:
        for pin in lock["clips"]:
            if pin["role"]!=role:
                continue
            index=pin["ordinal"]
            for case,motion,features,source in observations(archive,index):
                path=archive/"grid-native-clips-v149"/f"clip-{index:02d}"/Path(case["file"]).name
                require(sha(path)==case["file_sha256"],"Native input clip bytes differ")
                truth=case["edit_frames_half_open"] if case["condition"] in prior.NEURAL else None
                require(not truth or case["neural_provenance_verified"] and case["observed_neural_pixel_change"],"Actual verified changed mouth pictures required")
                record={"ordinal":index,"component":pin["component"],"condition":case["condition"],"truth":truth,"motion":motion,
                        "sha256":case["file_sha256"],"available_pictures":sum(item is not None for item in features),"arms":select(geometry(features),source)}
                stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
                count+=1
            stream.flush()
            print(role,"complete native source",index,flush=True)
    require(count==88,"Complete role case roster required")
    write(folder/f"{role}-seal.json",{"elapsed_s":time.monotonic()-start,"rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                     "journal_sha256":sha(journal),"freeze_sha256":sha(folder/"freeze.json")})


def calibrate(folder,archive):
    frozen(folder,archive)
    seal=read(folder/"development-seal.json")
    require(seal["journal_sha256"]==sha(folder/"development.jsonl") and seal["freeze_sha256"]==sha(folder/"freeze.json"),"Changed development rows")
    ordinary=[row for row in prior.jsonl(folder/"development.jsonl") if not row["truth"]]
    require(len(ordinary)==56,"Exact ordinary calibration roster required")
    points={}
    for arm in ARMS:
        scores=sorted(row["arms"][arm]["score"] for row in ordinary if row["arms"][arm])
        require(len(scores)>=42,"Enough supported ordinary calibration windows required")
        points[arm]=scores[-3]
    write(folder/"points.json",{"freeze_sha256":sha(folder/"freeze.json"),"development_sha256":sha(folder/"development.jsonl"),"points":points})


def outcome(record,point,truth):
    if record is None:
        return "unassessed"
    if record["score"]<=point:
        return "measured_no_range"
    if not truth:
        return "false_range"
    return "correct_range" if prior.exact_iou(record["clock_s"],[str(Fraction(i,25)) for i in truth])>=Fraction(1,2) else "misplaced_range"


def summary(folder,archive):
    lock=frozen(folder,archive)
    point=read(folder/"points.json")["points"]
    seal=read(folder/"held-seal.json")
    require(seal["journal_sha256"]==sha(folder/"held.jsonl") and seal["freeze_sha256"]==sha(folder/"freeze.json"),"Changed held rows")
    expected={(row["ordinal"],case["condition"]):case for row in prior.jsonl(PREVIOUS/"held.jsonl") for case in row["cases"]}
    direct_point=read(PREVIOUS/"calibration.json")["points"]["direct"]
    counts=defaultdict(lambda:defaultdict(Counter))
    gains=defaultdict(Counter)
    groups=defaultdict(set)
    rows=prior.jsonl(folder/"held.jsonl")
    for row in rows:
        prior_case=expected[(row["ordinal"],row["condition"])]
        require(row["sha256"]==prior_case["sha256"] and row["truth"]==prior_case["truth_frames"],"Previous comparison media or truth differs")
        compatible=bool(row["truth"] and row["motion"]==prior.SOURCE_CLOCK)
        band="compatible" if compatible else "other_neural" if row["truth"] else "control"
        old=prior_case["arms"]["direct"]["best"]
        old_case=None if old is None else {"score":old["score"],"clock_s":[str(Fraction(i,25)) for i in old["frames"]]}
        baseline=outcome(old_case,direct_point,row["truth"])
        counts["direct"][band][baseline]+=1
        for arm in ARMS:
            status=outcome(row["arms"][arm],point[arm],row["truth"])
            counts[arm][band][status]+=1
            gains[arm]["assessed"]+=status!="unassessed"
            gains[arm]["correct"]+=status=="correct_range"
            gains[arm]["added_over_direct"]+=status=="correct_range" and baseline!="correct_range"
            gains[arm]["source_compatible_correct"]+=compatible and status=="correct_range"
            gains[arm]["extra_control_ranges"]+=status=="false_range" and baseline!="false_range"
            if status=="correct_range":
                groups[arm].add(row["component"])
    require(len(rows)==88 and sum(counts["direct"]["compatible"].values())==7,"Exact held motion subset required")
    gate={}
    for arm in ARMS:
        gains[arm]["source_components"]=len(groups[arm])
        g=gains[arm]
        limits=SPEC["gate"]
        gate[arm]={"coverage":g["assessed"]>=88*limits["minimum_assessed_fraction"],
                   **{k:g[k]>=limits[k] for k in ("correct","added_over_direct","source_components","source_compatible_correct")},
                   "ordinary":g["extra_control_ranges"]<=limits["maximum_extra_control_ranges"]}
    return {"protocol":lock["protocol"],"counts":counts,"comparisons":gains,"gate":gate,"points":point,
            "advances":any(all(gate[arm].values()) for arm in ARMS if arm!="amplitude"),"held_runtime":seal,"new_neural_calls":0,
            "serving_change":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","development","calibrate","held","summary"))
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=ARCHIVE)
    args=parser.parse_args()
    if args.phase=="freeze":
        lock=sources(args.archive)
        args.folder.mkdir(mode=0o700)
        write(args.folder/"freeze.json",lock)
    elif args.phase=="calibrate":
        calibrate(args.folder,args.archive)
    elif args.phase=="summary":
        write(args.folder/"aggregate.json",summary(args.folder,args.archive))
    else:
        score(args.folder,args.archive,args.phase)


if __name__=="__main__":
    main()
