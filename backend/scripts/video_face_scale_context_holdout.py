# SPDX-License-Identifier: Apache-2.0
"""Confirm scale-only context using new genuine clips and the actual Full face path."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from scripts import screen_video_native_background_challenge as worker
from scripts.video_common_sources_v89 import read, require, sha, write

if __name__=="__main__" and len(sys.argv)>2 and sys.argv[1]=="full":
    worker.configure(Path(sys.argv[2]))
    os.environ["VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW"]="true"

from scripts import video_face_context_correction as candidate  # noqa: E402

near = candidate.near
PRIOR = candidate.DATA/"video-face-context-correction-2026-11-12"
ROLES = ("unchanged", "cutaway", "tandem_cutaway", "punch_in", "tandem_punch_in", "gain")
PLACEMENTS = ((56, 72), (63, 81))
SPEC = {
    "schema": "chipmunk.video-face-scale-context-holdout/1",
    "sources": "Ten separately archived CREMA-D originals, outside the previous near-head source roster. Prefer the fixed "
               "older selection role, then confidence-calibration role, with at least 55 qualified pictures.",
    "scope": [1280, 720, 30, 120, "h264", "yuv420p"],
    "interventions": list(map(list, PLACEMENTS)),
    "scale_point": [3., .45],
    "required_central_improvement": 2.,
    "reciprocal_geometry": {"minimum_scale_displacement": .07, "maximum_absolute_scale_product_error": .07},
    "conditions": list(ROLES),
    "observed": "Measure only the first available short, refined, or coarse published Full range; search within eight native "
                "pictures of each endpoint before correction. The authored arm independently uses exact boundaries.",
    "output": "Optional visual context only: aligning the pictures at different scales reduced the measured change. "
              "Retain the existing evidence, assessment and confidence.",
    "timed_gate": {"cutaway_groups_retained": 4, "zoom_files_explained": 7, "maximum_wrong_explanations": 1},
    "app_gate": {"minimum_zoom_ranges": 3, "minimum_zoom_explanations": 2, "minimum_cutaway_ranges": 6,
                 "minimum_cutaway_retained_share": .75, "maximum_wrong_explanations": 0},
}


def plan():
    from app.detectors.video.adapters import face_appearance

    previous = read(PRIOR/"calibration.json")
    aggregate = read(PRIOR/"aggregate.json")
    require(previous["point"]==SPEC["scale_point"] and not aggregate["timed_gate_passed"]
            and sha(PRIOR/"calibration.json")==aggregate["runtime"]["calibration_sha256"], "Prior experiment differs")
    old_sources = {row["source_group"] for row in read(candidate.CREMA/"freeze.json")["crema_originals"]}
    entries = {}
    for path in sorted(near.crema.NATIVE.glob("extract-*/output/cases.jsonl")):
        for row in map(json.loads,path.read_text().splitlines()):
            if (row["condition"]!="original" or row["status"]!="ready" or row["source_group"] in old_sources
                    or row["study_role"] not in ("selection","confidence_cal") or len(row["expected"]["rgb_sha256"])<55):
                continue
            relative = Path(*Path(row["file"]).parts[-3:])
            source = near.crema.ARCHIVE/"native-mouth-clips-v120"/relative
            require(source.is_file(),"Retained genuine source missing")
            entry = {**row,"file":str(source.relative_to(near.crema.ARCHIVE)),"journal_sha256":sha(path)}
            entries.setdefault(row["source_group"],entry)
    sources = sorted(entries.values(), key=lambda row:(row["study_role"]!="selection",row["asset_index"]))[:10]
    require(len(sources)==10 and len({r["source_group"] for r in sources})==10
            and all(r["source_group"] not in old_sources for r in sources),"Genuine source groups overlap")
    return {"protocol":SPEC,"sources":sources,"prior_calibration_sha256":sha(PRIOR/"calibration.json"),
            "prior_panel_sha256":sha(candidate.CREMA/"freeze.json"),
            "bindings":{Path(m.__file__).name:sha(m.__file__) for m in (near,candidate,worker,face_appearance)}
            |{"runner":sha(__file__),"face_asset":sha(near.development.ROOT/"models/weights/face_detection_yunet_2023mar.onnx")}}


def author(path,role,span,first,second,previous):
    if role.startswith("tandem_"):
        near.hd.authored(path,source=previous,crf=35)
    else:
        near.hd.authored(path,pictures=near.compose(first,second,span,role))


def prepare(output,lock):
    cv2.setNumThreads(1)
    media=output/"media"
    media.mkdir(mode=0o700)
    sources=lock["sources"]
    rows=[]
    for group,(a,b) in enumerate(zip(sources[::2],sources[1::2],strict=True),1):
        first,second=near.genuine_frames(a),near.genuine_frames(b)
        span=PLACEMENTS[(group-1)%2]
        for role in ROLES:
            path=media/f"h{group}-{role}.mp4"
            previous=media/f"h{group}-{role.removeprefix('tandem_')}.mp4"
            author(path,role,span,first,second,previous)
            rows.append(near.hd.describe(path,group,role,[list(span)]))
        print("prepared",group,flush=True)
    require(len(rows)==len({row["sha256"] for row in rows})==30,"Incomplete genuine holdout")
    write(output/"media.json",{"freeze_sha256":sha(output/"freeze.json"),"rows":rows})


def rows_for(output):
    media=read(output/"media.json")
    require(media["freeze_sha256"]==sha(output/"freeze.json") and len(media["rows"])==30,"Frozen input ledger differs")
    return media["rows"]


def worker_score(output):
    state=worker.configure(output)
    os.environ["VERITAS_VIDEO_FACE_SECOND_RETURN_REVIEW"]="true"
    require(read(output/"freeze.json")==plan(),"Full worker code or source roster changed")
    run=worker.scorer(state)
    began=time.monotonic()
    journal=output/"full.jsonl"
    with journal.open("x") as stream:
        for row in rows_for(output):
            result=run(output/"media"/row["file"],row)
            stream.write(json.dumps({key:row[key] for key in ("file","sha256","condition","group","truth")}|{"baseline":result},
                                    allow_nan=False)+"\n")
            stream.flush()
            print("Full",row["condition"],row["group"],bool(candidate.selection(result,row)),flush=True)
    write(output/"full.json",{"journal_sha256":sha(journal),"media_sha256":sha(output/"media.json"),
                             "elapsed_s":time.monotonic()-began})


def correction(result):
    if result["status"]!="assessed":
        return {"raw":False,"scale":False,"explanation":False}
    raw,_=candidate.disposition(result,"raw",SPEC["scale_point"])
    adjusted,_=candidate.disposition(result,"scale",SPEC["scale_point"])
    changes=[]
    for boundary in result["boundaries"]:
        fit=boundary["fitted"]["scale"]
        if fit is None:
            break
        improvement=fit["held_raw"]-fit["held_corrected"]
        scale=fit["transform"][0]
        if (improvement<max(SPEC["scale_point"][0],SPEC["scale_point"][1]*fit["held_raw"])
                or fit["central_improvement"]<SPEC["required_central_improvement"]
                or abs(scale-1)<SPEC["reciprocal_geometry"]["minimum_scale_displacement"]):
            break
        changes.append(scale)
    explanation=(raw and not adjusted and len(changes)==2 and (changes[0]-1)*(changes[1]-1)<0
                 and abs(changes[0]*changes[1]-1)<=SPEC["reciprocal_geometry"]["maximum_absolute_scale_product_error"])
    return {"raw":raw,"scale":adjusted,"explanation":explanation}


def measure(output):
    from app.detectors.image.faces import load_yunet

    cv2.setNumThreads(1)
    detector=load_yunet(near.development.ROOT/"models/weights/face_detection_yunet_2023mar.onnx")
    full=read(output/"full.json")
    require(full["journal_sha256"]==sha(output/"full.jsonl") and full["media_sha256"]==sha(output/"media.json"),
            "Full records differ")
    journal=output/"observations.jsonl"
    started=time.monotonic()
    with (output/"full.jsonl").open() as scores,journal.open("x") as out:
        for entry,scored in zip(rows_for(output),map(json.loads,scores),strict=True):
            require(all(scored[key]==entry[key] for key in ("file","sha256","condition","group","truth")),"Full case differs")
            observed=candidate.selection(scored["baseline"],entry)
            record={}
            for arm,intervals,radius in (("timed",entry["truth"],0),("app",observed,8)):
                result,reads=candidate.measure(output/"media"/entry["file"],entry,intervals,detector,radius)
                record[arm]={"reads":reads,"results":result}
            out.write(json.dumps({key:entry[key] for key in ("sha256","condition","group","truth")}|record,allow_nan=False)+"\n")
            print(entry["condition"],entry["group"],{k:[correction(x) for x in v["results"]] for k,v in record.items()},flush=True)
    write(output/"run.json",{"journal_sha256":sha(journal),"full_sha256":sha(output/"full.json"),"elapsed_s":time.monotonic()-started})


def report(output,lock):
    runtime=read(output/"run.json")
    require(runtime["journal_sha256"]==sha(output/"observations.jsonl") and runtime["full_sha256"]==sha(output/"full.json"),
            "Measured journal changed")
    ledger=rows_for(output)
    source_reads=0
    edge_count=0
    counts={arm:{role:Counter() for role in ROLES} for arm in ("timed","app")}
    retained={arm:set() for arm in counts}
    qualified_groups=set()
    with (output/"observations.jsonl").open() as stream:
        for entry,row in zip(ledger,map(json.loads,stream),strict=True):
            require(all(entry[key]==row[key] for key in ("sha256","condition","group","truth")),"Evidence source differs")
            role=row["condition"]
            for arm in counts:
                counts[arm][role]["files"]+=1
                source_reads+=row[arm]["reads"]
                for result in row[arm]["results"]:
                    count=counts[arm][role]
                    decision=correction(result)
                    count.update(ranges=1,assessed=result["status"]=="assessed",**decision)
                    if decision["scale"] and role in ("cutaway","tandem_cutaway"):
                        retained[arm].add(row["group"])
                    if decision["explanation"] and arm=="app":
                        qualified_groups.add(row["group"])
                    boundary_indices=[]
                    for boundary in result.get("boundaries",[]):
                        i=boundary["index"]
                        require(boundary["pts"]==entry["clock"]["pts"][i]
                                and boundary["native_hashes"]==entry["rgb_sha256"][i-1:i+1],"Observed edge differs from native source")
                        edge_count+=1
                        boundary_indices.append(i)
                    if boundary_indices and all(abs(a-b)<=2 for a,b in zip(boundary_indices,entry["truth"][0],strict=True)):
                        count["edges_within_two"]+=1
    rebuilds=0
    with TemporaryDirectory(prefix="native-rebuild-",dir=output) as name:
        tmp=Path(name)
        for group,(a,b) in enumerate(zip(lock["sources"][::2],lock["sources"][1::2],strict=True),1):
            first,second=near.genuine_frames(a),near.genuine_frames(b)
            span=PLACEMENTS[(group-1)%2]
            for role in ROLES:
                entry=next(row for row in ledger if row["group"]==group and row["condition"]==role)
                dst=tmp/entry["file"]
                previous=tmp/f"h{group}-{role.removeprefix('tandem_')}.mp4"
                author(dst,role,span,first,second,previous)
                require(sha(dst)==entry["sha256"],"Rebuilt delivery differs")
                rebuilds+=1
    timed,app=counts["timed"],counts["app"]
    zoom=("punch_in","tandem_punch_in")
    cuts=("cutaway","tandem_cutaway")
    nuisance=tuple(role for role in ROLES if role not in zoom)
    limits=SPEC["timed_gate"]
    cutaway_ranges=sum(app[k]["ranges"] for k in cuts)
    timed_gate={"retains_cutaway_groups":len(retained["timed"])>=limits["cutaway_groups_retained"],
                "explains_zoom":sum(timed[k]["explanation"] for k in zoom)>=limits["zoom_files_explained"],
                "limits_wrong_explanations":sum(timed[k]["explanation"] for k in nuisance)<=limits["maximum_wrong_explanations"]}
    limits=SPEC["app_gate"]
    app_gate={"eligible_zoom":sum(app[k]["ranges"] for k in zoom)>=limits["minimum_zoom_ranges"],
              "explains_zoom":sum(app[k]["explanation"] for k in zoom)>=limits["minimum_zoom_explanations"],
              "eligible_cutaways":cutaway_ranges>=limits["minimum_cutaway_ranges"],
              "retains_cutaways":bool(cutaway_ranges and sum(app[k]["scale"] for k in cuts)/cutaway_ranges>=limits["minimum_cutaway_retained_share"]),
              "limits_wrong_explanations":sum(app[k]["explanation"] for k in nuisance)<=limits["maximum_wrong_explanations"]}
    write(output/"aggregate.json",{"protocol":SPEC,"counts":counts,"runtime":runtime,"full":read(output/"full.json"),
                                    "source_reads":source_reads,"verified_boundaries":edge_count,"rebuilt_deliveries":rebuilds,
                                    "retained_cutaway_groups":{k:len(v) for k,v in retained.items()},
                                    "app_explanation_groups":len(qualified_groups),"timed_gate":timed_gate,"app_gate":app_gate,
                                    "advances":all((*timed_gate.values(),*app_gate.values()))})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","prepare","full","measure","report"))
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.phase=="full":
        worker_score(args.output)
        return
    frozen=plan()
    if args.phase=="freeze":
        args.output.mkdir(mode=0o700,parents=True)
        write(args.output/"freeze.json",frozen)
        return
    require(read(args.output/"freeze.json")==frozen,"Frozen scale check differs")
    if args.phase=="prepare":
        prepare(args.output,frozen)
    elif args.phase=="measure":
        measure(args.output)
    else:
        report(args.output,frozen)


if __name__=="__main__":
    main()
