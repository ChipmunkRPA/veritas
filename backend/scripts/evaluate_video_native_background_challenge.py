# SPDX-License-Identifier: Apache-2.0
"""Score the frozen guard only after the full tracked challenge screen seals."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import cv2

from scripts import video_native_background_challenge as panel
from scripts import video_native_background_edges as guard

common, require = panel.common, panel.require


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def lock(folder):
    candidate = common.digest(Path(guard.__file__))
    require(candidate == panel.CANDIDATE_SHA256, "The paired guard differs from the previously frozen method")
    return {"evaluator_sha256":common.digest(Path(__file__)),"source_freeze_sha256":common.digest(folder/"freeze.json"),
            "candidate_sha256":candidate,"gate":panel.SPEC["gate"]}


def frame_span(record, table):
    if record is None:
        return None
    require(record["time_base"]==table.time_base,"Range uses the wrong native clock")
    return [table.pts.index(record["start_pts"]),table.pts.index(record["end_pts"])]


def useful(values, interval):
    return bool(interval and any(value is not None and panel.old.iou(value,interval)>=Fraction(1,2) for value in values))


def outcome(path, row, metadata):
    require(common.digest(path)==metadata["sha256"],"Delivery identity changed")
    table=panel.earlier.previous.prior.clock(path)
    require(table.to_json()==metadata["clock"],"Verified native delivery clock changed")
    coarse=row["coarse"]
    result=guard.examine(path,coarse,table=table) if coarse else None
    ranges={key:frame_span(row[key],table) for key in ("coarse","boundary_refinement","short_range")}
    if result:
        a,b=result["edge_frames"]
        require(result["method"]==guard.METHOD and result["time_base"]==table.time_base
                and result["coarse_frames"]==ranges["coarse"] and result["edge_pts"]==[table.pts[a],table.pts[b]],"Incorrect accepted cut clock")
        require([p["index"] for p in result["pictures"]]==[a-1,a,b-1,b],"Four adjacent edge pictures required")
        for p in result["pictures"]:
            i=p["index"]
            require(p["pts"]==table.pts[i] and p["rgb_sha256"]==metadata["rgb_sha256"][i],"Accepted source RGB or PTS differs")
    retained={key:value for key,value in ranges.items() if not result or key=="short_range"}
    exact=bool(result and metadata["genuine_cut"] and result["edge_frames"]==metadata["interval"])
    close=bool(result and metadata["genuine_cut"] and all(abs(a-b)<=1 for a,b in zip(result["edge_frames"],metadata["interval"],strict=True)))
    true=metadata["truth"]
    coarse_useful=useful((ranges["coarse"],ranges["boundary_refinement"]),true)
    return {"source_sha256":metadata["sha256"],"source_pair":metadata["source_pair"],"direction":metadata["direction"],
            "condition":metadata["condition"],"authored_interval":metadata["interval"],"genuine_cut":metadata["genuine_cut"],
            "shot_boundaries":row["shot_boundaries"],"track_status":row["track_status"],"coarse_status":row["coarse_status"],
            "incumbent":ranges,"counterfactual":retained,"candidate":result,"exact_cut":exact,"cut_within_one_picture":close,
            "useful_current_swap":useful(ranges.values(),true),"useful_counterfactual_swap":useful(retained.values(),true),
            "useful_coarse_swap":coarse_useful,"lost_useful_coarse_swap":bool(result and coarse_useful),
            "false_cut_match":bool(result and not metadata["genuine_cut"])}


def report(rows):
    categories=defaultdict(Counter)
    exposed, improved=set(),set()
    for row in rows:
        name="graded_cut" if row["condition"].startswith("graded_") else "direct_cut" if row["condition"].startswith("direct_") else row["condition"]
        v=categories[name]
        v["cases"]+=1
        v["shot_boundary"]+=bool(row["shot_boundaries"])
        v["coarse_measured"]+=row["coarse_status"]=="ok"
        v["coarse_selected"]+=row["incumbent"]["coarse"] is not None
        v["short_selected"]+=row["incumbent"]["short_range"] is not None
        v["candidate_matches"]+=row["candidate"] is not None
        for key in ("exact_cut","cut_within_one_picture","useful_current_swap","useful_counterfactual_swap",
                    "useful_coarse_swap","lost_useful_coarse_swap","false_cut_match"):
            v[key]+=row[key]
        if row["genuine_cut"] and row["incumbent"]["coarse"]:
            exposed.add(row["source_pair"])
        if row["cut_within_one_picture"]:
            improved.add(row["source_pair"])
    required=panel.SPEC["gate"]
    gates={"useful_cut_removals":sum(x["cut_within_one_picture"] for x in categories.values())>=required["minimum_cut_removals"],
           "independent_pairs":len(improved)>=required["minimum_independent_pairs"],
           "useful_swap_coarse_coverage":sum(x["useful_coarse_swap"] for x in categories.values())>=required["minimum_useful_swap_coarse"],
           "useful_swap_preservation":not any(x["lost_useful_coarse_swap"] for x in categories.values()),
           "no_false_matches":not any(x["false_cut_match"] for x in categories.values())}
    return {"held_counts":{key:dict(value) for key,value in categories.items()},"alerted_cut_source_pairs":len(exposed),
            "improved_cut_source_pairs":len(improved),"gate":gates,"advances":all(gates.values())}


def run(folder,archive):
    source=panel.frozen(folder,archive)
    bound=lock(folder)
    require(common.read(folder/"evaluation-freeze.json")==bound,"Evaluation bindings were not frozen before screening")
    seal=common.read(folder/"incumbent-seal.json")
    screen=common.read(folder/"screen-freeze.json")
    require(seal["status"]=="verified" and seal["cases"]==108 and screen["candidate_sha256"]==bound["candidate_sha256"]
            and seal["journal_sha256"]==common.digest(folder/"private-incumbent.jsonl")
            and seal["screen_freeze_sha256"]==common.digest(folder/"screen-freeze.json")
            and seal["media_sha256"]==common.digest(folder/"media.json"),"Candidate cannot see an incomplete or changed incumbent screen")
    roster=common.read(folder/"media.json")["media"]
    scores=read_jsonl(folder/"private-incumbent.jsonl")
    rows=[]
    started=time.monotonic()
    journal=folder/"private-candidate.jsonl"
    with journal.open("x") as f:
        for row,item in zip(scores,roster,strict=True):
            require(all(row[k]==item[k] for k in ("file","sha256","condition","source_pair","direction","interval","truth","genuine_cut")),"Incumbent scored a different input")
            scored=outcome(folder/"media"/item["file"],row,item)
            rows.append(scored)
            f.write(json.dumps(scored,sort_keys=True,allow_nan=False)+"\n")
    require(len(rows)==108 and seal["candidate_challenge_source_pairs"]==sorted({row["source_pair"] for row in rows
            if row["genuine_cut"] and row["incumbent"]["coarse"]}),"Actual source-pair challenge roster differs")
    result={**report(rows),"protocol":source["spec"],"source_pair_count":len(source["pairs"]),
            "source_group_count":len(source["roster"]),"available_source_groups":source["available_source_groups"],
            "pair_gray_distances":[row["gray"] for row in source["pairs"]],"pair_palette_distances":[row["palette"] for row in source["pairs"]],
            "candidate_sha256":bound["candidate_sha256"],"source_freeze_sha256":bound["source_freeze_sha256"],
            "evaluation_freeze_sha256":common.digest(folder/"evaluation-freeze.json"),"incumbent_seal_sha256":common.digest(folder/"incumbent-seal.json"),
            "candidate_journal_sha256":common.digest(journal),"incumbent_elapsed_s":seal["elapsed_s"],"candidate_elapsed_s":time.monotonic()-started,
            "serving_change":False}
    common.write(folder/"aggregate.json",result)
    print(json.dumps(report(rows),sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("freeze","run"))
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=panel.ARCHIVE)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    if args.phase=="freeze":
        panel.frozen(args.folder,args.archive)
        require(not (args.folder/"private-incumbent.jsonl").exists(),"Freeze before held scoring")
        common.write(args.folder/"evaluation-freeze.json",lock(args.folder))
    else:
        run(args.folder,args.archive)


if __name__=="__main__":
    main()
