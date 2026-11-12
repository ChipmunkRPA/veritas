# SPDX-License-Identifier: Apache-2.0
"""Independently rebuild challenge deliveries and verify conditional native decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from app.detectors.video.decode import FrameTable, decode_frames
from scripts import review_video_native_background_authoring as previous
from scripts import video_native_background_challenge as panel

common, require = panel.common, panel.require
EXPECTED = ("direct", "graded")


def rows(folder):
    manifest=common.read(folder/"media.json")
    require(manifest["freeze_sha256"]==common.digest(folder/"freeze.json") and len(manifest["media"])==108,"Different or incomplete media manifest")
    return manifest["media"]


def authored_names(direction):
    choices={f"{kind}_{n}":(span,None) for n,span in enumerate(((59,69),(65,77),(71,81))) for kind in EXPECTED}
    choices.update(long_swap=(((50,68),(56,78))[direction],("swap_a","swap_b")[direction]),original=(None,None))
    choices[("camera_punch","light")[direction]]=((65,77),None)
    return choices


def independent_grade(frame, gain, bias):
    return np.dstack([np.uint8(np.clip(np.rint(frame[:,:,j].astype(np.float32)*np.float32(gain[j])+np.float32(bias[j])),0,255))
                      for j in range(3)])


def reconstruct(folder,archive):
    freeze=panel.frozen(folder,archive)
    indexed={(r["source_pair"],r["direction"],r["condition"]):r for r in rows(folder)}
    pins={r["index"]:r for r in freeze["roster"]}
    require(len(indexed)==108 and len(pins)==12,"Exact paired roster required")
    counts=Counter()
    with TemporaryDirectory(prefix="private-authoring-",dir=folder) as scratch:
        temp=Path(scratch)
        for pair, matched in enumerate(freeze["pairs"]):
            a,b=matched["groups"]
            originals={key:previous._read(archive,pins[key],"original",range(120)) for key in (a,b)}
            counts["verified_native_source_pictures"]+=240
            for direction,(host,guest) in enumerate(((a,b),(b,a))):
                grade=matched["grades"][direction]
                require(all(.80<=x<=1.20 for x in grade["scale"]) and all(-24<=x<=24 for x in grade["offset"]),"Color grade violates frozen bounds")
                base,alternative=originals[host],originals[guest]
                for case,(span,swap) in authored_names(direction).items():
                    row=indexed[pair,direction,case]
                    output=temp/row["file"]
                    frames=base.copy()
                    require(row["interval"]==(list(span) if span else None) and row["truth"]==(list(span) if swap else None),"Authored truth differs")
                    if span:
                        start,stop=span
                        if case.startswith("direct_"):
                            replacement=alternative[start:stop]
                        elif case.startswith("graded_"):
                            replacement=[independent_grade(rgb,grade["scale"],grade["offset"]) for rgb in alternative[start:stop]]
                        elif swap:
                            replacement=previous._read(archive,pins[host],swap,range(start,stop))
                            counts["verified_native_source_pictures"]+=len(replacement)
                        elif case=="light":
                            replacement=[np.rint(np.minimum(rgb.astype(np.float32)*1.2,255)).astype(np.uint8) for rgb in base[start:stop]]
                        else:
                            require(case=="camera_punch","Uncommitted video edit")
                            replacement=[cv2.resize(rgb[8:-8,8:-8],(640,360),interpolation=cv2.INTER_LINEAR) for rgb in base[start:stop]]
                        require(len(replacement)==stop-start and all(not np.array_equal(a,b) for a,b in zip(base[start:stop],replacement,strict=True)),"Wrong support for authored pictures")
                        frames[start:stop]=replacement
                        counts["authored_intervals"]+=1
                    panel.old.encoded.encode(output,frames=frames,crf=23)
                    require(common.digest(output)==row["sha256"]==common.digest(folder/"media"/row["file"]),"Authored source replay differs in bytes")
                    counts["byte_identical_mp4s"]+=1
                    output.unlink()
            print("independently rebuilt source pair",pair,flush=True)
    require(counts["byte_identical_mp4s"]==108 and counts["authored_intervals"]==96,"Incomplete byte and truth audit")
    return {"status":"verified","counts":dict(counts),"media_sha256":common.digest(folder/"media.json"),"freeze_sha256":common.digest(folder/"freeze.json")}


def useful(spans, truth):
    return bool(truth and any(span and 2*max(0,min(span[1],truth[1])-max(span[0],truth[0]))>=max(span[1],truth[1])-min(span[0],truth[0]) for span in spans))


def replay(folder):
    manifest=rows(folder)
    aggregate=common.read(folder/"aggregate.json")
    authoring=common.read(folder/"independent-authoring.json")
    journal=folder/"private-candidate.jsonl"
    seal=common.read(folder/"incumbent-seal.json")
    require(authoring["status"]=="verified" and authoring["media_sha256"]==seal["media_sha256"]==common.digest(folder/"media.json")
            and aggregate["candidate_journal_sha256"]==common.digest(journal) and aggregate["incumbent_seal_sha256"]==common.digest(folder/"incumbent-seal.json"),"Audit requires all unchanged sealed receipts")
    reports=[json.loads(line) for line in journal.read_text().splitlines()]
    tracked=[json.loads(line) for line in (folder/"private-incumbent.jsonl").read_text().splitlines()]
    categories=defaultdict(Counter)
    flagged_pairs,removed_pairs=set(),set()
    verified_frames=0
    public=[]
    for row,source,incumbent in zip(reports,manifest,tracked,strict=True):
        require(row["source_sha256"]==source["sha256"]==incumbent["sha256"] and row["genuine_cut"]==source["genuine_cut"]
                and row["authored_interval"]==source["interval"],"Reported fixture identity or authored interval differs")
        shot=row["shot_boundaries"]
        before, after, found=row["incumbent"],row["counterfactual"],row["candidate"]
        table=FrameTable.from_json(source["clock"])
        for name, value in before.items():
            measured=incumbent[name]
            expected=None if measured is None else [table.pts.index(measured[key]) for key in ("start_pts","end_pts")]
            require(value==expected,"Candidate was passed a different measured range")
        require(after=={key:value for key,value in before.items() if found is None or key=="short_range"},"Counterfactual rerouted an existing range")
        if found:
            require(before["coarse"]==found["coarse_frames"],"Cut explanation moved the triggering coarse range")
            a,b=found["edge_frames"]
            valid,tiles=found["eligible_tiles"],found["support_tiles"]
            require(5<=b-a<=12 and len(valid)>=16 and 100*len(tiles)>=35*len(valid) and set(tiles)<=set(valid)
                    and sum(n>=2 for n in Counter((i//8>=3,i%8>=4) for i in tiles).values())>=3,"Invalid broad non-face support")
            indices=[a-1,a,b-1,b]
            require([x["index"] for x in found["pictures"]]==indices and found["edge_pts"]==[table.pts[a],table.pts[b]],"Exact native paired edges differ")
            path=folder/"media"/source["file"]
            for frame, item in zip(decode_frames(path,table,indices,timeout_s=10),found["pictures"],strict=True):
                i=frame.index
                require(frame.pts_verified and i==item["index"] and frame.pts==item["pts"]==table.pts[i]
                        and hashlib.sha256(frame.rgb.tobytes()).hexdigest()==item["rgb_sha256"]==source["rgb_sha256"][i],"Cut source picture cannot be replayed")
                verified_frames+=1
        cut=source["genuine_cut"]
        close=bool(found and cut and max(abs(a-b) for a,b in zip(found["edge_frames"],source["interval"],strict=True))<=1)
        exact=bool(close and found["edge_frames"]==source["interval"])
        true=source["truth"]
        protection=useful((before["coarse"],before["boundary_refinement"]),true)
        decisions={"exact_cut":exact,"cut_within_one_picture":close,"useful_current_swap":useful(before.values(),true),
                   "useful_counterfactual_swap":useful(after.values(),true),"useful_coarse_swap":protection,
                   "lost_useful_coarse_swap":bool(found and protection),"false_cut_match":bool(found and not cut)}
        require(all(row[key] is value for key,value in decisions.items()),"A held outcome fails independent arithmetic")
        if cut and before["coarse"]:
            flagged_pairs.add(row["source_pair"])
        if close:
            removed_pairs.add(row["source_pair"])
        condition=row["condition"]
        label=condition.split("_")[0]+"_cut" if cut else condition
        categories[label].update({"cases":1,"shot_boundary":bool(shot),"coarse_measured":row["coarse_status"]=="ok",
                                  "coarse_selected":before["coarse"] is not None,"short_selected":before["short_range"] is not None,
                                  "candidate_matches":found is not None,**decisions})
        if found or cut and before["coarse"]:
            public.append({"source_pair":row["source_pair"],"direction":row["direction"],"condition":condition,
                           "authored_frames":source["interval"],"coarse_frames":before["coarse"],
                           "candidate_frames":found["edge_frames"] if found else None,"matched_true_cut":close})
    require({key:dict(value) for key,value in categories.items()}==aggregate["held_counts"] and len(reports)==108
            and len(flagged_pairs)==aggregate["alerted_cut_source_pairs"] and len(removed_pairs)==aggregate["improved_cut_source_pairs"],"Aggregate differs from independently replayed decisions")
    gate={"useful_cut_removals":sum(x["cut_within_one_picture"] for x in categories.values())>=2,
          "independent_pairs":len(removed_pairs)>=2,"useful_swap_coarse_coverage":sum(x["useful_coarse_swap"] for x in categories.values())>=1,
          "useful_swap_preservation":not any(x["lost_useful_coarse_swap"] for x in categories.values()),
          "no_false_matches":not any(x["false_cut_match"] for x in categories.values())}
    require(gate==aggregate["gate"] and all(gate.values())==aggregate["advances"],"Frozen gate differs")
    return {"status":"verified","decisions":len(reports),"accepted_candidate_native_frames_verified":verified_frames,
            "aggregate_sha256":common.digest(folder/"aggregate.json"),"selected_cut_or_candidate_intervals":public}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("authoring","results"))
    parser.add_argument("folder",type=Path)
    parser.add_argument("--archive",type=Path,default=panel.ARCHIVE)
    args=parser.parse_args()
    cv2.setNumThreads(1)
    result=reconstruct(args.folder,args.archive) if args.phase=="authoring" else replay(args.folder)
    common.write(args.folder/f"independent-{args.phase}.json",result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
