# SPDX-License-Identifier: Apache-2.0
"""Review two-pane RGB witness arithmetic and independently replay exact changed masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "chipmunk.image-embedded-pair-difference-transfer/1"
ROLES = ("sharp","feather","ordinary_local","duplicate","global_tone","unrelated","different_crop","ui")
POSITIVE = {"sharp","feather","ordinary_local"}


def need(check, why):
    if not check:
        raise ValueError(why)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def bounds(box):
    x,y,r,b=box
    return all(type(value) is int for value in box) and 0<=x<r<=768 and 0<=y<b<=640


def outcome(rows):
    expected={("panel",i,role) for i in range(12) for role in ROLES}|{("standalone",i,"photo") for i in range(8)}
    need(len(rows)==104 and {tuple(row["key"]) for row in rows}==expected,"Sealed denominator changed")
    counts,groups={},set()
    for row in rows:
        panel,slot,role=row["key"]
        base=row["baseline"]
        point=.8292734622955322
        need(bounds(base["box"]) and math.isfinite(base["score"]) and base["flag"]==(base["score"]>point),"Baseline arithmetic differs")
        need((row["target"]>0)==(role in POSITIVE),"Exact target role differs")
        evidence=row["evidence"] or {}
        if evidence:
            a,b=evidence["first"],evidence["second"]
            x,y,r,d=a
            w,h=r-x,d-y
            dx,dy=b[0]-x,b[1]-y
            changed=evidence["different_pixels"]
            need(bounds(a) and bounds(b) and w>=160 and h>=160 and b==[i+shift for i,shift in zip(a,(dx,dy,dx,dy),strict=True)]
                 and (abs(dx)>=w+24 or abs(dy)>=h+24) and max(abs(dx),abs(dy))>=192,"Comparison rectangle or translation differs")
            need(changed>=0 and evidence["exact_match_pixels"]+changed==w*h and changed*100<=15*w*h
                 and evidence["found"]==2*changed and 0<=evidence["false_pixels"]<=evidence["found"],"Native comparison counts differ")
            points=evidence["witnesses"]
            occupied={(min(2,3*(a-x)//w),min(2,3*(b-y)//h)) for a,b in points}
            need(len(points)>=12 and len(points)==len({tuple(p) for p in points}) and len(occupied)>=6
                 and all(x<=a<=r-16 and y<=b<=d-16 for a,b in points),"Distributed witness coverage differs")
            box=evidence["difference_box"]
            need((box is not None)==(changed>0),"Difference bounds do not match the count")
            if box:
                a,b,c,d=box
                need(8<=a<c<=w-8 and 8<=b<d<=h-8 and (c-a)*(d-b)<=.45*w*h,"Difference scope exceeds its support")
            expected_exact=role in POSITIVE and evidence["found"]==2*row["target"] and evidence["false_pixels"]==0
            need(evidence["exact_both"]==expected_exact,"Exact two-side mask counts differ")
        c=counts.setdefault(role,Counter())
        c.update(views=1,baseline_flags=base["flag"],baseline_useful=base["useful"],comparisons=bool(evidence),
                 difference_flags=bool(evidence.get("different_pixels")),exact_both=bool(evidence.get("exact_both")),
                 new_exact=bool(evidence.get("exact_both") and not base["useful"]))
        if role not in POSITIVE and evidence.get("different_pixels"):
            groups.add((panel,slot))
    return {"by_role":{key:dict(value) for key,value in counts.items()},"control_groups":len(groups),
            "passed":counts["sharp"]["new_exact"]>=6 and counts["feather"]["new_exact"]>=6 and not groups}


def verify(record):
    need(record["schema"]==SCHEMA and record["audit"]["views"]==104,"Unsupported study receipt")
    need(outcome(record["rows"])==record["result"],"Held results changed")
    return record["result"]


def native(args, output):
    import numpy as np

    from scripts import image_embedded_pair_difference_transfer as trial
    from scripts import image_viewport_frame_perturbation as sides

    freeze=args.run/"freeze.json"
    result=args.run/"result.json"
    lock,report=read(freeze),read(result)
    need(lock==trial.freeze(args) and sha(freeze.read_bytes())==report["freeze_sha256"],"Source freeze changed")
    audit=Counter(views=0,comparisons=0,witnesses=0,exact_masks=0,baseline_side_replays=0)
    published=[]
    for case, row, expected in zip(trial.cases(args),report["rows"],lock["cases"],strict=True):
        key,_,rgb,first,second,truth=case
        need(trial.identity(case)==expected and all(row[name]==value for name,value in expected.items()),"Exact frozen input differs")
        audit["views"]+=1
        known=np.zeros(rgb.shape[:2],dtype=bool)
        if first:
            for x,y,r,b in (first,second):
                known[y:b,x:r]=truth
        baseline=sides.side_support(rgb,row["baseline"]["box"])
        need(abs(baseline["score"]-row["baseline"]["score"])<=2e-5,"Existing score fails direct side replay")
        audit["baseline_side_replays"]+=1
        evidence=row["evidence"]
        public_row={name:value for name,value in row.items() if name not in {"rgb_sha256","truth_sha256"}}
        if evidence:
            audit["comparisons"]+=1
            x,y,r,b=evidence["first"]
            u,v,s,t=evidence["second"]
            for i,j in evidence["witnesses"]:
                dx,dy=u-x,v-y
                first_tile=rgb[j:j+16,i:i+16]
                second_tile=rgb[j+dy:j+dy+16,i+dx:i+dx+16]
                distinct=np.unique(first_tile.reshape(-1,3),axis=0).shape[0]
                need(distinct>=48 and np.array_equal(first_tile,second_tile),"Native exact texture witness differs")
                audit["witnesses"]+=1
            mask=np.max(np.abs(rgb[y:b,x:r].astype(np.int16)-rgb[v:t,u:s]),axis=2)>0
            changed=int(np.count_nonzero(mask))
            need(sha(mask.tobytes())==evidence["mask_sha256"] and changed==evidence["different_pixels"],"Exact paired difference differs")
            matched=np.zeros(rgb.shape[:2],dtype=bool)
            matched[y:b,x:r]=mask
            matched[v:t,u:s]=mask
            need(int(np.count_nonzero(matched&~known))==evidence["false_pixels"] and int(matched.sum())==evidence["found"],
                 "Independent native-region mapping differs")
            if evidence["exact_both"]:
                need(np.array_equal(matched,known),"The claimed exact mask differs")
                audit["exact_masks"]+=1
            public_row["evidence"]={name:value for name,value in evidence.items() if name!="mask_sha256"}
        published.append(public_row)
    portable={"schema":SCHEMA,"parameters":lock["parameters"],"code_sha256":lock["implementation_sha256"],
              "source_sha256":lock["source_sha256"],"freeze_sha256":report["freeze_sha256"],
              "result_sha256":sha(result.read_bytes()),"audit":dict(audit),"rows":published,
              "elapsed_s":report["elapsed_s"],"result":report["result"]}
    verify(portable)
    trial.write(output,portable)
    print(dict(audit),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt",type=Path)
    parser.add_argument("--native",nargs=4,type=Path)
    args=parser.parse_args()
    if args.native:
        native(SimpleNamespace(**dict(zip(("commons","lama","generated","run"),args.native,strict=True))),args.receipt)
    else:
        print(json.dumps(verify(read(args.receipt)),sort_keys=True))


if __name__=="__main__":
    main()
