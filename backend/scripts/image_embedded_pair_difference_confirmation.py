# SPDX-License-Identifier: Apache-2.0
"""Confirm the unchanged two-pane native comparison on separate source and layout roles."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

SCHEMA = "chipmunk.image-embedded-pair-difference-confirmation/1"
SHAPE = (768,768)
ROLES = ("sharp","feather","ordinary_local","duplicate","global_tone","unrelated","crop_shift","resampled","ui_card","triple")
POSITIVE = {"sharp","feather","ordinary_local"}
PILOT = "0ff4694f313a8389453ba57ccaabb3a7b3eea69a1a3d856d0fe7a4b8674666b7"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def write(path,value):
    with path.open("x") as out:
        json.dump(value,out,sort_keys=True,indent=2,allow_nan=False)
        out.write("\n")


def need(value,message):
    if not value:
        raise ValueError(message)


def choose(args):
    original=read(args.primary/"qualified-private.json")
    need(file_sha(args.primary/"qualified-private.json")==read(args.primary/"summary-v225.json")["qualified_private_sha256"],
         "Qualified primary archive changed")
    archive=read(args.earlier/"plan.json")["cases"]
    prior_lama=read(args.backup/"qualified-private.json")["qualified"]
    prior_generated=read(args.pilot_generated/"plan.json")["cases"]
    groups=[[row["source"] for row in a] for a in (original["qualified"],prior_lama)]+[archive,prior_generated]
    for key in ("flickr_photo_id","flickr_owner"):
        ids=[str(row[key]).casefold() for group in groups for row in group]
        need(len(ids)==len(set(ids))==55,"Source and contributor roles overlap")
    chosen=[]
    for split in sorted({row["source"]["publisher_split"] for row in original["qualified"]}):
        items=[row for row in original["qualified"] if row["source"]["publisher_split"]==split]
        chosen.extend(sorted(items,key=lambda row:sha(f"embedded-pair-confirm/1/{row['source']['flickr_photo_id']}".encode()))[:2])
    generated=[]
    for family in ("sdxl","kandinsky"):
        choices=[row for row in archive if row["generator"]==family]
        generated.extend(sorted(choices,key=lambda row:sha(f"embedded-pair-confirm/1/{row['flickr_photo_id']}".encode()))[:2])
    need(len(chosen)==8 and len(generated)==4,"Confirmation roles incomplete")
    return Path(original["cache"]),chosen,generated


def photo_pairs(args):
    import numpy as np
    from PIL import Image

    from scripts import image_crop_layout_transfer as p

    cache,qualified,other=choose(args)
    for slot,row in enumerate(qualified):
        parts=[]
        for role in ("original","edited"):
            asset=row["measurement"]["files"][role]
            path=cache/asset["file"]
            need(file_sha(path)==asset["sha256"],"Qualified photo bytes changed")
            parts.append(p.pixels(path))
        before,after=parts
        truth=(before!=after).any(axis=2)
        path=cache/row["truth_file"]
        need(file_sha(path)==row["truth_png_sha256"],"Qualified mask changed")
        with Image.open(path) as mask:
            need(np.array_equal(np.asarray(mask.convert("L"))>0,truth),"Original exact mask changed")
        yield slot,"lama",before,after,p.feather(before,after,truth)
    variants={case["slot"]:p.REGIONAL.native_variants(images,case["box"])
              for case,images,_ in p.REGIONAL.paired.inputs(args.earlier)}
    for slot,row in enumerate(other,8):
        rgb=variants[row["slot"]]
        yield slot,row["generator"],rgb["original"],rgb["edited"],rgb["feathered_edit"]


def canvas(a,b,slot,vertical,card=False):
    import numpy as np
    from PIL import Image, ImageDraw

    ph,pw=a.shape[:2]
    x,y=(63+slot%10,17+slot%4) if vertical else (17+slot%4,63+slot%10)
    dx,dy=((slot%7)-3,ph+39+slot%7) if vertical else (pw+39+slot%7,(slot%7)-3)
    first=[x,y,x+pw,y+ph]
    second=[x+dx,y+dy,x+dx+pw,y+dy+ph]
    rgb=np.full((*SHAPE,3),(31,38,53) if slot%3==1 else (232,237,244),dtype=np.uint8)
    for index,(box,photo) in enumerate(((first,a),(second,b))):
        left,top,right,bottom=box
        rgb[top-3:bottom+3,left-3:right+3]=(115+23*index,137+17*index,159+11*index)
        rgb[top:bottom,left:right]=photo
    if card:
        view=Image.fromarray(rgb)
        draw=ImageDraw.Draw(view)
        for index,(x0,y0,x1,y1) in enumerate((first,second)):
            draw.rectangle((x0,y0,x1-1,y0+23),fill=(215,226,242))
            draw.text((x0+12,y0+5),f"Reference panel {1+index} / {slot:02d}",fill=(32,57,86))
            draw.rectangle((x0,y1-19,x1-1,y1-1),fill=(207,220,237))
        rgb=np.asarray(view)
    return rgb,first,second


def triple(rgb,slot):
    import numpy as np

    h,w=rgb.shape[:2]
    cut=rgb[:min(444,h),max(0,(w-220)//2):max(0,(w-220)//2)+220]
    canvas=np.full((*SHAPE,3),(227,237,245),dtype=np.uint8)
    for i in range(3):
        y=90+3*(i%2)
        x=22+i*246
        canvas[y:y+len(cut),x:x+cut.shape[1]]=cut
        if i==2:
            sample=canvas[y+107:y+129,x+81:x+106]
            sample[:]=np.minimum(sample.astype(np.uint16)+17,255)
    return canvas


def cases(args):
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    from scripts import image_crop_layout_transfer as p
    from scripts import image_embedded_pair_difference_transfer as pilot

    available=p.sources.binding(args.commons)["sources"]
    previous={row["slot"] for row in pilot.donor_roster(args)}
    chosen=sorted((row for row in available if row["slot"] not in previous),
                  key=lambda row:sha(f"embedded-pair-confirm/1/control/{row['sha256']}".encode()))
    need(len(chosen)==18,"Expected ordinary photo groups disjoint from pilot")
    for slot,family,before,after,blend in photo_pairs(args):
        truth=(before!=after).any(axis=2)|(before!=blend).any(axis=2)
        top_left_x,top_left_y,right,bottom=pilot.bounds(truth)
        h,w=truth.shape
        vertical=bool(slot%2)
        pw,ph=(min(w,592),344) if vertical else (344,min(h,592))
        x=min(max(0,(top_left_x+right-pw)//2),w-pw)
        y=min(max(0,(top_left_y+bottom-ph)//2),h-ph)
        photo=(x,y,x+pw,y+ph)
        need(pilot.crop(truth,photo).sum()>=.85*truth.sum(),"Native pane loses too much source edit")
        original,edited,feathered=(pilot.crop(rgb,photo) for rgb in (before,after,blend))
        pillow=Image.fromarray(original)
        ordinary=original.copy()
        blurred=np.asarray(pillow.filter(ImageFilter.GaussianBlur(1.3)))
        where=np.s_[ph//3:ph//3+ph//6,pw//3:pw//3+pw//6]
        ordinary[where]=blurred[where]
        tone=np.asarray(ImageEnhance.Contrast(pillow).enhance(.91))
        unrelated=np.asarray(p.native.source(args.commons,chosen[slot],(pw,ph)))
        shifted=np.roll(original,(4,-7),axis=(0,1))
        resampled=np.asarray(pillow.resize((pw-11,ph-13),Image.Resampling.LANCZOS).resize((pw,ph),Image.Resampling.BICUBIC))
        alternatives=(edited,feathered,ordinary,original,tone,unrelated,shifted,resampled,original,original)
        for role,alternate in zip(ROLES,alternatives,strict=True):
            if role=="triple":
                yield ("panel",slot,role),family,triple(original,slot),None,None,None
                continue
            rgb,a,b=canvas(original,alternate,slot,vertical,role=="ui_card")
            truth=(original!=alternate).any(axis=2) if role in POSITIVE else np.zeros((ph,pw),dtype=bool)
            need((role in POSITIVE)==bool(truth.any()),"Known change missing")
            yield ("panel",slot,role),family,rgb,a,b,truth
    for slot,row in enumerate(chosen[12:]):
        rgb=np.asarray(p.native.source(args.commons,row,SHAPE[::-1]))
        yield ("standalone",slot,"photo"),"credited-photo",rgb,None,None,None


def freeze(args):
    from scripts import image_embedded_pair_difference_transfer as pilot

    need(file_sha(args.pilot_run/"freeze.json")==PILOT,"Pilot evidence freeze changed")
    source=read(args.pilot_run/"freeze.json")
    need(file_sha(Path(pilot.__file__))==source["implementation_sha256"],"Pilot comparison implementation changed")
    rows=[pilot.identity(case) for case in cases(args)]
    expected={("panel",i,role) for i in range(12) for role in ROLES}|{("standalone",i,"photo") for i in range(6)}
    need(len(rows)==len(expected)==126 and {tuple(row["key"]) for row in rows}==expected,"Confirmation roster changed")
    return {"schema":SCHEMA,"pilot_sha256":PILOT,"compare_sha256":source["implementation_sha256"],
            "implementation_sha256":file_sha(Path(__file__)),"parameters":pilot.PARAMETERS,
            "shape":list(SHAPE),"layouts":["horizontal","vertical"],"cases":rows,
            "source_sha256":{name:file_sha(path) for name,path in {"primary":args.primary/"qualified-private.json",
                  "earlier":args.earlier/"result.json","commons":args.commons/"acquisition.json"}.items()}}


def portable(record):
    rows=record["rows"]
    keys={("panel",i,role) for i in range(12) for role in ROLES}|{("standalone",i,"photo") for i in range(6)}
    need(record["schema"]==SCHEMA and len(rows)==len(keys) and {tuple(row["key"]) for row in rows}==keys,"Confirmation roles changed")
    counts={}
    groups=set()
    for row in rows:
        panel,slot,role=row["key"]
        baseline=row["baseline"]
        need(baseline["flag"]==(baseline["score"]>.8292734622955322),"Existing review point changed")
        item=row["evidence"] or {}
        if item:
            a,b=item["first"],item["second"]
            w,h=a[2]-a[0],a[3]-a[1]
            need(w>=160 and h>=160 and (w,h)==(b[2]-b[0],b[3]-b[1]) and item["exact_match_pixels"]+item["different_pixels"]==w*h,
                 "Aligned rectangle and exact counts disagree")
            need(item["found"]==item["different_pixels"]*2 and item["exact_both"]==
                 bool(role in POSITIVE and item["found"]==row["target"]*2 and item["false_pixels"]==0),"Two-sided truth counts differ")
        c=counts.setdefault(role,Counter())
        c.update(views=1,baseline_flags=baseline["flag"],baseline_useful=baseline["useful"],comparisons=bool(item),
                 difference_flags=bool(item.get("different_pixels")),exact_both=bool(item.get("exact_both")),
                 new_exact=bool(item.get("exact_both") and not baseline["useful"]))
        if role not in POSITIVE and item.get("different_pixels"):
            groups.add((panel,slot))
    result={"by_role":{role:dict(c) for role,c in counts.items()},"control_groups":len(groups),
            "passed":counts["sharp"]["new_exact"]>=6 and counts["feather"]["new_exact"]>=6 and not groups}
    need(record["result"]==result,"Reported confirmation count changed")
    return result


def audit(args,lock,report):
    import numpy as np

    from scripts import image_embedded_pair_difference_transfer as pilot
    from scripts import image_viewport_frame_perturbation as proof

    checked=Counter(views=0,comparisons=0,witnesses=0,exact_masks=0)
    public=[]
    for case,row,expected in zip(cases(args),report["rows"],lock["cases"],strict=True):
        key,_,rgb,first,second,truth=case
        need(pilot.identity(case)==expected and all(row[name]==value for name,value in expected.items()),"Exact source commitment differs")
        checked["views"]+=1
        baseline=proof.side_support(rgb,row["baseline"]["box"])
        need(abs(baseline["score"]-row["baseline"]["score"])<=2e-5,"Baseline side replay differs")
        entry={name:value for name,value in row.items() if name not in {"rgb_sha256","truth_sha256"}}
        if evidence:=row["evidence"]:
            a,b=evidence["first"],evidence["second"]
            x,y,r,d=a
            u,v,s,t=b
            checked["comparisons"]+=1
            for i,j in evidence["witnesses"]:
                left=rgb[j:j+16,i:i+16]
                right=rgb[j+v-y:j+v-y+16,i+u-x:i+u-x+16]
                need(np.array_equal(left,right) and np.unique(left.reshape(-1,3),axis=0).shape[0]>=48,"Exact texture witness differs")
                checked["witnesses"]+=1
            difference=np.max(np.abs(rgb[y:d,x:r].astype(np.int16)-rgb[v:t,u:s]),axis=2)>0
            need(sha(difference.tobytes())==evidence["mask_sha256"] and difference.sum()==evidence["different_pixels"],"Native difference differs")
            selected=pilot.target_views(rgb.shape,a,b,difference)
            target=pilot.target_views(rgb.shape,first,second,truth)
            need(int((selected&~target).sum())==evidence["false_pixels"],"Mapped region overlaps differ")
            if evidence["exact_both"]:
                need(np.array_equal(selected,target) and key[-1] in POSITIVE,"Exact two-sided mask differs")
                checked["exact_masks"]+=1
            entry["evidence"]={name:value for name,value in evidence.items() if name!="mask_sha256"}
        public.append(entry)
    record={name:value for name,value in report.items() if name!="rows"}|{"rows":public,"audit":dict(checked),
             "parameters":lock["parameters"],"source_sha256":lock["source_sha256"],"pilot_sha256":lock["pilot_sha256"],
             "compare_sha256":lock["compare_sha256"]}
    portable(record)
    write(args.run/"public.json",record)
    print(dict(checked),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","run","audit","review"))
    parser.add_argument("paths",nargs="+",type=Path)
    args=parser.parse_args()
    if args.stage=="review":
        path,=args.paths
        print(json.dumps(portable(read(path)),sort_keys=True))
        return
    from scripts import image_embedded_pair_difference_transfer as pilot

    args.commons,args.primary,args.earlier,args.backup,args.pilot_generated,args.pilot_run,args.run=args.paths
    lock=freeze(args)
    path=args.run/"freeze.json"
    if args.stage=="freeze":
        args.run.mkdir(mode=0o700)
        write(path,lock)
        print(file_sha(path),flush=True)
        return
    need(lock==read(path),"Frozen benchmark or original implementation changed")
    if args.stage=="audit":
        report=read(args.run/"result.json")
        need(file_sha(path)==report["freeze_sha256"],"Confirmed result used a different seal")
        audit(args,lock,report)
        return
    started=time.monotonic()
    rows=[]
    for case,expected in zip(cases(args),lock["cases"],strict=True):
        need(pilot.identity(case)==expected,"Exact source changed")
        rows.append(pilot.evaluate(case))
    result={"schema":SCHEMA,"freeze_sha256":file_sha(path),"elapsed_s":time.monotonic()-started,
            "rows":rows,"result":pilot.aggregate(rows)}
    write(args.run/"result.json",result)
    print(json.dumps(result["result"],sort_keys=True),flush=True)


if __name__=="__main__":
    main()
