# SPDX-License-Identifier: Apache-2.0
"""Verify localized native differences between two photo panes in one submitted PNG."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from app.detectors.image import sharp_boundary
from scripts import image_crop_layout_transfer as prior
from scripts import image_source_groups_v22 as source_groups

SCHEMA = "chipmunk.image-embedded-pair-difference-transfer/1"
SHAPE = (640, 768)
TILE = 16
STRIDE = 19
OFFSETS = ((0, 0), (1, 13), (4, 6), (6, 15), (8, 1), (10, 10), (14, 4), (15, 15))
MIX = (0x9e3779b185ebca87, 0xc2b2ae3d27d4eb4f, 0x165667b19e3779f9, 0x85ebca77c2b2ae63,
       0x27d4eb2f165667c5, 0xd6e8feb86659fd93, 0xa0761d6478bd642f, 0x8ebc6af09c88c6e3)
POSITIVE = {"sharp", "feather", "ordinary_local"}
ROLES = ("sharp", "feather", "ordinary_local", "duplicate", "global_tone", "unrelated", "different_crop", "ui")
PARAMETERS = {"tile": TILE, "stride": STRIDE, "max_occurrences": 3, "min_textured_values": 48,
              "min_shift": 192, "min_witnesses": 12, "min_span": 160, "min_spread_cells": 6,
              "min_separation": 24, "minimum_exact_match_fraction": .85, "max_difference_box_share": .45,
              "minimum_component": 48, "minimum_difference_margin": 8, "max_area": 1000000,
              "gates": {"new_sharp_sources": 6, "new_feathered_sources": 6, "new_control_groups": 0},
              "confidence_weight": 0}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    with path.open("x") as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")


def need(check, reason):
    if not check:
        raise ValueError(reason)


def crop(array, box):
    x, y, r, b = box
    return array[y:b, x:r]


def bounds(mask):
    y, x = np.nonzero(mask)
    return int(x.min()), int(y.min()), int(x.max()+1), int(y.max()+1)


def compare(rgb):
    """Use exact native texture witnesses to establish a single repeated translation."""
    h, w = rgb.shape[:2]
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or min(h,w) < 384 or h*w > PARAMETERS["max_area"]:
        return None, None
    packed = rgb.astype(np.uint32)
    values = (packed[:,:,0]<<16) | (packed[:,:,1]<<8) | packed[:,:,2]
    rows, cols = h-TILE+1, w-TILE+1
    hashes = np.zeros((rows,cols), dtype=np.uint64)
    for (dy,dx), factor in zip(OFFSETS,MIX,strict=True):
        hashes ^= values[dy:dy+rows, dx:dx+cols].astype(np.uint64)*np.uint64(factor)
    ordered = np.argsort(hashes,axis=None,kind="stable")
    flattened = hashes.ravel()[ordered]
    yy,xx = np.meshgrid(np.arange(0,rows,STRIDE),np.arange(0,cols,STRIDE),indexing="ij")
    samples = hashes[yy,xx].ravel()
    lo,hi = (np.searchsorted(flattened,samples,side=side) for side in ("left","right"))
    shifts = defaultdict(set)
    for sy,sx,a,b in zip(yy.ravel(),xx.ravel(),lo,hi,strict=True):
        if not 2 <= b-a <= 3:
            continue
        patch = values[sy:sy+TILE,sx:sx+TILE]
        if np.unique(patch).size < 48:
            continue
        for index in ordered[a:b]:
            dy,dx = divmod(int(index),cols)
            tx,ty = int(dx-sx),int(dy-sy)
            if max(abs(tx),abs(ty)) < 192 or not np.array_equal(patch,values[dy:dy+TILE,dx:dx+TILE]):
                continue
            if tx > 0 or (tx == 0 and ty > 0):
                shifts[tx,ty].add((int(sx),int(sy)))
            else:
                shifts[-tx,-ty].add((dx,dy))
    eligible = []
    for (dx,dy), points in shifts.items():
        if len(points) < 12:
            continue
        x,y = (min(p[axis] for p in points) for axis in (0,1))
        r,b = (max(p[axis] for p in points)+TILE for axis in (0,1))
        dw,dh = r-x,b-y
        if min(dw,dh) < 160 or (abs(dx) < dw+24 and abs(dy) < dh+24):
            continue
        occupied = {(min(2,3*(a-x)//dw),min(2,3*(c-y)//dh)) for a,c in points}
        if len(occupied) < 6:
            continue
        first, second = (x,y,r,b), (x+dx,y+dy,r+dx,b+dy)
        a,b = crop(rgb,first),crop(rgb,second)
        difference = (a!=b).any(axis=2)
        if 100*int(difference.sum()) > 15*difference.size:
            continue
        eligible.append((first,second,sorted(points),difference))
    if len(eligible) != 1:
        return None, None
    first,second,points,difference = eligible[0]
    measured = int(difference.sum())
    box = bounds(difference) if measured else None
    if box:
        x,y,r,b = box
        h,w = difference.shape
        _,_,regions,_ = cv2.connectedComponentsWithStats(difference.astype(np.uint8),connectivity=8)
        if min(x,y,w-r,h-b) < 8 or (r-x)*(b-y) > .45*difference.size or int(regions[1:,4].max()) < 48:
            return None, None
    return {"first": list(first), "second": list(second), "witnesses": [list(point) for point in points],
            "exact_match_pixels": difference.size-measured, "different_pixels": measured,
            "difference_box": list(box) if box else None, "mask_sha256": sha(difference.tobytes())}, difference


def source_pairs(args):
    cache, records = prior.qualified_lama(args.lama)
    ranked = sorted(records,key=lambda row:sha(f"embedded-pair/1/{row['source']['flickr_photo_id']}".encode()))[:8]
    for slot,row in enumerate(ranked):
        pair=[]
        for role in ("original","edited"):
            asset=row["measurement"]["files"][role]
            path=cache/asset["file"]
            need(file_sha(path)==asset["sha256"],"Qualified native photo bytes changed")
            pair.append(prior.pixels(path))
        truth=(pair[0]!=pair[1]).any(axis=2)
        truth_path=cache/row["truth_file"]
        need(file_sha(truth_path)==row["truth_png_sha256"],"Qualified exact edit mask changed")
        with Image.open(truth_path) as mask:
            need(np.array_equal(np.asarray(mask.convert("L"))>0,truth),"Actual changed pixels differ")
        yield slot,"lama",*pair,prior.feather(*pair,truth)
    originals=read(args.lama/"qualified-private.json")["qualified"]
    legacy=read(args.generated/"plan.json")["cases"]
    by_slot={row["slot"]:row for row in legacy}
    for name in ("flickr_owner","flickr_photo_id"):
        need(not {r["source"][name] for r in originals}&{r[name] for r in legacy},"Generated and LaMa source roles overlap")
    cases=list(prior.REGIONAL.paired.inputs(args.generated))
    chosen=[]
    for editor in ("sdxl","kandinsky"):
        subset=sorted((case for case in cases if case[0]["generator"]==editor),
                      key=lambda case:sha(f"embedded-pair/1/{by_slot[case[0]['slot']]['flickr_photo_id']}".encode()))[:2]
        chosen.extend(subset)
    need(len(chosen)==4,"Expected source-separated generated photo panel")
    for slot,(case,images,known) in enumerate(chosen,8):
        views=prior.REGIONAL.native_variants(images,case["box"])
        need(all(not ((views["original"]!=views[role]).any(axis=2)&~known).any() for role in ("edited","feathered_edit")),
             "Generated pixels escape qualification")
        yield slot,case["generator"],views["original"],views["edited"],views["feathered_edit"]


def donor_roster(args):
    roster=prior.sources.binding(args.commons)["sources"]
    details=read(args.commons/"plan.json")["sources"]
    chosen=sorted(roster,key=lambda row:sha(f"embedded-pair/1/donor/{row['sha256']}".encode()))[:20]
    need(len({details[r["slot"]]["author"] for r in chosen})==20,"Ordinary source credits overlap")
    return chosen


def panel(original,alternative,slot):
    h,w=SHAPE
    ph,pw=original.shape[:2]
    x,y=25+slot%6,63+slot%9
    dx,dy=pw+56+slot%11,(slot%5)-2
    first=[x,y,x+pw,y+ph]
    second=[x+dx,y+dy,x+dx+pw,y+dy+ph]
    rgb=np.full((h,w,3),(234,238,242),dtype=np.uint8)
    rgb[:42]=(42+slot,63,80)
    for box,source in ((first,original),(second,alternative)):
        a,b,r,d=box
        rgb[b:d,a:r]=source
    return rgb,first,second


def ui_fixture(size,slot):
    image=Image.new("RGB",size,(232,238,243))
    draw=ImageDraw.Draw(image)
    for row in range(18):
        y=16+row*24
        draw.rectangle((10,y,size[0]-12,y+21),fill=(218,227,237) if row%2 else (239,244,249))
        draw.text((24,y+5),f"Review item {slot:02d} / {row:02d}",fill=(35,52,71))
        for col in range(3):
            x=171+col*40
            draw.rectangle((x,y+6,x+15,y+16),outline=(56,98,139))
    return np.asarray(image)


def cases(args):
    roster=donor_roster(args)
    for slot,family,original,sharp,feather in source_pairs(args):
        exact=(original!=sharp).any(axis=2)|(original!=feather).any(axis=2)
        a,b,r,d=bounds(exact)
        h,w=exact.shape
        pw,ph=320,min(480,h)
        x=min(max(0,(a+r-pw)//2),w-pw)
        y=min(max(0,(b+d-ph)//2),h-ph)
        source_box=(x,y,x+pw,y+ph)
        need(crop(exact,source_box).sum() >= .9*exact.sum(),"Paired photo viewport loses too much known content")
        before,edited,blended=(crop(rgb,source_box) for rgb in (original,sharp,feather))
        toned=np.asarray(ImageEnhance.Contrast(Image.fromarray(before)).enhance(.91))
        mild=before.copy()
        ah,aw=ph//7,pw//6
        ay,ax=ph//3,pw//3
        mild[ay:ay+ah,ax:ax+aw]=np.minimum(mild[ay:ay+ah,ax:ax+aw].astype(np.uint16)+13,255).astype(np.uint8)
        different=np.asarray(prior.native.source(args.commons,roster[slot],(pw,ph)))
        shifted=np.roll(before,5,axis=1)
        pattern=ui_fixture((pw,ph),slot)
        pattern_b=pattern.copy()
        pattern_b[55:70,18:80]=(234,208,173)
        variants=(edited,blended,mild,before,toned,different,shifted,pattern_b)
        for role,second in zip(ROLES,variants,strict=True):
            first=pattern if role=="ui" else before
            rgb,box1,box2=panel(first,second,slot)
            truth=(first!=second).any(axis=2) if role in POSITIVE else np.zeros((ph,pw),dtype=bool)
            if role in POSITIVE:
                need(truth.any(),"Selected positive lost all changes")
            yield ("panel",slot,role),family,rgb,box1,box2,truth
    for slot,record in enumerate(roster[12:]):
        rgb=np.asarray(prior.native.source(args.commons,record,SHAPE[::-1]))
        yield ("standalone",slot,"photo"),"credited-photo",rgb,None,None,None


def identity(case):
    key,family,rgb,a,b,truth=case
    return {"key":list(key),"family":family,"rgb_sha256":sha(rgb.tobytes()),"first":a,"second":b,
            "truth_sha256":sha(truth.tobytes()) if truth is not None else None,"target":int(truth.sum()) if truth is not None else 0}


def freeze(args):
    inputs=list(cases(args))
    rows=[identity(case) for case in inputs]
    need(len(rows)==104 and len({tuple(row["key"]) for row in rows})==104,"Held roster incomplete")
    photos={case[0][1]:crop(case[2],case[3]) for case in inputs if case[0][2]=="duplicate"}
    controls=[case[2] for case in inputs if case[0][0]=="standalone"]
    controls.extend(crop(case[2],case[4]) for case in inputs if case[0][-1]=="unrelated")
    def mark(rgb):
        with Image.fromarray(rgb) as image:
            return {"size":image.size,"sketch":f"{source_groups.sketch(image):064x}","rgb":sha(rgb.tobytes())}
    left,right=([mark(rgb) for rgb in values] for values in (photos.values(),controls))
    need(not any(a["rgb"]==b["rgb"] or source_groups.near(a,b) for a in left for b in right),"Ordinary and edited source roles overlap")
    return {"schema":SCHEMA,"parameters":PARAMETERS,"implementation_sha256":file_sha(Path(__file__)),
            "source_sha256":{"lama":file_sha(args.lama/"qualified-private.json"),"generated":file_sha(args.generated/"result.json"),
                              "commons":file_sha(args.commons/"acquisition.json")},
            "baseline_sha256":file_sha(Path(sharp_boundary.__file__)),"point":sharp_boundary.REVIEW_POINT,"cases":rows}


def target_views(rgb_shape,a,b,truth):
    mask=np.zeros(rgb_shape[:2],dtype=bool)
    if a:
        for x,y,r,d in (a,b):
            mask[y:d,x:r]=truth
    return mask


def evaluate(case):
    key,_,rgb,first,second,truth=case
    measured=sharp_boundary.measure(rgb)
    baseline_flag=measured.score>sharp_boundary.REVIEW_POINT
    target=target_views(rgb.shape,first,second,truth)
    box=list(measured.box)
    base_iou=base_recall=0.
    if identity(case)["target"]:
        candidates=[]
        for a in (first,second):
            source=target_views(rgb.shape,a,a,truth)
            overlap=int(crop(source,box).sum())
            candidates.append((overlap/(int(source.sum())+(box[2]-box[0])*(box[3]-box[1])-overlap),overlap/int(source.sum())))
        base_iou,base_recall=max(candidates)
    record,mask=compare(rgb)
    evidence=None
    if record is not None:
        selected=target_views(rgb.shape,record["first"],record["second"],mask)
        detected=int(selected.sum())
        exact=np.array_equal(selected,target) if key[-1] in POSITIVE else False
        evidence=record|{"exact_both":bool(exact),"false_pixels":int((selected&~target).sum()),"found":detected}
    return identity(case)|{"baseline":{"box":box,"score":measured.score,"flag":baseline_flag,
                                          "useful":bool(baseline_flag and base_iou>=.30 and base_recall>=.5)},"evidence":evidence}


def aggregate(rows):
    counts={}
    for row in rows:
        role=row["key"][-1]
        c=counts.setdefault(role,Counter())
        evidence=row["evidence"] or {}
        c.update(views=1, baseline_flags=row["baseline"]["flag"], baseline_useful=row["baseline"]["useful"],
                 comparisons=bool(evidence), difference_flags=bool(evidence.get("different_pixels")),exact_both=bool(evidence.get("exact_both")),
                 new_exact=bool(evidence.get("exact_both") and not row["baseline"]["useful"]))
    controls={(row["key"][0],row["key"][1]) for row in rows if row["key"][-1] not in POSITIVE
              and (row["evidence"] or {}).get("different_pixels")}
    return {"by_role":{key:dict(value) for key,value in counts.items()},"control_groups":len(controls),
            "passed":counts["sharp"]["new_exact"]>=6 and counts["feather"]["new_exact"]>=6 and not controls}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","run"))
    for name in ("commons","lama","generated","run"):
        parser.add_argument(name,type=Path)
    args=parser.parse_args()
    lock=freeze(args)
    path=args.run/"freeze.json"
    if args.stage=="freeze":
        args.run.mkdir(mode=0o700)
        write(path,lock)
        print(file_sha(path),flush=True)
        return
    need(read(path)==lock,"Frozen cases or method changed")
    start=time.monotonic()
    rows=[]
    for case,expected in zip(cases(args),lock["cases"],strict=True):
        need(identity(case)==expected,"Native case changed")
        rows.append(evaluate(case))
    result={"schema":SCHEMA,"freeze_sha256":file_sha(path),"elapsed_s":time.monotonic()-start,"rows":rows,"result":aggregate(rows)}
    write(args.run/"result.json",result)
    print(json.dumps(result["result"],sort_keys=True),flush=True)


if __name__=="__main__":
    main()
