# SPDX-License-Identifier: Apache-2.0
"""Prospectively test exact and independently verified resized third-copy abstention."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "chipmunk.image-embedded-pair-third-guard/1"
CONFIRM = "58e314577bbf37bba612e3037995b00a9f387ed8141bd0ec8ae3d9a5927223f5"
ROLES = ("local", "sharp", "feather", "unrelated", "exact_third", "partial_third", "resampled_third", "duplicate")
POSITIVE = {"local", "sharp", "feather", "unrelated"}
THIRD = {"exact_third", "partial_third", "resampled_third"}
GATE = {"minimum_true_pair_admissions": 12,"minimum_exact_challenge": 6,"minimum_partial_challenge": 4,
        "minimum_resampled_challenge": 8,"minimum_resampled_suppression_fraction": .75,
        "lost_exact_pair_masks": 0,"lost_exact_partial_third_challenges": 0}
RULE = {"exact_tiles":12,"exact_grid_cells":6,"exact_minimum_span":128,"exact_max_occurrences":8,
        "third_separation":8,"sift_ratio":.72,"sift_ransac_pixels":2.,"sift_minimum_inliers":12,
        "sift_minimum_inlier_share":.55,"sift_scale":[.60,1.30],"sift_angle_degrees":5.,
        "sift_reference_span":160,"sift_third_span":128,"blurred_patch_correlation":.97,
        "minimum_independent_patch_matches":10,"gate":GATE,"confidence_weight":0}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fs(path):
    return sha(path.read_bytes())


def read(path):
    return json.loads(path.read_bytes())


def write(path,value):
    with path.open("x") as out:
        json.dump(value,out,sort_keys=True,indent=2,allow_nan=False)
        out.write("\n")


def need(value,reason):
    if not value:
        raise ValueError(reason)


def separated(a,b,margin=8):
    return a[2]+margin<=b[0] or b[2]+margin<=a[0] or a[3]+margin<=b[1] or b[3]+margin<=a[1]


def spread(points, minimum):
    x,y=(min(p[axis] for p in points) for axis in (0,1))
    r,b=(max(p[axis] for p in points)+16 for axis in (0,1))
    if min(r-x,b-y)<minimum:
        return None
    cells={(min(2,3*(a-x)//(r-x)),min(2,3*(c-y)//(b-y))) for a,c in points}
    return [x,y,r,b] if len(cells)>=6 else None


def exact_third(rgb,evidence):
    import numpy as np

    from scripts import image_embedded_pair_difference_transfer as base

    h,w=rgb.shape[:2]
    packed=rgb.astype(np.uint32)
    values=(packed[:,:,0]<<16)|(packed[:,:,1]<<8)|packed[:,:,2]
    rows,cols=h-15,w-15
    hashes=np.zeros((rows,cols),dtype=np.uint64)
    for (dy,dx),factor in zip(base.OFFSETS,base.MIX,strict=True):
        hashes^=values[dy:dy+rows,dx:dx+cols].astype(np.uint64)*np.uint64(factor)
    order=np.argsort(hashes,axis=None,kind="stable")
    keys=hashes.ravel()[order]
    a,b=evidence["first"],evidence["second"]
    groups=defaultdict(set)
    for x,y in evidence["witnesses"]:
        key=hashes[y,x]
        left,right=(int(np.searchsorted(keys,key,side=which)) for which in ("left","right"))
        if not 3<=right-left<=8:
            continue
        for index in order[left:right]:
            ty,tx=divmod(int(index),cols)
            patch=[tx,ty,tx+16,ty+16]
            if (separated(patch,a,0) and separated(patch,b,0)
                    and np.array_equal(values[y:y+16,x:x+16],values[ty:ty+16,tx:tx+16])):
                groups[tx-x,ty-y].add((tx,ty))
    choices=[]
    for shift,points in groups.items():
        if len(points)<12:
            continue
        third=spread(points,128)
        if third is not None and separated(third,a) and separated(third,b):
            choices.append((len(points),shift,third,sorted(points)))
    if not choices:
        return None
    count,shift,box,points=max(choices)
    return {"kind":"exact_third","box":box,"translation":list(shift),"count":count,"points":[list(p) for p in points]}


def approximate_third(rgb,evidence):
    import cv2
    import numpy as np

    h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    a,b=evidence["first"],evidence["second"]
    inside=np.zeros((h,w),np.uint8)
    outside=np.full((h,w),255,np.uint8)
    x,y,r,d=a
    inside[y+8:d-8,x+8:r-8]=255
    for x,y,r,d in (a,b):
        outside[max(0,y-24):min(h,d+24),max(0,x-24):min(w,r+24)]=0
    feature=cv2.SIFT_create(nfeatures=1200)
    first,fd=feature.detectAndCompute(gray,inside)
    third,td=feature.detectAndCompute(gray,outside)
    if fd is None or td is None or len(td)<2:
        return None
    nearest=cv2.BFMatcher(cv2.NORM_L2).knnMatch(fd,td,k=2)
    best={}
    for m,n in nearest:
        if m.distance<.72*n.distance and (m.trainIdx not in best or m.distance<best[m.trainIdx].distance):
            best[m.trainIdx]=m
    matches=sorted(best.values(),key=lambda m:(m.queryIdx,m.trainIdx))
    if len(matches)<12:
        return None
    src=np.array([first[m.queryIdx].pt for m in matches],dtype=np.float32)
    dst=np.array([third[m.trainIdx].pt for m in matches],dtype=np.float32)
    cv2.setRNGSeed(1189)
    matrix,flag=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.,
                                          maxIters=1000,confidence=.995,refineIters=10)
    if matrix is None or flag is None:
        return None
    kept=flag.reshape(-1).astype(bool)
    count=int(kept.sum())
    scale=math.hypot(matrix[0,0],matrix[1,0])
    angle=math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
    if count<12 or count<.55*len(matches) or not .60<=scale<=1.30 or abs(angle)>5:
        return None
    source,destination=src[kept],dst[kept]
    source_box=spread([(int(x)-8,int(y)-8) for x,y in source],160)
    target_box=spread([(int(x)-8,int(y)-8) for x,y in destination],128)
    if source_box is None or target_box is None or not separated(target_box,a) or not separated(target_box,b):
        return None
    smooth=cv2.GaussianBlur(gray.astype(np.float32),(0,0),1.)
    mapped=cv2.warpAffine(smooth,matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
    witnesses=[]
    used=set()
    for x,y in source:
        x,y=round(float(x)),round(float(y))
        if not 8<=x<w-8 or not 8<=y<h-8 or (x//16,y//16) in used:
            continue
        p,q=(image[y-8:y+8,x-8:x+8].astype(np.float64).ravel() for image in (smooth,mapped))
        if min(p.std(),q.std())<8:
            continue
        correlation=float(np.corrcoef(p,q)[0,1])
        if correlation>=.97:
            used.add((x//16,y//16))
            witnesses.append([x,y,correlation])
    if len(witnesses)<10:
        return None
    return {"kind":"resampled_third","box":target_box,"matrix":matrix.tolist(),"count":count,
            "matched_features":len(matches),"witnesses":witnesses}


def roster(args):
    from scripts import image_embedded_pair_difference_confirmation as prior

    cache,former,_=prior.choose(args)
    selected={row["source"]["flickr_photo_id"] for row in former}
    all_rows=read(args.primary/"qualified-private.json")["qualified"]
    available=[row for row in all_rows if row["source"]["flickr_photo_id"] not in selected]
    role=[]
    for split in sorted({row["source"]["publisher_split"] for row in available}):
        subset=[row for row in available if row["source"]["publisher_split"]==split]
        role.extend(sorted(subset,key=lambda row:sha(f"embedded-guard/1/{row['source']['flickr_photo_id']}".encode()))[:3])
    target={row["source"]["flickr_photo_id"] for row in role}
    controls=[row for row in available if row["source"]["flickr_photo_id"] not in target]
    need(len(role)==12 and len(controls)==4,"Fresh held and third-photo roles differ")
    return cache,role,controls


def cases(args):
    import numpy as np
    from PIL import Image

    from scripts import image_crop_layout_transfer as prior

    cache,qualified,controls=roster(args)
    def native(row,role):
        item=row["measurement"]["files"][role]
        path=cache/item["file"]
        need(fs(path)==item["sha256"],"Qualified image bytes changed")
        return prior.pixels(path)
    donors=[native(row,"original") for row in controls]
    for slot,row in enumerate(qualified):
        before,after=(native(row,name) for name in ("original","edited"))
        mask=(before!=after).any(axis=2)
        path=cache/row["truth_file"]
        need(fs(path)==row["truth_png_sha256"],"Exact generated mask changed")
        with Image.open(path) as image:
            need(np.array_equal(mask,np.asarray(image.convert("L"))>0),"Generated mask pixels changed")
        blend=prior.feather(before,after,mask)
        l,t,r,d=prior.source_box(mask)
        h,w=mask.shape
        pw,ph=320,min(440,h)
        x=min(max(0,(l+r-pw)//2),w-pw)
        y=min(max(0,(t+d-ph)//2),h-ph)
        original,edited,feather=(image[y:y+ph,x:x+pw] for image in (before,after,blend))
        local=original.copy()
        area=np.s_[ph//3:ph//3+ph//6,pw//3:pw//3+pw//6]
        local[area]=np.minimum(local[area].astype(np.uint16)+17,255)
        if slot%3!=2:
            size=(688,1088)
            first=(18+slot,83+slot)
            stride=338+slot%5
            second=(first[0]+2*stride,first[1]+(7 if slot%3==1 else 0))
            third=(first[0]+stride,first[1]+3)
        else:
            size=(944,944)
            first=(29,29+slot)
            second=(545,493+slot)
            third=(480,27+slot)
        unrelated=donors[slot%len(donors)][:ph,:pw]
        partial=original[ph//8:ph-ph//8,83:235]
        resampled=np.asarray(Image.fromarray(original).resize((round(pw*.78),round(ph*.78)),Image.Resampling.BICUBIC))
        variants=((local,None),(edited,None),(feather,None),(local,unrelated),(local,original),
                  (local,partial),(local,resampled),(original,None))
        for role,(changed,other) in zip(ROLES,variants,strict=True):
            rgb=np.full((*size,3),(226,233,242),dtype=np.uint8)
            rgb[:14]=(62+slot,75,92)
            for (px,py),picture in ((first,original),(second,changed),(third,other)):
                if picture is None:
                    continue
                hh,ww=picture.shape[:2]
                need(px+ww<=size[1] and py+hh<=size[0],"Pane does not fit its native canvas")
                rgb[py:py+hh,px:px+ww]=picture
            truth=(original!=changed).any(axis=2)
            need(role=="duplicate" or truth.any(),"Known changed pixels missing")
            yield slot,role,rgb,list(first),list(second),list(third) if other is not None else None,truth


def identity(case):
    slot,role,rgb,a,b,c,truth=case
    return {"slot":slot,"role":role,"shape":list(rgb.shape[:2]),"rgb_sha256":sha(rgb.tobytes()),
            "first":a,"second":b,"third":c,"truth_sha256":sha(truth.tobytes()),"target":int(truth.sum())}


def freeze(args):
    from scripts import image_embedded_pair_difference_confirmation as prior
    from scripts import image_embedded_pair_difference_transfer as base

    need(fs(args.confirmation/"freeze.json")==CONFIRM,"Original failed confirmation seal changed")
    sealed=read(args.confirmation/"freeze.json")
    need(fs(Path(base.__file__))==sealed["compare_sha256"] and fs(Path(prior.__file__))==sealed["implementation_sha256"],
         "Earlier frozen comparison changed")
    rows=[identity(case) for case in cases(args)]
    need(len(rows)==96 and len({(row["slot"],row["role"]) for row in rows})==96,"Frozen denominator changed")
    return {"schema":SCHEMA,"rule":RULE,"confirmation_sha256":CONFIRM,"base_sha256":sealed["compare_sha256"],
            "code_sha256":fs(Path(__file__)),"source_sha256":fs(args.primary/"qualified-private.json"),"cases":rows}


def evaluate(case):
    import numpy as np

    from scripts import image_embedded_pair_difference_transfer as base

    slot,role,rgb,first,second,_,truth=case
    evidence,mask=base.compare(rgb)
    raw=bool(evidence and evidence["different_pixels"])
    exact=approximate=None
    confirmed=False
    if raw:
        expected=np.zeros(rgb.shape[:2],dtype=bool)
        h,w=truth.shape
        for x,y in (first,second):
            expected[y:y+h,x:x+w]=truth
        matched=base.target_views(rgb.shape,evidence["first"],evidence["second"],mask)
        confirmed=bool(np.array_equal(matched,expected))
        exact=exact_third(rgb,evidence)
        approximate=approximate_third(rgb,evidence)
    return identity(case)|{"base_comparison":bool(evidence),"base_flag":raw,"exact_mask":confirmed,
                            "exact_guard":exact,"approximate_guard":approximate,"base_evidence":evidence,
                            "retained":raw and exact is None and approximate is None}


def summary(rows):
    results={}
    for row in rows:
        role=row["role"]
        cell=results.setdefault(role,Counter())
        cell.update(views=1,base_flags=row["base_flag"],base_exact_masks=row["exact_mask"],
                    exact_guard=bool(row["exact_guard"]),approximate_guard=bool(row["approximate_guard"]),
                    combined_guard=row["base_flag"] and not row["retained"],retained=row["retained"],
                    retained_exact_masks=row["retained"] and row["exact_mask"])
    controls=sum(results[k]["base_exact_masks"] for k in POSITIVE)
    losses=controls-sum(results[k]["retained_exact_masks"] for k in POSITIVE)
    exact,partial,resampled=(results[key] for key in ("exact_third","partial_third","resampled_third"))
    success=(controls>=12 and losses==0 and exact["base_flags"]>=6 and partial["base_flags"]>=4
             and resampled["base_flags"]>=8 and exact["retained"]==partial["retained"]==0
             and resampled["combined_guard"]>=.75*resampled["base_flags"] and not results["duplicate"]["retained"])
    return {"by_role":{name:dict(count) for name,count in results.items()},"lost_exact_pair_masks":losses,"passed":success}


def portable(record):
    rows=record["rows"]
    need(record["schema"]==SCHEMA and len(rows)==96 and {(row["slot"],row["role"]) for row in rows}==
         {(slot,role) for slot in range(12) for role in ROLES},"Frozen role count differs")
    for row in rows:
        need(row["retained"]==bool(row["base_flag"] and not (row["exact_guard"] or row["approximate_guard"])),"Abstention record differs")
        if row["exact_guard"]:
            need(row["exact_guard"]["count"]>=12,"Exact third-copy witnesses missing")
        if row["approximate_guard"]:
            approx=row["approximate_guard"]
            need(approx["count"]>=12 and approx["count"]>=.55*approx["matched_features"] and approx["pixel_witnesses"]>=10,
                 "Resampled third-copy witnesses missing")
    need(summary(rows)==record["result"],"Held aggregate differs")
    return record["result"]


def audit(args,lock,report):
    import numpy as np

    from scripts import image_embedded_pair_difference_transfer as base

    published=[]
    verified=Counter(views=0,exact_guard_tiles=0,approximate_witnesses=0,exact_masks=0)
    for case,row,expected in zip(cases(args),report["rows"],lock["cases"],strict=True):
        need(identity(case)==expected and all(row[key]==value for key,value in expected.items()),"Frozen case differs")
        _,_,rgb,first,second,_,truth=case
        verified["views"]+=1
        if evidence:=row["base_evidence"]:
            mask=(base.crop(rgb,evidence["first"])!=base.crop(rgb,evidence["second"])).any(axis=2)
            need(sha(mask.tobytes())==evidence["mask_sha256"] and int(mask.sum())==evidence["different_pixels"],"Native difference differs")
            if row["exact_mask"]:
                expected_mask=np.zeros(rgb.shape[:2],dtype=bool)
                h,w=truth.shape
                for x,y in (first,second):
                    expected_mask[y:y+h,x:x+w]=truth
                need(np.array_equal(base.target_views(rgb.shape,evidence["first"],evidence["second"],mask),expected_mask),
                     "Exact paired mask differs")
                verified["exact_masks"]+=1
        raw={key:value for key,value in row.items() if key not in {"base_evidence","rgb_sha256","truth_sha256"}}
        if exact:=row["exact_guard"]:
            dx,dy=exact["translation"]
            need(len(exact["points"])==exact["count"],"Third tile count differs")
            for x,y in exact["points"]:
                need(np.array_equal(rgb[y:y+16,x:x+16],rgb[y-dy:y-dy+16,x-dx:x-dx+16]),"Third native texture differs")
                verified["exact_guard_tiles"]+=1
            raw["exact_guard"]={key:value for key,value in exact.items() if key!="points"}
        if approx:=row["approximate_guard"]:
            replay=approximate_third(rgb,row["base_evidence"])
            need(replay==approx,"Independent resampled support replay differs")
            verified["approximate_witnesses"]+=len(approx["witnesses"])
            raw["approximate_guard"]={key:value for key,value in approx.items() if key!="witnesses"}|{"pixel_witnesses":len(approx["witnesses"])}
        published.append(raw)
    result={key:value for key,value in report.items() if key!="rows"}|{"rows":published,"audit":dict(verified),"rule":RULE,
               "confirmation_sha256":lock["confirmation_sha256"],"source_sha256":lock["source_sha256"]}
    portable(result)
    write(args.run/"public.json",result)
    print(dict(verified),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","run","audit","review"))
    parser.add_argument("paths",nargs="+",type=Path)
    args=parser.parse_args()
    if args.stage=="review":
        path,=args.paths
        print(json.dumps(portable(read(path)),sort_keys=True))
        return
    import cv2

    cv2.setNumThreads(1)
    args.primary,args.earlier,args.backup,args.pilot_generated,args.confirmation,args.run=args.paths
    lock=freeze(args)
    path=args.run/"freeze.json"
    if args.stage=="freeze":
        args.run.mkdir(mode=0o700)
        write(path,lock)
        print(fs(path),flush=True)
        return
    need(lock==read(path),"Frozen inputs or method differ")
    if args.stage=="audit":
        report=read(args.run/"result.json")
        need(report["freeze_sha256"]==fs(path),"Result freeze differs")
        audit(args,lock,report)
        return
    start=time.monotonic()
    rows=[]
    for case,expected in zip(cases(args),lock["cases"],strict=True):
        need(identity(case)==expected,"Native case changed")
        rows.append(evaluate(case))
        if len(rows)%24==0:
            print(len(rows),96,flush=True)
    result={"schema":SCHEMA,"rows":rows,"result":summary(rows),"freeze_sha256":fs(path),"elapsed_s":time.monotonic()-start}
    write(args.run/"result.json",result)
    print(json.dumps(result["result"],sort_keys=True),flush=True)


if __name__=="__main__":
    main()
